import argparse
import csv
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformers import get_linear_schedule_with_warmup
from datasets import load_dataset

from models.vaegpt import VAEGPT
from prepare_dataset import SentencePairDataset, build_tokenizers


def set_seed(seed: int, deterministic: bool = False):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_tsv(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            rows.append({
                "sentence1": r["sentence1"],
                "sentence2": r["sentence2"],
                "label": float(r["score"]),
            })
    return rows


def load_stsb(local_train_tsv: str, local_dev_tsv: str):
    if local_train_tsv or local_dev_tsv:
        assert local_train_tsv and local_dev_tsv, "Please provide both train/dev TSV paths."
        ds_train = load_tsv(local_train_tsv)
        ds_dev = load_tsv(local_dev_tsv)
        return ds_train, ds_dev

    dsd = load_dataset("glue", "stsb")
    return dsd["train"], dsd["validation"]


def kl_loss(mu, logvar, free_bits: float = 0.0):
    kl = 0.5 * (torch.exp(logvar) + mu ** 2 - 1.0 - logvar)
    if free_bits > 0:
        kl = torch.clamp(kl, min=free_bits)
    return kl.sum(dim=1).mean()


def kl_anneal(step: int, warmup_steps: int, start: float = 0.0, end: float = 1.0):#KL 权重退火，防止一上来 KL 太强导致 latent collapse
    if warmup_steps <= 0:
        return end
    progress = min(1.0, step / float(warmup_steps))
    return start + (end - start) * progress


def shift_decoder_inputs(dec_ids, dec_attn):#右移
    # Teacher forcing shift: inputs are tokens[:-1], labels are tokens[1:]
    dec_in = dec_ids[:, :-1]
    dec_labels = dec_ids[:, 1:]
    dec_attn_in = dec_attn[:, :-1]
    dec_attn_labels = dec_attn[:, 1:]
    return dec_in, dec_labels, dec_attn_in, dec_attn_labels


def apply_decoder_token_dropout(dec_in, drop_p: float, eos_id: int):#随机把一部分 token“替换/抹掉/置为某个占位”
    if drop_p <= 0:
        return dec_in
    keep = torch.rand_like(dec_in.float()) > drop_p
    # Keep the first token stable
    keep[:, 0] = True
    dec_in_dropped = dec_in.clone()
    dec_in_dropped[~keep] = eos_id
    return dec_in_dropped


def unfreeze_last_n_layers(model: VAEGPT, n_last: int):#最后 n 层从冻结状态改为可训练
    total = len(model.decoder.transformer.h)
    keep_frozen = max(0, total - n_last)
    for i, blk in enumerate(model.decoder.transformer.h):
        req = (i >= keep_frozen)
        for p in blk.parameters():
            p.requires_grad = req
    # Keep embeddings frozen for stability
    for p in model.decoder.transformer.wte.parameters():
        p.requires_grad = False


def staged_unfreeze(model: VAEGPT, step: int, stage: int, stage1_at: int, stage2_at: int, n1: int, n2: int) -> int:#分阶段解冻
    if stage < 1 and stage1_at and step >= stage1_at:
        unfreeze_last_n_layers(model, n1)
        print(f"[Unfreeze] step {step}: last {n1} GPT-2 blocks")
        return 1
    if stage < 2 and stage2_at and step >= stage2_at:
        unfreeze_last_n_layers(model, n2)
        print(f"[Unfreeze] step {step}: last {n2} GPT-2 blocks")
        return 2
    return stage


def build_optimizer(model: VAEGPT, sim_head: nn.Module, lr_main: float, lr_sim: float, lr_dec: float, weight_decay: float):#配置创建 optimizer
    dec_params = []
    other_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("decoder.transformer.h"):
            dec_params.append(p)
        else:
            other_params.append(p)

    param_groups = [
        {"params": other_params, "lr": lr_main, "initial_lr": lr_main, "weight_decay": weight_decay},
        {"params": sim_head.parameters(), "lr": lr_sim, "initial_lr": lr_sim, "weight_decay": 0.0},
    ]
    if dec_params:
        param_groups.append({"params": dec_params, "lr": lr_dec, "initial_lr": lr_dec, "weight_decay": 0.0})

    return torch.optim.AdamW(param_groups)


def build_scheduler(optimizer, warmup_steps: int, num_steps: int, last_epoch: int = -1):#创建学习率调度器
    return get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_steps,
        last_epoch=last_epoch,
    )


def extract_state_dict(state):#从模型对象里抽取可保存的权重字典
    if isinstance(state, dict):
        if "model" in state:
            return state["model"], state.get("sim_head"), state
        if "state_dict" in state:
            return state["state_dict"], None, state
    return state, None, {}


def adapt_single_prefix_ckpt_to_multi(state_dict, model):#把“单 prefix”的 checkpoint 权重改造成“多 prefix”结构可加载的形式。
    w_key, b_key = "z_to_embed.weight", "z_to_embed.bias"
    if w_key in state_dict and b_key in state_dict:
        w = state_dict[w_key]
        b = state_dict[b_key]
        exp_w = model.z_to_embed.weight.data
        exp_b = model.z_to_embed.bias.data
        if w.shape != exp_w.shape:
            old_out, in_dim = w.shape
            new_out, in_dim_exp = exp_w.shape
            if in_dim != in_dim_exp:
                return state_dict
            if new_out % old_out == 0:
                mult = new_out // old_out
                state_dict[w_key] = w.repeat(mult, 1)
                state_dict[b_key] = b.repeat(mult)
            else:
                new_w = exp_w.clone()
                new_b = exp_b.clone()
                copy_rows = min(old_out, new_out)
                new_w[:copy_rows, :] = w[:copy_rows, :]
                new_b[:copy_rows] = b[:copy_rows]
                state_dict[w_key] = new_w
                state_dict[b_key] = new_b
    return state_dict


def maybe_load_checkpoint(path: str, model: VAEGPT, sim_head: nn.Module, device: torch.device):#有就加载，没有就跳过”的加载逻辑
    if not path or not os.path.exists(path):
        return 0, -1.0
    print(f"[Resume] loading checkpoint: {path}")
    state = torch.load(path, map_location=device)
    model_state, sim_state, extra = extract_state_dict(state)
    model_state = adapt_single_prefix_ckpt_to_multi(model_state, model)
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    if missing or unexpected:
        print("[load_state_dict] missing:", missing)
        print("[load_state_dict] unexpected:", unexpected)
    if sim_state is not None:
        sim_head.load_state_dict(sim_state, strict=False)
    return int(extra.get("step", 0)), float(extra.get("best_spearman", -1.0))


def save_train_checkpoint(path: str, model: VAEGPT, sim_head: nn.Module, step: int, best_spearman: float):
    torch.save({
        "model": model.state_dict(),
        "sim_head": sim_head.state_dict(),
        "step": step,
        "best_spearman": best_spearman,
    }, path)


@torch.no_grad()
def encode_sentences(model, bert_tok, device, sentences, max_len=128, batch_size=64):
    mus = []
    for i in range(0, len(sentences), batch_size):
        chunk = sentences[i:i + batch_size]
        enc = bert_tok(
            chunk,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=max_len,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        mu, _ = model.encode(input_ids, attention_mask)
        mus.append(mu.cpu())
    return torch.cat(mus, dim=0)


@torch.no_grad()
def eval_stsb_dev(model, bert_tok, device, ds_dev, max_len=128, batch_size=64):
    model.eval()
    s1 = [ex["sentence1"] for ex in ds_dev]
    s2 = [ex["sentence2"] for ex in ds_dev]
    labels = torch.tensor([float(ex["label"]) for ex in ds_dev])

    mu1 = encode_sentences(model, bert_tok, device, s1, max_len=max_len, batch_size=batch_size)
    mu2 = encode_sentences(model, bert_tok, device, s2, max_len=max_len, batch_size=batch_size)

    cos = F.cosine_similarity(mu1, mu2, dim=-1).numpy()
    labels_np = labels.numpy()

    from scipy.stats import spearmanr, pearsonr
    sp = spearmanr(cos, labels_np).correlation
    pr = pearsonr(cos, labels_np).statistic
    return float(sp), float(pr), len(labels)


def parse_args():#解析运行命令里传的参数
    ap = argparse.ArgumentParser()

    # data / env
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--bert_dir", default=None)
    ap.add_argument("--gpt2_dir", default=None)
    ap.add_argument("--local_stsb_train_tsv", default=None)
    ap.add_argument("--local_stsb_dev_tsv", default=None)
    ap.add_argument("--min_label", type=float, default=4.0)

    # model
    ap.add_argument("--latent_size", type=int, default=32)
    ap.add_argument("--latent_prefix_len", type=int, default=12)
    ap.add_argument("--max_len", type=int, default=128)

    # training
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_steps", type=int, default=2000)
    ap.add_argument("--grad_accum_steps", type=int, default=1)
    ap.add_argument("--lr_main", type=float, default=1.5e-5)
    ap.add_argument("--lr_sim", type=float, default=3e-4)
    ap.add_argument("--lr_dec", type=float, default=5e-6)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--warmup_steps", type=int, default=-1)
    ap.add_argument("--kl_warmup_steps", type=int, default=1500)
    ap.add_argument("--kl_start", type=float, default=0.0)
    ap.add_argument("--kl_max", type=float, default=0.5)
    ap.add_argument("--free_bits", type=float, default=0.05)
    ap.add_argument("--lambda_sim", type=float, default=3.0)
    ap.add_argument("--decoder_drop_p", type=float, default=0.3)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--amp", action="store_true")

    # freeze / unfreeze
    ap.add_argument("--freeze_gpt2_all", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--unfreeze_stage1_at", type=int, default=0)
    ap.add_argument("--unfreeze_stage2_at", type=int, default=0)
    ap.add_argument("--unfreeze_last_n1", type=int, default=1)
    ap.add_argument("--unfreeze_last_n2", type=int, default=2)

    # eval / save
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--eval_batch_size", type=int, default=64)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--save_dir", default="checkpoints")
    ap.add_argument("--resume_from", default=None)
    ap.add_argument("--early_stop_patience", type=int, default=5)
    ap.add_argument("--early_stop_min_delta", type=float, default=0.0005)
    ap.add_argument("--early_stop_start_step", type=int, default=0)
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--eval_ckpt", default=None)

    return ap.parse_args()


def main():
    args = parse_args()

    if args.offline:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"

    set_seed(args.seed, args.deterministic)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    bert_tok, gpt2_tok = build_tokenizers(
        bert_name_or_dir=args.bert_dir or "bert-base-uncased",
        gpt2_name_or_dir=args.gpt2_dir or "gpt2",
        offline=args.offline,
    )

    ds_train, ds_dev = load_stsb(args.local_stsb_train_tsv, args.local_stsb_dev_tsv)
    if args.min_label is not None and args.min_label > 0:
        if hasattr(ds_train, "filter"):
            ds_train = ds_train.filter(lambda ex: float(ex["label"]) >= args.min_label)
            train_len = len(ds_train)
        else:
            ds_train = [ex for ex in ds_train if float(ex["label"]) >= args.min_label]
            train_len = len(ds_train)
        print(f"[Data] train filtered by label >= {args.min_label}: {train_len} samples")
    dataset_train = SentencePairDataset(ds_train, bert_tok, gpt2_tok, max_len=args.max_len)
    loader = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model = VAEGPT(
        latent_size=args.latent_size,
        latent_prefix_len=args.latent_prefix_len,
        bert_name_or_dir=args.bert_dir or "bert-base-uncased",
        gpt2_name_or_dir=args.gpt2_dir or "gpt2",
    ).to(device)

    if args.freeze_gpt2_all:
        for p in model.decoder.parameters():
            p.requires_grad = False
        print("[Freeze] GPT-2 all frozen")

    sim_head = nn.Sequential(
        nn.Linear(args.latent_size, 128),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(128, 1),
    ).to(device)

    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_latest = os.path.join(args.save_dir, "vaegpt.pt")
    ckpt_best = os.path.join(args.save_dir, "vaegpt_best.pt")
    ckpt_train = os.path.join(args.save_dir, "vaegpt_train.pt")

    if args.eval_only:
        ckpt_path = args.eval_ckpt or args.resume_from
        if not ckpt_path:
            if os.path.exists(ckpt_best):
                ckpt_path = ckpt_best
            elif os.path.exists(ckpt_latest):
                ckpt_path = ckpt_latest
        if not ckpt_path or not os.path.exists(ckpt_path):
            raise FileNotFoundError("No checkpoint found for eval. Provide --eval_ckpt or --resume_from.")
        state = torch.load(ckpt_path, map_location=device)
        model_state, _, _ = extract_state_dict(state)
        model_state = adapt_single_prefix_ckpt_to_multi(model_state, model)
        missing, unexpected = model.load_state_dict(model_state, strict=False)
        if missing or unexpected:
            print("[load_state_dict] missing:", missing)
            print("[load_state_dict] unexpected:", unexpected)
        sp, pr, n = eval_stsb_dev(
            model,
            bert_tok,
            device,
            ds_dev,
            max_len=args.max_len,
            batch_size=args.eval_batch_size,
        )
        print(f"[DEV] STS-B dev | Spearman: {sp:.4f} | Pearson: {pr:.4f} | n={n}")
        return

    start_step, best_spearman = maybe_load_checkpoint(args.resume_from, model, sim_head, device)

    warmup_steps = args.warmup_steps if args.warmup_steps >= 0 else int(args.num_steps * args.warmup_ratio)
    optimizer = build_optimizer(model, sim_head, args.lr_main, args.lr_sim, args.lr_dec, args.weight_decay)
    scheduler = build_scheduler(optimizer, warmup_steps, args.num_steps, last_epoch=start_step - 1)

    ce_loss = nn.CrossEntropyLoss(ignore_index=-100)
    mse_loss = nn.MSELoss()

    use_amp = args.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    stage = 0
    if best_spearman < 0:
        best_spearman = -1.0
    bad_evals = 0

    if args.unfreeze_stage1_at < 0:
        args.unfreeze_stage1_at = int(args.num_steps * 0.5)
    if args.unfreeze_stage2_at < 0:
        args.unfreeze_stage2_at = int(args.num_steps * 0.75)

    data_iter = iter(loader)
    current_step = start_step

    try:
        for step in range(start_step + 1, args.num_steps + 1):
            current_step = step
            model.train()
            sim_head.train()
            optimizer.zero_grad(set_to_none=True)

            accum_loss = 0.0
            accum_rec = 0.0
            accum_kl = 0.0
            accum_sim = 0.0

            for _ in range(args.grad_accum_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(loader)
                    batch = next(data_iter)

                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                dec_ids = batch["decoder_input_ids"].to(device)
                dec_attn = batch["decoder_attention_mask"].to(device)
                labels_float = batch["label"].to(device)

                dec_in, dec_labels, dec_attn_in, dec_attn_labels = shift_decoder_inputs(dec_ids, dec_attn)
                dec_in = apply_decoder_token_dropout(dec_in, args.decoder_drop_p, gpt2_tok.eos_token_id)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits, mu, logvar = model(input_ids, attention_mask, dec_in, decoder_attention_mask=dec_attn_in)

                    B, L = dec_labels.size()
                    Lz = model.latent_prefix_len
                    labels_rec = torch.full((B, Lz + L), -100, dtype=torch.long, device=device)
                    labels_rec[:, Lz:] = dec_labels
                    labels_rec[:, Lz:][dec_attn_labels == 0] = -100

                    loss_rec = ce_loss(logits.view(-1, logits.size(-1)), labels_rec.view(-1))
                    beta = kl_anneal(step, args.kl_warmup_steps, start=args.kl_start, end=args.kl_max)
                    loss_kl = kl_loss(mu, logvar, free_bits=args.free_bits)

                    pred = sim_head(mu).squeeze(-1)
                    loss_sim = mse_loss(pred, labels_float)

                    loss = loss_rec + beta * loss_kl + args.lambda_sim * loss_sim

                loss_scaled = loss / float(args.grad_accum_steps)
                if use_amp:
                    scaler.scale(loss_scaled).backward()
                else:
                    loss_scaled.backward()

                accum_loss += loss.item()
                accum_rec += loss_rec.item()
                accum_kl += loss_kl.item()
                accum_sim += loss_sim.item()

            if use_amp:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(sim_head.parameters()), args.max_grad_norm)

            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()

            if args.save_every > 0 and step % args.save_every == 0:
                save_train_checkpoint(ckpt_train, model, sim_head, step, best_spearman)

            if step % args.log_every == 0:
                with torch.no_grad():
                    mu_std = mu.std(dim=0).mean().item()
                print(
                    f"step {step:6d} | loss={accum_loss/args.grad_accum_steps:.4f} "
                    f"rec={accum_rec/args.grad_accum_steps:.4f} kl={accum_kl/args.grad_accum_steps:.4f} "
                    f"beta={beta:.2f} sim={accum_sim/args.grad_accum_steps:.4f} mu_std={mu_std:.4f}"
                )

            should_stop = False
            if step % args.eval_every == 0 or step == args.num_steps:
                sp, pr, n = eval_stsb_dev(
                    model,
                    bert_tok,
                    device,
                    ds_dev,
                    max_len=args.max_len,
                    batch_size=args.eval_batch_size,
                )
                print(f"[DEV] step {step} | STS-B dev | Spearman: {sp:.4f} | Pearson: {pr:.4f} | n={n}")
                torch.save(model.state_dict(), ckpt_latest)
                if sp > best_spearman + args.early_stop_min_delta:
                    best_spearman = sp
                    bad_evals = 0
                    torch.save(model.state_dict(), ckpt_best)
                    print(f"[SAVE] New best Spearman={sp:.4f} -> {ckpt_best}")
                else:
                    if step >= args.early_stop_start_step:
                        bad_evals += 1
                        if args.early_stop_patience > 0 and bad_evals >= args.early_stop_patience:
                            print(
                                f"[EarlyStop] no improvement in {bad_evals} evals "
                                f"(best={best_spearman:.4f})."
                            )
                            should_stop = True
                save_train_checkpoint(ckpt_train, model, sim_head, step, best_spearman)

            prev_stage = stage
            stage = staged_unfreeze(
                model,
                step,
                stage,
                args.unfreeze_stage1_at,
                args.unfreeze_stage2_at,
                args.unfreeze_last_n1,
                args.unfreeze_last_n2,
            )
            if stage != prev_stage:
                optimizer = build_optimizer(model, sim_head, args.lr_main, args.lr_sim, args.lr_dec, args.weight_decay)
                scheduler = build_scheduler(optimizer, warmup_steps, args.num_steps, last_epoch=step - 1)
                print(f"[Opt] rebuild optimizer at step {step}")

            if should_stop:
                break
    except KeyboardInterrupt:
        print(f"[Interrupt] Saving checkpoint at step {current_step}...")
        save_train_checkpoint(ckpt_train, model, sim_head, current_step, best_spearman)
        print(f"[Interrupt] Saved: {ckpt_train}")
        return

    print("Training complete")
    print(f"Latest checkpoint: {ckpt_latest}")
    if os.path.exists(ckpt_best):
        print(f"Best checkpoint: {ckpt_best} (best Spearman={best_spearman:.4f})")


if __name__ == "__main__":
    main()
