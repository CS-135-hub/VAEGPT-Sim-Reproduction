import argparse
import os
import torch
import torch.nn.functional as F
from models.vaegpt import VAEGPT
from prepare_dataset import build_tokenizers

# transformers>=4.53 cache API
try:
    from transformers.cache_utils import DynamicCache
    HAS_CACHE = True
except Exception:
    DynamicCache = None
    HAS_CACHE = False


def extract_state_dict(state):
    if isinstance(state, dict):
        if "model" in state:
            return state["model"]
        if "state_dict" in state:
            return state["state_dict"]
    return state


def infer_dims(state_dict):#自动推断模型各个关键维度大小
    latent_size = state_dict["to_mu.weight"].shape[0]
    hidden = state_dict["decoder.transformer.wte.weight"].shape[1]
    prefix_len = state_dict["z_to_embed.weight"].shape[0] // hidden
    return latent_size, prefix_len


def adapt_single_prefix_ckpt_to_multi(state_dict, model):
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


def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float("Inf")):#剪概率分布，在采样前，把“不该选的 token”从概率分布里直接干掉
    logits = logits.clone()
    if top_k and top_k > 0:
        top_k = min(top_k, logits.size(-1))
        thresh = torch.topk(logits, top_k)[0][..., -1, None]
        logits[logits < thresh] = filter_value
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumprobs = probs.cumsum(dim=-1)
        remove = cumprobs > top_p
        remove[..., 0] = False
        indices_to_remove = remove.scatter(1, sorted_idx, remove)
        logits[indices_to_remove] = filter_value
    return logits


def ban_repeating_ngram(cur_logits, generated, ngram_size: int):
    if ngram_size <= 1 or len(generated) < ngram_size - 1:
        return cur_logits
    ngram_set = set()
    for i in range(len(generated) - ngram_size + 1):
        ngram_set.add(tuple(generated[i:i + ngram_size]))

    prefix = tuple(generated[-(ngram_size - 1):])
    for ng in ngram_set:
        if ng[:-1] == prefix:
            cur_logits[0, ng[-1]] = -float("inf")
    return cur_logits


@torch.no_grad()
def generate_stepwise(#调用 generate_once_stepwise，循环生成完整序列
    model,
    gpt2_tok,
    z_embed,
    prompt_ids=None,
    max_new_tokens=50,
    temperature=1.0,
    top_p=1.0,
    top_k=0,
    do_sample=True,
    eos_token_id=None,
    device="cpu",
    repetition_penalty=1.0,
    no_repeat_ngram_size=3,
    block_repeat_token=True,
):
    model.eval()

    if prompt_ids is not None:
        prompt_emb = model.decoder.transformer.wte(prompt_ids)
        inputs_embeds = torch.cat([z_embed, prompt_emb], dim=1)
    else:
        inputs_embeds = z_embed

    attn_mask = torch.ones(inputs_embeds.size()[:-1], dtype=torch.long, device=device)

    outputs = model.decoder(
        inputs_embeds=inputs_embeds,
        attention_mask=attn_mask,
        use_cache=True,
        return_dict=True,
    )

    past = outputs.past_key_values
    if HAS_CACHE and not isinstance(past, DynamicCache):
        past = DynamicCache.from_legacy_cache(past)

    logits = outputs.logits[:, -1, :]

    generated = []
    for _ in range(max_new_tokens):
        cur_logits = logits / max(temperature, 1e-8)
        cur_logits = top_k_top_p_filtering(cur_logits, top_k=top_k, top_p=top_p)
        backup_logits = cur_logits.clone()

        if no_repeat_ngram_size and no_repeat_ngram_size >= 2:
            cur_logits = ban_repeating_ngram(cur_logits, generated, no_repeat_ngram_size)

        if block_repeat_token and len(generated) >= 1:
            last_tok = generated[-1]
            cur_logits[0, last_tok] = -float("inf")

        if repetition_penalty and repetition_penalty > 1.0 and len(generated) > 0:
            penalty = torch.log(torch.tensor(repetition_penalty, device=device, dtype=cur_logits.dtype))
            for tok in generated:
                if 0 <= tok < cur_logits.size(-1) and torch.isfinite(cur_logits[0, tok]):
                    cur_logits[0, tok] -= penalty

        if torch.isinf(cur_logits).all():
            cur_logits = backup_logits

        cur_logits = torch.nan_to_num(cur_logits, nan=-1e9, posinf=1e9, neginf=-1e9)

        if do_sample:
            probs = F.softmax(cur_logits, dim=-1)
            if (not torch.isfinite(probs).all()) or probs.sum().item() == 0.0:
                next_token = cur_logits.argmax(dim=-1, keepdim=True)
            else:
                next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = cur_logits.argmax(dim=-1, keepdim=True)

        generated.append(next_token.item())

        if eos_token_id is not None and next_token.item() == eos_token_id:
            break

        outputs = model.decoder(
            input_ids=next_token,
            past_key_values=past,
            use_cache=True,
            return_dict=True,
        )
        past = outputs.past_key_values
        logits = outputs.logits[:, -1, :]

    if len(generated) == 0:
        return torch.empty((inputs_embeds.size(0), 0), dtype=torch.long, device=device)
    return torch.tensor(generated, device=device).unsqueeze(0)


@torch.no_grad()
def generate_once_stepwise(#只生成“下一个 token
    model,
    bert_tok,
    gpt2_tok,
    src_text,
    prompt="",
    max_len=64,
    max_new_tokens=40,
    temperature=0.9,
    top_p=0.9,
    top_k=50,
    do_sample=True,
    use_mu=True,
    device="cpu",
    repetition_penalty=1.0,
    no_repeat_ngram_size=3,
    block_repeat_token=True,
):
    enc = bert_tok(src_text, return_tensors="pt", truncation=True, padding=True, max_length=max_len)
    input_ids = enc["input_ids"].to(device)
    attn_mask = enc["attention_mask"].to(device)

    mu, logvar = model.encode(input_ids, attn_mask)

    if use_mu:
        z = mu
    else:
        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std

    Lz = model.latent_prefix_len
    H = model.decoder.config.n_embd
    z_embed = model.z_to_embed(z).view(z.size(0), Lz, H)

    text_prompt = (prompt or "") + (src_text if src_text else "")
    prompt_ids = None
    if text_prompt.strip():
        prompt_ids = gpt2_tok(text_prompt, return_tensors="pt").input_ids.to(device)

    gen_ids = generate_stepwise(
        model=model,
        gpt2_tok=gpt2_tok,
        z_embed=z_embed,
        prompt_ids=prompt_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        do_sample=do_sample,
        eos_token_id=gpt2_tok.eos_token_id,
        device=device,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        block_repeat_token=block_repeat_token,
    )

    return gpt2_tok.decode(gen_ids[0], skip_special_tokens=True)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/vaegpt.pt")
    ap.add_argument("--src", required=True)
    ap.add_argument("--prompt", default="Paraphrase the following sentence in natural English:\n")
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--max_new_tokens", type=int, default=24)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.85)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--num_return_sequences", type=int, default=1)
    ap.add_argument("--bert_dir", default=None)
    ap.add_argument("--gpt2_dir", default=None)
    ap.add_argument("--repetition_penalty", type=float, default=1.4)
    ap.add_argument("--no_repeat_ngram_size", type=int, default=4)
    ap.add_argument("--stop_on_repeat_line", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--disable_repeat_token_block", action="store_true")
    ap.add_argument("--latent_size", type=int, default=None)
    ap.add_argument("--latent_prefix_len", type=int, default=None)
    ap.add_argument("--offline", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()

    if args.offline:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    state = torch.load(args.ckpt, map_location=device)
    state_dict = extract_state_dict(state)

    latent_size, prefix_len = infer_dims(state_dict)
    if args.latent_size is not None:
        latent_size = args.latent_size
    if args.latent_prefix_len is not None:
        prefix_len = args.latent_prefix_len

    model = VAEGPT(
        latent_size=latent_size,
        latent_prefix_len=prefix_len,
        bert_name_or_dir=args.bert_dir or "bert-base-uncased",
        gpt2_name_or_dir=args.gpt2_dir or "gpt2",
    ).to(device)

    state_dict = adapt_single_prefix_ckpt_to_multi(state_dict, model)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print("[load_state_dict] missing:", missing)
        print("[load_state_dict] unexpected:", unexpected)

    model.eval()

    bert_tok, gpt2_tok = build_tokenizers(
        bert_name_or_dir=args.bert_dir or "bert-base-uncased",
        gpt2_name_or_dir=args.gpt2_dir or "gpt2",
        offline=args.offline,
    )

    use_mu = (args.sample == 0)
    do_sample = (args.sample == 1)

    def dedup_repeat_lines(text: str) -> str:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            return text.strip()
        deduped = [lines[0]]
        for ln in lines[1:]:
            if ln == deduped[-1]:
                break
            deduped.append(ln)
        return "\n".join(deduped)

    for i in range(max(1, args.num_return_sequences)):
        text = generate_once_stepwise(
            model,
            bert_tok,
            gpt2_tok,
            src_text=args.src,
            prompt=args.prompt,
            max_len=args.max_len,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            do_sample=do_sample,
            use_mu=use_mu,
            device=device,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            block_repeat_token=(not args.disable_repeat_token_block),
        )
        if args.stop_on_repeat_line:
            text = dedup_repeat_lines(text)
        print(f"\n=== Generation #{i+1} ===")
        print(text)


if __name__ == "__main__":
    main()
