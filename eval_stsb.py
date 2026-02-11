import argparse
import csv
import os

import numpy as np
import torch
from datasets import load_dataset
from transformers import BertTokenizer
from scipy.stats import spearmanr, pearsonr

from models.vaegpt import VAEGPT


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


def extract_state_dict(state):
    if isinstance(state, dict):
        if "model" in state:
            return state["model"]
        if "state_dict" in state:
            return state["state_dict"]
    return state


def infer_dims(state_dict):
    latent_size = state_dict["to_mu.weight"].shape[0]
    hidden = state_dict["decoder.transformer.wte.weight"].shape[1]
    prefix_len = state_dict["z_to_embed.weight"].shape[0] // hidden
    return latent_size, prefix_len


def encode_sentences(model, bert_tok, device, sentences, max_len=64, batch_size=64):
    mus = []
    for i in range(0, len(sentences), batch_size):
        chunk = sentences[i:i + batch_size]
        enc = bert_tok(
            chunk,
            return_tensors="pt",
            max_length=max_len,
            truncation=True,
            padding="max_length",
        )
        input_ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)
        mu, _ = model.encode(input_ids, attn)
        mu = torch.nn.functional.normalize(mu, p=2, dim=1)
        mus.append(mu.cpu())
    return torch.cat(mus, dim=0).numpy()


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/vaegpt.pt")
    ap.add_argument("ckpt_pos", nargs="?", default=None)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--bert_dir", default=None)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--local_stsb_dev_tsv", default=None)
    ap.add_argument("--latent_size", type=int, default=None)
    ap.add_argument("--latent_prefix_len", type=int, default=None)
    return ap.parse_args()


def main():
    args = parse_args()

    if args.offline:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    ckpt_path = args.ckpt_pos or args.ckpt
    print(f"Loading checkpoint from: {ckpt_path}")

    state = torch.load(ckpt_path, map_location=device)
    state_dict = extract_state_dict(state)

    latent_size, prefix_len = infer_dims(state_dict)
    if args.latent_size is not None:
        latent_size = args.latent_size
    if args.latent_prefix_len is not None:
        prefix_len = args.latent_prefix_len

    model = VAEGPT(latent_size=latent_size, latent_prefix_len=prefix_len).to(device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print("[load_state_dict] missing:", missing)
        print("[load_state_dict] unexpected:", unexpected)
    model.eval()

    bert_tokenizer = BertTokenizer.from_pretrained(args.bert_dir or "bert-base-uncased", local_files_only=args.offline)

    if args.local_stsb_dev_tsv:
        ds = load_tsv(args.local_stsb_dev_tsv)
    else:
        ds = load_dataset("glue", "stsb")["validation"]

    s1 = [ex["sentence1"] for ex in ds]
    s2 = [ex["sentence2"] for ex in ds]
    y = np.array([ex["label"] for ex in ds], dtype=np.float32)

    z1 = encode_sentences(model, bert_tokenizer, device, s1, max_len=args.max_len, batch_size=args.batch_size)
    z2 = encode_sentences(model, bert_tokenizer, device, s2, max_len=args.max_len, batch_size=args.batch_size)

    cos_scores = np.sum(z1 * z2, axis=1)
    sr, _ = spearmanr(cos_scores, y)
    pr, _ = pearsonr(cos_scores, y)

    print(f"STS-B dev | Spearman: {sr:.4f} | Pearson: {pr:.4f} | n={len(y)}")


if __name__ == "__main__":
    main()
