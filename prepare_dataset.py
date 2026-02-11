import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, GPT2Tokenizer


def build_tokenizers(bert_name_or_dir="bert-base-uncased", gpt2_name_or_dir="gpt2", offline=False):
    bert_tokenizer = BertTokenizer.from_pretrained(bert_name_or_dir, local_files_only=offline)
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_name_or_dir, local_files_only=offline)
    if gpt2_tokenizer.pad_token is None:
        gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
    return bert_tokenizer, gpt2_tokenizer

class SentencePairDataset(Dataset):
    def __init__(self, data, bert_tokenizer, gpt2_tokenizer, max_len=64):
        self.samples = []

        for example in data:
            s1, s2 = example["sentence1"], example["sentence2"]

            # 编码输入句子（喂给编码器）
            bert_enc = bert_tokenizer(
                s1,
                return_tensors="pt",
                max_length=max_len,
                truncation=True,
                padding="max_length",
            )

            # 构造 decoder input（目标句子）
            gpt2_enc = gpt2_tokenizer(
                s2,
                return_tensors="pt",
                max_length=max_len,
                truncation=True,
                padding="max_length",
            )

            self.samples.append({
                "input_ids": bert_enc["input_ids"].squeeze(0),
                "attention_mask": bert_enc["attention_mask"].squeeze(0),
                "decoder_input_ids": gpt2_enc["input_ids"].squeeze(0),
                "decoder_attention_mask": gpt2_enc["attention_mask"].squeeze(0),
                "label": torch.tensor(example["label"], dtype=torch.float),
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
