import torch
import torch.nn as nn
from transformers import BertModel, GPT2LMHeadModel

class VAEGPT(nn.Module):
    """
    VAE encoder (BERT) -> latent (mu, logvar) -> z -> multi-token prefix embeddings -> GPT-2 decoder
    """
    def __init__(self, latent_size=32, latent_prefix_len=8, bert_name_or_dir="bert-base-uncased", gpt2_name_or_dir="gpt2"):
        super().__init__()
        # ----- Encoder -----
        self.encoder = BertModel.from_pretrained(bert_name_or_dir)
        hidden_size = self.encoder.config.hidden_size  # 768 for bert-base

        self.to_mu = nn.Linear(hidden_size, latent_size)
        self.to_logvar = nn.Linear(hidden_size, latent_size)

        # ----- Decoder -----
        self.decoder = GPT2LMHeadModel.from_pretrained(gpt2_name_or_dir)
        gpt_hid = self.decoder.config.n_embd  # 768 for gpt2 small

        # ----- Multi-token latent prefix -----
        self.latent_prefix_len = latent_prefix_len
        # Map z (B, latent) -> (B, Lz * H) and then view to (B, Lz, H)
        self.z_to_embed = nn.Linear(latent_size, gpt_hid * latent_prefix_len)

        self.ignore_index = -100

    def encode(self, input_ids, attention_mask):
        enc_out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = enc_out.last_hidden_state[:, 0]  # [B, H]
        mu = self.to_mu(cls)
        logvar = self.to_logvar(cls)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask=None, use_cache=False):
        """
        Returns
            logits: [B, Lz+L, V]
            mu, logvar
        """
        mu, logvar = self.encode(input_ids, attention_mask)
        z = self.reparameterize(mu, logvar)  # [B, latent]

        B = z.size(0)
        H = self.decoder.config.n_embd
        Lz = self.latent_prefix_len

        # z -> [B, Lz, H]
        z_embed = self.z_to_embed(z).view(B, Lz, H)

        # decoder token embeddings
        tok_embed = self.decoder.transformer.wte(decoder_input_ids)  # [B, L, H]
        inputs_embeds = torch.cat([z_embed, tok_embed], dim=1)       # [B, Lz+L, H]

        if decoder_attention_mask is None:
            decoder_attention_mask = torch.ones_like(decoder_input_ids)
        prefix_mask = torch.ones((B, Lz), dtype=decoder_attention_mask.dtype, device=decoder_attention_mask.device)
        attn_mask = torch.cat([prefix_mask, decoder_attention_mask], dim=1)

        outputs = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            use_cache=use_cache,
            return_dict=True,
        )
        logits = outputs.logits
        return logits, mu, logvar
