import math
import os
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# ----------------------- Model -----------------------


class Seq2SeqLSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pos_vocab_size: int,
        num_genres: int,
        token_embed_dim: int = 128,
        pos_embed_dim: int = 8,
        genre_embed_dim: int = 8,
        hidden: int = 256,
        layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size

        self.token_emb = nn.Embedding(vocab_size, token_embed_dim)
        self.pos_emb = nn.Embedding(pos_vocab_size, pos_embed_dim)
        self.genre_emb = nn.Embedding(num_genres, genre_embed_dim)

        enc_in = token_embed_dim + pos_embed_dim + genre_embed_dim
        dec_in = token_embed_dim + pos_embed_dim + genre_embed_dim

        self.encoder = nn.LSTM(
            input_size=enc_in,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.decoder = nn.LSTM(
            input_size=dec_in,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.proj = nn.Linear(hidden, vocab_size)

    def encode(self, tokens, pos, genre_id):
        B, T = tokens.shape
        g = self.genre_emb(genre_id).unsqueeze(1).expand(B, T, -1)
        x = torch.cat([self.token_emb(tokens), self.pos_emb(pos), g], dim=-1)
        _, (h, c) = self.encoder(x)
        return (h, c)

    def forward(
        self,
        src_tokens,
        src_pos,
        genre_id,
        tgt_tokens=None,
        tgt_pos=None,
        teacher_forcing: float = 0.5,
        max_len: int = None,
        start_token_id: int = 1,  # define in your vocab
    ):
        """
        Train: provide tgt_tokens,tgt_pos → returns logits (B,T,V)
        Inference: omit tgt_* → greedy decode length=max_len
        """
        B, Tsrc = src_tokens.shape
        device = src_tokens.device
        h, c = self.encode(src_tokens, src_pos, genre_id)

        if tgt_tokens is not None and tgt_pos is not None:
            T = tgt_tokens.shape[1]
            logits_out = []
            # first decoder input = <start> (use provided first target token if you prefer)
            prev_tok = torch.full((B,), start_token_id, dtype=torch.long, device=device)
            for t in range(T):
                g = self.genre_emb(genre_id)
                pe = self.pos_emb(tgt_pos[:, t])
                te = self.token_emb(prev_tok)
                inp = torch.cat([te, pe, g], dim=-1).unsqueeze(1)  # (B,1,D)
                out, (h, c) = self.decoder(inp, (h, c))
                logit = self.proj(out)  # (B,1,V)
                logits_out.append(logit)
                use_tf = random.random() < teacher_forcing
                prev_tok = tgt_tokens[:, t] if use_tf else logit.squeeze(1).argmax(-1)
            return torch.cat(logits_out, dim=1)  # (B,T,V)

        # Inference
        assert max_len is not None, "set max_len for inference"
        outputs = []
        prev_tok = torch.full((B,), start_token_id, dtype=torch.long, device=device)
        for t in range(max_len):
            g = self.genre_emb(genre_id)
            # if you have decoder positions for generation, pass them; else reuse src_pos[:, -1] + t % S
            pe = self.pos_emb(src_pos[:, -1] + (t % self.pos_emb.num_embeddings))
            te = self.token_emb(prev_tok)
            inp = torch.cat([te, pe, g], dim=-1).unsqueeze(1)
            out, (h, c) = self.decoder(inp, (h, c))
            logit = self.proj(out)  # (B,1,V)
            prev_tok = logit.squeeze(1).argmax(-1)
            outputs.append(prev_tok.unsqueeze(1))
        return torch.cat(outputs, dim=1)  # (B,max_len)
