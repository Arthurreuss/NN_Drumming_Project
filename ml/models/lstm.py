import math
import random

import torch
import torch.nn as nn


class Seq2SeqLSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_genres: int,
        token_embed_dim: int,
        pos_embed_dim: int,
        genre_embed_dim: int,
        bpm_embed_dim: int,
        hidden: int,
        layers: int,
        period: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size

        self.token_emb = nn.Embedding(vocab_size, token_embed_dim)
        self.pos_emb = nn.Linear(2, pos_embed_dim)
        self.genre_emb = nn.Embedding(num_genres, genre_embed_dim)
        self.bpm_emb = nn.Linear(1, bpm_embed_dim)

        enc_in = token_embed_dim + pos_embed_dim + genre_embed_dim + bpm_embed_dim
        dec_in = token_embed_dim + pos_embed_dim + genre_embed_dim + bpm_embed_dim

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
        self.period = period

    @staticmethod
    def _phase_from_sincos(sin_cos):  # (B,2) -> (B,)
        # sin_cos[...,0]=sin, sin_cos[...,1]=cos
        return torch.atan2(sin_cos[..., 0], sin_cos[..., 1])  # [-pi, pi]

    def _sincos_vec(self, step_idx, device):  # step_idx: (B,)
        # step_idx in [0, period)
        angle = 2 * math.pi * (step_idx.float() % self.period) / self.period
        sin = torch.sin(angle)
        cos = torch.cos(angle)
        return torch.stack([sin, cos], dim=-1)  # (B,2)

    def encode(self, tokens, pos, genre_id, bpm):
        B, T = tokens.shape
        g = self.genre_emb(genre_id).unsqueeze(1).expand(B, T, -1)
        b = self.bpm_emb(bpm.unsqueeze(1).float()).expand(B, T, -1)

        x = torch.cat([self.token_emb(tokens), self.pos_emb(pos.float()), g, b], dim=-1)
        _, (h, c) = self.encoder(x)
        return (h, c)

    def forward(
        self,
        src_tokens,
        src_pos,
        genre_id,
        unk_id,
        bpm,
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
        h, c = self.encode(src_tokens, src_pos, genre_id, bpm)

        if tgt_tokens is not None and tgt_pos is not None:
            T = tgt_tokens.shape[1]
            logits_out = []
            prev_tok = torch.full((B,), start_token_id, dtype=torch.long, device=device)
            g = self.genre_emb(genre_id)
            for t in range(T):
                pe = self.pos_emb(tgt_pos[:, t].float())
                te = self.token_emb(prev_tok)
                b = self.bpm_emb(bpm.float())
                inp = torch.cat([te, pe, g, b], dim=-1).unsqueeze(1)  # (B,1,D)
                out, (h, c) = self.decoder(inp, (h, c))
                logit = self.proj(out)  # (B,1,V)
                logits_out.append(logit)
                use_tf = random.random() < teacher_forcing
                prev_tok = tgt_tokens[:, t] if use_tf else logit.squeeze(1).argmax(-1)
            return torch.cat(logits_out, dim=1)  # (B,T,V)

        # Inference
        assert max_len is not None, "set max_len for inference"
        # derive current step index from last encoder pos (sin,cos)
        last_sc = src_pos[:, -1, :]  # (B,2)
        phase = self._phase_from_sincos(last_sc)  # (B,)
        # map phase [-pi,pi] -> step idx [0, period)
        step0 = (
            (phase + math.pi) / (2 * math.pi) * self.period
        ).round().long() % self.period

        g = self.genre_emb(genre_id)
        outputs = []
        prev_tok = torch.full((B,), start_token_id, dtype=torch.long, device=device)
        for t in range(max_len):
            step_t = (step0 + t) % self.period  # (B,)
            sc_t = self._sincos_vec(step_t, device)  # (B,2)
            pe = self.pos_emb(sc_t)
            te = self.token_emb(prev_tok)
            b = self.bpm_emb(bpm.float())
            inp = torch.cat([te, pe, g, b], dim=-1).unsqueeze(1)
            out, (h, c) = self.decoder(inp, (h, c))
            logit = self.proj(out)  # (B,1,V)
            logit[:, :, unk_id] = float("-inf")

            prev_tok = logit.squeeze(1).argmax(-1)
            outputs.append(prev_tok.unsqueeze(1))
        return torch.cat(outputs, dim=1)  # (B,max_len)

    @torch.no_grad()
    def generate(
        self,
        prim_tokens,
        prim_pos,
        genre_id,
        unk_id,
        bpm,
        steps: int,
        temperature: float = 1.0,
        start_token_id: int = 1,
    ):
        """
        prim_*: (1,T) tensors to build encoder context
        returns (1, steps) token ids
        """
        self.eval()
        h, c = self.encode(prim_tokens, prim_pos, genre_id, bpm)
        B = prim_tokens.size(0)
        device = prim_tokens.device

        last_sc = prim_pos[:, -1, :]
        phase = self._phase_from_sincos(last_sc)
        step0 = (
            (phase + math.pi) / (2 * math.pi) * self.period
        ).round().long() % self.period

        prev = torch.full(
            (B,), start_token_id, dtype=torch.long, device=prim_tokens.device
        )
        out_tokens = []
        g = self.genre_emb(genre_id)
        for t in range(steps):
            step_t = (step0 + t) % self.period
            sc_t = self._sincos_vec(step_t, device)
            pe = self.pos_emb(sc_t)
            te = self.token_emb(prev)
            b = self.bpm_emb(bpm.float())

            x = torch.cat([te, pe, g, b], dim=-1).unsqueeze(1)
            y, (h, c) = self.decoder(x, (h, c))
            logits = self.proj(y).squeeze(1) / max(1e-6, temperature)
            logits[:, unk_id] = float("-inf")

            probs = torch.softmax(logits, dim=-1)
            prev = torch.multinomial(probs, num_samples=1).squeeze(1)
            out_tokens.append(prev.unsqueeze(1))
        return torch.cat(out_tokens, dim=1)
