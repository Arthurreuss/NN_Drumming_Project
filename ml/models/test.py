import math
import os
import token
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ml.data.tokenizer import SimpleTokenizer
from ml.models.lstm import Seq2SeqLSTM
from utils.cfg import load_config


@dataclass
class TrainConfig:
    batch_size: int = 256
    lr: float = 5e-4
    weight_decay: float = 1e-4
    epochs: int = 20
    teacher_forcing_start: float = 0.9
    teacher_forcing_end: float = 0.2


def linear_tf(epoch, epochs, tf_start, tf_end):
    a = (tf_end - tf_start) / max(1, epochs - 1)
    return tf_start + a * epoch


def train_epoch(model, loader, opt, crit, device, tf_ratio):
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        tok = batch["tokens"].to(device)  # (B,T)
        pos = batch["positions"].to(device)  # (B,T)
        genre = batch["genre_id"].to(device)  # (B,)
        tgt = batch["targets"].to(device)  # (B,T)

        opt.zero_grad()
        logits = model(
            tok, pos, genre, tgt_tokens=tgt, tgt_pos=pos, teacher_forcing=tf_ratio
        )
        # CE expects (N,C) and targets (N,)
        loss = crit(logits.reshape(-1, model.vocab_size), tgt.reshape(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        total += loss.item() * tok.size(0)
        n += tok.size(0)
    return total / n


@torch.no_grad()
def eval_epoch(model, loader, crit, device):
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        tok = batch["tokens"].to(device)
        pos = batch["positions"].to(device)
        genre = batch["genre_id"].to(device)
        tgt = batch["targets"].to(device)
        logits = model(
            tok, pos, genre, tgt_tokens=tgt, tgt_pos=pos, teacher_forcing=0.0
        )
        loss = crit(logits.reshape(-1, model.vocab_size), tgt.reshape(-1))
        total += loss.item() * tok.size(0)
        n += tok.size(0)
    return total / n


def save_ckpt(path, model, opt, epoch, metrics: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optim_state": opt.state_dict(),
            "metrics": metrics,
        },
        path,
    )


# ----------------------- Example wiring -----------------------
# Assumes your DrumDataset returns dict with keys: tokens, positions, genre_id, targets

if __name__ == "__main__":
    from ml.data.dataset import DrumDataset

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    # Datasets
    train_dir = "dataset/processed/q_16/seg_64/train"
    val_dir = "dataset/processed/q_16/seg_64/test"

    train_set = DrumDataset(train_dir, include_genre=True)
    val_set = DrumDataset(val_dir, include_genre=True)

    # Infer vocab sizes from data
    # Scan a few files for max token and position ids
    tokenizer = SimpleTokenizer()
    cfg = load_config()

    vocab_size, pos_vocab_size = (
        len(tokenizer) + 1,
        cfg["dataset"]["quantization"] + 1,
    )  # maybe segment length?
    num_genres = len(train_set.genres)

    model = Seq2SeqLSTM(
        vocab_size=vocab_size,
        pos_vocab_size=pos_vocab_size,
        num_genres=num_genres,
        token_embed_dim=128,
        pos_embed_dim=8,
        genre_embed_dim=8,
        hidden=256,
        layers=2,
        dropout=0.1,
    ).to(device)

    train_loader = DataLoader(
        train_set, batch_size=256, shuffle=True, num_workers=2, persistent_workers=True
    )
    val_loader = DataLoader(
        val_set, batch_size=256, shuffle=False, num_workers=2, persistent_workers=True
    )

    cfg = TrainConfig()
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    crit = nn.CrossEntropyLoss()

    best = math.inf
    for epoch in range(1, cfg.epochs + 1):
        tf_ratio = linear_tf(
            epoch - 1, cfg.epochs, cfg.teacher_forcing_start, cfg.teacher_forcing_end
        )
        tr = train_epoch(model, train_loader, opt, crit, device, tf_ratio)
        va = eval_epoch(model, val_loader, crit, device)
        save_ckpt(
            f"checkpoints/seq2seq_epoch_{epoch:03d}.pt",
            model,
            opt,
            epoch,
            {"train": tr, "val": va},
        )
        if va < best:
            best = va
            save_ckpt(
                "checkpoints/seq2seq_best.pt",
                model,
                opt,
                epoch,
                {"train": tr, "val": va},
            )
        print(f"ep {epoch:02d}  train {tr:.4f}  val {va:.4f}  tf={tf_ratio:.2f}")

    # ----------------------- Generation -----------------------
    @torch.no_grad()
    def generate(
        model,
        prim_tokens,
        prim_pos,
        genre_id,
        steps: int,
        temperature: float = 1.0,
        start_token_id: int = 1,
    ):
        """
        prim_*: (1,T) tensors to build encoder context
        returns (1, steps) token ids
        """
        model.eval()
        h, c = model.encode(prim_tokens, prim_pos, genre_id)
        B = prim_tokens.size(0)
        prev = torch.full(
            (B,), start_token_id, dtype=torch.long, device=prim_tokens.device
        )
        out_tokens = []
        for t in range(steps):
            pe = model.pos_emb((prim_pos[:, -1] + t) % model.pos_emb.num_embeddings)
            g = model.genre_emb(genre_id)
            te = model.token_emb(prev)
            x = torch.cat([te, pe, g], dim=-1).unsqueeze(1)
            y, (h, c) = model.decoder(x, (h, c))
            logits = model.proj(y).squeeze(1) / max(1e-6, temperature)
            probs = torch.softmax(logits, dim=-1)
            prev = torch.multinomial(probs, num_samples=1).squeeze(1)
            out_tokens.append(prev.unsqueeze(1))
        return torch.cat(out_tokens, dim=1)


# Example use after training:
# sample = train_set[0]
# prim_tok = sample["tokens"].unsqueeze(0).to(device)     # (1,T)
# prim_pos = sample["positions"].unsqueeze(0).to(device)  # (1,T)
# genre_id = sample["genre_id"].unsqueeze(0).to(device)   # (1,)
# gen = generate(model, prim_tok, prim_pos, genre_id, steps=32, temperature=0.9)
# np.save("generated_tokens.npy", gen.cpu().numpy())
