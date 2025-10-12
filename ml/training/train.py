import math
import os
import re

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ml.data.dataset import DrumDataset
from ml.utils.cfg import load_config


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


def train(model):
    cfg = load_config()
    training_cfg = cfg["training"]

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    train_set = DrumDataset(training_cfg["train_dir"], include_genre=True)
    val_set = DrumDataset(training_cfg["test_dir"], include_genre=True)

    train_loader = DataLoader(
        train_set, batch_size=256, shuffle=True, num_workers=2, persistent_workers=True
    )
    val_loader = DataLoader(
        val_set, batch_size=256, shuffle=False, num_workers=2, persistent_workers=True
    )

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=training_cfg["learning_rate"],
        weight_decay=training_cfg["weight_decay"],
    )
    crit = nn.CrossEntropyLoss()

    best = math.inf
    for epoch in range(1, training_cfg["epochs"] + 1):
        tf_ratio = linear_tf(
            epoch - 1,
            training_cfg["epochs"],
            training_cfg["teacher_forcing_start"],
            training_cfg["teacher_forcing_end"],
        )
        tr = train_epoch(model, train_loader, opt, crit, device, tf_ratio)
        va = eval_epoch(model, val_loader, crit, device)
        save_ckpt(
            f"{training_cfg['checkpoint_dir']}/seq2seq_epoch_{epoch:03d}.pt",
            model,
            opt,
            epoch,
            {"train": tr, "val": va},
        )
        if va < best:
            best = va
            save_ckpt(
                f"{training_cfg['checkpoint_dir']}/seq2seq_best.pt",
                model,
                opt,
                epoch,
                {"train": tr, "val": va},
            )
        print(f"ep {epoch:02d}  train {tr:.4f}  val {va:.4f}  tf={tf_ratio:.2f}")
    return model
