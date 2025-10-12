import csv
import math
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ml.utils.cfg import load_config
from ml.utils.metrics import compute_eval_metrics


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


def train(model, device, train_set, val_set, tokenizer):
    cfg = load_config()
    training_cfg = cfg["training"]

    base_dir = f"{training_cfg['checkpoint_dir']}/seg_{cfg['pipeline']['train']['segment_len']}/{cfg['pipeline']['model']}"
    log_path = f"{base_dir}/training_log.csv"

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    if not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "epoch",
                    "train_loss",
                    "val_loss",
                    "teacher_forcing",
                    "accuracy",
                    "f1_macro",
                    "groove_similarity",
                    "pattern_entropy",
                ]
            )

    train_loader = DataLoader(
        train_set,
        batch_size=training_cfg["batch_size"],
        shuffle=True,
        num_workers=2,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=training_cfg["batch_size"],
        shuffle=False,
        num_workers=2,
        persistent_workers=True,
    )

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=training_cfg["learning_rate"],
        weight_decay=training_cfg["weight_decay"],
    )

    print("[Loss] Computing class weights...")
    token_counts = np.zeros(model.vocab_size, dtype=np.int64)

    for batch in train_loader:
        tokens = batch["tokens"].numpy().flatten()
        for t in tokens:
            if t < model.vocab_size:
                token_counts[t] += 1

    token_counts[token_counts == 0] = 1

    weights = 1.0 / token_counts
    weights = weights / weights.mean()  # normalize around 1
    weights = torch.tensor(weights, dtype=torch.float32, device=device)
    print(f"[Loss] Class weights computed for {len(weights)} tokens.")

    crit = nn.CrossEntropyLoss(weight=weights)

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

        # --- Compute additional metrics on a validation subset ---
        model.eval()
        all_preds, all_targets = [], []
        for i, batch in enumerate(val_loader):
            if i > 10:  # only first few batches to save time
                break
            tok = batch["tokens"].to(device)
            pos = batch["positions"].to(device)
            genre = batch["genre_id"].to(device)
            tgt = batch["targets"].to(device)
            logits = model(
                tok, pos, genre, tgt_tokens=tgt, tgt_pos=pos, teacher_forcing=0.0
            )
            preds = logits.argmax(-1).cpu().numpy().flatten()
            targets = tgt.cpu().numpy().flatten()
            all_preds.extend(preds)
            all_targets.extend(targets)

        try:
            extra_metrics = compute_eval_metrics(
                np.array(all_preds), np.array(all_targets), tokenizer
            )
        except Exception as e:
            print(f"[Warning] Metric computation failed: {e}")
            extra_metrics = {}

        # --- Combine and log ---
        metrics = {"train_loss": tr, "val_loss": va, **extra_metrics}
        print(
            f"ep {epoch:02d} | tr {tr:.4f} | val {va:.4f} | tf={tf_ratio:.2f} | acc={metrics['accuracy']:.3f} | f1={metrics['f1_macro']:.3f}"
        )

        # Save to CSV
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch,
                    tr,
                    va,
                    tf_ratio,
                    metrics["accuracy"],
                    metrics["f1_macro"],
                    metrics["groove_similarity"],
                    metrics["pattern_entropy"],
                ]
            )

        # save checkpoints
        save_ckpt(
            f"{base_dir}/seq2seq_epoch_{epoch:03d}.pt",
            model,
            opt,
            epoch,
            {"train": tr, "val": va, **extra_metrics},
        )
        if va < best:
            best = va
            save_ckpt(
                f"{base_dir}/seq2seq_best.pt",
                model,
                opt,
                epoch,
                {"train": tr, "val": va, **extra_metrics},
            )

    return model
