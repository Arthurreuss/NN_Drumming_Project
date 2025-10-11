# pytorch_lstm_drum.py
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from ml.data.dataset import DrumDataset
from torch.utils.data import DataLoader, Dataset, random_split

# -------- Config --------
SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

TIMESTEPS = 63  # required
FEATURES = 9  # required
BATCH_SMALL = 32  # for sanity overfit
BATCH_FULL = 128
LR = 1e-3
EPOCHS_SANITY = 200
EPOCHS_FULL = 30
WEIGHT_DECAY = 1e-4
CKPT_DIR = "./checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)


# -------- Model --------
class LSTMNextStep(nn.Module):
    """
    One LSTM over the sequence; linear head at each step.
    Input  : (B, 63, 9)
    Output : (B, 63, 9) logits (use BCEWithLogitsLoss for multi-label hits)
    """

    def __init__(self, input_size=FEATURES, hidden_size=128, num_layers=2, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        # x: (B, T, F)
        out, _ = self.lstm(x)  # (B, T, H)
        logits = self.head(out)  # (B, T, F)
        return logits


def save_ckpt(path, model, optim, epoch, metrics):
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optim_state": optim.state_dict(),
            "metrics": metrics,
        },
        path,
    )


@torch.no_grad()
def eval_loss(model, loader, criterion):
    model.eval()
    loss_sum = 0.0
    n = 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss_sum += loss.item() * xb.size(0)
        n += xb.size(0)
    return loss_sum / max(1, n)


def train_epoch(model, loader, optim, criterion, clip=1.0, amp=True):
    model.train()
    total = 0.0
    n = 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optim.zero_grad()
        if amp:
            with torch.autocast(device_type="mps", dtype=torch.float16):
                logits = model(xb)
                loss = criterion(logits, yb)
        else:
            logits = model(xb)
            loss = criterion(logits, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optim.step()
        total += loss.item() * xb.size(0)
        n += xb.size(0)
    return total / max(1, n)


# -------- Main --------
if __name__ == "__main__":
    # 1) Load/prepare data
    # X = np.load("your_sequences.npy")  # shape (N, 63, 9)
    dataset = DrumDataset(
        "dataset/processed/q_16/seg_64/jazz", "./config.yaml", include_genre=False
    )

    # Split
    val_frac = 0.1
    val_size = int(len(dataset) * val_frac)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(SEED)
    )

    # --- Small model, sanity overfit ---
    small = LSTMNextStep(hidden_size=128, num_layers=1).to(DEVICE)
    crit = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(small.parameters(), lr=1e-3, weight_decay=1e-4)

    SANITY_N = 1024
    small_train, _ = random_split(
        train_set,
        [min(SANITY_N, len(train_set)), len(train_set) - min(SANITY_N, len(train_set))],
        generator=torch.Generator().manual_seed(SEED),
    )
    dl_sanity = DataLoader(
        small_train,
        batch_size=128,
        shuffle=True,
        num_workers=2,
        persistent_workers=True,
    )
    dl_sanity_val = DataLoader(val_set, batch_size=1024, shuffle=False)

    # print("Sanity overfit (small model)...")
    # best = float("inf")
    # for e in range(20):
    #     tr = train_epoch(small, dl_sanity, opt, crit, clip=1.0)
    #     va = eval_loss(small, dl_sanity_val, crit)
    #     if va < best:
    #         best = va
    #         save_ckpt(
    #             os.path.join(CKPT_DIR, "small_sanity_best.pt"),
    #             small,
    #             opt,
    #             e + 1,
    #             {"train": tr, "val": va},
    #         )
    #     print(f"[sanity] ep{e+1:02d} train {tr:.4f} val {va:.4f}")

    # --- Bigger model, full training ---
    big = LSTMNextStep(hidden_size=256, num_layers=3, dropout=0.1).to(DEVICE)
    crit_big = nn.BCEWithLogitsLoss()
    opt_big = torch.optim.AdamW(big.parameters(), lr=8e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_big, factor=0.5, patience=3)

    dl_train = DataLoader(
        train_set, batch_size=256, shuffle=True, num_workers=4, persistent_workers=True
    )
    dl_val = DataLoader(
        val_set, batch_size=256, shuffle=False, num_workers=2, persistent_workers=True
    )

    print("Full training (big model)...")
    best = float("inf")
    for e in range(EPOCHS_FULL):
        tr = train_epoch(
            big, dl_train, opt_big, crit_big, clip=1.0, amp=(DEVICE.type == "cuda")
        )
        va = eval_loss(big, dl_val, crit_big)
        sched.step(va)
        save_ckpt(
            os.path.join(CKPT_DIR, f"big_jazz_epoch_{e+1:03d}.pt"),
            big,
            opt_big,
            e + 1,
            {"train": tr, "val": va},
        )
        if va < best:
            best = va
            save_ckpt(
                os.path.join(CKPT_DIR, "big_jazz_best.pt"),
                big,
                opt_big,
                e + 1,
                {"train": tr, "val": va},
            )
        print(
            f"[full ] ep{e+1:02d} train {tr:.4f} val {va:.4f} lr={opt_big.param_groups[0]['lr']:.2e}"
        )

    print("Checkpoints in", CKPT_DIR)
