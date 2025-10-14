import csv
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ml.data.dataset import DrumDataset
from ml.data.tokenizer import SimpleTokenizer
from ml.models.lstm import Seq2SeqLSTM
from ml.utils.cfg import load_config
from ml.utils.metrics import compute_eval_metrics


@torch.no_grad()
def evaluate_model(model, loader, device, tokenizer):
    """Run full evaluation on a dataset loader."""
    model.eval()
    all_preds, all_targets = [], []

    for batch in tqdm(loader, desc="Evaluating"):
        tok = batch["tokens"].to(device)
        pos = batch["positions"].to(device)
        genre = batch["genre_id"].to(device)
        tgt = batch["targets"].to(device)

        logits = model(
            tok,
            pos,
            genre,
            tgt_tokens=tgt,
            tgt_pos=pos,
            teacher_forcing=0.0,
            unk_id=tokenizer.unk_id,
        )
        preds = logits.argmax(-1).cpu().numpy().flatten()
        targets = tgt.cpu().numpy().flatten()

        all_preds.extend(preds)
        all_targets.extend(targets)

    metrics = compute_eval_metrics(
        np.array(all_preds), np.array(all_targets), tokenizer
    )
    return metrics
