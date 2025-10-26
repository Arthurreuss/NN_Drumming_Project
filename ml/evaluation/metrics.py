import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def token_metrics(targets, preds):
    """Standard token-level metrics."""
    return {
        "accuracy": accuracy_score(targets, preds),
        "f1_macro": f1_score(targets, preds, average="macro", zero_division=0),
        "precision": precision_score(targets, preds, average="macro", zero_division=0),
        "recall": recall_score(targets, preds, average="macro", zero_division=0),
    }


def compute_eval_metrics(preds, targets, tokenizer=None):
    result = token_metrics(targets, preds)
    return result
