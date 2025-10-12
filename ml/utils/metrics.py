import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


# ---- Core Metric Helpers ----
def token_metrics(targets, preds):
    """Standard token-level metrics."""
    return {
        "accuracy": accuracy_score(targets, preds),
        "f1_macro": f1_score(targets, preds, average="macro", zero_division=0),
        "precision": precision_score(targets, preds, average="macro", zero_division=0),
        "recall": recall_score(targets, preds, average="macro", zero_division=0),
    }


def hit_precision_recall(pred_matrix, true_matrix, threshold=0.5):
    """Per-instrument binary precision and recall."""
    pred_bin = (pred_matrix > threshold).astype(int)
    true_bin = (true_matrix > threshold).astype(int)
    eps = 1e-9
    precision = np.sum(pred_bin * true_bin, axis=0) / (np.sum(pred_bin, axis=0) + eps)
    recall = np.sum(pred_bin * true_bin, axis=0) / (np.sum(true_bin, axis=0) + eps)
    return precision.mean(), recall.mean()


def groove_similarity(pred_matrix, true_matrix):
    """Cosine similarity between flattened binary onset patterns."""
    pred_flat = pred_matrix.flatten()
    true_flat = true_matrix.flatten()
    num = np.dot(pred_flat, true_flat)
    denom = np.linalg.norm(pred_flat) * np.linalg.norm(true_flat)
    return num / (denom + 1e-9)


def pattern_entropy(matrix):
    """Shannon entropy across timesteps (measures rhythmic diversity)."""
    p = matrix.mean(axis=0)
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -np.sum(p * np.log2(p) + (1 - p) * np.log2(1 - p))


# ---- High-level Wrapper ----
def compute_eval_metrics(preds, targets, tokenizer=None):
    """
    Combines standard and musical metrics.
    preds/targets: flattened numpy arrays of token IDs.
    tokenizer: optional, if you want to compute musical metrics via detokenization.
    """
    result = token_metrics(targets, preds)

    if tokenizer is not None:
        # Compute musical metrics on a small subset
        N = min(100, len(preds))
        groove_sims, entropies, hit_precs, hit_recs = [], [], [], []

        for i in range(N):
            try:
                p_vec = tokenizer.detokenize(int(preds[i]))
                t_vec = tokenizer.detokenize(int(targets[i]))
            except Exception:
                continue
            p_vec = np.array(p_vec)
            t_vec = np.array(t_vec)
            gp = groove_similarity(p_vec, t_vec)
            hp, hr = hit_precision_recall(p_vec, t_vec)
            he = pattern_entropy(p_vec)
            groove_sims.append(gp)
            hit_precs.append(hp)
            hit_recs.append(hr)
            entropies.append(he)

        result.update(
            {
                "groove_similarity": np.mean(groove_sims),
                "hit_precision": np.mean(hit_precs),
                "hit_recall": np.mean(hit_recs),
                "pattern_entropy": np.mean(entropies),
            }
        )

    return result
