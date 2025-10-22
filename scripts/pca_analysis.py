import os
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from torch import ge
from tqdm import tqdm

from ml.utils.cfg import load_config


def count_tokens_by_genre(dataset_dir, genres):
    """
    Count token frequencies overall and per genre across all .npz files.
    Each .npz file must include 'tokens' and 'genre_id' fields.
    Returns:
        total_counter: Counter of all token frequencies
        genre_counters: dict[genre_id] -> Counter of token frequencies in that genre
    """
    dataset_dir = Path(dataset_dir)
    files = list(dataset_dir.rglob("*.npz"))
    total_counter = Counter()
    genre_counters = {g: Counter() for g in genres}

    print(f"Scanning {len(files)} files under {dataset_dir}...")

    for f in tqdm(files, desc=f"Counting tokens in {dataset_dir.name}"):
        data = np.load(f, allow_pickle=True)
        tokens = data["tokens"]
        genre = str(data["genre"].item())
        total_counter.update(tokens.tolist())

        if genre is not None:
            genre_counters[genre].update(tokens.tolist())

    for counter in genre_counters.values():
        del counter[0]

    return total_counter, genre_counters


def summarize_by_genre(total_counter, genre_counters, genre_labels=None, top_n=20):
    total = sum(total_counter.values())
    print(f"\nTotal tokens: {total:,}")
    print(f"Unique tokens: {len(total_counter):,}")

    # --- per-genre summary ---
    print("\nPer-genre token stats:")
    for gid, counter in genre_counters.items():
        name = genre_labels.get(gid, f"Genre {gid}") if genre_labels else f"Genre {gid}"
        top_tokens = ", ".join(f"{t}:{f}" for t, f in counter.most_common(5))
        print(f"  {name:<12} -> {len(counter):4} unique tokens, top: {top_tokens}")

    genres = sorted(genre_counters.keys())
    token_ids = sorted(total_counter.keys())

    heatmap = np.zeros((len(genres), len(token_ids)))
    for i, gid in enumerate(genres):
        for j, tid in enumerate(token_ids):
            heatmap[i, j] = genre_counters[gid][tid]


import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def assign_token_genres(genre_counters, min_ratio=1.05, min_total=10):
    """
    Returns:
        assignments: dict[token_id] -> dominant_genre | 'none'
        all_tokens: sorted list of token IDs (excluding 0 if missing)
        genres: list of genre names
    """
    # collect all token IDs (should already exclude 0)
    all_tokens = sorted({t for c in genre_counters.values() for t in c})
    genres = list(genre_counters.keys())

    # build frequency matrix [tokens x genres]
    freq = np.zeros((len(all_tokens), len(genres)))
    for j, g in enumerate(genres):
        for t, f in genre_counters[g].items():
            if t == 0:
                continue
            i = all_tokens.index(t)
            freq[i, j] = f

    assignments = {}
    for i, t in enumerate(all_tokens):
        row = freq[i]
        total = row.sum()
        if total < min_total:
            assignments[t] = "none"
            continue
        # determine dominance
        top = np.argsort(row)[::-1]
        if len(top) < 2:
            assignments[t] = genres[top[0]] if total >= min_total else "none"
            continue
        max1, max2 = row[top[0]], row[top[1]]
        ratio = (max1 / max2) if max2 > 0 else np.inf
        assignments[t] = genres[top[0]] if ratio >= min_ratio else "none"

    return assignments, all_tokens, genres


def pca(model):
    """
    Plots PCA of token embeddings colored by their dominant genre.
    Expects token 0 already excluded.
    """
    dataset_root = Path("dataset/processed/q_16/seg_512")
    cfg = load_config()
    genres = cfg["dataset_creation"]["genres"]

    total_counter, genre_counters = count_tokens_by_genre(dataset_root, genres)

    assignments, all_tokens, genres = assign_token_genres(genre_counters)
    E = model.token_emb.weight.detach().cpu().numpy()[1:]
    E_pca = PCA(n_components=2).fit_transform(E)

    unique_labels = list(dict.fromkeys(assignments.values()))
    label_to_id = {g: i for i, g in enumerate(unique_labels)}
    colors = np.array(
        [label_to_id[assignments.get(t, "none")] for t in all_tokens if t != 0],
        dtype=int,
    )

    n = min(len(E_pca), len(colors))
    E_pca, colors = E_pca[:n], colors[:n]

    cmap = plt.cm.get_cmap("tab20", len(unique_labels))
    alphas = np.where(
        np.array([assignments.get(t, "none") for t in all_tokens if t != 0]) == "none",
        0.1,
        0.8,
    )

    plt.figure(figsize=(9, 7))
    for g, i in label_to_id.items():
        mask = np.array([assignments.get(t, "none") == g for t in all_tokens if t != 0])
        plt.scatter(
            E_pca[mask, 0],
            E_pca[mask, 1],
            s=10,
            color=cmap(i / len(unique_labels)),
            alpha=0.1 if g == "none" else 0.8,
            label=g,
        )

    plt.title(
        "Token embeddings colored by dominant genre (unk removed, none transparent)"
    )
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    dataset_root = Path("dataset/processed/q_16/seg_512")
    cfg = load_config()
    genres = cfg["dataset_creation"]["genres"]
    genre_labels = {idx: g for idx, g in enumerate(genres)}

    total_counter, genre_counters = count_tokens_by_genre(dataset_root, genres)

    summarize_by_genre(total_counter, genre_counters, genre_labels=genre_labels)
