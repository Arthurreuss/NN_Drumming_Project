import os
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def count_tokens_in_dataset(dataset_dir):
    """
    Count token frequencies across all .npz files in a dataset folder.
    Expects files with fields: 'tokens' (and optionally 'targets').
    """
    dataset_dir = Path(dataset_dir)
    files = list(dataset_dir.rglob("*.npz"))
    counter = Counter()

    print(f"Scanning {len(files)} files under {dataset_dir}...")

    for f in tqdm(files, desc=f"Counting tokens in {dataset_dir.name}"):
        try:
            data = np.load(f, allow_pickle=True)
            tokens = data["tokens"].flatten()
            counter.update(tokens.tolist())

            if "targets" in data:
                counter.update(data["targets"].flatten().tolist())
        except Exception as e:
            print(f"[Warning] Skipped {f.name}: {e}")

    return counter


def summarize(counter, title="Token Distribution", top_n=20, save_path=None):
    total = sum(counter.values())
    unique = len(counter)
    print(f"\nTotal tokens: {total:,}")
    print(f"Unique tokens: {unique:,}")
    print(f"Most frequent {top_n} tokens:")
    for token, freq in counter.most_common(top_n):
        pct = freq / total * 100
        print(f"  Token {token:<6} {freq:>8} ({pct:5.2f}%)")

    # --- Plot top N ---
    tokens, freqs = zip(*counter.most_common(top_n))
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(tokens)), freqs, color="tab:red", alpha=0.8)
    plt.xticks(range(len(tokens)), tokens, rotation=45)
    plt.xlabel("Token ID")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


if __name__ == "__main__":
    dataset_root = Path("dataset/processed/q_16/seg_256")
    train_dir = dataset_root / "train"
    val_dir = dataset_root / "val"
    test_dir = dataset_root / "test"

    all_counts = Counter()

    for split_dir in [train_dir, val_dir, test_dir]:
        if split_dir.exists():
            split_counts = count_tokens_in_dataset(split_dir)
            print(f"\n--- {split_dir.name.upper()} SPLIT ---")
            summarize(
                split_counts,
                title=f"{split_dir.name.capitalize()} Token Distribution",
                save_path=f"outputs/{split_dir.name}_token_freq.png",
            )
            all_counts.update(split_counts)
        else:
            print(f"[Skip] {split_dir} does not exist")

    print("\n--- COMBINED DATASET ---")
    summarize(
        all_counts,
        title="Combined Token Distribution",
        top_n=20,
        save_path="outputs/combined_token_freq.png",
    )
