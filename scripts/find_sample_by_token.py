from pathlib import Path

import numpy as np

from ml.data.tokenizer import BeatTokenizer
from ml.utils.cfg import load_config
from scripts.plotting_inference import plot_drum_matrix


def find_npz_with_token(dataset_dir, target_token, tokenizer):
    """
    Searches all .npz files in dataset_dir for a specific token ID.
    If found, loads the first matching file and converts its tokens to matrix form.
    """
    dataset_dir = Path(dataset_dir)
    files = list(dataset_dir.rglob("*.npz"))

    print(
        f"Searching {len(files)} files under {dataset_dir} for token {target_token}..."
    )
    f_count = 0
    count_finds = 0
    max = 0
    for f in files:
        f_count += 1
        try:
            data = np.load(f, allow_pickle=True)
            tokens = data["tokens"]

            if target_token in tokens:
                if np.sum(tokens == target_token) < 10:
                    continue
                # else:
                #     count_finds += 1
                #     if np.sum(tokens == target_token) > max:
                #         max = np.sum(tokens == target_token)
                #     continue

                print(f"Found token {target_token} in file: {f.name}")

                # Example: convert back to matrix form using your tokenizer
                # Adjust depending on your tokenizer interface

                matrices = [tokenizer.detokenize(t) for t in tokens]

                # Option 1: keep them separate along a new axis → shape (N, 13, 16)
                matrix = np.concatenate(matrices, axis=0)

                print(f"Matrix shape: {matrix.shape}")
                return matrix, f
        except Exception as e:
            print(f"[Warning] Skipped {f.name}: {e}")

    print(f"Token {target_token} not found in dataset.")
    print(
        f"Scanned {f_count} files, found {count_finds} occurrences, max count in a file: {max}"
    )
    return None, None


# Example usage
if __name__ == "__main__":
    cfg = load_config()
    tokenizer = BeatTokenizer(
        cfg, path="dataset/processed/q_16/seg_512/beat_tokenizer.npy"
    )
    print(len(tokenizer))

    dataset_dir = "dataset/processed/q_16/seg_512"
    target_token = 1  # for example

    matrix, file_path = find_npz_with_token(dataset_dir, target_token, tokenizer)
    if matrix is not None:
        print(f"✅ Token found in {file_path}")
        plot_drum_matrix(
            matrix, f"Drum Matrix for Token {target_token}", "outputs/token_matrix.png"
        )

    else:
        print("❌ Token not found.")
