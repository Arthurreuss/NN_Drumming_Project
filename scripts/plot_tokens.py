import numpy as np

from ml.data.tokenizer import BeatTokenizer
from scripts.plotting_inference import plot_drum_matrix

tokenizer = BeatTokenizer("dataset/processed_test/q_16/seg_128/beat_tokenizer.npy")

D = len(tokenizer.cfg["dataset_creation"]["pitch_groups"])

for motif, tid in tokenizer.vocab.items():
    if motif == "<UNK>":
        continue
    arr = np.array(motif, dtype=float)
    if arr.size % D != 0:
        print(f"Skipping malformed motif (token {tid}) of length {arr.size}")
        continue
    matrix = arr.reshape(-1, D)
    plot_drum_matrix(
        matrix,
        title=f"Token {tid} | shape={matrix.shape}",
        save_path=f"outputs/token_{tid}.png",
    )
