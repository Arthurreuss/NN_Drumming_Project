import numpy as np

from ml.data.tokenizer import BeatTokenizer
from ml.utils import cfg
from ml.utils.cfg import load_config
from scripts.plotting_inference import plot_drum_matrix

cfg = load_config()
tokenizer = BeatTokenizer(cfg, "dataset/processed/q_16/seg_128/beat_tokenizer.npy")
print(f"Loaded tokenizer with {len(tokenizer)} tokens.")
D = len(tokenizer.cfg["dataset_creation"]["pitch_groups"])
idx = 0
for motif, tid in tokenizer.vocab.items():
    if idx > 200:
        break
    idx += 1
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
        save_path=f"outputs/l1_token_{tid}.png",
    )
