from pathlib import Path

import numpy as np

from ml.data.midi import Midi
from ml.data.tokenizer import SimpleTokenizer
from ml.utils.cfg import load_config

# --- setup ---
cfg = load_config()
midi_reader = Midi(cfg["dataset"]["quantization"])
tokenizer = SimpleTokenizer()

# Paths to 3 example MIDI files (replace with real ones)
midi_paths = [
    "outputs/test/4_rock_100_beat_4-4_31.midi",
    "outputs/test/27_jazz_92_beat_4-4_16.midi",
    "outputs/test/48_rock_110_beat_4-4_6.midi",
]
import matplotlib.pyplot as plt


def plot_drum_matrix(matrix, title="Generated Drum Pattern", save_path=None):
    """
    matrix: (T, 9) numpy array with values 0–1
    """
    instruments = [
        "Kick",
        "Snare",
        "HH Closed",
        "HH Open",
        "Tom L",
        "Tom M",
        "Tom H",
        "Crash",
        "Ride",
    ]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(matrix.T, aspect="auto", origin="lower", cmap="magma", vmin=0, vmax=1)
    ax.set_yticks(range(len(instruments)))
    ax.set_yticklabels(instruments)
    ax.set_xlabel("Timesteps")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)


def simplify_matrix(mat: np.ndarray, pitch_groups: dict[str, list[int]]):
    out = np.zeros((mat.shape[0], len(pitch_groups)), dtype=np.float32)
    for idx, key in enumerate(pitch_groups):
        for p in pitch_groups[key]:
            if p < mat.shape[1]:
                out[:, idx] = np.maximum(out[:, idx], mat[:, p])
    out /= 127.0
    return out


# --- read and tokenize ---
for i, midi_path in enumerate(midi_paths):
    print(f"\n--- Processing {midi_path} ---")

    tracks = midi_reader.read_file(midi_path)  # returns dict of name → matrix
    for name, mat in tracks.items():
        mat = simplify_matrix(mat, cfg["dataset"]["pitch_groups"])
        plot_drum_matrix(
            mat, title=f"Drum Pattern from '{Path(midi_path).name}' - Track '{name}'"
        )

        print(f"Track '{name}': {mat.shape[0]} timesteps")

        # tokenize each timestep (vector of 9 drums)
        tokens = [tokenizer.tokenize(vec) for vec in mat]
        print(f"  → Tokenized into {len(tokenizer.vocab)} tokens")

        # detokenize a few for sanity check
        recon = np.stack([tokenizer.detokenize(t) for t in tokens])

        # --- recreate MIDI and save ---
        midi_reader.create_midi(
            recon,
            output_path=f"outputs/test/test_detok_{i+1}_{name}.mid",
            tempo=100,
        )

print("\n✅ Test complete — check 'outputs/' for resulting MIDI files.")
