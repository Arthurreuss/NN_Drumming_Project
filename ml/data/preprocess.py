import re
from pathlib import Path

import numpy as np
from ml.data.midi import Midi


def _extract_metadata(midi_path):
    """
    Extract genre and bpm from E-GMD filename patterns such as:
    1_funk-groove1_138_beat_4-4_1.midi
    """
    name = Path(midi_path).stem
    m = re.match(r"\d+_([a-zA-Z]+)-.*_(\d+)_beat.*", name)
    if m:
        genre = m.group(1).lower()
        bpm = int(m.group(2))
    else:
        genre, bpm = "unknown", -1
    return genre, bpm


def _simplify_matrix(mat):
    """
    Convert full 128-pitch matrix to 9 canonical drum instruments.
    Groups ranges of pitches per instrument.
    Output: (timesteps, 9) float matrix in [0, 1].
    """

    # General MIDI Drum Map ranges
    pitch_groups = {
        "kick": [35, 36],
        "snare": [38, 40],
        "hh_closed": [42, 44],
        "hh_open": [46],
        "tom_low": [41, 43],
        "tom_mid": [45, 47],
        "tom_high": [48, 50],
        "crash": [49, 57],
        "ride": [51, 59],
    }

    out = np.zeros((mat.shape[0], 9), dtype=np.float32)
    for idx, (_, pitches) in enumerate(pitch_groups.items()):
        for p in pitches:
            if p < mat.shape[1]:
                out[:, idx] = np.maximum(out[:, idx], mat[:, p])

    # normalize 0–1
    out /= 127.0
    return out


def preprocess_dataset(midi_dir, save_dir, quantization=32, segment_len=128):
    """
    Convert all MIDI files under midi_dir to .npz segments stored in save_dir.
    Each .npz contains X (timesteps×9), genre (string), bpm (int).
    """
    reader = Midi(quantization=quantization)
    midi_dir = Path(midi_dir)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for i, midi_path in enumerate(midi_dir.rglob("*.midi")):
        if i >= 100:  # stop after 100 files
            break

        genre, bpm = _extract_metadata(midi_path)
        genre_dir = save_dir / genre
        genre_dir.mkdir(exist_ok=True)

        tracks = reader.read_file(str(midi_path))
        for name, data in tracks.items():
            mat = data["drum_matrix"]
            mat = _simplify_matrix(mat)

            # Split into fixed-length segments
            n_steps = mat.shape[0]
            for i in range(0, n_steps - segment_len + 1, segment_len):
                seg = mat[i : i + segment_len]
                out_path = genre_dir / f"{midi_path.stem}_{name}_bpm{bpm}_seg{i}.npz"
                np.savez(out_path, X=seg, genre=genre, bpm=bpm)


if __name__ == "__main__":
    input_dir = "dataset/e-gmd-v1.0.0"
    output_dir = "dataset/preprocessed"
    preprocess_dataset(input_dir, output_dir, 16)
