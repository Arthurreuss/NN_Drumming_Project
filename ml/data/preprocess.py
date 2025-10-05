import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from ml.data.midi import Midi
from sympy import im
from tqdm import tqdm
from utils.cfg import load_config


def _extract_metadata(midi_path: Path):
    name = midi_path.stem
    m = re.match(r"\d+_([a-zA-Z]+)-.*_(\d+)_beat.*", name)
    if m:
        return m.group(1).lower(), int(m.group(2))
    return "unknown", -1


def _simplify_matrix(mat: np.ndarray, pitch_groups: dict[str, list[int]]):
    out = np.zeros((mat.shape[0], 9), dtype=np.float32)
    for idx, key in enumerate(pitch_groups):  # rely on YAML key order
        for p in pitch_groups[key]:
            if p < mat.shape[1]:
                out[:, idx] = np.maximum(out[:, idx], mat[:, p])
    out /= 127.0
    return out


def _trim_trailing_zeros_full_segments(mat: np.ndarray, segment_len: int):
    """Trim to last nonzero, then floor to full segment multiple."""
    nz = np.any(mat > 0, axis=1)
    if not nz.any():
        return mat[:0]
    last_idx = np.where(nz)[0][-1] + 1  # make it a length
    trimmed_len = (last_idx // segment_len) * segment_len
    return mat[:trimmed_len]


def preprocess_dataset():
    """
    Build a balanced dataset of fixed-length drum segments from MIDI files.

    Steps:
      1) Load config: pitch_groups, genres, total_target, paths, quantization, segment_len, max_samples_per_file.
      2) Init MIDI reader and output dirs: {save_dir}/q_{quantization}/seg_{segment_len}/<genre>/.
      3) Discover and shuffle .mid/.midi files with a fixed seed.
      4) For each file (until per-genre quota is met):
         - Extract (genre, bpm); skip if not allowed or quota reached.
         - Read tracks; for each track: simplify → trim → make non-overlapping starts → shuffle.
         - Save up to min(remaining genre quota, max_samples_per_file) segments as .npz with fields:
           X (float32, shape [segment_len, 9]), genre (str), bpm (int).
      5) Stop when all genres hit their quota; print per-genre counts.

    I/O:
      - Reads config and MIDI under cfg['raw_data_dir'].
      - Writes .npz segments under cfg['preprocessed_data_dir'] grouped by genre.

    Guarantees:
      - Even per-genre cap (total_target / |genres|).
      - No partial tail segments.
      - Values normalized to [0, 1].

    Note:
      - Create train/val/test splits later by song to avoid leakage.
    """
    cfg = load_config("config.yaml")["dataset"]
    pitch_groups = cfg["pitch_groups"]
    genres_cfg = cfg["genres"]
    total_target = cfg["total_target"]
    midi_dir = cfg["raw_data_dir"]
    save_dir = cfg["preprocessed_data_dir"]
    quantization = cfg["quantization"]
    segment_len = cfg["segment_len"]
    max_samples_per_file = cfg["max_samples_per_file"]
    seed = cfg["seed"]

    genres = set(genres_cfg)
    samples_per_genre = total_target // len(genres)

    reader = Midi(quantization=quantization)
    midi_dir = Path(midi_dir)
    out_root = Path(save_dir) / f"q_{quantization}" / f"seg_{segment_len}"
    out_root.mkdir(parents=True, exist_ok=True)

    midi_files = list(midi_dir.rglob("*.mid")) + list(midi_dir.rglob("*.midi"))
    rng = random.Random(seed)
    rng.shuffle(midi_files)

    counts = defaultdict(int)
    saved = 0
    pbar = tqdm(total=total_target, desc="Saved samples", unit="seg")

    for midi_path in midi_files:
        genre, bpm = _extract_metadata(midi_path)
        if genre not in genres or counts[genre] >= samples_per_genre:
            continue

        tracks = reader.read_file(str(midi_path))
        if not tracks:
            continue

        for name, mat in tracks.items():
            mat = _simplify_matrix(mat, pitch_groups)
            mat = _trim_trailing_zeros_full_segments(mat, segment_len)
            n_steps = mat.shape[0]
            if n_steps < segment_len:
                continue

            # non-overlapping random order
            starts = np.arange(0, n_steps - segment_len + 1, segment_len)
            np.random.shuffle(starts)

            genre_dir = out_root / genre
            genre_dir.mkdir(exist_ok=True)

            picks = 0
            for s in starts:
                if counts[genre] >= samples_per_genre or picks >= max_samples_per_file:
                    break
                seg = mat[s : s + segment_len]
                out_path = (
                    genre_dir / f"{midi_path.stem}_{name}_bpm{bpm}_seg{s}_id{saved}.npz"
                )
                np.savez(out_path, X=seg, genre=genre, bpm=bpm)
                counts[genre] += 1
                picks += 1
                saved += 1
                pbar.update(1)

        # stop early if all genre quotas hit
        if all(counts[g] >= samples_per_genre for g in genres):
            print(f"Reached target of {total_target} samples. Stopping.")
            break

    print("Final counts per genre:")
    for g in sorted(genres):
        print(f"{g}: {counts[g]}")
