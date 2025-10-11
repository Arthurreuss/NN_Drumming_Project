import random
import re
import tokenize
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ml.data.dataset import DrumDataset
from ml.data.midi import Midi
from ml.data.tokenizer import SimpleTokenizer
from utils.cfg import load_config


def _extract_metadata(midi_path: Path):
    name = midi_path.stem
    m = re.match(r"\d+_([a-zA-Z]+)-.*_(\d+)_beat.*", name)
    if m:
        return m.group(1).lower(), int(m.group(2))
    return "unknown", -1


def _simplify_matrix(mat: np.ndarray, pitch_groups: dict[str, list[int]]):
    out = np.zeros((mat.shape[0], 9), dtype=np.float32)
    for idx, key in enumerate(pitch_groups):
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
    last_idx = np.where(nz)[0][-1] + 1
    trimmed_len = (last_idx // segment_len) * segment_len
    return mat[:trimmed_len]


def _split_files_by_genre(
    midi_files: list[Path], train_ratio: float, genres: set, seed: int
) -> tuple[list[Path], list[Path]]:
    """
    Split MIDI files into train/test sets while maintaining genre balance.

    Args:
        midi_files: List of all MIDI file paths
        train_ratio: Fraction of files to use for training (e.g., 0.8)
        genres: Set of valid genre names
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_files, test_files)
    """
    rng = random.Random(seed)

    # Group files by genre
    genre_files = defaultdict(list)
    for f in midi_files:
        genre, _ = _extract_metadata(f)
        if genre in genres:
            genre_files[genre].append(f)

    train_files = []
    test_files = []

    # Split each genre independently
    for genre, files in genre_files.items():
        rng.shuffle(files)
        split_idx = int(len(files) * train_ratio)
        train_files.extend(files[:split_idx])
        test_files.extend(files[split_idx:])

    # Shuffle the combined lists to mix genres
    rng.shuffle(train_files)
    rng.shuffle(test_files)

    print(f"\nFile split summary:")
    print(f"{'Genre':<15} {'Train':<10} {'Test':<10} {'Total':<10}")
    print("-" * 45)
    for genre in sorted(genres):
        files = genre_files[genre]
        split_idx = int(len(files) * train_ratio)
        train_count = split_idx
        test_count = len(files) - split_idx
        print(f"{genre:<15} {train_count:<10} {test_count:<10} {len(files):<10}")
    print(
        f"{'Total':<15} {len(train_files):<10} {len(test_files):<10} {len(train_files) + len(test_files):<10}\n"
    )

    return train_files, test_files


def _process_midi_files(
    midi_files,
    output_dir,
    reader,
    genres,
    samples_per_genre,
    split_target,
    pitch_groups,
    segment_len,
    max_samples_per_file,
    tokenizer,
):
    """Process MIDI files and save segments to the specified output directory."""
    counts = defaultdict(int)
    saved = 0
    pbar = tqdm(total=split_target, desc=f"Saving to {output_dir.name}", unit="seg")

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

            genre_dir = output_dir / genre
            genre_dir.mkdir(exist_ok=True)

            picks = 0
            for s in starts:
                if counts[genre] >= samples_per_genre or picks >= max_samples_per_file:
                    break

                seg = mat[s : s + segment_len]  # (T, 9)

                tokens = [tokenizer.tokenize(vec) for vec in seg]  # list[int] length=T

                positions = np.arange(segment_len, dtype=np.int32)  # 0..T-1

                out_path = (
                    genre_dir / f"{midi_path.stem}_{name}_bpm{bpm}_seg{s}_id{saved}.npz"
                )

                np.savez(
                    out_path,
                    tokens=np.array(tokens, dtype=np.int32),
                    positions=positions,
                    genre=genre,
                    bpm=bpm,
                )
                counts[genre] += 1
                picks += 1
                saved += 1
                pbar.update(1)

        # stop early if all genre quotas hit
        if all(counts[g] >= samples_per_genre for g in genres):
            break

    pbar.close()
    print(f"\nFinal counts per genre in {output_dir.name}:")
    for g in sorted(genres):
        print(f"  {g}: {counts[g]}")


def preprocess_dataset() -> tuple[DrumDataset, DrumDataset]:
    """
    Build a balanced dataset of fixed-length drum segments from MIDI files.

    Steps:
      1) Load config: pitch_groups, genres, total_target, paths, quantization, segment_len, max_samples_per_file.
      2) Init MIDI reader and output dirs: {save_dir}/q_{quantization}/seg_{segment_len}/<split>/.
      3) Discover all .mid/.midi files and split by genre into train/test sets.
      4) For each split (train/test):
         - Process files until per-genre quota is met.
         - Extract (genre, bpm); skip if not allowed or quota reached.
         - Read tracks; for each track: simplify → trim → make non-overlapping starts → shuffle.
         - Save up to min(remaining genre quota, max_samples_per_file) segments as .npz with fields:
           X (float32, shape [segment_len, 9]), genre (str), bpm (int).
      5) Stop when all genres hit their quota; print per-genre counts.

    I/O:
      - Reads config and MIDI under cfg['raw_data_dir'].
      - Writes .npz segments under cfg['preprocessed_data_dir'] grouped by split and genre.

    Guarantees:
      - Even per-genre cap (total_target / |genres|) in each split.
      - No data leakage - files are split before processing.
      - Balanced genre distribution in both train and test sets.
      - No partial tail segments.
      - Values normalized to [0, 1].
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
    train_test_split = cfg["train_test_split"]
    seed = cfg["seed"]
    tokenizer = SimpleTokenizer()

    genres = set(genres_cfg)
    midi_dir = Path(midi_dir)

    # Discover all MIDI files
    midi_files = list(midi_dir.rglob("*.mid")) + list(midi_dir.rglob("*.midi"))
    print(f"Found {len(midi_files)} MIDI files")

    # Split files by genre while maintaining balance
    train_files, test_files = _split_files_by_genre(
        midi_files, train_test_split, genres, seed
    )

    reader = Midi(quantization=quantization)

    # Calculate split sizes
    train_samples = int(total_target * train_test_split)
    test_samples = total_target - train_samples
    train_samples_per_genre = train_samples // len(genres)
    test_samples_per_genre = test_samples // len(genres)

    # Create output directories
    train_dir = Path(save_dir) / f"q_{quantization}" / f"seg_{segment_len}" / "train"
    test_dir = Path(save_dir) / f"q_{quantization}" / f"seg_{segment_len}" / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # Process train and test files separately
    print("Processing training files...")
    _process_midi_files(
        train_files,
        train_dir,
        reader,
        genres,
        train_samples_per_genre,
        train_samples,
        pitch_groups,
        segment_len,
        max_samples_per_file,
        tokenizer,
    )

    print("\nProcessing test files...")
    _process_midi_files(
        test_files,
        test_dir,
        reader,
        genres,
        test_samples_per_genre,
        test_samples,
        pitch_groups,
        segment_len,
        max_samples_per_file,
        tokenizer,
    )

    return DrumDataset(train_dir, include_genre=True), DrumDataset(
        test_dir, include_genre=True
    )
