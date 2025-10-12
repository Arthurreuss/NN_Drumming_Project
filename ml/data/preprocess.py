import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ml.data.dataset import DrumDataset
from ml.data.tokenizer import SimpleTokenizer
from ml.utils.cfg import load_config


class DrumPreprocessor:
    def __init__(self, midi_reader, tokenizer):
        self.cfg = load_config()
        self.dataset_cfg = self.cfg["dataset"]
        self.midi_reader = midi_reader
        self.tokenizer = tokenizer
        self.pitch_groups = self.dataset_cfg["pitch_groups"]
        self.genres = set(self.dataset_cfg["genres"])
        self.total_target = self.dataset_cfg["total_target"]
        self.midi_dir = self.dataset_cfg["raw_data_dir"]
        self.save_dir = self.dataset_cfg["preprocessed_data_dir"]
        self.quantization = self.dataset_cfg["quantization"]
        self.segment_len = self.dataset_cfg["segment_len"]
        self.max_samples_per_file = self.dataset_cfg["max_samples_per_file"]
        self.train_test_split = self.dataset_cfg["train_test_split"]
        self.seed = self.dataset_cfg["seed"]
        np.random.seed(self.seed)
        random.seed(self.seed)

    def _extract_metadata(self, midi_path: Path):
        name = midi_path.stem
        m = re.match(r"\d+_([a-zA-Z]+)-.*_(\d+)_beat.*", name)
        if m:
            return m.group(1).lower(), int(m.group(2))
        return "unknown", -1

    def _simplify_matrix(self, mat: np.ndarray, pitch_groups: dict[str, list[int]]):
        out = np.zeros((mat.shape[0], len(self.pitch_groups)), dtype=np.float32)
        for idx, key in enumerate(pitch_groups):
            for p in pitch_groups[key]:
                if p < mat.shape[1]:
                    out[:, idx] = np.maximum(out[:, idx], mat[:, p])
        out /= 127.0
        return out

    def _trim_trailing_zeros_full_segments(self, mat: np.ndarray, segment_len: int):
        """Trim to last nonzero, then floor to full segment multiple."""
        nz = np.any(mat > 0, axis=1)
        if not nz.any():
            return mat[:0]
        last_idx = np.where(nz)[0][-1] + 1
        trimmed_len = (last_idx // segment_len) * segment_len
        return mat[:trimmed_len]

    def _split_files_by_genre(
        self, midi_files: list[Path]
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
        genres = set(self.genres_cfg)
        rng = random.Random()

        # Group files by genre
        genre_files = defaultdict(list)
        for f in midi_files:
            genre, _ = self._extract_metadata(f)
            if genre in genres:
                genre_files[genre].append(f)

        train_files = []
        test_files = []

        # Split each genre independently
        for genre, files in genre_files.items():
            rng.shuffle(files)
            split_idx = int(len(files) * self.train_test_split)
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
            split_idx = int(len(files) * self.train_test_split)
            train_count = split_idx
            test_count = len(files) - split_idx
            print(f"{genre:<15} {train_count:<10} {test_count:<10} {len(files):<10}")
        print(
            f"{'Total':<15} {len(train_files):<10} {len(test_files):<10} {len(train_files) + len(test_files):<10}\n"
        )

        return train_files, test_files

    def _process_midi_files(
        self,
        midi_files,
        output_dir,
        samples_per_genre,
        split_target,
    ):
        """Process MIDI files and save segments to the specified output directory."""
        counts = defaultdict(int)
        saved = 0
        pbar = tqdm(total=split_target, desc=f"Saving to {output_dir.name}", unit="seg")

        for midi_path in midi_files:
            genre, bpm = self._extract_metadata(midi_path)
            if genre not in self.genres or counts[genre] >= samples_per_genre:
                continue

            tracks = self.midi_reader.read_file(str(midi_path))
            if not tracks:
                continue

            for name, mat in tracks.items():
                mat = self._simplify_matrix(mat, self.pitch_groups)
                mat = self._trim_trailing_zeros_full_segments(mat, self.segment_len)
                n_steps = mat.shape[0]
                if n_steps < self.segment_len:
                    continue

                # non-overlapping random order
                starts = np.arange(0, n_steps - self.segment_len + 1, self.segment_len)

                np.random.shuffle(starts)

                genre_dir = output_dir / genre
                genre_dir.mkdir(exist_ok=True)

                picks = 0
                for s in starts:
                    if (
                        counts[genre] >= samples_per_genre
                        or picks >= self.max_samples_per_file
                    ):
                        break

                    seg = mat[s : s + self.segment_len]  # (T, 9)

                    tokens = [
                        self.tokenizer.tokenize(vec) for vec in seg
                    ]  # list[int] length=T

                    positions = np.arange(self.segment_len, dtype=np.int32)  # 0..T-1

                    out_path = (
                        genre_dir
                        / f"{midi_path.stem}_{name}_bpm{bpm}_seg{s}_id{saved}.npz"
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
            if all(counts[g] >= samples_per_genre for g in self.genres):
                break

        pbar.close()
        print(f"\nFinal counts per genre in {output_dir.name}:")
        for g in sorted(self.genres):
            print(f"  {g}: {counts[g]}")

    def preprocess_dataset(self):
        midi_dir = Path(self.midi_dir)

        # Discover all MIDI files
        midi_files = list(midi_dir.rglob("*.mid")) + list(midi_dir.rglob("*.midi"))
        print(f"Found {len(midi_files)} MIDI files")

        # Split files by genre while maintaining balance
        train_files, test_files = self._split_files_by_genre(midi_files)

        # Calculate split sizes
        train_samples = int(self.total_target * self.train_test_split)
        test_samples = self.total_target - train_samples
        train_samples_per_genre = train_samples // len(self.genres)
        test_samples_per_genre = test_samples // len(self.genres)

        # Create output directories
        train_dir = (
            Path(self.save_dir)
            / f"q_{self.quantization}"
            / f"seg_{self.segment_len}"
            / "train"
        )
        test_dir = (
            Path(self.save_dir)
            / f"q_{self.quantization}"
            / f"seg_{self.segment_len}"
            / "test"
        )
        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)

        # Process train and test files separately
        print("Processing training files...")
        self._process_midi_files(
            train_files,
            train_dir,
            train_samples_per_genre,
            train_samples,
        )

        print("\nProcessing test files...")
        self._process_midi_files(
            test_files,
            test_dir,
            test_samples_per_genre,
            test_samples,
        )
        print("Saving tokenizer vocabulary...")
        self.tokenizer.save()
        print(f"Tokenizer size: {len(self.tokenizer)} tokens")

        return (
            DrumDataset(train_dir, include_genre=True),
            DrumDataset(test_dir, include_genre=True),
            self.tokenizer,
        )
