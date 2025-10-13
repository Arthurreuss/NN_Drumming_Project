import random
import re
from collections import defaultdict
from os import path
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ml.data.dataset import DrumDataset
from ml.utils.cfg import load_config


class DrumPreprocessor:
    def __init__(self, midi_reader, tokenizer):
        self.cfg = load_config()
        self.dataset_cfg = self.cfg["dataset_creation"]
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
        self.train_test_val_split = self.dataset_cfg["train_test_val_split"]
        self.seed = self.dataset_cfg["seed"]
        np.random.seed(self.seed)
        random.seed(self.seed)

    def _extract_metadata(self, midi_path: Path):
        name = midi_path.stem
        genres = self.dataset_cfg["genres"]
        for g in genres:
            if g.lower() in name.lower() and "beat" in name.lower():
                return g
        return "unknown", -1

    def _simplify_matrix(self, mat: np.ndarray, pitch_groups: dict[str, list[int]]):
        out = np.zeros((mat.shape[0], len(self.pitch_groups)), dtype=np.float32)
        for idx, key in enumerate(pitch_groups):
            for p in pitch_groups[key]:
                if p < mat.shape[1]:
                    out[:, idx] = np.maximum(out[:, idx], mat[:, p])
        out /= 127.0
        return out

    def _trim_trailing_zeros_full_segments(self, mat: np.ndarray):
        """Trim to last nonzero, then floor to full segment multiple."""
        nz = np.any(mat > 0, axis=1)
        if not nz.any():
            return mat[:0]
        last_idx = np.where(nz)[0][-1] + 1
        trimmed_len = (last_idx // self.segment_len) * self.segment_len
        return mat[:trimmed_len]

    def _split_files_by_genre(self, midi_files: list[Path]):
        """Split files into train/val/test while maintaining genre balance."""
        rng = random.Random(self.seed)
        genre_files = defaultdict(list)
        for f in midi_files:
            genre = self._extract_metadata(f)
            if genre in self.genres:
                genre_files[genre].append(f)

        splits = {"train": [], "val": [], "test": []}
        per_genre_counts = {}

        for genre, files in genre_files.items():
            rng.shuffle(files)
            n = len(files)
            n_train = int(n * self.train_test_val_split[0])
            n_val = int(n * self.train_test_val_split[1])
            n_test = n - n_train - n_val

            train_files = files[:n_train]
            val_files = files[n_train : n_train + n_val]
            test_files = files[n_train + n_val :]

            splits["train"].extend(train_files)
            splits["val"].extend(val_files)
            splits["test"].extend(test_files)

            per_genre_counts[genre] = {
                "train": len(train_files),
                "val": len(val_files),
                "test": len(test_files),
                "total": n,
            }

        # --- Clean table print ---
        print("\nSplit summary:")
        print(f"{'Genre':<12} {'Train':>6} {'Val':>6} {'Test':>6} {'Total':>6}")
        print("-" * 44)
        for genre, c in sorted(per_genre_counts.items()):
            print(
                f"{genre:<12} {c['train']:>6} {c['val']:>6} {c['test']:>6} {c['total']:>6}"
            )
        print("-" * 44)
        print(
            f"{'Total':<12} "
            f"{len(splits['train']):>6} "
            f"{len(splits['val']):>6} "
            f"{len(splits['test']):>6} "
            f"{len(midi_files):>6}\n"
        )

        return splits

    def _cyclic_positional_encoding(self):
        t = np.arange(self.segment_len)
        period = self.quantization * 4
        return np.stack(
            [np.sin(2 * np.pi * t / period), np.cos(2 * np.pi * t / period)], axis=1
        )  # shape: (length, 2)

    def _process_midi_files(
        self, midi_files, output_dir, samples_per_genre, split_target
    ):
        counts = defaultdict(int)
        saved = 0
        pbar = tqdm(total=split_target, desc=f"Saving to {output_dir.name}", unit="seg")

        # Shuffle all files globally for randomness
        random.shuffle(midi_files)

        for midi_path in midi_files:
            genre = self._extract_metadata(midi_path)
            if genre not in self.genres:
                continue
            if counts[genre] >= samples_per_genre:
                continue  # already enough for this genre

            tracks = self.midi_reader.read_file(str(midi_path))
            if not tracks:
                continue

            for name, mat in tracks.items():
                mat = self._simplify_matrix(mat, self.pitch_groups)
                mat = self._trim_trailing_zeros_full_segments(mat)
                n_steps = mat.shape[0]
                if n_steps < self.segment_len:
                    continue

                # non-overlapping starts
                starts = np.arange(0, n_steps - self.segment_len + 1, self.segment_len)
                np.random.shuffle(starts)

                genre_dir = output_dir / genre
                genre_dir.mkdir(exist_ok=True)

                for s in starts:
                    if counts[genre] >= samples_per_genre or saved >= split_target:
                        break

                    seg = mat[s : s + self.segment_len]
                    tokens = [self.tokenizer.tokenize(vec) for vec in seg]
                    positions = self._cyclic_positional_encoding()

                    out_path = (
                        genre_dir / f"{midi_path.stem}_{name}_seg{s}_id{saved}.npz"
                    )
                    np.savez(
                        out_path,
                        tokens=np.array(tokens, dtype=np.int32),
                        positions=positions,
                        genre=genre,
                    )

                    counts[genre] += 1
                    saved += 1
                    if saved % 50 == 0:  # less tqdm overhead
                        pbar.update(50)

            # Stop early if all genre quotas hit
            if all(counts[g] >= samples_per_genre for g in self.genres):
                break

        pbar.close()
        print(f"\nFinal counts per genre in {output_dir.name}:")
        for g in sorted(self.genres):
            print(f"  {g}: {counts[g]} samples")

    def preprocess_dataset(self):
        midi_dir = Path(self.midi_dir)
        midi_files = list(midi_dir.rglob("*.mid")) + list(midi_dir.rglob("*.midi"))
        random.shuffle(midi_files)  # shuffle globally
        print(f"[Preprocessor] Found {len(midi_files)} MIDI files")

        splits = self._split_files_by_genre(midi_files)
        ratios = dict(zip(["train", "val", "test"], self.train_test_val_split))

        base_dir = (
            Path(self.save_dir) / f"q_{self.quantization}" / f"seg_{self.segment_len}"
        )

        datasets = {}
        for split, files in splits.items():
            print(f"\n[Preprocessor] Processing {split} set...")
            out_dir = base_dir / split
            out_dir.mkdir(parents=True, exist_ok=True)

            split_target = int(self.total_target * ratios[split])
            samples_per_genre = split_target // len(self.genres)

            self._process_midi_files(files, out_dir, samples_per_genre, split_target)
            datasets[split] = DrumDataset(out_dir, include_genre=True)

        self.tokenizer.prune(min_freq=self.dataset_cfg["token_min_freq"])
        self.tokenizer.save()
        print(f"[Tokenizer] Vocabulary size: {len(self.tokenizer)} tokens")
        print("[Preprocessor] Preprocessing complete.\n")

        return datasets["train"], datasets["val"], datasets["test"], self.tokenizer
