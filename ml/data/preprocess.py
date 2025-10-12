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
        self.train_test_val_split = self.dataset_cfg["train_test_val_split"]
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
            genre, _ = self._extract_metadata(f)
            if genre in self.genres:
                genre_files[genre].append(f)

        splits = {"train": [], "val": [], "test": []}
        for genre, files in genre_files.items():
            rng.shuffle(files)
            n = len(files)
            n_train = int(n * self.train_test_val_split[0])
            n_val = int(n * self.train_test_val_split[1])
            splits["train"].extend(files[:n_train])
            splits["val"].extend(files[n_train : n_train + n_val])
            splits["test"].extend(files[n_train + n_val :])

        print("\nSplit summary:")
        print(f"{'Genre':<12} {'Train':<8} {'Val':<8} {'Test':<8} {'Total':<8}")
        print("-" * 50)
        for genre, files in genre_files.items():
            n = len(files)
            n_train = int(n * self.train_test_val_split[0])
            n_val = int(n * self.train_test_val_split[1])
            n_test = n - n_train - n_val
            print(f"{genre:<12} {n_train:<8} {n_val:<8} {n_test:<8} {n:<8}")
        total = len(midi_files)
        print(
            f"{'Total':<12} {len(splits['train']):<8} {len(splits['val']):<8} {len(splits['test']):<8} {total:<8}"
        )
        print()
        return splits

    def _process_midi_files(
        self, midi_files, output_dir, samples_per_genre, split_target
    ):
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
                mat = self._trim_trailing_zeros_full_segments(mat)
                n_steps = mat.shape[0]
                if n_steps < self.segment_len:
                    continue

                starts = np.arange(0, n_steps - self.segment_len + 1, self.segment_len)
                np.random.shuffle(starts)

                genre_dir = output_dir / genre
                genre_dir.mkdir(exist_ok=True)

                for s in starts:
                    if counts[genre] >= samples_per_genre or saved >= split_target:
                        break
                    seg = mat[s : s + self.segment_len]
                    tokens = [self.tokenizer.tokenize(vec) for vec in seg]
                    positions = np.arange(self.segment_len, dtype=np.int32)

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
                    saved += 1
                    pbar.update(1)

            if all(counts[g] >= samples_per_genre for g in self.genres):
                break

        pbar.close()
        print(f"\nFinal counts per genre in {output_dir.name}:")
        for g in sorted(self.genres):
            print(f"  {g}: {counts[g]}")

    def preprocess_dataset(self):
        midi_dir = Path(self.midi_dir)
        midi_files = list(midi_dir.rglob("*.mid")) + list(midi_dir.rglob("*.midi"))
        print(f"Found {len(midi_files)} MIDI files")

        splits = self._split_files_by_genre(midi_files)
        ratios = dict(zip(["train", "val", "test"], self.train_test_val_split))

        base_dir = (
            Path(self.save_dir) / f"q_{self.quantization}" / f"seg_{self.segment_len}"
        )

        datasets = {}
        for split, files in splits.items():
            print(f"\nProcessing {split} set...")
            out_dir = base_dir / split
            out_dir.mkdir(parents=True, exist_ok=True)

            split_target = int(self.total_target * ratios[split])
            samples_per_genre = split_target // len(self.genres)

            self._process_midi_files(files, out_dir, samples_per_genre, split_target)
            datasets[split] = DrumDataset(out_dir, include_genre=True)

        print("\nSaving tokenizer vocabulary...")
        self.tokenizer.save()
        print(f"[Tokenizer] Vocabulary size: {len(self.tokenizer)} tokens")

        return datasets["train"], datasets["val"], datasets["test"], self.tokenizer
