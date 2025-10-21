import logging
import random
import re
from collections import defaultdict
from os import path
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ml.data.dataset import DrumDataset


class DrumPreprocessor:
    def __init__(self, cfg, midi_reader, tokenizer):
        self.cfg = cfg
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
                pattern = r"_(\d+)_beat"
                match = re.search(pattern, name.lower())
                if match:
                    bpm = int(match.group(1))
                    return g, bpm
        return "unknown", -1

    def _simplify_matrix(self, mat: np.ndarray, pitch_groups: dict[str, list[int]]):
        out = np.zeros((mat.shape[0], len(self.pitch_groups)), dtype=np.float32)
        for idx, key in enumerate(pitch_groups):
            for p in pitch_groups[key]:
                if p < mat.shape[1]:
                    out[:, idx] = np.maximum(out[:, idx], mat[:, p])
        out /= 127.0
        return out

    def _trim_silence_full_segments(self, mat: np.ndarray):
        """Trim leading and trailing all-zero 16-timestep blocks, keeping full segments."""
        if mat.size == 0:
            return mat

        beat = 16
        n_beats = mat.shape[0] // beat

        start = 0
        end = n_beats

        # trim silent beats from front
        for i in range(n_beats):
            if np.any(mat[i * beat : (i + 1) * beat] > 0):
                start = i
                break

        # trim silent beats from back
        for i in range(n_beats - 1, -1, -1):
            if np.any(mat[i * beat : (i + 1) * beat] > 0):
                end = i + 1
                break

        trimmed = mat[start * beat : end * beat]
        trimmed_len = (trimmed.shape[0] // self.segment_len) * self.segment_len
        return trimmed[:trimmed_len]

    def _split_files_by_genre(self, midi_files: list[Path]):
        """Split files into train/val/test while maintaining genre balance."""
        rng = random.Random(self.seed)
        genre_files = defaultdict(list)
        for f in midi_files:
            genre, bpm = self._extract_metadata(f)
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

        # --- Clean table logging.info ---
        logging.info("\nSplit summary:")
        logging.info(f"{'Genre':<12} {'Train':>6} {'Val':>6} {'Test':>6} {'Total':>6}")
        logging.info("-" * 44)
        for genre, c in sorted(per_genre_counts.items()):
            logging.info(
                f"{genre:<12} {c['train']:>6} {c['val']:>6} {c['test']:>6} {c['total']:>6}"
            )
        logging.info("-" * 44)
        logging.info(
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
            genre, bpm = self._extract_metadata(midi_path)
            if genre not in self.genres:
                continue
            if 0 > bpm > 250:
                continue
            if counts[genre] >= samples_per_genre:
                continue  # already enough for this genre

            tracks = self.midi_reader.read_file(str(midi_path))
            if not tracks:
                continue

            for name, mat in tracks.items():
                mat = self._simplify_matrix(mat, self.pitch_groups)
                mat = self._trim_silence_full_segments(mat)
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
                    tokens = self.tokenizer.tokenize(seg)

                    positions = self._cyclic_positional_encoding()
                    beat_positions = positions[:: self.quantization][: len(tokens)]

                    out_path = (
                        genre_dir / f"{midi_path.stem}_{name}_seg{s}_id{saved}.npz"
                    )
                    np.savez(
                        out_path,
                        tokens=np.array(tokens, dtype=np.int32),
                        positions=beat_positions,
                        genre=genre,
                        bpm=np.array([bpm], dtype=np.float32),
                    )

                    counts[genre] += 1
                    saved += 1
                    if saved % 50 == 0:
                        pbar.update(50)

            # Stop early if all genre quotas hit
            if all(counts[g] >= samples_per_genre for g in self.genres):
                break

        pbar.close()
        logging.info(f"\nFinal counts per genre in {output_dir.name}:")
        for g in sorted(self.genres):
            logging.info(f"  {g}: {counts[g]} samples")

    def _filter_dataset_unknowns(self, dataset_dir, unk_id=0, unk_threshold=0.5):
        """
        Remove dataset samples (.npz files) where more than `unk_threshold`
        fraction of tokens are <UNK> (id == unk_id).
        """
        dataset_dir = Path(dataset_dir)
        npz_files = list(dataset_dir.rglob("*.npz"))
        removed = 0
        total = len(npz_files)

        for file in npz_files:
            data = np.load(file, allow_pickle=True)
            tokens = data["tokens"]
            unk_ratio = np.mean(tokens == unk_id)
            if unk_ratio >= unk_threshold:
                file.unlink()  # delete file
                removed += 1

        logging.info(
            f"[Filter] {removed}/{total} samples removed "
            f"({removed/total*100:.1f}%) — unk_ratio > {unk_threshold}"
        )

    def _remap_tokens_to_pruned_vocab(self, dataset_dir):
        """
        Replace all tokens not in the pruned vocab with UNK id.
        """
        dataset_dir = Path(dataset_dir)
        npz_files = list(dataset_dir.rglob("*.npz"))
        vocab_keys = set(self.tokenizer.vocab.values())
        unk_id = self.tokenizer.unk_id
        remapped = 0

        for f in npz_files:
            data = np.load(f, allow_pickle=True)
            tokens = data["tokens"]

            invalid_mask = ~np.isin(tokens, list(vocab_keys))
            if np.any(invalid_mask):
                tokens[invalid_mask] = unk_id
                remapped += 1
                np.savez(
                    f,
                    tokens=tokens,
                    positions=data["positions"],
                    genre=data["genre"],
                    bpm=data["bpm"],
                )

        logging.info(
            f"[Remap] {remapped}/{len(npz_files)} files contained pruned tokens."
        )

    def preprocess_dataset(self):
        midi_dir = Path(self.midi_dir)
        midi_files = list(midi_dir.rglob("*.mid")) + list(midi_dir.rglob("*.midi"))
        random.shuffle(midi_files)
        logging.info(f"[Preprocessor] Found {len(midi_files)} MIDI files")

        splits = self._split_files_by_genre(midi_files)
        ratios = dict(zip(["train", "val", "test"], self.train_test_val_split))

        base_dir = (
            Path(self.save_dir) / f"q_{self.quantization}" / f"seg_{self.segment_len}"
        )

        datasets = {}
        for split, files in splits.items():
            logging.info(f"\n[Preprocessor] Processing {split} set...")
            out_dir = base_dir / split
            out_dir.mkdir(parents=True, exist_ok=True)

            split_target = int(self.total_target * ratios[split])
            samples_per_genre = split_target // len(self.genres)

            self._process_midi_files(files, out_dir, samples_per_genre, split_target)

        self.tokenizer.prune(keep_ratio=self.dataset_cfg["keep_ratio"])
        self.tokenizer.save()
        logging.info(f"[Tokenizer] Vocabulary size: {len(self.tokenizer)} tokens")
        self.tokenizer.analyze_tokens()
        self._remap_tokens_to_pruned_vocab(base_dir)

        for split in ["train", "val", "test"]:
            out_dir = base_dir / split
            datasets[split] = DrumDataset(self.cfg, out_dir, include_genre=True)

        logging.info("[Preprocessor] Preprocessing complete.\n")

        return datasets["train"], datasets["val"], datasets["test"], self.tokenizer
