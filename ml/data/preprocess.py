import logging
import os
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from sympy import rem
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

    def _cyclic_positional_encoding(self):
        t = np.arange(self.segment_len)
        period = self.quantization * 4
        return np.stack(
            [np.sin(2 * np.pi * t / period), np.cos(2 * np.pi * t / period)], axis=1
        )  # shape: (length, 2)

    def _remap_tokens_to_pruned_vocab(self, dataset_dir):
        """
        Remap old token IDs to new contiguous indices after vocab pruning.
        Tokens not in the kept vocab are replaced with UNK id.
        """
        dataset_dir = Path(dataset_dir)
        npz_files = list(dataset_dir.rglob("*.npz"))
        unk_id = self.tokenizer.unk_id

        # Map old IDs -> new contiguous IDs
        # self.tokenizer.vocab is a dict {token_str: old_id}
        old_to_new = {
            old_id: new_id
            for new_id, old_id in enumerate(self.tokenizer.vocab.values())
        }

        remapped_files = 0
        remapped_tokens = 0

        for f in npz_files:
            data = np.load(f, allow_pickle=True)
            tokens = data["tokens"]

            # Remap each token ID, fallback to unk_id
            remapped = np.vectorize(
                lambda t: old_to_new.get(t, unk_id), otypes=[np.int32]
            )(tokens)
            n_changed = np.sum(tokens != remapped)

            if n_changed > 0:
                np.savez(
                    f,
                    tokens=remapped.astype(np.int32),
                    positions=data["positions"],
                    genre=data["genre"],
                    bpm=data["bpm"],
                )
                remapped_files += 1
                remapped_tokens += n_changed

        logging.info(
            f"[Remap] {remapped_files}/{len(npz_files)} files remapped | {remapped_tokens} tokens changed."
        )

    def _shuffle_and_store_samples(self, temp_dir, dataset_dir):
        """Split files into train/val/test while maintaining genre balance."""
        rng = random.Random(self.seed)
        for genre in self.genres:
            genre_dir = temp_dir / genre
            files = [p for p in genre_dir.glob("*.*") if p.is_file()]
            rng.shuffle(files)

            n = len(files)
            n_train = int(n * self.train_test_val_split[0])
            n_val = int(n * self.train_test_val_split[1])
            n_test = n - n_train - n_val

            train_files = files[:n_train]
            val_files = files[n_train : n_train + n_val]
            test_files = files[n_train + n_val :]

            os.makedirs(dataset_dir / "train" / genre, exist_ok=True)
            os.makedirs(dataset_dir / "val" / genre, exist_ok=True)
            os.makedirs(dataset_dir / "test" / genre, exist_ok=True)

            for f in train_files:
                target = dataset_dir / "train" / genre / f.name
                shutil.move(str(f), str(target))

            for f in val_files:
                target = dataset_dir / "val" / genre / f.name
                shutil.move(str(f), str(target))

            for f in test_files:
                target = dataset_dir / "test" / genre / f.name
                shutil.move(str(f), str(target))
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            logging.info(f"[Preprocessor] Warning: failed to remove temp dir: {e}")

    def _process_midi_files(self, midi_files, output_dir):
        counts = defaultdict(int)
        saved = 0
        samples_per_genre = int(self.total_target / len(self.genres))
        pbar = tqdm(
            total=self.total_target, desc=f"Saving to {output_dir.name}", unit="seg"
        )

        # Shuffle all files globally for randomness
        random.shuffle(midi_files)

        for midi_path in midi_files:
            genre, bpm = self._extract_metadata(midi_path)
            if genre not in self.genres:
                continue
            if 0 > bpm > 250:
                continue
            if counts[genre] >= samples_per_genre:
                continue

            tracks = self.midi_reader.read_file(str(midi_path))
            if not tracks:
                continue

            for name, mat in tracks.items():
                mat = self._simplify_matrix(mat, self.pitch_groups)
                mat = self._trim_silence_full_segments(mat)
                n_steps = mat.shape[0]
                if n_steps < self.segment_len:
                    continue

                # overlapping starts
                starts = np.arange(
                    0, n_steps - self.segment_len + 1, self.segment_len // 2
                )
                np.random.shuffle(starts)

                genre_dir = output_dir / genre
                genre_dir.mkdir(exist_ok=True)

                for s in starts:
                    if counts[genre] >= samples_per_genre or saved >= self.total_target:
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

        pbar.close()
        logging.info(f"\nFinal counts per genre in {output_dir.name}:")
        for g in sorted(self.genres):
            logging.info(f"  {g}: {counts[g]} samples")

    def preprocess_dataset(self):
        midi_dir = Path(self.midi_dir)
        midi_files = list(midi_dir.rglob("*.mid")) + list(midi_dir.rglob("*.midi"))
        logging.info(f"[Preprocessor] Found {len(midi_files)} MIDI files")

        base_dir = (
            Path(self.save_dir) / f"q_{self.quantization}" / f"seg_{self.segment_len}"
        )
        logging.info(f"\n[Preprocessor] Extracting all smamples...")
        temp_dir = base_dir / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        self._process_midi_files(midi_files, temp_dir)

        logging.info(f"\n[Preprocessor] Shuffling all smamples...")
        self._shuffle_and_store_samples(temp_dir, base_dir)

        self.tokenizer.prune(keep_ratio=self.dataset_cfg["keep_ratio"])
        self.tokenizer.save()
        logging.info(f"[Tokenizer] Vocabulary size: {len(self.tokenizer)} tokens")
        self.tokenizer.analyze_tokens()
        self._remap_tokens_to_pruned_vocab(base_dir)

        datasets = {}
        for split in ["train", "val", "test"]:
            out_dir = base_dir / split
            datasets[split] = DrumDataset(self.cfg, out_dir, include_genre=True)

        logging.info("[Preprocessor] Preprocessing complete.\n")

        return datasets["train"], datasets["val"], datasets["test"], self.tokenizer
