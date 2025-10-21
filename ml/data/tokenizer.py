import logging
from collections import Counter

import numpy as np


class SimpleTokenizer:
    def __init__(self, cfg, path, vocab=None):
        self.cfg = cfg
        self.path = path
        self.freqs = Counter()
        self.unk_token = "<UNK>"
        self.unk_id = 0

        if vocab is None:
            try:
                self.vocab = self.load()
                logging.info(
                    f"[Tokenizer] Loaded existing vocab ({len(self.vocab)} tokens)"
                )
            except Exception as e:
                logging.info(f"[Tokenizer] Warning: failed to load existing vocab: {e}")
                self.vocab = {}
        else:
            self.vocab = vocab

    def __len__(self):
        return len(self.vocab)

    def _round_to_loudness(self, vector):
        ranges = self.cfg["dataset_creation"]["loudness_ranges"]
        rounded = np.zeros_like(vector)
        for i, val in enumerate(vector):
            for r in ranges:
                if val <= r:
                    rounded[i] = r
                    break
        return rounded

    def tokenize(self, vector):
        """Quantize, record frequency, and map vector to integer token ID."""
        vector = self._round_to_loudness(vector)
        key = tuple(vector.tolist())

        # Track frequency for pruning
        self.freqs[key] += 1

        # Create new token if unseen
        if key not in self.vocab:
            token_id = len(self.vocab) + 1
            self.vocab[key] = token_id
        else:
            token_id = self.vocab[key]

        return token_id

    def detokenize(self, token_id):
        """Convert token ID back to loudness vector."""
        for key, val in self.vocab.items():
            if val == token_id:
                return np.array(key)
        # fallback for unknowns
        return np.zeros(9)

    def prune(self, min_freq=50):
        """Remove rare tokens and map them to UNK (id=0)."""
        if not self.freqs:
            logging.info("[Tokenizer] No frequency data found — skipping pruning.")
            return

        kept = {}
        new_id = 1  # reserve 0 for UNK
        removed = 0

        for key, freq in self.freqs.items():
            if freq >= min_freq:
                kept[key] = new_id
                new_id += 1
            else:
                removed += 1

        kept[self.unk_token] = self.unk_id
        self.vocab = kept

        logging.info(f"[Tokenizer] Pruned vocab to {len(self.vocab)} tokens (+ UNK).")
        logging.info(f"[Tokenizer] Removed {removed} rare tokens (freq < {min_freq}).")

    def save(self):
        """Save vocab and freqs to .npy file."""
        np.save(
            self.path, {"vocab": self.vocab, "freqs": self.freqs}, allow_pickle=True
        )
        logging.info(f"[Tokenizer] Saved to {self.path}")

    def load(self):  # TODO: make path configurable
        """Load vocab and freqs from .npy file."""
        data = np.load(self.path, allow_pickle=True).item()
        self.vocab = data.get("vocab", {})
        self.freqs = data.get("freqs", Counter())
        logging.info(
            f"[Tokenizer] Loaded vocab with {len(self.vocab)} tokens from {self.path}"
        )
        return self.vocab


class BeatTokenizer:
    def __init__(self, cfg, path, vocab=None):
        self.cfg = cfg
        self.path = path
        self.freqs = Counter()
        self.unk_token = "<UNK>"
        self.unk_id = 0
        self.q = self.cfg["dataset_creation"]["quantization"]

        if vocab is None:
            try:
                self.vocab = self.load()
                logging.info(
                    f"[Tokenizer] Loaded existing vocab ({len(self.vocab)} tokens)"
                )
            except Exception as e:
                logging.info(f"[Tokenizer] Warning: failed to load existing vocab: {e}")
                self.vocab = {}
        else:
            self.vocab = vocab

    def __len__(self):
        return len(self.vocab)

    def _round_to_loudness(self, vector):
        """
        Quantize each value in `vector` to the nearest upper threshold
        from loudness_ranges (sorted ascending).
        """
        ranges = np.array(
            self.cfg["dataset_creation"]["loudness_ranges"], dtype=np.float32
        )
        x = np.asarray(vector, dtype=np.float32)

        idx = np.searchsorted(ranges, x, side="left")
        idx = np.clip(idx, 0, len(ranges) - 1)
        return ranges[idx]

    def tokenize(self, seq):
        """Tokenize a sequence every 16 timesteps (one beat). Each beat (16x num_drums) becomes one token."""

        tokens = []
        T = len(seq)
        for start in range(0, T, self.q):
            beat_chunk = seq[start : start + self.q]
            if len(beat_chunk) < self.q:
                continue  # ignore incomplete beat

            # flatten and quantize to loudness grid
            flat = self._round_to_loudness(beat_chunk.flatten())
            key = tuple(flat.tolist())

            self.freqs[key] += 1
            if key not in self.vocab:
                self.vocab[key] = len(self.vocab) + 1
            tokens.append(self.vocab[key])
        return tokens

    def detokenize(self, token_id):
        """Convert token ID back to loudness matrix."""
        D = len(self.cfg["dataset_creation"]["pitch_groups"])

        for key, val in self.vocab.items():
            if key == self.unk_token:
                continue
            if val == token_id:
                arr = np.array(key)
                # safety: pad or trim to expected length
                if arr.size < self.q * D:
                    arr = np.pad(arr, (0, self.q * D - arr.size))
                elif arr.size > self.q * D:
                    arr = arr[: self.q * D]
                return arr.reshape(self.q, D)

        # fallback for unknowns
        return np.zeros((self.q, D))

    def prune(self, keep_ratio=0.95):
        if not self.freqs:
            logging.info("[Tokenizer] No frequency data found — skipping pruning.")
            return

        # sort tokens by frequency (descending)
        sorted_items = sorted(self.freqs.items(), key=lambda x: x[1], reverse=True)
        total_freq = sum(freq for _, freq in sorted_items)
        cutoff = total_freq * keep_ratio

        kept = {}
        new_id = 1
        cum_freq = 0
        for key, freq in sorted_items:
            if cum_freq < cutoff:
                kept[key] = new_id
                new_id += 1
                cum_freq += freq
            else:
                break

        pruned_tokens = len(sorted_items) - len(kept)
        pruned_freq = total_freq - cum_freq
        kept_freq_share = cum_freq / total_freq * 100

        kept[self.unk_token] = self.unk_id
        self.vocab = kept

        logging.info(f"[Tokenizer] Kept {len(kept)-1} tokens (+UNK).")
        logging.info(f"[Tokenizer] Pruned {pruned_tokens} rare tokens.")
        logging.info(
            f"[Tokenizer] Kept freq share: {kept_freq_share:.2f}%  |  "
            f"Pruned freq share: {100 - kept_freq_share:.2f}%"
        )

    def save(self):
        np.save(
            self.path, {"vocab": self.vocab, "freqs": self.freqs}, allow_pickle=True
        )
        logging.info(f"[Tokenizer] Saved to {self.path}")

    def load(self):
        data = np.load(self.path, allow_pickle=True).item()
        self.vocab = data.get("vocab", {})
        self.freqs = data.get("freqs", Counter())
        logging.info(
            f"[Tokenizer] Loaded vocab with {len(self.vocab)} tokens from {self.path}"
        )
        return self.vocab

    def analyze_tokens(self):
        """
        Inspect vocabulary contents and statistics interactively.
        - show_matrices=True  -> visualize each token as matrix (requires plot_drum_matrix)
        - pause=True          -> press Enter to view next token
        """
        total_tokens = len(self.vocab)
        total_freq = sum(self.freqs.values())

        kept = [k for k in self.vocab if k in self.freqs]
        pruned = [k for k in self.freqs if k not in self.vocab]

        kept_freq = sum(self.freqs[k] for k in kept)
        pruned_freq = sum(self.freqs[k] for k in pruned)

        logging.info(f"[Tokenizer] Total observed frequency count: {total_freq:,}")
        logging.info(
            f"[Tokenizer] Kept tokens: {len(kept)}  (freq sum = {kept_freq:,})"
        )
        logging.info(
            f"[Tokenizer] Pruned tokens: {len(pruned)}  (freq sum = {pruned_freq:,})"
        )

        if total_freq > 0:
            kept_ratio = kept_freq / total_freq * 100
            pruned_ratio = pruned_freq / total_freq * 100
            logging.info(
                f"[Tokenizer] Kept freq share: {kept_ratio:.2f}%  |  Pruned freq share: {pruned_ratio:.2f}%"
            )

        if pruned:
            min_kept_freq = min(self.freqs[k] for k in kept) if kept else 0
            max_pruned_freq = max(self.freqs[k] for k in pruned)
            logging.info(
                f"[Tokenizer] Lowest kept freq: {min_kept_freq}, Highest pruned freq: {max_pruned_freq}"
            )
