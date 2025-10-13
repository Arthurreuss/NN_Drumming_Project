from collections import Counter

import numpy as np

from ml.utils.cfg import load_config


class SimpleTokenizer:
    def __init__(self, path, vocab=None):
        self.cfg = load_config()
        self.path = path
        self.freqs = Counter()
        self.unk_token = "<UNK>"
        self.unk_id = 0

        if vocab is None:
            try:
                self.vocab = self.load()
                print(f"[Tokenizer] Loaded existing vocab ({len(self.vocab)} tokens)")
            except Exception as e:
                print(f"[Tokenizer] Warning: failed to load existing vocab: {e}")
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
            print("[Tokenizer] No frequency data found — skipping pruning.")
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

        print(f"[Tokenizer] Pruned vocab to {len(self.vocab)} tokens (+ UNK).")
        print(f"[Tokenizer] Removed {removed} rare tokens (freq < {min_freq}).")

    def save(self):
        """Save vocab and freqs to .npy file."""
        np.save(
            self.path, {"vocab": self.vocab, "freqs": self.freqs}, allow_pickle=True
        )
        print(f"[Tokenizer] Saved to {self.path}")

    def load(self):  # TODO: make path configurable
        """Load vocab and freqs from .npy file."""
        data = np.load(self.path, allow_pickle=True).item()
        self.vocab = data.get("vocab", {})
        self.freqs = data.get("freqs", Counter())
        print(
            f"[Tokenizer] Loaded vocab with {len(self.vocab)} tokens from {self.path}"
        )
        return self.vocab
