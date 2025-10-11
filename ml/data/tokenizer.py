import numpy as np

from utils.cfg import load_config


class SimpleTokenizer:
    def __init__(self, vocab=None):
        self.cfg = load_config()
        self.vocab = vocab or {}

    def __len__(self):
        return len(self.vocab)

    def _round_to_loudness(self, vector):
        ranges = self.cfg["dataset"]["loudness_ranges"]
        rounded = np.zeros_like(vector)
        for i, val in enumerate(vector):
            for r in ranges:
                if val <= r:
                    rounded[i] = r
                    break
        return rounded

    def tokenize(self, vector):
        vector = self._round_to_loudness(vector)
        key = tuple(vector.tolist())

        # create new token if unseen
        if key not in self.vocab:
            token_id = len(self.vocab) + 1
            self.vocab[key] = token_id
        else:
            token_id = self.vocab[key]
        return token_id

    def detokenize(self, token_id):
        for key, val in self.vocab.items():
            if val == token_id:
                return np.array(key)
        raise ValueError(f"Token ID {token_id} not found in vocabulary.")
