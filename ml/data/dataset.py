from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class DrumDataset(Dataset):
    def __init__(self, cfg, data_dir, include_genre=True):
        self.files = list(Path(data_dir).rglob("*.npz"))
        assert self.files, f"No .npz files found in {data_dir}"
        self.include_genre = include_genre
        self.genres = cfg["dataset_creation"]["genres"]
        self.genre_to_idx = {g: i for i, g in enumerate(self.genres)}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx], allow_pickle=True)

        tokens = torch.tensor(data["tokens"], dtype=torch.long)  # (T,)
        positions = torch.tensor(data["positions"], dtype=torch.long)  # (T,)
        genre_val = data["genre"]
        if isinstance(genre_val, bytes):  # npz may store as bytes
            genre_val = genre_val.decode()
        genre_id = self.genre_to_idx.get(str(genre_val), 0)
        genre_id = torch.tensor(genre_id, dtype=torch.long)

        # Inputs = tokens[:-1], Targets = tokens[1:]
        X_in = tokens[:-1]
        X_out = tokens[1:]
        pos_in = positions[:-1]
        bpm = torch.from_numpy(data["bpm"]).float()

        return {
            "tokens": X_in,
            "positions": pos_in,
            "genre_id": genre_id,
            "targets": X_out,
            "bpm": bpm,
        }
