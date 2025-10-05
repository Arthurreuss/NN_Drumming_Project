from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset


class DrumDataset(Dataset):
    def __init__(self, data_dir, config_path, include_genre=True):
        """
        Args:
            data_dir (str): Directory containing .npz preprocessed files.
            config_path (str): Path to config.yaml containing list of genres.
            include_genre (bool): Whether to append one-hot genre encoding to X.
        """
        self.files = list(Path(data_dir).rglob("*.npz"))
        assert len(self.files) > 0, f"No .npz files found in {data_dir}"

        self.include_genre = include_genre
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        self.genres = cfg.get("dataset", {}).get("genres", [])
        self.genre_to_idx = {g: i for i, g in enumerate(self.genres)}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx], allow_pickle=True)
        X = torch.tensor(data["X"], dtype=torch.float32)  # (T, 9)
        genre = data["genre"].item() if "genre" in data else "unknown"
        # bpm = int(data["bpm"]) if "bpm" in data else -1

        if self.include_genre and genre in self.genre_to_idx:
            onehot = torch.zeros(len(self.genres))
            onehot[self.genre_to_idx[genre]] = 1.0
            onehot = onehot.repeat(X.shape[0], 1)  # (T, num_genres)
            X = torch.cat([X, onehot], dim=1)  # (T, 9 + num_genres)

        X_in = X[:-1]
        X_out = X[1:]
        return X_in, X_out
