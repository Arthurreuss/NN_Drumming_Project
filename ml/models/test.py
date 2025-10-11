from re import T

import numpy as np
import torch
from ml.data.dataset import DrumDataset
from ml.models.lstm import LSTMNextStep

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
T = 63  # model trained on sequences of length 63
F = 9  # model trained on features of size 9 (no genre)


# ---- Predict / Generate ----
@torch.no_grad()
def generate_next(model, context, steps=16, threshold=0.5, sample=False):
    """
    context: tensor (1, L, F) with 1<=L<=63; model trained on (T=63,F=10)
    steps: number of future steps to generate
    threshold: for deterministic binarization if sample=False
    sample: if True, Bernoulli-sample from sigmoid probs
    Returns tensor (1, L+steps, F) in {0,1}
    """
    model.eval()
    x = context.clone().to(DEVICE)  # (1,L,F)
    while x.size(1) < T:  # left-pad with zeros up to 63 if needed
        pad = torch.zeros(1, 1, F, device=DEVICE)
        x = torch.cat([pad, x], dim=1)
    seq = x.clone()  # working window length 63
    out_all = context.clone().to(DEVICE)

    for _ in range(steps):
        logits = model(seq)[:, -1, :]  # last step logits
        p = torch.sigmoid(logits)
        if sample:
            step = torch.bernoulli(p).float()
        else:
            step = (p >= threshold).float()
        out_all = torch.cat([out_all, step.unsqueeze(1)], dim=1)
        seq = torch.cat([seq[:, 1:, :], step.unsqueeze(1)], dim=1)  # slide window
    return out_all.detach().cpu()


from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_drum_sequence(file_path, threshold=0.5, title=None, save_to=None, show=True):
    """
    file_path: path to generated .npy or .npz
    Expects array shape (T, F) or (1, T, F). Values in [0,1] or {0,1}.
    """
    file_path = Path(file_path)
    if file_path.suffix == ".npy":
        arr = np.load(file_path)
    elif file_path.suffix == ".npz":
        data = np.load(file_path)
        # try common keys
        for k in ("generated", "arr_0"):
            if k in data:
                arr = data[k]
                break
        else:
            raise KeyError(
                "No suitable array key found in npz (tried 'generated', 'arr_0')."
            )
    else:
        raise ValueError("Unsupported file type. Use .npy or .npz")

    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Expected (T,F) or (1,T,F). Got {arr.shape}")

    # binarize if needed
    arr_bin = (arr >= threshold).astype(float)

    plt.figure(figsize=(12, 3 + 0.2 * arr_bin.shape[1]))
    plt.imshow(arr_bin.T, aspect="auto", interpolation="nearest", origin="lower")
    plt.xlabel("Time steps")
    plt.ylabel("Drum channels")
    plt.title(title or f"{file_path.name}  |  shape {arr_bin.shape}")
    plt.colorbar(label="hit")
    plt.tight_layout()

    if save_to:
        plt.savefig(save_to, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":

    # define the same model class and hyperparams used for training
    model = LSTMNextStep(hidden_size=256, num_layers=3, dropout=0.1).to(DEVICE)

    # load checkpoint
    ckpt = torch.load("checkpoints/big_jazz_best.pt", map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])  # key name from your training code
    model.eval()

    with torch.no_grad():
        dataset = DrumDataset(
            data_dir="dataset/processed",
            config_path="config.yaml",
            include_genre=False,
        )

        # take one real sequence as context (first 16 steps)
        x0, _ = dataset[110000]
        ctx = x0[:16].unsqueeze(0)  # (1,16,9)
        gen = generate_next(
            model, ctx, steps=64, threshold=0.5, sample=False
        )  # (1, 16+32, 9)
        # gen contains binary predictions; save if you want
        np.save("generated.npy", gen.numpy())

    plot_drum_sequence("generated.npy", threshold=0.5, save_to="generated_plot.png")
