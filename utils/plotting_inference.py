import matplotlib.pyplot as plt
import numpy as np
import torch

from ml.data.dataset import DrumDataset
from ml.data.tokenizer import SimpleTokenizer
from ml.models.lstm import Seq2SeqLSTM
from utils.cfg import load_config


# -------------------- Visualization --------------------
def plot_drum_matrix(matrix, title="Generated Drum Pattern"):
    """
    matrix: (T, 9) numpy array with values 0–1
    """
    instruments = [
        "Kick",
        "Snare",
        "HH Closed",
        "HH Open",
        "Tom L",
        "Tom M",
        "Tom H",
        "Crash",
        "Ride",
    ]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(matrix.T, aspect="auto", origin="lower", cmap="magma", vmin=0, vmax=1)
    ax.set_yticks(range(len(instruments)))
    ax.set_yticklabels(instruments)
    ax.set_xlabel("Timesteps")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def show_interactive_predictions(
    model, dataset, tokenizer, device, steps=64, temperature=0.9
):
    """
    Use ← and → arrows to browse samples and generate new predictions.
    Press ESC or Q to quit.
    """
    idx = 0
    fig = None

    def generate_sample(index):
        sample = dataset[index]
        prim_tok = sample["tokens"].unsqueeze(0).to(device)
        prim_pos = sample["positions"].unsqueeze(0).to(device)
        genre_id = sample["genre_id"].unsqueeze(0).to(device)

        gen_tokens = (
            model.generate(
                prim_tok, prim_pos, genre_id, steps=steps, temperature=temperature
            )
            .squeeze(0)
            .cpu()
            .numpy()
        )

        detok = []
        for t in gen_tokens:
            try:
                vec = tokenizer.detokenize(int(t))
            except KeyError:
                vec = np.zeros(9)
            detok.append(vec)
        matrix = np.stack(detok, axis=0)
        genre_name = dataset.genres[int(sample["genre_id"].item())]
        return matrix, genre_name

    def draw(index):
        nonlocal fig, ax
        matrix, genre = generate_sample(index)
        if fig is None:
            fig, ax = plt.subplots(figsize=(10, 4))
        ax.clear()
        ax.imshow(matrix.T, aspect="auto", origin="lower", cmap="magma", vmin=0, vmax=1)
        ax.set_yticks(range(9))
        ax.set_yticklabels(
            [
                "Kick",
                "Snare",
                "HH Cl",
                "HH Op",
                "Tom L",
                "Tom M",
                "Tom H",
                "Crash",
                "Ride",
            ]
        )
        ax.set_title(f"Generated (Sample {index+1}/{len(dataset)} | {genre})")
        ax.set_xlabel("Timesteps")
        fig.canvas.draw_idle()

    def on_key(event):
        nonlocal idx, running
        if event.key in ["right", "left"]:
            if event.key == "right":
                idx = (idx + 1) % len(dataset)
            elif event.key == "left":
                idx = (idx - 1) % len(dataset)
            draw(idx)
        elif event.key in ["escape", "q"]:
            running = False
            plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.canvas.mpl_connect("key_press_event", on_key)
    draw(idx)
    plt.ion()  # turn on interactive mode

    running = True
    while running and plt.fignum_exists(fig.number):
        plt.pause(0.1)
