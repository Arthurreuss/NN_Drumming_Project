from xml.etree.ElementInclude import include

import matplotlib.pyplot as plt
import numpy as np
from ml.data.dataset import DrumDataset


def plot_drum_sample(dataset, index=0, include_genre=True):
    """
    Plot input (X_in) and label (X_out) of one sample in the dataset.
    Press any key in the plot window to advance to the next sample.
    """
    current_index = index

    def _draw(idx):
        X_in, X_out = dataset[idx]
        X_in = X_in.numpy()
        X_out = X_out.numpy()

        num_features = X_in.shape[1]
        num_genres = len(dataset.genres)
        base_instruments = [
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

        has_genre = num_features > len(base_instruments)

        if has_genre:
            instr_in = X_in
            instr_out = X_out
            instruments = base_instruments + dataset.genres
        else:
            instr_in = X_in
            instr_out = X_out
            instruments = base_instruments

        # --- Plot ---
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        title = f"Sample {idx+1}/{len(dataset)}"
        fig.suptitle(title)

        # Input
        axes[0].imshow(
            instr_in.T, aspect="auto", origin="lower", cmap="magma", vmin=0, vmax=1
        )
        axes[0].set_yticks(range(len(instruments)))
        axes[0].set_yticklabels(instruments)
        axes[0].set_ylabel("Instruments")
        axes[0].set_title("Input (X_in)")

        # Target
        axes[1].imshow(
            instr_out.T, aspect="auto", origin="lower", cmap="magma", vmin=0, vmax=1
        )
        axes[1].set_yticks(range(len(instruments)))
        axes[1].set_yticklabels(instruments)
        axes[1].set_xlabel("Timesteps")
        axes[1].set_ylabel("Instruments")
        axes[1].set_title("Target (X_out)")

        plt.tight_layout()

        # Interactive browsing
        def on_key(event):
            plt.close(fig)
            nonlocal current_index
            if event.key == "right":
                current_index = (current_index + 1) % len(dataset)
            elif event.key == "left":
                current_index = (current_index - 1) % len(dataset)
            _draw(current_index)

        fig.canvas.mpl_connect("key_press_event", on_key)
        plt.show()

    _draw(current_index)


if __name__ == "__main__":
    data_dir = "dataset/preprocessed"
    config_path = "config.yaml"

    dataset = DrumDataset(data_dir, config_path, include_genre=False)
    X_in, X_out = dataset[0]

    print("Input shape:", X_in.shape)
    print("Output shape:", X_out.shape)
    plot_drum_sample(dataset, index=0)
