import matplotlib.pyplot as plt

from ml.data.dataset import DrumDataset


def plot_drum_sample(dataset, index=0):
    """
    Visualize a tokenized drum sequence sample (input & target tokens).
    Press ← / → to navigate.
    """
    current_index = index

    def _draw(idx):
        sample = dataset[idx]
        tokens_in = sample["tokens"].numpy()
        tokens_out = sample["targets"].numpy()
        positions = sample["positions"].numpy()
        genre_id = sample["genre_id"].item()

        genre_name = (
            dataset.genres[genre_id] if genre_id < len(dataset.genres) else "unknown"
        )

        fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
        fig.suptitle(f"Sample {idx+1}/{len(dataset)}  |  Genre: {genre_name}")

        axes[0].plot(positions, tokens_in, drawstyle="steps-mid")
        axes[0].set_ylabel("Token ID")
        axes[0].set_title("Input tokens (X_in)")

        axes[1].plot(positions, tokens_out, drawstyle="steps-mid", color="orange")
        axes[1].set_xlabel("Timesteps")
        axes[1].set_ylabel("Token ID")
        axes[1].set_title("Target tokens (X_out)")

        plt.tight_layout()

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
    data_dir = "dataset/processed/q_16/seg_64/test"

    dataset = DrumDataset(data_dir, include_genre=True)
    print(f"Dataset size: {len(dataset)} samples")
    plot_drum_sample(dataset, index=0)
