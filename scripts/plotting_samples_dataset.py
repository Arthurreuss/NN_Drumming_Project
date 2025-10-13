import matplotlib.pyplot as plt
import numpy as np

from ml.data.dataset import DrumDataset
from ml.data.midi import Midi
from ml.data.tokenizer import SimpleTokenizer


def plot_drum_sample(dataset, matrix=True, index=0):
    """
    Visualize a tokenized or detokenized drum sequence sample (input & target).
    Press ← / → to navigate between samples.
    """
    current_index = index
    tokenizer = SimpleTokenizer()

    def _draw(idx):
        sample = dataset[idx]
        tokens_in = sample["tokens"].numpy()
        tokens_out = sample["targets"].numpy()
        positions = sample["positions"].numpy()
        genre_id = int(sample["genre_id"].item())

        if matrix:
            detok_in, detok_out = [], []
            for t_in, t_out in zip(tokens_in, tokens_out):
                try:
                    vec_in = tokenizer.detokenize(int(t_in))
                    vec_out = tokenizer.detokenize(int(t_out))
                except Exception:
                    vec_in = np.zeros(9)
                    vec_out = np.zeros(9)
                detok_in.append(vec_in)
                detok_out.append(vec_out)

            tokens_in = np.stack(detok_in, axis=0)  # (T, 9)
            tokens_out = np.stack(detok_out, axis=0)  # (T, 9)
            output_path = f"outputs/sample_{idx+1}_input_matrix.mid"
            Midi(16).create_midi(tokens_in, output_path=output_path)

        genre_name = (
            dataset.genres[genre_id] if genre_id < len(dataset.genres) else "unknown"
        )

        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        fig.suptitle(f"Sample {idx+1}/{len(dataset)}  |  Genre: {genre_name}")

        if not matrix:
            # Plot token IDs as step curves
            axes[0].plot(positions, tokens_in, drawstyle="steps-mid", label="Input")
            axes[0].set_ylabel("Token ID")
            axes[0].set_title("Input Tokens (X_in)")

            axes[1].plot(
                positions,
                tokens_out,
                drawstyle="steps-mid",
                color="orange",
                label="Target",
            )
            axes[1].set_xlabel("Timesteps")
            axes[1].set_ylabel("Token ID")
            axes[1].set_title("Target Tokens (X_out)")
        else:
            # Plot drum matrices
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

            axes[0].imshow(
                tokens_in.T, aspect="auto", origin="lower", cmap="magma", vmin=0, vmax=1
            )
            axes[0].set_yticks(range(len(instruments)))
            axes[0].set_yticklabels(instruments)
            axes[0].set_title("Input Drum Pattern (X_in)")

            axes[1].imshow(
                tokens_out.T,
                aspect="auto",
                origin="lower",
                cmap="magma",
                vmin=0,
                vmax=1,
            )
            axes[1].set_yticks(range(len(instruments)))
            axes[1].set_yticklabels(instruments)
            axes[1].set_xlabel("Timesteps")
            axes[1].set_title("Target Drum Pattern (X_out)")

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        def on_key(event):
            nonlocal current_index
            plt.close(fig)
            if event.key == "right":
                current_index = (current_index + 1) % len(dataset)
            elif event.key == "left":
                current_index = (current_index - 1) % len(dataset)
            elif event.key in ["escape", "q"]:
                return
            _draw(current_index)

        fig.canvas.mpl_connect("key_press_event", on_key)
        plt.savefig("outputs/drum_sample.png", dpi=300, bbox_inches="tight")
        plt.show()
        plt.close(fig)

    _draw(current_index)


if __name__ == "__main__":
    data_dir = "dataset/processed/q_32/seg_512/test"
    dataset = DrumDataset(data_dir, include_genre=True)
    print(f"Dataset size: {len(dataset)} samples")
    plot_drum_sample(dataset, matrix=True, index=0)
