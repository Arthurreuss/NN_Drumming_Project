import matplotlib.pyplot as plt


def plot_drum_matrix(matrix, title="Generated Drum Pattern", save_path=None):
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
    if save_path:
        fig.savefig(save_path)
    plt.close(fig)
