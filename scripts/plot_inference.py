import matplotlib.pyplot as plt

from ml.utils.cfg import load_config


def plot_drum_matrix(matrix, title="Generated Drum Pattern", save_path=None):
    """
    matrix: (T, 9) numpy array with values 0–1
    """
    cfg = load_config()
    instruments = cfg["dataset_creation"]["pitch_groups"].keys()

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(
        matrix.T,
        aspect="auto",
        origin="lower",
        cmap="magma",
        vmin=0,
        vmax=1,
        interpolation="none",
    )
    ax.set_yticks(range(len(instruments)))
    ax.set_yticklabels(instruments)
    ax.set_xlabel("Timesteps")
    ax.set_title(title)

    # --- add x-axis ticks every 4 steps ---
    step = 64
    timesteps = matrix.shape[0]
    ax.set_xticks(range(0, timesteps, step))
    ax.set_xticklabels(range(0, timesteps, step))

    # add colorbar legend
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Velocity / Activation (0–1)", rotation=270, labelpad=15)

    plt.tight_layout()
    # plt.show()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)
