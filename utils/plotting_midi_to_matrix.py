import matplotlib.pyplot as plt
from ml.data.midi import Midi


def compare_quantizations(
    midi_path, quantizations, track_key=None, pitch_range=(30, 70)
):
    """
    Compare drum matrices for different quantizations.

    Args:
        midi_path (str): path to MIDI file.
        quantizations (list[int]): list of quantization values to compare.
        track_key (str or None): name of the track to plot. If None, first track is used.
        pitch_range (tuple): (min_pitch, max_pitch) to focus on.
    """

    n = len(quantizations)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6), sharey=True)

    for i, q in enumerate(quantizations):
        reader = Midi(quantization=q)
        tracks = reader.read_file(midi_path)

        # choose a track
        if track_key and track_key in tracks:
            mat = tracks[track_key]["drum_matrix"]
            title = f"{track_key} (q={q})"
        else:
            first_key = list(tracks.keys())[0]
            mat = tracks[first_key]["drum_matrix"]
            title = f"{first_key} (q={q})"

        # focus on pitch range
        mat_range = mat[:, pitch_range[0] : pitch_range[1]]

        ax = axes[i] if n > 1 else axes
        im = ax.imshow(
            mat_range.T,
            aspect="auto",
            origin="lower",
            cmap="magma",
            interpolation="nearest",
            vmin=0,
            vmax=127,
        )
        ax.set_title(title)
        ax.set_xlabel("Time steps")
        if i == 0:
            ax.set_ylabel("MIDI Pitch")
            # add tick labels for some common drums
            ax.set_yticks(
                [
                    36 - pitch_range[0],
                    38 - pitch_range[0],
                    42 - pitch_range[0],
                    44 - pitch_range[0],
                    46 - pitch_range[0],
                    49 - pitch_range[0],
                ]
            )
            ax.set_yticklabels(
                ["Kick(36)", "Snare(38)", "HHc(42)", "HHp(44)", "HHo(46)", "Crash(49)"]
            )

    fig.colorbar(im, ax=axes.ravel().tolist(), label="Velocity")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    compare_quantizations(
        "/Users/arthurreuss/Library/Mobile Documents/iCloud~md~obsidian/Documents/Brain Extension/Bachelor AI/Neural Networks/Project/NN_Drumming_Project/dataset/e-gmd-v1.0.0/drummer1/eval_session/1_funk-groove1_138_beat_4-4_1.midi",
        quantizations=[4, 8, 16, 32],
        pitch_range=(30, 70),
    )
