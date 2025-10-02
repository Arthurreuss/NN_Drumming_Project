import numpy as np
from expirimental.read_midi import Read_midi
import os

# Map MIDI pitches to simplified drum classes
DRUM_MAP = {
    36: "Kick", 35: "Kick",
    38: "Snare", 40: "Snare",
    42: "HiHatClosed", 44: "HiHatPedal", 46: "HiHatOpen",
    49: "Crash", 57: "Crash",
    51: "Ride", 53: "Ride", 59: "Ride",
    45: "LowTom", 47: "LowTom",
    48: "HighTom", 50: "HighTom"
}

INSTRUMENTS = sorted(set(DRUM_MAP.values()))
INST_TO_IDX = {inst: i for i, inst in enumerate(INSTRUMENTS)}

# Quantization settings (steps per beat)
QUANTIZATION = 8  # 4 = 16th note resolution, 8 = 32nd note, 2 = 8th note
BEATS_PER_BAR = 4  # Assuming 4/4 time signature
STEPS_PER_BAR = QUANTIZATION * BEATS_PER_BAR

def pianoroll_to_matrix(pianoroll_dict, normalize_velocity=True):
    matrices = []
    for pr in pianoroll_dict.values():
        T, _ = pr.shape
        reduced = np.zeros((T, len(INSTRUMENTS)), dtype=np.float16)
        for pitch, inst in DRUM_MAP.items():
            col = INST_TO_IDX[inst]
            if pitch < pr.shape[1]:
                velocity_values = pr[:, pitch].astype(np.float16)
                if normalize_velocity:
                    # Normalize velocity from [0, 127] to [0, 1]
                    velocity_values = velocity_values / 127.0
                reduced[:, col] = np.maximum(reduced[:, col], velocity_values)
        matrices.append(reduced)
    return np.maximum.reduce(matrices) if len(matrices) > 1 else matrices[0]


def create_sequences(matrix, seq_len=None):
    """
    Slice pianoroll into input/output sequences for RNN training.
    Input: (seq_len × num_instruments)
    Target: next timestep (seq_len × num_instruments) or autoregressive
    """
    if seq_len is None:
        seq_len = STEPS_PER_BAR
    X, y = [], []
    for start in range(0, len(matrix) - seq_len):
        X.append(matrix[start:start+seq_len])
        y.append(matrix[start+1:start+seq_len+1])
    return np.array(X), np.array(y)


def create_bars(matrix, bar_len=None):
    """
    Slice pianoroll into non-overlapping bars.
    Input: matrix (timesteps × num_instruments)
    Output: X = bar N, y = bar N+1
    """
    if bar_len is None:
        bar_len = STEPS_PER_BAR
    num_bars = len(matrix) // bar_len
    X, y = [], []
    for b in range(num_bars - 1):
        X.append(matrix[b*bar_len:(b+1)*bar_len])
        y.append(matrix[(b+1)*bar_len:(b+2)*bar_len])
    return np.array(X), np.array(y)





# Example usage
if __name__ == "__main__":
    # Suppose you already built pianoroll dict "aaa" from Read_midi
    abs_path = os.path.abspath(__file__)
    dir_path = os.path.dirname(abs_path)
    filepath = os.path.join(dir_path, "1_funk-groove1_138_beat_4-4_2.midi")
    aaa = Read_midi(filepath, quantization=QUANTIZATION).read_file()

    reduced_matrix = pianoroll_to_matrix(aaa)
    print("Reduced matrix shape:", reduced_matrix.shape)  # (timesteps × num_instruments)

    # X, y = create_sequences(reduced_matrix)
    X, y = create_bars(reduced_matrix)
    print("X shape:", X.shape)  # (num_sequences × 16 × num_instruments)
    print("y shape:", y.shape)
    import matplotlib.pyplot as plt

    # pick one sequence
    sample_idx = 0
    X_sample = X[sample_idx]  # shape (16, 9)

    plt.figure(figsize=(10,4))
    plt.imshow(X_sample.T, aspect="auto", cmap="viridis", origin="lower", vmin=0, vmax=1)
    plt.yticks(range(len(INSTRUMENTS)), INSTRUMENTS)
    plt.xlabel("Time step")
    plt.ylabel("Instrument")
    plt.title(f"Drum pattern ({STEPS_PER_BAR} steps)")
    plt.colorbar(label="Velocity (0=silent, 1=max)")
    plt.show()

