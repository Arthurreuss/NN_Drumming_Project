import numpy as np
from experimental.read_midi import Read_midi

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

def pianoroll_to_matrix(pianoroll_dict):
    matrices = []
    for pr in pianoroll_dict.values():
        T, _ = pr.shape
        reduced = np.zeros((T, len(INSTRUMENTS)), dtype=np.float32)
        for pitch, inst in DRUM_MAP.items():
            col = INST_TO_IDX[inst]
            if pitch < pr.shape[1]:
                reduced[:, col] = np.maximum(reduced[:, col],
                                             (pr[:, pitch] > 0).astype(np.float32))
        matrices.append(reduced)
    return np.maximum.reduce(matrices) if len(matrices) > 1 else matrices[0]


def create_sequences(matrix, seq_len=16):
    """
    Slice pianoroll into input/output sequences for RNN training.
    Input: (seq_len × num_instruments)
    Target: next timestep (seq_len × num_instruments) or autoregressive
    """
    X, y = [], []
    for start in range(0, len(matrix) - seq_len):
        X.append(matrix[start:start+seq_len])
        y.append(matrix[start+1:start+seq_len+1])
    return np.array(X), np.array(y)


def create_bars(matrix, bar_len=16):
    """
    Slice pianoroll into non-overlapping bars.
    Input: matrix (timesteps × num_instruments)
    Output: X = bar N, y = bar N+1
    """
    num_bars = len(matrix) // bar_len
    X, y = [], []
    for b in range(num_bars - 1):
        X.append(matrix[b*bar_len:(b+1)*bar_len])
        y.append(matrix[(b+1)*bar_len:(b+2)*bar_len])
    return np.array(X), np.array(y)


# Example usage
if __name__ == "__main__":
    # Suppose you already built pianoroll dict "aaa" from Read_midi
    filepath = "/Users/arthurreuss/Library/Mobile Documents/iCloud~md~obsidian/Documents/Brain Extension/Bachelor AI/Neural Networks/Project/NN_Drumming_Project/datasets/e-gmd-v1.0.0/drummer1/eval_session/1_funk-groove1_138_beat_4-4_1.midi"
    aaa = Read_midi(filepath, quantization=4).read_file()

    reduced_matrix = pianoroll_to_matrix(aaa)
    print("Reduced matrix shape:", reduced_matrix.shape)  # (timesteps × num_instruments)

    # X, y = create_sequences(reduced_matrix, seq_len=16)
    X, y = create_bars(reduced_matrix, bar_len=16)
    print("X shape:", X.shape)  # (num_sequences × 16 × num_instruments)
    print("y shape:", y.shape)
    import matplotlib.pyplot as plt

    # pick one sequence
    sample_idx = 0
    X_sample = X[sample_idx]  # shape (16, 9)

    plt.figure(figsize=(10,4))
    plt.imshow(X_sample.T, aspect="auto", cmap="Greys", origin="lower")
    plt.yticks(range(len(INSTRUMENTS)), INSTRUMENTS)
    plt.xlabel("Time step")
    plt.ylabel("Instrument")
    plt.title("Drum pattern (16 steps)")
    plt.colorbar(label="Active (0/1 or velocity)")
    plt.show()

