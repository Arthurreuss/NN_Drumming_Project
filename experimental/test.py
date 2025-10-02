#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import mido
from mido import MidiFile

from experimental.create_dataset import pianoroll_to_matrix, create_bars, INSTRUMENTS
from experimental.read_midi import Read_midi
from experimental.rnn import DrumRNN

# --------------------------------------------------
# Generation
# --------------------------------------------------
def generate_bars(model, seed_bar, num_bars=10, threshold=0.2):
    model.eval()
    bars = [seed_bar]
    current = torch.tensor(seed_bar, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        for _ in range(num_bars - 1):
            pred = model(current)
            pred = torch.sigmoid(pred)
            binary = (pred[0].numpy() > threshold).astype(np.int16) * 100
            bars.append(binary)
            current = torch.tensor(binary, dtype=torch.float32).unsqueeze(0)

    return np.vstack(bars)  # (num_bars*bar_len, num_instruments)

# --------------------------------------------------
# Map reduced instruments back to MIDI pitches
# --------------------------------------------------
INSTRUMENT_TO_PITCH = {
    "Kick": 36,
    "Snare": 38,
    "HiHatClosed": 42,
    "HiHatOpen": 46,
    "Crash": 49,
    "Ride": 51,
    "LowTom": 45,
    "HighTom": 48,
}

def bars_to_pianoroll(matrix, instrument_names):
    T, num_instruments = matrix.shape
    pr = np.zeros((T, 128), dtype=np.int16)
    for i, inst in enumerate(instrument_names):
        if inst in INSTRUMENT_TO_PITCH:
            pitch = INSTRUMENT_TO_PITCH[inst]
            pr[:, pitch] = matrix[:, i]
    return {"Drums": pr}

# --------------------------------------------------
# MIDI writing
# --------------------------------------------------
import mido
from mido import MidiFile

def write_midi(pr, ticks_per_beat, write_path, tempo=100, steps_per_bar=16):
    """
    pr: dictionary {track_name: pianoroll_matrix} where matrix = (timesteps × 128 pitches)
    ticks_per_beat: resolution, usually 480
    write_path: output .mid file
    tempo: BPM
    steps_per_bar: how many discrete steps per bar (e.g. 16)
    """
    microseconds_per_beat = mido.bpm2tempo(tempo)
    mid = MidiFile()
    mid.ticks_per_beat = ticks_per_beat

    # Each bar = 4 beats → ticks_per_beat*4 ticks
    # Each step = that / steps_per_bar
    ticks_per_step = int((ticks_per_beat * 4) / steps_per_bar)

    for instrument_name, matrix in pr.items():
        track = mid.add_track(instrument_name)
        track.append(mido.MetaMessage('set_tempo', tempo=microseconds_per_beat))
        channel = 9  # General MIDI percussion channel

        T, N = matrix.shape
        pr_tm1 = np.zeros(N)
        t_last = 0

        for t in range(T):
            pr_t = matrix[t]
            mask = (pr_t != pr_tm1)
            if mask.any():
                for n in range(N):
                    if mask[n]:
                        pitch = n
                        velocity = int(pr_t[n])
                        # convert step difference to ticks
                        t_event = (t - t_last) * ticks_per_step
                        t_last = t
                        if velocity == 0:
                            track.append(mido.Message('note_off', note=pitch, velocity=0,
                                                      time=t_event, channel=channel))
                        else:
                            track.append(mido.Message('note_on', note=pitch, velocity=100,  # uniform velocity
                                                      time=t_event, channel=channel))
            pr_tm1 = pr_t

    mid.save(write_path)

import matplotlib.pyplot as plt
import numpy as np

def plot_generated(generated, instrument_names, bar_len=16):
    """
    generated: numpy array (num_bars*bar_len, num_instruments)
    instrument_names: list of instrument labels (len=num_instruments)
    bar_len: steps per bar (default=16)
    """
    T, num_instruments = generated.shape
    num_bars = T // bar_len

    plt.figure(figsize=(12, 6))
    plt.imshow(generated.T, aspect="auto", cmap="Greys", origin="lower")

    # Y-axis labels = instruments
    plt.yticks(range(num_instruments), instrument_names)
    plt.xlabel("Time steps")
    plt.ylabel("Instrument")
    plt.title(f"Generated Drum Pattern ({num_bars} bars, {bar_len} steps/bar)")

    # Vertical lines at bar boundaries
    for b in range(1, num_bars):
        plt.axvline(b * bar_len - 0.5, color="red", linestyle="--", alpha=0.5)

    plt.colorbar(label="Velocity (0–127)")
    plt.show()

# Example usage

# --------------------------------------------------
# Main script
# --------------------------------------------------
if __name__ == "__main__":
    filepath = "/Users/arthurreuss/Library/Mobile Documents/iCloud~md~obsidian/Documents/Brain Extension/Bachelor AI/Neural Networks/Project/NN_Drumming_Project/datasets/e-gmd-v1.0.0/drummer1/eval_session/1_funk-groove1_138_beat_4-4_1.midi"

    aaa = Read_midi(filepath, quantization=4).read_file()

    reduced_matrix = pianoroll_to_matrix(aaa)
    print("Reduced matrix shape:", reduced_matrix.shape)

    X, y = create_bars(reduced_matrix, bar_len=16)
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # --- Dataset ---
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

    # --- Model ---
    input_size = X.shape[2]
    hidden_size = 64
    output_size = input_size
    bar_len = X.shape[1]

    model = DrumRNN(input_size, hidden_size, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # --- Training ---
    for epoch in range(50):
        total_loss = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: loss={total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), "drum_rnn.pth")

    # --- Test on first bar ---
    with torch.no_grad():
        sample_in = X_tensor[0].unsqueeze(0)
        pred = model(sample_in)
        print("Prediction shape:", pred.shape)
        print("First timestep prediction:", pred[0, 0])

    # --- Generate 10 bars ---
    seed_bar = X[0]
    generated = generate_bars(model, seed_bar, num_bars=10, threshold=0.5)
    print("Generated sequence shape:", generated.shape)

    # --- Convert to MIDI ---
    pianoroll_dict = bars_to_pianoroll(generated, INSTRUMENTS)
    write_midi(pianoroll_dict, ticks_per_beat=480, write_path="generated_drums2.mid", tempo=100)
    plot_generated(generated, INSTRUMENTS, bar_len=16)
    
    

