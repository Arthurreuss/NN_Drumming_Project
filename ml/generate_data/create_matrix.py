import numpy as np
from ml.generate_data.read_midi import Read_midi
import os
import matplotlib.pyplot as plt


class CreateMatrix:
    """
    A class for processing MIDI drum data into matrices for machine learning.
    """
    
    def __init__(self, quantization=8, beats_per_bar=4, normalize_velocity=True, verbose=False):
        """
        Initialize the CreateMatrix class.
        
        Args:
            quantization (int): Steps per beat (4=16th note, 8=32nd note, 2=8th note)
            beats_per_bar (int): Number of beats per bar (default 4 for 4/4 time)
            normalize_velocity (bool): Whether to normalize velocity values to [0,1]
            verbose (bool): Whether to print debug information
        """
        # Map MIDI pitches to simplified drum classes
        self.DRUM_MAP = {
            36: "Kick", 35: "Kick",
            38: "Snare", 40: "Snare",
            42: "HiHatClosed", 44: "HiHatPedal", 46: "HiHatOpen",
            49: "Crash", 57: "Crash",
            51: "Ride", 53: "Ride", 59: "Ride",
            45: "LowTom", 47: "LowTom",
            48: "HighTom", 50: "HighTom"
        }
        
        self.INSTRUMENTS = sorted(set(self.DRUM_MAP.values()))
        self.INST_TO_IDX = {inst: i for i, inst in enumerate(self.INSTRUMENTS)}
        
        self.quantization = quantization
        self.beats_per_bar = beats_per_bar
        self.steps_per_bar = quantization * beats_per_bar
        self.normalize_velocity = normalize_velocity
        self.verbose = verbose
    
    def _print(self, *args, **kwargs):
        """Print only if verbose mode is enabled."""
        if self.verbose:
            print(*args, **kwargs)
    
    def pianoroll_to_matrix(self, pianoroll_dict):
        """Convert pianoroll dictionary to reduced matrix representation."""
        matrices = []
        for pr in pianoroll_dict.values():
            T, _ = pr.shape
            reduced = np.zeros((T, len(self.INSTRUMENTS)), dtype=np.float16)
            for pitch, inst in self.DRUM_MAP.items():
                col = self.INST_TO_IDX[inst]
                if pitch < pr.shape[1]:
                    velocity_values = pr[:, pitch].astype(np.float16)
                    if self.normalize_velocity:
                        # Normalize velocity from [0, 127] to [0, 1]
                        velocity_values = velocity_values / 127.0
                    reduced[:, col] = np.maximum(reduced[:, col], velocity_values)
            matrices.append(reduced)
        return np.maximum.reduce(matrices) if len(matrices) > 1 else matrices[0]

    def create_sequences(self, matrix, seq_len=None):
        """
        Slice pianoroll into input/output sequences for RNN training.
        Input: (seq_len × num_instruments)
        Target: next timestep (seq_len × num_instruments) or autoregressive
        """
        if seq_len is None:
            seq_len = self.steps_per_bar
        X, y = [], []
        for start in range(0, len(matrix) - seq_len):
            X.append(matrix[start:start+seq_len])
            y.append(matrix[start+1:start+seq_len+1])
        return np.array(X), np.array(y)

    def create_bars(self, matrix, bar_len=None):
        """
        Slice pianoroll into non-overlapping bars.
        Input: matrix (timesteps × num_instruments)
        Output: X = bar N, y = bar N+1
        """
        if bar_len is None:
            bar_len = self.steps_per_bar
        num_bars = len(matrix) // bar_len
        X, y = [], []
        for b in range(num_bars - 1):
            X.append(matrix[b*bar_len:(b+1)*bar_len])
            y.append(matrix[(b+1)*bar_len:(b+2)*bar_len])
        return np.array(X), np.array(y)
    
    def get_all_bars(self, matrix, bar_len=None):
        """
        Slice pianoroll into all non-overlapping bars.
        Input: matrix (timesteps × num_instruments)
        Output: All bars in the MIDI file
        """
        if bar_len is None:
            bar_len = self.steps_per_bar
        num_bars = len(matrix) // bar_len
        bars = []
        for b in range(num_bars):
            bars.append(matrix[b*bar_len:(b+1)*bar_len])
        return np.array(bars)
    
    def plot_matrix(self, matrix, sample_idx=0, title=None):
        """
        Plot a drum pattern matrix.
        
        Args:
            matrix (np.array): Input matrix with shape (num_samples, timesteps, instruments)
            sample_idx (int): Index of the sample to plot
            title (str): Custom title for the plot
        """
        if len(matrix.shape) == 3:
            # If 3D array, select one sample
            sample = matrix[sample_idx]
        else:
            # If 2D array, use as is
            sample = matrix
        
        plt.figure(figsize=(10, 4))
        plt.imshow(sample.T, aspect="auto", cmap="viridis", origin="lower", vmin=0, vmax=1)
        plt.yticks(range(len(self.INSTRUMENTS)), self.INSTRUMENTS)
        plt.xlabel("Time step")
        plt.ylabel("Instrument")
        
        if title is None:
            title = f"Drum pattern ({self.steps_per_bar} steps per bar)"
        plt.title(title)
        plt.colorbar(label="Velocity (0=silent, 1=max)")
        plt.show()
    
    def run(self, filepath, bar_len=None, enable_plot=False, sample_idx=0):
        """
        Main method to process MIDI file and return all bars.
        
        Args:
            filepath (str): Path to MIDI file
            bar_len (int): Bar length (optional, defaults to steps_per_bar)
            enable_plot (bool): Whether to plot the first bar
            sample_idx (int): Index of bar to plot if enable_plot is True
        
        Returns:
            list: List of all bars in the MIDI file, where each bar is a numpy array
                  with shape (steps_per_bar, num_instruments)
        """
        # Read MIDI file
        midi_reader = Read_midi(filepath, quantization=self.quantization)
        pianoroll_dict = midi_reader.read_file()
        
        # Convert to reduced matrix
        reduced_matrix = self.pianoroll_to_matrix(pianoroll_dict)
        self._print(f"Reduced matrix shape: {reduced_matrix.shape}")
        
        # Get all bars
        all_bars = self.get_all_bars(reduced_matrix, bar_len)
        self._print(f"Number of bars found: {len(all_bars)}")
        self._print(f"Each bar shape: {all_bars[0].shape if len(all_bars) > 0 else 'No bars found'}")
        
        # Plot if requested
        if enable_plot and len(all_bars) > sample_idx:
            self.plot_matrix(all_bars, sample_idx, f"Bar {sample_idx} - Drum Pattern")
        
        # Convert to list of individual bar matrices
        return [bar for bar in all_bars]




    
