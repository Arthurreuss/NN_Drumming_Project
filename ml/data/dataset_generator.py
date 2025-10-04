import os

import pandas as pd
from ml.data.create_matrix import CreateMatrix


class DatasetGenerator:
    def __init__(
        self, verbose=True, dataset_path="datasets/e-gmd-v1.0.0", dataset_config=None
    ):
        if not dataset_config:
            raise ValueError("dataset_config must be provided")

        self.beats_per_bar = dataset_config["beats_per_bar"]
        self.quantization = dataset_config["quantization"]
        self.normalize_velocity = dataset_config["normalize_velocity"]
        print(f"Dataset config: {dataset_config}")

        self.verbose = verbose
        if not os.path.exists(dataset_path):
            raise ValueError(f"Dataset path {dataset_path} does not exist.")

        self.dataset_path = dataset_path
        self.data_csv = pd.read_csv(os.path.join(dataset_path, "e-gmd-v1.0.0.csv"))
        self.midi_file_names = self.data_csv["midi_filename"].tolist()

    def _print(self, message):
        if self.verbose:
            print(message)

    def create_samples(self):
        samples = []
        for file_name in self.midi_file_names[:3]:
            if not os.path.exists(os.path.join(self.dataset_path, file_name)):
                self._print(f"Warning: {file_name} does not exist in dataset path.")
                continue
            data_point = self.data_csv[
                self.data_csv["midi_filename"] == file_name
            ].iloc[0]
            bpm = data_point["bpm"]
            beat_type = data_point["beat_type"]
            style = data_point["style"]
            file_path = os.path.join(self.dataset_path, file_name)
            matrix_gen = CreateMatrix(
                quantization=self.quantization,
                beats_per_bar=self.beats_per_bar,
                normalize_velocity=self.normalize_velocity,
            )
            matrixes = matrix_gen.run(file_path)
            samples.append(
                {
                    "file_name": file_name,
                    "bpm": bpm,
                    "beat_type": beat_type,
                    "style": style,
                    "matrixes": matrixes,
                }
            )
            print(f"Progress: {len(samples)}/{len(self.midi_file_names)}", end="\r")
        self.samples = samples

    def save_samples(
        self, save_path="datasets/preprocessed_datasets/processed_samples.csv"
    ):
        # check if dir exists else create it
        dir_name = os.path.dirname(save_path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        if not hasattr(self, "samples"):
            raise ValueError("No samples to save. Please run create_samples() first.")
        df = pd.DataFrame(self.samples)
        df.to_csv(save_path, index=False)
        self._print(f"Samples saved to {save_path}")
