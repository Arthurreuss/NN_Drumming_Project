import csv
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ml.data.dataset import DrumDataset
from ml.data.midi import Midi
from ml.data.preprocess import DrumPreprocessor
from ml.data.tokenizer import BeatTokenizer
from ml.evaluation.eval import evaluate_model
from ml.models.lstm import Seq2SeqLSTM
from ml.training.train import train
from scripts.plotting_inference import plot_drum_matrix


class Pipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset_cfg = self.cfg["dataset_creation"]
        self.pipeline_cfg = self.cfg["pipeline"]
        self.training_cfg = self.cfg["training"]
        self.model_cfg = self.cfg["model"]

        self.dataset_path = (
            Path(self.dataset_cfg["preprocessed_data_dir"])
            / f"q_{self.pipeline_cfg['quantization']}"
            / f"seg_{self.pipeline_cfg['segment_len']}"
        )
        self.checkpoint_path = (
            Path("checkpoints")
            / f"seg_{self.pipeline_cfg['segment_len']}"
            / self.pipeline_cfg["model"]
        )

        self.midi_reader = Midi(self.pipeline_cfg["quantization"])
        self.tokenizer = BeatTokenizer(
            self.cfg, path=f"{self.dataset_path}/beat_tokenizer.npy"
        )
        self.preprocessor = DrumPreprocessor(self.cfg, self.midi_reader, self.tokenizer)
        self.train_set = None
        self.val_set = None
        self.test_set = None
        self.model = None
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )

    def _build_model(self, model_name):
        if model_name in ["small", "medium", "large"]:
            self.model = Seq2SeqLSTM(
                vocab_size=len(self.tokenizer) + 1,
                num_genres=len(self.dataset_cfg["genres"]),
                token_embed_dim=self.model_cfg[model_name]["token_embed_dim"],
                pos_embed_dim=self.model_cfg[model_name]["pos_embed_dim"],
                genre_embed_dim=self.model_cfg[model_name]["genre_embed_dim"],
                bpm_embed_dim=self.model_cfg[model_name]["bpm_embed_dim"],
                hidden=self.model_cfg[model_name]["hidden_dim"],
                layers=self.model_cfg[model_name]["num_layers"],
                period=self.pipeline_cfg["inference"]["period"],
                dropout=self.model_cfg[model_name]["dropout"],
            ).to(self.device)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

    def _load_model(self):
        ckpt_path = self.checkpoint_path / "seq2seq_best.pt"
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"], strict=False)
        self.model.eval()
        logging.info(
            f"[Model] Loaded checkpoint from {ckpt_path}, epoch {ckpt['epoch']}"
        )

    def run(self):
        # load or preprocess dataset
        if self.pipeline_cfg["create_dataset"]:
            self.train_set, self.val_set, self.test_set, self.tokenizer = (
                self.preprocessor.preprocess_dataset()
            )

        else:
            try:
                self.train_set = DrumDataset(
                    self.cfg, self.dataset_path / "train", include_genre=True
                )
                self.val_set = DrumDataset(
                    self.cfg, self.dataset_path / "val", include_genre=True
                )
                self.test_set = DrumDataset(
                    self.cfg, self.dataset_path / "test", include_genre=True
                )
            except Exception as e:
                logging.info(f"[Pipeline] Error loading datasets: {e}")
                return

        # build model
        self._build_model(self.pipeline_cfg["model"])

        # train
        if self.pipeline_cfg["train_model"]:
            self.model = train(
                self.cfg,
                self.model,
                self.device,
                self.train_set,
                self.val_set,
                self.tokenizer,
                self.checkpoint_path,
            )

        # inference
        if self.pipeline_cfg["inference"]["enabled"]:
            self._load_model()

            genre_indices = [
                i
                for i, s in enumerate(self.val_set)
                if self.val_set.genres[s["genre_id"].item()]
                == self.pipeline_cfg["inference"]["genre"]
            ]

            if not genre_indices:
                raise ValueError(
                    f"No samples found for genre '{self.pipeline_cfg['inference']['genre']}'"
                )

            # Pick one random sample
            idx = random.choice(genre_indices)
            sample = self.val_set[idx]
            logging.info(
                f"[Inference] Picked sample {idx} ({self.pipeline_cfg['inference']['genre']})"
            )

            prim_tok = sample["tokens"].unsqueeze(0).to(self.device)
            prim_pos = sample["positions"].unsqueeze(0).to(self.device)
            genre_id = sample["genre_id"].unsqueeze(0).to(self.device)
            bpm = torch.tensor(
                sample["bpm"], dtype=torch.float32, device=self.device
            ).unsqueeze(0)

            gen_tokens = self.model.generate(
                prim_tok,
                prim_pos,
                genre_id,
                bpm=bpm,
                unk_id=self.tokenizer.unk_id,
                steps=self.pipeline_cfg["inference"]["generation_length"],
                temperature=self.pipeline_cfg["inference"]["temperature"],
            )

            gen_tokens = gen_tokens.squeeze(0).cpu().numpy()
            detok = []
            for t in gen_tokens:
                vec = self.tokenizer.detokenize(int(t))
                detok.append(vec)
            matrix = np.concat(detok, axis=0)  # (T, 9)

            if self.pipeline_cfg["inference"]["plot"]:
                genre_idx = int(genre_id.squeeze().cpu().item())
                plot_drum_matrix(
                    matrix,
                    title=f"Generated Drum Pattern (Genre: {self.val_set.genres[genre_idx]})",
                    save_path=(
                        f"outputs/{self.pipeline_cfg['inference']['genre']}_{idx}.png"
                    ),
                )
            if self.pipeline_cfg["inference"]["create_midi"]:
                os.makedirs("outputs", exist_ok=True)
                self.midi_reader.create_midi(
                    self.cfg,
                    matrix,
                    output_path=(
                        f"outputs/{self.pipeline_cfg['inference']['genre']}_{idx}.mid"
                    ),
                    tempo=int(bpm),
                )

        if self.pipeline_cfg["evaluate"]:
            self._load_model()
            test_loader = DataLoader(
                self.test_set,
                batch_size=self.training_cfg["batch_size"],
                shuffle=False,
                num_workers=8,
            )
            logging.info("[Eval] Running evaluation on test set...")
            metrics = evaluate_model(
                self.model, test_loader, self.device, self.tokenizer
            )

            logging.info("=== Evaluation Results (Full Test Set) ===")
            for k, v in metrics.items():
                logging.info(f"{k:20s}: {v:.4f}")

            os.makedirs(self.checkpoint_path, exist_ok=True)
            csv_path = os.path.join(self.checkpoint_path, "test_eval_metrics.csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["metric", "value"])
                for k, v in metrics.items():
                    writer.writerow([k, v])

            logging.info(f"\n[Eval] Results written to {csv_path}")
