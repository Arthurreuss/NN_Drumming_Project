import csv
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from ml.data.dataset import DrumDataset
from ml.data.midi import Midi
from ml.data.preprocess import DrumPreprocessor
from ml.data.tokenizer import SimpleTokenizer
from ml.models.lstm import Seq2SeqLSTM
from ml.training.train import train
from ml.utils.cfg import load_config
from ml.utils.eval import evaluate_model
from scripts.plotting_inference import plot_drum_matrix


class Pipeline:
    def __init__(self):
        self.cfg = load_config()
        self.dataset_cfg = self.cfg["dataset"]
        self.pipeline_cfg = self.cfg["pipeline"]
        self.training_cfg = self.cfg["training"]
        self.model_cfg = self.cfg["model"]
        self.midi_reader = Midi(self.dataset_cfg["quantization"])
        self.tokenizer = SimpleTokenizer()
        self.preprocessor = DrumPreprocessor(self.midi_reader, self.tokenizer)
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
                num_pos=self.dataset_cfg["segment_len"],
                num_genres=len(self.dataset_cfg["genres"]),
                token_embed_dim=self.model_cfg[model_name]["token_embed_dim"],
                pos_embed_dim=self.model_cfg[model_name]["pos_embed_dim"],
                genre_embed_dim=self.model_cfg[model_name]["genre_embed_dim"],
                hidden=self.model_cfg[model_name]["hidden_dim"],
                layers=self.model_cfg[model_name]["num_layers"],
                dropout=self.model_cfg[model_name]["dropout"],
            ).to(self.device)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

    def _load_model(self):
        ckpt_path = f"checkpoints/seg_{self.pipeline_cfg['train']['segment_len']}/{self.pipeline_cfg['model']}/seq2seq_best.pt"
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"], strict=False)
        self.model.eval()
        print(f"[Model] Loaded checkpoint from {ckpt_path}, epoch {ckpt['epoch']}")

    def run(self):
        # load or preprocess dataset
        if self.pipeline_cfg["data"]["preprocess"]:
            self.trainset, self.val_set, self.testset, self.tokenizer = (
                self.preprocessor.preprocess_dataset()
            )
        else:
            try:
                self.train_set = DrumDataset(
                    self.pipeline_cfg["data"]["train_dir"], include_genre=True
                )
                self.val_set = DrumDataset(
                    self.pipeline_cfg["data"]["val_dir"], include_genre=True
                )
                self.test_set = DrumDataset(
                    self.pipeline_cfg["data"]["test_dir"], include_genre=True
                )
            except Exception as e:
                print(f"[Pipeline] Error loading datasets: {e}")
                return

        # build model
        self._build_model(self.pipeline_cfg["model"])

        # train
        if self.pipeline_cfg["train"]["enabled"]:
            self.model = train(
                self.model, self.device, self.train_set, self.test_set, self.tokenizer
            )

        # inference
        if self.pipeline_cfg["inference"]["enabled"]:
            self._load_model()

            sample = self.val_set[self.pipeline_cfg["inference"]["starting_sample"]]
            prim_tok = sample["tokens"].unsqueeze(0).to(self.device)
            prim_pos = sample["positions"].unsqueeze(0).to(self.device)
            genre_id = sample["genre_id"].unsqueeze(0).to(self.device)

            gen_tokens = self.model.generate(
                prim_tok,
                prim_pos,
                genre_id,
                steps=self.pipeline_cfg["inference"]["generation_length"],
                temperature=self.pipeline_cfg["inference"]["temperature"],
            )

            gen_tokens = gen_tokens.squeeze(0).cpu().numpy()
            detok = []
            for t in gen_tokens:
                vec = self.tokenizer.detokenize(int(t))
                detok.append(vec)
            matrix = np.stack(detok, axis=0)  # (T, 9)

            if self.pipeline_cfg["inference"]["plot"]:
                genre_idx = int(genre_id.squeeze().cpu().item())
                plot_drum_matrix(
                    matrix,
                    title=f"Generated Drum Pattern (Genre: {self.val_set.genres[genre_idx]})",
                    save_path=(
                        self.pipeline_cfg["inference"]["save_plot_filename"]
                        if self.pipeline_cfg["inference"]["save_plot_filename"]
                        else None
                    ),
                )
            if self.pipeline_cfg["inference"]["create_midi"]:
                os.makedirs("outputs", exist_ok=True)
                midi = self.midi_reader.create_midi(
                    matrix,
                    output_path=(
                        self.pipeline_cfg["inference"]["save_midi_filename"]
                        if self.pipeline_cfg["inference"]["save_midi_filename"]
                        else "generated.mid"
                    ),
                )

        if self.pipeline_cfg["evaluate"]["enabled"]:
            self._load_model()
            test_loader = DataLoader(
                self.test_set,
                batch_size=self.pipeline_cfg["evaluate"]["batch_size"],
                shuffle=False,
                num_workers=2,
            )
            print("[Eval] Running evaluation on test set...")
            metrics = evaluate_model(
                self.model, test_loader, self.device, self.tokenizer
            )

            print("\n=== Evaluation Results (Full Test Set) ===")
            for k, v in metrics.items():
                print(f"{k:20s}: {v:.4f}")

            out_dir = f"checkpoints/seg_{self.pipeline_cfg['train']['segment_len']}/{self.pipeline_cfg['model']}"
            os.makedirs(out_dir, exist_ok=True)
            csv_path = os.path.join(out_dir, "test_eval_metrics.csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["metric", "value"])
                for k, v in metrics.items():
                    writer.writerow([k, v])

            print(f"\n[Saved] Results written to {csv_path}")
