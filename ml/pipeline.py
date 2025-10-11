from re import S

import numpy as np
import torch

from ml.data.dataset import DrumDataset
from ml.data.preprocess import preprocess_dataset
from ml.data.tokenizer import SimpleTokenizer
from ml.models.lstm import Seq2SeqLSTM
from ml.models.model import Model
from ml.training.train import train
from utils.cfg import load_config
from utils.plotting_inference import plot_drum_matrix, show_interactive_predictions


class Pipeline:
    def __init__(self):
        self.cfg = load_config()
        self.dataset_cfg = self.cfg["dataset"]
        self.pipeline_cfg = self.cfg["pipeline"]
        self.training_cfg = self.cfg["training"]
        self.tokenizer = SimpleTokenizer()
        self.train_set = None
        self.test_set = None
        self.model = None
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )

    def build_model(self) -> Model:
        self.model = Seq2SeqLSTM(
            vocab_size=len(self.tokenizer) + 1,
            pos_vocab_size=self.dataset_cfg["quantization"]
            + 1,  # maybe seg length instead
            num_genres=len(self.train_set.genres),
            token_embed_dim=128,
            pos_embed_dim=8,
            genre_embed_dim=8,
            hidden=256,
            layers=2,
            dropout=0.1,
        ).to(self.device)

    def run(self):
        if self.pipeline_cfg["preprocess"]:
            self.trainset, self.testset, self.tokenizer = preprocess_dataset()
        else:
            self.train_set = DrumDataset(
                self.training_cfg["train_dir"], include_genre=True
            )
            self.test_set = DrumDataset(
                self.training_cfg["test_dir"], include_genre=True
            )

        self.build_model()
        if self.pipeline_cfg["train"]:
            self.model = train(self.model)
        if self.pipeline_cfg["inference"]["enabled"]:
            ckpt_path = "checkpoints/seq2seq_best.pt"
            ckpt = torch.load(ckpt_path, map_location=self.device)

            self.model.load_state_dict(ckpt["model_state"], strict=False)
            self.model.eval()

            sample = self.test_set[0]
            prim_tok = sample["tokens"].unsqueeze(0).to(self.device)
            prim_pos = sample["positions"].unsqueeze(0).to(self.device)
            genre_id = sample["genre_id"].unsqueeze(0).to(self.device)

            # --- Generate sequence ---
            gen_tokens = self.model.generate(
                prim_tok,
                prim_pos,
                genre_id,
                steps=self.pipeline_cfg["inference"]["generation_length"],
                temperature=self.pipeline_cfg["inference"]["temperature"],
            )

            if self.pipeline_cfg["inference"]["plot"]:
                gen_tokens = gen_tokens.squeeze(0).cpu().numpy()

                # --- Detokenize into drum matrix ---
                detok = []
                for t in gen_tokens:
                    try:
                        vec = self.tokenizer.detokenize(int(t))
                    except KeyError:
                        vec = np.zeros(9)  # fallback if unseen token
                    detok.append(vec)

                matrix = np.stack(detok, axis=0)  # (T, 9)

                # --- Plot ---
                genre_idx = int(genre_id.squeeze().cpu().item())

                show_interactive_predictions(
                    self.model,
                    self.test_set,
                    self.tokenizer,
                    self.device,
                    steps=self.pipeline_cfg["inference"]["generation_length"],
                    temperature=self.pipeline_cfg["inference"]["temperature"],
                )
