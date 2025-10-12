import numpy as np
import torch

from ml.data.dataset import DrumDataset
from ml.data.midi import Midi
from ml.data.preprocess import DrumPreprocessor
from ml.data.tokenizer import SimpleTokenizer
from ml.models.lstm import Seq2SeqLSTM
from ml.models.plotting_inference import plot_drum_matrix
from ml.training.train import train
from ml.utils.cfg import load_config


class Pipeline:
    def __init__(self):
        self.cfg = load_config()
        self.dataset_cfg = self.cfg["dataset"]
        self.pipeline_cfg = self.cfg["pipeline"]
        self.training_cfg = self.cfg["training"]
        self.model_cfg = self.cfg["model"]
        self.midi_reader = Midi(self.dataset_cfg["quantization"])
        self.tokenizer = SimpleTokenizer()  # TODO: if we have multiple write funciton
        self.preprocessor = DrumPreprocessor(self.midi_reader, self.tokenizer)
        self.train_set = None
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
                num_pos=self.dataset_cfg["segment_length"],
                num_genres=len(self.train_set.genres),
                token_embed_dim=self.model_cfg[model_name]["token_embed_dim"],
                pos_embed_dim=self.model_cfg[model_name]["pos_embed_dim"],
                genre_embed_dim=self.model_cfg[model_name]["genre_embed_dim"],
                hidden=self.model_cfg[model_name]["hidden_dim"],
                layers=self.model_cfg[model_name]["num_layers"],
                dropout=self.model_cfg[model_name]["dropout"],
            ).to(self.device)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

    def run(self):
        # load or preprocess dataset
        if self.pipeline_cfg["preprocess"]:
            self.trainset, self.testset, self.tokenizer = (
                self.preprocessor.preprocess_dataset()
            )
        else:
            try:
                self.train_set = DrumDataset(
                    self.training_cfg["train_dir"], include_genre=True
                )
                self.test_set = DrumDataset(
                    self.training_cfg["test_dir"], include_genre=True
                )
            except Exception as e:
                print(f"[Pipeline] Error loading datasets: {e}")
                return

        # build model
        self._build_model()

        # train
        if self.pipeline_cfg["train"]:
            self.model = train(self.model, self.device, self.train_set, self.test_set)

        # inference
        if self.pipeline_cfg["inference"]["enabled"]:
            # load model
            ckpt_path = "checkpoints/seq2seq_best.pt"
            ckpt = torch.load(ckpt_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state"], strict=False)
            self.model.eval()

            sample = self.test_set[self.pipeline_cfg["inference"]["starting_sample"]]
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
                    title=f"Generated Drum Pattern (Genre: {self.test_set.genres[genre_idx]})",
                    save_path=(
                        self.pipeline_cfg["inference"]["save_plot_filename"]
                        if self.pipeline_cfg["inference"]["save_plot_filename"]
                        else None
                    ),
                )
            if self.pipeline_cfg["inference"]["create_midi"]:
                midi = self.midi_reader.create_midi(
                    matrix,
                    output_path=(
                        self.pipeline_cfg["inference"]["save_midi_filename"]
                        if self.pipeline_cfg["inference"]["save_midi_filename"]
                        else "generated.mid"
                    ),
                )
                midi.save()
