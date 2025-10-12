import os
import sys

from ml.models.model import Model


class ModelManager:
    def __init__(self, model_dir: str = "models/"):
        self.model_dir = model_dir
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        self.current_models = self._list_models()

    def _list_models(self):
        return [
            f
            for f in os.listdir(self.model_dir)
            if os.path.isfile(os.path.join(self.model_dir, f))
        ]

    def save(self, model: Model, model_name: str):
        model_path = os.path.join(self.model_dir, model_name)
        model.save(model_path)
        self.current_models.append(model_name)

    def load(self, model_name: str) -> Model:
        pass
