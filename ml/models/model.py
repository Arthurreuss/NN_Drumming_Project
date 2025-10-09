from abc import ABC, abstractmethod
from typing import Any, Dict
from ml.data.dataset import DrumDataset

class Model(ABC):
    def __init__(self):
        self._model = None

    @abstractmethod
    def save(self, path: str) -> None:
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        pass

    @abstractmethod
    def train(self, dataset: DrumDataset, plot: bool = True, TensorBoard: bool = False) -> None:
        pass

    @abstractmethod
    def predict(self, dataset: DrumDataset) -> Any:
        pass

    @abstractmethod
    def evaluate(self, dataset: DrumDataset, plot: bool = True) -> Dict[str, float]:
        pass

