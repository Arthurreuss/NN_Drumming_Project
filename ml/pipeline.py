from ml.data.preprocess import preprocess_dataset
from ml.models.model import Model
from ml.data.dataset import DrumDataset


class Pipeline:
    def __init__(self, config):
        self.cfg = config
        self.dataset_cfg = config['dataset']
        self.pipeline_cfg = config['pipeline']
        self._model = None

    def run(self):
        if self.pipeline_cfg['dataset'].get('preprocess', False):
            dataset: DrumDataset = self.preprocess_data()
        else:
            dataset: DrumDataset = self._load_dataset()
        if self.pipeline_cfg['train'].get("enabled", False):
            plot = self.pipeline_cfg['train'].get("plot", False)
            tensor_board = self.pipeline_cfg['train'].get("tensor_board", False)
            self.train_model
                
        

    def preprocess_data(self) -> DrumDataset:
        return preprocess_dataset()

    def train_model(self, Dataset: DrumDataset = None, training_plot: bool=True, TensorBoard: bool=False) -> Model:
        pass

    def _load_dataset(self, train_test_split: float) -> DrumDataset:
        pass


    