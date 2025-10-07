from ml.data.preprocess import preprocess_dataset


class Pipeline:
    def __init__(self, config):
        self.cfg = config

    def preprocess_data(self):
        preprocess_dataset()

    def train_model(self, training_plot: bool=True, TensorBoard: bool=False):
        pass
