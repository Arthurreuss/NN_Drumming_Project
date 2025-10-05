from ml.data.preprocess import preprocess_dataset


class Pipeline:
    def __init__(self, config):
        self.cfg = config

    def preprocess_data(self):
        preprocess_dataset()
