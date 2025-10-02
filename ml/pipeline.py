from ml.generate_data.dataset_generator import DatasetGenerator


class Pipeline:
    def __init__(self, config):
        self.config = config
        self.dataset_config = config['dataset']
        if self.dataset_config['generate']:
            self.generate_data()
        


    def generate_data(self):
        # Code to generate data based on config
        generator = DatasetGenerator(dataset_config=self.dataset_config)
        generator.create_samples()
        generator.save_samples()

        