import os
import sys

import yaml

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

from ml.pipeline import Pipeline
from utils.cfg import load_config


def main():
    cfg = load_config()
    pipeline = Pipeline(cfg)
    pipeline.preprocess_data()


if __name__ == "__main__":
    main()
