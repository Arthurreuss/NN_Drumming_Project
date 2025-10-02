import yaml

import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from ml.pipeline import Pipeline


def load_config(path="config.yaml") -> dict:
    """
    Load a YAML configuration file for model, dataset, and training settings.

    :param path: Path to the configuration YAML file. Defaults to "config.yaml".
    :type path: str, optional
    :return: Parsed configuration dictionary.
    :rtype: dict
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    pipeline = Pipeline(config)
    print("Boem Pow tsss!!! Drumming time!")

if __name__ == "__main__":
    main()