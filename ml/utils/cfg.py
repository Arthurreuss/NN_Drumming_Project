import yaml


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
