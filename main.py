import argparse
import logging
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

from datetime import datetime

from ml.pipeline import Pipeline
from ml.utils.cfg import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        choices=["small", "medium", "large"],
        required=True,
        help="Model size to use",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        choices=[0.001, 0.0005, 0.0001],
        required=True,
        help="Learning rate to override",
    )
    args = parser.parse_args()

    cfg = load_config()
    cfg["pipeline"]["model"] = args.model
    cfg["training"]["learning_rate"] = args.learning_rate

    os.makedirs("logs", exist_ok=True)
    logfile = f"logs/{args.model}_lr{args.learning_rate}{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.FileHandler(logfile), logging.StreamHandler()],
    )

    logger = logging.getLogger(__name__)
    logger.info("Initializing pipeline...")
    pipeline = Pipeline(cfg)
    pipeline.run()


if __name__ == "__main__":
    main()
