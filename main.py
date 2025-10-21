import logging
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

from datetime import datetime

from ml.pipeline import Pipeline
from ml.utils.cfg import load_config


def main():
    os.makedirs("logs", exist_ok=True)
    cfg = load_config()
    model_size = cfg["pipeline"]["model"]
    q = cfg["pipeline"]["quantization"]
    segment_len = cfg["pipeline"]["segment_len"]
    logfile = f"logs/{model_size}_q{q}_seg{segment_len}{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.FileHandler(logfile), logging.StreamHandler()],
    )

    logger = logging.getLogger(__name__)
    logger.info("Initializing pipeline...")
    pipeline = Pipeline()
    pipeline.run()


if __name__ == "__main__":
    main()
