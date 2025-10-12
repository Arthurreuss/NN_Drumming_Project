import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

from ml.pipeline import Pipeline


def main():
    pipeline = Pipeline()
    pipeline.run()


if __name__ == "__main__":
    main()
