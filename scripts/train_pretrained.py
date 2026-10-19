"""Fine-tune COCO-pretrained Faster R-CNN."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training.train import run_training
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/train.yaml")
    args = parser.parse_args(); run_training(args.config, "pretrained")
