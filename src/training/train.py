"""Config-driven scratch/pretrained training entrypoint."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from torch.utils.data import DataLoader

from src.datasets import build_dataset
from src.models import build_model
from src.training.trainer import train_model
from src.utils.config import load_yaml, choose_device
from src.utils.seed import set_seed


def run_training(config_path: str, initialization: str | None = None):
    config = load_yaml(config_path)
    if int(config.get("batch_size", 0)) < 1 or int(config.get("epochs", 0)) < 1:
        raise ValueError("batch_size and epochs must be positive integers")
    if float(config.get("learning_rate", 0)) <= 0 or float(config.get("weight_decay", 0)) < 0:
        raise ValueError("learning_rate must be positive and weight_decay must be non-negative")
    if int(config.get("num_workers", 0)) < 0 or int(config.get("save_every", 1)) < 1:
        raise ValueError("num_workers must be non-negative and save_every must be positive")
    base = Path(config_path).resolve().parent.parent
    dataset_config = load_yaml(base / config.get("dataset_config", "configs/dataset.yaml"))
    model_config = load_yaml(base / config.get("model_config", "configs/model.yaml"))
    if initialization:
        model_config["initialization"] = initialization
    model_config["class_names"] = dataset_config["class_names"]
    model_config["num_classes"] = len(dataset_config["class_names"]) + 1
    train_cfg = {**config, "validation_fraction": dataset_config.get("validation_fraction", config.get("validation_fraction", .15))}
    train_cfg["checkpoint_dir"] = str(base / config["checkpoint_dir"] / model_config["initialization"])
    train_cfg["log_dir"] = str(base / config["log_dir"])
    set_seed(int(config.get("seed", 42)))
    device = choose_device(config.get("device", "auto"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1 and config.get("distributed", "auto") is not False
    if distributed:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if device.type == "cuda":
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            backend = "nccl"
        else:
            backend = "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
    train_set = build_dataset(dataset_config, "train", training=True)
    val_set = build_dataset(dataset_config, "val", training=False)
    overlap = set(train_set.sequences) & set(val_set.sequences)
    if overlap:
        raise RuntimeError(f"Train/validation sequence leakage: {sorted(overlap)[:5]}")
    model = build_model(model_config, int(config.get("seed", 42))).to(device)
    if distributed:
        model = DistributedDataParallel(model, device_ids=[device.index] if device.type == "cuda" else None)
    class_mapping = {"0": "__background__", **{str(i): name for i, name in enumerate(model_config["class_names"], start=1)}}
    if not distributed or dist.get_rank() == 0:
        print(f"Device: {device}; train frames: {len(train_set)}; validation frames: {len(val_set)}")
        print(f"Sequences: {len(train_set.sequences)} train / {len(val_set.sequences)} validation")
        print(f"Distributed workers: {world_size if distributed else 1}")
        print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    result = train_model(model, train_set, val_set, train_cfg, model_config, class_mapping, device, int(config.get("seed", 42)))
    if distributed:
        dist.destroy_process_group()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--initialization", choices=["scratch", "pretrained"])
    args = parser.parse_args()
    run_training(args.config, args.initialization)


if __name__ == "__main__":
    main()
