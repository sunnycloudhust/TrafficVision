"""Checkpoint persistence with experiment metadata."""
from __future__ import annotations

from pathlib import Path
import platform
import torch
import torchvision


def save_checkpoint(path, model, optimizer, epoch: int, best_metric: float, model_config: dict,
                    class_mapping: dict, seed: int, extra: dict | None = None, scaler=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    base_model = model.module if hasattr(model, "module") else model
    torch.save({"model": base_model.state_dict(), "optimizer": optimizer.state_dict(),
                "epoch": epoch, "best_metric": best_metric, "model_config": model_config,
                "class_mapping": class_mapping, "seed": seed,
                "versions": {"python": platform.python_version(), "torch": torch.__version__,
                             "torchvision": torchvision.__version__}, "extra": extra or {},
                "scaler": scaler.state_dict() if scaler is not None else None}, path)


def load_checkpoint(path, model, optimizer=None, map_location="cpu"):
    try:
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # compatibility with earlier torch versions
        checkpoint = torch.load(path, map_location=map_location)
    base_model = model.module if hasattr(model, "module") else model
    base_model.load_state_dict(checkpoint["model"])
    if optimizer is not None and checkpoint.get("optimizer"):
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint
