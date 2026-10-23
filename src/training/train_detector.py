"""Train a configured DETRAC detector with sequence-level validation."""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from src.data import DETRACDataset, detection_collate_fn
from src.models import build_model


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value


def _resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _mean_loss(model, loader, device: torch.device, optimizer=None) -> float:
    training = optimizer is not None
    # Torchvision detectors and the custom region-based models return loss
    # dictionaries in train mode. Validation disables gradients and optimizer steps.
    model.train()
    if not training:
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()
    total_loss = 0.0
    total_batches = 0

    for images, targets in loader:
        images = [image.to(device) for image in images]
        targets = [{key: value.to(device) if isinstance(value, torch.Tensor) else value
                    for key, value in target.items()} for target in targets]

        if training:
            optimizer.zero_grad(set_to_none=True)
            losses = model(images, targets)
            loss = sum(losses.values())
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss: {loss.item()}")
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                losses = model(images, targets)
                loss = sum(losses.values())

        total_loss += float(loss.detach().item())
        total_batches += 1

    return total_loss / total_batches if total_batches else 0.0


def train(config_path: Path, *, architecture=None, initialization=None, output_dir=None) -> Path:
    config = _load_yaml(config_path)
    project_root = Path(__file__).resolve().parents[2]
    dataset_root = Path(config.get("dataset_root", "dataset/DETRAC"))
    if not dataset_root.is_absolute():
        dataset_root = project_root / dataset_root
    model_config_path = Path(config.get("model_config", "configs/model.yaml"))
    if not model_config_path.is_absolute():
        model_config_path = project_root / model_config_path
    model_config = _load_yaml(model_config_path)
    architecture = architecture or config.get("architecture")
    if architecture:
        model_config["architecture"] = architecture
    initialization = initialization or config.get("initialization")
    if initialization:
        model_config["initialization"] = initialization
    architecture = model_config.get("architecture", "fasterrcnn_resnet50_fpn")
    model_name = {
        "fasterrcnn_resnet50_fpn": "faster_rcnn",
        "fast_rcnn": "fast_rcnn",
        "rcnn": "rcnn",
    }.get(architecture)
    if model_name is None:
        raise ValueError(f"Unsupported model architecture: {architecture}")
    selected_output_dir = output_dir or config.get("output_dir") or f"outputs/detrac_{model_name}"
    selected_output_dir = Path(selected_output_dir)
    if not selected_output_dir.is_absolute():
        selected_output_dir = project_root / selected_output_dir
    selected_output_dir.mkdir(parents=True, exist_ok=True)
    class_names = model_config["class_names"]

    seed = int(config.get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    dataset = DETRACDataset(root=dataset_root, split="train", class_names=class_names)
    all_sequences = sorted({record["sequence"] for record in dataset.records})
    val_fraction = float(config.get("validation_fraction", 0.15))
    if not 0 < val_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    shuffled_sequences = all_sequences.copy()
    random.Random(seed).shuffle(shuffled_sequences)
    val_count = min(max(1, round(len(shuffled_sequences) * val_fraction)), len(shuffled_sequences) - 1)
    val_sequences = set(shuffled_sequences[:val_count])
    train_sequences = set(shuffled_sequences[val_count:])
    train_indices = [i for i, record in enumerate(dataset.records) if record["sequence"] in train_sequences]
    val_indices = [i for i, record in enumerate(dataset.records) if record["sequence"] in val_sequences]
    train_subset = torch.utils.data.Subset(dataset, train_indices)
    val_subset = torch.utils.data.Subset(dataset, val_indices)

    device = _resolve_device(str(config.get("device", "auto")))
    workers = int(config.get("num_workers", 4))
    loader_options = {
        "batch_size": int(config.get("batch_size", 2)),
        "num_workers": workers,
        "collate_fn": detection_collate_fn,
        "pin_memory": bool(config.get("pin_memory", True)) and device.type == "cuda",
        "persistent_workers": workers > 0,
    }
    train_loader = DataLoader(train_subset, shuffle=True, **loader_options)
    val_loader = DataLoader(val_subset, shuffle=False, **loader_options)

    model = build_model(model_config).to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.SGD(
        parameters,
        lr=float(config.get("learning_rate", 0.005)),
        momentum=float(config.get("momentum", 0.9)),
        weight_decay=float(config.get("weight_decay", 0.0005)),
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(config.get("lr_step_size", 8)),
        gamma=float(config.get("lr_gamma", 0.1)),
    )

    best_val_loss = float("inf")
    epochs = int(config.get("epochs", 20))
    print(f"Device: {device}; train sequences: {len(train_sequences)}; validation sequences: {len(val_sequences)}")
    print(f"Train frames: {len(train_subset)}; validation frames: {len(val_subset)}")
    print(f"Architecture: {architecture}; classes: {class_names}; "
          f"initialization: {model_config.get('initialization', 'scratch')}")

    for epoch in range(1, epochs + 1):
        train_loss = _mean_loss(model, train_loader, device, optimizer)
        val_loss = _mean_loss(model, val_loader, device)
        scheduler.step()
        print(f"Epoch {epoch}/{epochs} - train loss: {train_loss:.5f} - val loss: {val_loss:.5f}")

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "class_names": class_names,
            "model_config": model_config,
            "train_sequences": sorted(train_sequences),
            "validation_sequences": sorted(val_sequences),
            "train_loss": train_loss,
            "validation_loss": val_loss,
        }
        torch.save(checkpoint, selected_output_dir / "last.pth")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, selected_output_dir / "best.pth")

    print(f"Best validation loss: {best_val_loss:.5f}")
    print(f"Checkpoints saved to: {selected_output_dir}")
    return selected_output_dir / "best.pth"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    args = parser.parse_args()
    train(args.config.resolve())


if __name__ == "__main__":
    main()
