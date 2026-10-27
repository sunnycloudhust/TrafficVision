"""Train a configured DETRAC detector with sequence-level validation."""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data import DETRACDataset, detection_collate_fn
from src.models import build_model
from src.utils import ensure_dir, load_yaml, project_path, resolve_device


def _mean_loss(model, loader, device: torch.device, optimizer=None) -> float:
    training = optimizer is not None
    model.train()
    if not training:
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()

    total_loss = 0.0
    total_batches = 0

    for images, targets in loader:
        # Move images and tensor-valued targets to the selected device.
        images = [image.to(device) for image in images]
        targets = [{key: value.to(device) if isinstance(value, torch.Tensor) else value
                    for key, value in target.items()} for target in targets]

        if training:
            # Training: calculate the loss and gradients, then update model weights.
            optimizer.zero_grad(set_to_none=True)
            losses = model(images, targets)
            loss = sum(losses.values())
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss: {loss.item()}")
            loss.backward()
            optimizer.step()
        else:
            # Validation: measure the loss without computing gradients or updating weights.
            with torch.no_grad():
                losses = model(images, targets)
                loss = sum(losses.values())

        total_loss += float(loss.detach().item())
        total_batches += 1

    return total_loss / total_batches if total_batches else 0.0


def train(
    config_path: Path,
    *,
    architecture=None,
    initialization=None,
    output_dir=None,
    resume=None,
    resume_from=None,
) -> Path:
    # Load the training and model configs; resolve relative paths from the project root.
    config = load_yaml(config_path)
    dataset_root = project_path(config.get("dataset_root", "dataset/DETRAC"))
    model_config_path = project_path(config.get("model_config", "configs/model.yaml"))
    model_config = load_yaml(model_config_path)
    
    # Function arguments take precedence over values in the YAML files.
    architecture = architecture or config.get("architecture")
    if architecture:
        model_config["architecture"] = architecture
    initialization = initialization or config.get("initialization")
    if initialization:
        model_config["initialization"] = initialization
    architecture = model_config.get("architecture", "fasterrcnn_resnet50_fpn")
    # Map the architecture to a short name for the output directory.
    model_name = {
        "fasterrcnn_resnet50_fpn": "faster_rcnn",
        "fast_rcnn": "fast_rcnn",
        "rcnn": "rcnn",
    }.get(architecture)
    if model_name is None:
        raise ValueError(f"Unsupported model architecture: {architecture}")

    initialization = model_config.get("initialization", "scratch")
    
    configured_output_dir = output_dir or config.get("output_dir")
    selected_output_dir = configured_output_dir or f"outputs/detrac_{model_name}_{initialization}"
    selected_output_dir = project_path(selected_output_dir)
    selected_output_dir = ensure_dir(selected_output_dir)

    class_names = model_config["class_names"]

    seed = int(config.get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    dataset = DETRACDataset(
        root=dataset_root,
        split="train",
        class_names=class_names,
        max_frames_per_sequence=config.get("max_frames_per_sequence"),
    )
    if dataset.missing_images:
        print(f"Warning: skipped {len(dataset.missing_images)} annotated frame(s) with missing image files.")
        for missing_path in dataset.missing_images[:5]:
            print(f"  Missing: {missing_path}")
        if len(dataset.missing_images) > 5:
            print(f"  ... and {len(dataset.missing_images) - 5} more")
    
    all_sequences = sorted({record["sequence"] for record in dataset.records})
    val_fraction = float(config.get("validation_fraction", 0.15))

    if not 0 < val_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    
    # Shuffle the video list, then reserve whole videos for validation.
    shuffled_sequences = all_sequences.copy()
    random.Random(seed).shuffle(shuffled_sequences)
    
    # Choose the validation video count, keeping at least one video in each split.
    val_count = min(max(1, round(len(shuffled_sequences) * val_fraction)), len(shuffled_sequences) - 1)
    val_sequences = set(shuffled_sequences[:val_count])
    train_sequences = set(shuffled_sequences[val_count:])
    
    train_indices = [i for i, record in enumerate(dataset.records) if record["sequence"] in train_sequences]
    val_indices = [i for i, record in enumerate(dataset.records) if record["sequence"] in val_sequences]
    train_subset = torch.utils.data.Subset(dataset, train_indices)
    val_subset = torch.utils.data.Subset(dataset, val_indices)

    # Select a device and configure how DataLoaders read and group batches.
    device = resolve_device(str(config.get("device", "auto")))
    workers = int(config.get("num_workers", 4))
    loader_options = {
        "batch_size": int(config.get("batch_size", 32)),
        "num_workers": workers,
        "collate_fn": detection_collate_fn,
        "pin_memory": bool(config.get("pin_memory", True)) and device.type == "cuda",
        "persistent_workers": workers > 0,
    }
    # Shuffle training samples each epoch; keep validation order fixed for consistent evaluation.
    train_loader = DataLoader(train_subset, shuffle=True, **loader_options)
    val_loader = DataLoader(val_subset, shuffle=False, **loader_options)

    resume_enabled = bool(config.get("resume", True)) if resume is None else resume
    configured_resume_path = resume_from or (config.get("resume_from") if resume_enabled else None)
    resume_path = project_path(configured_resume_path) if configured_resume_path else None
    if resume_path is None and resume_enabled:
        candidate = selected_output_dir / "last.pth"
        if candidate.is_file():
            resume_path = candidate

    # A full checkpoint contains all model weights, so avoid downloading pretrained initialization weights when resuming it.
    build_config = dict(model_config)
    if resume_path is not None:
        build_config["initialization"] = "scratch"
    model = build_model(build_config).to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    
    # SGD updates the weights; learning rate, momentum, and weight decay come from YAML.
    optimizer = torch.optim.SGD(
        parameters,
        lr=float(config.get("learning_rate", 0.005)),
        momentum=float(config.get("momentum", 0.9)),
        weight_decay=float(config.get("weight_decay", 0.0005)),
    )
    # Reduce the learning rate by gamma every step_size epochs.
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(config.get("lr_step_size", 8)),
        gamma=float(config.get("lr_gamma", 0.1)),
    )

    # Track the lowest validation loss to select the best checkpoint.
    best_val_loss = float("inf")
    epochs = int(config.get("epochs", 20))
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    start_epoch = 1
    if resume_path is not None:
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        saved_model_config = checkpoint.get("model_config", {})
        if saved_model_config:
            saved_architecture = saved_model_config.get("architecture", "fasterrcnn_resnet50_fpn")
            if saved_architecture != architecture:
                raise ValueError(
                    f"Cannot resume {resume_path}: checkpoint architecture is {saved_architecture}, "
                    f"but requested architecture is {architecture}"
                )
            saved_classes = checkpoint.get("class_names")
            if saved_classes is not None and list(saved_classes) != list(class_names):
                raise ValueError(
                    f"Cannot resume {resume_path}: checkpoint classes {saved_classes} "
                    f"do not match configured classes {class_names}"
                )
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        completed_epoch = int(checkpoint.get("epoch", 0))
        start_epoch = completed_epoch + 1
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        else:
            # Backward compatibility with checkpoints written before scheduler state was saved.
            scheduler.last_epoch = completed_epoch
            scheduler._step_count = completed_epoch + 1
        best_val_loss = float(checkpoint.get("best_validation_loss", float("inf")))
        if best_val_loss == float("inf") and (selected_output_dir / "best.pth").is_file():
            best_checkpoint = torch.load(
                selected_output_dir / "best.pth", map_location="cpu", weights_only=False
            )
            best_val_loss = float(best_checkpoint.get("validation_loss", float("inf")))
            del best_checkpoint
        saved_train_sequences = checkpoint.get("train_sequences")
        saved_validation_sequences = checkpoint.get("validation_sequences")
        if saved_train_sequences is not None and set(saved_train_sequences) != train_sequences:
            raise ValueError("Cannot resume: training sequence split differs from the checkpoint.")
        if saved_validation_sequences is not None and set(saved_validation_sequences) != val_sequences:
            raise ValueError("Cannot resume: validation sequence split differs from the checkpoint.")
        print(f"Resuming from {resume_path}; will train {epochs} more epoch(s), starting at epoch {start_epoch}")
        del checkpoint
    else:
        print("Starting a new training run (no resume checkpoint selected).")

    print(f"Device: {device}; train sequences: {len(train_sequences)}; validation sequences: {len(val_sequences)}")
    print(f"Train frames: {len(train_subset)}; validation frames: {len(val_subset)}")
    print(f"Architecture: {architecture}; classes: {class_names}; "
          f"initialization: {model_config.get('initialization', 'scratch')}")

    final_epoch = start_epoch + epochs - 1
    for epoch in range(start_epoch, final_epoch + 1):
        train_loss = _mean_loss(model, train_loader, device, optimizer)
        val_loss = _mean_loss(model, val_loader, device)
        scheduler.step()
        print(f"Epoch {epoch} - train loss: {train_loss:.5f} - val loss: {val_loss:.5f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "class_names": class_names,
            "model_config": model_config,
            "train_sequences": sorted(train_sequences),
            "validation_sequences": sorted(val_sequences),
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "best_validation_loss": best_val_loss,
        }
        # last.pth stores the latest epoch; update best.pth only when validation loss improves.
        torch.save(checkpoint, selected_output_dir / "last.pth")
        if val_loss <= best_val_loss:
            torch.save(checkpoint, selected_output_dir / "best.pth")

    # Return best.pth so the caller can pass it to evaluation.
    print(f"Best validation loss: {best_val_loss:.5f}")
    print(f"Checkpoints saved to: {selected_output_dir}")
    return selected_output_dir / "best.pth"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    args = parser.parse_args()
    train(project_path(args.config))


if __name__ == "__main__":
    main()
