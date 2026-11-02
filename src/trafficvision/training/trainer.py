"""Reproducible training, validation AP, atomic last/best checkpoints and resume."""

import json
import os
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset
import yaml
from ..data.datasets.ua_detrac import build_dataset, collate_detection
from ..models.base import TorchDetector
from ..registry import build_model
from ..evaluation.detection_metrics import evaluate_detector


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def data_loader(config, split, generator=None):
    dataset = build_dataset(config["data"], split)
    train = split == "train"
    return DataLoader(
        dataset,
        batch_size=config.get("training", {}).get("batch_size", 2),
        shuffle=train and not isinstance(dataset, IterableDataset),
        num_workers=config.get("training", {}).get("workers", 0),
        collate_fn=collate_detection,
        generator=generator,
    )


def save_checkpoint(path, data):
    temporary = str(path) + ".tmp"
    torch.save(data, temporary)
    os.replace(temporary, path)


def train(config, model_name=None, init=None, resume=None):
    train_config = config.get("training", {})
    seed = config.get("seed", 42)
    seed_everything(seed)
    model_config = config["model"]
    name = model_name or model_config["name"]
    initialization = init or model_config.get("init", "random")
    output = Path(train_config.get("output", f"outputs/{name}"))
    output.mkdir(parents=True, exist_ok=True)
    if not resume and ((output / "last.pt").exists() or (output / "train/weights/last.pt").exists()):
        raise FileExistsError("Output already contains a run; choose a new output directory or --resume")
    if name == "yolov8":
        return train_yolo(config, initialization, output, resume)
    params = model_config.get("params", {})
    device = train_config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(name, initialization, **params).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.get("lr", 1e-4),
        weight_decay=train_config.get("weight_decay", 1e-4),
    )
    epochs = train_config.get("epochs", 30)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=train_config.get("lr_step", 10), gamma=0.1
    )
    generator = torch.Generator().manual_seed(seed)
    start, best = 0, float("-inf")
    if resume:
        saved = torch.load(resume, map_location=device, weights_only=True)
        if saved["model_name"] != name or saved["model_params"] != params:
            raise ValueError("Resume checkpoint architecture differs from config")
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        scheduler.load_state_dict(saved["scheduler_state"])
        start, best = saved["epoch"] + 1, saved["best_map"]
        torch.set_rng_state(saved["torch_rng"].cpu())
        random.setstate(saved["python_rng"])
        if "numpy_rng" in saved:
            state = saved["numpy_rng"]
            np.random.set_state(
                (
                    state["algorithm"],
                    np.array(state["keys"], dtype=np.uint32),
                    state["position"],
                    state["has_gauss"],
                    state["cached_gaussian"],
                )
            )
        generator.set_state(saved["loader_rng"].cpu())
        if torch.cuda.is_available() and saved.get("cuda_rng"):
            torch.cuda.set_rng_state_all([state.cpu() for state in saved["cuda_rng"]])
    resolved = {**config, "model": {**model_config, "name": name, "init": initialization}}
    (output / "config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False))
    training = data_loader(config, "train", generator)
    validation = data_loader(config, "val")
    for epoch in range(start, epochs):
        if hasattr(training.dataset, "set_epoch"):
            training.dataset.set_epoch(epoch)
        model.train()
        losses, steps = 0.0, 0
        for images, targets in training:
            if train_config.get("max_steps") and steps >= train_config["max_steps"]:
                break
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in target.items()} for target in targets]
            optimizer.zero_grad(set_to_none=True)
            components = model(images, targets)
            loss = sum(components.values())
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}: {components}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.get("gradient_clip", 10.0))
            optimizer.step()
            losses += loss.item()
            steps += 1
        if not steps:
            raise ValueError("Training dataset produced zero batches")
        scheduler.step()
        metrics = evaluate_detector(
            TorchDetector(model, device, confidence=0.0),
            validation,
            max_batches=train_config.get("validation_batches"),
        )
        current = metrics["map"]
        improved = current > best
        best = max(current, best)
        record = dict(epoch=epoch, loss=losses / steps, lr=optimizer.param_groups[0]["lr"], metrics=metrics)
        with (output / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        numpy_rng = np.random.get_state()
        saved = dict(
            model_name=name,
            model_params=params,
            model_state=model.state_dict(),
            init=initialization,
            config=resolved,
            optimizer_state=optimizer.state_dict(),
            scheduler_state=scheduler.state_dict(),
            epoch=epoch,
            best_map=best,
            torch_rng=torch.get_rng_state(),
            python_rng=random.getstate(),
            loader_rng=generator.get_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            class_map={"car": 1, "bus": 2, "van": 3, "others": 4},
            torch_version=str(torch.__version__),
            numpy_rng=dict(
                algorithm=numpy_rng[0],
                keys=numpy_rng[1].tolist(),
                position=numpy_rng[2],
                has_gauss=numpy_rng[3],
                cached_gaussian=numpy_rng[4],
            ),
        )
        save_checkpoint(output / "last.pt", saved)
        if improved:
            save_checkpoint(output / "best.pt", saved)
        print(json.dumps(record), flush=True)
    return output


def train_yolo(config, init, output, resume=None):
    from ultralytics import YOLO
    from ..models.baselines.yolov8 import export_yolo_dataset

    if init != "pretrained":
        raise ValueError("YOLO baseline expects --init pretrained")
    tc = config.get("training", {})
    path = output / "dataset/dataset.yaml"
    if not resume or not path.exists():
        path = export_yolo_dataset(config["data"], output / "dataset")
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    model = YOLO(resume or config["model"].get("weights", "yolov8n.pt"))
    return model.train(
        data=str(path),
        epochs=tc.get("epochs", 30),
        batch=tc.get("batch_size", 8),
        imgsz=config["model"].get("image_size", 640),
        device=tc.get("device", "cpu"),
        seed=config.get("seed", 42),
        workers=tc.get("workers", 0),
        project=str(output),
        name="train",
        exist_ok=True,
        resume=bool(resume),
        lr0=tc.get("lr", 1e-3),
        optimizer="AdamW",
    )
