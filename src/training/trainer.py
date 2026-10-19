"""Epoch loop, validation, checkpoint selection and CSV/loss-curve logging."""
from __future__ import annotations

import csv
from pathlib import Path
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from src.datasets.vehicle_dataset import detection_collate
from src.evaluation.detection_metrics import compute_detection_metrics, predict_dataset
from src.training.checkpoint import save_checkpoint, load_checkpoint
from src.utils.logger import get_logger


def make_optimizer(model, config):
    params = [p for p in model.parameters() if p.requires_grad]
    name = config.get("optimizer", "AdamW").lower()
    if name == "adamw":
        return torch.optim.AdamW(params, lr=float(config["learning_rate"]), weight_decay=float(config.get("weight_decay", 0)))
    if name == "sgd":
        return torch.optim.SGD(params, lr=float(config["learning_rate"]), momentum=float(config.get("momentum", 0.9)), weight_decay=float(config.get("weight_decay", 0)))
    raise ValueError("optimizer must be AdamW or SGD")


def train_model(model, train_dataset, val_dataset, config: dict, model_config: dict,
                class_mapping: dict, device: torch.device, seed: int):
    distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    is_main = rank == 0
    sampler = DistributedSampler(train_dataset, shuffle=True, seed=seed) if distributed else None
    train_loader = DataLoader(train_dataset, batch_size=int(config["batch_size"]), shuffle=sampler is None,
                              sampler=sampler, num_workers=int(config.get("num_workers", 0)), collate_fn=detection_collate)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                            num_workers=int(config.get("num_workers", 0)), collate_fn=detection_collate)
    model.to(device)
    optimizer = make_optimizer(model, config)
    amp_enabled = bool(config.get("mixed_precision", False))
    if amp_enabled and device.type != "cuda":
        raise ValueError("mixed_precision is supported only on CUDA; set mixed_precision: false for CPU")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    start_epoch, best = 0, float("-inf")
    if config.get("resume"):
        state = load_checkpoint(config["resume"], model, optimizer, device)
        start_epoch, best = int(state["epoch"]), float(state.get("best_metric", best))
        if state.get("scaler"):
            scaler.load_state_dict(state["scaler"])
    ckpt_dir = Path(config["checkpoint_dir"]); ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(config["log_dir"]); log_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger(f"trafficvision.{model_config['initialization']}", log_dir / f"{model_config['initialization']}.log") if is_main else None
    csv_path = log_dir / f"{model_config['initialization']}_training.csv"
    write_header = not csv_path.exists() or start_epoch == 0
    stream = csv_path.open("a" if not write_header else "w", newline="", encoding="utf-8") if is_main else None
    writer = csv.DictWriter(stream, fieldnames=["epoch", "train_loss", "loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg", "val_precision", "val_recall", "val_map50"]) if is_main else None
    if is_main and write_header: writer.writeheader()
    best_metric_key = config.get("best_metric", "val_map50")
    if best_metric_key not in {"val_map50", "val_precision", "val_recall"}:
        raise ValueError("best_metric must be val_map50, val_precision, or val_recall")
    checkpoint_metadata = {"training_config": dict(config), "best_metric_name": best_metric_key}
    try:
        for epoch in range(start_epoch, int(config["epochs"])):
            if sampler is not None: sampler.set_epoch(epoch)
            model.train(); loss_totals = {}; steps = 0
            for images, targets in train_loader:
                images = [x.to(device) for x in images]
                targets = [{k: v.to(device) for k, v in y.items() if k in {"boxes", "labels"}} for y in targets]
                optimizer.zero_grad(set_to_none=True)
                try:
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                        losses = model(images, targets)
                        loss = sum(losses.values())
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Non-finite training loss: {loss.item()}")
                    scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
                except torch.cuda.OutOfMemoryError as exc:
                    raise RuntimeError("CUDA out of memory. Reduce configured batch_size; it was not changed automatically.") from exc
                for key, value in losses.items(): loss_totals[key] = loss_totals.get(key, 0.0) + float(value.detach())
                loss_totals["total"] = loss_totals.get("total", 0.0) + float(loss.detach())
                steps += 1
            mean_loss = {key: value / max(steps, 1) for key, value in loss_totals.items()}
            if distributed:
                keys = sorted(mean_loss)
                values = torch.tensor([mean_loss[k] for k in keys], dtype=torch.float64, device=device)
                dist.all_reduce(values, op=dist.ReduceOp.SUM); values /= dist.get_world_size()
                mean_loss = dict(zip(keys, values.cpu().tolist()))
            if is_main:
                eval_model = model.module if hasattr(model, "module") else model
                predictions, targets = predict_dataset(eval_model, val_loader, device)
                metrics = compute_detection_metrics(predictions, targets, model_config["class_names"],
                                                    float(config.get("validation_confidence_threshold", .5)),
                                                    float(config.get("validation_iou_threshold", .5)))
                row = {"epoch": epoch + 1, "train_loss": mean_loss.get("total", 0.0),
                       "loss_classifier": mean_loss.get("loss_classifier", 0.0), "loss_box_reg": mean_loss.get("loss_box_reg", 0.0),
                       "loss_objectness": mean_loss.get("loss_objectness", 0.0), "loss_rpn_box_reg": mean_loss.get("loss_rpn_box_reg", 0.0),
                       "val_precision": metrics["precision"], "val_recall": metrics["recall"], "val_map50": metrics["map50"]}
                writer.writerow(row); stream.flush()
                logger.info(f"epoch {epoch+1}/{config['epochs']} loss={row['train_loss']:.4f} val mAP@50={row['val_map50']:.4f}")
                selected_metric = row[best_metric_key]
                is_best = selected_metric > best
                if is_best: best = selected_metric
                save_checkpoint(ckpt_dir / "last.pt", model, optimizer, epoch + 1, best, model_config, class_mapping, seed,
                                extra=checkpoint_metadata, scaler=scaler)
                if (epoch + 1) % int(config.get("save_every", 1)) == 0:
                    save_checkpoint(ckpt_dir / f"epoch_{epoch+1:03d}.pt", model, optimizer, epoch + 1, best, model_config, class_mapping, seed,
                                    extra=checkpoint_metadata, scaler=scaler)
                if is_best:
                    save_checkpoint(ckpt_dir / "best.pt", model, optimizer, epoch + 1, best, model_config, class_mapping, seed,
                                    {**checkpoint_metadata, "validation_metrics": metrics}, scaler=scaler)
            if distributed: dist.barrier()
    finally:
        if stream is not None: stream.close()
    if is_main:
        try:
            import matplotlib.pyplot as plt
            with csv_path.open(encoding="utf-8") as source: rows = list(csv.DictReader(source))
            plt.figure(); plt.plot([float(x["epoch"]) for x in rows], [float(x["train_loss"]) for x in rows], label="train loss")
            plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.tight_layout()
            plt.savefig(log_dir / f"{model_config['initialization']}_loss.png"); plt.close()
        except ImportError:
            pass
    return csv_path
