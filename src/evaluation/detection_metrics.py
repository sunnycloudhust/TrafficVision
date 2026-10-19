"""Image-level precision, recall and class-averaged AP at IoU 0.50."""
from __future__ import annotations

import torch


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((len(boxes1), len(boxes2)), dtype=torch.float32)
    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    a = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    b = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    return inter / (a[:, None] + b[None, :] - inter).clamp(min=1e-12)


def compute_detection_metrics(predictions: list[dict], targets: list[dict], class_names: list[str],
                              confidence_threshold: float = 0.5, iou_threshold: float = 0.5) -> dict:
    """Greedy one-to-one matching per image/class; AP is global, not batch-averaged."""
    if not 0 <= confidence_threshold <= 1 or not 0 <= iou_threshold <= 1:
        raise ValueError("confidence_threshold and iou_threshold must be between 0 and 1")
    per_class = {}
    for label, name in enumerate(class_names, start=1):
        gt_by_image = []
        total_gt = 0
        all_predictions = []
        for image_index, (pred, target) in enumerate(zip(predictions, targets)):
            gt_mask = target["labels"].cpu() == label
            gt_boxes = target["boxes"].cpu()[gt_mask]
            gt_by_image.append({"boxes": gt_boxes, "matched": torch.zeros(len(gt_boxes), dtype=torch.bool)})
            total_gt += len(gt_boxes)
            pred_boxes = pred["boxes"].cpu()
            pred_scores = pred["scores"].cpu()
            pred_labels = pred["labels"].cpu()
            ignored = target.get("ignore_boxes", torch.empty((0, 4))).cpu()
            keep = torch.ones(len(pred_boxes), dtype=torch.bool)
            if ignored.numel() and pred_boxes.numel():
                lt = torch.maximum(pred_boxes[:, None, :2], ignored[None, :, :2])
                rb = torch.minimum(pred_boxes[:, None, 2:], ignored[None, :, 2:])
                intersection = (rb - lt).clamp(min=0).prod(dim=2)
                pred_area = ((pred_boxes[:, 2] - pred_boxes[:, 0]).clamp(min=0) *
                             (pred_boxes[:, 3] - pred_boxes[:, 1]).clamp(min=0)).clamp(min=1e-12)
                keep = (intersection / pred_area[:, None]).amax(dim=1) < 0.5
            pred_mask = (pred_labels == label) & keep
            for box, score in zip(pred_boxes[pred_mask], pred_scores[pred_mask]):
                all_predictions.append((float(score), image_index, box))
        all_predictions.sort(key=lambda item: item[0], reverse=True)
        tp, fp = [], []
        threshold_tp = threshold_fp = 0
        for score, image_index, box in all_predictions:
            entry = gt_by_image[image_index]
            overlaps = box_iou(box.reshape(1, 4), entry["boxes"])[0]
            if overlaps.numel():
                best = int(overlaps.argmax())
                is_tp = float(overlaps[best]) >= iou_threshold and not bool(entry["matched"][best])
            else:
                is_tp = False
                best = -1
            if is_tp:
                entry["matched"][best] = True
                tp.append(1); fp.append(0)
                if score >= confidence_threshold: threshold_tp += 1
            else:
                tp.append(0); fp.append(1)
                if score >= confidence_threshold: threshold_fp += 1
        if total_gt and tp:
            cum_tp = torch.tensor(tp, dtype=torch.float64).cumsum(0)
            cum_fp = torch.tensor(fp, dtype=torch.float64).cumsum(0)
            recall = cum_tp / total_gt
            precision = cum_tp / (cum_tp + cum_fp).clamp(min=1)
            mrec = torch.cat([torch.tensor([0.], dtype=torch.float64), recall, torch.tensor([1.], dtype=torch.float64)])
            mpre = torch.cat([torch.tensor([0.], dtype=torch.float64), precision, torch.tensor([0.], dtype=torch.float64)])
            for i in range(len(mpre) - 2, -1, -1):
                mpre[i] = torch.maximum(mpre[i], mpre[i + 1])
            changes = torch.where(mrec[1:] != mrec[:-1])[0]
            ap = float(((mrec[changes + 1] - mrec[changes]) * mpre[changes + 1]).sum())
        else:
            ap = 0.0 if total_gt else None
        per_class[name] = {"precision": threshold_tp / max(threshold_tp + threshold_fp, 1),
                           "recall": threshold_tp / max(total_gt, 1), "ap50": ap,
                           "ground_truths": total_gt, "true_positives": threshold_tp,
                           "false_positives": threshold_fp, "false_negatives": total_gt - threshold_tp}
    included = [v for v in per_class.values() if v["ground_truths"] > 0]
    return {"precision": sum(x["precision"] for x in included) / max(1, len(included)),
            "recall": sum(x["recall"] for x in included) / max(1, len(included)),
            "map50": sum(x["ap50"] for x in included) / max(1, len(included)),
            "confidence_threshold": confidence_threshold, "iou_threshold": iou_threshold,
            "aggregation": "unweighted mean over configured classes with at least one ground-truth object",
            "per_class": per_class}


@torch.inference_mode()
def predict_dataset(model, loader, device):
    model.eval()
    predictions, targets = [], []
    roi_heads = getattr(model, "roi_heads", None)
    old_threshold = roi_heads.score_thresh if roi_heads is not None else None
    # Preserve low-confidence candidates for ranked AP; precision/recall apply their own threshold later.
    if roi_heads is not None:
        roi_heads.score_thresh = 0.0
    try:
        for images, batch_targets in loader:
            outputs = model([image.to(device) for image in images])
            predictions.extend({k: v.detach().cpu() for k, v in out.items()} for out in outputs)
            targets.extend({k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in batch_targets)
    finally:
        if roi_heads is not None:
            roi_heads.score_thresh = old_threshold
    return predictions, targets
