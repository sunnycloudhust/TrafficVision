"""Precision, recall and AP@50 for Faster R-CNN style predictions.
"""
from __future__ import annotations

import torch
from torchvision.ops import box_iou


def _tensor(value, dtype=None):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().to(dtype=dtype) if dtype else value.detach().cpu()
    return torch.as_tensor(value, dtype=dtype)


def _ap101(tp, fp, n_gt):
    if n_gt == 0:
        return None
    tp, fp = torch.cumsum(tp, 0), torch.cumsum(fp, 0)
    recall = tp / n_gt
    precision = tp / torch.clamp(tp + fp, min=1)
    return sum(float(precision[recall >= r].max()) if torch.any(recall >= r) else 0.0
               for r in torch.linspace(0, 1, 101)) / 101


def evaluate_detection(predictions, targets, iou_threshold=0.5, score_threshold=0.5):
    """Return micro precision/recall at score_threshold and class-macro AP@50."""
    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must have the same number of images")
    pred = []
    gt_count = {}
    for image_idx, (p, t) in enumerate(zip(predictions, targets)):
        pb, pl = _tensor(p["boxes"], torch.float32), _tensor(p["labels"], torch.int64)
        ps = _tensor(p["scores"], torch.float32)
        gb, gl = _tensor(t["boxes"], torch.float32), _tensor(t["labels"], torch.int64)
        if len(pb) != len(pl) or len(pb) != len(ps) or len(gb) != len(gl):
            raise ValueError("boxes, labels and scores must have matching lengths")
        for c in gl.tolist():
            if c > 0:
                gt_count[c] = gt_count.get(c, 0) + 1
        for j, score in enumerate(ps.tolist()):
            if pl[j].item() > 0:
                pred.append((score, image_idx, int(pl[j]), pb[j]))

    matched = [set() for _ in targets]
    tp_all = fp_all = 0
    ap_by_class = {}
    for cls in sorted(set(gt_count) | {r[2] for r in pred}):
        rows = sorted((r for r in pred if r[2] == cls), key=lambda r: -r[0])
        class_matches = [set() for _ in targets]
        tps, fps = [], []
        for score, image_idx, _, box in rows:
            gt_boxes = _tensor(targets[image_idx]["boxes"], torch.float32)
            gt_labels = _tensor(targets[image_idx]["labels"], torch.int64)
            indices = torch.where(gt_labels == cls)[0]
            if len(indices):
                overlaps = box_iou(box.reshape(1, 4), gt_boxes[indices])[0]
                order = torch.argsort(overlaps, descending=True)
                best = next((int(indices[k]) for k in order if float(overlaps[k]) >= iou_threshold
                             and int(indices[k]) not in class_matches[image_idx]), None)
            else:
                best = None
            is_tp = best is not None
            tps.append(float(is_tp)); fps.append(float(not is_tp))
            if is_tp:
                class_matches[image_idx].add(best)
                if score >= score_threshold:
                    matched[image_idx].add((cls, best))
        ap = _ap101(torch.tensor(tps), torch.tensor(fps), gt_count.get(cls, 0))
        if ap is not None:
            ap_by_class[cls] = ap

    tp_all = sum(len(x) for x in matched)
    pred_count = sum(1 for score, *_ in pred if score >= score_threshold)
    gt_total = sum(gt_count.values())
    fp_all, fn_all = pred_count - tp_all, gt_total - tp_all
    
    return {
        "precision": tp_all / (tp_all + fp_all) if tp_all + fp_all else 0.0,
        "recall": tp_all / gt_total if gt_total else 0.0,
        "map50": sum(ap_by_class.values()) / len(ap_by_class) if ap_by_class else 0.0,
        "ap50_per_class": ap_by_class,
        "tp": tp_all, "fp": fp_all, "fn": fn_all,
    }
