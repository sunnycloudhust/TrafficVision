"""Precision, recall and AP@50 for Faster R-CNN style predictions."""
from __future__ import annotations

from collections import defaultdict

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


class DetectionMetricAccumulator:
    """Incrementally calculate dataset-level precision, recall and AP@50.

    ``update`` accepts one image at a time, allowing full-dataset inference to
    calculate metrics without retaining every image's predictions in memory.
    """

    def __init__(self, iou_threshold=0.5, score_threshold=0.5):
        if not 0 <= iou_threshold <= 1 or not 0 <= score_threshold <= 1:
            raise ValueError("IoU and score thresholds must be between 0 and 1")
        self.iou_threshold = float(iou_threshold)
        self.score_threshold = float(score_threshold)
        self.gt_count = defaultdict(int)
        self.scores_by_class = defaultdict(list)
        self.tp_by_class = defaultdict(list)
        self.gt_total = 0
        self.pred_count = 0
        self.tp_count = 0

    def update(self, prediction, target):
        pb = _tensor(prediction["boxes"], torch.float32)
        pl = _tensor(prediction["labels"], torch.int64)
        ps = _tensor(prediction["scores"], torch.float32)
        gb = _tensor(target["boxes"], torch.float32)
        gl = _tensor(target["labels"], torch.int64)
        if len(pb) != len(pl) or len(pb) != len(ps) or len(gb) != len(gl):
            raise ValueError("boxes, labels and scores must have matching lengths")

        self.gt_total += sum(int(label > 0) for label in gl.tolist())
        self.pred_count += sum(int(score >= self.score_threshold and label > 0)
                               for score, label in zip(ps.tolist(), pl.tolist()))
        for label in gl.tolist():
            if label > 0:
                self.gt_count[label] += 1

        classes = {int(label) for label in gl.tolist() if label > 0}
        classes.update(int(label) for label in pl.tolist() if label > 0)
        for cls in classes:
            pred_indices = torch.where(pl == cls)[0]
            if len(pred_indices):
                pred_indices = pred_indices[torch.argsort(ps[pred_indices], descending=True)]
            gt_indices = torch.where(gl == cls)[0]
            used_gt = set()
            overlaps = box_iou(pb[pred_indices], gb[gt_indices]) if len(pred_indices) and len(gt_indices) else None

            for rank, pred_index in enumerate(pred_indices.tolist()):
                best = None
                if overlaps is not None:
                    order = torch.argsort(overlaps[rank], descending=True)
                    best = next((int(gt_indices[k]) for k in order
                                 if float(overlaps[rank, k]) >= self.iou_threshold
                                 and int(gt_indices[k]) not in used_gt), None)
                is_tp = best is not None
                if is_tp:
                    used_gt.add(best)
                    if float(ps[pred_index]) >= self.score_threshold:
                        self.tp_count += 1
                self.scores_by_class[cls].append(float(ps[pred_index]))
                self.tp_by_class[cls].append(is_tp)

    def compute(self):
        ap_by_class = {}
        for cls, n_gt in sorted(self.gt_count.items()):
            indices = sorted(range(len(self.scores_by_class[cls])),
                             key=lambda idx: self.scores_by_class[cls][idx], reverse=True)
            tp = torch.tensor([self.tp_by_class[cls][idx] for idx in indices], dtype=torch.float32)
            fp = 1 - tp
            ap = _ap101(tp, fp, n_gt)
            if ap is not None:
                ap_by_class[cls] = ap

        fp_count = self.pred_count - self.tp_count
        fn_count = self.gt_total - self.tp_count
        return {
            "precision": self.tp_count / (self.tp_count + fp_count)
            if self.tp_count + fp_count else 0.0,
            "recall": self.tp_count / self.gt_total if self.gt_total else 0.0,
            "map50": sum(ap_by_class.values()) / len(ap_by_class) if ap_by_class else 0.0,
            "ap50_per_class": ap_by_class,
            "tp": self.tp_count,
            "fp": fp_count,
            "fn": fn_count,
        }


def evaluate_detection(predictions, targets, iou_threshold=0.5, score_threshold=0.5):
    """Return micro precision/recall at score_threshold and class-macro AP@50."""
    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must have the same number of images")
    accumulator = DetectionMetricAccumulator(iou_threshold, score_threshold)
    for prediction, target in zip(predictions, targets):
        accumulator.update(prediction, target)
    return accumulator.compute()
