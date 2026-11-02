"""Hungarian assignment and IoU-aware set losses for the educational RT-DETR."""

import torch
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment
from torchvision.ops import box_convert, generalized_box_iou, box_iou
from ..data.ignore import centers_in_regions


def set_loss(logits, boxes, targets):
    total_cls = logits.sum() * 0
    total_l1 = boxes.sum() * 0
    total_giou = boxes.sum() * 0
    for pred_logits, pred_boxes, target in zip(logits, boxes, targets):
        truth = target["boxes_normalized"]
        labels = target["labels"] - 1
        desired = torch.zeros_like(pred_logits)
        if len(truth):
            xyxy = box_convert(pred_boxes, "cxcywh", "xyxy")
            truth_xyxy = box_convert(truth, "cxcywh", "xyxy")
            cost = (
                -pred_logits.sigmoid()[:, labels]
                + 5 * torch.cdist(pred_boxes, truth, p=1)
                - 2 * generalized_box_iou(xyxy, truth_xyxy)
            )
            row, col = linear_sum_assignment(cost.detach().cpu().numpy())
            row = torch.as_tensor(row, device=logits.device)
            col = torch.as_tensor(col, device=logits.device)
            quality = box_iou(xyxy[row], truth_xyxy[col]).diag().detach().clamp(0, 1)
            desired[row, labels[col]] = quality
            total_l1 = total_l1 + F.l1_loss(pred_boxes[row], truth[col], reduction="sum") / max(len(truth), 1)
            total_giou = total_giou + (
                1 - generalized_box_iou(xyxy[row], truth_xyxy[col]).diag()
            ).sum() / max(len(truth), 1)
        weights = 0.75 * pred_logits.sigmoid().detach().pow(2) * (desired == 0) + desired
        if "ignored_normalized" in target:
            ignored = centers_in_regions(
                box_convert(pred_boxes, "cxcywh", "xyxy"), target["ignored_normalized"]
            )
            weights[ignored & (desired.sum(-1) == 0)] = 0
        total_cls = total_cls + (
            F.binary_cross_entropy_with_logits(pred_logits, desired, reduction="none") * weights
        ).sum() / max(len(truth), 1)
    batch = len(targets)
    return {
        "loss_class": total_cls / batch,
        "loss_bbox": 5 * total_l1 / batch,
        "loss_giou": 2 * total_giou / batch,
    }
