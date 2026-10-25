"""Fast R-CNN and R-CNN detectors that use Selective Search proposals."""
from __future__ import annotations

from collections import defaultdict

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.ops import box_iou, nms, roi_align

from .selective_search import SelectiveSearchProposer
from src.utils import project_path


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _build_resnet(config):
    weights = ResNet50_Weights.DEFAULT if config.get("initialization", "scratch") == "pretrained" else None
    backbone = resnet50(weights=weights)
    if config.get("freeze_backbone", False):
        for parameter in backbone.parameters():
            parameter.requires_grad = False
    return backbone


def _normalize(image):
    mean = image.new_tensor(_IMAGENET_MEAN).view(3, 1, 1)
    std = image.new_tensor(_IMAGENET_STD).view(3, 1, 1)
    return (image - mean) / std


def _encode_boxes(proposals, targets):
    p_size = (proposals[:, 2:] - proposals[:, :2]).clamp(min=1)
    p_center = proposals[:, :2] + 0.5 * p_size
    t_size = (targets[:, 2:] - targets[:, :2]).clamp(min=1)
    t_center = targets[:, :2] + 0.5 * t_size
    delta_center = (t_center - p_center) / p_size
    delta_size = torch.log(t_size / p_size)
    return torch.cat((delta_center, delta_size), dim=1)


def _decode_boxes(proposals, deltas):
    p_size = (proposals[:, 2:] - proposals[:, :2]).clamp(min=1)
    p_center = proposals[:, :2] + 0.5 * p_size
    center = deltas[:, :2] * p_size + p_center
    size = deltas[:, 2:].clamp(max=4).exp() * p_size
    return torch.cat((center - 0.5 * size, center + 0.5 * size), dim=1)


def _sample_training_rois(proposals, target, batch_size=128, positive_fraction=0.25):
    gt_boxes = target["boxes"]
    gt_labels = target["labels"]
    if len(gt_boxes):
        proposals = torch.cat((proposals, gt_boxes), dim=0)
        overlaps = box_iou(proposals, gt_boxes)
        max_iou, matched = overlaps.max(dim=1)
        labels = gt_labels[matched].clone()
        positive = max_iou >= 0.5
        negative = max_iou < 0.3
        labels[~positive] = 0
    else:
        max_iou = proposals.new_zeros((len(proposals),))
        matched = torch.zeros((len(proposals),), dtype=torch.long, device=proposals.device)
        labels = torch.zeros((len(proposals),), dtype=torch.long, device=proposals.device)
        positive = torch.zeros_like(labels, dtype=torch.bool)
        negative = torch.ones_like(labels, dtype=torch.bool)

    max_positive = int(batch_size * positive_fraction)
    positive_indices = torch.where(positive)[0]
    negative_indices = torch.where(negative)[0]
    if len(positive_indices) > max_positive:
        positive_indices = positive_indices[torch.randperm(len(positive_indices), device=proposals.device)[:max_positive]]
    max_negative = max(0, batch_size - len(positive_indices))
    if len(negative_indices) > max_negative:
        negative_indices = negative_indices[torch.randperm(len(negative_indices), device=proposals.device)[:max_negative]]
    selected = torch.cat((positive_indices, negative_indices))
    sampled_boxes = proposals[selected]
    sampled_labels = labels[selected]
    if len(positive_indices):
        matched_gt = gt_boxes[matched[positive_indices]]
        regression_targets = _encode_boxes(proposals[positive_indices], matched_gt)
    else:
        regression_targets = proposals.new_empty((0, 4))
    return sampled_boxes, sampled_labels, positive_indices.numel(), regression_targets


def _losses(logits, box_deltas, labels, positive_count, regression_targets):
    if not len(labels):
        zero = logits.sum() * 0
        return {"loss_classifier": zero, "loss_box_reg": zero}
    classification_loss = F.cross_entropy(logits, labels)
    if positive_count:
        positive_labels = labels[:positive_count]
        selected_deltas = box_deltas[:positive_count].reshape(len(positive_labels), -1, 4)
        selected_deltas = selected_deltas[torch.arange(positive_count, device=labels.device), positive_labels]
        regression_loss = F.smooth_l1_loss(selected_deltas, regression_targets, beta=1 / 9, reduction="sum")
        regression_loss = regression_loss / max(1, len(labels))
    else:
        regression_loss = box_deltas.sum() * 0
    return {"loss_classifier": classification_loss, "loss_box_reg": regression_loss}


def _detections(proposals, logits, box_deltas, image_size, config):
    class_count = logits.shape[1]
    scores = logits.softmax(dim=1)
    deltas = box_deltas.reshape(len(proposals), class_count, 4)
    height, width = image_size
    all_boxes, all_scores, all_labels = [], [], []
    threshold = float(config.get("score_threshold", 0.05))
    nms_threshold = float(config.get("nms_threshold", 0.5))
    max_detections = int(config.get("box_detections_per_img", 100))

    for class_id in range(1, class_count):
        class_scores = scores[:, class_id]
        keep = torch.where(class_scores >= threshold)[0]
        if not len(keep):
            continue
        boxes = _decode_boxes(proposals[keep], deltas[keep, class_id])
        boxes[:, 0::2].clamp_(0, width)
        boxes[:, 1::2].clamp_(0, height)
        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes, selected_scores = boxes[valid], class_scores[keep][valid]
        if not len(boxes):
            continue
        keep_nms = nms(boxes, selected_scores, nms_threshold)
        all_boxes.append(boxes[keep_nms])
        all_scores.append(selected_scores[keep_nms])
        all_labels.append(torch.full((len(keep_nms),), class_id, dtype=torch.int64, device=boxes.device))

    if not all_boxes:
        return {
            "boxes": proposals.new_empty((0, 4)),
            "scores": proposals.new_empty((0,)),
            "labels": torch.empty((0,), dtype=torch.int64, device=proposals.device),
        }
    boxes, result_scores, labels = torch.cat(all_boxes), torch.cat(all_scores), torch.cat(all_labels)
    order = torch.argsort(result_scores, descending=True)[:max_detections]
    return {"boxes": boxes[order], "scores": result_scores[order], "labels": labels[order]}


class FastRCNNSelectiveSearch(nn.Module):
    """Fast R-CNN: shared image features and RoI pooling on Selective Search boxes."""

    def __init__(self, config):
        super().__init__()
        backbone = _build_resnet(config)
        self.backbone = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3,
        )
        self.num_classes = len(config["class_names"]) + 1
        roi_size = int(config.get("roi_pool_size", 7))
        self.roi_size = roi_size
        self.roi_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.Linear(1024, 1024), nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(1024, self.num_classes)
        self.box_regressor = nn.Linear(1024, self.num_classes * 4)
        self.proposer = _make_proposer(config)
        self.config = dict(config)
        self.roi_batch_size = int(config.get("roi_batch_size", 128))
        self.positive_fraction = float(config.get("positive_fraction", 0.25))

    def _features(self, image):
        return self.backbone(_normalize(image).unsqueeze(0))

    def _head(self, feature_map, proposals):
        logits, box_deltas = [], []
        for start in range(0, len(proposals), self.roi_batch_size):
            roi_boxes = proposals[start:start + self.roi_batch_size]
            pooled = roi_align(feature_map, [roi_boxes], output_size=self.roi_size,
                              spatial_scale=1 / 16, sampling_ratio=2, aligned=True)
            features = self.roi_head(pooled)
            logits.append(self.classifier(features))
            box_deltas.append(self.box_regressor(features))
        return torch.cat(logits), torch.cat(box_deltas)

    def forward(self, images, targets=None, image_paths=None):
        losses = defaultdict(list)
        results = []
        for index, image in enumerate(images):
            cache_key = image_paths[index] if image_paths is not None else _target_path(targets, index)
            proposals = self.proposer(image, cache_key)
            features = self._features(image)
            if self.training:
                if targets is None:
                    raise ValueError("targets are required while Fast R-CNN is training")
                boxes, labels, positive_count, regression_targets = _sample_training_rois(
                    proposals, targets[index], self.roi_batch_size, self.positive_fraction
                )
                logits, box_deltas = self._head(features, boxes)
                for name, value in _losses(logits, box_deltas, labels, positive_count, regression_targets).items():
                    losses[name].append(value)
            else:
                logits, box_deltas = self._head(features, proposals)
                results.append(_detections(proposals, logits, box_deltas, image.shape[-2:], self.config))
        if self.training:
            return {name: torch.stack(values).mean() for name, values in losses.items()}
        return results


class RCNNSelectiveSearch(nn.Module):
    """R-CNN: run a CNN on each proposed region independently."""

    def __init__(self, config):
        super().__init__()
        backbone = _build_resnet(config)
        self.backbone = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4,
        )
        self.num_classes = len(config["class_names"]) + 1
        self.classifier = nn.Sequential(nn.Linear(2048, 1024), nn.ReLU(inplace=True))
        self.class_logits = nn.Linear(1024, self.num_classes)
        self.box_regressor = nn.Linear(1024, self.num_classes * 4)
        self.proposer = _make_proposer(config)
        self.config = dict(config)
        self.roi_batch_size = int(config.get("rcnn_roi_batch_size", config.get("roi_batch_size", 128)))
        self.crop_batch_size = int(config.get("rcnn_crop_batch_size", 16))
        self.positive_fraction = float(config.get("positive_fraction", 0.25))

    def _crop_features(self, image, boxes):
        if not len(boxes):
            return image.new_empty((0, 2048))
        height, width = image.shape[-2:]
        box_values = boxes.detach().round().to(device="cpu", dtype=torch.int64).tolist()
        features = []
        for start in range(0, len(box_values), self.crop_batch_size):
            crops = []
            for x1, y1, x2, y2 in box_values[start:start + self.crop_batch_size]:
                x1, y1 = min(width - 1, max(0, x1)), min(height - 1, max(0, y1))
                x2, y2 = min(width, max(x1 + 1, x2)), min(height, max(y1 + 1, y2))
                crop = image[:, y1:y2, x1:x2]
                crop = F.interpolate(crop.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False)[0]
                crops.append(_normalize(crop))
            batch = torch.stack(crops)
            feature = self.backbone(batch)
            features.append(F.adaptive_avg_pool2d(feature, 1).flatten(1))
        return torch.cat(features)

    def forward(self, images, targets=None, image_paths=None):
        losses = defaultdict(list)
        results = []
        for index, image in enumerate(images):
            cache_key = image_paths[index] if image_paths is not None else _target_path(targets, index)
            proposals = self.proposer(image, cache_key)
            if self.training:
                if targets is None:
                    raise ValueError("targets are required while R-CNN is training")
                boxes, labels, positive_count, regression_targets = _sample_training_rois(
                    proposals, targets[index], self.roi_batch_size, self.positive_fraction
                )
            else:
                boxes = proposals
                labels = None
                positive_count = 0
                regression_targets = None

            features = self.classifier(self._crop_features(image, boxes))
            logits = self.class_logits(features)
            box_deltas = self.box_regressor(features)
            if self.training:
                for name, value in _losses(logits, box_deltas, labels, positive_count, regression_targets).items():
                    losses[name].append(value)
            else:
                results.append(_detections(boxes, logits, box_deltas, image.shape[-2:], self.config))
        if self.training:
            return {name: torch.stack(values).mean() for name, values in losses.items()}
        return results


def _make_proposer(config):
    cache_dir = config.get("selective_search_cache_dir", "outputs/selective_search_cache")
    if cache_dir:
        cache_dir = project_path(cache_dir)
    return SelectiveSearchProposer(
        cache_dir=cache_dir,
        max_proposals=config.get("selective_search_max_proposals", 1000),
        mode=config.get("selective_search_mode", "fast"),
    )


def _target_path(targets, index):
    if targets is None:
        return None
    return targets[index].get("image_path")
