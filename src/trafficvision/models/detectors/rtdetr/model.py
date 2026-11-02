"""Compact educational RT-DETR variant, implemented here from scratch.

Multi-scale CNN, high-level attention, bidirectional feature fusion, IoU-aware
top-k query selection, iterative Transformer decoding and Hungarian set loss.
Uses dense cross-attention instead of deformable attention; no denoising queries.
Do not label these results as reproduction of the official RT-DETR benchmark.
"""

import math
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import box_convert
from ...backbones.convnet import block
from ....training.losses import set_loss


def inverse_sigmoid(x):
    return torch.logit(x.clamp(1e-4, 1 - 1e-4))


def positional(height, width, channels, device):
    y, x = torch.meshgrid(
        torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij"
    )
    omega = 1 / (10000 ** (torch.arange(channels // 4, device=device) / (channels // 4)))
    x = x.flatten()[:, None] * omega
    y = y.flatten()[:, None] * omega
    return torch.cat([x.sin(), x.cos(), y.sin(), y.cos()], dim=1)[None]


class HybridEncoder(nn.Module):
    def __init__(self, hidden, heads):
        super().__init__()
        self.attention = nn.TransformerEncoderLayer(hidden, heads, hidden * 4, dropout=0.0, batch_first=True)
        self.top = nn.ModuleList([block(hidden * 2, hidden) for _ in range(2)])
        self.bottom = nn.ModuleList([block(hidden * 2, hidden) for _ in range(2)])

    def forward(self, maps):
        low, mid, high = maps
        batch, channels, height, width = high.shape
        sequence = high.flatten(2).transpose(1, 2)
        high = (
            self.attention(sequence + positional(height, width, channels, high.device))
            .transpose(1, 2)
            .reshape(batch, channels, height, width)
        )
        mid = self.top[0](torch.cat([mid, F.interpolate(high, size=mid.shape[-2:], mode="nearest")], 1))
        low = self.top[1](torch.cat([low, F.interpolate(mid, size=low.shape[-2:], mode="nearest")], 1))
        mid = self.bottom[0](torch.cat([mid, F.adaptive_avg_pool2d(low, mid.shape[-2:])], 1))
        high = self.bottom[1](torch.cat([high, F.adaptive_avg_pool2d(mid, high.shape[-2:])], 1))
        return [low, mid, high]


class RTDETR(nn.Module):
    def __init__(
        self,
        num_classes=4,
        init="random",
        hidden=64,
        heads=4,
        queries=100,
        decoder_layers=3,
        image_size=512,
        score_threshold=0.05,
        **kwargs,
    ):
        super().__init__()
        if init != "random":
            raise ValueError("Custom RT-DETR supports random init; no hidden pretrained backbone")
        if hidden % 8 or hidden % heads or image_size < 32 or queries < 1:
            raise ValueError("hidden must divide heads and 8; image_size>=32; queries>=1")
        self.stem = nn.Sequential(block(3, 32, 2), block(32, hidden, 2))
        self.stages = nn.ModuleList([block(hidden, hidden, 2) for _ in range(3)])
        self.encoder = HybridEncoder(hidden, heads)
        self.encoder_class = nn.Linear(hidden, num_classes)
        self.encoder_box = nn.Linear(hidden, 4)
        self.layers = nn.ModuleList(
            [
                nn.TransformerDecoderLayer(hidden, heads, hidden * 4, dropout=0.0, batch_first=True)
                for _ in range(decoder_layers)
            ]
        )
        self.class_heads = nn.ModuleList([nn.Linear(hidden, num_classes) for _ in self.layers])
        self.box_heads = nn.ModuleList(
            [nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 4)) for _ in self.layers]
        )
        self.query_position = nn.Sequential(nn.Linear(4, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.queries, self.image_size, self.score_threshold = queries, image_size, score_threshold
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        for head in [self.encoder_class, *self.class_heads]:
            nn.init.constant_(head.bias, -math.log(99))

    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError("Training requires targets")
        sizes = [image.shape[-2:] for image in images]
        batch = torch.stack(
            [
                F.interpolate(
                    image[None], (self.image_size, self.image_size), mode="bilinear", align_corners=False
                )[0]
                for image in images
            ]
        )
        feature = self.stem((batch - self.mean) / self.std)
        maps = []
        for stage in self.stages:
            feature = stage(feature)
            maps.append(feature)
        maps = self.encoder(maps)
        sequences, anchors, positions = [], [], []
        for level, feature in enumerate(maps):
            _, channels, height, width = feature.shape
            y, x = torch.meshgrid(
                torch.arange(height, device=batch.device),
                torch.arange(width, device=batch.device),
                indexing="ij",
            )
            center = torch.stack([(x + 0.5) / width, (y + 0.5) / height], -1).reshape(-1, 2)
            anchors.append(torch.cat([center, torch.full_like(center, 0.05 * 2**level)], -1))
            sequences.append(feature.flatten(2).transpose(1, 2))
            positions.append(positional(height, width, channels, batch.device))
        memory = torch.cat(sequences, 1)
        memory_pos = torch.cat(positions, 1)
        enc_logits = self.encoder_class(memory)
        enc_boxes = (self.encoder_box(memory) + inverse_sigmoid(torch.cat(anchors)[None])).sigmoid()
        top = enc_logits.sigmoid().max(-1).values.topk(min(self.queries, memory.shape[1]), 1).indices
        query = memory.gather(1, top[..., None].expand(-1, -1, memory.shape[-1])).detach()
        reference = enc_boxes.gather(1, top[..., None].expand(-1, -1, 4)).detach()
        outputs = []
        for layer, cls, box in zip(self.layers, self.class_heads, self.box_heads):
            query = layer(query + self.query_position(reference), memory + memory_pos)
            boxes = (box(query) + inverse_sigmoid(reference)).sigmoid()
            outputs.append((cls(query), boxes))
            reference = boxes.detach()
        if self.training:
            normalized = []
            for target, (height, width) in zip(targets, sizes):
                scale = batch.new_tensor([width, height, width, height])
                normalized.append(
                    {
                        **target,
                        "boxes_normalized": box_convert(target["boxes"] / scale, "xyxy", "cxcywh"),
                        "ignored_normalized": target.get("ignored_regions", batch.new_empty((0, 4))) / scale,
                    }
                )
            losses = {}
            for index, (logits, boxes) in enumerate([(enc_logits, enc_boxes), *outputs]):
                losses.update({f"{k}_{index}": v for k, v in set_loss(logits, boxes, normalized).items()})
            return losses
        result = []
        logits, boxes = outputs[-1]
        for pred_logits, pred_boxes, (height, width) in zip(logits, boxes, sizes):
            scores, labels = pred_logits.sigmoid().max(-1)
            keep = scores >= self.score_threshold
            xyxy = box_convert(pred_boxes[keep], "cxcywh", "xyxy").clamp(0, 1) * batch.new_tensor(
                [width, height, width, height]
            )
            result.append(dict(boxes=xyxy, scores=scores[keep], labels=labels[keep] + 1))
        return result
