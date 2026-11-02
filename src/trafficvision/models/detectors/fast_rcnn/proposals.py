"""External proposals. Selective Search requires opencv-contrib.

Grid mode is explicit and intended for smoke tests, not Selective Search results.
"""

import cv2
import numpy as np
import torch
from torch import nn


class ExternalProposals(nn.Module):
    def __init__(self, strategy="selective_search", limit=300):
        super().__init__()
        if strategy not in ("selective_search", "grid"):
            raise ValueError("Unknown proposal strategy")
        self.strategy, self.limit = strategy, limit

    def forward(self, images, features, targets=None):
        result = []
        for image, (height, width) in zip(images.tensors, images.image_sizes):
            if self.strategy == "selective_search":
                if not hasattr(cv2, "ximgproc"):
                    raise RuntimeError(
                        "Selective Search needs opencv-contrib; install a single contrib OpenCV wheel"
                    )
                rgb = image[:, :height, :width].detach().cpu().permute(1, 2, 0).numpy()
                rgb = np.clip((rgb * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255, 0, 255).astype(
                    "uint8"
                )
                search = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
                search.setBaseImage(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                search.switchToSelectiveSearchFast()
                rects = search.process()[: self.limit]
                boxes = [[x, y, x + w, y + h] for x, y, w, h in rects]
            else:
                boxes = []
                for fraction in (0.125, 0.25, 0.5, 1.0):
                    for ratio in (0.5, 1.0, 2.0):
                        w, h = max(4, int(width * fraction)), max(4, int(height * fraction * ratio))
                        for y in range(0, height, max(h // 2, 1)):
                            for x in range(0, width, max(w // 2, 1)):
                                boxes.append([x, y, min(x + w, width), min(y + h, height)])
                # Retain spatial and scale coverage under a small test budget.
                indices = np.linspace(0, len(boxes) - 1, min(self.limit, len(boxes))).astype(int)
                boxes = [boxes[i] for i in indices]
            result.append(torch.tensor(boxes, dtype=image.dtype, device=image.device).reshape(-1, 4))
        return result, {}
