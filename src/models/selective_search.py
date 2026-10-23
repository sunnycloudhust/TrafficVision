"""Selective Search region proposals with an optional on-disk cache."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch


class SelectiveSearchProposer:
    def __init__(self, cache_dir, max_proposals=1000, mode="fast"):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_proposals = int(max_proposals)
        self.mode = mode
        if self.max_proposals < 1:
            raise ValueError("selective_search_max_proposals must be positive")
        if mode not in {"fast", "quality"}:
            raise ValueError("selective_search_mode must be 'fast' or 'quality'")

    def __call__(self, image: torch.Tensor, cache_key: str | Path | None = None) -> torch.Tensor:
        height, width = image.shape[-2:]
        cache_path = self._cache_path(cache_key)
        if cache_path is not None and cache_path.is_file():
            boxes = np.load(cache_path, allow_pickle=False)
            return torch.as_tensor(boxes, dtype=torch.float32, device=image.device)

        try:
            import cv2
        except ImportError as exc:
            raise ImportError(
                "Fast R-CNN and R-CNN with Selective Search require "
                "opencv-contrib-python (cv2.ximgproc.segmentation)."
            ) from exc
        if not hasattr(cv2, "ximgproc") or not hasattr(cv2.ximgproc, "segmentation"):
            raise RuntimeError(
                "The installed OpenCV build has no Selective Search module; "
                "install opencv-contrib-python."
            )

        rgb = image.detach().to(device="cpu", dtype=torch.float32).permute(1, 2, 0).numpy()
        rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        segmentation = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        segmentation.setBaseImage(bgr)
        if self.mode == "quality":
            segmentation.switchToSelectiveSearchQuality()
        else:
            segmentation.switchToSelectiveSearchFast()
        raw = segmentation.process()

        boxes = []
        seen = set()
        for x, y, box_width, box_height in raw:
            x1, y1 = max(0, int(x)), max(0, int(y))
            x2 = min(width, int(x + box_width))
            y2 = min(height, int(y + box_height))
            key = (x1, y1, x2, y2)
            if x2 > x1 and y2 > y1 and key not in seen:
                boxes.append(key)
                seen.add(key)
                if len(boxes) >= self.max_proposals:
                    break

        if not boxes:
            boxes = [(0, 0, width, height)]
        result = np.asarray(boxes, dtype=np.float32)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = cache_path.with_suffix(".tmp.npy")
            np.save(temporary_path, result, allow_pickle=False)
            temporary_path.replace(cache_path)
        return torch.as_tensor(result, dtype=torch.float32, device=image.device)

    def _cache_path(self, cache_key):
        if self.cache_dir is None or cache_key is None:
            return None
        source = Path(cache_key)
        try:
            stat = source.stat()
            fingerprint = f"{source.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
        except OSError:
            fingerprint = str(cache_key)
        signature = f"{fingerprint}:{self.mode}:{self.max_proposals}"
        digest = hashlib.sha1(signature.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.npy"
