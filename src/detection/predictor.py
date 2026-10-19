"""Single-image and sequential video prediction."""
from __future__ import annotations

import cv2
import torch
from src.datasets.transforms import to_image_tensor


class Predictor:
    def __init__(self, model, device, score_threshold: float = 0.25):
        self.model = model.to(device).eval()
        self.device = device
        self.score_threshold = score_threshold

    @torch.inference_mode()
    def predict_rgb(self, rgb):
        image = to_image_tensor(rgb).to(self.device)
        result = self.model([image])[0]
        keep = result["scores"] >= self.score_threshold
        return {key: value[keep].detach().cpu() for key, value in result.items()}

    def predict_bgr(self, bgr):
        return self.predict_rgb(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
