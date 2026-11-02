"""Unified adapters: RGB float CHW tensors -> original-image xyxy, labels 1..4."""

import torch


class TorchDetector:
    def __init__(self, model, device="cpu", confidence=0.05):
        self.model = model.to(device).eval()
        self.device, self.confidence = device, confidence

    @torch.inference_mode()
    def predict(self, image):
        result = self.model([image.to(self.device)])[0]
        keep = result["scores"] >= self.confidence
        return {key: result[key][keep].detach().cpu() for key in ("boxes", "scores", "labels")}
