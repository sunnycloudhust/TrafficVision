"""Small deterministic fixtures shared by pipeline tests."""

import torch


class SyntheticDetector:
    """Return a known moving box so tests can verify tracking and speed export."""

    def __init__(self):
        self.index = 0

    def predict(self, image):
        x = 20 + self.index * 2
        self.index += 1
        return dict(
            boxes=torch.tensor([[x, 50.0, x + 30, 80.0]]),
            scores=torch.tensor([0.9]),
            labels=torch.tensor([1]),
        )
