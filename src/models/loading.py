"""Load an experiment checkpoint for inference without redownloading pretraining weights."""
from __future__ import annotations

import torch
from .faster_rcnn_custom import build_scratch_model
from src.training.checkpoint import load_checkpoint


def load_model_checkpoint(path, device):
    try:
        state = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(path, map_location=device)
    config = dict(state["model_config"])
    # Both strategies share identical architecture; the serialized state supplies all weights.
    model = build_scratch_model(config, int(state.get("seed", 42)))
    model.load_state_dict(state["model"])
    model.to(device).eval()
    return model, state
