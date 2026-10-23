"""Dataset readers used by the training and evaluation pipelines."""

from .detrac_dataset import DETRACDataset, detection_collate_fn

__all__ = ["DETRACDataset", "detection_collate_fn"]
