"""Dataset adapters."""
from .vehicle_dataset import VehicleDataset, build_dataset, detection_collate, resolve_dataset_paths

__all__ = ["VehicleDataset", "build_dataset", "detection_collate", "resolve_dataset_paths"]
