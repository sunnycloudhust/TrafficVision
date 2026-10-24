"""Shared YAML, project path, and directory helpers."""
from pathlib import Path

import yaml
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(path: str | Path) -> Path:
    """Return an absolute path, resolving relative paths from the project root."""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parent_dir(path: str | Path) -> Path:
    """Return the parent directory of a file or directory path."""
    return Path(path).parent


def ensure_dir(path: str | Path) -> Path:
    """Create a directory and its parents if they do not already exist."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def ensure_parent_dir(path: str | Path) -> Path:
    """Create the parent directory for a file path and return it."""
    return ensure_dir(parent_dir(path))


def load_yaml(path: str | Path) -> dict:
    """Load a YAML file and require its top-level value to be a mapping."""
    path = project_path(path)
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value


def resolve_device(requested: str) -> torch.device:
    """Resolve an explicit device or choose CUDA, MPS, then CPU automatically."""
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
