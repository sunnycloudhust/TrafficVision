"""YAML configs with relative inheritance and explicit override semantics."""

from pathlib import Path
import yaml


def merge(base, override):
    result = dict(base)
    for key, value in override.items():
        result[key] = (
            merge(result[key], value)
            if isinstance(value, dict) and isinstance(result.get(key), dict)
            else value
        )
    return result


def load_config(path, _seen=None):
    path = Path(path).resolve()
    seen = set() if _seen is None else set(_seen)
    if path in seen:
        raise ValueError(f"Circular config inheritance: {path}")
    seen.add(path)
    with path.open() as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    parents = config.pop("extends", [])
    parents = [parents] if isinstance(parents, str) else parents
    result = {}
    for parent in parents:
        result = merge(result, load_config(path.parent / parent, seen))
    return merge(result, config)
