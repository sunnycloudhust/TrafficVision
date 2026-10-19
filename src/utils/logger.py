"""Consistent console and file logging."""
import logging
from pathlib import Path


def get_logger(name: str, log_file: str | Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    console = logging.StreamHandler(); console.setFormatter(formatter); logger.addHandler(console)
    if log_file is not None:
        path = Path(log_file); path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8"); file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger
