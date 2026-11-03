"""Atomic downloads with bounded retries, checksum and optional ephemeral cache."""

from pathlib import Path
import hashlib
import os
import tempfile
import time
import fsspec
from tqdm.auto import tqdm


def sha256_file(path):
    digest = hashlib.sha256()
    path = Path(path)
    with open(path, "rb") as handle:
        with tqdm(
            total=path.stat().st_size,
            desc=f"Verifying {path.name}",
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            leave=False,
        ) as progress:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
                progress.update(len(chunk))
    return digest.hexdigest()


def cached_file(url, cache_dir, sha256=None, size=None, retries=3):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / (sha256 or hashlib.sha256(url.encode()).hexdigest())

    def valid(path):
        return (
            path.exists()
            and (size is None or path.stat().st_size == size)
            and (sha256 is None or sha256_file(path) == sha256)
        )

    if valid(dest):
        return dest
    error = None
    for attempt in range(retries):
        fd, temporary = tempfile.mkstemp(dir=cache_dir, suffix=".part")
        try:
            with os.fdopen(fd, "wb") as out, fsspec.open(url, "rb").open() as source:
                total = size or getattr(source, "size", None)
                with tqdm(
                    total=total,
                    desc=f"Downloading {Path(url).name or 'dataset shard'}",
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    leave=False,
                ) as progress:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        out.write(chunk)
                        progress.update(len(chunk))
            if not valid(Path(temporary)):
                raise ValueError("Downloaded file failed size/SHA256 validation")
            os.replace(temporary, dest)
            return dest
        except Exception as exc:
            error = exc
            Path(temporary).unlink(missing_ok=True)
            if attempt + 1 < retries:
                time.sleep(min(2**attempt, 8))
    raise RuntimeError(
        f"Download failed after {retries} attempts; check source permissions, quota and manifest"
    ) from error
