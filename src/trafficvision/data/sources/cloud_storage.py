"""Sequential TAR streaming: reads image+JSON pairs without extracting archives.

Manifest is small local JSON; shard URLs may be HTTP, hf://, s3://, gs:// or files.
Cache verifies whole-shard SHA256. Stream checks each image's embedded checksum.
"""

from io import BytesIO
import hashlib
import json
from pathlib import Path
import random
import tarfile
import tempfile
import fsspec
from PIL import Image
from torch.utils.data import get_worker_info
from .base import DatasetSource
from ..cache import cached_file
from ..splits import validate_splits


class CloudObjectStorageSource(DatasetSource):
    def __init__(self, manifest, split, mode="auto", cache_dir=None, seed=42, shuffle=False):
        if mode not in ("auto", "stream", "cache"):
            raise ValueError("mode must be auto, stream or cache")
        self.manifest_path = Path(manifest).resolve()
        metadata = json.loads(self.manifest_path.read_text())
        validate_splits(metadata["splits"])
        self.videos = set(metadata["splits"][split])
        self.shards = [s for s in metadata["shards"] if s["split"] == split]
        if not self.shards:
            raise ValueError(f"No shards for {split}")
        self.mode = metadata.get("preferred_mode", "stream") if mode == "auto" else mode
        if self.mode not in ("cache", "stream"):
            raise ValueError("Manifest preferred_mode must be cache or stream")
        self._temporary = tempfile.TemporaryDirectory(prefix="trafficvision-") if cache_dir is None else None
        self.cache_dir = cache_dir or self._temporary.name
        self.seed, self.shuffle, self.epoch = seed, shuffle, 0

    def __getstate__(self):
        # Spawned workers share the path, while the parent owns cache cleanup.
        state = self.__dict__.copy()
        state["_temporary"] = None
        return state

    def __iter__(self):
        shards = list(self.shards)
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(shards)
        worker = get_worker_info()
        if worker is not None:
            shards = shards[worker.id :: worker.num_workers]
        for shard in shards:
            url = shard["url"]
            if "://" not in url and not Path(url).is_absolute():
                url = str(self.manifest_path.parent / url)
            if self.mode == "cache":
                url = str(cached_file(url, self.cache_dir, shard.get("sha256"), shard.get("size")))
            # stream=True requests sequential HTTP access; no snapshot download.
            with (
                fsspec.open(url, "rb", block_size=0).open() as handle,
                tarfile.open(fileobj=handle, mode="r|*") as archive,
            ):
                pending = {}
                for member in archive:
                    if not member.isfile():
                        continue
                    key, ext = member.name.rsplit(".", 1)
                    if ext not in ("jpg", "json"):
                        continue
                    if key not in pending and pending:
                        raise ValueError("Shard members must be adjacent image/JSON pairs")
                    pending.setdefault(key, {})[ext] = archive.extractfile(member).read()
                    if len(pending[key]) == 2:
                        pair = pending.pop(key)
                        row = json.loads(pair["json"])
                        if row["video_id"] not in self.videos or row["split"] != shard["split"]:
                            raise ValueError("Shard contains a sequence outside its declared split")
                        if hashlib.sha256(pair["jpg"]).hexdigest() != row["image_sha256"]:
                            raise ValueError("Image checksum mismatch")
                        with Image.open(BytesIO(pair["jpg"])) as image:
                            yield image.convert("RGB"), row
                if pending:
                    raise ValueError("Incomplete image/JSON pair in shard")
