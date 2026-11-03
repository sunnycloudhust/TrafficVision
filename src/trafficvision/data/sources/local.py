import json
from pathlib import Path
from PIL import Image
from .base import DatasetSource


class LocalDatasetSource(DatasetSource):
    def __init__(self, manifest, split):
        self.path = Path(manifest)
        self.records = []
        with self.path.open() as handle:
            for line in handle:
                row = json.loads(line)
                if row["split"] == split:
                    self.records.append(row)
        if not self.records:
            raise ValueError(f"No records for split {split!r} in {manifest}")
        videos = {}
        with self.path.open() as handle:
            for line in handle:
                row = json.loads(line)
                previous = videos.setdefault(row["video_id"], row["split"])
                if previous != row["split"]:
                    raise ValueError("Sequence leakage in manifest")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(self.path.parent / record["image"]) as image:
            return image.convert("RGB"), record

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]
