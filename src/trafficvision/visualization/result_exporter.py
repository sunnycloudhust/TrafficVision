import csv
import json
from pathlib import Path
from ..data.parsers.xml_parser import CLASS_NAMES

FIELDS = [
    "video_id",
    "frame_id",
    "timestamp",
    "track_id",
    "label",
    "class",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "world_x",
    "world_y",
    "speed_kmh",
    "zones",
]


class ResultExporter:
    """Write a CSV and streaming JSON array; memory does not grow with video length."""

    def __init__(self, output):
        output = Path(output)
        self.csv_file = output.with_suffix(".csv").open("w", newline="")
        self.json_file = output.with_suffix(".json").open("w")
        self.writer = csv.DictWriter(self.csv_file, fieldnames=FIELDS)
        self.writer.writeheader()
        self.json_file.write("[")
        self.first = True

    def write(self, video_id, frame_id, timestamp, results):
        for result in results:
            track, world = result["track"], result["world"]
            row = dict(
                video_id=video_id,
                frame_id=frame_id,
                timestamp=float(timestamp),
                track_id=int(track.track_id),
                label=int(track.label),
                confidence=float(track.score),
                world_x=None if world is None else float(world[0]),
                world_y=None if world is None else float(world[1]),
                speed_kmh=result["speed_kmh"],
                zones=";".join(result["zones"]),
            )
            row["class"] = CLASS_NAMES[track.label - 1]
            row.update(zip(("x1", "y1", "x2", "y2"), map(float, track.box)))
            self.writer.writerow(row)
            self.json_file.write(("" if self.first else ",\n") + json.dumps(row, allow_nan=False))
            self.first = False

    def close(self):
        self.csv_file.close()
        self.json_file.write("]\n")
        self.json_file.close()
