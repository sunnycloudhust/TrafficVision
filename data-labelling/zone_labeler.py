"""Interactive polygon-zone labeler for TrafficVision video files."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml


WINDOW_NAME = "TrafficVision zone labeler"
COLORS = ((0, 180, 255), (0, 230, 90), (255, 180, 0), (220, 80, 220))


class ZoneLabeler:
    def __init__(self, frame, camera_id, output):
        self.frame = frame
        self.camera_id = camera_id
        self.output = Path(output)
        self.height, self.width = frame.shape[:2]
        self.zones = []
        self.current = []
        self.display_scale = 1.0
        self.display_size = (self.width, self.height)

    def display_frame(self):
        max_width, max_height = 1400, 850
        self.display_scale = min(max_width / self.width, max_height / self.height, 1.0)
        self.display_size = (
            max(1, round(self.width * self.display_scale)),
            max(1, round(self.height * self.display_scale)),
        )
        image = cv2.resize(self.frame, self.display_size, interpolation=cv2.INTER_AREA)
        for index, zone in enumerate(self.zones):
            points = zone["polygon"]
            polygon = self.to_display_points(points)
            color = COLORS[index % len(COLORS)]
            cv2.polylines(image, [polygon], True, color, 2)
            for point_index, (x, y) in enumerate(polygon):
                cv2.circle(image, (x, y), 5, color, -1)
                cv2.putText(
                    image,
                    str(point_index + 1),
                    (x + 7, y - 7),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                )
            cv2.putText(
                image,
                zone["name"],
                tuple(polygon[0]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        if self.current:
            points = self.to_display_points(self.current)
            cv2.polylines(image, [points], False, (255, 255, 255), 2)
            for point in points:
                cv2.circle(image, tuple(point), 5, (255, 255, 255), -1)

        return image

    def to_display_points(self, points):
        return np.asarray(
            [[round(x * self.display_scale), round(y * self.display_scale)] for x, y in points],
            dtype=np.int32,
        )

    def to_frame_point(self, x, y):
        return [round(x / self.display_scale), round(y / self.display_scale)]

    def on_click(self, event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current.append(self.to_frame_point(x, y))

    def save(self):
        config = {
            "camera_id": self.camera_id,
            "frame_size": [self.width, self.height],
            "roi_only": True,
            "default_threshold": 0.25,
            "zones": {
                zone["name"]: {
                    "polygon": zone["polygon"],
                    "confidence_threshold": 0.05,
                    "inference_scale": 1.0,
                    "measure_speed": True,
                }
                for zone in self.zones
            },
        }
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        print(f"Saved {len(self.zones)} zones to {self.output}")

    def run(self):
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, self.on_click)
        print(
            "Click points; Enter=finish zone, U=undo, C=clear current, "
            "S=save, Q/Esc=quit."
        )
        while True:
            cv2.imshow(WINDOW_NAME, self.display_frame())
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("u") and self.current:
                self.current.pop()
            elif key == ord("c"):
                self.current.clear()
            elif key in (10, 13):
                if len(self.current) < 3:
                    print("A zone needs at least 3 points.")
                    continue
                name = input(f"Zone name ({len(self.zones) + 1}): ").strip()
                if not name:
                    print("Zone name cannot be empty.")
                    continue
                if name in {zone["name"] for zone in self.zones}:
                    print(f"Zone already exists: {name}")
                    continue
                self.zones.append({"name": name, "polygon": self.current.copy()})
                self.current.clear()
            elif key == ord("s"):
                self.save()
        cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Video file used for labelling")
    parser.add_argument("--output", required=True, help="Output zone YAML path")
    parser.add_argument("--camera-id", help="Camera ID; defaults to the video filename stem")
    return parser.parse_args()


def main():
    args = parse_args()
    source = Path(args.source)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open video source: {source}")
    try:
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok:
        raise ValueError(f"Video has no readable frames: {source}")
    ZoneLabeler(frame, args.camera_id or source.stem, args.output).run()


if __name__ == "__main__":
    main()
