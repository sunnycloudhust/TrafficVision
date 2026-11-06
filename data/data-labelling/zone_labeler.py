"""Interactive polygon-zone labeler for UA-DETRAC camera frames."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml


WINDOW_NAME = "TrafficVision zone labeler"
COLORS = ((0, 180, 255), (0, 230, 90), (255, 180, 0), (220, 80, 220))
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FRAME_DIR = (
    REPO_ROOT / "data" / "ua_detrac" / "raw" / "DETRAC" / "DETRAC-Images"
    / "DETRAC-Images"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "configs" / "zones"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


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
        return [
            min(max(round(x / self.display_scale), 0), self.width - 1),
            min(max(round(y / self.display_scale), 0), self.height - 1),
        ]

    def on_click(self, event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current.append(self.to_frame_point(x, y))

    def save(self):
        if self.current:
            raise ValueError("Finish the current polygon with Enter before saving.")
        if not self.zones:
            raise ValueError("Create at least one zone before saving.")
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

    def finish_zone(self, name):
        name = name.strip()
        if not name or name in {zone["name"] for zone in self.zones}:
            raise ValueError("Zone name must be nonempty and unique.")
        points = np.asarray(self.current, dtype=np.float32).reshape(-1, 2)
        if (
            len(points) < 3
            or len(np.unique(points, axis=0)) != len(points)
            or abs(cv2.contourArea(points)) < 1e-6
        ):
            raise ValueError("A zone needs >=3 distinct points with nonzero area.")
        self.zones.append({"name": name, "polygon": self.current.copy()})
        self.current.clear()

    def run(self):
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, self.on_click)
        print("Click points; Enter=finish zone, U=undo, C=clear current, " "S=save, Q/Esc=quit.")
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
                try:
                    self.finish_zone(name)
                except ValueError as exc:
                    print(exc)
            elif key == ord("s"):
                try:
                    self.save()
                except ValueError as exc:
                    print(exc)
        cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    return parser.parse_args()


def discover_frames(frame_dir):
    """Pick the first available image from each camera sequence directory."""
    frame_dir = Path(frame_dir)
    if not frame_dir.is_dir():
        raise FileNotFoundError(f"Frame directory does not exist: {frame_dir}")
    frames = []
    for camera_dir in sorted(path for path in frame_dir.iterdir() if path.is_dir()):
        images = sorted(
            path for path in camera_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if images:
            frames.append(images[0])
    return frames


def has_zones(config_path):
    """Return whether a saved zone config contains at least one zone."""
    config_path = Path(config_path)
    if not config_path.is_file():
        return False
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return False
    if not isinstance(config, dict):
        return False
    zones = config.get("zones")
    return isinstance(zones, dict) and bool(zones)


def choose_frame(frame_dir, output_dir):
    frames = discover_frames(frame_dir)
    if not frames:
        raise FileNotFoundError(f"No camera frames found in: {frame_dir}")
    print("Chọn camera/frame đại diện để kẻ zone:")
    for index, frame in enumerate(frames, 1):
        status = " --done" if has_zones(Path(output_dir) / f"{frame.parent.name}.yaml") else ""
        print(f"  {index}. {frame.parent.name}{status}")
    while True:
        answer = input(f"Nhập số camera (1-{len(frames)}): ").strip()
        try:
            index = int(answer)
        except ValueError:
            print("Vui lòng nhập một số hợp lệ.")
            continue
        if 1 <= index <= len(frames):
            return frames[index - 1]
        print(f"Số camera phải nằm trong khoảng 1-{len(frames)}.")


def main():
    parse_args()
    source = choose_frame(DEFAULT_FRAME_DIR, DEFAULT_OUTPUT_DIR)
    camera_id = source.parent.name
    output = DEFAULT_OUTPUT_DIR / f"{camera_id}.yaml"
    print(f"Dùng frame đại diện: {source}")
    print(f"File zone sẽ lưu tại: {output}")
    frame = cv2.imread(str(source))
    if frame is None:
        raise ValueError(f"Cannot read image frame: {source}")
    ZoneLabeler(frame, camera_id, output).run()


if __name__ == "__main__":
    main()
