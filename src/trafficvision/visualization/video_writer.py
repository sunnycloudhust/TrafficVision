from pathlib import Path
import cv2


class VideoWriter:
    def __init__(self, path, fps, frame_size):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, tuple(frame_size))
        if not self.writer.isOpened():
            raise RuntimeError(f"Cannot create MP4 writer: {path}")

    def write(self, frame):
        self.writer.write(frame)

    def close(self):
        self.writer.release()
