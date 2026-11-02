import cv2
import numpy as np
from ..data.parsers.xml_parser import CLASS_NAMES


def annotate(frame, results, zones, fps=None):
    image = frame.copy()
    for zone in zones.zones:
        polygon = zone.polygon.astype(np.int32)
        cv2.polylines(image, [polygon], True, (255, 180, 0), 2)
        origin = tuple(polygon[0])
        cv2.putText(image, zone.name, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 180, 0), 2)
    for result in results:
        track = result["track"]
        x1, y1, x2, y2 = map(int, track.box)
        speed = result["speed_kmh"]
        speed_text = "N/A" if speed is None else f"{speed:.1f} km/h"
        name = CLASS_NAMES[track.label - 1] if 1 <= track.label <= len(CLASS_NAMES) else str(track.label)
        text = f"#{track.track_id} {name} {speed_text}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 230, 90), 2)
        cv2.putText(
            image, text, (max(0, x1), max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 230, 90), 2
        )
    if fps is not None:
        cv2.putText(image, f"processing {fps:.1f} FPS", (12, 24), 0, 0.6, (255, 255, 255), 2)
    return image
