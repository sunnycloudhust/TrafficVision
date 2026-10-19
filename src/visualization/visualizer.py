"""Draw model classes, confidence and persistent track IDs."""
import cv2


def draw_tracks(frame, tracks, class_names):
    for item in tracks:
        x1, y1, x2, y2 = map(int, item.box)
        label = item.class_id - 1
        name = class_names[label] if 0 <= label < len(class_names) else str(item.class_id)
        caption = f"{name} {item.score:.2f} id:{item.track_id}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 220, 80), 2)
        cv2.putText(frame, caption, (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, .5, (40, 220, 80), 1, cv2.LINE_AA)
    return frame
