"""UA-DETRAC XML: one-based frame indices, xywh -> xyxy, stable track IDs.

Labels are 1..4; zero is exclusively the detector background class.
Coordinates retain annotation continuous pixel coordinates (no silent -1).
"""

from pathlib import Path
import math
import xml.etree.ElementTree as ET

CLASS_NAMES = ("car", "bus", "van", "others")
CLASS_MAP = {name: index + 1 for index, name in enumerate(CLASS_NAMES)}


def box_xyxy(element):
    x, y, w, h = (float(element.attrib[k]) for k in ("left", "top", "width", "height"))
    if not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0:
        raise ValueError(f"Invalid bounding box: {element.attrib}")
    return [x, y, x + w, y + h]


def parse_sequence(path, unknown_class="error"):
    if unknown_class not in ("error", "others"):
        raise ValueError("unknown_class must be error or others")
    root = ET.parse(path).getroot()
    video_id = root.get("name", Path(path).stem)
    ignored = [box_xyxy(e) for e in root.findall("./ignored_region/box")]
    frames = []
    seen = set()
    for frame in root.findall("frame"):
        frame_id = int(frame.attrib["num"])
        if frame_id < 1 or frame_id in seen:
            raise ValueError(f"Invalid/duplicate frame number {frame_id} in {video_id}")
        seen.add(frame_id)
        boxes, labels, ids, attributes = [], [], [], []
        for target in frame.findall("./target_list/target"):
            attr = target.find("attribute")
            attr = {} if attr is None else dict(attr.attrib)
            name = attr.get("vehicle_type", "").lower()
            if name not in CLASS_MAP:
                if unknown_class == "error":
                    raise ValueError(f"Unknown vehicle_type={name!r} in {path}; choose an explicit mapping")
                name = "others"
            boxes.append(box_xyxy(target.find("box")))
            labels.append(CLASS_MAP[name])
            ids.append(int(target.attrib["id"]))
            attributes.append(attr)
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate track ID in {video_id}:{frame_id}")
        frames.append(
            dict(
                video_id=video_id,
                frame_id=frame_id,
                boxes=boxes,
                labels=labels,
                track_ids=ids,
                ignored_regions=ignored,
                attributes=attributes,
            )
        )
    return sorted(frames, key=lambda f: f["frame_id"])
