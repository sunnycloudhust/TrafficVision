"""Headless calibration from point pairs, or click image points with --interactive.

world-points JSON is always measured coordinates in meters; no invented distances.
"""

import argparse
import json
from pathlib import Path
import sys

import cv2
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from trafficvision.geometry.homography import Homography  


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", required=True)
    p.add_argument("--image-points", help="JSON [[u,v], ...]")
    p.add_argument("--world-points", required=True, help="JSON [[X_m,Y_m], ...] in matching order")
    p.add_argument("--interactive", action="store_true")
    p.add_argument("--camera-id", required=True)
    p.add_argument("--fps", type=float, default=25)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(args.image)
    world = json.loads(Path(args.world_points).read_text())
    points = []
    if args.interactive:

        def click(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append([x, y])
                cv2.circle(image, (x, y), 4, (0, 255, 0), -1)
                cv2.putText(image, str(len(points)), (x + 5, y), 0, 0.5, (0, 255, 0), 1)

        try:
            cv2.namedWindow("calibration: click in world-point order, Enter to finish, Esc to cancel")
            window = "calibration: click in world-point order, Enter to finish, Esc to cancel"
            cv2.setMouseCallback(window, click)
            while True:
                cv2.imshow(window, image)
                key = cv2.waitKey(30) & 255
                if key == 27:
                    raise ValueError("Calibration cancelled")
                if key == 13:
                    break
        finally:
            cv2.destroyAllWindows()
    elif args.image_points:
        points = json.loads(Path(args.image_points).read_text())
    else:
        p.error("Provide --image-points or --interactive (desktop OpenCV required)")
    homography = Homography.from_points(points, world)
    config = dict(
        camera_id=args.camera_id,
        fps=args.fps,
        frame_size=[image.shape[1], image.shape[0]],
        world_unit="meter",
        homography=dict(matrix=homography.matrix.tolist()),
        calibration=dict(
            image_points=points, world_points=world, reprojection_error_m=homography.reprojection_error_m
        ),
        speed=dict(smoothing_window=5, min_track_observations=3, max_reasonable_kmh=160),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(config, sort_keys=False))
    print(
        f"Saved {output}; fit error {homography.reprojection_error_m:.5f} m (validate on independent points)"
    )


if __name__ == "__main__":
    main()
