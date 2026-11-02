from .homography import Homography


def load_calibration(config):
    if config.get("world_unit", "meter") != "meter":
        raise ValueError("World coordinates must be in meters")
    matrix = config.get("homography", {}).get("matrix")
    return Homography(matrix) if matrix is not None else None
