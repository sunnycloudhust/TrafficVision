import numpy as np


def bottom_center(box):
    x1, _, x2, y2 = box
    return np.array([(x1 + x2) / 2, y2], dtype=float)
