import numpy as np
import pytest
from trafficvision.geometry.homography import Homography


def test_known_metric_projection():
    h = Homography.from_points([[0, 0], [100, 0], [100, 100], [0, 100]], [[0, 0], [10, 0], [10, 20], [0, 20]])
    np.testing.assert_allclose(h.project([50, 50]), [5, 10], atol=1e-6)
    with pytest.raises(ValueError):
        Homography.from_points([[0, 0], [1, 1], [2, 2], [3, 3]], [[0, 0], [1, 1], [2, 2], [3, 3]])
    with pytest.raises(ValueError):
        Homography(np.zeros((3, 3)))
