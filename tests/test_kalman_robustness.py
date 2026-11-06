import numpy as np
import pytest
from trafficvision.geometry.homography import Homography
from trafficvision.speed.kalman_speed import KalmanSpeedEstimator, pixel_jacobian


@pytest.mark.parametrize(
    "params",
    [
        dict(sigma_px=0),
        dict(sigma_px=float("nan")),
        dict(accel_std=-1),
        dict(gate_chi2=0),
        dict(max_gap_seconds=0),
        dict(min_observations=1),
        dict(max_rejects=0),
        dict(max_speed_std_kmh=float("inf")),
        dict(max_reasonable_kmh=-1),
        dict(min_observations=2.5),
    ],
)
def test_invalid_config_fails_early(params):
    with pytest.raises(ValueError):
        KalmanSpeedEstimator(**params)


@pytest.mark.parametrize("jacobian", [np.zeros((2, 2)), np.ones((3, 2)), [[1, 0], [0, float("nan")]]])
def test_bad_jacobian_resets_track(jacobian):
    estimator = KalmanSpeedEstimator()
    estimator.update(1, [0, 0], 0)
    with pytest.raises(ValueError, match="Jacobian"):
        estimator.update(1, [1, 0], 0.04, jacobian)
    assert 1 not in estimator.state


def test_perspective_jacobian_matches_small_step_difference():
    h = Homography([[0.3, 0.1, 2], [0.05, 0.5, 4], [0.001, 0.002, 1]])
    point = np.array([200.0, 100.0])
    epsilon = 1e-4
    expected = np.column_stack(
        [
            (h.project(point + axis * epsilon) - h.project(point - axis * epsilon)) / (2 * epsilon)
            for axis in np.eye(2)
        ]
    )
    np.testing.assert_allclose(pixel_jacobian(h, point), expected, rtol=1e-7, atol=1e-8)
    with pytest.raises(ValueError, match="horizon"):
        Homography([[1, 0, 0], [0, 1, 0], [1, 0, 1]]).jacobian([-1, 0])


def test_covariance_remains_symmetric_psd_with_correlated_noise():
    estimator = KalmanSpeedEstimator(sigma_px=0.2)
    J = np.array([[0.1, 0.04], [0.04, 0.2]])
    for i in range(150):
        estimator.update(1, [i * 0.08, i * 0.16], i * 0.04, J)
        covariance = estimator.state[1]["P"]
        np.testing.assert_allclose(covariance, covariance.T, atol=1e-12)
        assert np.linalg.eigvalsh(covariance).min() >= -1e-12


def test_rejected_observations_do_not_keep_old_track_alive():
    estimator = KalmanSpeedEstimator(max_rejects=100, max_gap_seconds=0.15)
    for i in range(30):
        estimator.update(1, [i * 0.4, 0], i * 0.04)
    for i in range(30, 33):
        assert estimator.update(1, [1000, 0], i * 0.04) is None
    estimator.prune(1.32)
    assert 1 not in estimator.state


def test_gap_and_duplicate_timestamp_restart_warmup():
    estimator = KalmanSpeedEstimator(sigma_px=0.1)
    for i in range(30):
        estimator.update(1, [i * 0.4, 0], i * 0.04)
    assert estimator.update(1, [11.6, 0], 1.16) is None
    assert estimator.state[1]["n"] == 1
    assert estimator.update(1, [30, 0], 3) is None
    assert estimator.state[1]["n"] == 1
