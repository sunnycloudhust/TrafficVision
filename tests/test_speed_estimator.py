import pytest
from trafficvision.speed.estimator import SpeedEstimator
from trafficvision.evaluation.speed_metrics import speed_metrics


def test_known_velocity_timestamps_gaps_and_jumps():
    estimator = SpeedEstimator(max_gap_seconds=1.0)
    assert estimator.update(1, [0, 0], 0) is None
    assert estimator.update(1, [1, 0], 0.1) is None
    assert estimator.update(1, [3, 0], 0.3) == pytest.approx(36)
    assert estimator.update(1, [4, 0], 0.3) is None  # duplicate timestamp
    assert estimator.update(1, [1000, 0], 0.4) is None  # impossible ID jump
    assert estimator.update(1, [1001, 0], 3.0) is None  # long observation gap
    assert estimator.update(2, [0, 0], 0) is None


def test_speed_metric_zero_ground_truth():
    result = speed_metrics([0, 10], [1, 12])
    assert result["mae"] == 1.5 and result["mape"] == 20

def test_kalman_constant_speed_and_jump_rejection():
    from trafficvision.speed.kalman_speed import KalmanSpeedEstimator
    est = KalmanSpeedEstimator(sigma_px=0.1)
    out = [est.update(1, [0.0, 10.0 * i / 25], i / 25) for i in range(30)]  # 10 m/s
    assert out[2] is None                      # chưa đủ quan sát
    assert out[-1] == pytest.approx(36, abs=0.5)
    assert est.update(1, [0.0, 500.0], 30 / 25) is None  # nhảy ID bị loại