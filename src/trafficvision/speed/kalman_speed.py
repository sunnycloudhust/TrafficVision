"""Kalman constant-velocity speed estimator in world (metre) space.

Measurement noise is propagated from pixel noise through the homography Jacobian, so
far-away (noisy) observations are automatically down-weighted. Gating is done on the
Mahalanobis distance of the innovation, not on instantaneous speed.
"""
import math
import numpy as np


class KalmanSpeedEstimator:
    def __init__(self, sigma_px=1.0, accel_std=2.0, gate_chi2=16.0, min_observations=8,
                 max_speed_std_kmh=4.0, max_gap_seconds=0.5, max_reasonable_kmh=160, max_rejects=3):
        self.sigma_px, self.q = sigma_px, accel_std
        self.gate, self.min_obs, self.max_std = gate_chi2, min_observations, max_speed_std_kmh / 3.6
        self.max_gap, self.vmax, self.max_rejects = max_gap_seconds, max_reasonable_kmh / 3.6, max_rejects
        self.state = {}

    def reset(self, track_id=None):
        self.state.clear() if track_id is None else self.state.pop(track_id, None)

    def prune(self, timestamp):
        for k in [k for k, s in self.state.items() if timestamp - s["t"] > self.max_gap]:
            self.state.pop(k)

    def _init(self, z, R, t):
        x = np.array([z[0], z[1], 0.0, 0.0])
        P = np.zeros((4, 4)); P[:2, :2] = R; P[2:, 2:] = np.eye(2) * (self.vmax / 2) ** 2
        return dict(x=x, P=P, t=t, n=1, rejects=0)

    def update(self, track_id, world, timestamp, jacobian=None):
        z = np.asarray(world, float)
        if z.shape != (2,) or not np.isfinite(z).all() or not math.isfinite(timestamp):
            self.reset(track_id); return None
        J = np.eye(2) if jacobian is None else np.asarray(jacobian, float)
        R = (self.sigma_px ** 2) * J @ J.T
        s = self.state.get(track_id)
        if s is None or not (0 < timestamp - s["t"] <= self.max_gap):
            self.state[track_id] = self._init(z, R, timestamp); return None
        dt = timestamp - s["t"]
        F = np.eye(4); F[0, 2] = F[1, 3] = dt
        G = np.array([[dt**2 / 2, 0], [0, dt**2 / 2], [dt, 0], [0, dt]])
        Q = (self.q ** 2) * G @ G.T
        x, P = F @ s["x"], F @ s["P"] @ F.T + Q
        Hm = np.eye(2, 4); S = Hm @ P @ Hm.T + R
        innov = z - Hm @ x
        if innov @ np.linalg.solve(S, innov) > self.gate:      # outlier / ID jump
            s.update(x=x, P=P, t=timestamp, rejects=s["rejects"] + 1)
            if s["rejects"] >= self.max_rejects:
                self.state[track_id] = self._init(z, R, timestamp)
            return None
        K = P @ Hm.T @ np.linalg.inv(S)
        x = x + K @ innov; P = (np.eye(4) - K @ Hm) @ P
        s.update(x=x, P=P, t=timestamp, n=s["n"] + 1, rejects=0)
        v = x[2:]; speed = float(np.linalg.norm(v))
        vstd = math.sqrt(max(P[2, 2], P[3, 3]))
        if s["n"] < self.min_obs or vstd > self.max_std or speed > self.vmax:
            return None
        return speed * 3.6


def pixel_jacobian(homography, point, eps=0.5):
    """2x2 Jacobian d(world_xy)/d(pixel_uv) at `point`, via central differences (metres per pixel)."""
    p = np.asarray(point, float)
    du = (homography.project(p + [eps, 0]) - homography.project(p - [eps, 0])) / (2 * eps)
    dv = (homography.project(p + [0, eps]) - homography.project(p - [0, eps])) / (2 * eps)
    return np.stack([du, dv], axis=1)


