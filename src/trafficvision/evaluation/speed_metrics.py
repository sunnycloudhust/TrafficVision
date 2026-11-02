import numpy as np


def speed_metrics(truth, predicted):
    truth, predicted = np.asarray(truth, dtype=float), np.asarray(predicted, dtype=float)
    if truth.shape != predicted.shape or truth.ndim != 1 or len(truth) == 0:
        raise ValueError("Need nonempty aligned 1D arrays of matched ground-truth and predictions")
    if not np.isfinite(truth).all() or not np.isfinite(predicted).all() or (truth < 0).any():
        raise ValueError("Speed metrics require finite, nonnegative ground-truth speeds")
    error = predicted - truth
    nonzero = truth != 0
    return dict(
        mae=float(abs(error).mean()),
        rmse=float(np.sqrt((error**2).mean())),
        mape=float((abs(error[nonzero]) / truth[nonzero]).mean() * 100) if nonzero.any() else None,
        observations=len(truth),
        mape_observations=int(nonzero.sum()),
    )
