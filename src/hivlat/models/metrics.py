from __future__ import annotations
import numpy as np

def expected_calibration_error_multiclass(proba: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    """ECE using max-probability confidence and 0/1 correctness of argmax."""
    conf = proba.max(axis=1)
    preds = proba.argmax(axis=1)
    correct = (preds == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins+1)
    ece = 0.0
    for i in range(n_bins):
        mask = (conf >= bins[i]) & (conf < bins[i+1])
        if mask.sum() == 0:
            continue
        acc = correct[mask].mean()
        avg_conf = conf[mask].mean()
        ece += (mask.mean()) * abs(acc - avg_conf)
    return float(ece)
