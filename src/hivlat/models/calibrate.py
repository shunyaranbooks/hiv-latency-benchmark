from __future__ import annotations
import numpy as np
from sklearn.calibration import CalibratedClassifierCV

def calibrate_sklearn(model, method='isotonic', cv=5):
    """Wrap a scikit-learn classifier for probability calibration."""
    return CalibratedClassifierCV(model, method=method, cv=cv)

def conformal_thresholds(proba_calib: np.ndarray, y_calib: np.ndarray, alpha=0.1):
    """Split-conformal for multiclass: returns per-class q-hat on nonconformity scores.
    Nonconformity: 1 - p_trueclass.
    """
    K = proba_calib.shape[1]
    scores = []
    for i in range(proba_calib.shape[0]):
        c = y_calib[i]
        scores.append(1.0 - proba_calib[i, c])
    scores = np.array(scores)
    # Single global threshold (simple variant)
    qhat = np.quantile(scores, 1 - alpha, interpolation='higher')
    return float(qhat)

def conformal_coverage(proba: np.ndarray, y_true: np.ndarray, qhat: float):
    sets = (1.0 - proba) <= qhat  # cells x classes boolean
    covered = [sets[i, y_true[i]] for i in range(len(y_true))]
    avg_set_size = sets.sum(axis=1).mean()
    return float(np.mean(covered)), float(avg_set_size)
