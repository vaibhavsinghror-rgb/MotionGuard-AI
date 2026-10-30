"""Uncertainty and calibration metrics (ECE, Brier, NLL) for MotionGuard AI."""

from typing import Dict, List, Tuple
import numpy as np


def compute_expected_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> Tuple[float, float, List[Dict]]:
    """Computes Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).
    
    Args:
        probabilities: (N, num_classes) predicted softmax probabilities
        labels: (N,) true class indices
        n_bins: number of confidence bins
        
    Returns:
        ece: float
        mce: float
        bin_data: list of dicts with bin accuracy, confidence, and sample count
    """
    assert len(probabilities) == len(labels), "Length mismatch"
    confidences = np.max(probabilities, axis=1)
    predictions = np.argmax(probabilities, axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    bin_data = []

    N = len(labels)
    for b in range(n_bins):
        low, high = bin_boundaries[b], bin_boundaries[b + 1]
        mask = (confidences > low) & (confidences <= high) if b > 0 else (confidences >= low) & (confidences <= high)
        bin_count = int(np.sum(mask))

        if bin_count > 0:
            bin_acc = float(np.mean(accuracies[mask]))
            bin_conf = float(np.mean(confidences[mask]))
            diff = abs(bin_acc - bin_conf)
            ece += (bin_count / N) * diff
            mce = max(mce, diff)
        else:
            bin_acc = 0.0
            bin_conf = (low + high) / 2.0
            diff = 0.0

        bin_data.append({
            "bin": b,
            "range": [float(low), float(high)],
            "count": bin_count,
            "accuracy": bin_acc,
            "confidence": bin_conf,
            "calibration_error": diff,
        })

    return float(ece), float(mce), bin_data


def compute_brier_score(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Computes multi-class Brier score: mean squared error of predicted probabilities."""
    N, num_classes = probabilities.shape
    one_hot = np.zeros((N, num_classes), dtype=np.float32)
    one_hot[np.arange(N), labels] = 1.0
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def compute_nll(probabilities: np.ndarray, labels: np.ndarray, eps: float = 1e-12) -> float:
    """Computes categorical Negative Log-Likelihood."""
    N = len(labels)
    probs_clamped = np.clip(probabilities[np.arange(N), labels], eps, 1.0)
    return float(-np.mean(np.log(probs_clamped)))
