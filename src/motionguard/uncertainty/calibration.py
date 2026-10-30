"""Temperature scaling and confidence calibration for MotionGuard AI."""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax

from motionguard.uncertainty.metrics import (
    compute_brier_score,
    compute_expected_calibration_error,
    compute_nll,
)
from motionguard.utils.logging import get_logger

logger = get_logger(__name__)


class TemperatureScaler:
    """Post-hoc confidence calibration using Temperature Scaling on validation logits."""

    def __init__(self, initial_temperature: float = 1.0):
        self.temperature: float = float(initial_temperature)
        self.is_fitted: bool = False
        self.pre_calibration_metrics: Dict = {}
        self.post_calibration_metrics: Dict = {}

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> "TemperatureScaler":
        """Fits temperature T > 0 by minimizing NLL on validation logits."""
        assert logits.ndim == 2, f"Expected 2D logits, got {logits.shape}"
        assert len(logits) == len(labels), "Length mismatch"

        # Baseline metrics with T = 1.0
        base_probs = softmax(logits, axis=1)
        base_ece, base_mce, base_bins = compute_expected_calibration_error(base_probs, labels)
        base_brier = compute_brier_score(base_probs, labels)
        base_nll = compute_nll(base_probs, labels)

        self.pre_calibration_metrics = {
            "ece": base_ece,
            "mce": base_mce,
            "brier": base_brier,
            "nll": base_nll,
            "temperature": 1.0,
        }

        # Optimization objective: NLL as function of temperature T
        def objective(t_arr: np.ndarray) -> float:
            t = t_arr[0]
            if t <= 1e-4:
                return 1e6
            scaled_logits = logits / t
            # Stable log-sum-exp
            max_logits = np.max(scaled_logits, axis=1, keepdims=True)
            log_sum_exp = max_logits.ravel() + np.log(np.sum(np.exp(scaled_logits - max_logits), axis=1))
            true_logits = scaled_logits[np.arange(len(labels)), labels]
            nll = np.mean(log_sum_exp - true_logits)
            return float(nll)

        res = minimize(objective, x0=np.array([1.0]), bounds=[(0.05, 10.0)], method="L-BFGS-B")
        self.temperature = float(res.x[0])
        self.is_fitted = True

        # Post-calibration metrics
        cal_probs = self.calibrate(logits)
        cal_ece, cal_mce, cal_bins = compute_expected_calibration_error(cal_probs, labels)
        cal_brier = compute_brier_score(cal_probs, labels)
        cal_nll = compute_nll(cal_probs, labels)

        self.post_calibration_metrics = {
            "ece": cal_ece,
            "mce": cal_mce,
            "brier": cal_brier,
            "nll": cal_nll,
            "temperature": self.temperature,
            "bins": cal_bins,
        }

        logger.info(
            f"Fitted Temperature: {self.temperature:.4f}. "
            f"ECE: {base_ece:.4f} -> {cal_ece:.4f}, NLL: {base_nll:.4f} -> {cal_nll:.4f}"
        )
        return self

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """Scales logits and applies softmax."""
        t = self.temperature if self.is_fitted else 1.0
        return softmax(logits / t, axis=1)

    def to_dict(self) -> Dict:
        return {
            "temperature": self.temperature,
            "is_fitted": self.is_fitted,
            "pre_calibration": self.pre_calibration_metrics,
            "post_calibration": self.post_calibration_metrics,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "TemperatureScaler":
        scaler = cls(initial_temperature=data.get("temperature", 1.0))
        scaler.is_fitted = data.get("is_fitted", False)
        scaler.pre_calibration_metrics = data.get("pre_calibration", {})
        scaler.post_calibration_metrics = data.get("post_calibration", {})
        return scaler

    def save(self, filepath: Path) -> None:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Calibration metadata saved to {filepath}")

    @classmethod
    def load(cls, filepath: Path) -> "TemperatureScaler":
        with open(filepath, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
