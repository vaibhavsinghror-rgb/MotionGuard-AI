"""Unfamiliar movement pattern and anomaly detection for MotionGuard AI."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from motionguard.utils.logging import get_logger

logger = get_logger(__name__)


class MovementAnomalyDetector:
    """Detects unfamiliar or out-of-distribution motion patterns using Isolation Forest on engineered features.
    
    LIMITATION NOTICE: This module detects statistical deviation from normal training activities.
    It does NOT provide clinical fall detection or certified medical/workplace safety verification.
    """

    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 150,
        random_state: int = 42,
    ):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.detector = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )
        self.is_fitted = False

    def fit(self, features: np.ndarray) -> "MovementAnomalyDetector":
        """Fits normal baseline patterns on in-distribution training features."""
        logger.info(f"Fitting MovementAnomalyDetector on {features.shape[0]} feature vectors...")
        scaled_x = self.scaler.fit_transform(features)
        self.detector.fit(scaled_x)
        self.is_fitted = True
        return self

    def score_samples(self, features: np.ndarray) -> np.ndarray:
        """Computes anomaly scores (lower means more anomalous; invert so higher = more abnormal)."""
        if not self.is_fitted:
            raise RuntimeError("Anomaly detector must be fitted before scoring!")
        scaled_x = self.scaler.transform(features)
        # Decision function is positive for inliers, negative for outliers
        # Invert so higher score represents higher anomalousness
        raw_scores = -self.detector.decision_function(scaled_x)
        # Normalize roughly to 0..1 range using sigmoid or minmax
        return raw_scores.astype(np.float32)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Returns 1 for normal patterns, -1 for anomalous / unfamiliar patterns."""
        if not self.is_fitted:
            raise RuntimeError("Anomaly detector must be fitted before predicting!")
        scaled_x = self.scaler.transform(features)
        return self.detector.predict(scaled_x)

    def save(self, filepath: Path) -> None:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        logger.info(f"Saved MovementAnomalyDetector to {filepath}")

    @classmethod
    def load(cls, filepath: Path) -> "MovementAnomalyDetector":
        return joblib.load(filepath)
