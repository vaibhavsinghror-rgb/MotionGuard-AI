"""Waveform preprocessing and normalization for MotionGuard AI."""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple
import joblib
import numpy as np

from motionguard.utils.logging import get_logger

logger = get_logger(__name__)


class WaveformNormalizer:
    """Channel-wise standardizer for 6-channel sensor waveforms (N, 6, T).
    
    Fits strictly on training data: computes channel means and stds across (N_train, T).
    """

    def __init__(self, eps: float = 1e-7):
        self.eps = eps
        self.channel_means: Optional[np.ndarray] = None  # shape: (6, 1)
        self.channel_stds: Optional[np.ndarray] = None   # shape: (6, 1)
        self.is_fitted: bool = False

    def fit(self, waveforms: np.ndarray) -> "WaveformNormalizer":
        """Fits mean and std per channel from training waveforms (N, 6, T)."""
        assert waveforms.ndim == 3, f"Expected 3D waveforms (N, C, T), got {waveforms.shape}"
        num_channels = waveforms.shape[1]
        
        # Compute mean and std per channel across all windows and timepoints
        # waveforms is (N, C, T) -> reshape to (C, N*T)
        transposed = waveforms.transpose(1, 0, 2).reshape(num_channels, -1)
        means = np.mean(transposed, axis=1, keepdims=True)  # (C, 1)
        stds = np.std(transposed, axis=1, keepdims=True)    # (C, 1)
        
        # Avoid division by zero
        stds = np.where(stds < self.eps, 1.0, stds)

        self.channel_means = means.astype(np.float32)
        self.channel_stds = stds.astype(np.float32)
        self.is_fitted = True
        logger.info(f"Fitted WaveformNormalizer across {waveforms.shape[0]} windows. Channel means: {self.channel_means.ravel()}, stds: {self.channel_stds.ravel()}")
        return self

    def transform(self, waveforms: np.ndarray) -> np.ndarray:
        """Applies normalization: (x - mean) / std per channel."""
        if not self.is_fitted:
            raise RuntimeError("WaveformNormalizer must be fitted before calling transform!")
        assert waveforms.ndim == 3, f"Expected 3D waveforms (N, C, T), got {waveforms.shape}"
        
        # Expand channel_means and stds to (1, C, 1) for broadcasting over (N, C, T)
        means = self.channel_means.reshape(1, -1, 1)
        stds = self.channel_stds.reshape(1, -1, 1)
        
        normalized = (waveforms - means) / stds
        return normalized.astype(np.float32)

    def fit_transform(self, waveforms: np.ndarray) -> np.ndarray:
        return self.fit(waveforms).transform(waveforms)

    def to_dict(self) -> Dict:
        return {
            "channel_means": self.channel_means.tolist() if self.channel_means is not None else [],
            "channel_stds": self.channel_stds.tolist() if self.channel_stds is not None else [],
            "eps": self.eps,
            "is_fitted": self.is_fitted,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "WaveformNormalizer":
        norm = cls(eps=data.get("eps", 1e-7))
        if data.get("is_fitted"):
            norm.channel_means = np.array(data["channel_means"], dtype=np.float32)
            norm.channel_stds = np.array(data["channel_stds"], dtype=np.float32)
            norm.is_fitted = True
        return norm

    def save(self, filepath: Path) -> None:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        if filepath.suffix == ".json":
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2)
        else:
            joblib.dump(self, filepath)
        logger.info(f"Saved WaveformNormalizer to {filepath}")

    @classmethod
    def load(cls, filepath: Path) -> "WaveformNormalizer":
        if not filepath.exists():
            raise FileNotFoundError(f"Normalizer file not found: {filepath}")
        if filepath.suffix == ".json":
            with open(filepath, "r", encoding="utf-8") as f:
                return cls.from_dict(json.load(f))
        return joblib.load(filepath)
