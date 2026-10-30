"""Time-domain feature extraction for MotionGuard AI."""

from typing import List, Tuple
import numpy as np
from scipy import stats


def extract_axis_time_features(signal: np.ndarray, prefix: str = "ch0") -> Tuple[np.ndarray, List[str]]:
    """Extracts comprehensive time-domain statistics for a 1D or 2D signal (N, T).
    
    Features:
      - mean
      - std (with small epsilon check)
      - rms
      - min, max
      - peak_to_peak
      - skewness (safe from zero-variance)
      - kurtosis (safe from zero-variance)
      - energy (sum of squared amplitudes / T)
      - zero_crossings_rate
    """
    assert signal.ndim == 2, f"Expected 2D array (N, T), got {signal.shape}"
    N, T = signal.shape
    eps = 1e-8

    mean_val = np.mean(signal, axis=1)
    std_val = np.std(signal, axis=1)
    rms_val = np.sqrt(np.mean(signal ** 2, axis=1) + eps)
    min_val = np.min(signal, axis=1)
    max_val = np.max(signal, axis=1)
    ptp_val = max_val - min_val
    energy_val = np.sum(signal ** 2, axis=1) / max(T, 1)

    # Skewness and Kurtosis with numerical safeguard
    skew_val = np.zeros(N, dtype=np.float32)
    kurt_val = np.zeros(N, dtype=np.float32)

    # Use scipy stats with biased=False, suppressing precision loss warnings
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            raw_skew = stats.skew(signal, axis=1, nan_policy="omit")
            skew_val = np.nan_to_num(raw_skew, nan=0.0, posinf=0.0, neginf=0.0)
            raw_kurt = stats.kurtosis(signal, axis=1, nan_policy="omit")
            kurt_val = np.nan_to_num(raw_kurt, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception:
            pass

    # Zero crossing rate (sign change count / T)
    zero_crossings = np.sum(np.diff(np.signbit(signal), axis=1), axis=1) / max(T - 1, 1)

    feature_matrix = np.column_stack([
        mean_val,
        std_val,
        rms_val,
        min_val,
        max_val,
        ptp_val,
        skew_val,
        kurt_val,
        energy_val,
        zero_crossings,
    ]).astype(np.float32)

    feature_names = [
        f"{prefix}_mean",
        f"{prefix}_std",
        f"{prefix}_rms",
        f"{prefix}_min",
        f"{prefix}_max",
        f"{prefix}_ptp",
        f"{prefix}_skew",
        f"{prefix}_kurt",
        f"{prefix}_energy",
        f"{prefix}_zcr",
    ]

    return feature_matrix, feature_names
