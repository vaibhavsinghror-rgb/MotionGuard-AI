"""Data loading and tensor reshaping for MotionGuard AI (KU-HAR)."""

from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd

from motionguard.config import DataConfig
from motionguard.utils.logging import get_logger

logger = get_logger(__name__)


def load_raw_csv_as_arrays(
    csv_path: Path,
    config: Optional[DataConfig] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Loads KU-HAR CSV into structured (N, 6, 300) waveforms, labels (N,), and metadata.
    
    Layout according to KU-HAR specification:
      Col 0..299:    Acc X
      Col 300..599:  Acc Y
      Col 600..899:  Acc Z
      Col 900..1199: Gyro X
      Col 1200..1499: Gyro Y
      Col 1500..1799: Gyro Z
      Col 1800: Class ID (0..17)
      Col 1801: Length of each channel (300)
      Col 1802: Serial number
    """
    config = config or DataConfig.from_yaml()
    logger.info(f"Loading raw dataset from {csv_path}...")
    
    # Read without header assumption; check if first row is header
    with open(csv_path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
        tokens = first_line.split(",")
        has_header = False
        try:
            float(tokens[0])
        except ValueError:
            has_header = True

    header_arg = 0 if has_header else None
    df = pd.read_csv(csv_path, header=header_arg, low_memory=False)
    
    num_rows, num_cols = df.shape
    logger.info(f"Read {num_rows} rows x {num_cols} columns.")

    # Extract sensor matrix (first 1800 columns)
    sensor_vals = df.iloc[:, :1800].to_numpy(dtype=np.float32)

    # Impute any NaN or infinite values safely with forward/zero fill
    if np.isnan(sensor_vals).any() or np.isinf(sensor_vals).any():
        logger.warning("Replacing NaN/Inf values with 0.0 in sensor waveforms.")
        sensor_vals = np.nan_to_num(sensor_vals, nan=0.0, posinf=0.0, neginf=0.0)

    num_samples = sensor_vals.shape[0]
    expected_samples_per_ch = config.window_samples  # 300
    num_channels = config.num_channels  # 6

    # Reshape from (N, 1800) to (N, 6, 300)
    # Channels are concatenated sequentially: 300 acc_x, 300 acc_y, etc.
    waveforms = np.zeros((num_samples, num_channels, expected_samples_per_ch), dtype=np.float32)
    for c in range(num_channels):
        start_idx = c * expected_samples_per_ch
        end_idx = (c + 1) * expected_samples_per_ch
        waveforms[:, c, :] = sensor_vals[:, start_idx:end_idx]

    # Extract labels
    label_col_idx = 1800 if num_cols >= 1801 else num_cols - 1
    labels = df.iloc[:, label_col_idx].to_numpy(dtype=np.int64)

    # Serial number metadata if available
    serial_col_idx = 1802 if num_cols >= 1803 else None
    serials = df.iloc[:, serial_col_idx].to_numpy() if serial_col_idx is not None else None

    logger.info(f"Loaded waveforms shape: {waveforms.shape}, labels shape: {labels.shape}")
    return waveforms, labels, serials


def save_processed_arrays(
    waveforms: np.ndarray,
    labels: np.ndarray,
    output_dir: Path,
    serials: Optional[np.ndarray] = None,
) -> None:
    """Saves processed numpy arrays to disk for fast caching."""
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "waveforms.npy", waveforms)
    np.save(output_dir / "labels.npy", labels)
    if serials is not None:
        np.save(output_dir / "serials.npy", serials)
    logger.info(f"Saved processed arrays to {output_dir}")


def load_processed_arrays(processed_dir: Path) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Loads processed arrays from disk."""
    wf_path = processed_dir / "waveforms.npy"
    lb_path = processed_dir / "labels.npy"
    sr_path = processed_dir / "serials.npy"

    if not wf_path.exists() or not lb_path.exists():
        raise FileNotFoundError(f"Processed arrays not found in {processed_dir}. Run preprocessing first.")

    waveforms = np.load(wf_path)
    labels = np.load(lb_path)
    serials = np.load(sr_path) if sr_path.exists() else None
    return waveforms, labels, serials
