"""Data validation pipeline for MotionGuard AI (KU-HAR)."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from motionguard.config import DataConfig
from motionguard.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationReport:
    is_valid: bool = False
    row_count: int = 0
    column_count: int = 0
    expected_rows: int = 20750
    expected_cols: int = 1803
    num_classes_found: int = 0
    unique_classes: List[int] = field(default_factory=list)
    missing_values_count: int = 0
    infinite_values_count: int = 0
    duplicate_rows_count: int = 0
    class_distribution: Dict[int, int] = field(default_factory=dict)
    channel_statistics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    near_constant_channels: List[str] = field(default_factory=list)
    schema_errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, filepath: Path) -> None:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Validation report saved to {filepath}")


class DataValidator:
    """Validates the KU-HAR raw dataset against schema, integrity, and domain rules."""

    def __init__(self, config: Optional[DataConfig] = None):
        self.config = config or DataConfig.from_yaml()

    def validate_csv(self, csv_path: Path) -> ValidationReport:
        report = ValidationReport()
        logger.info(f"Starting rigorous validation of {csv_path}...")

        if not csv_path.exists():
            report.schema_errors.append(f"CSV file does not exist: {csv_path}")
            report.is_valid = False
            return report

        try:
            # First peek at header and column count
            with open(csv_path, "r", encoding="utf-8") as f:
                first_line = f.readline().strip()
                tokens = first_line.split(",")
                num_cols = len(tokens)
                has_header = False
                try:
                    float(tokens[0])
                except ValueError:
                    has_header = True

            logger.info(f"File layout peek: {num_cols} columns detected. Has string header: {has_header}")

            # Read CSV with pandas
            header_arg = 0 if has_header else None
            df = pd.read_csv(csv_path, header=header_arg, low_memory=False)
            
            # If the CSV has a header row or index col, normalize
            report.row_count, report.column_count = df.shape
            logger.info(f"Read dataframe of shape: {df.shape}")

            # Check column count
            # According to KU-HAR specification:
            # 1800 sensor points (6 * 300) + 1 class_id + 1 length + 1 serial_no = 1803 columns
            if report.column_count not in [1803, 1804]:
                if report.column_count < 1801:
                    report.schema_errors.append(
                        f"Expected at least 1801 columns (6x300 sensor samples + label), got {report.column_count}"
                    )
            
            # Identify label column (typically 1800 index if 0-based, or last column)
            # Col 1801 in 1-based index is index 1800
            label_col_idx = 1800 if report.column_count >= 1801 else report.column_count - 1
            labels = df.iloc[:, label_col_idx]

            # Coerce labels to integer
            if not pd.api.types.is_numeric_dtype(labels):
                try:
                    labels = pd.to_numeric(labels, errors="coerce")
                except Exception as e:
                    report.schema_errors.append(f"Label column could not be parsed as numeric: {e}")

            unique_labels = sorted([int(x) for x in labels.dropna().unique()])
            report.unique_classes = unique_labels
            report.num_classes_found = len(unique_labels)
            report.class_distribution = {int(k): int(v) for k, v in labels.value_counts().to_dict().items()}

            if report.num_classes_found != self.config.num_classes:
                msg = f"Expected {self.config.num_classes} classes (0-17), but found {report.num_classes_found}: {unique_labels}"
                if report.num_classes_found < 10:
                    report.schema_errors.append(msg)
                else:
                    report.warnings.append(msg)

            # Check missing and infinite values in sensor data (first 1800 columns)
            sensor_data = df.iloc[:, :1800]
            missing_count = int(sensor_data.isna().sum().sum())
            report.missing_values_count = missing_count
            if missing_count > 0:
                report.warnings.append(f"Found {missing_count} missing values in sensor readings.")

            # Infinite values check
            # Convert to numpy float32
            sensor_vals = sensor_data.to_numpy(dtype=np.float32, na_value=np.nan)
            inf_count = int(np.isinf(sensor_vals).sum())
            report.infinite_values_count = inf_count
            if inf_count > 0:
                report.schema_errors.append(f"Found {inf_count} infinite values in sensor matrix.")

            # Duplicate rows check
            dup_count = int(sensor_data.duplicated().sum())
            report.duplicate_rows_count = dup_count
            if dup_count > 0:
                report.warnings.append(f"Found {dup_count} duplicate sensor windows.")

            # Validate each of 6 channels (300 points each)
            channel_names = self.config.channels
            for ch_idx, ch_name in enumerate(channel_names):
                start_col = ch_idx * 300
                end_col = (ch_idx + 1) * 300
                ch_slice = sensor_vals[:, start_col:end_col]
                
                ch_mean = float(np.nanmean(ch_slice))
                ch_std = float(np.nanstd(ch_slice))
                ch_min = float(np.nanmin(ch_slice))
                ch_max = float(np.nanmax(ch_slice))

                report.channel_statistics[ch_name] = {
                    "mean": ch_mean,
                    "std": ch_std,
                    "min": ch_min,
                    "max": ch_max,
                }

                # Check for near-constant signal (std < 1e-5)
                if ch_std < 1e-5:
                    report.near_constant_channels.append(ch_name)
                    report.warnings.append(f"Channel {ch_name} has near-zero standard deviation ({ch_std:.2e}).")

            # Final validation check
            if len(report.schema_errors) == 0:
                report.is_valid = True
                logger.info("Data validation passed successfully.")
            else:
                report.is_valid = False
                logger.error(f"Data validation failed with errors: {report.schema_errors}")

            return report

        except Exception as e:
            report.schema_errors.append(f"Fatal error during validation: {str(e)}")
            report.is_valid = False
            return report
