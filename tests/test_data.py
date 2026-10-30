"""Automated tests for data loading, schema, and manifest."""

import numpy as np
import pytest
from motionguard.config import DataConfig, PROJECT_ROOT
from motionguard.data.validate import DataValidator


def test_data_config_loading():
    config = DataConfig.from_yaml()
    assert config.num_classes == 18
    assert config.num_channels == 6
    assert config.window_samples == 300
    assert len(config.classes) == 18


def test_validator_with_synthetic_valid_data(tmp_path):
    # Create valid synthetic CSV with 1803 columns
    rows = 20
    cols = 1803
    synthetic = np.random.randn(rows, cols).astype(np.float32)
    # Col 1800 is class ID (0..17)
    synthetic[:, 1800] = np.random.randint(0, 18, size=rows)
    synthetic[:, 1801] = 300
    synthetic[:, 1802] = np.arange(rows)

    csv_path = tmp_path / "synthetic_valid.csv"
    np.savetxt(csv_path, synthetic, delimiter=",", fmt="%.4f")

    validator = DataValidator()
    report = validator.validate_csv(csv_path)
    assert report.row_count == rows
    assert report.column_count == cols
    assert report.missing_values_count == 0
    assert report.infinite_values_count == 0


def test_validator_detects_infinite_values(tmp_path):
    rows = 10
    cols = 1803
    synthetic = np.random.randn(rows, cols).astype(np.float32)
    synthetic[2, 50] = np.inf
    synthetic[:, 1800] = 0

    csv_path = tmp_path / "synthetic_inf.csv"
    np.savetxt(csv_path, synthetic, delimiter=",")

    validator = DataValidator()
    report = validator.validate_csv(csv_path)
    assert not report.is_valid
    assert report.infinite_values_count > 0
