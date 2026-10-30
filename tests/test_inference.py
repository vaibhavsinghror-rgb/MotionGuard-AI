"""Automated tests for inference API schemas and input validation."""

import pytest
from pydantic import ValidationError
from motionguard.inference.schemas import WindowInput


def test_window_input_valid():
    # 6 channels x 300 samples
    valid_data = [[0.0] * 300 for _ in range(6)]
    payload = WindowInput(channels=valid_data, confidence_threshold=0.75)
    assert len(payload.channels) == 6
    assert payload.confidence_threshold == 0.75


def test_window_input_invalid_channel_count():
    # Only 5 channels
    invalid_data = [[0.0] * 300 for _ in range(5)]
    with pytest.raises(ValidationError):
        WindowInput(channels=invalid_data)


def test_window_input_invalid_window_length():
    # 6 channels but only 200 samples
    invalid_data = [[0.0] * 200 for _ in range(6)]
    with pytest.raises(ValidationError):
        WindowInput(channels=invalid_data)


def test_window_input_rejects_nan():
    # 6 channels with NaN
    invalid_data = [[0.0] * 300 for _ in range(6)]
    invalid_data[0][10] = float("nan")
    with pytest.raises(ValidationError):
        WindowInput(channels=invalid_data)
