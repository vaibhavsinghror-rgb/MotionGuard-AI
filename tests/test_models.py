"""Automated tests for model architectures and training forward passes."""

import numpy as np
import pytest
import torch
from motionguard.models.cnn1d import MotionGuardCNN1D
from motionguard.models.random_forest import build_random_forest_pipeline


def test_cnn1d_forward_pass_dimensions():
    batch_size = 4
    channels = 6
    sequence_length = 300
    num_classes = 18

    model = MotionGuardCNN1D(
        in_channels=channels,
        sequence_length=sequence_length,
        num_classes=num_classes,
    )

    dummy_input = torch.randn(batch_size, channels, sequence_length)
    output = model(dummy_input)

    assert output.shape == (batch_size, num_classes)
    assert not torch.isnan(output).any()
    assert torch.isfinite(output).all()


def test_random_forest_pipeline_fit_predict():
    N, D = 50, 144
    X = np.random.randn(N, D)
    y = np.random.randint(0, 5, size=N)

    pipeline = build_random_forest_pipeline(params={"n_estimators": 10})
    pipeline.fit(X, y)

    preds = pipeline.predict(X[:5])
    assert len(preds) == 5
    probs = pipeline.predict_proba(X[:5])
    assert probs.shape == (5, len(np.unique(y)))
