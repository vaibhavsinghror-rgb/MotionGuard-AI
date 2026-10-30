"""Automated tests for evaluation, calibration, and abstention metrics."""

import numpy as np
import pytest
from motionguard.evaluation.metrics import compute_classification_metrics
from motionguard.uncertainty.abstention import SelectivePredictor
from motionguard.uncertainty.metrics import compute_expected_calibration_error


def test_classification_metrics_perfect_predictions():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 1, 2])

    metrics = compute_classification_metrics(y_true, y_pred)
    assert metrics.accuracy == 1.0
    assert metrics.macro_f1 == 1.0
    assert metrics.macro_precision == 1.0
    assert metrics.macro_recall == 1.0


def test_ece_perfect_calibration():
    # 2 samples, 100% confidence matching truth
    probs = np.array([[1.0, 0.0], [0.0, 1.0]])
    labels = np.array([0, 1])
    ece, mce, _ = compute_expected_calibration_error(probs, labels, n_bins=10)
    assert ece == pytest.approx(0.0, abs=1e-5)


def test_selective_prediction_abstention():
    predictor = SelectivePredictor(confidence_threshold=0.80)

    # High confidence sample
    high_conf_prob = np.array([0.90, 0.05, 0.05])
    res_high = predictor.predict_sample(high_conf_prob)
    assert res_high.is_accepted
    assert res_high.status == "prediction_available"
    assert res_high.confidence == pytest.approx(0.90)

    # Low confidence sample
    low_conf_prob = np.array([0.45, 0.35, 0.20])
    res_low = predictor.predict_sample(low_conf_prob)
    assert not res_low.is_accepted
    assert res_low.status == "uncertain_human_review_recommended"
    assert "below the required acceptance threshold" in res_low.abstention_reason
