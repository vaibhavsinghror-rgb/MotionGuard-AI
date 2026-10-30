"""Automated tests for saved artifact integrity and persistence."""

from pathlib import Path
import pytest
from motionguard.config import PROJECT_ROOT
from motionguard.evaluation.compare_models import compute_file_sha256
from motionguard.inference.predictor import MotionGuardPredictor


def test_saved_artifacts_exist():
    artifacts_dir = PROJECT_ROOT / "artifacts"
    models_dir = artifacts_dir / "models"
    manifests_dir = artifacts_dir / "manifests"

    assert (models_dir / "best_model_metadata.json").exists()
    assert (models_dir / "class_mapping.json").exists()
    assert (manifests_dir / "model_manifest.json").exists()
    assert (artifacts_dir / "calibration" / "calibration.json").exists()


def test_predictor_loads_saved_artifact():
    predictor = MotionGuardPredictor()
    assert predictor.is_loaded
    assert predictor.model is not None
    assert len(predictor.class_mapping) == 18
