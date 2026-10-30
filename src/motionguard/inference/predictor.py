"""Inference predictor engine for MotionGuard AI."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union
import joblib
import numpy as np
import torch

from motionguard.config import PROJECT_ROOT
from motionguard.data.preprocess import WaveformNormalizer
from motionguard.features.frequency_domain import extract_all_features
from motionguard.inference.schemas import PredictionResponse, WindowInput
from motionguard.models.cnn1d import MotionGuardCNN1D
from motionguard.uncertainty.abstention import SelectivePredictor
from motionguard.uncertainty.calibration import TemperatureScaler
from motionguard.utils.logging import get_logger

logger = get_logger(__name__)


class MotionGuardPredictor:
    """Production-grade inference predictor loading saved model artifacts from the project directory."""

    def __init__(
        self,
        artifacts_dir: Optional[Path] = None,
        device: str = "cpu",
        default_confidence_threshold: float = 0.70,
    ):
        self.artifacts_dir = artifacts_dir or (PROJECT_ROOT / "artifacts")
        self.models_dir = self.artifacts_dir / "models"
        self.preprocessors_dir = self.artifacts_dir / "preprocessors"
        self.calibration_dir = self.artifacts_dir / "calibration"
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.default_confidence_threshold = default_confidence_threshold

        self.model: Optional[Union[torch.nn.Module, Any]] = None
        self.model_metadata: Dict = {}
        self.class_mapping: Dict[int, str] = {}
        self.normalizer: Optional[WaveformNormalizer] = None
        self.feature_scaler: Optional[Any] = None
        self.calibrator: Optional[TemperatureScaler] = None
        self.selective_predictor = SelectivePredictor(confidence_threshold=default_confidence_threshold)
        self.is_loaded = False

        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Loads all required persistent artifacts safely from artifacts/ directory."""
        metadata_file = self.models_dir / "best_model_metadata.json"
        if not metadata_file.exists():
            logger.warning(
                f"Model metadata not found at {metadata_file}. "
                "Training pipeline must be executed first to generate model artifacts."
            )
            self.is_loaded = False
            return

        with open(metadata_file, "r", encoding="utf-8") as f:
            self.model_metadata = json.load(f)

        # Class mapping
        class_mapping_file = self.models_dir / "class_mapping.json"
        if class_mapping_file.exists():
            with open(class_mapping_file, "r", encoding="utf-8") as f:
                raw_mapping = json.load(f)
                self.class_mapping = {int(k): str(v) for k, v in raw_mapping.items()}

        # Load Temperature Scaler if present
        calib_file = self.calibration_dir / "calibration.json"
        if calib_file.exists():
            self.calibrator = TemperatureScaler.load(calib_file)
            logger.info(f"Loaded calibration with temperature: {self.calibrator.temperature:.4f}")

        # Load Normalizer if present
        norm_file = self.preprocessors_dir / "waveform_normalizer.joblib"
        if norm_file.exists():
            self.normalizer = WaveformNormalizer.load(norm_file)

        model_family = self.model_metadata.get("model_family", "deep_learning")

        if model_family == "deep_learning":
            pt_path = self.models_dir / "best_model.pt"
            if not pt_path.exists():
                raise FileNotFoundError(f"PyTorch model checkpoint missing at {pt_path}")
            
            checkpoint = torch.load(pt_path, map_location=self.device, weights_only=False)
            model_config = checkpoint.get("config", {})
            self.model = MotionGuardCNN1D.from_config(model_config).to(self.device)
            self.model.load_state_dict(checkpoint["state_dict"])
            self.model.eval()
            logger.info(f"Loaded PyTorch 1D-CNN from {pt_path} on {self.device}")
        else:
            joblib_path = self.models_dir / "best_classical_model.joblib"
            if not joblib_path.exists():
                raise FileNotFoundError(f"Classical model file missing at {joblib_path}")
            self.model = joblib.load(joblib_path)
            logger.info(f"Loaded classical model from {joblib_path}")

        self.is_loaded = True

    def predict(
        self,
        waveform: Union[np.ndarray, List[List[float]]],
        threshold: Optional[float] = None,
    ) -> PredictionResponse:
        """Runs end-to-end inference on a 6x300 sensor window.
        
        Args:
            waveform: array of shape (6, 300) or (1, 6, 300)
            threshold: optional custom confidence acceptance threshold
        """
        if not self.is_loaded or self.model is None:
            raise RuntimeError("MotionGuardPredictor has no loaded model. Execute training pipeline first.")

        # Ensure numpy array
        if isinstance(waveform, list):
            arr = np.array(waveform, dtype=np.float32)
        else:
            arr = np.asarray(waveform, dtype=np.float32)

        if arr.ndim == 2:
            assert arr.shape == (6, 300), f"Expected waveform shape (6, 300), got {arr.shape}"
            arr = np.expand_dims(arr, axis=0)  # (1, 6, 300)
        elif arr.ndim == 3:
            assert arr.shape[1:] == (6, 300), f"Expected shape (N, 6, 300), got {arr.shape}"
        else:
            raise ValueError(f"Invalid input dimensions: {arr.shape}")

        # Check for NaN / Inf
        if np.isnan(arr).any() or np.isinf(arr).any():
            raise ValueError("Input waveform contains NaN or Infinite values.")

        model_family = self.model_metadata.get("model_family", "deep_learning")

        if model_family == "deep_learning":
            # Normalize with fitted normalizer if available
            norm_arr = self.normalizer.transform(arr) if self.normalizer else arr
            tensor_x = torch.from_numpy(norm_arr).float().to(self.device)

            with torch.no_grad():
                logits = self.model(tensor_x).cpu().numpy()[0]  # (num_classes,)

            # Calibrate logits
            if self.calibrator and self.calibrator.is_fitted:
                calib_probs = self.calibrator.calibrate(np.expand_dims(logits, 0))[0]
            else:
                # Standard softmax
                exp_l = np.exp(logits - np.max(logits))
                calib_probs = exp_l / np.sum(exp_l)
        else:
            # Classical model requires feature extraction
            channel_names = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
            features, _ = extract_all_features(arr, channel_names=channel_names)
            calib_probs = self.model.predict_proba(features)[0]

        # Selective prediction check
        thresh = threshold if threshold is not None else self.default_confidence_threshold
        result = self.selective_predictor.predict_sample(calib_probs, threshold=thresh)

        # Construct named probability mapping
        named_probs = {}
        for class_idx, prob in enumerate(calib_probs):
            class_name = self.class_mapping.get(class_idx, f"Class_{class_idx}")
            named_probs[class_name] = float(prob)

        predicted_activity = self.class_mapping.get(result.predicted_class, f"Class_{result.predicted_class}")

        return PredictionResponse(
            activity=predicted_activity,
            class_id=result.predicted_class,
            confidence=result.confidence,
            accepted=result.is_accepted,
            status=result.status,
            probabilities=named_probs,
            model_version=self.model_metadata.get("model_name", "1.0.0"),
            abstention_reason=result.abstention_reason,
        )
