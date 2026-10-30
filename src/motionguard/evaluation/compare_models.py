"""Model comparison, selection, and promotion pipeline for MotionGuard AI."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import pandas as pd
import torch

from motionguard.config import EvaluationConfig, TrainingConfig
from motionguard.evaluation.metrics import EvaluationMetrics
from motionguard.utils.logging import get_logger

logger = get_logger(__name__)


def compute_file_sha256(filepath: Path) -> str:
    """Computes SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


class ModelSelector:
    """Selects the best performing candidate model based strictly on validation Macro-F1."""

    def __init__(
        self,
        eval_config: Optional[EvaluationConfig] = None,
        train_config: Optional[TrainingConfig] = None,
    ):
        self.eval_config = eval_config or EvaluationConfig.from_yaml()
        self.train_config = train_config or TrainingConfig.from_yaml()
        self.artifacts_dir = self.train_config.artifacts_dir
        self.models_dir = self.train_config.models_dir

    def compare_and_rank(
        self,
        model_results: Dict[str, Tuple[Any, EvaluationMetrics, float]],
    ) -> pd.DataFrame:
        """Ranks models by validation Macro-F1, compiling comparison table."""
        rows = []
        for model_name, (model_obj, metrics, train_time) in model_results.items():
            rows.append({
                "model_name": model_name,
                "val_macro_f1": metrics.macro_f1,
                "val_accuracy": metrics.accuracy,
                "val_balanced_acc": metrics.balanced_accuracy,
                "val_macro_precision": metrics.macro_precision,
                "val_macro_recall": metrics.macro_recall,
                "inference_latency_ms": metrics.avg_inference_latency_ms,
                "model_size_mb": metrics.model_size_bytes / (1024 * 1024),
                "training_time_sec": train_time,
            })

        df = pd.DataFrame(rows).sort_values(by="val_macro_f1", ascending=False).reset_index(drop=True)
        logger.info("\n=== Model Comparison Leaderboard (Validation Set) ===\n" + df.to_string(index=False))
        return df

    def promote_best_model(
        self,
        ranked_df: pd.DataFrame,
        model_objects: Dict[str, Any],
        class_mapping: Dict[int, str],
        dataset_manifest_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Promotes the best candidate model to artifacts/models/best_model.*."""
        if ranked_df.empty:
            raise ValueError("No candidate models to evaluate!")

        best_row = ranked_df.iloc[0]
        best_name = str(best_row["model_name"])
        best_f1 = float(best_row["val_macro_f1"])

        logger.info(f"Top candidate: {best_name} with validation Macro-F1: {best_f1:.4f}")

        # Check quality gate
        min_gate = self.eval_config.min_val_macro_f1
        if best_f1 < min_gate:
            logger.warning(
                f"Top candidate {best_name} (Macro-F1 {best_f1:.4f}) does not meet target quality gate ({min_gate:.4f}). "
                "Saving with experimental status."
            )
            is_production_ready = False
        else:
            logger.info(f"Candidate {best_name} passed all validation quality gates.")
            is_production_ready = True

        self.models_dir.mkdir(parents=True, exist_ok=True)
        manifests_dir = self.train_config.manifests_dir
        manifests_dir.mkdir(parents=True, exist_ok=True)

        best_model_obj = model_objects[best_name]
        is_dl = "cnn" in best_name.lower() or isinstance(best_model_obj, torch.nn.Module)

        # Save class mapping
        class_mapping_path = self.models_dir / "class_mapping.json"
        with open(class_mapping_path, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in class_mapping.items()}, f, indent=2)

        artifact_file = None
        if is_dl:
            artifact_file = self.models_dir / "best_model.pt"
            # PyTorch checkpoint with full metadata
            checkpoint = {
                "state_dict": best_model_obj.state_dict(),
                "architecture": "MotionGuardCNN1D",
                "config": getattr(best_model_obj, "get_config", lambda: {})(),
                "class_mapping": class_mapping,
                "in_channels": 6,
                "sequence_length": 300,
                "val_macro_f1": best_f1,
                "val_accuracy": float(best_row["val_accuracy"]),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            torch.save(checkpoint, artifact_file)
        else:
            artifact_file = self.models_dir / "best_classical_model.joblib"
            joblib.dump(best_model_obj, artifact_file)

        # Compute checksum
        artifact_sha = compute_file_sha256(artifact_file)

        # Metadata
        metadata = {
            "model_family": "deep_learning" if is_dl else "classical",
            "model_name": best_name,
            "artifact_path": str(artifact_file.relative_to(self.artifacts_dir.parent)),
            "artifact_sha256": artifact_sha,
            "is_production_ready": is_production_ready,
            "val_metrics": best_row.to_dict(),
            "class_mapping_path": str(class_mapping_path.relative_to(self.artifacts_dir.parent)),
            "input_shape": [1, 6, 300] if is_dl else [1, "num_features"],
            "promoted_at": datetime.now(timezone.utc).isoformat(),
        }

        metadata_path = self.models_dir / "best_model_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        # Comprehensive Model Manifest
        manifest_payload = {
            "model_version": "1.0.0",
            "model_name": best_name,
            "created_timestamp": datetime.now(timezone.utc).isoformat(),
            "selected_family": metadata["model_family"],
            "artifacts": {
                "model_file": str(artifact_file.name),
                "model_sha256": artifact_sha,
                "metadata_file": str(metadata_path.name),
                "class_mapping": str(class_mapping_path.name),
            },
            "validation_performance": {
                "macro_f1": best_f1,
                "accuracy": float(best_row["val_accuracy"]),
                "balanced_accuracy": float(best_row["val_balanced_acc"]),
                "inference_latency_ms": float(best_row["inference_latency_ms"]),
            },
            "dataset_manifest_reference": str(dataset_manifest_path) if dataset_manifest_path else None,
            "known_limitations": [
                "Dataset collected from 90 participants using smartphone sensors in controlled trials.",
                "Not clinically validated for medical diagnostic fall detection or real-world safety certification.",
                "Performance may vary under irregular sensor placements or sampling jitter.",
            ],
        }

        manifest_file = manifests_dir / "model_manifest.json"
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest_payload, f, indent=2)

        logger.info(f"Promoted best model {best_name} -> {artifact_file}")
        return manifest_payload
