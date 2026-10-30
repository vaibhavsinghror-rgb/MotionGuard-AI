"""Model training pipelines for classical and deep learning models in MotionGuard AI."""

import copy
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from motionguard.config import PROJECT_ROOT, TrainingConfig
from motionguard.data.dataset import create_dataloaders
from motionguard.evaluation.metrics import (
    EvaluationMetrics,
    compute_classification_metrics,
    measure_inference_latency,
)
from motionguard.models.cnn1d import MotionGuardCNN1D
from motionguard.models.gradient_boosting import build_gradient_boosting_pipeline
from motionguard.models.random_forest import build_random_forest_pipeline
from motionguard.models.svm import build_svm_pipeline
from motionguard.utils.logging import get_logger

logger = get_logger(__name__)


def train_classical_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: Optional[TrainingConfig] = None,
) -> Tuple[Any, EvaluationMetrics, float]:
    """Trains a classical machine learning pipeline on engineered features.
    
    Supported model_name: 'random_forest', 'gradient_boosting', 'svm'
    """
    config = config or TrainingConfig.from_yaml()
    seed = config.seed

    logger.info(f"Training classical model: {model_name} on {X_train.shape[0]} samples, {X_train.shape[1]} features...")
    
    if model_name == "random_forest":
        pipeline = build_random_forest_pipeline(params=config.classical_rf, random_state=seed)
    elif model_name == "gradient_boosting":
        pipeline = build_gradient_boosting_pipeline(params=config.classical_gb, random_state=seed)
    elif model_name == "svm":
        pipeline = build_svm_pipeline(params=config.classical_svm, random_state=seed)
    else:
        raise ValueError(f"Unknown classical model: {model_name}")

    start_time = time.perf_counter()
    pipeline.fit(X_train, y_train)
    training_time = time.perf_counter() - start_time
    logger.info(f"Finished {model_name} training in {training_time:.2f}s.")

    # Validation evaluation
    y_val_pred = pipeline.predict(X_val)
    single_sample = X_val[0:1]
    latency_ms = measure_inference_latency(pipeline.predict, single_sample)

    # Estimate model size
    temp_file = PROJECT_ROOT / f"temp_{model_name}.joblib"
    joblib.dump(pipeline, temp_file)
    model_size_bytes = temp_file.stat().st_size if temp_file.exists() else 0
    if temp_file.exists():
        temp_file.unlink()

    metrics = compute_classification_metrics(
        y_true=y_val,
        y_pred=y_val_pred,
        latency_ms=latency_ms,
        model_size_bytes=model_size_bytes,
    )
    logger.info(f"Validation results for {model_name}: Acc={metrics.accuracy:.4f}, Macro-F1={metrics.macro_f1:.4f}")
    return pipeline, metrics, training_time


def train_cnn1d_model(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    config: Optional[TrainingConfig] = None,
    save_checkpoint_path: Optional[Path] = None,
) -> Tuple[MotionGuardCNN1D, EvaluationMetrics, Dict[str, List[float]], float]:
    """Trains 1D-CNN on raw 6-channel waveforms with validation monitoring and early stopping."""
    config = config or TrainingConfig.from_yaml()
    dl_cfg = config.dl_config

    device_str = "cuda" if torch.cuda.is_available() and config.device == "cuda" else "cpu"
    device = torch.device(device_str)
    logger.info(f"Training 1D-CNN on device: {device}...")

    # DataLoaders
    batch_size = dl_cfg.get("batch_size", 64)
    train_loader, val_loader, _ = create_dataloaders(
        train_x=train_x,
        train_y=train_y,
        val_x=val_x,
        val_y=val_y,
        batch_size=batch_size,
        augment_train=True,
    )

    # Model architecture
    num_classes = dl_cfg.get("num_classes", 18)
    model = MotionGuardCNN1D(
        in_channels=dl_cfg.get("in_channels", 6),
        sequence_length=dl_cfg.get("sequence_length", 300),
        num_classes=num_classes,
        conv_channels=dl_cfg.get("conv_channels", [64, 128, 256]),
        kernel_sizes=dl_cfg.get("kernel_sizes", [7, 5, 3]),
        dropout=dl_cfg.get("dropout", 0.3),
        fc_hidden=dl_cfg.get("fc_hidden", 128),
    ).to(device)

    # Loss function with optional class weights
    if dl_cfg.get("use_class_weights", True):
        classes = np.arange(num_classes)
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=train_y)
        class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(
        model.parameters(),
        lr=float(dl_cfg.get("learning_rate", 0.001)),
        weight_decay=float(dl_cfg.get("weight_decay", 1e-4)),
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        patience=int(dl_cfg.get("lr_patience", 2)),
        factor=float(dl_cfg.get("lr_factor", 0.5)),
        min_lr=float(dl_cfg.get("min_lr", 1e-6)),
    )

    epochs = dl_cfg.get("epochs", 35)
    patience = dl_cfg.get("early_stopping_patience", 8)
    best_macro_f1 = -1.0
    best_model_weights = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
    }

    start_time = time.perf_counter()

    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        train_loss_total = 0.0
        train_correct = 0
        train_total = 0

        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits = model(x_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()

            train_loss_total += loss.item() * x_b.size(0)
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == y_b).sum().item()
            train_total += x_b.size(0)

        epoch_train_loss = train_loss_total / train_total
        epoch_train_acc = train_correct / train_total

        # Validation phase
        model.eval()
        val_loss_total = 0.0
        val_preds_list = []
        val_targets_list = []

        with torch.no_grad():
            for x_v, y_v in val_loader:
                x_v, y_v = x_v.to(device), y_v.to(device)
                v_logits = model(x_v)
                v_loss = criterion(v_logits, y_v)
                val_loss_total += v_loss.item() * x_v.size(0)

                v_preds = torch.argmax(v_logits, dim=1)
                val_preds_list.extend(v_preds.cpu().numpy())
                val_targets_list.extend(y_v.cpu().numpy())

        epoch_val_loss = val_loss_total / len(val_targets_list)
        val_preds_arr = np.array(val_preds_list)
        val_targets_arr = np.array(val_targets_list)
        
        val_metrics = compute_classification_metrics(y_true=val_targets_arr, y_pred=val_preds_arr)
        epoch_val_acc = val_metrics.accuracy
        epoch_val_f1 = val_metrics.macro_f1

        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)
        history["val_macro_f1"].append(epoch_val_f1)

        scheduler.step(epoch_val_f1)

        logger.info(
            f"Epoch {epoch:02d}/{epochs:02d} - "
            f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}, Val Macro-F1: {epoch_val_f1:.4f}"
        )

        if epoch_val_f1 > best_macro_f1:
            best_macro_f1 = epoch_val_f1
            best_model_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            if save_checkpoint_path:
                save_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_model_weights, save_checkpoint_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch} (no improvement for {patience} epochs).")
                break

    training_time = time.perf_counter() - start_time
    model.load_state_dict(best_model_weights)
    model.eval()

    # Latency and size measurement
    sample_tensor = torch.from_numpy(val_x[0:1]).float().to(device)
    
    def run_inference(t):
        with torch.no_grad():
            return model(t)

    latency_ms = measure_inference_latency(run_inference, sample_tensor)
    
    # Compute size in bytes
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    model_size_bytes = param_size + buffer_size

    # Final validation metrics on best weights
    final_preds = []
    with torch.no_grad():
        for x_v, _ in val_loader:
            x_v = x_v.to(device)
            final_preds.extend(torch.argmax(model(x_v), dim=1).cpu().numpy())

    final_metrics = compute_classification_metrics(
        y_true=val_y,
        y_pred=np.array(final_preds),
        latency_ms=latency_ms,
        model_size_bytes=model_size_bytes,
    )
    logger.info(f"Final 1D-CNN Best Validation Macro-F1: {final_metrics.macro_f1:.4f}, Accuracy: {final_metrics.accuracy:.4f}")

    return model, final_metrics, history, training_time
