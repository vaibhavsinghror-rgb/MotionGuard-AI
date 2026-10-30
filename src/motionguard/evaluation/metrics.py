"""Comprehensive evaluation metrics for MotionGuard AI."""

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


@dataclass
class EvaluationMetrics:
    accuracy: float = 0.0
    balanced_accuracy: float = 0.0
    macro_precision: float = 0.0
    macro_recall: float = 0.0
    macro_f1: float = 0.0
    weighted_f1: float = 0.0
    per_class_f1: Dict[int, float] = field(default_factory=dict)
    per_class_precision: Dict[int, float] = field(default_factory=dict)
    per_class_recall: Dict[int, float] = field(default_factory=dict)
    confusion_matrix: List[List[int]] = field(default_factory=list)
    avg_inference_latency_ms: float = 0.0
    model_size_bytes: int = 0
    num_samples: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[Dict[int, str]] = None,
    latency_ms: float = 0.0,
    model_size_bytes: int = 0,
) -> EvaluationMetrics:
    """Computes full suite of academic and production classification metrics."""
    assert len(y_true) == len(y_pred), f"Mismatched shapes: {len(y_true)} vs {len(y_pred)}"
    
    unique_classes = sorted(np.unique(np.concatenate([y_true, y_pred])))

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    m_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    m_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    m_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    w_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=unique_classes).tolist()

    # Per-class metrics
    p_prec = precision_score(y_true, y_pred, average=None, labels=unique_classes, zero_division=0)
    p_rec = recall_score(y_true, y_pred, average=None, labels=unique_classes, zero_division=0)
    p_f1 = f1_score(y_true, y_pred, average=None, labels=unique_classes, zero_division=0)

    per_class_precision = {int(c): float(p) for c, p in zip(unique_classes, p_prec)}
    per_class_recall = {int(c): float(r) for c, r in zip(unique_classes, p_rec)}
    per_class_f1 = {int(c): float(f) for c, f in zip(unique_classes, p_f1)}

    return EvaluationMetrics(
        accuracy=acc,
        balanced_accuracy=bal_acc,
        macro_precision=m_prec,
        macro_recall=m_rec,
        macro_f1=m_f1,
        weighted_f1=w_f1,
        per_class_f1=per_class_f1,
        per_class_precision=per_class_precision,
        per_class_recall=per_class_recall,
        confusion_matrix=cm,
        avg_inference_latency_ms=latency_ms,
        model_size_bytes=model_size_bytes,
        num_samples=len(y_true),
    )


def measure_inference_latency(predict_fn, sample_input, num_warmup: int = 5, num_runs: int = 50) -> float:
    """Measures mean inference latency in milliseconds per single sample."""
    # Warmup
    for _ in range(num_warmup):
        _ = predict_fn(sample_input)

    times = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = predict_fn(sample_input)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    return float(np.mean(times))
