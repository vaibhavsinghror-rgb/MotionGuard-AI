"""Selective prediction and confidence abstention mechanism for MotionGuard AI."""

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class SelectivePredictionResult:
    predicted_class: int
    confidence: float
    is_accepted: bool
    status: str
    probabilities: List[float]
    abstention_reason: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)


class SelectivePredictor:
    """Confidence-aware selective classifier with configurable thresholding."""

    def __init__(self, confidence_threshold: float = 0.70):
        self.confidence_threshold = float(confidence_threshold)

    def predict_sample(
        self,
        probabilities: np.ndarray,
        threshold: Optional[float] = None,
    ) -> SelectivePredictionResult:
        """Evaluates single sample probability distribution for prediction or abstention."""
        thresh = threshold if threshold is not None else self.confidence_threshold
        pred_class = int(np.argmax(probabilities))
        conf = float(np.max(probabilities))

        if conf >= thresh:
            return SelectivePredictionResult(
                predicted_class=pred_class,
                confidence=conf,
                is_accepted=True,
                status="prediction_available",
                probabilities=[float(p) for p in probabilities],
                abstention_reason=None,
            )
        else:
            return SelectivePredictionResult(
                predicted_class=pred_class,
                confidence=conf,
                is_accepted=False,
                status="uncertain_human_review_recommended",
                probabilities=[float(p) for p in probabilities],
                abstention_reason=(
                    f"Predicted confidence ({conf:.2%}) is below the required acceptance threshold ({thresh:.2%})."
                ),
            )

    def evaluate_risk_coverage(
        self,
        probabilities: np.ndarray,
        true_labels: np.ndarray,
        thresholds: Optional[List[float]] = None,
    ) -> List[Dict]:
        """Calculates coverage, selective accuracy, and risk across multiple thresholds."""
        thresholds = thresholds or [0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        confidences = np.max(probabilities, axis=1)
        predictions = np.argmax(probabilities, axis=1)
        total_samples = len(true_labels)

        tradeoff_curve = []
        for thresh in sorted(thresholds):
            accepted_mask = confidences >= thresh
            accepted_count = int(np.sum(accepted_mask))
            coverage = accepted_count / total_samples

            if accepted_count > 0:
                acc_on_accepted = float(np.mean(predictions[accepted_mask] == true_labels[accepted_mask]))
                error_rate = 1.0 - acc_on_accepted
            else:
                acc_on_accepted = 1.0
                error_rate = 0.0

            tradeoff_curve.append({
                "threshold": float(thresh),
                "coverage": float(coverage),
                "accepted_count": accepted_count,
                "total_count": total_samples,
                "accuracy_accepted": acc_on_accepted,
                "error_rate": error_rate,
            })

        return tradeoff_curve
