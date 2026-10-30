"""Evaluation protocols for movement anomaly detection."""

from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def evaluate_anomaly_detection(
    in_distribution_scores: np.ndarray,
    out_of_distribution_scores: np.ndarray,
) -> Dict[str, float]:
    """Computes ROC-AUC and PR-AUC for separating in-distribution vs held-out activities.
    
    Args:
        in_distribution_scores: anomaly scores for normal validation samples
        out_of_distribution_scores: anomaly scores for held-out unfamiliar activity
        
    Returns:
        Dictionary with roc_auc, pr_auc, and score statistics
    """
    y_true = np.concatenate([
        np.zeros(len(in_distribution_scores)),
        np.ones(len(out_of_distribution_scores)),
    ])
    y_scores = np.concatenate([in_distribution_scores, out_of_distribution_scores])

    roc_auc = float(roc_auc_score(y_true, y_scores))
    pr_auc = float(average_precision_score(y_true, y_scores))

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "mean_in_score": float(np.mean(in_distribution_scores)),
        "std_in_score": float(np.std(in_distribution_scores)),
        "mean_ood_score": float(np.mean(out_of_distribution_scores)),
        "std_ood_score": float(np.std(out_of_distribution_scores)),
    }
