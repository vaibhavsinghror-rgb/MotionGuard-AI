"""Research Experiment: Unfamiliar-pattern and movement anomaly detection.

Fulfills Section 12 of specification.
"""

import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from motionguard.anomaly.detector import MovementAnomalyDetector
from motionguard.anomaly.evaluation import evaluate_anomaly_detection
from motionguard.config import DataConfig
from motionguard.data.load import load_processed_arrays
from motionguard.data.split import load_split_manifest
from motionguard.utils.logging import get_logger

logger = get_logger("anomaly_experiments")


def run_unfamiliar_pattern_experiment(held_out_class_id: int = 8):
    """Holds out an activity (default: class 8 'Jump') as unfamiliar/anomalous,
    trains detector on remaining in-distribution classes, and evaluates detection power.
    """
    logger.info("Initializing Movement Anomaly Detection Experiment...")
    data_config = DataConfig.from_yaml()
    held_out_name = data_config.classes.get(held_out_class_id, f"Class {held_out_class_id}")
    logger.info(f"Designated Unfamiliar / Held-Out Activity: {held_out_name} (ID {held_out_class_id})")

    # Load features and labels
    split_manifest = data_config.manifests_dir / "split_manifest.json"
    train_idx, val_idx, _ = load_split_manifest(split_manifest)

    X_train = np.load(data_config.processed_dir / "features_train.npy")
    X_val = np.load(data_config.processed_dir / "features_val.npy")

    _, labels, _ = load_processed_arrays(data_config.processed_dir)
    y_train = labels[train_idx]
    y_val = labels[val_idx]

    # In-distribution training data: exclude held_out_class_id
    train_in_mask = y_train != held_out_class_id
    X_train_in = X_train[train_in_mask]
    logger.info(f"Training Anomaly Detector on {len(X_train_in)} in-distribution training vectors (excluding {held_out_name})...")

    detector = MovementAnomalyDetector(contamination=0.05, n_estimators=150, random_state=42)
    detector.fit(X_train_in)

    # Save anomaly detector artifact
    detector_path = PROJECT_ROOT / "artifacts" / "models" / "anomaly_detector.joblib"
    detector.save(detector_path)

    # Validation evaluation: in-distribution validation vs out-of-distribution (held-out activity)
    val_in_mask = y_val != held_out_class_id
    val_ood_mask = y_val == held_out_class_id

    scores_in = detector.score_samples(X_val[val_in_mask])
    scores_ood = detector.score_samples(X_val[val_ood_mask])

    results = evaluate_anomaly_detection(scores_in, scores_ood)
    results["held_out_class_id"] = held_out_class_id
    results["held_out_class_name"] = held_out_name
    results["in_distribution_count"] = int(np.sum(val_in_mask))
    results["out_of_distribution_count"] = int(np.sum(val_ood_mask))
    results["scientific_limitation"] = (
        "Statistically detects deviation from normal training activities. "
        "Does NOT constitute clinical fall detection, injury diagnosis, or certified real-world hazard detection."
    )

    out_dir = PROJECT_ROOT / "experiments" / "results" / "anomaly"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "anomaly_detection_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Plot anomaly score distribution
    plt.figure(figsize=(8, 5))
    plt.hist(scores_in, bins=40, density=True, alpha=0.6, color="navy", label=f"In-Distribution Normal ({results['in_distribution_count']} windows)")
    plt.hist(scores_ood, bins=40, density=True, alpha=0.6, color="crimson", label=f"Unfamiliar: {held_out_name} ({results['out_of_distribution_count']} windows)")
    plt.xlabel("Anomaly Score (Higher = More Deviant/Unfamiliar)", fontsize=11)
    plt.ylabel("Probability Density", fontsize=11)
    plt.title(f"Unfamiliar Pattern Detection: In-Distribution vs {held_out_name} (ROC-AUC = {results['roc_auc']:.3f})", fontsize=12, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "anomaly_score_distribution.png", dpi=200)
    plt.close()

    logger.info(f"Anomaly detection experiment complete: ROC-AUC = {results['roc_auc']:.4f}, PR-AUC = {results['pr_auc']:.4f}")
    print("\n" + "=" * 65)
    print("      UNFAMILIAR-PATTERN ANOMALY DETECTION RESULTS")
    print("=" * 65)
    print(f"Held-out Unfamiliar Class: {held_out_name} (ID {held_out_class_id})")
    print(f"ROC-AUC:                  {results['roc_auc']:.4f}")
    print(f"PR-AUC:                   {results['pr_auc']:.4f}")
    print(f"Mean In-Dist Score:       {results['mean_in_score']:.4f} ± {results['std_in_score']:.4f}")
    print(f"Mean Unfamiliar Score:    {results['mean_ood_score']:.4f} ± {results['std_ood_score']:.4f}")
    print("=" * 65)
    print(f"\n[ARTIFACTS]: Saved to {detector_path} and {out_dir}")


def main():
    run_unfamiliar_pattern_experiment()


if __name__ == "__main__":
    main()
