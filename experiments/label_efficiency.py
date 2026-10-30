"""Research Experiment: Label-efficient learning across 10%, 25%, 50%, 100% budgets.

Fulfills Section 11 of specification.
"""

import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from motionguard.config import DataConfig
from motionguard.data.load import load_processed_arrays
from motionguard.data.split import load_split_manifest
from motionguard.evaluation.metrics import compute_classification_metrics
from motionguard.models.gradient_boosting import build_gradient_boosting_pipeline
from motionguard.utils.logging import get_logger

logger = get_logger("label_efficiency")


def run_label_efficiency_experiment():
    logger.info("Initializing Label Efficiency Experiment...")
    data_config = DataConfig.from_yaml()
    split_manifest = data_config.manifests_dir / "split_manifest.json"

    train_idx, val_idx, test_idx = load_split_manifest(split_manifest)

    # Load engineered features
    X_train_full = np.load(data_config.processed_dir / "features_train.npy")
    X_val = np.load(data_config.processed_dir / "features_val.npy")
    X_test = np.load(data_config.processed_dir / "features_test.npy")

    waveforms, labels, _ = load_processed_arrays(data_config.processed_dir)
    y_train_full = labels[train_idx]
    y_val = labels[val_idx]
    y_test = labels[test_idx]

    budgets = [0.10, 0.25, 0.50, 1.00]
    seeds = [42, 101, 2024]

    experiment_results = []

    for budget in budgets:
        budget_f1s = []
        budget_accs = []
        n_samples = int(len(X_train_full) * budget)
        logger.info(f"\n--- Testing Label Budget: {budget:.0%} ({n_samples} training samples) ---")

        for seed in seeds:
            if budget < 1.0:
                sss = StratifiedShuffleSplit(n_splits=1, train_size=budget, random_state=seed)
                sub_train_idx, _ = next(sss.split(X_train_full, y_train_full))
                x_sub = X_train_full[sub_train_idx]
                y_sub = y_train_full[sub_train_idx]
            else:
                x_sub = X_train_full
                y_sub = y_train_full

            pipeline = build_gradient_boosting_pipeline(random_state=seed)
            pipeline.fit(x_sub, y_sub)

            y_val_pred = pipeline.predict(X_val)
            metrics = compute_classification_metrics(y_true=y_val, y_pred=y_val_pred)
            budget_f1s.append(metrics.macro_f1)
            budget_accs.append(metrics.accuracy)

        mean_f1 = float(np.mean(budget_f1s))
        std_f1 = float(np.std(budget_f1s))
        mean_acc = float(np.mean(budget_accs))

        logger.info(f"Budget {budget:.0%}: Mean Macro-F1 = {mean_f1:.4f} ± {std_f1:.4f}, Accuracy = {mean_acc:.4f}")

        experiment_results.append({
            "budget_ratio": budget,
            "sample_count": n_samples,
            "mean_macro_f1": mean_f1,
            "std_macro_f1": std_f1,
            "mean_accuracy": mean_acc,
            "individual_f1s": [float(f) for f in budget_f1s],
        })

    out_dir = PROJECT_ROOT / "experiments" / "results" / "label_efficiency"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "label_efficiency_results.json", "w", encoding="utf-8") as f:
        json.dump(experiment_results, f, indent=2)

    # Plot curve
    budgets_pct = [r["budget_ratio"] * 100 for r in experiment_results]
    f1_means = [r["mean_macro_f1"] for r in experiment_results]
    f1_stds = [r["std_macro_f1"] for r in experiment_results]

    plt.figure(figsize=(8, 5))
    plt.errorbar(budgets_pct, f1_means, yerr=f1_stds, fmt="o-", color="teal", lw=2, capsize=5, label="Supervised Baseline (3 Seeds)")
    for i, txt in enumerate(f1_means):
        plt.annotate(f"{txt:.3f}", (budgets_pct[i], f1_means[i]), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9, fontweight="bold")
    plt.xlabel("Label Budget (% of Training Data)", fontsize=11)
    plt.ylabel("Validation Macro-F1", fontsize=11)
    plt.title("Label-Efficiency Learning Curve (KU-HAR)", fontsize=13, fontweight="bold")
    plt.ylim(0.7, 1.0)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "label_efficiency_curve.png", dpi=200)
    plt.close()

    logger.info(f"Label efficiency results saved to {out_dir}")
    print(f"\n[SUCCESS] Label efficiency experiment complete. Results saved in: {out_dir}")


def main():
    run_label_efficiency_experiment()


if __name__ == "__main__":
    main()
