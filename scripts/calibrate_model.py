"""CLI Script: Calibrate model confidence and evaluate prediction abstention."""

import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from motionguard.config import DataConfig, EvaluationConfig
from motionguard.data.load import load_processed_arrays
from motionguard.data.preprocess import WaveformNormalizer
from motionguard.data.split import load_split_manifest
from motionguard.evaluation.plots import plot_reliability_diagram
from motionguard.features.frequency_domain import extract_all_features
from motionguard.inference.predictor import MotionGuardPredictor
from motionguard.uncertainty.abstention import SelectivePredictor
from motionguard.uncertainty.calibration import TemperatureScaler
from motionguard.utils.logging import get_logger

logger = get_logger("calibrate_model")


def main():
    logger.info("Initializing confidence calibration and abstention analysis...")
    data_config = DataConfig.from_yaml()
    eval_config = EvaluationConfig.from_yaml()

    # Load validation data
    split_manifest = data_config.manifests_dir / "split_manifest.json"
    _, val_idx, _ = load_split_manifest(split_manifest)

    waveforms, labels, _ = load_processed_arrays(data_config.processed_dir)
    val_waveforms = waveforms[val_idx]
    val_labels = labels[val_idx]

    # Initialize predictor to load model
    predictor = MotionGuardPredictor()
    if not predictor.is_loaded:
        logger.error("Promoted model not loaded. Run train_all.py first.")
        sys.exit(1)

    model = predictor.model
    model_family = predictor.model_metadata.get("model_family", "deep_learning")
    logger.info(f"Extracting validation logits for {model_family} model...")

    if model_family == "deep_learning":
        norm_wf = predictor.normalizer.transform(val_waveforms) if predictor.normalizer else val_waveforms
        val_tensor = torch.from_numpy(norm_wf).float().to(predictor.device)
        model.eval()
        with torch.no_grad():
            logits = model(val_tensor).cpu().numpy()
    else:
        # Classical model: use predict_proba or decision_function
        X_val_feat = np.load(data_config.processed_dir / "features_val.npy")
        probs = model.predict_proba(X_val_feat)
        # Log-odds representation for temperature scaling
        eps = 1e-7
        logits = np.log(np.clip(probs, eps, 1.0 - eps))

    # Fit Temperature Scaler on validation logits
    scaler = TemperatureScaler(initial_temperature=1.0)
    scaler.fit(logits=logits, labels=val_labels)

    # Save calibration artifact
    calib_file = PROJECT_ROOT / "artifacts" / "calibration" / "calibration.json"
    scaler.save(calib_file)

    # Plot reliability diagram
    results_dir = PROJECT_ROOT / "experiments" / "results" / "calibration"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    bin_data = scaler.post_calibration_metrics.get("bins", [])
    plot_reliability_diagram(
        bin_data=bin_data,
        save_path=results_dir / "reliability_diagram.png",
        ece=scaler.post_calibration_metrics["ece"],
        title=f"Post-Calibration Reliability Diagram (T = {scaler.temperature:.3f})",
    )

    # Evaluate selective prediction / abstention trade-offs
    calib_probs = scaler.calibrate(logits)
    selective_engine = SelectivePredictor()
    thresholds = eval_config.evaluated_thresholds
    tradeoff_results = selective_engine.evaluate_risk_coverage(
        probabilities=calib_probs,
        true_labels=val_labels,
        thresholds=thresholds,
    )

    # Save risk-coverage JSON
    with open(results_dir / "risk_coverage_tradeoff.json", "w", encoding="utf-8") as f:
        json.dump(tradeoff_results, f, indent=2)

    # Plot Risk-Coverage Curve
    covs = [r["coverage"] for r in tradeoff_results]
    accs = [r["accuracy_accepted"] for r in tradeoff_results]
    thrs = [r["threshold"] for r in tradeoff_results]

    plt.figure(figsize=(8, 5))
    plt.plot(covs, accs, "o-", color="darkblue", lw=2, label="Accuracy on Accepted")
    for i, txt in enumerate(thrs):
        plt.annotate(f"Thr={txt}", (covs[i], accs[i]), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    plt.xlabel("Coverage (Proportion of Predictions Accepted)", fontsize=11)
    plt.ylabel("Accuracy on Accepted Predictions", fontsize=11)
    plt.title("Selective Prediction: Risk-Coverage Trade-Off", fontsize=13, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "risk_coverage_curve.png", dpi=200)
    plt.close()

    print("\n" + "=" * 65)
    print("      MOTIONGUARD AI CONFIDENCE CALIBRATION SUMMARY")
    print("=" * 65)
    print(f"Optimal Temperature: {scaler.temperature:.4f}")
    print(f"Pre-Calibration ECE:  {scaler.pre_calibration_metrics['ece']:.4f}")
    print(f"Post-Calibration ECE: {scaler.post_calibration_metrics['ece']:.4f}")
    print(f"Pre-Calibration NLL:  {scaler.pre_calibration_metrics['nll']:.4f}")
    print(f"Post-Calibration NLL: {scaler.post_calibration_metrics['nll']:.4f}")
    print("=" * 65)
    print("\n--- Selective Prediction Risk-Coverage Trade-Off ---")
    for r in tradeoff_results:
        print(f"Threshold: {r['threshold']:.2f} -> Coverage: {r['coverage']:.1%}, Accepted Acc: {r['accuracy_accepted']:.2%}")
    print(f"\n[ARTIFACTS]: Saved to {calib_file} and {results_dir}")


if __name__ == "__main__":
    main()
