"""CLI Script: Evaluate promoted model strictly on the held-out test set."""

import json
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from motionguard.config import DataConfig, EvaluationConfig
from motionguard.data.load import load_processed_arrays
from motionguard.data.split import load_split_manifest
from motionguard.evaluation.metrics import compute_classification_metrics
from motionguard.evaluation.plots import plot_confusion_matrix_heatmap
from motionguard.inference.predictor import MotionGuardPredictor
from motionguard.utils.logging import get_logger

logger = get_logger("evaluate_models")


def main():
    logger.info("Initializing held-out test evaluation...")
    data_config = DataConfig.from_yaml()
    eval_config = EvaluationConfig.from_yaml()

    # Verify split manifest
    split_manifest = data_config.manifests_dir / "split_manifest.json"
    if not split_manifest.exists():
        logger.error(f"Split manifest not found at {split_manifest}")
        sys.exit(1)

    _, _, test_idx = load_split_manifest(split_manifest)
    logger.info(f"Loaded held-out test set: {len(test_idx)} samples.")

    # Load test waveforms and labels
    waveforms, labels, _ = load_processed_arrays(data_config.processed_dir)
    test_waveforms = waveforms[test_idx]
    test_labels = labels[test_idx]

    # Initialize Predictor loading saved promoted artifacts from project directory
    predictor = MotionGuardPredictor()
    if not predictor.is_loaded:
        logger.error("Promoted model artifacts not found in artifacts/models/. Run train_all.py first.")
        sys.exit(1)

    logger.info(f"Loaded promoted model: {predictor.model_metadata.get('model_name')} for final test evaluation.")

    # Run inference across test set efficiently
    model_family = predictor.model_metadata.get("model_family", "deep_learning")
    
    if model_family == "classical":
        test_features_path = data_config.processed_dir / "features_test.npy"
        if test_features_path.exists():
            X_test_feat = np.load(test_features_path)
            y_pred = predictor.model.predict(X_test_feat)
        else:
            from motionguard.features.frequency_domain import extract_all_features
            features, _ = extract_all_features(test_waveforms, channel_names=data_config.channels)
            y_pred = predictor.model.predict(features)
    else:
        # Deep learning batch prediction
        norm_wf = predictor.normalizer.transform(test_waveforms) if predictor.normalizer else test_waveforms
        import torch
        tensor_test = torch.from_numpy(norm_wf).float().to(predictor.device)
        with torch.no_grad():
            logits = predictor.model(tensor_test)
            y_pred = torch.argmax(logits, dim=1).cpu().numpy()

    y_true = test_labels

    test_metrics = compute_classification_metrics(y_true=y_true, y_pred=y_pred)

    output_dir = eval_config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save test metrics JSON
    test_metrics_path = output_dir / "test_metrics.json"
    with open(test_metrics_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics.to_dict(), f, indent=2)

    # Save test confusion matrix
    class_names = [data_config.classes[i] for i in range(data_config.num_classes)]
    cm_path = output_dir / "test_confusion_matrix.png"
    plot_confusion_matrix_heatmap(
        cm=np.array(test_metrics.confusion_matrix),
        class_names=class_names,
        save_path=cm_path,
        title=f"Held-Out Test Confusion Matrix ({predictor.model_metadata.get('model_name')})",
    )

    # Save markdown summary
    summary_path = output_dir / "test_summary.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# MotionGuard AI - Held-Out Test Evaluation Summary\n\n")
        f.write(f"- **Evaluated Model:** {predictor.model_metadata.get('model_name')}\n")
        f.write(f"- **Total Test Samples:** {len(test_labels)}\n")
        f.write(f"- **Test Accuracy:** {test_metrics.accuracy:.4f}\n")
        f.write(f"- **Test Balanced Accuracy:** {test_metrics.balanced_accuracy:.4f}\n")
        f.write(f"- **Test Macro-F1:** {test_metrics.macro_f1:.4f}\n")
        f.write(f"- **Test Macro-Precision:** {test_metrics.macro_precision:.4f}\n")
        f.write(f"- **Test Macro-Recall:** {test_metrics.macro_recall:.4f}\n\n")
        f.write("## Per-Class Performance\n\n")
        f.write("| Class ID | Activity | Precision | Recall | F1-Score |\n")
        f.write("|---|---|---|---|---|\n")
        for cid in range(data_config.num_classes):
            cname = data_config.classes[cid]
            p = test_metrics.per_class_precision.get(cid, 0.0)
            r = test_metrics.per_class_recall.get(cid, 0.0)
            f1 = test_metrics.per_class_f1.get(cid, 0.0)
            f.write(f"| {cid} | {cname} | {p:.4f} | {r:.4f} | {f1:.4f} |\n")

    print("\n" + "=" * 65)
    print("      MOTIONGUARD AI HELD-OUT TEST EVALUATION RESULTS")
    print("=" * 65)
    print(f"Model:               {predictor.model_metadata.get('model_name')}")
    print(f"Test Accuracy:       {test_metrics.accuracy:.4f}")
    print(f"Test Balanced Acc:   {test_metrics.balanced_accuracy:.4f}")
    print(f"Test Macro-F1:       {test_metrics.macro_f1:.4f}")
    print(f"Test Macro-Precision:{test_metrics.macro_precision:.4f}")
    print(f"Test Macro-Recall:   {test_metrics.macro_recall:.4f}")
    print("=" * 65)
    print(f"\n[OUTPUT] Test report and plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
