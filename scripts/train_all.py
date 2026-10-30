"""CLI Script: Train all baseline classical models and 1D-CNN on KU-HAR."""

import json
import sys
import time
from pathlib import Path
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from motionguard.config import DataConfig, TrainingConfig
from motionguard.data.load import load_processed_arrays
from motionguard.data.preprocess import WaveformNormalizer
from motionguard.data.split import load_split_manifest
from motionguard.evaluation.compare_models import ModelSelector
from motionguard.evaluation.plots import plot_confusion_matrix_heatmap
from motionguard.models.training import train_classical_model, train_cnn1d_model
from motionguard.utils.logging import get_logger
from motionguard.utils.reproducibility import seed_everything

logger = get_logger("train_all")


def main():
    logger.info("Initializing MotionGuard AI training pipeline...")
    data_config = DataConfig.from_yaml()
    train_config = TrainingConfig.from_yaml()
    seed_everything(train_config.seed)

    processed_dir = data_config.processed_dir
    manifests_dir = data_config.manifests_dir
    split_manifest = manifests_dir / "split_manifest.json"

    if not (processed_dir / "waveforms.npy").exists() or not split_manifest.exists():
        logger.error("Processed data or split manifest missing. Run run_eda.py first.")
        sys.exit(1)

    # Load waveforms and labels
    waveforms, labels, _ = load_processed_arrays(processed_dir)
    train_idx, val_idx, test_idx = load_split_manifest(split_manifest)

    logger.info(f"Loaded splits: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

    # Load engineered features for classical baselines
    X_train_feat = np.load(processed_dir / "features_train.npy")
    X_val_feat = np.load(processed_dir / "features_val.npy")
    y_train = labels[train_idx]
    y_val = labels[val_idx]

    # Load normalized waveforms for Deep Learning
    normalizer_path = PROJECT_ROOT / "artifacts" / "preprocessors" / "waveform_normalizer.joblib"
    normalizer = WaveformNormalizer.load(normalizer_path)
    X_train_wf = normalizer.transform(waveforms[train_idx])
    X_val_wf = normalizer.transform(waveforms[val_idx])

    results_dir = PROJECT_ROOT / "experiments" / "results" / "models"
    results_dir.mkdir(parents=True, exist_ok=True)

    model_results = {}
    model_objects = {}

    # 1. Train Random Forest
    logger.info("--- Training Random Forest Baseline ---")
    rf_pipeline, rf_metrics, rf_time = train_classical_model(
        "random_forest", X_train_feat, y_train, X_val_feat, y_val, config=train_config
    )
    model_results["Random Forest"] = (rf_pipeline, rf_metrics, rf_time)
    model_objects["Random Forest"] = rf_pipeline

    # 2. Train Gradient Boosting
    logger.info("--- Training Gradient Boosting Baseline ---")
    gb_pipeline, gb_metrics, gb_time = train_classical_model(
        "gradient_boosting", X_train_feat, y_train, X_val_feat, y_val, config=train_config
    )
    model_results["Gradient Boosting"] = (gb_pipeline, gb_metrics, gb_time)
    model_objects["Gradient Boosting"] = gb_pipeline

    # 3. Train SVM (subsampled for computationally reasonable search on classical features)
    logger.info("--- Training Support Vector Machine Baseline ---")
    # Subsample training features to 5,000 for SVM to prevent excessive quadratic fit time
    svm_train_size = min(5000, len(X_train_feat))
    svm_idx = np.random.choice(len(X_train_feat), size=svm_train_size, replace=False)
    svm_pipeline, svm_metrics, svm_time = train_classical_model(
        "svm", X_train_feat[svm_idx], y_train[svm_idx], X_val_feat, y_val, config=train_config
    )
    model_results["SVM"] = (svm_pipeline, svm_metrics, svm_time)
    model_objects["SVM"] = svm_pipeline

    # 4. Train Deep Learning 1D-CNN
    logger.info("--- Training 1D-CNN Time-Series Model ---")
    cnn_checkpoint_path = results_dir / "cnn1d_best.pt"
    cnn_model, cnn_metrics, cnn_history, cnn_time = train_cnn1d_model(
        train_x=X_train_wf,
        train_y=y_train,
        val_x=X_val_wf,
        val_y=y_val,
        config=train_config,
        save_checkpoint_path=cnn_checkpoint_path,
    )
    model_results["1D-CNN"] = (cnn_model, cnn_metrics, cnn_time)
    model_objects["1D-CNN"] = cnn_model

    # Save training history
    with open(results_dir / "cnn1d_history.json", "w", encoding="utf-8") as f:
        json.dump(cnn_history, f, indent=2)

    # 5. Model Comparison and Selection
    logger.info("Comparing all candidate models on the validation set...")
    selector = ModelSelector()
    ranked_df = selector.compare_and_rank(model_results)
    ranked_df.to_csv(results_dir / "model_comparison.csv", index=False)

    # Promote the top model
    manifest = selector.promote_best_model(
        ranked_df=ranked_df,
        model_objects=model_objects,
        class_mapping=data_config.classes,
        dataset_manifest_path=data_config.dataset_manifest,
    )

    # Save validation confusion matrix plot for the promoted best model
    best_model_name = ranked_df.iloc[0]["model_name"]
    best_metrics = model_results[best_model_name][1]
    class_names = [data_config.classes[i] for i in range(data_config.num_classes)]
    cm_plot_path = results_dir / f"confusion_matrix_{best_model_name.replace(' ', '_').lower()}.png"
    plot_confusion_matrix_heatmap(
        cm=np.array(best_metrics.confusion_matrix),
        class_names=class_names,
        save_path=cm_plot_path,
        title=f"Validation Confusion Matrix ({best_model_name})",
    )

    print("\n" + "=" * 65)
    print("           MOTIONGUARD AI MODEL LEADERBOARD (VALIDATION)")
    print("=" * 65)
    print(ranked_df.to_string(index=False))
    print("=" * 65)
    print(f"\n[PROMOTED BEST MODEL]: {manifest['model_name']} (Val Macro-F1: {manifest['validation_performance']['macro_f1']:.4f})")
    print(f"[ARTIFACTS]: Saved to {train_config.models_dir}")


if __name__ == "__main__":
    main()
