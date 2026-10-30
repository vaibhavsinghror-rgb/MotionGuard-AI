"""CLI Script: Run Exploratory Data Analysis, Preprocessing, and Feature Engineering."""

import json
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from motionguard.config import DataConfig
from motionguard.data.load import load_raw_csv_as_arrays, save_processed_arrays
from motionguard.data.preprocess import WaveformNormalizer
from motionguard.data.split import create_stratified_splits, save_split_manifest
from motionguard.evaluation.plots import (
    plot_class_distribution,
    plot_fft_power_spectrum,
    plot_pca_features,
    plot_representative_waveforms,
)
from motionguard.features.frequency_domain import extract_all_features
from motionguard.utils.logging import get_logger
from motionguard.utils.reproducibility import seed_everything

logger = get_logger("run_eda")


def main():
    logger.info("Initializing EDA and preprocessing pipeline...")
    config = DataConfig.from_yaml()
    seed_everything(config.random_seed)

    csv_path = config.raw_dir / config.csv_filename
    if not csv_path.exists():
        logger.error(f"Raw CSV not found at {csv_path}. Run download_data.py first.")
        sys.exit(1)

    # 1. Load and reshape raw data
    waveforms, labels, serials = load_raw_csv_as_arrays(csv_path=csv_path, config=config)
    save_processed_arrays(waveforms, labels, config.processed_dir, serials=serials)

    # 2. Leakage-safe train/val/test splits
    train_idx, val_idx, test_idx = create_stratified_splits(
        y=labels,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        random_seed=config.random_seed,
    )
    split_manifest_path = config.manifests_dir / "split_manifest.json"
    save_split_manifest(
        train_idx, val_idx, test_idx, split_manifest_path,
        metadata={"random_seed": config.random_seed, "num_classes": config.num_classes}
    )

    # 3. Fit Waveform Normalizer on training data ONLY
    train_waveforms = waveforms[train_idx]
    normalizer = WaveformNormalizer()
    normalizer.fit(train_waveforms)
    normalizer_path = PROJECT_ROOT / "artifacts" / "preprocessors" / "waveform_normalizer.joblib"
    normalizer.save(normalizer_path)

    # 4. Extract engineered features for classical baselines
    logger.info("Extracting time & frequency domain features across all 6 channels...")
    channel_names = config.channels
    features, feature_names = extract_all_features(waveforms, channel_names=channel_names, fs=config.sampling_rate_hz)
    
    # Save features by split
    np.save(config.processed_dir / "features_train.npy", features[train_idx])
    np.save(config.processed_dir / "features_val.npy", features[val_idx])
    np.save(config.processed_dir / "features_test.npy", features[test_idx])
    with open(config.processed_dir / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2)
    logger.info(f"Extracted and saved {len(feature_names)} engineered features per sample.")

    # 5. Generate and save EDA visualizations
    eda_dir = PROJECT_ROOT / "experiments" / "results" / "eda"
    eda_dir.mkdir(parents=True, exist_ok=True)

    class_names_map = config.classes
    class_counts = {class_names_map[int(k)]: int(v) for k, v in zip(*np.unique(labels, return_counts=True))}

    logger.info("Generating EDA figures...")
    plot_class_distribution(class_counts, eda_dir / "class_distribution.png")
    plot_representative_waveforms(waveforms, labels, class_names_map, eda_dir / "representative_waveforms.png")
    plot_fft_power_spectrum(waveforms, labels, class_names_map, eda_dir / "frequency_spectra.png")
    plot_pca_features(features, labels, class_names_map, eda_dir / "feature_pca.png")

    # Save summary stats
    eda_summary = {
        "dataset_name": config.name,
        "total_windows": int(waveforms.shape[0]),
        "channels": config.channels,
        "samples_per_window": int(waveforms.shape[2]),
        "sampling_rate_hz": config.sampling_rate_hz,
        "num_classes": config.num_classes,
        "class_counts": class_counts,
        "engineered_feature_count": len(feature_names),
        "split_counts": {
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
    }
    with open(eda_dir / "eda_summary.json", "w", encoding="utf-8") as f:
        json.dump(eda_summary, f, indent=2)

    logger.info(f"EDA successfully completed. Artifacts and plots saved in {eda_dir}")
    print(f"\n[SUCCESS] EDA complete. Plots saved to: {eda_dir}")


if __name__ == "__main__":
    main()
