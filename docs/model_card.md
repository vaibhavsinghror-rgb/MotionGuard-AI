# Model Card: MotionGuard AI (Gradient Boosting Pipeline)

## Model Details
- **Developer:** MotionGuard AI Team
- **Model Name:** HistGradientBoostingClassifier with StandardScaler on 144 engineered kinematic features
- **Model Version:** 1.0.0
- **Model Type:** Gradient Boosted Decision Trees (Ensemble)
- **License:** MIT License
- **Framework:** scikit-learn 1.9+
- **Artifact Path:** `artifacts/models/best_classical_model.joblib`
- **Metadata Path:** `artifacts/models/best_model_metadata.json`
- **Manifest:** `artifacts/manifests/model_manifest.json`

## Intended Use
- **Primary Use:** Human activity recognition (18 activities) from 3-second tri-axial accelerometer and gyroscope windows recorded at 100 Hz.
- **Secondary Use:** Confidence-calibrated selective prediction and movement anomaly flagging.
- **Out of Scope:** Clinical fall alarms, real-time life safety systems, automated medical diagnostics without human validation.

## Training Data & Splits
- **Source:** KU-HAR Dataset (Mendeley DOI: 10.17632/45f952y38r.1)
- **Total Windows:** 20,750 (6 channels x 300 samples)
- **Training Partition (70%):** 14,525 windows
- **Validation Partition (15%):** 3,112 windows (used for model comparison and temperature scaling)
- **Held-Out Test Partition (15%):** 3,113 windows (evaluated only after model selection)

## Empirical Evaluation Results

### Validation Partition Performance
| Metric | Score |
|---|---|
| Validation Accuracy | **97.11%** |
| Validation Balanced Accuracy | **96.39%** |
| Validation Macro-F1 | **0.9685** |
| Validation Macro-Precision | **97.41%** |
| Validation Macro-Recall | **96.39%** |
| Single-Window Latency | **6.64 ms** |

### Held-Out Test Partition Performance (Unseen Test Data)
| Metric | Score |
|---|---|
| Test Accuracy | **97.62%** |
| Test Balanced Accuracy | **97.45%** |
| Test Macro-F1 | **0.9778** |
| Test Macro-Precision | **98.15%** |
| Test Macro-Recall | **97.45%** |

### Confidence Calibration Audit
- **Method:** Post-hoc Temperature Scaling ($T = 1.6390$)
- **Pre-Calibration Expected Calibration Error (ECE):** 0.0173
- **Post-Calibration Expected Calibration Error (ECE):** **0.0047** (72.8% reduction)
- **Post-Calibration Negative Log-Likelihood (NLL):** 0.1022

### Selective Prediction (Risk-Coverage)
- At threshold 0.70: **97.1% coverage**, **98.34% accuracy** on accepted predictions.
- At threshold 0.90: **93.1% coverage**, **99.00% accuracy** on accepted predictions.
- At threshold 0.95: **89.4% coverage**, **99.21% accuracy** on accepted predictions.
