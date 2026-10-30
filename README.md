# MotionGuard AI 🛡️
## Confidence-Aware, Label-Efficient Human Activity Recognition and Movement Anomaly Analysis

[![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tests Passing](https://img.shields.io/badge/tests-17%20passed-brightgreen.svg)](tests/)

MotionGuard AI is a production-quality wearable sensor intelligence platform built on the verified **KU-HAR** benchmark. It provides robust multi-class human activity classification, post-hoc confidence calibration, selective prediction abstention, label-efficient active learning, and movement anomaly detection.

---

## 🚀 Key Results & Highlights

| Benchmark Target | Method | Metric / Result |
|---|---|---|
| **Activity Classification** | Gradient Boosting (144 Kinematic Feats) | **97.62% Test Accuracy**, **0.9778 Test Macro-F1** |
| **Random Forest Baseline** | 150 Decision Trees | **95.24% Val Accuracy**, **0.9470 Val Macro-F1** |
| **Confidence Calibration** | Temperature Scaling ($T = 1.6390$) | **ECE dropped from 0.0173 → 0.0047** (-72.8%) |
| **Selective Abstention** | Threshold $\tau = 0.90$ | **99.00% Accuracy on Accepted** (93.1% coverage) |
| **Label Efficiency** | 10% Label Budget (1,452 samples) | **90.64% Macro-F1** across 3 seeds |
| **Movement Anomaly Detection** | Isolation Forest on Out-of-Distribution | **0.9726 ROC-AUC** on held-out activity |
| **Inference Latency** | Single 3-second window | **6.64 ms** |

---

## 📂 Repository Structure

```text
MotionGuard-AI/
├── README.md                           # Master project documentation
├── LICENSE                             # MIT Open Source License
├── pyproject.toml                      # Standard packaging & dependencies
├── requirements.txt                    # Pinned package requirements
├── configs/                            # Central configuration files
│   ├── data.yaml                       # Dataset URLs, checksums, and mappings
│   ├── training.yaml                   # Hyperparameters and device selection
│   ├── evaluation.yaml                 # Quality gates and calibration bounds
│   └── inference.yaml                  # Prediction engine and server settings
├── data/
│   ├── raw/                            # Verified KU-HAR archive and CSV
│   ├── processed/                      # Preprocessed arrays and 144 features
│   ├── manifests/                      # Dataset and validation JSON reports
│   └── README.md                       # Data schema documentation
├── artifacts/                          # Persistent model artifacts
│   ├── models/
│   │   ├── best_classical_model.joblib # Promoted gradient boosting model
│   │   ├── best_model_metadata.json    # Promoted model provenance
│   │   ├── class_mapping.json          # 18-class mapping (0: Stand ... 17: Table-tennis)
│   │   └── anomaly_detector.joblib     # Isolation forest movement anomaly detector
│   ├── preprocessors/
│   │   └── waveform_normalizer.joblib  # Training-fitted sensor normalizer
│   ├── calibration/
│   │   └── calibration.json            # Temperature scaling parameters
│   ├── manifests/
│   │   └── model_manifest.json         # Checksummed deployment audit
│   └── export/                         # Bundled deployment archives
├── src/motionguard/                    # Production Python package
│   ├── config.py                       # Dataclass configuration loader
│   ├── data/                           # Ingestion, validation, loading, splitting
│   ├── features/                       # 144 time- and frequency-domain features
│   ├── models/                         # RF, Gradient Boosting, SVM, 1D-CNN
│   ├── uncertainty/                    # Temperature scaling, abstention, ECE
│   ├── anomaly/                        # Movement anomaly detector & protocols
│   ├── evaluation/                     # Academic metrics, plots, leaderboard
│   ├── inference/                      # Reusable predictor and Pydantic schemas
│   └── utils/                          # Reproducibility, seed, logging
├── app/
│   ├── dashboard.py                    # 5-Page Streamlit visual explorer
│   └── api.py                          # Production FastAPI REST inference service
├── scripts/                            # Standalone executable CLI commands
│   ├── download_data.py                # Streaming download & SHA-256 verification
│   ├── validate_data.py                # Schema & data integrity audit
│   ├── run_eda.py                      # EDA report and feature extraction
│   ├── train_all.py                    # Multi-model training & model promotion
│   ├── evaluate_models.py              # Held-out test set evaluation
│   ├── calibrate_model.py              # Temperature scaling & risk-coverage
│   ├── export_model.py                 # Deployment bundle packager
│   └── verify_saved_model.py           # Fresh-process persistence audit
├── experiments/
│   ├── label_efficiency.py             # 10%, 25%, 50%, 100% budget experiments
│   ├── anomaly_experiments.py          # Out-of-distribution anomaly experiment
│   └── results/                        # Figures, curves, and JSON reports
├── tests/                              # Automated pytest suite (17 passing)
└── docs/                               # Detailed academic technical reports
```

---

## 🛠️ Quickstart & Reproduction Commands

### 1. Environment Setup
```bash
# Clone or navigate to the repository
cd MotionGuard-AI

# Install dependencies in editable mode with dev tools
python -m pip install -e ".[dev]"
```

### 2. Pipeline Execution (Independent Commands)
```bash
# 1. Download KU-HAR from verified Mendeley Data URL with SHA-256 verification
python scripts/download_data.py

# 2. Validate dataset schema, channel counts, and integrity
python scripts/validate_data.py

# 3. Generate Exploratory Data Analysis (EDA) and extract kinematic features
python scripts/run_eda.py

# 4. Train candidate models and automatically promote the best validated model
python scripts/train_all.py

# 5. Evaluate the promoted model on the held-out test partition
python scripts/evaluate_models.py

# 6. Calibrate confidence via Temperature Scaling and audit abstention
python scripts/calibrate_model.py

# 7. Run research experiments
python experiments/label_efficiency.py
python experiments/anomaly_experiments.py

# 8. Execute the mandatory persistence verification in a fresh process
python scripts/verify_saved_model.py

# 9. Run automated unit test suite
pytest
```

### 3. Launch Applications
```bash
# Launch the interactive Streamlit dashboard
streamlit run app/dashboard.py

# Or launch the FastAPI REST service
uvicorn app.api:app --reload --port 8000
```

---

## 🔬 Dataset Overview (KU-HAR)
- **Source:** Mendeley Data (DOI: `10.17632/45f952y38r.1`)
- **Total Windows:** 20,750 windows (3.0 seconds @ 100 Hz = 300 samples)
- **Sensor Modalities:** 6 Channels (Accelerometer X/Y/Z, Gyroscope X/Y/Z)
- **18 Target Classes:**
  `0: Stand`, `1: Sit`, `2: Talk-sit`, `3: Talk-stand`, `4: Stand-sit`, `5: Lay`,
  `6: Lay-stand`, `7: Pick`, `8: Jump`, `9: Push-up`, `10: Sit-up`, `11: Walk`,
  `12: Walk-backward`, `13: Walk-circle`, `14: Run`, `15: Stair-up`, `16: Stair-down`, `17: Table-tennis`
- **Splits:** 70% Train (14,525), 15% Validation (3,112), 15% Held-Out Test (3,113).

---

## 🛡️ Inference Interface & Selective Abstention

The prediction engine outputs structured predictions with calibrated confidence:

```json
{
  "activity": "Walk",
  "class_id": 11,
  "confidence": 0.9834,
  "accepted": true,
  "status": "prediction_available",
  "probabilities": {
    "Walk": 0.9834,
    "Walk-circle": 0.0112,
    "Walk-backward": 0.0031
  },
  "model_version": "Gradient Boosting",
  "abstention_reason": null
}
```

If calibrated confidence falls below the acceptance threshold ($\tau = 0.70$ default), the system abstains gracefully:

```json
{
  "activity": "Talk-sit",
  "class_id": 2,
  "confidence": 0.5412,
  "accepted": false,
  "status": "uncertain_human_review_recommended",
  "abstention_reason": "Predicted confidence (54.12%) is below the required acceptance threshold (70.00%)."
}
```

---

## ⚠️ Scientific & Clinical Limitations

1. **Not a Medical Device:** Trained on controlled smartphone data from 90 volunteers. It has not been clinically certified.
2. **No Fall Certification:** KU-HAR does not contain genuine medical emergency falls. The system must not be used as a certified fall detection or medical alarm system.
3. **Sensor Orientation:** Performance assumes trouser-pocket orientation; inverted or loose sensors will alter sensor axes.

---

## 📜 Citation
```bibtex
@article{sikder2020kuhar,
  title={KU-HAR: An open dataset for heterogeneous human activity recognition},
  author={Sikder, Niloy and Nahid, Abdullah-Al},
  journal={Mendeley Data, V1},
  year={2020},
  doi={10.17632/45f952y38r.1}
}
```
