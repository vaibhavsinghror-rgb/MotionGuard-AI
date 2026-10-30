# MotionGuard AI - System Architecture

## 1. High-Level Architecture

MotionGuard AI is designed as a modular, production-ready machine learning platform composed of 7 distinct layers:

```
[ Wearable Sensor Input (6 channels @ 100 Hz, 3 seconds) ]
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ 1. Data Ingestion & Preprocessing Layer                 │
│    - Streaming Download with SHA-256 verification       │
│    - Schema Validation & NaN/Inf Sanitization          │
│    - Waveform Reshaping to (N, 6, 300)                  │
│    - Leakage-Free Stratified Splitting (70/15/15)       │
└────────────────────────┬────────────────────────────────┘
                         │
        ┌────────────────┴────────────────┐
        ▼                                 ▼
┌───────────────────────────┐   ┌───────────────────────────┐
│ 2. Kinematic Feature Head │   │ 3. Deep Temporal Head     │
│    - 144 Time & Freq Feats│   │    - 1D Residual CNN      │
│    - FFT Band Energies    │   │    - Multi-scale stem     │
│    - Spectral Centroid    │   │    - Global Avg Pooling   │
└─────────────┬─────────────┘   └─────────────┬─────────────┘
              │                               │
              ▼                               ▼
┌───────────────────────────┐   ┌───────────────────────────┐
│ 4. Baseline Classifiers   │   │ Deep Learning Classifier  │
│    - Random Forest        │   │    - PyTorch AdamW / Cos  │
│    - Gradient Boosting    │   │    - Early Stopping       │
│    - SVM (RBF Kernel)     │   │                           │
└─────────────┬─────────────┘   └─────────────┬─────────────┘
              │                               │
              └───────────────┬───────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────┐
│ 5. Model Selection & Promotion Engine                   │
│    - Validation Macro-F1 Quality Gate (> 0.85)          │
│    - Artifact Persistence & SHA-256 Manifesting         │
└─────────────────────────────┬───────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────┐
│ 6. Confidence Calibration & Selective Prediction        │
│    - Temperature Scaling Optimization (NLL minimization)│
│    - Expected Calibration Error (ECE) audit             │
│    - Abstention logic: thresholding & human review flag │
└─────────────────────────────┬───────────────────────────┘
                              │
        ┌─────────────────────┴─────────────────────┐
        ▼                                           ▼
┌───────────────────────────┐               ┌───────────────────────────┐
│ 7a. Streamlit Dashboard   │               │ 7b. FastAPI REST Server   │
│     - 5-page Explorer     │               │     - POST /predict       │
│     - Plotly Waveforms    │               │     - GET /health         │
│     - Realtime Inference  │               │     - GET /model          │
└───────────────────────────┘               └───────────────────────────┘
```

## 2. Feature Engineering Pipeline (144 Physical Dimensions)
Across all 6 primary channels (`acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z`) plus 2 aggregate vector magnitudes (`acc_mag, gyro_mag`):

1. **Time-Domain (10 per channel):**
   - Mean, Standard Deviation, Root Mean Square (RMS)
   - Minimum, Maximum, Peak-to-Peak Amplitude
   - Unbiased Skewness, Kurtosis
   - Energy (mean square power)
   - Zero-Crossing Rate

2. **Frequency-Domain (8 per channel):**
   - Dominant Spectral Frequency & Peak Magnitude
   - Spectral Centroid (center of mass)
   - Spectral Entropy (distribution dispersion)
   - Power in Band 1: 0.0 – 2.5 Hz (posture & drift)
   - Power in Band 2: 2.5 – 5.0 Hz (cadence & locomotion)
   - Power in Band 3: 5.0 – 15.0 Hz (rapid limb transitions)
   - Power in Band 4: 15.0 – 50.0 Hz (high-frequency impacts)

## 3. Deep Learning 1D-CNN (`MotionGuardCNN1D`)
- Dual-scale kernel stem (`k=7` and `k=3`) capturing both broad gestures and sharp acceleration transients.
- Three residual convolutional stages with downsampling strides.
- Adaptive Average Pooling + Dropout regularization.
- Safe serialization using explicit `state_dict` and configuration dictionaries.
