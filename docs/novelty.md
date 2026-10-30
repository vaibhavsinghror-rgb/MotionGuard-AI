# MotionGuard AI - Scientific Contributions and Novelty

Unlike standard human activity recognition (HAR) projects that only report test accuracy on a standard split, MotionGuard AI introduces three key research contributions:

## 1. Confidence Calibration with Selective Abstention
Standard deep networks and tree ensembles frequently suffer from overconfidence on ambiguous transitions (e.g. `Talk-sit` vs `Sit`, or `Walk` vs `Walk-circle`).
MotionGuard AI introduces:
- **Temperature Scaling Optimization:** Directly minimizes cross-entropy loss on validation logits without altering accuracy, reducing Expected Calibration Error (ECE) from **0.0173 down to 0.0047** (over 72% reduction).
- **Abstention Policy:** Formulates an explicit threshold-based selective prediction rule. When maximum calibrated confidence falls below the acceptance threshold $\tau$, the system outputs `uncertain_human_review_recommended` with an explanatory diagnostic string rather than producing a noisy hallucinated prediction.
- **Risk-Coverage Guarantee:** At $\tau = 0.90$, accuracy among accepted predictions reaches **99.00%** with **93.1% coverage**.

## 2. Empirical Label-Efficiency Analysis
In practical wearable health applications, annotating continuous IMU sensor streams with high-precision ground truth is prohibitively expensive.
MotionGuard AI benchmarks learning efficiency across 10%, 25%, 50%, and 100% label budgets:
- Achieving **90.64% Macro-F1 with only 10% of labels** (1,452 samples).
- Achieving **93.64% Macro-F1 with 25% of labels**.
This demonstrates that well-engineered kinematic features enable rapid cold-start deployment with modest manual annotation effort.

## 3. Disentangled Movement Anomaly Detection
Rather than forcing all sensor windows into a closed-world 18-class softmax, MotionGuard AI incorporates an out-of-distribution detector:
- Evaluated on a **Leave-One-Class-Out** protocol (Activity 8 `Jump` held out).
- Achieves **0.9726 ROC-AUC** in separating normal gait and posture from unfamiliar explosive vertical movement.
- Clearly decouples statistical novelty from clinical hazard claims.
