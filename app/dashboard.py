"""Interactive Streamlit Dashboard for MotionGuard AI.

Fulfills Section 15 of project specifications.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Configure page layout
st.set_page_config(
    page_title="MotionGuard AI | Wearable Sensor Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Cache predictor and data loaders
@st.cache_resource
def load_predictor():
    from motionguard.inference.predictor import MotionGuardPredictor
    return MotionGuardPredictor()

@st.cache_data
def load_eda_summary():
    p = PROJECT_ROOT / "experiments" / "results" / "eda" / "eda_summary.json"
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

@st.cache_data
def load_comparison_data():
    p = PROJECT_ROOT / "experiments" / "results" / "models" / "model_comparison.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()

@st.cache_data
def load_test_metrics():
    p = PROJECT_ROOT / "experiments" / "results" / "evaluation" / "test_metrics.json"
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

@st.cache_data
def load_sample_waveforms():
    wf_path = PROJECT_ROOT / "data" / "processed" / "waveforms.npy"
    lb_path = PROJECT_ROOT / "data" / "processed" / "labels.npy"
    if wf_path.exists() and lb_path.exists():
        # load first 200 samples
        wf = np.load(wf_path, mmap_mode="r")[:200]
        lb = np.load(lb_path)[:200]
        return wf, lb
    return None, None


# Sidebar Navigation
st.sidebar.title("🛡️ MotionGuard AI")
st.sidebar.caption("Confidence-Aware Wearable Sensor Intelligence")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "A. Overview",
        "B. Dataset Explorer",
        "C. Model Performance",
        "D. Activity Prediction",
        "E. Model Information",
    ],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Dataset:** KU-HAR (20,750 windows, 18 classes)\n\n"
    "**Sensors:** Accelerometer & Gyroscope (100 Hz)\n\n"
    "**License:** CC BY 4.0 / Research Use"
)


# -------------------------------------------------------------
# PAGE A: OVERVIEW
# -------------------------------------------------------------
if page == "A. Overview":
    st.title("🛡️ MotionGuard AI: Wearable Sensor Intelligence")
    st.markdown(
        "A rigorous, production-quality platform for human activity recognition (HAR), "
        "confidence calibration, and movement anomaly analysis based on the verified **KU-HAR** benchmark."
    )

    eda = load_eda_summary()
    comp_df = load_comparison_data()
    test_met = load_test_metrics()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Windows", f"{eda.get('total_windows', 20750):,}")
    with col2:
        st.metric("Activity Classes", f"{eda.get('num_classes', 18)}")
    with col3:
        best_name = comp_df.iloc[0]["model_name"] if not comp_df.empty else "N/A"
        st.metric("Selected Model", best_name)
    with col4:
        val_f1 = comp_df.iloc[0]["val_macro_f1"] if not comp_df.empty else 0.0
        st.metric("Validation Macro-F1", f"{val_f1:.4f}")

    st.markdown("### 📋 Core System Capabilities")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            "#### 🏃 Robust Activity Recognition\n"
            "- Multi-model benchmarking: Random Forest, SVM, Gradient Boosting, 1D-CNN.\n"
            "- 144 physically grounded time- and frequency-domain kinematic features.\n"
            "- Leakage-free stratified splitting protocol."
        )
    with c2:
        st.markdown(
            "#### 🎯 Confidence & Abstention\n"
            "- Post-hoc Temperature Scaling to fix overconfidence.\n"
            "- Selective prediction: explicitly returns `uncertain` when confidence falls below acceptance threshold.\n"
            "- Empirical Risk-Coverage trade-off evaluation."
        )
    with c3:
        st.markdown(
            "#### 🔬 Academic Integrity\n"
            "- Programmatic dataset verification with SHA-256 integrity check.\n"
            "- Validation-based model selection; held-out test evaluation only after freeze.\n"
            "- Clear distinction between statistical anomaly detection and clinical claims."
        )

    # Verification status card
    st.markdown("### 🔍 Dataset & Artifact Integrity Status")
    manifest_path = PROJECT_ROOT / "data" / "manifests" / "dataset_manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            d_man = json.load(f)
        st.success(
            f"**Dataset Source Verified:** {d_man.get('dataset_name')} (DOI: {d_man.get('doi')})\n\n"
            f"- **SHA-256 Checksum:** `{d_man.get('archive_sha256')}`\n"
            f"- **Verification Status:** `{d_man.get('verification_status')}`\n"
            f"- **Primary File:** `{d_man.get('primary_csv_filename')}` ({d_man.get('csv_file_size_bytes', 0)/(1024*1024):.1f} MB)"
        )
    else:
        st.warning("Dataset manifest not generated yet. Please run download_data.py.")


# -------------------------------------------------------------
# PAGE B: DATASET EXPLORER
# -------------------------------------------------------------
elif page == "B. Dataset Explorer":
    st.title("📊 KU-HAR Dataset Explorer")
    st.markdown(
        "Inspection of the 20,750 3-second subsamples collected from 90 participants at 100 Hz across 18 activities."
    )

    eda = load_eda_summary()
    counts = eda.get("class_counts", {})
    if counts:
        df_counts = pd.DataFrame(list(counts.items()), columns=["Activity", "Count"]).sort_values(by="Count", ascending=False)
        fig = px.bar(
            df_counts,
            x="Activity",
            y="Count",
            color="Count",
            color_continuous_scale="Viridis",
            title="Class Distribution Across 18 Activities",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🌊 Interactive Sensor Waveform Explorer")
    waveforms, labels = load_sample_waveforms()
    if waveforms is not None:
        predictor = load_predictor()
        class_map = predictor.class_mapping

        sample_idx = st.slider("Select Sample Index (from first 200)", 0, len(waveforms) - 1, 0)
        selected_wf = waveforms[sample_idx]  # (6, 300)
        true_activity = class_map.get(int(labels[sample_idx]), f"Class {labels[sample_idx]}")

        st.subheader(f"Activity: **{true_activity}** (Sample #{sample_idx})")

        time_axis = np.arange(300) / 100.0
        ch_names = ["Acc X", "Acc Y", "Acc Z", "Gyro X", "Gyro Y", "Gyro Z"]

        fig_wf = go.Figure()
        for c in range(3):
            fig_wf.add_trace(go.Scatter(x=time_axis, y=selected_wf[c], mode="lines", name=f"{ch_names[c]} (m/s²)"))
        fig_wf.update_layout(title="Linear Acceleration Channels (X, Y, Z)", xaxis_title="Time (seconds)", yaxis_title="m/s²")
        st.plotly_chart(fig_wf, use_container_width=True)

        fig_gyro = go.Figure()
        for c in range(3, 6):
            fig_gyro.add_trace(go.Scatter(x=time_axis, y=selected_wf[c], mode="lines", name=f"{ch_names[c]} (rad/s)"))
        fig_gyro.update_layout(title="Angular Velocity Channels (Gyro X, Y, Z)", xaxis_title="Time (seconds)", yaxis_title="rad/s")
        st.plotly_chart(fig_gyro, use_container_width=True)


# -------------------------------------------------------------
# PAGE C: MODEL PERFORMANCE
# -------------------------------------------------------------
elif page == "C. Model Performance":
    st.title("📈 Model Comparison & Benchmark Results")
    st.markdown("Evaluation of classical and deep learning models on validation and held-out test partitions.")

    comp_df = load_comparison_data()
    if not comp_df.empty:
        st.subheader("Leaderboard (Validation Partition)")
        st.dataframe(
            comp_df.style.format({
                "val_macro_f1": "{:.4f}",
                "val_accuracy": "{:.4f}",
                "val_balanced_acc": "{:.4f}",
                "val_macro_precision": "{:.4f}",
                "val_macro_recall": "{:.4f}",
                "inference_latency_ms": "{:.2f} ms",
                "model_size_mb": "{:.2f} MB",
                "training_time_sec": "{:.2f} s",
            }),
            use_container_width=True,
        )

        fig_bar = px.bar(
            comp_df,
            x="model_name",
            y="val_macro_f1",
            color="val_accuracy",
            text="val_macro_f1",
            title="Model Comparison by Validation Macro-F1",
            labels={"model_name": "Model", "val_macro_f1": "Macro-F1 Score"},
        )
        fig_bar.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        st.plotly_chart(fig_bar, use_container_width=True)

    # Held-Out Test Evaluation
    st.markdown("---")
    st.subheader("🎯 Held-Out Test Evaluation (Final Unseen Test Partition)")
    test_met = load_test_metrics()
    if test_met:
        tc1, tc2, tc3, tc4 = st.columns(4)
        with tc1:
            st.metric("Test Accuracy", f"{test_met.get('accuracy', 0):.4f}")
        with tc2:
            st.metric("Test Balanced Acc", f"{test_met.get('balanced_accuracy', 0):.4f}")
        with tc3:
            st.metric("Test Macro-F1", f"{test_met.get('macro_f1', 0):.4f}")
        with tc4:
            st.metric("Test Samples", f"{test_met.get('num_samples', 0):,}")

        # Confusion Matrix Image
        cm_img = PROJECT_ROOT / "experiments" / "results" / "evaluation" / "test_confusion_matrix.png"
        if cm_img.exists():
            st.image(str(cm_img), caption="Normalized Confusion Matrix on Held-Out Test Partition", use_container_width=True)
    else:
        st.info("Run `python scripts/evaluate_models.py` to generate held-out test metrics.")

    # Calibration & Label Efficiency Results
    st.markdown("---")
    st.subheader("🔬 Calibration & Research Curves")
    rc1, rc2 = st.columns(2)
    with rc1:
        calib_img = PROJECT_ROOT / "experiments" / "results" / "calibration" / "reliability_diagram.png"
        if calib_img.exists():
            st.image(str(calib_img), caption="Post-Calibration Reliability Diagram", use_container_width=True)
    with rc2:
        le_img = PROJECT_ROOT / "experiments" / "results" / "label_efficiency" / "label_efficiency_curve.png"
        if le_img.exists():
            st.image(str(le_img), caption="Label-Efficiency Learning Curve (10% to 100%)", use_container_width=True)


# -------------------------------------------------------------
# PAGE D: ACTIVITY PREDICTION
# -------------------------------------------------------------
elif page == "D. Activity Prediction":
    st.title("⚡ Interactive Activity Inference")
    st.markdown("Run confidence-aware predictions on wearable sensor windows with selective abstention.")

    predictor = load_predictor()
    if not predictor.is_loaded:
        st.error("No saved model found in `artifacts/models/`. Please run the training pipeline first.")
    else:
        waveforms, labels = load_sample_waveforms()
        class_map = predictor.class_mapping

        st.sidebar.markdown("### Inference Settings")
        conf_thresh = st.sidebar.slider(
            "Confidence Acceptance Threshold",
            min_value=0.50,
            max_value=0.99,
            value=0.70,
            step=0.01,
            help="If confidence is below this threshold, the model abstains and flags for review.",
        )

        input_mode = st.radio("Input Source", ["Preprocessed Benchmark Example", "Simulated Sensor Window"], horizontal=True)

        if input_mode == "Preprocessed Benchmark Example" and waveforms is not None:
            sel_idx = st.number_input("Select Example Index (0 to 199)", min_value=0, max_value=len(waveforms) - 1, value=12)
            input_window = waveforms[sel_idx]
            actual_label = class_map.get(int(labels[sel_idx]), "Unknown")
            st.caption(f"Ground Truth Activity for selected window: **{actual_label}**")
        else:
            # Generate simulated walking or running pattern
            t = np.arange(300) / 100.0
            ax = 1.2 * np.sin(2 * np.pi * 1.8 * t) + np.random.normal(0, 0.2, 300)
            ay = 9.8 + 0.8 * np.cos(2 * np.pi * 1.8 * t) + np.random.normal(0, 0.2, 300)
            az = 0.5 * np.sin(2 * np.pi * 3.6 * t) + np.random.normal(0, 0.2, 300)
            gx = 0.8 * np.sin(2 * np.pi * 1.8 * t)
            gy = 0.4 * np.cos(2 * np.pi * 1.8 * t)
            gz = 0.2 * np.sin(2 * np.pi * 3.6 * t)
            input_window = np.vstack([ax, ay, az, gx, gy, gz]).astype(np.float32)

        if st.button("Run Inference", type="primary"):
            pred_resp = predictor.predict(input_window, threshold=conf_thresh)

            pcol1, pcol2 = st.columns([1, 2])
            with pcol1:
                st.markdown("### Prediction Outcome")
                if pred_resp.accepted:
                    st.success(f"### Recognized: **{pred_resp.activity}**")
                    st.metric("Confidence Score", f"{pred_resp.confidence:.2%}")
                    st.info(f"Status: `{pred_resp.status}`")
                else:
                    st.warning(f"### Proposed: **{pred_resp.activity}**")
                    st.metric("Confidence Score", f"{pred_resp.confidence:.2%}")
                    st.error(f"⚠️ **ABSTAINED:** {pred_resp.abstention_reason}")

            with pcol2:
                st.markdown("### Class Probability Distribution")
                probs = pred_resp.probabilities
                df_p = pd.DataFrame(list(probs.items()), columns=["Activity", "Probability"]).sort_values(by="Probability", ascending=False).head(8)
                fig_p = px.bar(
                    df_p,
                    x="Probability",
                    y="Activity",
                    orientation="h",
                    color="Probability",
                    color_continuous_scale="Blues",
                    range_x=[0, 1.0],
                )
                fig_p.update_layout(yaxis={"autorange": "reversed"})
                st.plotly_chart(fig_p, use_container_width=True)

            # Show waveform
            st.markdown("### Input 6-Channel Sensor Waveform")
            time_ax = np.arange(300) / 100.0
            fig_in = go.Figure()
            for c, name in enumerate(["Acc X", "Acc Y", "Acc Z", "Gyro X", "Gyro Y", "Gyro Z"]):
                fig_in.add_trace(go.Scatter(x=time_ax, y=input_window[c], mode="lines", name=name))
            fig_in.update_layout(xaxis_title="Time (seconds)", yaxis_title="Sensor Units (m/s² or rad/s)")
            st.plotly_chart(fig_in, use_container_width=True)


# -------------------------------------------------------------
# PAGE E: MODEL INFORMATION
# -------------------------------------------------------------
elif page == "E. Model Information":
    st.title("ℹ️ Model Architecture & Provenance Card")

    manifest_file = PROJECT_ROOT / "artifacts" / "manifests" / "model_manifest.json"
    if manifest_file.exists():
        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        st.json(manifest)
    else:
        st.warning("Model manifest not found. Please train models first.")

    st.markdown("### ⚠️ Known Scientific & Clinical Limitations")
    st.warning(
        "1. **Not a Clinical Device:** This system was trained on healthy volunteers under controlled smartphone data collection.\n"
        "2. **No Fall Certification:** KU-HAR does not contain genuine clinical emergency fall events; the model must NOT be used for real-time safety critical fall alarms.\n"
        "3. **Sensor Orientation Sensitivity:** Variations in pocket placement or smartphone orientation may alter performance."
    )
