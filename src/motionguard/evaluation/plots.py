"""Visualization and plotting utilities for MotionGuard AI."""

from pathlib import Path
from typing import Dict, List, Optional
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA


def plot_class_distribution(
    class_counts: Dict[str, int],
    save_path: Path,
    title: str = "KU-HAR Class Distribution (20,750 Windows)",
) -> None:
    """Plots and saves class frequency bar chart."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    labels = list(class_counts.keys())
    counts = list(class_counts.values())

    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(labels)), counts, color="steelblue", edgecolor="black", alpha=0.85)
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=10)
    plt.ylabel("Number of 3-Second Windows", fontsize=11)
    plt.title(title, fontsize=13, fontweight="bold")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, h + 15, f"{int(h)}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_representative_waveforms(
    waveforms: np.ndarray,
    labels: np.ndarray,
    class_mapping: Dict[int, str],
    save_path: Path,
    fs: float = 100.0,
    selected_classes: Optional[List[int]] = None,
) -> None:
    """Plots 6-channel raw waveforms for selected activities."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if selected_classes is None:
        selected_classes = [0, 8, 11, 14]  # Stand, Jump, Walk, Run

    time_sec = np.arange(waveforms.shape[2]) / fs
    ch_names = ["Acc X (m/s²)", "Acc Y (m/s²)", "Acc Z (m/s²)", "Gyro X (rad/s)", "Gyro Y (rad/s)", "Gyro Z (rad/s)"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    fig, axes = plt.subplots(len(selected_classes), 6, figsize=(18, 3.2 * len(selected_classes)), sharex=True)
    if len(selected_classes) == 1:
        axes = np.expand_dims(axes, 0)

    for row_idx, cls_id in enumerate(selected_classes):
        cls_name = class_mapping.get(cls_id, f"Class {cls_id}")
        # Find first sample matching class
        sample_indices = np.where(labels == cls_id)[0]
        if len(sample_indices) == 0:
            continue
        sample_idx = sample_indices[0]
        sample_wf = waveforms[sample_idx]  # (6, 300)

        for col_idx in range(6):
            ax = axes[row_idx, col_idx]
            ax.plot(time_sec, sample_wf[col_idx], color=colors[col_idx], lw=1.2)
            if row_idx == 0:
                ax.set_title(ch_names[col_idx], fontsize=11, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(f"{cls_name}\n(ID {cls_id})", fontsize=10, fontweight="bold")
            if row_idx == len(selected_classes) - 1:
                ax.set_xlabel("Time (seconds)", fontsize=9)
            ax.grid(True, linestyle=":", alpha=0.6)

    plt.suptitle("Representative 6-Channel Waveforms (3-Second Windows @ 100 Hz)", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_fft_power_spectrum(
    waveforms: np.ndarray,
    labels: np.ndarray,
    class_mapping: Dict[int, str],
    save_path: Path,
    fs: float = 100.0,
    selected_classes: Optional[List[int]] = None,
) -> None:
    """Plots frequency spectra (FFT magnitude) across activities."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if selected_classes is None:
        selected_classes = [0, 8, 11, 14]  # Stand, Jump, Walk, Run

    T = waveforms.shape[2]
    freqs = np.fft.rfftfreq(T, d=1.0 / fs)

    plt.figure(figsize=(10, 5))
    for cls_id in selected_classes:
        cls_name = class_mapping.get(cls_id, f"Class {cls_id}")
        sample_indices = np.where(labels == cls_id)[0]
        if len(sample_indices) == 0:
            continue
        # Average power spectrum of Acc magnitude across first 20 samples of this class
        cls_samples = waveforms[sample_indices[:20]]  # (M, 6, T)
        acc_mag = np.sqrt(np.sum(cls_samples[:, :3, :] ** 2, axis=1))  # (M, T)
        mags = np.abs(np.fft.rfft(acc_mag, axis=1))
        avg_mag = np.mean(mags, axis=0)

        plt.plot(freqs[1:], avg_mag[1:], label=cls_name, lw=1.8)

    plt.xlabel("Frequency (Hz)", fontsize=11)
    plt.ylabel("Mean FFT Magnitude", fontsize=11)
    plt.title("Acceleration Magnitude Power Spectra by Activity", fontsize=13, fontweight="bold")
    plt.xlim(0, 30)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_pca_features(
    features: np.ndarray,
    labels: np.ndarray,
    class_mapping: Dict[int, str],
    save_path: Path,
    n_samples: int = 1500,
) -> None:
    """2D PCA projection of engineered features to visualize class separability."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Subsample if large
    if len(features) > n_samples:
        indices = np.random.choice(len(features), size=n_samples, replace=False)
        x_sub = features[indices]
        y_sub = labels[indices]
    else:
        x_sub, y_sub = features, labels

    pca = PCA(n_components=2, random_state=42)
    pcs = pca.fit_transform(x_sub)
    var_exp = pca.explained_variance_ratio_

    plt.figure(figsize=(10, 8))
    # Select subset of classes or color palette
    scatter = plt.scatter(
        pcs[:, 0],
        pcs[:, 1],
        c=y_sub,
        cmap="tab20",
        alpha=0.65,
        s=25,
        edgecolor="none",
    )
    plt.xlabel(f"Principal Component 1 ({var_exp[0]:.1%} variance)", fontsize=11)
    plt.ylabel(f"Principal Component 2 ({var_exp[1]:.1%} variance)", fontsize=11)
    plt.title("2D PCA Projection of Engineered HAR Features", fontsize=13, fontweight="bold")
    plt.colorbar(scatter, label="Activity Class ID (0-17)", ticks=range(18))
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_confusion_matrix_heatmap(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Path,
    normalize: bool = True,
    title: str = "Confusion Matrix",
) -> None:
    """Plots normalized confusion matrix heatmap."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cm_display = cm.astype(float)
    if normalize:
        row_sums = cm_display.sum(axis=1, keepdims=True)
        cm_display = np.divide(cm_display, row_sums, out=np.zeros_like(cm_display), where=row_sums != 0)

    plt.figure(figsize=(14, 11))
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Normalized Recall" if normalize else "Counts"},
        annot_kws={"size": 7.5},
    )
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.xlabel("Predicted Activity", fontsize=12, fontweight="bold")
    plt.ylabel("True Activity", fontsize=12, fontweight="bold")
    plt.title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_reliability_diagram(
    bin_data: List[Dict],
    save_path: Path,
    ece: float,
    title: str = "Confidence Reliability Diagram",
) -> None:
    """Plots confidence calibration curve and expected vs observed accuracy."""
    save_path.parent.mkdir(parents=True, exist_ok=True)

    confs = [b["confidence"] for b in bin_data if b["count"] > 0]
    accs = [b["accuracy"] for b in bin_data if b["count"] > 0]

    plt.figure(figsize=(7, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration (Ideal)", lw=1.5)
    plt.plot(confs, accs, "s-", color="crimson", lw=2, label=f"Model (ECE = {ece:.3f})")
    plt.fill_between(confs, confs, accs, color="pink", alpha=0.4, label="Calibration Gap")

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Confidence (Predicted Probability)", fontsize=11)
    plt.ylabel("Observed Empirical Accuracy", fontsize=11)
    plt.title(title, fontsize=13, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
