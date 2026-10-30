"""Frequency-domain feature extraction and combined feature engineering for MotionGuard AI."""

from typing import List, Tuple
import numpy as np

from motionguard.features.time_domain import extract_axis_time_features


def extract_axis_frequency_features(
    signal: np.ndarray,
    fs: float = 100.0,
    prefix: str = "ch0",
) -> Tuple[np.ndarray, List[str]]:
    """Extracts frequency-domain features using FFT for a 2D signal (N, T) sampled at fs Hz.
    
    Features:
      - dominant_frequency
      - dominant_magnitude
      - spectral_centroid
      - spectral_entropy
      - band_energy_0_to_2_5Hz
      - band_energy_2_5_to_5Hz
      - band_energy_5_to_15Hz
      - band_energy_15_to_50Hz
    """
    assert signal.ndim == 2, f"Expected 2D array (N, T), got {signal.shape}"
    N, T = signal.shape
    eps = 1e-8

    # Compute one-sided FFT
    fft_vals = np.fft.rfft(signal, axis=1)
    magnitudes = np.abs(fft_vals)  # (N, T // 2 + 1)
    freqs = np.fft.rfftfreq(T, d=1.0 / fs)  # (T // 2 + 1,)

    # Power spectrum
    power = (magnitudes ** 2) / max(T, 1)
    total_power = np.sum(power, axis=1, keepdims=True) + eps

    # Dominant frequency (exclude DC bin at index 0)
    mag_no_dc = magnitudes[:, 1:]
    freqs_no_dc = freqs[1:]
    dom_indices = np.argmax(mag_no_dc, axis=1)
    dominant_frequency = freqs_no_dc[dom_indices]
    dominant_magnitude = np.max(mag_no_dc, axis=1)

    # Spectral centroid: sum(f * P(f)) / sum(P(f))
    spectral_centroid = np.sum(power * freqs.reshape(1, -1), axis=1) / total_power.ravel()

    # Spectral entropy: -sum(p_norm * log(p_norm))
    norm_power = power / total_power
    log_norm_power = np.log(norm_power + eps)
    spectral_entropy = -np.sum(norm_power * log_norm_power, axis=1)

    # Band energies
    def get_band_power(f_min: float, f_max: float) -> np.ndarray:
        mask = (freqs >= f_min) & (freqs < f_max)
        if not np.any(mask):
            return np.zeros(N, dtype=np.float32)
        return np.sum(power[:, mask], axis=1).astype(np.float32)

    band1 = get_band_power(0.0, 2.5)
    band2 = get_band_power(2.5, 5.0)
    band3 = get_band_power(5.0, 15.0)
    band4 = get_band_power(15.0, 50.0)

    feature_matrix = np.column_stack([
        dominant_frequency,
        dominant_magnitude,
        spectral_centroid,
        spectral_entropy,
        band1,
        band2,
        band3,
        band4,
    ]).astype(np.float32)

    feature_names = [
        f"{prefix}_dom_freq",
        f"{prefix}_dom_mag",
        f"{prefix}_spec_centroid",
        f"{prefix}_spec_entropy",
        f"{prefix}_band_0_2.5Hz",
        f"{prefix}_band_2.5_5Hz",
        f"{prefix}_band_5_15Hz",
        f"{prefix}_band_15_50Hz",
    ]

    return feature_matrix, feature_names


def extract_all_features(
    waveforms: np.ndarray,
    channel_names: List[str],
    fs: float = 100.0,
) -> Tuple[np.ndarray, List[str]]:
    """Extracts combined time and frequency features across all channels of (N, 6, 300).
    
    Returns:
      feature_matrix: np.ndarray of shape (N, num_features)
      all_feature_names: list of column names
    """
    assert waveforms.ndim == 3, f"Expected 3D array (N, C, T), got {waveforms.shape}"
    N, C, T = waveforms.shape
    assert C == len(channel_names), f"Channel names count ({len(channel_names)}) does not match C ({C})"

    all_matrices = []
    all_names = []

    for c_idx in range(C):
        ch_name = channel_names[c_idx]
        sig_2d = waveforms[:, c_idx, :]  # (N, T)

        time_mat, time_names = extract_axis_time_features(sig_2d, prefix=ch_name)
        freq_mat, freq_names = extract_axis_frequency_features(sig_2d, fs=fs, prefix=ch_name)

        all_matrices.extend([time_mat, freq_mat])
        all_names.extend(time_names + freq_names)

    # Cross-channel magnitude features for Accel and Gyro
    # Accel magnitude: sqrt(ax^2 + ay^2 + az^2)
    acc_mag = np.sqrt(
        waveforms[:, 0, :] ** 2 + waveforms[:, 1, :] ** 2 + waveforms[:, 2, :] ** 2 + 1e-8
    )
    acc_time_mat, acc_time_names = extract_axis_time_features(acc_mag, prefix="acc_mag")
    acc_freq_mat, acc_freq_names = extract_axis_frequency_features(acc_mag, fs=fs, prefix="acc_mag")

    # Gyro magnitude: sqrt(gx^2 + gy^2 + gz^2)
    gyro_mag = np.sqrt(
        waveforms[:, 3, :] ** 2 + waveforms[:, 4, :] ** 2 + waveforms[:, 5, :] ** 2 + 1e-8
    )
    gyro_time_mat, gyro_time_names = extract_axis_time_features(gyro_mag, prefix="gyro_mag")
    gyro_freq_mat, gyro_freq_names = extract_axis_frequency_features(gyro_mag, fs=fs, prefix="gyro_mag")

    all_matrices.extend([acc_time_mat, acc_freq_mat, gyro_time_mat, gyro_freq_mat])
    all_names.extend(acc_time_names + acc_freq_names + gyro_time_names + gyro_freq_names)

    combined_features = np.column_stack(all_matrices).astype(np.float32)
    # Check for any remaining NaNs or Infs
    combined_features = np.nan_to_num(combined_features, nan=0.0, posinf=0.0, neginf=0.0)

    return combined_features, all_names
