"""Automated tests for preprocessing and normalization."""

import numpy as np
import pytest
from motionguard.data.preprocess import WaveformNormalizer


def test_waveform_normalizer_fitting():
    # Synthetic waveforms (10, 6, 300)
    N, C, T = 10, 6, 300
    waveforms = np.random.normal(loc=5.0, scale=2.0, size=(N, C, T)).astype(np.float32)

    normalizer = WaveformNormalizer()
    assert not normalizer.is_fitted
    normalizer.fit(waveforms)
    assert normalizer.is_fitted
    assert normalizer.channel_means.shape == (6, 1)
    assert normalizer.channel_stds.shape == (6, 1)

    norm_x = normalizer.transform(waveforms)
    assert norm_x.shape == (N, C, T)

    # Check that transformed mean is ~0 and std is ~1 per channel
    for c in range(6):
        ch_slice = norm_x[:, c, :]
        np.testing.assert_allclose(np.mean(ch_slice), 0.0, atol=1e-1)
        np.testing.assert_allclose(np.std(ch_slice), 1.0, atol=1e-1)


def test_waveform_normalizer_save_load(tmp_path):
    waveforms = np.random.randn(5, 6, 300).astype(np.float32)
    norm = WaveformNormalizer().fit(waveforms)

    save_path = tmp_path / "normalizer.joblib"
    norm.save(save_path)

    loaded = WaveformNormalizer.load(save_path)
    assert loaded.is_fitted
    np.testing.assert_array_equal(norm.channel_means, loaded.channel_means)
    np.testing.assert_array_equal(norm.channel_stds, loaded.channel_stds)
