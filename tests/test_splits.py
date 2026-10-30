"""Automated tests for leakage-free data splits."""

import numpy as np
import pytest
from motionguard.data.split import create_stratified_splits


def test_stratified_splits_zero_leakage():
    # 100 samples across 5 classes
    y = np.repeat(np.arange(5), 20)
    train_idx, val_idx, test_idx = create_stratified_splits(
        y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42
    )

    # Zero overlap check
    s_train = set(train_idx)
    s_val = set(val_idx)
    s_test = set(test_idx)

    assert len(s_train.intersection(s_val)) == 0, "Train and Val overlap!"
    assert len(s_train.intersection(s_test)) == 0, "Train and Test overlap!"
    assert len(s_val.intersection(s_test)) == 0, "Val and Test overlap!"

    # Total coverage check
    assert len(s_train) + len(s_val) + len(s_test) == len(y)

    # Stratification check (each split should contain all 5 classes)
    assert len(np.unique(y[train_idx])) == 5
    assert len(np.unique(y[val_idx])) == 5
    assert len(np.unique(y[test_idx])) == 5
