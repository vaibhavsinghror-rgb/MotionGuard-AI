"""Leakage-safe train/val/test data splitting for MotionGuard AI."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from motionguard.config import DataConfig
from motionguard.utils.logging import get_logger

logger = get_logger(__name__)


def create_stratified_splits(
    y: np.ndarray,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Creates leakage-free stratified splits for train, validation, and test sets.
    
    Returns:
        (train_indices, val_indices, test_indices)
    """
    total = train_ratio + val_ratio + test_ratio
    train_ratio, val_ratio, test_ratio = train_ratio / total, val_ratio / total, test_ratio / total

    # First split: train vs (val + test)
    val_test_ratio = val_ratio + test_ratio
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=val_test_ratio, random_state=random_seed)
    train_idx, val_test_idx = next(sss1.split(np.zeros(len(y)), y))

    # Second split: val vs test
    val_in_val_test_ratio = val_ratio / val_test_ratio
    y_val_test = y[val_test_idx]
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=1.0 - val_in_val_test_ratio, random_state=random_seed)
    v_sub_idx, t_sub_idx = next(sss2.split(np.zeros(len(y_val_test)), y_val_test))

    val_idx = val_test_idx[v_sub_idx]
    test_idx = val_test_idx[t_sub_idx]

    logger.info(
        f"Created stratified splits: Train={len(train_idx)} ({len(train_idx)/len(y):.1%}), "
        f"Val={len(val_idx)} ({len(val_idx)/len(y):.1%}), "
        f"Test={len(test_idx)} ({len(test_idx)/len(y):.1%})"
    )

    # Sanity check: zero overlap
    assert len(set(train_idx).intersection(set(val_idx))) == 0, "Train and Val overlap!"
    assert len(set(train_idx).intersection(set(test_idx))) == 0, "Train and Test overlap!"
    assert len(set(val_idx).intersection(set(test_idx))) == 0, "Val and Test overlap!"

    return train_idx, val_idx, test_idx


def save_split_manifest(
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    manifest_path: Path,
    metadata: Optional[Dict] = None,
) -> None:
    """Saves split indices to disk for exact reproducibility."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": metadata or {},
        "counts": {
            "total": int(len(train_idx) + len(val_idx) + len(test_idx)),
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "indices": {
            "train": [int(i) for i in train_idx],
            "val": [int(i) for i in val_idx],
            "test": [int(i) for i in test_idx],
        },
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    logger.info(f"Split indices saved to {manifest_path}")


def load_split_manifest(manifest_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Loads split indices from manifest."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    train_idx = np.array(payload["indices"]["train"], dtype=np.int64)
    val_idx = np.array(payload["indices"]["val"], dtype=np.int64)
    test_idx = np.array(payload["indices"]["test"], dtype=np.int64)
    return train_idx, val_idx, test_idx
