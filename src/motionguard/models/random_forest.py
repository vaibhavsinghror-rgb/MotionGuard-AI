"""Random Forest baseline classifier for MotionGuard AI."""

from typing import Any, Dict, Optional
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_random_forest_pipeline(
    params: Optional[Dict[str, Any]] = None,
    random_state: int = 42,
) -> Pipeline:
    """Creates a scikit-learn Pipeline with StandardScaler and RandomForestClassifier."""
    default_params = {
        "n_estimators": 150,
        "max_depth": 25,
        "min_samples_split": 4,
        "min_samples_leaf": 2,
        "n_jobs": -1,
        "random_state": random_state,
    }
    if params:
        default_params.update(params)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(**default_params)),
    ])
    return pipeline
