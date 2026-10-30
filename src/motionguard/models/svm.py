"""Support Vector Machine baseline classifier for MotionGuard AI."""

from typing import Any, Dict, Optional
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def build_svm_pipeline(
    params: Optional[Dict[str, Any]] = None,
    random_state: int = 42,
) -> Pipeline:
    """Creates a scikit-learn Pipeline with StandardScaler and SVC with probability estimates."""
    default_params = {
        "C": 5.0,
        "kernel": "rbf",
        "gamma": "scale",
        "probability": True,
        "random_state": random_state,
    }
    if params:
        default_params.update(params)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(**default_params)),
    ])
    return pipeline
