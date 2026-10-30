"""Gradient Boosting baseline classifier for MotionGuard AI."""

from typing import Any, Dict, Optional
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_gradient_boosting_pipeline(
    params: Optional[Dict[str, Any]] = None,
    random_state: int = 42,
) -> Pipeline:
    """Creates a scikit-learn Pipeline with StandardScaler and HistGradientBoostingClassifier."""
    default_params = {
        "max_iter": 120,
        "learning_rate": 0.1,
        "max_depth": 7,
        "min_samples_leaf": 20,
        "random_state": random_state,
    }
    if params:
        # map n_estimators to max_iter if given in config
        p = dict(params)
        if "n_estimators" in p:
            p["max_iter"] = p.pop("n_estimators")
        p.pop("subsample", None)
        p.pop("max_depth", None)  # default max_depth is None/unconstrained in HistGB
        default_params.update(p)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", HistGradientBoostingClassifier(**default_params)),
    ])
    return pipeline
