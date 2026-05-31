"""
Clean inference wrapper for the trained XGBoost models.

Loads model + feature list from disk, validates feature alignment,
and exposes a single predict_proba() interface.

No training logic here — only inference.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from f1_predictor.common.config import settings
from f1_predictor.common.exceptions import ModelFeatureMismatchError, ModelNotFoundError
from f1_predictor.common.logger import get_logger

log = get_logger(__name__)

# Model task identifiers (must match filenames produced by train.py)
TASK_WINNER = "race_winner"
TASK_POLE   = "pole_position"
TASK_TOP3   = "top3"

_MODEL_REGISTRY: dict[str, object] = {}   # task → loaded model


def _model_path(task: str) -> Path:
    return Path(settings.paths.models) / f"{task}_model.json"


def _features_path(task: str) -> Path:
    return Path(settings.paths.models) / f"{task}_model_features.json"


def _load_model(task: str):
    """Load XGBoost model from disk. Cached in _MODEL_REGISTRY."""
    if task in _MODEL_REGISTRY:
        return _MODEL_REGISTRY[task]

    model_file = _model_path(task)
    if not model_file.exists():
        raise ModelNotFoundError(
            f"Model file not found: {model_file}. "
            f"Run `python src/f1_predictor/models/train.py` first."
        )

    from xgboost import XGBClassifier
    model = XGBClassifier()
    model.load_model(str(model_file))
    _MODEL_REGISTRY[task] = model
    log.info(f"Model loaded: {task}")
    return model


def _load_features(task: str) -> list[str]:
    """Load the feature list the model was trained on."""
    feat_file = _features_path(task)
    if not feat_file.exists():
        raise ModelNotFoundError(
            f"Feature list not found: {feat_file}. "
            f"Run training first."
        )
    with open(feat_file, encoding="utf-8") as fh:
        return json.load(fh)


def predict_proba(task: str, feature_df: pd.DataFrame) -> np.ndarray:
    """
    Run inference for the given task.

    Args:
        task:       One of TASK_WINNER, TASK_POLE, TASK_TOP3.
        feature_df: DataFrame with one row per driver.
                    May have extra columns — they are silently dropped.
                    Missing required columns are filled with 0.

    Returns:
        1-D numpy array of positive-class probabilities, shape (n_drivers,).

    Raises:
        ModelNotFoundError: Model or feature file missing from disk.
    """
    model    = _load_model(task)
    features = _load_features(task)

    # Align to model's expected feature set
    X = _align_features(feature_df, features)

    probs = model.predict_proba(X)[:, 1]
    log.debug(f"[{task}] inference: {len(probs)} drivers, "
              f"max_prob={probs.max():.3f}, min_prob={probs.min():.3f}")
    return probs


def _align_features(df: pd.DataFrame, expected: list[str]) -> pd.DataFrame:
    """
    Build a DataFrame with exactly the columns in `expected`, in order.
    - Extra columns in `df` are dropped.
    - Missing columns are filled with 0 (safe fallback).
    """
    missing = [c for c in expected if c not in df.columns]
    if missing:
        log.warning(
            f"Feature mismatch — missing columns filled with 0: {missing}"
        )

    X = pd.DataFrame(index=df.index)
    for col in expected:
        if col in df.columns:
            X[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        else:
            X[col] = 0.0

    return X


def reload_models() -> None:
    """Force reload all models from disk (call after retraining)."""
    _MODEL_REGISTRY.clear()
    log.info("Model registry cleared — will reload on next inference call.")


def models_available() -> dict[str, bool]:
    """Return dict of {task: model_file_exists} for UI status display."""
    return {
        task: _model_path(task).exists()
        for task in [TASK_WINNER, TASK_POLE, TASK_TOP3]
    }
