"""
F1 Multi-Task Model Training Pipeline
Uses the Gold parquet produced by pandas_processor.py.

Tasks:
  - race  : binary classifier → IsWinner  (XGBoost)
  - pole  : binary classifier → GridPosition==1 (XGBoost)
  - top3  : binary classifier → IsTop3    (XGBoost)

Run:  python src/f1_predictor/models/train.py
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (accuracy_score, brier_score_loss,
                             precision_score, roc_auc_score)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ── Feature columns used by ALL tasks ────────────────────────────────────────
# These must exist in the Gold parquet (pandas_processor.py produces them).
FEATURES = [
    "GridPosition",       # starting grid
    "AvgFinishLast5",     # rolling 5-race avg finish
    "AvgFinishLast3",     # rolling 3-race avg finish (recent form)
    "AvgGridLast5",       # rolling avg qualifying position
    "TeamForm",           # team points last 5 races
    "DNFRateLast10",      # reliability
    "CircuitAvgFinish",   # driver's historical avg at this circuit
    "CircuitWins",        # wins at this circuit
    "ChampPos",           # championship standing at race time
]

# Backwards-compatible 3-feature set (used if Gold was built by spark_processor.py)
FEATURES_COMPAT = ["GridPosition", "AvgFinishLast5", "TeamForm"]


def _available_features(df: pd.DataFrame) -> list[str]:
    """Return whichever feature set is available in the dataframe."""
    full = [f for f in FEATURES if f in df.columns]
    if len(full) >= 5:
        return full
    log.warning("Full feature set not available — falling back to 3-feature compat mode. "
                "Run pandas_processor.py to rebuild Gold.")
    return [f for f in FEATURES_COMPAT if f in df.columns]


def _time_split(df: pd.DataFrame, test_frac: float = 0.15):
    """
    Strict time-aware train/test split — NO random splitting.
    Test set = last `test_frac` of races by RaceIndex.
    """
    if "RaceIndex" in df.columns:
        cutoff = df["RaceIndex"].quantile(1 - test_frac)
        train = df[df["RaceIndex"] <= cutoff]
        test  = df[df["RaceIndex"] >  cutoff]
    else:
        n = len(df)
        split = int(n * (1 - test_frac))
        train = df.iloc[:split]
        test  = df.iloc[split:]
    return train, test


def train_task(df: pd.DataFrame, target_col: str, model_name: str,
               features: list[str]) -> dict:
    """
    Train one XGBoost binary classifier for a given target.
    Returns a metrics dict.
    """
    log.info(f"\n{'='*50}")
    log.info(f"Training: {model_name}  |  target: {target_col}  |  features: {features}")

    df_clean = df.dropna(subset=features + [target_col]).copy()
    X = df_clean[features]
    y = df_clean[target_col].astype(int)

    pos_rate = y.mean()
    log.info(f"  Dataset: {len(df_clean):,} rows  |  positive rate: {pos_rate:.3f}")

    if pos_rate == 0 or pos_rate == 1:
        log.error(f"  Skipping {model_name}: degenerate target (rate={pos_rate})")
        return {}

    train_df, test_df = _time_split(df_clean)
    X_train, y_train = train_df[features], train_df[target_col].astype(int)
    X_test,  y_test  = test_df[features],  test_df[target_col].astype(int)

    # Class imbalance weight
    scale_pos = max(1.0, (y_train == 0).sum() / max((y_train == 1).sum(), 1))

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    y_pred      = model.predict(X_test)
    y_prob      = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model":     model_name,
        "target":    target_col,
        "features":  features,
        "n_train":   int(len(X_train)),
        "n_test":    int(len(X_test)),
        "accuracy":  float(round(accuracy_score(y_test, y_pred), 4)),
        "precision": float(round(precision_score(y_test, y_pred, zero_division=0), 4)),
        "roc_auc":   float(round(roc_auc_score(y_test, y_prob), 4)),
        "brier":     float(round(brier_score_loss(y_test, y_prob), 4)),
    }

    log.info(f"  Accuracy={metrics['accuracy']}  AUC={metrics['roc_auc']}  "
             f"Brier={metrics['brier']}")

    # Save model
    os.makedirs("models", exist_ok=True)
    out = f"models/{model_name}.json"
    model.save_model(out)
    log.info(f"  Saved → {out}")

    # Save feature list alongside model (critical for inference)
    feat_file = f"models/{model_name}_features.json"
    with open(feat_file, "w") as fh:
        json.dump(features, fh)
    log.info(f"  Feature list → {feat_file}")

    return metrics


def run(gold_path: str = "data/gold/features.parquet") -> None:
    if not os.path.exists(gold_path):
        print(f"❌ Gold parquet not found at '{gold_path}'.")
        print("   Run first:  python src/f1_predictor/features/pandas_processor.py")
        return

    df = pd.read_parquet(gold_path)

    # Cast all expected columns to numeric (schema may have mixed types from CSV sources)
    all_numeric = FEATURES + FEATURES_COMPAT + [
        "IsWinner", "IsTop3", "IsTop10", "IsPole",
        "ClassifiedPosition", "Position", "GridPosition", "RaceIndex",
    ]
    for col in all_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    log.info(f"Loaded Gold: {len(df):,} rows, {df['FullName'].nunique()} drivers")

    feats = _available_features(df)

    # Ensure IsTop3 target exists
    if "IsTop3" not in df.columns and "ClassifiedPosition" in df.columns:
        df["IsTop3"] = (df["ClassifiedPosition"] <= 3).astype(int)
    if "IsWinner" not in df.columns and "ClassifiedPosition" in df.columns:
        df["IsWinner"] = (df["ClassifiedPosition"] == 1).astype(int)

    all_metrics = []

    # Task 1: Race Winner
    if "IsWinner" in df.columns:
        m = train_task(df, "IsWinner", "race_winner_model", feats)
        if m: all_metrics.append(m)

    # Task 2: Pole Position
    if "GridPosition" in df.columns:
        df["IsPole"] = (df["GridPosition"] == 1).astype(int)
        m = train_task(df, "IsPole", "pole_position_model", feats)
        if m: all_metrics.append(m)

    # Task 3: Top-3 Podium
    if "IsTop3" in df.columns:
        m = train_task(df, "IsTop3", "top3_model", feats)
        if m: all_metrics.append(m)

    # Save run metadata
    meta = {
        "run_at":   datetime.now(timezone.utc).isoformat(),
        "gold_path": gold_path,
        "features": feats,
        "tasks":    all_metrics,
    }
    os.makedirs("outputs", exist_ok=True)
    meta_file = "outputs/last_training_run.json"
    with open(meta_file, "w") as fh:
        json.dump(meta, fh, indent=2)

    print("\n✅ Training complete!")
    print(f"   Models saved in: models/")
    print(f"   Run metadata:    {meta_file}")
    print("\n📊 Metrics Summary:")
    for m in all_metrics:
        print(f"   {m['model']:30s}  AUC={m['roc_auc']:.3f}  Acc={m['accuracy']:.3f}  Brier={m['brier']:.3f}")


if __name__ == "__main__":
    run()
