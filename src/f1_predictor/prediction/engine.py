"""
Prediction orchestrator — 2-stage pipeline:
  Stage 1: Qualifying simulation → predicted grid order
  Stage 2: Race simulation       → predicted finishing order + probabilities

Reads from Gold feature store (data/gold/features.parquet).
Uses only active drivers from the current season (fully dynamic).
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from f1_predictor.common.config import settings
from f1_predictor.common.exceptions import GoldFeaturesMissingError, PredictionError
from f1_predictor.common.logger import get_logger
from f1_predictor.models.inference import TASK_POLE, TASK_TOP3, TASK_WINNER, predict_proba

log = get_logger(__name__)

# Driver name normalization map — built dynamically from data, not hardcoded
_ALIAS_OVERRIDE: dict[str, str] = {}   # populated by _load_gold()


def _load_gold() -> pd.DataFrame:
    gold_path = settings.gold_path
    if not gold_path.exists():
        raise GoldFeaturesMissingError(
            f"Gold features not found at {gold_path}. "
            f"Run pandas_processor.py first."
        )
    df = pd.read_parquet(gold_path)
    log.info(f"Gold loaded: {len(df):,} rows, {df['FullName'].nunique()} drivers")
    return df


def _get_active_drivers(gold: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the active driver set from the most recent season data.
    Falls back to the two most recent years if latest year has < 3 races.
    Returns one row per driver with aggregated historical features.
    """
    current_year = settings.current_year

    # Dynamic year detection — no hardcoding
    years_in_data = sorted(gold["Year"].astype(str).unique())
    year_str = str(current_year)

    if year_str not in years_in_data:
        # Use most recent year in the data
        year_str = years_in_data[-1]
        log.warning(f"No {current_year} data — falling back to {year_str}")

    latest = gold[gold["Year"].astype(str) == year_str]

    # If fewer than 3 races in latest year, also include previous year
    if latest["Circuit"].nunique() < 3 and len(years_in_data) >= 2:
        prev_year = years_in_data[-2]
        latest = gold[gold["Year"].astype(str).isin([year_str, prev_year])]
        log.info(f"Using {prev_year}+{year_str} data ({latest['Circuit'].nunique()} circuits)")

    active_drivers = set(latest["FullName"].dropna().unique())
    return gold[gold["FullName"].isin(active_drivers)].copy()


def _build_driver_feature_row(
    driver_history: pd.DataFrame,
    circuit_keyword: str,
) -> dict:
    """
    Aggregate all historical rows for one driver into a single feature vector.
    Everything is derived from data — no hardcoded values.
    """
    hist = driver_history.sort_values("RaceIndex")

    # Latest values (most recent race)
    latest = hist.iloc[-1]

    # Circuit-specific form
    circuit_mask = hist["Circuit"].str.lower().str.contains(circuit_keyword, na=False)
    circuit_hist = hist[circuit_mask]

    avg_finish_circuit = (
        circuit_hist["Position"].mean()
        if not circuit_hist.empty
        else hist["Position"].mean()
    )
    circuit_wins = int((circuit_hist["Position"] == 1).sum()) if not circuit_hist.empty else 0

    # Recent rolling features (last available values from Gold)
    def _last(col: str, fallback=10.0):
        vals = hist[col].dropna() if col in hist.columns else pd.Series(dtype=float)
        return float(vals.iloc[-1]) if not vals.empty else fallback

    return {
        "FullName":          latest["FullName"],
        "TeamName":          latest.get("TeamName", "Unknown"),
        "HeadshotUrl":       latest.get("HeadshotUrl", ""),
        # Features used by the model
        "AvgFinishLast3":    _last("AvgFinishLast3"),
        "AvgFinishLast5":    _last("AvgFinishLast5"),
        "AvgFinishLast10":   _last("AvgFinishLast10"),
        "AvgGridLast5":      _last("AvgGridLast5"),
        "DNFRateLast10":     _last("DNFRateLast10", 0.05),
        "TeamPointsLast5":   _last("TeamPointsLast5", 0.0),
        "TeamForm":          _last("TeamForm", 0.0),
        "CircuitAvgFinish":  avg_finish_circuit,
        "CircuitWins":       float(circuit_wins),
        "ChampPos":          _last("ChampPos", 10.0),
        # Aliases that older model feature lists may reference
        "GridPosition":      _last("AvgGridLast5"),
        "AvgFinishLast5_alias": _last("AvgFinishLast5"),
    }


def build_features_for_race(circuit_name: str) -> pd.DataFrame:
    """
    Build one-row-per-driver feature DataFrame for the given circuit.

    Args:
        circuit_name: Full circuit name (e.g. "Canadian Grand Prix").
                      First word used as keyword for circuit matching.
    Returns:
        DataFrame with one row per active driver, all features aligned.
    """
    gold   = _load_gold()
    active = _get_active_drivers(gold)

    keyword = circuit_name.split()[0].lower()  # "Canadian" → "canadian"
    log.info(f"Building features for circuit keyword='{keyword}', "
             f"active drivers={active['FullName'].nunique()}")

    rows = []
    for driver, driver_df in active.groupby("FullName"):
        row = _build_driver_feature_row(driver_df, keyword)
        rows.append(row)

    feats = pd.DataFrame(rows).reset_index(drop=True)
    log.info(f"Feature matrix: {len(feats)} drivers × {len(feats.columns)} columns")
    return feats


def run_prediction(
    race_name:    str,
    year:         int,
    round_number: int,
) -> pd.DataFrame:
    """
    Run the full 2-stage prediction pipeline.

    Stage 1 — Qualifying:
        Use pole_position model to predict grid order.

    Stage 2 — Race:
        Inject predicted grid as GridPosition, run race_winner + top3 models,
        combine scores, normalise to probabilities.

    Returns:
        DataFrame with columns:
        [PredictedRank, FullName, TeamName, PredictedGrid, PredictedFinish,
         WinProbability, PodiumProbability, CircuitAvgFinish, AvgFinishLast5,
         DNFRateLast10, ChampPos, HeadshotUrl, Reasons]

    Raises:
        GoldFeaturesMissingError: Features not built yet.
        PredictionError: Any unrecoverable error during simulation.
    """
    log.info(f"Starting prediction: {race_name} {year} (round {round_number})")

    feats = build_features_for_race(race_name)
    if feats.empty:
        raise PredictionError("No driver features available — check Bronze data.")

    # ── Stage 1: Qualifying / Grid Simulation ─────────────────────────────────
    try:
        qual_probs = predict_proba(TASK_POLE, feats)
    except Exception as exc:
        log.warning(f"Pole model failed ({exc}), falling back to AvgGridLast5")
        qual_probs = 1.0 / (feats["AvgGridLast5"].clip(lower=1).values)

    feats["_qual_score"] = qual_probs
    feats_sorted = feats.sort_values("_qual_score", ascending=False).reset_index(drop=True)
    feats_sorted["PredictedGrid"] = range(1, len(feats_sorted) + 1)

    # Inject predicted grid as the qualifying feature for the race model
    feats_sorted["GridPosition"] = feats_sorted["PredictedGrid"].astype(float)

    # ── Stage 2: Race Simulation ──────────────────────────────────────────────
    try:
        winner_probs = predict_proba(TASK_WINNER, feats_sorted)
        top3_probs   = predict_proba(TASK_TOP3,   feats_sorted)
    except Exception as exc:
        raise PredictionError(f"Race model inference failed: {exc}") from exc

    # Combine: 60% race winner model + 40% top3 model
    combined = 0.60 * winner_probs + 0.40 * top3_probs

    # Normalise to sum = 1
    total = combined.sum()
    if total > 0:
        combined = combined / total

    feats_sorted["_race_score"]   = combined
    feats_sorted["WinProbability"]    = winner_probs / winner_probs.sum()
    feats_sorted["PodiumProbability"] = top3_probs   / top3_probs.sum()

    # Sort by combined race score → predicted finishing order
    result = feats_sorted.sort_values("_race_score", ascending=False).reset_index(drop=True)
    result["PredictedFinish"] = range(1, len(result) + 1)
    result["PredictedRank"]   = result["PredictedFinish"]

    # ── Reasoning text ────────────────────────────────────────────────────────
    result["Reasons"] = result.apply(_build_reasons, axis=1)

    # ── Final column selection ────────────────────────────────────────────────
    output_cols = [
        "PredictedRank", "FullName", "TeamName",
        "PredictedGrid", "PredictedFinish",
        "WinProbability", "PodiumProbability",
        "CircuitAvgFinish", "AvgFinishLast5", "AvgGridLast5",
        "DNFRateLast10", "ChampPos",
        "HeadshotUrl", "Reasons",
    ]
    final = result[[c for c in output_cols if c in result.columns]].copy()
    final["WinProbability"]    = (final["WinProbability"] * 100).round(1)
    final["PodiumProbability"] = (final["PodiumProbability"] * 100).round(1)

    log.info(
        f"Prediction complete: P1={final.iloc[0]['FullName']} "
        f"({final.iloc[0]['WinProbability']:.1f}%)"
    )
    return final


def _build_reasons(row: pd.Series) -> str:
    """Generate a short human-readable reasoning string from feature values."""
    reasons = []

    grid = row.get("PredictedGrid", 20)
    if grid <= 3:
        reasons.append("🏁 Front-row starter")
    elif grid <= 6:
        reasons.append("🔵 Strong qualifier")

    circ = row.get("CircuitAvgFinish", 20)
    avg  = row.get("AvgFinishLast5", 20)
    if circ < avg - 2:
        reasons.append("🏟️ Track specialist")

    dnf = row.get("DNFRateLast10", 0.1)
    if dnf < 0.05:
        reasons.append("🔧 Reliable finisher")
    elif dnf > 0.25:
        reasons.append("⚠️ DNF risk")

    champ = row.get("ChampPos", 20)
    if champ <= 3:
        reasons.append("🏆 Championship leader")
    elif champ <= 6:
        reasons.append("📈 Title contender")

    form = row.get("TeamForm", 0)
    if form > 50:
        reasons.append("⚡ Team momentum")

    return " · ".join(reasons) if reasons else "📊 Historical data"
