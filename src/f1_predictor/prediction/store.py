"""
Prediction persistence layer — SQLAlchemy CRUD for the predictions table.

All database operations go through this module.
No raw SQL anywhere — uses the ORM defined in common/database.py.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd

from f1_predictor.common.database import Prediction, User, get_db, init_db
from f1_predictor.common.exceptions import (
    PredictionAlreadyScoredError,
    PredictionLockedError,
    PredictionNotFoundError,
    RaceAlreadyLockedError,
)
from f1_predictor.common.logger import get_logger
from f1_predictor.prediction.scorer import score as compute_scores

log = get_logger(__name__)


# ── Create / Update ───────────────────────────────────────────────────────────

def save_prediction(
    user_id:      int,
    race_name:    str,
    year:         int,
    round_number: int,
    top10:        list[dict],     # [{"pos":1,"driver":"X","team":"Y","prob":0.33},...]
    win_probs:    dict[str, float],
    fp1_start_utc: Optional[datetime] = None,
) -> Prediction:
    """
    Upsert a draft prediction for the given user + race.

    If a draft already exists it is overwritten (allows regenerating before lock).
    Raises RaceAlreadyLockedError if the FP1 has already started.
    """
    init_db()

    with get_db() as db:
        # Check for existing locked prediction first
        existing = (
            db.query(Prediction)
            .filter_by(user_id=user_id, race_name=race_name, year=year)
            .first()
        )
        if existing and existing.is_locked:
            raise PredictionLockedError(
                f"Prediction for '{race_name} {year}' is already locked."
            )

        if existing:
            pred = existing
        else:
            pred = Prediction(
                user_id=user_id,
                race_name=race_name,
                year=year,
                round_number=round_number,
                fp1_start_utc=fp1_start_utc,
            )
            db.add(pred)

        pred.predicted_at = datetime.utcnow()
        pred.set_top10(top10)
        pred.set_win_probs(win_probs)
        db.flush()
        pred_id = pred.id

    log.info(f"Prediction saved: user={user_id} race='{race_name}' year={year}")
    return pred_id


def lock_prediction(user_id: int, race_name: str, year: int) -> None:
    """
    Lock a draft prediction. Once locked, it cannot be modified.
    """
    init_db()

    with get_db() as db:
        pred = (
            db.query(Prediction)
            .filter_by(user_id=user_id, race_name=race_name, year=year)
            .first()
        )
        if pred is None:
            raise PredictionNotFoundError(
                f"No prediction found for '{race_name}' {year}."
            )
        if pred.is_locked:
            raise PredictionLockedError("Prediction is already locked.")

        pred.is_locked = True
        pred.locked_at = datetime.utcnow()

    log.info(f"Prediction locked: user={user_id} race='{race_name}' year={year}")


# ── Scoring ───────────────────────────────────────────────────────────────────

def score_prediction(
    user_id:       int,
    race_name:     str,
    year:          int,
    actual_top10:  list[str],     # ordered driver names, P1 first
) -> dict:
    """
    Score a locked prediction against the actual race result.

    Returns the score dict from scorer.score().
    Persists all metrics to the database.
    """
    init_db()

    with get_db() as db:
        pred = (
            db.query(Prediction)
            .filter_by(user_id=user_id, race_name=race_name, year=year)
            .first()
        )
        if pred is None:
            raise PredictionNotFoundError(
                f"No prediction for '{race_name}' {year}."
            )
        if pred.scored_at is not None:
            raise PredictionAlreadyScoredError(
                f"Prediction for '{race_name}' {year} has already been scored."
            )

        predicted_top10 = [row["driver"] for row in pred.get_top10()]
        scores = compute_scores(predicted_top10, actual_top10)

        pred.set_actual_top10(actual_top10)
        pred.winner_correct    = scores["winner_correct"]
        pred.podium_hits       = scores["podium_hits"]
        pred.top10_hits        = scores["top10_hits"]
        pred.position_mae      = scores["position_mae"]
        pred.spearman_rho      = scores["spearman_rho"]
        pred.scored_at         = datetime.utcnow()

    log.info(
        f"Prediction scored: user={user_id} race='{race_name}' year={year} "
        f"winner={scores['winner_correct']} podium={scores['podium_hits']}/3"
    )
    return scores


# ── Read ──────────────────────────────────────────────────────────────────────

def get_user_predictions(user_id: int) -> pd.DataFrame:
    """Return all predictions for a user as a DataFrame."""
    init_db()

    with get_db() as db:
        preds = (
            db.query(Prediction)
            .filter_by(user_id=user_id)
            .order_by(Prediction.year, Prediction.round_number)
            .all()
        )
        rows = [_pred_to_row(p) for p in preds]

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def get_all_predictions_admin() -> pd.DataFrame:
    """Return all predictions for all users (admin only)."""
    init_db()

    with get_db() as db:
        results = (
            db.query(Prediction, User.username)
            .join(User, Prediction.user_id == User.id)
            .order_by(Prediction.year, Prediction.round_number)
            .all()
        )
        rows = []
        for pred, username in results:
            row = _pred_to_row(pred)
            row["username"] = username
            rows.append(row)

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def get_leaderboard() -> pd.DataFrame:
    """
    Global leaderboard: all users ranked by average podium_hits descending.
    Only includes scored predictions.
    """
    init_db()

    with get_db() as db:
        results = (
            db.query(User.username, Prediction)
            .join(Prediction, User.id == Prediction.user_id)
            .filter(Prediction.scored_at.isnot(None))
            .all()
        )

    if not results:
        return pd.DataFrame()

    rows = []
    for username, pred in results:
        rows.append({
            "username":      username,
            "race":          pred.race_name,
            "year":          pred.year,
            "winner":        int(pred.winner_correct or 0),
            "podium_hits":   pred.podium_hits or 0,
            "top10_hits":    pred.top10_hits or 0,
            "position_mae":  pred.position_mae,
            "spearman_rho":  pred.spearman_rho,
        })

    df = pd.DataFrame(rows)
    summary = (
        df.groupby("username")
        .agg(
            races_scored=("race", "count"),
            winners_correct=("winner", "sum"),
            avg_podium_hits=("podium_hits", "mean"),
            avg_top10_hits=("top10_hits", "mean"),
            avg_position_mae=("position_mae", "mean"),
            avg_spearman=("spearman_rho", "mean"),
        )
        .reset_index()
        .sort_values("avg_podium_hits", ascending=False)
    )
    summary["rank"] = range(1, len(summary) + 1)
    return summary


def get_prediction(user_id: int, race_name: str, year: int) -> Optional[dict]:
    """Return a single prediction as a dict, or None if not found."""
    init_db()

    with get_db() as db:
        pred = (
            db.query(Prediction)
            .filter_by(user_id=user_id, race_name=race_name, year=year)
            .first()
        )
        if pred is None:
            return None
        return _pred_to_row(pred)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _pred_to_row(pred: Prediction) -> dict:
    return {
        "id":              pred.id,
        "race_name":       pred.race_name,
        "year":            pred.year,
        "round_number":    pred.round_number,
        "predicted_at":    pred.predicted_at,
        "is_locked":       pred.is_locked,
        "locked_at":       pred.locked_at,
        "fp1_start_utc":   pred.fp1_start_utc,
        "top10":           pred.get_top10(),
        "win_probs":       pred.get_win_probs(),
        "actual_top10":    pred.get_actual_top10(),
        "winner_correct":  pred.winner_correct,
        "podium_hits":     pred.podium_hits,
        "top10_hits":      pred.top10_hits,
        "position_mae":    pred.position_mae,
        "spearman_rho":    pred.spearman_rho,
        "scored_at":       pred.scored_at,
    }
