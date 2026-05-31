"""
Post-race accuracy scorer.

Computes all accuracy metrics when comparing a locked prediction
against the actual race result.

All metrics are pure functions — no database or Streamlit dependency.
"""
from __future__ import annotations

import math
from typing import Optional


def score(
    predicted_top10: list[str],
    actual_top10:    list[str],
) -> dict:
    """
    Compare a predicted finishing order against the actual result.

    Args:
        predicted_top10: Ordered list of driver names (P1 first), length 1–20.
        actual_top10:    Ordered list of driver names (P1 first), length 1–20.

    Returns dict with keys:
        winner_correct  (bool)   — Was P1 correct?
        podium_hits     (int)    — Drivers in pred top-3 that finished top-3 (0–3)
        top10_hits      (int)    — Drivers in pred top-10 that finished top-10 (0–10)
        position_mae    (float)  — Mean |predicted_rank - actual_rank| for matched drivers
        spearman_rho    (float)  — Rank correlation over all common drivers
    """
    if not predicted_top10 or not actual_top10:
        return _empty_scores()

    # Normalise to same length ceiling
    pred = predicted_top10[:20]
    act  = actual_top10[:20]

    winner_correct = (
        bool(pred) and bool(act) and pred[0] == act[0]
    )

    podium_pred = set(pred[:3])
    podium_act  = set(act[:3])
    podium_hits = len(podium_pred & podium_act)

    top10_pred = set(pred[:10])
    top10_act  = set(act[:10])
    top10_hits = len(top10_pred & top10_act)

    # Position error for drivers that appear in BOTH lists
    pred_ranks = {driver: (i + 1) for i, driver in enumerate(pred)}
    act_ranks  = {driver: (i + 1) for i, driver in enumerate(act)}
    common     = [d for d in pred if d in act_ranks]

    position_mae: float = 0.0
    if common:
        position_mae = sum(
            abs(pred_ranks[d] - act_ranks[d]) for d in common
        ) / len(common)

    spearman_rho = _spearman(pred_ranks, act_ranks)

    return {
        "winner_correct": winner_correct,
        "podium_hits":    podium_hits,
        "top10_hits":     top10_hits,
        "position_mae":   round(position_mae, 3),
        "spearman_rho":   round(spearman_rho, 4) if spearman_rho is not None else None,
    }


def _spearman(
    pred_ranks: dict[str, int],
    act_ranks:  dict[str, int],
) -> Optional[float]:
    """
    Compute Spearman rank correlation coefficient over common drivers.
    Returns None if fewer than 2 common drivers.
    """
    common = [d for d in pred_ranks if d in act_ranks]
    n = len(common)
    if n < 2:
        return None

    d_sq_sum = sum((pred_ranks[d] - act_ranks[d]) ** 2 for d in common)
    rho = 1 - (6 * d_sq_sum) / (n * (n ** 2 - 1))
    return max(-1.0, min(1.0, rho))   # clamp to [-1, 1]


def _empty_scores() -> dict:
    return {
        "winner_correct": None,
        "podium_hits":    None,
        "top10_hits":     None,
        "position_mae":   None,
        "spearman_rho":   None,
    }


def score_summary_text(scores: dict) -> str:
    """Return a human-readable one-line summary of the scores."""
    if scores.get("winner_correct") is None:
        return "Not yet scored."
    winner = "✅" if scores["winner_correct"] else "❌"
    return (
        f"Winner {winner} | "
        f"Podium: {scores['podium_hits']}/3 | "
        f"Top-10: {scores['top10_hits']}/10 | "
        f"Pos MAE: {scores['position_mae']:.1f} | "
        f"Spearman ρ: {scores['spearman_rho']:.3f}"
    )
