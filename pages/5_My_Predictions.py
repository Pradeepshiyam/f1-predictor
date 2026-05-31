"""
Page 5 — My Predictions

Shows the authenticated user's locked prediction history,
accuracy scores after race results, and the global leaderboard.
Admins also see a "Score a Prediction" panel.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import streamlit as st
import pandas as pd
import plotly.express as px

from f1_predictor.common.auth import require_auth, current_user_id, is_admin
from f1_predictor.common.config import settings
from f1_predictor.common.exceptions import (
    PredictionNotFoundError, PredictionAlreadyScoredError,
)
from f1_predictor.prediction import store as pred_store
from f1_predictor.prediction.scorer import score_summary_text

require_auth()

st.markdown("## 🔬 My Predictions & Accuracy")
st.markdown("---")

user_id = current_user_id()
year    = settings.current_year

# ── My prediction history ─────────────────────────────────────────────────────
st.markdown("### 📋 My Locked Predictions")

history = pred_store.get_user_predictions(user_id)

if history.empty:
    st.info("You haven't made any predictions yet. Go to 🔮 Race Prediction to start!")
else:
    # Status column
    def _status(row):
        if row.get("scored_at"):
            return score_summary_text({
                "winner_correct": row.get("winner_correct"),
                "podium_hits":    row.get("podium_hits"),
                "top10_hits":     row.get("top10_hits"),
                "position_mae":   row.get("position_mae"),
                "spearman_rho":   row.get("spearman_rho"),
            })
        return "🔒 Locked — awaiting race result" if row.get("is_locked") else "📝 Draft"

    history["Status"] = history.apply(_status, axis=1)

    display_cols = ["race_name", "year", "predicted_at", "is_locked", "Status"]
    st.dataframe(
        history[[c for c in display_cols if c in history.columns]].rename(columns={
            "race_name": "Race", "year": "Year",
            "predicted_at": "Predicted At", "is_locked": "Locked",
        }),
        use_container_width=True, hide_index=True,
    )

    # Accuracy trend (scored predictions only)
    scored = history[history["scored_at"].notna()].copy()
    if not scored.empty:
        st.markdown("### 📈 Accuracy Trend")
        fig = px.line(
            scored, x="race_name", y="podium_hits",
            title="Podium Hits per Race (out of 3)",
            markers=True, color_discrete_sequence=["#e10600"],
        )
        fig.update_layout(
            xaxis_tickangle=-45, height=350,
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            font=dict(color="#e0e0e0"),
        )
        st.plotly_chart(fig, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Winner Correct", f"{int(scored['winner_correct'].sum())}/{len(scored)}")
        with col2:
            st.metric("Avg Podium Hits", f"{scored['podium_hits'].mean():.2f}/3")
        with col3:
            st.metric("Avg Spearman ρ", f"{scored['spearman_rho'].mean():.3f}")

st.markdown("---")

# ── Score a prediction (admin or user after race) ────────────────────────────
st.markdown("### 📊 Score a Prediction Against Actual Results")
st.caption("Run this after the race result is ingested from FastF1.")

with st.expander("Enter Actual Race Result"):
    score_race = st.selectbox(
        "Race to score:",
        options=history["race_name"].tolist() if not history.empty else [],
    )
    actual_input = st.text_area(
        "Actual finishing order (one driver name per line, P1 first):",
        placeholder="Max Verstappen\nCharles Leclerc\nLando Norris\n...",
        height=200,
    )
    if st.button("📥 Submit Actual Result & Score", type="primary"):
        actual_list = [n.strip() for n in actual_input.strip().splitlines() if n.strip()]
        if len(actual_list) < 3:
            st.error("Please enter at least 3 driver names.")
        else:
            try:
                scores = pred_store.score_prediction(user_id, score_race, year, actual_list)
                st.success(f"Scored! {score_summary_text(scores)}")
                st.balloons()
            except PredictionNotFoundError:
                st.error(f"No locked prediction found for {score_race} {year}.")
            except PredictionAlreadyScoredError:
                st.warning("This prediction has already been scored.")

st.markdown("---")

# ── Global Leaderboard ────────────────────────────────────────────────────────
st.markdown("### 🏆 Global Leaderboard")

leaderboard = pred_store.get_leaderboard()

if leaderboard.empty:
    st.info("No scored predictions yet across all users.")
else:
    medals = {1: "🥇", 2: "🥈", 3: "🥉"}
    leaderboard["rank"] = leaderboard["rank"].astype(int)
    leaderboard["Medal"] = leaderboard["rank"].map(lambda r: medals.get(r, str(r)))

    st.dataframe(
        leaderboard[[
            "Medal", "username", "races_scored", "winners_correct",
            "avg_podium_hits", "avg_top10_hits", "avg_position_mae", "avg_spearman"
        ]].rename(columns={
            "username": "User",
            "races_scored": "Races",
            "winners_correct": "Winners ✅",
            "avg_podium_hits": "Avg Podium",
            "avg_top10_hits": "Avg Top-10",
            "avg_position_mae": "Avg MAE",
            "avg_spearman": "Avg ρ",
        }),
        use_container_width=True, hide_index=True,
    )
