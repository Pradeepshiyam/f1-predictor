"""
Page 1 — Race Prediction Engine

Features:
- Upcoming races only, calendar order (from FastF1 API)
- Auto-selects the next upcoming race
- Lock status derived from FP1 start time (live from FastF1)
- Prediction locked before FP1 — cannot be regenerated after
- Save & lock prediction to SQLite
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from f1_predictor.common.auth import require_auth, current_user_id
from f1_predictor.common.calendar import (
    get_upcoming_races, get_fp1_start, is_prediction_locked, get_season_schedule
)
from f1_predictor.common.config import settings
from f1_predictor.common.exceptions import (
    GoldFeaturesMissingError, ModelNotFoundError, PredictionError,
    PredictionLockedError, RaceAlreadyLockedError,
)
from f1_predictor.prediction.engine import run_prediction
from f1_predictor.prediction import store as pred_store

require_auth()

st.markdown("## 🔮 Race Prediction Engine")
st.markdown("---")

year = settings.current_year

# ── Race Selector (upcoming only, calendar order) ────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def _get_upcoming(yr):
    return get_upcoming_races(yr)

upcoming = _get_upcoming(year)

if upcoming.empty:
    st.warning("🏁 The season is over — no upcoming races.")
    st.stop()

race_options  = upcoming["race_name"].tolist()
round_numbers = upcoming["round_number"].tolist()

# Default index: next race (first in upcoming list)
selected_idx  = st.selectbox(
    "Select Upcoming Race:",
    options=range(len(race_options)),
    format_func=lambda i: race_options[i],
    index=0,
)
selected_race  = race_options[selected_idx]
selected_round = round_numbers[selected_idx]

# ── Lock status ───────────────────────────────────────────────────────────────
fp1_time = get_fp1_start(year, selected_round)
locked_now = is_prediction_locked(year, selected_round)

col_info1, col_info2 = st.columns(2)
with col_info1:
    fp1_display = fp1_time.strftime("%a %d %b %Y, %H:%M UTC") if fp1_time else "Unknown"
    st.info(f"🔒 Locks at FP1 start: **{fp1_display}**")
with col_info2:
    if locked_now:
        st.error("🔴 Predictions LOCKED — FP1 has started")
    else:
        st.success("🟢 Predictions OPEN — Submit before FP1")

st.markdown("---")

# ── Existing prediction check ─────────────────────────────────────────────────
user_id   = current_user_id()
existing  = pred_store.get_prediction(user_id, selected_race, year)
is_my_locked = existing and existing["is_locked"]

# ── Generate Button ───────────────────────────────────────────────────────────
if locked_now and not is_my_locked:
    st.warning("Prediction window closed for this race. You can still view others' predictions.")

generate_disabled = locked_now and not is_my_locked

if not generate_disabled:
    if st.button("🚀 Generate AI Prediction", use_container_width=True, type="primary"):
        st.session_state["pred_result"] = None
        with st.spinner(f"🧠 Simulating {selected_race}…"):
            try:
                result_df = run_prediction(selected_race, year, selected_round)
                st.session_state["pred_result"] = result_df
                st.session_state["pred_race"]   = selected_race
            except GoldFeaturesMissingError:
                st.error("❌ Feature data missing. Run the ETL pipeline first:\n"
                         "`.venv\\Scripts\\python.exe src/f1_predictor/features/pandas_processor.py`")
            except ModelNotFoundError as e:
                st.error(f"❌ Model missing: {e}")
            except PredictionError as e:
                st.error(f"❌ Prediction failed: {e}")

# ── Display Result ────────────────────────────────────────────────────────────
result_df: pd.DataFrame | None = st.session_state.get("pred_result")
pred_race: str = st.session_state.get("pred_race", "")

if result_df is not None and pred_race == selected_race:
    st.markdown(f"### 🏆 Predicted Result — {selected_race} {year}")

    # ── Top 3 podium cards ────────────────────────────────────────────────────
    podium = result_df.head(3)
    p_cols = st.columns(3)
    medals = ["🥇", "🥈", "🥉"]
    for i, (col, (_, row)) in enumerate(zip(p_cols, podium.iterrows())):
        with col:
            img = row.get("HeadshotUrl", "")
            if img:
                st.image(img, width=100)
            st.markdown(
                f"**{medals[i]} {row['FullName']}**  \n"
                f"{row.get('TeamName','')}  \n"
                f"Win Prob: **{row['WinProbability']:.1f}%**  \n"
                f"Grid P{int(row['PredictedGrid'])}  \n"
                f"*{row.get('Reasons','')}*"
            )

    st.markdown("---")

    # ── Grid vs Finish line chart ─────────────────────────────────────────────
    st.markdown(f"#### 📈 Predicted Grid vs Finishing Order — {selected_race}")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=result_df["FullName"], y=result_df["PredictedGrid"],
        mode="lines+markers", name="Predicted Grid (Qualifying)",
        line=dict(color="#3498db", width=2),
        marker=dict(size=7),
    ))
    fig.add_trace(go.Scatter(
        x=result_df["FullName"], y=result_df["PredictedFinish"],
        mode="lines+markers", name="Predicted Finish (Race)",
        line=dict(color="#e67e22", width=2, dash="dot"),
        marker=dict(size=7, symbol="diamond"),
    ))
    fig.update_layout(
        xaxis_tickangle=-45,
        yaxis=dict(title="Position", autorange="reversed"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=420,
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="#e0e0e0"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Win probability bar chart ─────────────────────────────────────────────
    st.markdown("#### 🎯 Win Probability — All Drivers")
    top10_df = result_df.head(settings.top_n_drivers)
    fig2 = go.Figure(go.Bar(
        x=top10_df["FullName"],
        y=top10_df["WinProbability"],
        marker_color=[
            "#e10600" if i == 0 else "#ff6b35" if i < 3 else "#3498db"
            for i in range(len(top10_df))
        ],
        text=[f"{v:.1f}%" for v in top10_df["WinProbability"]],
        textposition="outside",
    ))
    fig2.update_layout(
        yaxis_title="Win Probability (%)",
        height=350,
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="#e0e0e0"),
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ── Full prediction table ─────────────────────────────────────────────────
    st.markdown("#### 📋 Full Predicted Starting Grid & Finishing Order")
    display_cols = [c for c in [
        "PredictedGrid", "PredictedFinish", "FullName", "TeamName",
        "WinProbability", "PodiumProbability", "CircuitAvgFinish",
        "AvgFinishLast5", "DNFRateLast10", "Reasons"
    ] if c in result_df.columns]
    st.dataframe(result_df[display_cols], use_container_width=True, hide_index=True)

    # ── Lock / Save controls ──────────────────────────────────────────────────
    st.markdown("---")
    col_save, col_lock = st.columns(2)

    top10_payload = [
        {
            "pos":    int(row["PredictedFinish"]),
            "driver": row["FullName"],
            "team":   row.get("TeamName", ""),
            "prob":   round(float(row["WinProbability"]) / 100, 4),
        }
        for _, row in result_df.head(10).iterrows()
    ]
    win_probs = {
        row["FullName"]: round(float(row["WinProbability"]) / 100, 4)
        for _, row in result_df.iterrows()
    }

    with col_save:
        if not is_my_locked:
            if st.button("💾 Save as Draft", use_container_width=True):
                try:
                    pred_store.save_prediction(
                        user_id, selected_race, year, selected_round,
                        top10_payload, win_probs, fp1_time
                    )
                    st.success("Draft saved! You can regenerate until FP1 starts.")
                except PredictionLockedError as e:
                    st.error(str(e))

    with col_lock:
        if not is_my_locked and not locked_now:
            if st.button("🔒 Lock This Prediction", use_container_width=True, type="primary"):
                try:
                    pred_store.save_prediction(
                        user_id, selected_race, year, selected_round,
                        top10_payload, win_probs, fp1_time
                    )
                    pred_store.lock_prediction(user_id, selected_race, year)
                    st.success(f"✅ Prediction locked for {selected_race}!")
                    st.balloons()
                except (PredictionLockedError, PredictionError) as e:
                    st.error(str(e))
        elif is_my_locked:
            st.success("✅ Your prediction is locked for this race.")

# ── Show existing locked prediction ──────────────────────────────────────────
elif is_my_locked and result_df is None:
    st.info("You have a locked prediction for this race. View it in 🔬 My Predictions.")
