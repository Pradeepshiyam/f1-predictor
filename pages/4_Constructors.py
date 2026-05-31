"""
Page 4 — Constructor Analysis

All team lists derived dynamically from ingested data.
Features:
- Constructor points table + bar chart
- Both driver results per race per team
- Points contribution split (Driver 1 vs Driver 2)
- Circuit strengths per team
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import glob

from f1_predictor.common.auth import require_auth
from f1_predictor.common.config import settings

require_auth()

st.markdown("## 🏗️ Constructor Analysis")
st.markdown("---")

bronze_path = settings.bronze_path

@st.cache_data(ttl=300)
def load_all_bronze() -> pd.DataFrame:
    files = glob.glob(str(bronze_path / "*_results.csv"))
    frames = []
    for f in files:
        try:
            df = pd.read_csv(f)
            parts = os.path.basename(f).replace("_results.csv", "").split("_", 1)
            df["Year"]    = int(parts[0])
            df["Circuit"] = parts[1].replace("_", " ") if len(parts) > 1 else "Unknown"
            frames.append(df)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    for col in ["Position", "Points", "GridPosition"]:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")
    return combined

all_data = load_all_bronze()

if all_data.empty or "TeamName" not in all_data.columns:
    st.warning("No data available.")
    st.stop()

current_year = settings.current_year
curr_data = all_data[all_data["Year"].astype(str) == str(current_year)]
if curr_data.empty:
    curr_data = all_data

# Dynamic team selector from current season
teams = sorted(curr_data["TeamName"].dropna().unique().tolist())

col_t1, col_t2 = st.columns(2)
with col_t1:
    selected_team = st.selectbox("Select Constructor:", teams, index=0)
with col_t2:
    year_options = sorted(all_data["Year"].unique(), reverse=True)
    selected_year = st.selectbox("Season:", year_options, index=0)

team_data = all_data[
    (all_data["TeamName"] == selected_team) &
    (all_data["Year"].astype(str) == str(selected_year))
].copy()

st.markdown("---")

if team_data.empty:
    st.info(f"No data for {selected_team} in {selected_year}.")
else:
    # ── Team drivers for selected year ────────────────────────────────────────
    team_drivers = team_data["FullName"].dropna().unique().tolist()
    driver_colors = ["#e10600", "#3498db", "#2ecc71", "#f39c12"]
    color_map = {d: driver_colors[i % len(driver_colors)] for i, d in enumerate(team_drivers)}

    # ── Points tally ──────────────────────────────────────────────────────────
    st.markdown(f"### 📊 {selected_team} — {selected_year} Summary")

    stats = (
        team_data.groupby("FullName")
        .agg(Points=("Points", "sum"), Wins=("Position", lambda x: (x==1).sum()),
             Podiums=("Position", lambda x: (x<=3).sum()), Races=("Position", "count"))
        .reset_index()
    )
    cols = st.columns(len(team_drivers))
    for col, driver in zip(cols, team_drivers):
        row = stats[stats["FullName"] == driver]
        if not row.empty:
            r = row.iloc[0]
            with col:
                st.metric(driver, f"{int(r['Points'])} pts")
                st.caption(f"W:{int(r['Wins'])} P:{int(r['Podiums'])} R:{int(r['Races'])}")

    # ── Race-by-race results ──────────────────────────────────────────────────
    st.markdown(f"### 📈 Race-by-Race Finishing Positions — {selected_team}")

    fig = go.Figure()
    for driver in team_drivers:
        d_data = team_data[team_data["FullName"] == driver].sort_values("Circuit")
        fig.add_trace(go.Scatter(
            x=d_data["Circuit"], y=d_data["Position"],
            mode="lines+markers", name=driver,
            line=dict(color=color_map[driver], width=2),
            marker=dict(size=8),
        ))
    fig.update_layout(
        yaxis=dict(title="Finishing Position", autorange="reversed"),
        xaxis_tickangle=-45, height=400,
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#e0e0e0"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Points contribution split ─────────────────────────────────────────────
    st.markdown(f"### 🥧 Points Contribution Split")

    per_race_pts = (
        team_data.groupby(["Circuit", "FullName"])["Points"]
        .sum().reset_index()
    )
    fig2 = px.bar(
        per_race_pts, x="Circuit", y="Points", color="FullName",
        barmode="stack",
        color_discrete_map=color_map,
        title=f"{selected_team} Points Split by Race",
    )
    fig2.update_layout(
        xaxis_tickangle=-45, height=380,
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#e0e0e0"),
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ── Circuit strengths ─────────────────────────────────────────────────────
    st.markdown(f"### 🏟️ Circuit Performance — {selected_team}")
    circuit_avg = (
        team_data.groupby("Circuit")["Position"].mean()
        .reset_index().rename(columns={"Position": "AvgFinish"})
        .sort_values("AvgFinish")
    )
    fig3 = px.bar(
        circuit_avg, x="Circuit", y="AvgFinish",
        color="AvgFinish", color_continuous_scale="RdYlGn_r",
        title=f"Average Finishing Position by Circuit — {selected_team}",
        text=circuit_avg["AvgFinish"].round(1),
    )
    fig3.update_layout(
        xaxis_tickangle=-45, height=380,
        yaxis=dict(title="Avg Finish", autorange="reversed"),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#e0e0e0"),
    )
    st.plotly_chart(fig3, use_container_width=True)

# ── All-season constructor comparison ─────────────────────────────────────────
st.markdown("---")
st.markdown(f"### 🏆 All Teams — {selected_year} Championship")

all_teams_pts = (
    all_data[all_data["Year"].astype(str) == str(selected_year)]
    .groupby("TeamName")["Points"].sum()
    .reset_index().sort_values("Points", ascending=False)
)
fig4 = px.bar(
    all_teams_pts, x="TeamName", y="Points",
    color="Points", color_continuous_scale="reds", text="Points",
    title=f"{selected_year} Constructor Championship Standings",
)
fig4.update_layout(
    height=380, plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
    font=dict(color="#e0e0e0"), showlegend=False,
)
st.plotly_chart(fig4, use_container_width=True)
