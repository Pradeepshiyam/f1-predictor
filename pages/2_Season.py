"""
Page 2 — Season Tracker

Shows all live 2026 season data dynamically derived from ingested Bronze data:
- Race calendar progress (past vs upcoming)
- Driver championship standings
- Constructor standings
- Points gap over the season
- Win / Podium / DNF tally per driver
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from f1_predictor.common.auth import require_auth
from f1_predictor.common.calendar import get_season_schedule
from f1_predictor.common.config import settings

require_auth()

st.markdown("## 📅 Season Tracker")
st.markdown("---")

year = settings.current_year
bronze_path = settings.bronze_path

# ── Load current season Bronze data ──────────────────────────────────────────
@st.cache_data(ttl=300)
def load_season_data(yr: int) -> pd.DataFrame:
    import glob
    files = glob.glob(str(bronze_path / f"{yr}_*_results.csv"))
    if not files:
        return pd.DataFrame()
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


season_data = load_season_data(year)

# ── Calendar Progress ─────────────────────────────────────────────────────────
st.markdown("### 🗓️ Race Calendar Progress")

try:
    schedule = get_season_schedule(year)
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    schedule["status"] = schedule["event_date"].apply(
        lambda d: "✅ Completed" if d < now else "🔜 Upcoming"
    )
    completed = schedule[schedule["status"] == "✅ Completed"]
    upcoming  = schedule[schedule["status"] == "🔜 Upcoming"]

    col_c, col_u = st.columns(2)
    with col_c:
        st.metric("Races Completed", len(completed))
    with col_u:
        st.metric("Races Remaining", len(upcoming))

    st.dataframe(
        schedule[["round_number", "race_name", "country", "event_date", "status"]]
        .rename(columns={
            "round_number": "Round", "race_name": "Grand Prix",
            "country": "Country", "event_date": "Race Date", "status": "Status"
        }),
        use_container_width=True, hide_index=True,
    )
except Exception as e:
    st.warning(f"Calendar unavailable: {e}")

if season_data.empty:
    st.info(f"No {year} race results ingested yet. Run `ingest_historical.py` after each race.")
    st.stop()

st.markdown("---")

# ── Driver Championship Standings ─────────────────────────────────────────────
st.markdown("### 🏆 Driver Championship Standings")

driver_pts = (
    season_data.groupby("FullName")
    .agg(
        Points=("Points", "sum"),
        TeamName=("TeamName", "last"),
        Wins=("Position", lambda x: (x == 1).sum()),
        Podiums=("Position", lambda x: (x <= 3).sum()),
        Races=("Position", "count"),
    )
    .reset_index()
    .sort_values("Points", ascending=False)
    .reset_index(drop=True)
)
driver_pts["Position"] = range(1, len(driver_pts) + 1)
driver_pts["Gap"]      = driver_pts["Points"].iloc[0] - driver_pts["Points"]

st.dataframe(
    driver_pts[["Position", "FullName", "TeamName", "Points", "Gap", "Wins", "Podiums", "Races"]],
    use_container_width=True, hide_index=True,
)

# ── Points progression chart ──────────────────────────────────────────────────
st.markdown("### 📈 Points Progression")

race_order = sorted(season_data["Circuit"].unique())
cum_pts    = []
for circuit in race_order:
    race_slice = season_data[season_data["Circuit"] == circuit]
    for _, row in race_slice.iterrows():
        cum_pts.append({"Circuit": circuit, "FullName": row["FullName"], "Points": row["Points"]})

cum_df = pd.DataFrame(cum_pts)
if not cum_df.empty:
    cum_df["CumPoints"] = cum_df.groupby("FullName")["Points"].cumsum()
    top_drivers = driver_pts.head(10)["FullName"].tolist()
    chart_df = cum_df[cum_df["FullName"].isin(top_drivers)]

    fig = px.line(
        chart_df, x="Circuit", y="CumPoints", color="FullName",
        title="Cumulative Points — Top 10 Drivers",
        markers=True,
    )
    fig.update_layout(
        xaxis_tickangle=-45, height=450,
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#e0e0e0"),
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Constructor Standings ─────────────────────────────────────────────────────
st.markdown("### 🏗️ Constructor Championship")

if "TeamName" in season_data.columns:
    team_pts = (
        season_data.groupby("TeamName")["Points"].sum()
        .reset_index().sort_values("Points", ascending=False)
        .reset_index(drop=True)
    )
    team_pts["Position"] = range(1, len(team_pts) + 1)
    team_pts["Gap"]      = team_pts["Points"].iloc[0] - team_pts["Points"]

    fig3 = px.bar(
        team_pts, x="TeamName", y="Points", color="Points",
        color_continuous_scale="reds", text="Points",
        title="Constructor Points",
    )
    fig3.update_layout(
        height=380, plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#e0e0e0"), showlegend=False,
    )
    st.plotly_chart(fig3, use_container_width=True)
    st.dataframe(team_pts[["Position", "TeamName", "Points", "Gap"]],
                 use_container_width=True, hide_index=True)

# ── DNF Tracker ───────────────────────────────────────────────────────────────
st.markdown("### ⚠️ DNF & Reliability Tracker")

if "Position" in season_data.columns:
    dnf_df = (
        season_data.assign(DNF=season_data["Position"] > 20)
        .groupby("FullName")
        .agg(Races=("Position", "count"), DNFs=("DNF", "sum"))
        .reset_index()
    )
    dnf_df["DNF Rate"] = (dnf_df["DNFs"] / dnf_df["Races"] * 100).round(1)
    dnf_df = dnf_df.sort_values("DNFs", ascending=False)
    st.dataframe(dnf_df, use_container_width=True, hide_index=True)
