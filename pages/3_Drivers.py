"""
Page 3 — Driver Deep Dive

All driver and circuit lists derived dynamically from ingested data.
Features:
- Driver selector (active drivers from current season)
- Career stats: races, wins, podiums, DNFs, avg finish, avg grid
- Season form: Grid vs Finish per race (line chart)
- Circuit heatmap: avg finish per track
- Head-to-head comparison against any rival
- Qualifying vs Race delta chart
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

st.markdown("## 🏎️ Driver Deep Dive")
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

if all_data.empty:
    st.warning("No historical data available. Run ingest_historical.py first.")
    st.stop()

# Dynamic driver list from current season
current_year = settings.current_year
yr_data = all_data[all_data["Year"].astype(str) == str(current_year)]
if yr_data.empty:
    yr_data = all_data  # fallback to all data if no current year

active_drivers = sorted(yr_data["FullName"].dropna().unique().tolist())

col_d1, col_d2 = st.columns(2)
with col_d1:
    driver1 = st.selectbox("Select Driver:", active_drivers, index=0)
with col_d2:
    rival_options = ["None"] + [d for d in active_drivers if d != driver1]
    rival = st.selectbox("Compare with Rival (optional):", rival_options, index=0)

st.markdown("---")

# ── Driver Data ───────────────────────────────────────────────────────────────
d1_data = all_data[all_data["FullName"] == driver1].copy()
d1_curr = d1_data[d1_data["Year"].astype(str) == str(current_year)]

if d1_data.empty:
    st.warning(f"No data found for {driver1}.")
    st.stop()

# ── Career Stats ──────────────────────────────────────────────────────────────
st.markdown(f"### 📊 Career Stats — {driver1}")

total_races  = len(d1_data)
total_wins   = int((d1_data["Position"] == 1).sum())
total_podiums = int((d1_data["Position"] <= 3).sum())
total_dnf    = int((d1_data["Position"] > 20).sum())
avg_finish   = d1_data["Position"].mean()
avg_grid     = d1_data["GridPosition"].mean() if "GridPosition" in d1_data.columns else None

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Races", total_races)
c2.metric("Wins", total_wins)
c3.metric("Podiums", total_podiums)
c4.metric("DNFs", total_dnf)
c5.metric("Avg Finish", f"{avg_finish:.1f}")

# ── Season Form: Grid vs Finish ───────────────────────────────────────────────
if not d1_curr.empty:
    st.markdown(f"### 📈 {current_year} Season Form — Grid vs Finish")
    d1_curr_sorted = d1_curr.sort_values("Circuit")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=d1_curr_sorted["Circuit"], y=d1_curr_sorted["GridPosition"],
        mode="lines+markers", name="Grid Position",
        line=dict(color="#3498db", width=2), marker=dict(size=8),
    ))
    fig.add_trace(go.Scatter(
        x=d1_curr_sorted["Circuit"], y=d1_curr_sorted["Position"],
        mode="lines+markers", name="Finishing Position",
        line=dict(color="#e10600", width=2, dash="dot"), marker=dict(size=8, symbol="diamond"),
    ))
    fig.update_layout(
        yaxis=dict(title="Position", autorange="reversed"),
        xaxis_tickangle=-45, height=380,
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#e0e0e0"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Circuit Heatmap ───────────────────────────────────────────────────────────
st.markdown(f"### 🗺️ Circuit Heatmap — {driver1}")

circuit_avg = (
    d1_data.groupby("Circuit")["Position"]
    .mean().reset_index()
    .rename(columns={"Position": "AvgFinish"})
    .sort_values("AvgFinish")
)

fig2 = px.bar(
    circuit_avg, x="Circuit", y="AvgFinish",
    title=f"Average Finishing Position by Circuit — {driver1}",
    color="AvgFinish", color_continuous_scale="RdYlGn_r",
    text=circuit_avg["AvgFinish"].round(1),
)
fig2.update_layout(
    xaxis_tickangle=-45, height=400,
    yaxis=dict(title="Avg Finish Position", autorange="reversed"),
    plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
    font=dict(color="#e0e0e0"),
)
st.plotly_chart(fig2, use_container_width=True)

# ── Qualifying vs Race Delta ──────────────────────────────────────────────────
if "GridPosition" in d1_data.columns:
    st.markdown(f"### ⚡ Qualifying vs Race Delta — {driver1}")
    d1_data_clean = d1_data.dropna(subset=["GridPosition", "Position"])
    d1_data_clean = d1_data_clean.copy()
    d1_data_clean["Delta"] = d1_data_clean["GridPosition"] - d1_data_clean["Position"]
    # Positive delta = gained positions

    circuit_delta = d1_data_clean.groupby("Circuit")["Delta"].mean().reset_index()
    fig3 = px.bar(
        circuit_delta, x="Circuit", y="Delta",
        color="Delta", color_continuous_scale="RdYlGn",
        title=f"Positions Gained/Lost vs Grid — {driver1} (positive = gained)",
        text=circuit_delta["Delta"].round(1),
    )
    fig3.update_layout(
        xaxis_tickangle=-45, height=380,
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#e0e0e0"),
    )
    st.plotly_chart(fig3, use_container_width=True)

# ── Head-to-Head ──────────────────────────────────────────────────────────────
if rival != "None":
    st.markdown(f"### ⚔️ Head-to-Head: {driver1} vs {rival}")

    d2_data = all_data[all_data["FullName"] == rival]

    common_races = set(zip(d1_data["Year"], d1_data["Circuit"])) & \
                   set(zip(d2_data["Year"], d2_data["Circuit"]))

    if not common_races:
        st.info("No common races found between these two drivers.")
    else:
        d1_common = d1_data[d1_data.apply(lambda r: (r["Year"], r["Circuit"]) in common_races, axis=1)]
        d2_common = d2_data[d2_data.apply(lambda r: (r["Year"], r["Circuit"]) in common_races, axis=1)]

        h2h_stats = {
            "Metric": ["Races Together", "Wins", "Podiums", "Avg Finish", "Avg Grid", "DNFs"],
            driver1:  [
                len(d1_common),
                int((d1_common["Position"] == 1).sum()),
                int((d1_common["Position"] <= 3).sum()),
                round(d1_common["Position"].mean(), 2),
                round(d1_common.get("GridPosition", pd.Series()).mean(), 2) if "GridPosition" in d1_common.columns else "—",
                int((d1_common["Position"] > 20).sum()),
            ],
            rival: [
                len(d2_common),
                int((d2_common["Position"] == 1).sum()),
                int((d2_common["Position"] <= 3).sum()),
                round(d2_common["Position"].mean(), 2),
                round(d2_common.get("GridPosition", pd.Series()).mean(), 2) if "GridPosition" in d2_common.columns else "—",
                int((d2_common["Position"] > 20).sum()),
            ],
        }
        st.dataframe(pd.DataFrame(h2h_stats), use_container_width=True, hide_index=True)
