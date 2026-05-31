"""Page 6 — Historical Deep Dive (2021–now)"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import streamlit as st
import pandas as pd
import plotly.express as px
import glob

from f1_predictor.common.auth import require_auth
from f1_predictor.common.config import settings

require_auth()
st.markdown("## 📊 Historical Deep Dive")
st.markdown("---")

bronze_path = settings.bronze_path

@st.cache_data(ttl=600)
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
    st.warning("No historical data available.")
    st.stop()

all_years = sorted(all_data["Year"].unique(), reverse=True)
selected_year = st.selectbox("Season:", ["All Time"] + [str(y) for y in all_years])
yr_data = all_data if selected_year == "All Time" else all_data[all_data["Year"].astype(str) == selected_year]

st.markdown("---")
st.markdown(f"### 🏆 Driver Stats — {selected_year}")

stats = (
    yr_data.groupby("FullName")
    .agg(Races=("Position","count"), Wins=("Position", lambda x:(x==1).sum()),
         Podiums=("Position", lambda x:(x<=3).sum()), Points=("Points","sum"),
         AvgFinish=("Position","mean"), DNFs=("Position", lambda x:(x>20).sum()))
    .reset_index().sort_values("Points", ascending=False).reset_index(drop=True)
)
stats["Rank"] = range(1, len(stats)+1)
stats["AvgFinish"] = stats["AvgFinish"].round(2)
st.dataframe(stats[["Rank","FullName","Points","Wins","Podiums","Races","AvgFinish","DNFs"]],
             use_container_width=True, hide_index=True)

top_winners = stats[stats["Wins"]>0].sort_values("Wins", ascending=False).head(15)
if not top_winners.empty:
    fig = px.bar(top_winners, x="FullName", y="Wins", color="Wins",
                 color_continuous_scale="reds", text="Wins",
                 title=f"Race Wins — {selected_year}")
    fig.update_layout(height=360, xaxis_tickangle=-30,
                      plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                      font=dict(color="#e0e0e0"), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

if selected_year != "All Time":
    st.markdown(f"### 📈 {selected_year} Championship Replay")
    race_order = sorted(yr_data["Circuit"].unique())
    yr_data = yr_data.copy()
    yr_data["CircuitIdx"] = yr_data["Circuit"].map({c:i for i,c in enumerate(race_order)})
    yr_data = yr_data.sort_values(["FullName","CircuitIdx"])
    yr_data["CumPoints"] = yr_data.groupby("FullName")["Points"].cumsum()
    top10 = stats.head(10)["FullName"].tolist()
    fig2 = px.line(yr_data[yr_data["FullName"].isin(top10)],
                   x="Circuit", y="CumPoints", color="FullName", markers=True,
                   title=f"{selected_year} Cumulative Points")
    fig2.update_layout(xaxis_tickangle=-45, height=420,
                       plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                       font=dict(color="#e0e0e0"))
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")
st.markdown("### 🏟️ Circuit Records")
circuits = sorted(all_data["Circuit"].dropna().unique())
circ = st.selectbox("Circuit:", circuits)
circ_data = all_data[all_data["Circuit"]==circ]
if not circ_data.empty:
    c1, c2 = st.columns(2)
    with c1:
        winners = circ_data[circ_data["Position"]==1]["FullName"].value_counts().reset_index().rename(columns={"count":"Wins"})
        st.markdown(f"**Most Wins**"); st.dataframe(winners.head(10), use_container_width=True, hide_index=True)
    with c2:
        avg = circ_data.groupby("FullName")["Position"].mean().reset_index().rename(columns={"Position":"AvgFinish"}).sort_values("AvgFinish").head(10)
        avg["AvgFinish"] = avg["AvgFinish"].round(2)
        st.markdown(f"**Best Avg Finish**"); st.dataframe(avg, use_container_width=True, hide_index=True)
