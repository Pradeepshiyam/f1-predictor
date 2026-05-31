import json
import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
import plotly.express as px
import plotly.graph_objects as go

# ── Driver name aliases (FastF1 changed Antonelli's name mid-career) ─────────
DRIVER_ALIASES: dict[str, str] = {
    "Andrea Kimi Antonelli": "Kimi Antonelli",
    "Isack HADJAR": "Isack Hadjar",
    "Oliver BEARMAN": "Oliver Bearman",
    "Franco COLAPINTO": "Franco Colapinto",
    "Nico Hülkenberg": "Nico Hulkenberg",
    "Nico HÜLKENBERG": "Nico Hulkenberg",
}

def _norm(name: str) -> str:
    if pd.isna(name): return "Unknown"
    return DRIVER_ALIASES.get(str(name).strip(), str(name).strip())

st.set_page_config(page_title="F1 Prediction Center", layout="wide", page_icon="🏎️")

st.title("🏎️ F1 Prediction Command Center")
st.markdown("---")

BRONZE_PATH = "data/bronze/historical"
GOLD_PATH = "data/gold/features.parquet"

# ── Sidebar ─────────────────────────────────────────────────────────────────
st.sidebar.header("Controls")
st.sidebar.subheader("Data Status")
if os.path.exists(BRONZE_PATH):
    total_files = len(os.listdir(BRONZE_PATH))
    st.sidebar.success(f"Ingested {total_files} data files.")
else:
    st.sidebar.warning("No data found.")
st.sidebar.markdown("---")
st.sidebar.info("Built with Pandas · XGBoost · Streamlit\nPySpark ready for ETL scale-up")


# ── Helper: Load ALL bronze results into one DataFrame ──────────────────────
@st.cache_data(ttl=300)
def load_all_bronze() -> pd.DataFrame:
    """Load every *_results.csv from bronze layer, tag Year + Circuit."""
    files = [f for f in glob.glob(f"{BRONZE_PATH}/*_results.csv")
             if os.path.getsize(f) > 200]
    if not files:
        return pd.DataFrame()
    frames = []
    for f in files:
        try:
            fname = os.path.basename(f)
            parts = fname.replace("_results.csv", "").split("_")
            year = parts[0]
            circuit = " ".join(parts[1:])          # e.g. "Australian Grand Prix"
            df = pd.read_csv(f)
            # Normalise driver names to remove duplicates like Antonelli
            if "FullName" in df.columns:
                df["FullName"] = df["FullName"].apply(_norm)
            df["Year"]    = year
            df["Circuit"] = circuit
            df["Source"]  = fname
            frames.append(df)
        except Exception:
            continue
    combined = pd.concat(frames, ignore_index=True)
    # Final dedup: keep last occurrence of each (Year, Circuit, FullName)
    if "FullName" in combined.columns:
        combined = combined.drop_duplicates(
            subset=["Year", "Circuit", "FullName"], keep="last"
        )
    return combined


@st.cache_data(ttl=300)
def load_all_qualifying() -> pd.DataFrame:
    """Load every *_qualifying.csv from bronze layer."""
    files = [f for f in glob.glob(f"{BRONZE_PATH}/*_qualifying.csv")
             if os.path.getsize(f) > 200]
    if not files:
        return pd.DataFrame()
    frames = []
    for f in files:
        try:
            fname = os.path.basename(f)
            parts = fname.replace("_qualifying.csv", "").split("_")
            year = parts[0]
            circuit = " ".join(parts[1:])
            df = pd.read_csv(f)
            df["Year"]    = year
            df["Circuit"] = circuit
            frames.append(df)
        except Exception:
            continue
    return pd.concat(frames, ignore_index=True)


all_bronze = load_all_bronze()
all_qual   = load_all_qualifying()


# ── Helper: Build rich driver feature table ─────────────────────────────────
def build_driver_features(selected_circuit: str) -> pd.DataFrame:
    """
    For each driver, compute:
      - avg_quali_pos      : historical average qualifying position (all circuits)
      - avg_finish_overall : historical average race finish (all circuits)
      - avg_finish_circuit : historical average race finish AT THIS circuit
      - team_form          : team's total points in last 5 races
      - track_specialist   : True if avg_finish_circuit < avg_finish_overall - 2
    Returns one row per driver.
    """
    if all_bronze.empty:
        return pd.DataFrame()

    # ── Step 0: Get ACTIVE drivers from the most recent season only ───────────
    # This prevents retired drivers from 2021-2024 appearing in predictions.
    latest_year = str(all_bronze["Year"].astype(str).max())
    latest_year_data = all_bronze[all_bronze["Year"].astype(str) == latest_year]

    # If the latest year has very few races (e.g. only 1-3), also allow prior year
    if latest_year_data["Circuit"].nunique() < 3:
        second_year = str(sorted(all_bronze["Year"].astype(str).unique())[-2])
        latest_year_data = all_bronze[all_bronze["Year"].astype(str).isin([latest_year, second_year])]

    active_drivers = set(latest_year_data["FullName"].dropna().unique())

    # Filter all_bronze to only active drivers for feature computation
    hist = all_bronze[all_bronze["FullName"].isin(active_drivers)].copy()

    # --- Overall driver averages (from full history, active drivers only) ---
    driver_overall = (
        hist
        .groupby("FullName")
        .agg(
            avg_finish_overall=("Position", "mean"),
            TeamName=("TeamName", "last"),
            HeadshotUrl=("HeadshotUrl", "last"),
        )
        .reset_index()
    )

    # --- Qualifying averages (active drivers only) ---
    if not all_qual.empty and "GridPosition" in all_qual.columns:
        qual_active = all_qual[all_qual["FullName"].isin(active_drivers)]
        quali_avg = (
            qual_active
            .groupby("FullName")["GridPosition"]
            .mean()
            .reset_index()
            .rename(columns={"GridPosition": "avg_quali_pos"})
        )
    else:
        quali_avg = (
            hist
            .groupby("FullName")["GridPosition"]
            .mean()
            .reset_index()
            .rename(columns={"GridPosition": "avg_quali_pos"})
        )

    # --- Circuit-specific finishing average (active drivers only) ---
    keyword = selected_circuit.split()[0].lower()
    circuit_mask = hist["Circuit"].str.lower().str.contains(keyword, na=False)
    circuit_data = hist[circuit_mask]

    if not circuit_data.empty:
        circuit_avg = (
            circuit_data
            .groupby("FullName")["Position"]
            .mean()
            .reset_index()
            .rename(columns={"Position": "avg_finish_circuit"})
        )
    else:
        circuit_avg = pd.DataFrame(columns=["FullName", "avg_finish_circuit"])

    # --- Team form: last 5 races from most recent data ---
    sorted_hist = hist.sort_values(["Year", "Circuit"])
    team_form = (
        sorted_hist
        .groupby("TeamName")["Points"]
        .apply(lambda x: x.iloc[-5:].sum() if len(x) >= 5 else x.sum())
        .reset_index()
        .rename(columns={"Points": "team_form"})
    )

    # --- Merge everything ---
    feats = (
        driver_overall
        .merge(quali_avg,   on="FullName", how="left")
        .merge(circuit_avg, on="FullName", how="left")
        .merge(team_form,   on="TeamName", how="left")
    )

    # Fill missing with sensible defaults
    feats["avg_quali_pos"]      = feats["avg_quali_pos"].fillna(12.0)
    feats["avg_finish_overall"] = feats["avg_finish_overall"].fillna(12.0)
    feats["avg_finish_circuit"] = feats["avg_finish_circuit"].fillna(feats["avg_finish_overall"])
    feats["team_form"]          = feats["team_form"].fillna(feats["team_form"].median())

    # Track specialist flag
    feats["track_delta"]     = feats["avg_finish_overall"] - feats["avg_finish_circuit"]
    feats["track_specialist"] = feats["track_delta"] > 2.0

    return feats



# ── Section 1: Historical Data Viewer ───────────────────────────────────────
st.header("📊 Historical Data Viewer")
if not all_bronze.empty:
    files = sorted([f for f in os.listdir(BRONZE_PATH) if f.endswith("_results.csv")
                    and os.path.getsize(os.path.join(BRONZE_PATH, f)) > 200])
    if files:
        selected_file = st.selectbox("Select a race to view:", files)
        df_view = pd.read_csv(os.path.join(BRONZE_PATH, selected_file))
        cols_show = [c for c in ["Position", "FullName", "TeamName", "GridPosition", "Points"]
                     if c in df_view.columns]
        st.dataframe(df_view[cols_show], height=380)
else:
    st.info("No data ingested yet.")


# ── Section 2: Prediction Engine ─────────────────────────────────────────────
st.markdown("---")
st.header("🔮 Dynamic Prediction Engine")

# Build GP list from ALL unique circuits in bronze data
if not all_bronze.empty:
    available_gps = sorted(all_bronze["Circuit"].unique())
else:
    available_gps = ["Miami Grand Prix"]

# Also add upcoming races not yet in historical data
upcoming = [
    "Miami Grand Prix", "Monaco Grand Prix", "Spanish Grand Prix",
    "Canadian Grand Prix", "British Grand Prix", "Hungarian Grand Prix",
    "Belgian Grand Prix", "Dutch Grand Prix", "Italian Grand Prix",
    "Singapore Grand Prix", "Japanese Grand Prix", "United States Grand Prix",
    "Mexican Grand Prix", "Brazilian Grand Prix", "Las Vegas Grand Prix",
    "Abu Dhabi Grand Prix",
]
all_gps = sorted(set(available_gps) | set(upcoming))

col1, col2 = st.columns([1, 2])
with col1:
    st.subheader("Race Configuration")
    selected_gp = st.selectbox("Select Upcoming GP:", all_gps,
                               index=all_gps.index("Miami Grand Prix") if "Miami Grand Prix" in all_gps else 0)
    predict_btn = st.button("🚀 Generate AI Prediction", use_container_width=True)


# ── Prediction Logic ──────────────────────────────────────────────────────────
if predict_btn:
    try:
        import xgboost as xgb

        model_file = "models/race_winner_model.json"
        qual_model_file = "models/pole_position_model.json"

        if not os.path.exists(model_file):
            st.error(f"Race model not found at `{model_file}`. Please run the training script.")
            st.stop()
        if not os.path.exists(qual_model_file):
            st.error(f"Qualifying model not found at `{qual_model_file}`. Please run the training script.")
            st.stop()

        model      = xgb.XGBClassifier(); model.load_model(model_file)
        qual_model = xgb.XGBClassifier(); qual_model.load_model(qual_model_file)

        with st.spinner(f"🧠 AI is simulating the {selected_gp}..."):
            # Build rich features
            feats = build_driver_features(selected_gp)

            if feats.empty:
                st.error("No historical data available to build driver features.")
                st.stop()

            # Load the feature list the model was trained on (from sidecar JSON)
            def load_model_features(model_path: str, fallback: list) -> list:
                feat_file = model_path.replace(".json", "_features.json")
                if os.path.exists(feat_file):
                    with open(feat_file) as fh:
                        return json.load(fh)
                return fallback

            DEFAULT_FEATS = ["GridPosition", "AvgFinishLast5", "TeamForm"]
            qual_feats = load_model_features(qual_model_file, DEFAULT_FEATS)
            race_feats = load_model_features(model_file, DEFAULT_FEATS)

            # Map our rich feature table to whatever feature names the model expects
            FEAT_MAP = {
                "GridPosition":    "avg_quali_pos",
                "AvgFinishLast5":  "avg_finish_overall",
                "AvgFinishLast3":  "avg_finish_overall",
                "AvgGridLast5":    "avg_quali_pos",
                "TeamForm":        "team_form",
                "TeamPointsLast5": "team_form",
                "CircuitAvgFinish":"avg_finish_circuit",
                "CircuitWins":     "circuit_wins",
                "DNFRateLast10":   "dnf_rate",
                "ChampPos":        "champ_pos",
                "AvgFinishLast10": "avg_finish_overall",
            }
            # Ensure the columns exist in feats (create them if missing)
            if "circuit_wins" not in feats.columns:
                feats["circuit_wins"] = 0
            if "dnf_rate" not in feats.columns:
                feats["dnf_rate"] = 0.05
            if "champ_pos" not in feats.columns:
                feats["champ_pos"] = 10.0

            def build_X(feature_names: list) -> pd.DataFrame:
                cols = {}
                for fn in feature_names:
                    src = FEAT_MAP.get(fn, None)
                    if src and src in feats.columns:
                        cols[fn] = feats[src].values
                    elif fn in feats.columns:
                        cols[fn] = feats[fn].values
                    else:
                        cols[fn] = np.full(len(feats), 10.0)
                return pd.DataFrame(cols).fillna(0)

            # ── STEP 1: Predict Qualifying (Grid) ────────────────────────────
            X_qual = build_X(qual_feats)
            qual_probs = qual_model.predict_proba(X_qual)[:, 1]
            feats["QualScore"] = qual_probs
            feats = feats.sort_values("QualScore", ascending=False).reset_index(drop=True)
            feats["PredictedGrid"] = range(1, len(feats) + 1)

            # ── STEP 2: Predict Race Finish ──────────────────────────────────
            # Inject predicted grid as GridPosition for race model
            feats["avg_quali_pos"] = feats["PredictedGrid"].astype(float)
            X_race = build_X(race_feats)

            base_probs = model.predict_proba(X_race)[:, 1]

            # Apply circuit-specific track score as probability multiplier
            # Better avg at this circuit → higher multiplier (max 1.3x, min 0.7x)
            max_finish = feats["avg_finish_circuit"].max()
            min_finish = feats["avg_finish_circuit"].min()
            rng = max(max_finish - min_finish, 1)
            # Lower finish number = better, so invert for multiplier
            track_multiplier = 1.0 + 0.3 * (
                (max_finish - feats["avg_finish_circuit"]) / rng - 0.5
            )
            feats["WinProbability"] = base_probs * track_multiplier.values

            # Normalize so probabilities sum to ~1 (makes them interpretable)
            total_prob = feats["WinProbability"].sum()
            if total_prob > 0:
                feats["WinProbability"] = feats["WinProbability"] / total_prob

            feats = feats.sort_values("WinProbability", ascending=False).reset_index(drop=True)
            feats["PredictedFinish"] = range(1, len(feats) + 1)

        # ── OUTPUT A: Podium ─────────────────────────────────────────────────
        st.write(f"### 🏁 AI Race Prediction — {selected_gp}")
        top3 = feats.head(3)
        pod_cols = st.columns(3)
        medals = ["🥇", "🥈", "🥉"]

        for i, (_, row) in enumerate(top3.iterrows()):
            with pod_cols[i]:
                img_url = row.get("HeadshotUrl", "")
                if pd.notna(img_url) and str(img_url).startswith("http"):
                    st.image(str(img_url), width=150)
                else:
                    st.markdown(f"## {medals[i]}")
                st.markdown(f"**{row['FullName']}**")
                st.metric("Win Probability", f"{row['WinProbability']*100:.1f}%")

                # AI Reasoning — multi-factor
                reasons = []
                if row["track_specialist"]:
                    reasons.append("📍 Track Specialist")
                if row["PredictedGrid"] <= 3:
                    reasons.append("🎯 Pole Contender")
                if row["avg_finish_circuit"] <= 5:
                    reasons.append("🔥 Circuit Ace")
                if row["team_form"] >= feats["team_form"].quantile(0.75):
                    reasons.append("🚀 Team Momentum")
                if row["avg_finish_overall"] <= 6:
                    reasons.append("💪 Season Form")
                if not reasons:
                    reasons.append("📈 Consistent Pace")

                st.caption(" · ".join(reasons))
                st.caption(
                    f"Pred. Grid: P{int(row['PredictedGrid'])} | "
                    f"Circuit Avg: P{row['avg_finish_circuit']:.1f} | "
                    f"Overall Avg: P{row['avg_finish_overall']:.1f}"
                )

        # ── OUTPUT B: Full Grid Chart (Predicted Grid vs Predicted Finish) ───
        st.markdown("---")
        st.subheader(f"📊 Predicted Starting Grid vs. Finishing Order — {selected_gp}")

        chart_data = feats.sort_values("PredictedGrid").copy()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=chart_data["FullName"], y=chart_data["PredictedGrid"],
            mode="lines+markers", name="Predicted Grid (Qualifying)",
            line=dict(color="#3b82f6", width=2),
            marker=dict(size=8, symbol="circle")
        ))
        fig.add_trace(go.Scatter(
            x=chart_data["FullName"], y=chart_data["PredictedFinish"],
            mode="lines+markers", name="Predicted Finish (Race)",
            line=dict(color="#f59e0b", width=2),
            marker=dict(size=8, symbol="diamond")
        ))
        fig.update_layout(
            title=f"Grid vs. Finish Forecast — {selected_gp}",
            xaxis_title="Driver",
            yaxis_title="Position",
            yaxis=dict(autorange="reversed"),
            template="plotly_dark",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=500,
        )
        fig.update_xaxes(tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

        # ── OUTPUT C: Win probability bar chart (top 10) ─────────────────────
        st.subheader("🏆 Win Probability — Top 10 Drivers")
        top10 = feats.head(10).sort_values("WinProbability")
        fig_bar = px.bar(
            top10, x="WinProbability", y="FullName", orientation="h",
            color="WinProbability", color_continuous_scale="Viridis",
            labels={"WinProbability": "Win Probability", "FullName": "Driver"},
            title=f"Top-10 Win Probability Distribution — {selected_gp}",
            template="plotly_dark",
        )
        fig_bar.update_layout(coloraxis_showscale=False, height=400)
        fig_bar.update_xaxes(tickformat=".1%")
        st.plotly_chart(fig_bar, use_container_width=True)

        st.info(
            f"💡 **How the AI predicts**: "
            f"Each driver's historical average qualifying position drives the grid prediction. "
            f"Their circuit-specific finishing average (at {selected_gp.split()[0]} races) "
            f"adjusts the race outcome. The model was trained on 5 years of F1 data."
        )

    except Exception as e:
        st.error(f"Prediction error: {e}")
        import traceback
        st.code(traceback.format_exc())


# ── Section 3: Track Historical Winners ──────────────────────────────────────
st.markdown("---")
st.header("📍 Track Historical Winners")

if not all_bronze.empty and "selected_gp" in locals():
    keyword = selected_gp.split()[0].lower()
    track_hist = all_bronze[all_bronze["Circuit"].str.lower().str.contains(keyword, na=False)]
    if not track_hist.empty:
        winners = (
            track_hist[track_hist["Position"] == 1][["Year", "FullName", "TeamName", "Circuit"]]
            .sort_values("Year", ascending=False)
            .rename(columns={"FullName": "Winner", "TeamName": "Team"})
        )
        st.dataframe(winners, use_container_width=True, hide_index=True)
    else:
        st.info(f"No historical data found for '{selected_gp}'. Run ingestion for more seasons.")
else:
    st.info("Generate a prediction first to see historical context.")


# ── Section 4: Championship Standings ────────────────────────────────────────
st.markdown("---")
st.header("🏆 Championship Standings")

if not all_bronze.empty:
    available_years = sorted(all_bronze["Year"].unique(), reverse=True)
    sel_year = st.selectbox("Select Season:", available_years, index=0)

    year_data = all_bronze[all_bronze["Year"] == sel_year]

    col_d, col_c = st.columns(2)
    with col_d:
        st.subheader(f"👤 {sel_year} Drivers")
        d_stand = (
            year_data.groupby("FullName")["Points"]
            .sum().sort_values(ascending=False)
            .reset_index().rename(columns={"FullName": "Driver"})
        )
        d_stand.index = range(1, len(d_stand) + 1)
        st.dataframe(d_stand, use_container_width=True)

    with col_c:
        st.subheader(f"🏎️ {sel_year} Constructors")
        c_stand = (
            year_data.groupby("TeamName")["Points"]
            .sum().sort_values(ascending=False)
            .reset_index().rename(columns={"TeamName": "Constructor"})
        )
        c_stand.index = range(1, len(c_stand) + 1)
        st.dataframe(c_stand, use_container_width=True)
else:
    st.info("No data loaded yet.")


# ── Section 5: Season Performance Analysis ───────────────────────────────────
st.markdown("---")
st.header("📈 Season Performance Analysis")

if not all_bronze.empty:
    # Use the already-selected year if available, otherwise take most recent
    analysis_year = sel_year if "sel_year" in locals() else sorted(all_bronze["Year"].unique())[-1]
    year_df = all_bronze[all_bronze["Year"] == analysis_year].copy()

    if not year_df.empty:
        drivers_in_year = sorted(year_df["FullName"].dropna().unique())
        sel_driver = st.selectbox("Select a Driver:", drivers_in_year, key="perf_driver")

        driver_df = year_df[year_df["FullName"] == sel_driver].copy()

        # Ensure numeric
        for col in ["GridPosition", "Position"]:
            driver_df[col] = pd.to_numeric(driver_df[col], errors="coerce")

        driver_df = driver_df.dropna(subset=["GridPosition", "Position"])
        driver_df = driver_df.sort_values("Circuit")

        if not driver_df.empty:
            fig_perf = go.Figure()
            fig_perf.add_trace(go.Scatter(
                x=driver_df["Circuit"], y=driver_df["GridPosition"],
                mode="lines+markers", name="Starting Grid",
                line=dict(color="#3b82f6", width=2),
                marker=dict(size=8)
            ))
            fig_perf.add_trace(go.Scatter(
                x=driver_df["Circuit"], y=driver_df["Position"],
                mode="lines+markers", name="Race Finish",
                line=dict(color="#f59e0b", width=2, dash="dot"),
                marker=dict(size=8, symbol="diamond")
            ))
            fig_perf.update_layout(
                title=f"{sel_driver} — Grid vs. Finish ({analysis_year})",
                yaxis=dict(autorange="reversed", title="Position (1 = best)"),
                xaxis_title="Grand Prix",
                template="plotly_dark",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                height=450,
            )
            fig_perf.update_xaxes(tickangle=-45)
            st.plotly_chart(fig_perf, use_container_width=True)
            st.caption("💡 When the yellow line is ABOVE the blue line, the driver gained positions during the race.")
        else:
            st.info(f"No clean race data available for {sel_driver} in {analysis_year}.")
    else:
        st.info("No data for the selected season.")
else:
    st.info("No data loaded yet.")
