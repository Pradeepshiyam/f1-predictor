"""
Pandas-based Feature Engineering Pipeline (Primary ETL)
Replaces PySpark as the daily runner for speed on Windows.
PySpark (spark_processor.py) is retained for large-scale/future use.

Run:  python src/f1_predictor/features/pandas_processor.py
"""
from __future__ import annotations

import glob
import logging
import os
from pathlib import Path

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Known driver name aliases (FastF1 changed Antonelli's name in 2026) ──────
DRIVER_ALIASES: dict[str, str] = {
    "Andrea Kimi Antonelli": "Kimi Antonelli",
    "Andrea Kimi ANTONELLI": "Kimi Antonelli",
    "ANTONELLI": "Kimi Antonelli",
    "Isack HADJAR": "Isack Hadjar",
    "Oliver BEARMAN": "Oliver Bearman",
    "Franco COLAPINTO": "Franco Colapinto",
    "Jack DOOHAN": "Jack Doohan",
    "Gabriel BORTOLETO": "Gabriel Bortoleto",
    "Liam LAWSON": "Liam Lawson",
    "Nico HÜLKENBERG": "Nico Hulkenberg",
    "Nico Hülkenberg": "Nico Hulkenberg",
}

F1_POINTS = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10,
             6: 8,  7: 6,  8: 4,  9: 2,  10: 1}


def normalize_name(name: str) -> str:
    """Return canonical driver name, resolving known aliases."""
    if pd.isna(name):
        return "Unknown"
    name = str(name).strip()
    return DRIVER_ALIASES.get(name, name)


def load_bronze(bronze_path: str) -> pd.DataFrame:
    """
    Load all *_results.csv files from bronze layer.
    Tags each row with Year, Circuit, and RaceOrder (for time-aware windows).
    """
    files = sorted([
        f for f in glob.glob(f"{bronze_path}/*_results.csv")
        if os.path.getsize(f) > 200
    ])
    if not files:
        raise FileNotFoundError(f"No valid results CSVs found in {bronze_path}")

    frames: list[pd.DataFrame] = []
    for f in files:
        fname = Path(f).stem                         # e.g. 2024_Miami_Grand_Prix_results
        parts = fname.replace("_results", "").split("_")
        year = int(parts[0])
        circuit = " ".join(parts[1:])                # e.g. "Miami Grand Prix"

        try:
            df = pd.read_csv(f)
        except Exception as e:
            log.warning(f"Skipping {f}: {e}")
            continue

        df["Year"]    = year
        df["Circuit"] = circuit
        df["Source"]  = Path(f).name
        frames.append(df)

    raw = pd.concat(frames, ignore_index=True)
    log.info(f"Loaded {len(raw):,} rows from {len(files)} result files")
    return raw


def load_qualifying(bronze_path: str) -> pd.DataFrame:
    """Load all *_qualifying.csv files from bronze layer."""
    files = sorted([
        f for f in glob.glob(f"{bronze_path}/*_qualifying.csv")
        if os.path.getsize(f) > 200
    ])
    if not files:
        log.warning("No qualifying CSVs found — quali features will use result GridPosition")
        return pd.DataFrame()

    frames = []
    for f in files:
        fname = Path(f).stem
        parts = fname.replace("_qualifying", "").split("_")
        year = int(parts[0])
        circuit = " ".join(parts[1:])
        try:
            df = pd.read_csv(f)
            df["Year"]    = year
            df["Circuit"] = circuit
            frames.append(df)
        except Exception as e:
            log.warning(f"Skipping {f}: {e}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def clean_silver(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Silver layer: clean, type-cast, normalise names.
    Drops rows where Position is non-numeric (e.g. DNF stored as 'R').
    """
    df = raw.copy()

    # Normalise driver names (fix Antonelli duplicate etc.)
    if "FullName" in df.columns:
        df["FullName"] = df["FullName"].apply(normalize_name)

    # Cast numeric columns
    for col in ["Position", "GridPosition", "Points"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ClassifiedPosition fallback
    if "ClassifiedPosition" in df.columns:
        df["ClassifiedPosition"] = pd.to_numeric(df["ClassifiedPosition"], errors="coerce")
    else:
        df["ClassifiedPosition"] = df["Position"]

    # Drop rows with no usable finish position
    df = df.dropna(subset=["Position", "FullName"])
    df["Position"] = df["Position"].astype(int)

    # Sort by Year then Circuit (alphabetical as race proxy — good enough for rolling)
    df = df.sort_values(["Year", "Circuit", "Position"]).reset_index(drop=True)

    # Race-level ordering: assign a RaceIndex for correct rolling windows
    race_keys = df[["Year", "Circuit"]].drop_duplicates().sort_values(["Year", "Circuit"])
    race_keys["RaceIndex"] = range(len(race_keys))
    df = df.merge(race_keys, on=["Year", "Circuit"], how="left")

    log.info(f"Silver: {len(df):,} rows, {df['FullName'].nunique()} unique drivers, "
             f"{df['Circuit'].nunique()} circuits")
    return df


def build_gold(silver: pd.DataFrame, qual_df: pd.DataFrame) -> pd.DataFrame:
    """
    Gold layer: build per-driver-per-race feature rows for model training.

    Features produced:
      GridPosition         – qualifying grid (from quali file or result column)
      AvgFinishLast3/5/10  – rolling avg finish (3, 5, 10 races)
      AvgGridLast5         – rolling avg qualifying position (proxy for speed)
      DNFRateLast10        – reliability signal
    """
    df = silver.copy()

    # ── Qualifying grid position ──────────────────────────────────────────────
    if not qual_df.empty and "GridPosition" in qual_df.columns and "FullName" in qual_df.columns:
        qual_clean = qual_df.copy()
        qual_clean["FullName"] = qual_clean["FullName"].apply(normalize_name)
        qual_clean["GridPosition"] = pd.to_numeric(qual_clean["GridPosition"], errors="coerce")
        quali_map = (
            qual_clean
            .dropna(subset=["GridPosition", "FullName"])
            .set_index(["Year", "Circuit", "FullName"])["GridPosition"]
            .to_dict()
        )
        df["GridPosition"] = df.apply(
            lambda r: quali_map.get((r["Year"], r["Circuit"], r["FullName"]),
                                    r.get("GridPosition", np.nan)), axis=1
        )
    df["GridPosition"] = pd.to_numeric(df.get("GridPosition", np.nan), errors="coerce").fillna(10)

    # ── Rolling driver features ───────────────────────────────────────────────
    df = df.sort_values(["FullName", "RaceIndex"]).copy()
    grp = df.groupby("FullName")
    df["AvgFinishLast3"]  = grp["Position"].transform(lambda s: s.shift(1).rolling(3,  min_periods=1).mean())
    df["AvgFinishLast5"]  = grp["Position"].transform(lambda s: s.shift(1).rolling(5,  min_periods=1).mean())
    df["AvgFinishLast10"] = grp["Position"].transform(lambda s: s.shift(1).rolling(10, min_periods=1).mean())
    df["AvgGridLast5"]    = grp["GridPosition"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    df["DNFRateLast10"]   = grp["Position"].transform(
        lambda s: (s.shift(1) > 20).astype(float).rolling(10, min_periods=1).mean()
    )

    # ── Team points rolling ───────────────────────────────────────────────────
    if "TeamName" in df.columns and "Points" in df.columns:
        team_pts = (
            df.groupby(["TeamName", "RaceIndex"])["Points"].sum()
            .reset_index().sort_values(["TeamName", "RaceIndex"])
        )
        team_pts["TeamPointsLast5"] = (
            team_pts.groupby("TeamName")["Points"]
            .transform(lambda s: s.shift(1).rolling(5, min_periods=1).sum())
        )
        df = df.merge(team_pts[["TeamName", "RaceIndex", "TeamPointsLast5"]],
                      on=["TeamName", "RaceIndex"], how="left")
        df["TeamForm"] = df["TeamPointsLast5"].fillna(0)
    else:
        df["TeamPointsLast5"] = 0.0
        df["TeamForm"]        = 0.0

    # ── Circuit-specific features (fast cumulative approach) ──────────────────
    df = df.sort_values(["FullName", "Circuit", "RaceIndex"]).copy()
    grp2 = df.groupby(["FullName", "Circuit"])
    df["_pos_cumsum"]   = grp2["Position"].transform(lambda s: s.shift(1).expanding().sum())
    df["_pos_cumcount"] = grp2["Position"].transform(lambda s: s.shift(1).expanding().count())
    df["_win_cumsum"]   = grp2["Position"].transform(lambda s: (s.shift(1) == 1).astype(float).expanding().sum())
    df["CircuitAvgFinish"] = (df["_pos_cumsum"] / df["_pos_cumcount"]).fillna(df["AvgFinishLast5"])
    df["CircuitWins"]      = df["_win_cumsum"].fillna(0)
    df.drop(columns=["_pos_cumsum", "_pos_cumcount", "_win_cumsum"], inplace=True)
    df = df.sort_values(["FullName", "RaceIndex"]).copy()

    # ── Championship position ─────────────────────────────────────────────────
    df = df.sort_values(["RaceIndex", "FullName"]).copy()
    df["_pts_shifted"] = df.groupby("FullName")["Points"].transform(lambda s: s.shift(1).fillna(0))
    df["CumPoints"]    = df.groupby("FullName")["_pts_shifted"].transform("cumsum")
    df["ChampPos"]     = df.groupby("RaceIndex")["CumPoints"].rank(ascending=False, method="min")
    df.drop(columns=["_pts_shifted"], inplace=True)

    # ── Targets ───────────────────────────────────────────────────────────────
    df["ClassifiedPosition"] = df["Position"]
    df["IsWinner"] = (df["Position"] == 1).astype(int)
    df["IsTop3"]   = (df["Position"] <= 3).astype(int)
    df["IsTop10"]  = (df["Position"] <= 10).astype(int)

    # ── Cast to float64 for clean Parquet schema ──────────────────────────────
    all_num = [
        "Position", "GridPosition", "Points", "RaceIndex", "CumPoints",
        "AvgFinishLast3", "AvgFinishLast5", "AvgFinishLast10",
        "AvgGridLast5", "DNFRateLast10", "TeamPointsLast5", "TeamForm",
        "CircuitAvgFinish", "CircuitWins", "ChampPos",
        "ClassifiedPosition", "IsWinner", "IsTop3", "IsTop10",
    ]
    for c in all_num:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")

    log.info(f"Gold: {len(df):,} rows with {len(df.columns)} columns")
    return df


def run(bronze_path: str = "data/bronze/historical",
        silver_path: str = "data/silver/historical.parquet",
        gold_path:   str = "data/gold/features.parquet") -> None:
    """End-to-end Pandas ETL pipeline: Bronze → Silver → Gold."""
    os.makedirs(Path(silver_path).parent, exist_ok=True)
    os.makedirs(Path(gold_path).parent,   exist_ok=True)

    log.info("=== Pandas ETL Pipeline Starting ===")

    raw    = load_bronze(bronze_path)
    silver = clean_silver(raw)
    silver.to_parquet(silver_path, index=False)
    log.info(f"Silver saved → {silver_path}")

    qual_df = load_qualifying(bronze_path)
    gold    = build_gold(silver, qual_df)
    gold_tmp = gold_path + ".tmp"
    if os.path.exists(gold_tmp):
        os.remove(gold_tmp)
    gold.to_parquet(gold_tmp, index=False)
    if os.path.exists(gold_path):
        os.remove(gold_path)
    os.rename(gold_tmp, gold_path)
    log.info(f"Gold saved -> {gold_path}")

    log.info("=== ETL Complete ===")
    print(f"\nFeatures ready: {gold_path}")
    print(f"   Rows: {len(gold):,} | Drivers: {gold['FullName'].nunique()} | Circuits: {gold['Circuit'].nunique()}")


if __name__ == "__main__":
    run()
