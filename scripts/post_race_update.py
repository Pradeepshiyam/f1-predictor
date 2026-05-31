"""
Post-Race Update Pipeline

One command. No arguments required.
Auto-detects the most recently completed race from the FastF1 schedule.

Steps:
  1. Detect current year (datetime.now().year)
  2. Detect last completed race (EventDate < today, highest RoundNumber)
  3. Ingest race result + qualifying via FastF1
  4. Run pandas ETL (Bronze -> Silver -> Gold)
  5. Retrain all models (race_winner, pole_position, top3)
  6. Score all locked predictions for this race
  7. Log model run to database
  8. Print summary report

Usage:
    python scripts/post_race_update.py
"""
from __future__ import annotations

import sys
import os
import time
from datetime import datetime, timezone

# Make src/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from f1_predictor.common.config import settings
from f1_predictor.common.database import ModelRunLog, get_db, init_db
from f1_predictor.common.logger import get_logger
from f1_predictor.models.inference import reload_models

log = get_logger(__name__)


def detect_last_race(year: int) -> dict:
    """Return the most recently completed race from FastF1 schedule."""
    import fastf1
    cache_dir = os.path.join("data", "cache")
    os.makedirs(cache_dir, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)

    schedule = fastf1.get_event_schedule(year, include_testing=False)
    now = datetime.now(timezone.utc)

    past = schedule[schedule["EventDate"].dt.tz_localize("UTC") < now]
    if past.empty:
        raise RuntimeError(f"No completed races found for {year}.")

    last = past.sort_values("RoundNumber").iloc[-1]
    return {
        "race_name":    last["OfficialEventName"],
        "round_number": int(last["RoundNumber"]),
        "year":         year,
    }


def ingest_race(year: int, round_number: int) -> None:
    """Pull race + qualifying results from FastF1 into Bronze."""
    import fastf1
    cache_dir = os.path.join("data", "cache")
    fastf1.Cache.enable_cache(cache_dir)

    bronze_path = str(settings.bronze_path)
    os.makedirs(bronze_path, exist_ok=True)

    for session_type, suffix in [("R", "results"), ("Q", "qualifying")]:
        try:
            session = fastf1.get_session(year, round_number, session_type)
            session.load(laps=False, telemetry=False, weather=False, messages=False)
            results = session.results
            if results is None or results.empty:
                log.warning(f"No {suffix} data for round {round_number}")
                continue

            event_name = session.event["OfficialEventName"].replace(" ", "_")
            fname = f"{year}_{event_name}_{suffix}.csv"
            out_path = os.path.join(bronze_path, fname)
            results.to_csv(out_path, index=False)
            log.info(f"Ingested: {out_path}")
        except Exception as exc:
            log.error(f"Failed to ingest {session_type} for round {round_number}: {exc}")


def run_etl() -> None:
    """Run the pandas ETL pipeline: Bronze → Silver → Gold."""
    import importlib.util, subprocess
    etl_script = os.path.join("src", "f1_predictor", "features", "pandas_processor.py")
    log.info("Running ETL pipeline…")
    result = subprocess.run(
        [sys.executable, etl_script],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        log.error(f"ETL failed:\n{result.stderr}")
        raise RuntimeError("ETL pipeline failed.")
    log.info("ETL complete.")


def run_training() -> dict:
    """Retrain all models. Returns metrics dict."""
    import subprocess
    train_script = os.path.join("src", "f1_predictor", "models", "train.py")
    log.info("Retraining models…")
    start = time.time()
    result = subprocess.run(
        [sys.executable, train_script],
        capture_output=True, text=True
    )
    elapsed = time.time() - start
    if result.returncode != 0:
        log.error(f"Training failed:\n{result.stderr}")
        raise RuntimeError("Model training failed.")
    log.info(f"Training complete in {elapsed:.1f}s.")
    return {"training_sec": round(elapsed, 2)}


def score_all_predictions(race_name: str, year: int) -> int:
    """
    Score all locked, unscored predictions for this race using
    actual results from the ingested Bronze data.
    Returns count of predictions scored.
    """
    import glob, pandas as pd
    from f1_predictor.common.database import Prediction, User, SessionLocal
    from f1_predictor.prediction.scorer import score as compute_score

    # Load actual race result from Bronze
    pattern = os.path.join(str(settings.bronze_path), f"{year}_*_results.csv")
    keyword = race_name.split()[0].lower()
    files = [f for f in glob.glob(pattern)
             if keyword in os.path.basename(f).lower()]

    if not files:
        log.warning(f"No results file found for {race_name} {year} — cannot auto-score.")
        return 0

    actual_df = pd.read_csv(files[0])
    actual_df["Position"] = pd.to_numeric(actual_df.get("Position", pd.Series()), errors="coerce")
    actual_df = actual_df.dropna(subset=["Position"]).sort_values("Position")
    actual_top10 = actual_df["FullName"].head(20).tolist()

    if not actual_top10:
        log.warning("Empty actual results — cannot score.")
        return 0

    count = 0
    with SessionLocal() as db:
        preds = (
            db.query(Prediction)
            .filter_by(race_name=race_name, year=year, is_locked=True)
            .filter(Prediction.scored_at.is_(None))
            .all()
        )
        for pred in preds:
            predicted_top10 = [r["driver"] for r in pred.get_top10()]
            scores = compute_score(predicted_top10, actual_top10)
            pred.set_actual_top10(actual_top10)
            pred.winner_correct = scores["winner_correct"]
            pred.podium_hits    = scores["podium_hits"]
            pred.top10_hits     = scores["top10_hits"]
            pred.position_mae   = scores["position_mae"]
            pred.spearman_rho   = scores["spearman_rho"]
            pred.scored_at      = datetime.utcnow()
            count += 1
        db.commit()

    log.info(f"Scored {count} predictions for {race_name} {year}.")
    return count


def log_model_run(race: dict, metrics: dict) -> None:
    """Write training run summary to model_run_log table."""
    import pandas as pd
    gold_path = settings.gold_path
    gold_rows = 0
    if gold_path.exists():
        try:
            gold_rows = len(pd.read_parquet(gold_path))
        except Exception:
            pass

    init_db()
    with get_db() as db:
        entry = ModelRunLog(
            run_at=datetime.utcnow(),
            triggered_by="post_race_update.py",
            race_name=race["race_name"],
            year=race["year"],
            gold_row_count=gold_rows,
            training_sec=metrics.get("training_sec"),
        )
        db.add(entry)
    log.info("Model run logged to database.")


def main() -> None:
    print("\n" + "=" * 55)
    print("  F1 Post-Race Update Pipeline")
    print("=" * 55)

    year = settings.current_year
    print(f"\n[1/7] Detected season: {year}")

    race = detect_last_race(year)
    print(f"[2/7] Last completed race: {race['race_name']} "
          f"(Round {race['round_number']})")

    print(f"[3/7] Ingesting from FastF1…")
    ingest_race(race["year"], race["round_number"])

    print(f"[4/7] Running ETL pipeline…")
    run_etl()

    print(f"[5/7] Retraining models…")
    metrics = run_training()

    print(f"[6/7] Scoring locked predictions…")
    scored = score_all_predictions(race["race_name"], race["year"])
    print(f"      Scored {scored} prediction(s).")

    print(f"[7/7] Logging model run…")
    log_model_run(race, metrics)

    reload_models()

    print("\n" + "=" * 55)
    print(f"  Update complete for: {race['race_name']} {race['year']}")
    print(f"  Training time: {metrics.get('training_sec', '?')}s")
    print(f"  Predictions scored: {scored}")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
