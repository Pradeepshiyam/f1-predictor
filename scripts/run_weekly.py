import argparse
from pathlib import Path

import pandas as pd
from pyspark.sql import SparkSession

from f1_predictor.config import load_config
from f1_predictor.features.feature_builder import build_driver_form_features
from f1_predictor.models.baseline_model import heuristic_predict
from f1_predictor.reporting.writer import write_prediction_markdown


def _fallback_rows() -> list[dict]:
    return [
        {"driver_number": 12, "full_name": "Kimi Antonelli", "team_name": "Mercedes", "position": 1, "driver_points": 72},
        {"driver_number": 63, "full_name": "George Russell", "team_name": "Mercedes", "position": 2, "driver_points": 63},
        {"driver_number": 16, "full_name": "Charles Leclerc", "team_name": "Ferrari", "position": 3, "driver_points": 49},
        {"driver_number": 44, "full_name": "Lewis Hamilton", "team_name": "Ferrari", "position": 4, "driver_points": 41},
        {"driver_number": 4, "full_name": "Lando Norris", "team_name": "McLaren", "position": 5, "driver_points": 25},
    ]


def run(config_path: str) -> Path:
    cfg = load_config(config_path)

    spark = (
        SparkSession.builder.appName("f1_predictor_local")
        .master("local[*]")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )

    rows = _fallback_rows()
    results_df = spark.createDataFrame(rows)

    features_df = build_driver_form_features(results_df)
    top_df = features_df.toPandas()

    prediction = heuristic_predict(top_df)

    output_dir = Path(cfg["paths"]["outputs"])
    output_file = output_dir / f"{cfg['season']}_{cfg['race_name']}_prediction.md"
    write_prediction_markdown(
        output_path=str(output_file),
        race_name=cfg["race_name"],
        season=int(cfg["season"]),
        pole_setter=prediction.pole_setter,
        race_winner=prediction.race_winner,
        winning_constructor=prediction.winning_constructor,
    )

    spark.stop()
    return output_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Run weekly F1 local prediction pipeline")
    parser.add_argument("--config", default="configs/project.yaml", help="Path to project config")
    args = parser.parse_args()

    output_file = run(args.config)
    print(f"Prediction generated: {output_file}")


if __name__ == "__main__":
    main()
