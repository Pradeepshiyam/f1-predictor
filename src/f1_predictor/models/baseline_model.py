from dataclasses import dataclass

import pandas as pd


@dataclass
class BaselinePrediction:
    pole_setter: str
    race_winner: str
    winning_constructor: str


def heuristic_predict(standings: pd.DataFrame) -> BaselinePrediction:
    """Baseline rule: pick top driver and top constructor by points."""
    top_driver = standings.sort_values("driver_points", ascending=False).iloc[0]
    top_constructor = standings.groupby("team_name", as_index=False)["driver_points"].sum()
    top_constructor = top_constructor.sort_values("driver_points", ascending=False).iloc[0]

    return BaselinePrediction(
        pole_setter=top_driver["full_name"],
        race_winner=top_driver["full_name"],
        winning_constructor=top_constructor["team_name"],
    )
