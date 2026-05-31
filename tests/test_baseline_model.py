from f1_predictor.models.baseline_model import heuristic_predict


def test_heuristic_predict_uses_top_points_driver():
    rows = [
        {"full_name": "A", "team_name": "T1", "driver_points": 10},
        {"full_name": "B", "team_name": "T2", "driver_points": 20},
    ]

    pred = heuristic_predict(__import__("pandas").DataFrame(rows))

    assert pred.pole_setter == "B"
    assert pred.race_winner == "B"
    assert pred.winning_constructor == "T2"
