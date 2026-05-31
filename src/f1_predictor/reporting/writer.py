from pathlib import Path


def write_prediction_markdown(
    output_path: str,
    race_name: str,
    season: int,
    pole_setter: str,
    race_winner: str,
    winning_constructor: str,
) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    content = f"""# {season} {race_name.title()} GP Prediction\n\n- Pole Setter: {pole_setter}\n- Race Winner: {race_winner}\n- Winning Constructor: {winning_constructor}\n"""
    out.write_text(content, encoding="utf-8")
