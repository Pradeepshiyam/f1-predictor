from pathlib import Path

import requests


def fetch_openf1_endpoint(endpoint: str, params: dict | None = None) -> list[dict]:
    base = "https://api.openf1.org/v1"
    url = f"{base}/{endpoint}"
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    return response.json()


def persist_json(records: list[dict], output_path: str) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    import json

    with out.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
