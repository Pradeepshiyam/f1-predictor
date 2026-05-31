.PHONY: setup run lint test

setup:
	python -m venv .venv
	.venv\Scripts\pip install -r requirements.txt
	.venv\Scripts\pip install -e .

run:
	.venv\Scripts\python scripts\run_weekly.py --config configs\project.yaml

lint:
	.venv\Scripts\ruff check src scripts tests
	.venv\Scripts\black --check src scripts tests

test:
	.venv\Scripts\pytest -q
