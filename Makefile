PY ?= python

.PHONY: install lint format test data experiment tables sql-bench serve all

install:
	$(PY) -m pip install -e ".[dev]"

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

format:
	$(PY) -m ruff format .
	$(PY) -m ruff check --fix .

test:
	$(PY) -m pytest

data:
	$(PY) -m forecast_optimize generate --out data/synthetic_load.csv

experiment:
	$(PY) -m forecast_optimize experiment

tables:
	$(PY) -m forecast_optimize tables

sql-bench:
	$(PY) scripts/sql_benchmark.py

serve:
	$(PY) -m forecast_optimize serve

all: lint test experiment sql-bench
