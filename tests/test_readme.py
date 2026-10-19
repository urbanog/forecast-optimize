"""The README result tables must be exactly what the experiment produced (results/metrics.json)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from forecast_optimize.experiment import markdown_tables

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not (ROOT / "results" / "metrics.json").exists(), reason="run `make experiment` first")
def test_readme_tables_match_results() -> None:
    tables = markdown_tables(json.loads((ROOT / "results" / "metrics.json").read_text()))
    readme = (ROOT / "README.md").read_text()
    for line in tables.splitlines():
        if line.strip():
            assert line in readme, f"README is out of date: {line!r}"
