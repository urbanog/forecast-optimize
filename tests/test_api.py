from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from forecast_optimize.api.app import create_app
from forecast_optimize.pipeline import ForecastService
from forecast_optimize.storage.db import connect


@pytest.fixture(scope="module")
def client(small_data: pd.DataFrame, fast_lgbm: dict[str, object]):
    service = ForecastService(small_data, pd.Timestamp("2024-03-01"), cal_days=42, model_kwargs=fast_lgbm)
    db = connect("sqlite:///:memory:")
    db.migrate()
    with TestClient(create_app(service, db)) as c:
        c.db = db  # type: ignore[attr-defined]
        yield c
    db.close()


def test_health_and_openapi(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"
    spec = client.get("/openapi.json").json()
    assert {"/forecast", "/plan", "/health"} <= set(spec["paths"])
    assert "PlanRequest" in spec["components"]["schemas"]


def test_forecast_endpoint(client: TestClient) -> None:
    r = client.post("/forecast", json={"date": "2024-03-12", "persist": True})
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["hours"]) == 24 and body["run_id"] is not None
    for h in body["hours"]:
        values = [q["value_mw"] for q in h["quantiles"]]
        assert values == sorted(values)
        assert 300 < h["mean_mw"] < 2000
    stored = client.db.latest_forecast("2024-03-12", "2024-03-13", 0.5)  # type: ignore[attr-defined]
    assert len(stored) == 24


def test_forecast_rejects_in_sample_dates(client: TestClient) -> None:
    r = client.post("/forecast", json={"date": "2023-06-01"})
    assert r.status_code == 422 and "date must lie" in r.json()["detail"]


def test_forecast_validates_payload(client: TestClient) -> None:
    assert client.post("/forecast", json={"date": "not-a-date"}).status_code == 422


@pytest.mark.parametrize("strategy", ["deterministic", "quantile", "stochastic"])
def test_plan_endpoint(client: TestClient, strategy: str) -> None:
    r = client.post("/plan", json={"date": "2024-03-12", "strategy": strategy, "n_scenarios": 20})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["solver_status"] == "Optimal"
    assert len(body["hours"]) == 24
    assert body["critical_ratio"] == pytest.approx(40 / 55)
    assert all(0 <= h["soc_end_mwh"] <= 100.0 + 1e-6 for h in body["hours"])


def test_plan_rejects_oracle_and_bad_market(client: TestClient) -> None:
    assert client.post("/plan", json={"date": "2024-03-12", "strategy": "oracle"}).status_code == 422
    bad = {"date": "2024-03-12", "market": {"lambda_up": -1}}
    assert client.post("/plan", json=bad).status_code == 422
