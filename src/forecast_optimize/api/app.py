"""FastAPI application: ``GET /health``, ``POST /forecast``, ``POST /plan``.

Run with ``python -m forecast_optimize serve`` and open http://127.0.0.1:8000/docs
for the interactive OpenAPI documentation.

The app is built by :func:`create_app` around a :class:`ForecastService`, so the
tests can inject a small, fast service.  When started from the CLI the service is
trained on synthetic data with model cutoff ``FO_CUTOFF`` (default 2024-07-01).
"""

from __future__ import annotations

import os

import pandas as pd
from fastapi import FastAPI, HTTPException

from forecast_optimize.api.schemas import (
    ForecastRequest,
    ForecastResponse,
    Health,
    HourlyForecast,
    HourlyPlan,
    PlanRequest,
    PlanResponse,
    QuantileValue,
)
from forecast_optimize.optimize.dayahead import Battery, MarketParams
from forecast_optimize.pipeline import ForecastService
from forecast_optimize.storage.db import Database


def create_app(service: ForecastService, db: Database | None = None) -> FastAPI:
    app = FastAPI(
        title="forecast-optimize",
        version="0.1.0",
        description="Probabilistic day-ahead electricity load forecasts and the procurement / battery plans derived from them.",
    )

    @app.get("/health", response_model=Health, tags=["meta"])
    def health() -> Health:
        return Health(
            status="ok", model=service.model_name, cutoff=service.cutoff.date(), last_day=service.last_day.date()
        )

    @app.post("/forecast", response_model=ForecastResponse, tags=["forecast"])
    def forecast(req: ForecastRequest) -> ForecastResponse:
        try:
            idx, point, Q = service.forecast(pd.Timestamp(req.date))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        run_id = None
        if req.persist:
            if db is None:
                raise HTTPException(status_code=503, detail="no database configured")
            run_id = db.save_forecast(service.model_name, idx[0], idx, point, Q, service.levels)
        hours = [
            HourlyForecast(
                ts=t.to_pydatetime(),
                mean_mw=float(point[i]),
                quantiles=[QuantileValue(level=a, value_mw=float(Q[i, j])) for j, a in enumerate(service.levels)],
            )
            for i, t in enumerate(idx)
        ]
        return ForecastResponse(
            model=service.model_name,
            issued_at=idx[0].to_pydatetime(),
            levels=list(service.levels),
            interval_90=(0.05, 0.95),
            hours=hours,
            run_id=run_id,
        )

    @app.post("/plan", response_model=PlanResponse, tags=["optimise"])
    def plan(req: PlanRequest) -> PlanResponse:
        mk = req.market
        params = MarketParams(
            lambda_up=mk.lambda_up,
            lambda_down=mk.lambda_down,
            capacity_cost=mk.capacity_cost,
            exceed_penalty=mk.exceed_penalty,
            battery=Battery(**mk.battery.model_dump()),
        )
        try:
            idx, pl, prices = service.plan(pd.Timestamp(req.date), req.strategy, params, req.n_scenarios, req.seed)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        hours = [
            HourlyPlan(
                ts=t.to_pydatetime(),
                price_eur_mwh=float(prices[i]),
                buy_mwh=float(pl.buy[i]),
                charge_mw=float(pl.charge[i]),
                discharge_mw=float(pl.discharge[i]),
                soc_end_mwh=float(pl.soc[i + 1]),
            )
            for i, t in enumerate(idx)
        ]
        return PlanResponse(
            strategy=req.strategy,
            capacity_mw=pl.capacity,
            expected_cost_eur=pl.objective,
            critical_ratio=params.critical_ratio,
            solver_status=pl.status,
            solve_seconds=pl.solve_seconds,
            hours=hours,
        )

    return app


def default_app() -> FastAPI:  # pragma: no cover - exercised manually / by uvicorn
    from forecast_optimize.data.synthetic import generate
    from forecast_optimize.storage.db import connect

    data = generate()
    service = ForecastService(data, pd.Timestamp(os.environ.get("FO_CUTOFF", "2024-07-01")))
    db = connect()
    db.migrate()
    return create_app(service, db)
