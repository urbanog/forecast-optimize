"""Pydantic request/response models of the REST API (they also define the OpenAPI schema)."""

from __future__ import annotations

import datetime as dt
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class QuantileValue(BaseModel):
    level: float = Field(..., gt=0, lt=1, description="Quantile level tau")
    value_mw: float


class HourlyForecast(BaseModel):
    ts: dt.datetime = Field(..., description="Start of the delivery hour (local standard time, CET)")
    mean_mw: float = Field(..., description="Point forecast (conditional mean)")
    quantiles: list[QuantileValue] = Field(..., description="Conformally calibrated quantiles")


class ForecastRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {"date": "2024-06-12", "persist": False}})

    date: dt.date = Field(..., description="Delivery day; the forecast is issued at 00:00 of this day")
    persist: bool = Field(False, description="Store the forecast in the database")


class ForecastResponse(BaseModel):
    model: str
    issued_at: dt.datetime
    levels: list[float]
    interval_90: tuple[float, float] = Field(..., description="Quantile levels of the 90% central interval")
    hours: list[HourlyForecast]
    run_id: int | None = Field(None, description="Database id when persisted")


class BatteryIn(BaseModel):
    energy_mwh: float = Field(100.0, ge=0)
    power_mw: float = Field(25.0, ge=0)
    eta_charge: float = Field(0.95, gt=0, le=1)
    eta_discharge: float = Field(0.95, gt=0, le=1)
    initial_frac: float = Field(0.5, ge=0, le=1)
    degradation_eur_per_mwh: float = Field(2.0, ge=0)


class MarketIn(BaseModel):
    lambda_up: float = Field(40.0, ge=0, description="Shortfall premium over the day-ahead price, EUR/MWh")
    lambda_down: float = Field(15.0, ge=0, description="Surplus discount under the day-ahead price, EUR/MWh")
    capacity_cost: float = Field(4.0, ge=0, description="EUR per MW of nominated capacity per day")
    exceed_penalty: float = Field(60.0, ge=0, description="EUR per MW of realised peak above the nomination")
    battery: BatteryIn = Field(default_factory=BatteryIn)

    @model_validator(mode="after")
    def _sane(self) -> MarketIn:
        if self.lambda_up + self.lambda_down == 0:
            raise ValueError("lambda_up + lambda_down must be positive")
        return self


class PlanRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={"example": {"date": "2024-06-12", "strategy": "stochastic", "n_scenarios": 50}}
    )

    date: dt.date
    strategy: Literal["deterministic", "quantile", "stochastic"] = "stochastic"
    n_scenarios: int = Field(50, ge=1, le=500, description="Scenarios for the stochastic strategy")
    seed: int = 0
    market: MarketIn = Field(default_factory=MarketIn)


class HourlyPlan(BaseModel):
    ts: dt.datetime
    price_eur_mwh: float
    buy_mwh: float
    charge_mw: float
    discharge_mw: float
    soc_end_mwh: float


class PlanResponse(BaseModel):
    strategy: str
    capacity_mw: float
    expected_cost_eur: float = Field(..., description="Objective value of the optimisation model")
    critical_ratio: float = Field(..., description="lambda_up / (lambda_up + lambda_down)")
    solver_status: str
    solve_seconds: float
    hours: list[HourlyPlan]


class Health(BaseModel):
    status: Literal["ok"]
    model: str
    cutoff: dt.date
    last_day: dt.date
