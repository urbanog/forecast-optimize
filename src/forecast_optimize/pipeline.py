"""Predict-then-optimise glue: a forecasting service and the decision back-test.

Strategies compared for every test day (all share the same battery and market):

``deterministic``  solve the MILP with one scenario = conditional-mean forecast,
``quantile``       one scenario = the hourly critical-ratio quantile q_{tau*},
``stochastic``     sample-average approximation with S joint scenarios drawn from
                   the conformal marginals + Gaussian copula,
``oracle``         one scenario = the realised load (perfect foresight).

Realised costs are computed with the closed-form recourse of
:func:`forecast_optimize.optimize.dayahead.realized_cost` on the actual load.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from forecast_optimize.features import TARGET, make_features
from forecast_optimize.models.conformal import ConformalQuantileCalibrator
from forecast_optimize.models.copula import GaussianCopula, _z_grid, z_to_quantile
from forecast_optimize.models.cv import FoldResult
from forecast_optimize.models.forecasters import DEFAULT_LEVELS, make_forecaster
from forecast_optimize.optimize.dayahead import MarketParams, Plan, realized_cost, solve_day

STRATEGIES = ("deterministic", "quantile", "stochastic", "oracle")


def quantile_path(Q_day: np.ndarray, levels: Sequence[float], tau: float) -> np.ndarray:
    """Hourly tau-quantile, interpolating the calibrated quantile curves in normal-score space."""
    from scipy.special import ndtri

    z = np.full((1, Q_day.shape[0]), float(ndtri(tau)))
    return z_to_quantile(Q_day, _z_grid(levels), z)[0]


def plan_day(
    strategy: str,
    prices: np.ndarray,
    point: np.ndarray,
    Q_day: np.ndarray,
    levels: Sequence[float],
    copula: GaussianCopula | None,
    params: MarketParams,
    n_scenarios: int = 50,
    rng: np.random.Generator | None = None,
    actual: np.ndarray | None = None,
) -> Plan:
    if strategy == "deterministic":
        scen = point[None, :]
    elif strategy == "quantile":
        scen = quantile_path(Q_day, levels, params.critical_ratio)[None, :]
    elif strategy == "stochastic":
        if copula is None:
            raise ValueError("stochastic strategy needs a fitted copula")
        scen = copula.sample(Q_day, n_scenarios, rng or np.random.default_rng(0))
    elif strategy == "oracle":
        if actual is None:
            raise ValueError("oracle needs the realised load")
        scen = actual[None, :]
    else:
        raise ValueError(f"unknown strategy {strategy!r}; choose from {STRATEGIES}")
    return solve_day(prices, scen, params)


def decision_backtest(
    data: pd.DataFrame,
    folds: Sequence[FoldResult],
    params: MarketParams | None = None,
    n_scenarios: int = 50,
    n_eval: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """Plan every test day with each strategy and record realised and in-model costs.

    In-model columns (``eev_*``) evaluate each plan on ``n_eval`` *fresh* scenarios
    from the same predictive distribution, so ``eev_deterministic - eev_stochastic``
    is an (out-of-sample) estimate of the textbook value of the stochastic solution.
    """
    p = params or MarketParams()
    rng = np.random.default_rng(seed)
    rows = []
    for fr in folds:
        days = fr.index.normalize()
        for day in days.unique():
            m = np.asarray(days == day)
            if m.sum() != 24:
                continue
            prices = data.loc[fr.index[m], "da_price"].to_numpy()
            y, point, Q = fr.y[m], fr.point[m], fr.q_cal[m]
            eval_scen = fr.copula.sample(Q, n_eval, rng) if fr.copula is not None else None
            row: dict[str, float | pd.Timestamp] = {"day": day}
            for s in STRATEGIES:
                plan = plan_day(s, prices, point, Q, fr.levels, fr.copula, p, n_scenarios, rng, actual=y)
                row[f"cost_{s}"] = float(realized_cost(plan, prices, y, p)["total"][0])
                row[f"cap_{s}"] = plan.capacity
                row[f"solve_s_{s}"] = plan.solve_seconds
                row[f"mip_{s}"] = float(plan.used_mip)
                if eval_scen is not None and s != "oracle":
                    row[f"eev_{s}"] = float(realized_cost(plan, prices, eval_scen, p)["total"].mean())
            row["peak_actual"] = float(y.max())
            rows.append(row)
    return pd.DataFrame(rows).set_index("day")


@dataclass
class ForecastService:
    """A trained, conformally calibrated model that serves forecasts and plans for dates >= cutoff.

    The model is fitted on ``[start, cutoff - cal_days)`` and calibrated on
    ``[cutoff - cal_days, cutoff)``; requests for days before ``cutoff`` are rejected
    because they would be in-sample.
    """

    data: pd.DataFrame
    cutoff: pd.Timestamp
    model_name: str = "lightgbm"
    cal_days: int = 112
    levels: tuple[float, ...] = DEFAULT_LEVELS
    model_kwargs: dict | None = None

    def __post_init__(self) -> None:
        self.cutoff = pd.Timestamp(self.cutoff).normalize()
        self.feats = make_features(self.data)
        idx = self.data.index
        train_start = idx[0] + pd.Timedelta(days=15)
        cal_start = self.cutoff - pd.Timedelta(days=self.cal_days)
        train = idx[(idx >= train_start) & (idx < cal_start)]
        cal = idx[(idx >= cal_start) & (idx < self.cutoff)]
        if len(train) < 24 * 60:
            raise ValueError("need at least 60 days of training data before the calibration window")
        self.model = make_forecaster(self.model_name, **(self.model_kwargs or {}))
        self.model.fit(self.data.loc[train], self.feats.loc[train])
        _, q_cal_raw = self.model.predict(self.data.loc[cal], self.feats.loc[cal], self.levels)
        y_cal = self.data.loc[cal, TARGET].to_numpy()
        self.calibrator = ConformalQuantileCalibrator(self.levels).fit(y_cal, q_cal_raw)
        self.copula = GaussianCopula(self.levels).fit(y_cal, self.calibrator.transform(q_cal_raw), cal)
        self.lock = threading.Lock()

    @property
    def last_day(self) -> pd.Timestamp:
        return self.data.index[-1].normalize()

    def _day_index(self, day: pd.Timestamp | str) -> pd.DatetimeIndex:
        d = pd.Timestamp(day).normalize()
        if d < self.cutoff or d > self.last_day:
            raise ValueError(
                f"date must lie in [{self.cutoff.date()}, {self.last_day.date()}] (model cutoff .. last data day)"
            )
        return pd.date_range(d, periods=24, freq="h")

    def forecast(self, day: pd.Timestamp | str) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
        """Calibrated day-ahead forecast issued at 00:00 of ``day`` (uses data before ``day`` only)."""
        idx = self._day_index(day)
        with self.lock:
            point, q_raw = self.model.predict(self.data.loc[idx], self.feats.loc[idx], self.levels)
        return idx, point, self.calibrator.transform(q_raw)

    def plan(
        self, day: pd.Timestamp | str, strategy: str, params: MarketParams, n_scenarios: int = 50, seed: int = 0
    ) -> tuple[pd.DatetimeIndex, Plan, np.ndarray]:
        if strategy == "oracle":
            raise ValueError("the oracle uses realised load and is not available as a live plan")
        idx, point, Q = self.forecast(day)
        prices = self.data.loc[idx, "da_price"].to_numpy()
        plan = plan_day(
            strategy, prices, point, Q, self.levels, self.copula, params, n_scenarios, np.random.default_rng(seed)
        )
        return idx, plan, prices
