"""Small end-to-end run: data -> rolling back-test -> conformal -> copula scenarios -> MILP -> realised costs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from forecast_optimize.models.cv import run_backtest
from forecast_optimize.pipeline import STRATEGIES, decision_backtest


def test_predict_then_optimise_end_to_end(small_data: pd.DataFrame, fast_lgbm: dict[str, object]) -> None:
    folds = run_backtest(
        small_data,
        ["lightgbm"],
        first_origin="2024-03-01",
        n_folds=1,
        test_days=6,
        cal_days=42,
        model_kwargs={"lightgbm": fast_lgbm},
    )["lightgbm"]
    dec = decision_backtest(small_data, folds, n_scenarios=20, n_eval=200)
    assert len(dec) == 6
    costs = dec[[f"cost_{s}" for s in STRATEGIES]]
    assert np.isfinite(costs.to_numpy()).all()
    # perfect foresight is a lower bound for every implementable plan, day by day
    for s in ("deterministic", "quantile", "stochastic"):
        assert (dec[f"cost_{s}"] >= dec["cost_oracle"] * (1 - 1e-6)).all()
