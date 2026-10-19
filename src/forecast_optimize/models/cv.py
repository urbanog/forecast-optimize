"""Rolling-origin (expanding window) evaluation with a split-conformal calibration block.

For every origin ``o`` (a midnight) the timeline is split into three disjoint,
time-ordered blocks::

    [ start ............ o - cal_days )[ o - cal_days ... o )[ o ... o + test_days )
                 train                        calibration              test

The forecaster is fitted on *train* only, the conformal shifts are estimated on
*calibration* only, and metrics are computed on *test* only.  Successive origins
move forward by ``test_days``, so the test blocks tile the evaluation period
without overlap and every test hour is forecast by a model that has seen only
the past.  Because features for day D use only information available at
00:00 of D (see :mod:`forecast_optimize.features`), no block boundary leaks
future information.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from forecast_optimize import metrics as M
from forecast_optimize.features import TARGET, make_features
from forecast_optimize.models.conformal import ConformalQuantileCalibrator, coverage, interval_columns
from forecast_optimize.models.copula import GaussianCopula
from forecast_optimize.models.forecasters import DEFAULT_LEVELS, Forecaster, make_forecaster


@dataclass(frozen=True)
class Fold:
    origin: pd.Timestamp
    train: pd.DatetimeIndex
    cal: pd.DatetimeIndex
    test: pd.DatetimeIndex


def rolling_origin_folds(
    index: pd.DatetimeIndex,
    first_origin: str | pd.Timestamp,
    n_folds: int,
    test_days: int,
    cal_days: int,
    min_train_start: str | pd.Timestamp | None = None,
) -> Iterator[Fold]:
    """Yield expanding-window folds; raises if a block would fall outside ``index``."""
    origin = pd.Timestamp(first_origin).normalize()
    start = pd.Timestamp(min_train_start) if min_train_start is not None else index[0]
    hour = pd.Timedelta(hours=1)
    for _ in range(n_folds):
        cal_start = origin - pd.Timedelta(days=cal_days)
        test_end = origin + pd.Timedelta(days=test_days)
        if cal_start <= start or test_end - hour > index[-1]:
            raise ValueError(f"fold with origin {origin.date()} does not fit in the data")
        yield Fold(
            origin=origin,
            train=index[(index >= start) & (index < cal_start)],
            cal=index[(index >= cal_start) & (index < origin)],
            test=index[(index >= origin) & (index < test_end)],
        )
        origin = test_end


@dataclass
class FoldResult:
    model: str
    origin: pd.Timestamp
    index: pd.DatetimeIndex
    y: np.ndarray
    point: np.ndarray
    q_raw: np.ndarray
    q_cal: np.ndarray
    levels: tuple[float, ...]
    copula: GaussianCopula | None = None
    extra: dict[str, float] = field(default_factory=dict)


def fit_calibrate_predict(
    model: Forecaster,
    data: pd.DataFrame,
    feats: pd.DataFrame,
    fold: Fold,
    levels: Sequence[float] = DEFAULT_LEVELS,
) -> FoldResult:
    """Fit on fold.train, conformalise + fit the copula on fold.cal, predict fold.test."""
    levels = tuple(levels)
    model.fit(data.loc[fold.train], feats.loc[fold.train])
    _, q_cal_raw = model.predict(data.loc[fold.cal], feats.loc[fold.cal], levels)
    y_cal = data.loc[fold.cal, TARGET].to_numpy()
    calib = ConformalQuantileCalibrator(levels).fit(y_cal, q_cal_raw)
    copula = GaussianCopula(levels).fit(y_cal, calib.transform(q_cal_raw), fold.cal)
    point, q_raw = model.predict(data.loc[fold.test], feats.loc[fold.test], levels)
    return FoldResult(
        model=model.name,
        origin=fold.origin,
        index=fold.test,
        y=data.loc[fold.test, TARGET].to_numpy(),
        point=point,
        q_raw=q_raw,
        q_cal=calib.transform(q_raw),
        levels=levels,
        copula=copula,
        extra={"n_cal": float(calib.n_cal)},
    )


def score(res: FoldResult, alphas: Sequence[float] = (0.1, 0.2)) -> dict[str, float]:
    """Point and probabilistic metrics for one fold (or a concatenation of folds)."""
    out = {
        "MAE": M.mae(res.y, res.point),
        "RMSE": M.rmse(res.y, res.point),
        "MAPE": M.mape(res.y, res.point),
        "pinball_raw": M.mean_pinball(res.y, res.q_raw, res.levels),
        "pinball_conformal": M.mean_pinball(res.y, res.q_cal, res.levels),
    }
    for a in alphas:
        lo, hi = interval_columns(res.levels, a)
        tag = f"{round(100 * (1 - a))}"
        out[f"cov{tag}_raw"] = coverage(res.y, res.q_raw[:, lo], res.q_raw[:, hi])
        out[f"cov{tag}_conformal"] = coverage(res.y, res.q_cal[:, lo], res.q_cal[:, hi])
        out[f"width{tag}_conformal"] = float(np.mean(res.q_cal[:, hi] - res.q_cal[:, lo]))
    return out


def concat(results: Sequence[FoldResult]) -> FoldResult:
    r0 = results[0]
    return FoldResult(
        model=r0.model,
        origin=r0.origin,
        index=pd.DatetimeIndex(np.concatenate([r.index.to_numpy() for r in results])),
        y=np.concatenate([r.y for r in results]),
        point=np.concatenate([r.point for r in results]),
        q_raw=np.vstack([r.q_raw for r in results]),
        q_cal=np.vstack([r.q_cal for r in results]),
        levels=r0.levels,
    )


def run_backtest(
    data: pd.DataFrame,
    model_names: Sequence[str],
    first_origin: str,
    n_folds: int,
    test_days: int,
    cal_days: int,
    levels: Sequence[float] = DEFAULT_LEVELS,
    model_kwargs: dict[str, dict[str, object]] | None = None,
    min_train_start: str | None = None,
) -> dict[str, list[FoldResult]]:
    """Run the rolling-origin evaluation for several models on identical folds."""
    feats = make_features(data)
    start = min_train_start or str((data.index[0] + pd.Timedelta(days=15)).date())  # lags need 14 days
    folds = list(rolling_origin_folds(data.index, first_origin, n_folds, test_days, cal_days, start))
    model_kwargs = model_kwargs or {}
    out: dict[str, list[FoldResult]] = {}
    for name in model_names:
        out[name] = [
            fit_calibrate_predict(make_forecaster(name, **model_kwargs.get(name, {})), data, feats, f, levels)
            for f in folds
        ]
    return out
