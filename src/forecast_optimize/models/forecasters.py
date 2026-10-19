"""Point and quantile forecasters sharing one small interface.

Every forecaster implements

* ``fit(frame, feats)`` on a training window,
* ``predict(frame, feats, levels)`` returning ``(point, Q)`` where ``Q`` has one
  column per quantile level (sorted, non-crossing).

``frame`` is the raw hourly data (needed by the seasonal naive model) and
``feats`` the output of :func:`forecast_optimize.features.make_features` on the
same index.  Quantiles of the two baselines come from empirical quantiles of the
in-sample residuals; LightGBM fits one model per level with the pinball loss.
In all cases the raw quantiles are afterwards recalibrated by split conformal
prediction (:mod:`forecast_optimize.models.conformal`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np
import pandas as pd

from forecast_optimize.features import TARGET, fourier_design

DEFAULT_LEVELS: tuple[float, ...] = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)


class Forecaster(ABC):
    name: str = "base"

    @abstractmethod
    def fit(self, frame: pd.DataFrame, feats: pd.DataFrame) -> Forecaster: ...

    @abstractmethod
    def predict(
        self, frame: pd.DataFrame, feats: pd.DataFrame, levels: Sequence[float] = DEFAULT_LEVELS
    ) -> tuple[np.ndarray, np.ndarray]: ...


class _ResidualQuantileMixin:
    """Quantiles = point forecast + empirical quantiles of in-sample residuals."""

    _resid: np.ndarray

    def _quantiles(self, point: np.ndarray, levels: Sequence[float]) -> np.ndarray:
        offs = np.quantile(self._resid, np.asarray(levels))
        return point[:, None] + offs[None, :]


class SeasonalNaive(_ResidualQuantileMixin, Forecaster):
    """y_hat(t) = y(t - 168h): same hour one week earlier."""

    name = "seasonal_naive"

    def fit(self, frame: pd.DataFrame, feats: pd.DataFrame) -> SeasonalNaive:
        r = (frame[TARGET] - feats["lag_168"]).dropna()
        self._resid = r.to_numpy()
        return self

    def predict(
        self, frame: pd.DataFrame, feats: pd.DataFrame, levels: Sequence[float] = DEFAULT_LEVELS
    ) -> tuple[np.ndarray, np.ndarray]:
        point = feats["lag_168"].to_numpy(float)
        return point, self._quantiles(point, levels)


class FourierLinear(_ResidualQuantileMixin, Forecaster):
    """Ordinary least squares on Fourier terms x day type + temperature hinges (see features.fourier_design)."""

    name = "fourier_linear"

    def __init__(self, ridge: float = 1e-6) -> None:
        self.ridge = ridge

    def _design(self, frame: pd.DataFrame, feats: pd.DataFrame) -> np.ndarray:
        X = fourier_design(frame.index, frame["temp_fc_c"], feats["prevday_mean"])
        return X.to_numpy(float)

    def fit(self, frame: pd.DataFrame, feats: pd.DataFrame) -> FourierLinear:
        X = self._design(frame, feats)
        y = frame[TARGET].to_numpy(float)
        ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X, y = X[ok], y[ok]
        self._mu = X.mean(axis=0)
        self._sd = X.std(axis=0)
        self._sd[self._sd == 0] = 1.0
        Z = (X - self._mu) / self._sd
        Z1 = np.column_stack([np.ones(len(Z)), Z])
        A = Z1.T @ Z1 + self.ridge * len(Z) * np.diag([0.0] + [1.0] * Z.shape[1])
        self._coef = np.linalg.solve(A, Z1.T @ y)
        self._resid = y - Z1 @ self._coef
        return self

    def _point(self, frame: pd.DataFrame, feats: pd.DataFrame) -> np.ndarray:
        Z = (self._design(frame, feats) - self._mu) / self._sd
        return np.column_stack([np.ones(len(Z)), Z]) @ self._coef

    def predict(
        self, frame: pd.DataFrame, feats: pd.DataFrame, levels: Sequence[float] = DEFAULT_LEVELS
    ) -> tuple[np.ndarray, np.ndarray]:
        point = self._point(frame, feats)
        return point, self._quantiles(point, levels)


class LightGBMQuantile(Forecaster):
    """Gradient-boosted trees: one L2 model for the mean, one pinball-loss model per quantile level."""

    name = "lightgbm"

    def __init__(
        self,
        levels: Sequence[float] = DEFAULT_LEVELS,
        n_estimators: int = 200,
        learning_rate: float = 0.1,
        num_leaves: int = 31,
        min_child_samples: int = 40,
        seed: int = 0,
        n_jobs: int = 1,
    ) -> None:
        self.levels = tuple(float(a) for a in levels)
        self.params = dict(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            min_child_samples=min_child_samples,
            colsample_bytree=0.9,
            random_state=seed,
            n_jobs=n_jobs,
            verbose=-1,
            force_col_wise=True,
        )
        self.feature_names: list[str] = []

    def fit(self, frame: pd.DataFrame, feats: pd.DataFrame) -> LightGBMQuantile:
        import lightgbm as lgb

        y = frame[TARGET]
        ok = feats.notna().all(axis=1) & y.notna()
        X, yy = feats.loc[ok], y.loc[ok]
        self.feature_names = list(X.columns)
        self._mean = lgb.LGBMRegressor(objective="regression", **self.params).fit(X, yy)
        self._q = {a: lgb.LGBMRegressor(objective="quantile", alpha=a, **self.params).fit(X, yy) for a in self.levels}
        return self

    def predict(
        self, frame: pd.DataFrame, feats: pd.DataFrame, levels: Sequence[float] = DEFAULT_LEVELS
    ) -> tuple[np.ndarray, np.ndarray]:
        missing = [a for a in levels if float(a) not in self._q]
        if missing:
            raise ValueError(f"levels {missing} were not fitted (fitted: {self.levels})")
        X = feats[self.feature_names]
        point = self._mean.predict(X)
        Q = np.column_stack([self._q[float(a)].predict(X) for a in levels])
        return point, np.sort(Q, axis=1)  # rearrangement removes rare quantile crossings


def make_forecaster(name: str, **kwargs: object) -> Forecaster:
    registry: dict[str, type[Forecaster]] = {
        SeasonalNaive.name: SeasonalNaive,
        FourierLinear.name: FourierLinear,
        LightGBMQuantile.name: LightGBMQuantile,
    }
    if name not in registry:
        raise KeyError(f"unknown forecaster {name!r}; choose from {sorted(registry)}")
    return registry[name](**kwargs)  # type: ignore[arg-type]
