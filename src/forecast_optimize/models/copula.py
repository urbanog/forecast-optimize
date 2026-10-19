r"""From 24 marginal quantile forecasts to joint day-ahead scenarios.

The optimisation in :mod:`forecast_optimize.optimize` contains a *daily peak*
term, whose distribution depends on the dependence between the 24 hours of a
day, not just on the 24 marginals.  We therefore combine

* the calibrated marginal quantiles :math:`q_{t,\tau_1} < \dots < q_{t,\tau_m}`
  (inverse CDF reconstructed by linear interpolation in the normal-score
  variable :math:`z = \Phi^{-1}(\tau)`, linear extrapolation in the tails - exact
  if the marginal is Gaussian), with
* a Gaussian copula: :math:`Z = (Z_0,\dots,Z_{23}) \sim N(0, R)`, scenario
  :math:`y_t = F_t^{-1}(\Phi(Z_t))`.

:math:`R` is the correlation matrix of the normal scores
:math:`z_{d,h} = \Phi^{-1}(\hat F_{d,h}(y_{d,h}))` of the calibration days,
shrunk slightly towards the identity so that it is well conditioned.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import ndtr, ndtri


def _z_grid(levels: Sequence[float]) -> np.ndarray:
    return ndtri(np.asarray(levels, float))


def quantile_to_z(q_row: np.ndarray, zs: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Normal score z with F^{-1}(Phi(z)) = y for a piecewise-linear-in-z quantile curve (vectorised over y)."""
    lo_slope = (q_row[1] - q_row[0]) / (zs[1] - zs[0])
    hi_slope = (q_row[-1] - q_row[-2]) / (zs[-1] - zs[-2])
    z = np.interp(y, q_row, zs)
    below, above = y < q_row[0], y > q_row[-1]
    z[below] = zs[0] + (y[below] - q_row[0]) / max(lo_slope, 1e-9)
    z[above] = zs[-1] + (y[above] - q_row[-1]) / max(hi_slope, 1e-9)
    return z


def z_to_quantile(Q: np.ndarray, zs: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Evaluate the quantile curves of the rows of Q (T x m) at normal scores Z (S x T)."""
    S, T = Z.shape
    out = np.empty((S, T))
    for t in range(T):
        q = Q[t]
        lo_slope = (q[1] - q[0]) / (zs[1] - zs[0])
        hi_slope = (q[-1] - q[-2]) / (zs[-1] - zs[-2])
        z = Z[:, t]
        v = np.interp(z, zs, q)
        v = np.where(z < zs[0], q[0] + (z - zs[0]) * lo_slope, v)
        v = np.where(z > zs[-1], q[-1] + (z - zs[-1]) * hi_slope, v)
        out[:, t] = v
    return out


@dataclass
class GaussianCopula:
    levels: tuple[float, ...]
    corr: np.ndarray | None = None
    shrinkage: float = 0.05

    def fit(self, y: np.ndarray, Q: np.ndarray, index: pd.DatetimeIndex) -> GaussianCopula:
        """Estimate R from calibration observations y, calibrated quantiles Q (n x m) and their time stamps."""
        zs = _z_grid(self.levels)
        z = np.array([quantile_to_z(Q[i], zs, np.array([y[i]]))[0] for i in range(len(y))])
        z = np.clip(z, -4.0, 4.0)
        frame = pd.DataFrame({"z": z, "day": index.normalize(), "hour": index.hour})
        wide = frame.pivot(index="day", columns="hour", values="z").dropna()
        if wide.shape[1] != 24 or len(wide) < 10:
            raise ValueError("need at least 10 complete calibration days")
        R = np.corrcoef(wide.to_numpy(), rowvar=False)
        self.corr = (1 - self.shrinkage) * R + self.shrinkage * np.eye(24)
        return self

    def sample(self, Q_day: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw n joint scenarios (n x 24) for one day given its calibrated quantiles Q_day (24 x m)."""
        if self.corr is None:
            raise RuntimeError("copula not fitted")
        L = np.linalg.cholesky(self.corr)
        Z = rng.standard_normal((n, 24)) @ L.T
        return z_to_quantile(Q_day, _z_grid(self.levels), Z)

    def pit(self, y: np.ndarray, Q: np.ndarray) -> np.ndarray:
        """Probability integral transform u = F_hat(y) under the interpolated marginals."""
        zs = _z_grid(self.levels)
        return ndtr(np.array([quantile_to_z(Q[i], zs, np.array([y[i]]))[0] for i in range(len(y))]))
