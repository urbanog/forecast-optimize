r"""Split-conformal recalibration of quantile forecasts.

Setting.  A quantile model :math:`\hat q_\tau` is fitted on a *training* window.
On a disjoint *calibration* window with :math:`n` points we compute the scores

.. math::  s_i = y_i - \hat q_\tau(x_i), \qquad i = 1, \dots, n,

and shift the quantile by an order statistic of the scores,

.. math::  \tilde q_\tau(x) = \hat q_\tau(x) + s_{(k)}, \qquad
           k = \lceil (n+1)\tau \rceil \ (\tau \ge 1/2), \quad
           k = \lfloor (n+1)\tau \rfloor \ (\tau < 1/2),

with :math:`s_{(0)} = -\infty` and :math:`s_{(n+1)} = +\infty`.

Guarantee.  If the calibration pairs and the test pair are *exchangeable*, the
rank of the test score among all :math:`n+1` scores is uniform (ties broken at
random), hence for every :math:`\tau \ge 1/2`

.. math::  \mathbb P\bigl(Y \le \tilde q_\tau(X)\bigr) \ge k/(n+1) \ge \tau,

and for every :math:`\tau < 1/2`

.. math::  \mathbb P\bigl(Y < \tilde q_\tau(X)\bigr) \le k/(n+1) \le \tau .

Consequently the interval :math:`[\tilde q_{\alpha/2}, \tilde q_{1-\alpha/2}]`
has marginal coverage at least :math:`1-\alpha` (union bound on the two tails).
The guarantee is *marginal* (averaged over X), not conditional on X.

Caveat.  Hourly load residuals are autocorrelated and the series is not
stationary, so exchangeability fails and the bound is only approximate here;
that is why the empirical coverage on held-out data is always reported.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


def conformal_rank(n: int, tau: float) -> int:
    """Order-statistic index k (1-based; 0 means -inf, n+1 means +inf)."""
    if not 0.0 < tau < 1.0:
        raise ValueError("tau must lie in (0, 1)")
    # tiny epsilon so that e.g. (n+1)*0.9 == 90.00000000001 is not rounded up to 91
    x = (n + 1) * tau
    return math.ceil(x - 1e-9) if tau >= 0.5 else math.floor(x + 1e-9)


@dataclass
class ConformalQuantileCalibrator:
    """Per-level additive conformal shifts for a vector of quantile levels."""

    levels: tuple[float, ...]
    shifts: np.ndarray | None = None
    n_cal: int = 0

    def fit(self, y_cal: np.ndarray, q_cal: np.ndarray) -> ConformalQuantileCalibrator:
        y_cal = np.asarray(y_cal, float)
        q_cal = np.asarray(q_cal, float)
        if q_cal.shape != (len(y_cal), len(self.levels)):
            raise ValueError("q_cal must have shape (n, len(levels))")
        n = len(y_cal)
        scores = np.sort(y_cal[:, None] - q_cal, axis=0)
        shifts = np.empty(len(self.levels))
        for j, tau in enumerate(self.levels):
            k = conformal_rank(n, tau)
            shifts[j] = -np.inf if k == 0 else (np.inf if k > n else scores[k - 1, j])
        self.shifts, self.n_cal = shifts, n
        return self

    def transform(self, q: np.ndarray) -> np.ndarray:
        if self.shifts is None:
            raise RuntimeError("calibrator is not fitted")
        return np.sort(np.asarray(q, float) + self.shifts[None, :], axis=1)


def coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    """Empirical coverage of the intervals [lo, hi]."""
    y = np.asarray(y)
    return float(np.mean((y >= lo) & (y <= hi)))


def interval_columns(levels: Sequence[float], alpha: float) -> tuple[int, int]:
    """Column indices of the levels alpha/2 and 1-alpha/2."""
    lv = np.round(np.asarray(levels, float), 10)
    lo = np.flatnonzero(np.isclose(lv, alpha / 2))
    hi = np.flatnonzero(np.isclose(lv, 1 - alpha / 2))
    if not len(lo) or not len(hi):
        raise ValueError(f"levels {tuple(levels)} do not contain {alpha / 2} and {1 - alpha / 2}")
    return int(lo[0]), int(hi[0])
