"""Forecast evaluation metrics."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def mae(y: np.ndarray, f: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(f))))


def rmse(y: np.ndarray, f: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(f)) ** 2)))


def mape(y: np.ndarray, f: np.ndarray) -> float:
    """Mean absolute percentage error in percent (load is bounded away from zero)."""
    y = np.asarray(y, float)
    return float(100.0 * np.mean(np.abs(y - np.asarray(f)) / np.abs(y)))


def pinball(y: np.ndarray, q: np.ndarray, tau: float) -> float:
    r"""Mean pinball (quantile) loss :math:`\rho_\tau(u) = u(\tau - 1\{u<0\})`, u = y - q.

    Its expectation is minimised by the tau-quantile of Y, which is why quantile
    regression trains on it.
    """
    u = np.asarray(y, float) - np.asarray(q, float)
    return float(np.mean(np.maximum(tau * u, (tau - 1.0) * u)))


def mean_pinball(y: np.ndarray, Q: np.ndarray, levels: Sequence[float]) -> float:
    """Pinball loss averaged over quantile levels (a discretised CRPS / 2)."""
    return float(np.mean([pinball(y, Q[:, j], a) for j, a in enumerate(levels)]))
