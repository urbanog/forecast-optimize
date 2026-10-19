from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecast_optimize import metrics as M
from forecast_optimize.models.conformal import ConformalQuantileCalibrator, conformal_rank, coverage
from forecast_optimize.models.copula import GaussianCopula, _z_grid, quantile_to_z, z_to_quantile
from forecast_optimize.models.cv import concat, run_backtest, score
from forecast_optimize.models.forecasters import DEFAULT_LEVELS


def test_metrics_basic() -> None:
    y = np.array([100.0, 200.0])
    f = np.array([110.0, 180.0])
    assert M.mae(y, f) == 15.0
    assert M.rmse(y, f) == pytest.approx(np.sqrt((100 + 400) / 2))
    assert M.mape(y, f) == pytest.approx(100 * (0.1 + 0.1) / 2)
    # pinball: u = y - q; tau=0.9: u=-10 -> 1.0, u=20 -> 18.0
    assert M.pinball(y, f, 0.9) == pytest.approx((1.0 + 18.0) / 2)


def test_pinball_is_minimised_by_the_empirical_quantile() -> None:
    rng = np.random.default_rng(1)
    y = rng.gamma(2.0, 3.0, 2001)
    for tau in (0.1, 0.5, 0.73, 0.95):
        grid = np.linspace(y.min(), y.max(), 4001)
        losses = [M.pinball(y, np.full_like(y, g), tau) for g in grid]
        best = grid[int(np.argmin(losses))]
        lo, hi = np.quantile(y, tau, method="inverted_cdf"), np.quantile(y, tau, method="higher")
        step = grid[1] - grid[0]
        assert lo - step <= best <= hi + step


def test_conformal_rank() -> None:
    assert conformal_rank(99, 0.95) == 95
    assert conformal_rank(99, 0.05) == 5
    assert conformal_rank(9, 0.9) == 9  # (n+1)tau = 9 exactly: no spurious rounding up
    assert conformal_rank(5, 0.99) == 6  # > n means +inf
    assert conformal_rank(5, 0.01) == 0  # 0 means -inf


def test_split_conformal_marginal_coverage_under_exchangeability() -> None:
    """Monte Carlo check of P(L <= Y <= U) >= 1 - alpha with a deliberately biased, too-narrow base model."""
    rng = np.random.default_rng(2)
    n, reps, alpha = 49, 4000, 0.1
    levels = (alpha / 2, 1 - alpha / 2)
    hits = np.empty(reps)
    for r in range(reps):
        x = rng.uniform(0, 1, n + 1)
        y = 3 * x + rng.standard_t(4, n + 1) * (0.5 + x)  # heteroskedastic, heavy tailed
        q = np.column_stack([3 * x + 1.0 - 0.2, 3 * x + 1.0 + 0.2])  # biased and too narrow
        cal = ConformalQuantileCalibrator(levels).fit(y[:n], q[:n])
        qt = cal.transform(q[n:])
        hits[r] = coverage(y[n:], qt[:, 0], qt[:, 1])
    cov = hits.mean()
    se = np.sqrt(0.1 * 0.9 / reps)
    assert cov >= 1 - alpha - 3 * se
    # With continuous scores the coverage is exactly k_hi/(n+1) - k_lo/(n+1) = 48/50 - 2/50 = 0.92.
    k_hi, k_lo = conformal_rank(n, 1 - alpha / 2), conformal_rank(n, alpha / 2)
    assert cov == pytest.approx((k_hi - k_lo) / (n + 1), abs=3 * se)


def test_conformal_one_sided_quantile_validity() -> None:
    rng = np.random.default_rng(3)
    n, reps, tau = 39, 4000, 0.8
    hits = np.empty(reps)
    for r in range(reps):
        y = rng.normal(size=n + 1)
        q = np.full((n + 1, 1), -1.0)  # terrible base quantile
        cal = ConformalQuantileCalibrator((tau,)).fit(y[:n], q[:n])
        hits[r] = float(y[n] <= cal.transform(q[n:])[0, 0])
    k = conformal_rank(n, tau)
    # exact value for continuous scores: P(Y <= q~) = k/(n+1) = 32/40 = 0.8
    assert hits.mean() == pytest.approx(k / (n + 1), abs=3 * np.sqrt(0.16 / reps) + 1e-3)


def test_quantile_curve_roundtrip_and_gaussian_exactness() -> None:
    zs = _z_grid(DEFAULT_LEVELS)
    mu, sd = 1000.0, 40.0
    q = mu + sd * zs  # Gaussian quantiles
    y = np.array([850.0, 990.0, 1001.0, 1100.0, 1200.0])
    z = quantile_to_z(q, zs, y)
    np.testing.assert_allclose(z, (y - mu) / sd, atol=1e-9)  # exact for a Gaussian, also in the tails
    back = z_to_quantile(q[None, :], zs, z[:, None]).ravel()
    np.testing.assert_allclose(back, y, atol=1e-9)


def test_copula_samples_have_right_marginals_and_dependence() -> None:
    rng = np.random.default_rng(4)
    levels = DEFAULT_LEVELS
    zs = _z_grid(levels)
    # calibration days with AR(1)-correlated hourly normal scores
    days, rho = 120, 0.8
    Z = np.empty((days, 24))
    Z[:, 0] = rng.standard_normal(days)
    for h in range(1, 24):
        Z[:, h] = rho * Z[:, h - 1] + np.sqrt(1 - rho**2) * rng.standard_normal(days)
    idx = pd.date_range("2024-01-01", periods=days * 24, freq="h")
    Q = 500.0 + 20.0 * zs[None, :].repeat(days * 24, axis=0)
    y = 500.0 + 20.0 * Z.ravel()
    cop = GaussianCopula(levels).fit(y, Q, idx)
    assert cop.corr is not None and cop.corr[0, 1] == pytest.approx(0.95 * rho, abs=0.1)
    S = cop.sample(Q[:24], 4000, rng)
    assert S.mean() == pytest.approx(500.0, abs=1.0)
    assert S[:, 5].std() == pytest.approx(20.0, rel=0.05)


def test_backtest_runs_and_lightgbm_beats_naive(small_data: pd.DataFrame, fast_lgbm: dict[str, object]) -> None:
    res = run_backtest(
        small_data,
        ["seasonal_naive", "fourier_linear", "lightgbm"],
        first_origin="2024-02-01",
        n_folds=2,
        test_days=21,
        cal_days=56,
        model_kwargs={"lightgbm": fast_lgbm},
    )
    s = {m: score(concat(v)) for m, v in res.items()}
    assert s["lightgbm"]["MAE"] < s["seasonal_naive"]["MAE"]
    assert s["fourier_linear"]["MAE"] < s["seasonal_naive"]["MAE"]
    for m in s:
        assert 0.6 < s[m]["cov90_conformal"] <= 1.0
        q = np.vstack([r.q_cal for r in res[m]])
        assert np.all(np.diff(q, axis=1) >= 0)  # no quantile crossing
