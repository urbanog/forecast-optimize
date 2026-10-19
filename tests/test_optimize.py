from __future__ import annotations

import numpy as np
import pytest

from forecast_optimize.optimize.dayahead import Battery, MarketParams, realized_cost, solve_day

NO_BATTERY = Battery(energy_mwh=0.0, power_mw=0.0, degradation_eur_per_mwh=0.0)


def _prices() -> np.ndarray:
    return 90.0 + 30.0 * np.sin(2 * np.pi * (np.arange(24) - 8) / 24)


def test_newsvendor_critical_quantile_energy() -> None:
    """Without battery and capacity terms, the SAA optimum x_t is the ceil(S tau*)-th order statistic of y_st."""
    rng = np.random.default_rng(0)
    S = 401
    Y = 800 + 120 * rng.gamma(3.0, 1.0, (S, 24))
    p = MarketParams(lambda_up=40.0, lambda_down=15.0, capacity_cost=0.0, exceed_penalty=0.0, battery=NO_BATTERY)
    tau = p.critical_ratio
    assert tau == pytest.approx(40 / 55)
    plan = solve_day(_prices(), Y, p)
    k = int(np.ceil(S * tau))  # S*tau = 291.6 is not an integer, so the minimiser is unique
    expected = np.sort(Y, axis=0)[k - 1]
    np.testing.assert_allclose(plan.buy, expected, rtol=1e-7)


@pytest.mark.parametrize(("lam_up", "lam_down"), [(40.0, 15.0), (10.0, 30.0), (20.0, 20.0)])
def test_newsvendor_gaussian_matches_formula(lam_up: float, lam_down: float) -> None:
    """With many Gaussian scenarios x* approaches mu + sigma * Phi^{-1}(lam_up / (lam_up + lam_down))."""
    from scipy.stats import norm

    rng = np.random.default_rng(1)
    mu, sd, S = 1000.0, 50.0, 4000
    Y = mu + sd * rng.standard_normal((S, 24))
    p = MarketParams(lambda_up=lam_up, lambda_down=lam_down, capacity_cost=0, exceed_penalty=0, battery=NO_BATTERY)
    plan = solve_day(_prices(), Y, p)
    target = mu + sd * norm.ppf(lam_up / (lam_up + lam_down))
    # sampling s.e. of a quantile ~ sd * sqrt(tau(1-tau)/S) / phi(z) < 1.5 MW here
    assert np.abs(plan.buy - target).max() < 6.0


def test_capacity_is_critical_quantile_of_daily_peak() -> None:
    rng = np.random.default_rng(2)
    S = 301
    Y = 900 + 60 * rng.standard_normal((S, 24))
    p = MarketParams(capacity_cost=4.0, exceed_penalty=60.0, battery=NO_BATTERY)
    plan = solve_day(_prices(), Y, p)
    peaks = np.sort(Y.max(axis=1))
    k = int(np.ceil(S * p.capacity_ratio))  # 301 * 14/15 = 280.93 -> 281st order statistic
    assert plan.capacity == pytest.approx(peaks[k - 1], rel=1e-7)


def test_objective_equals_closed_form_recourse() -> None:
    rng = np.random.default_rng(3)
    Y = 900 + 80 * rng.standard_normal((30, 24))
    plan = solve_day(_prices(), Y)
    cost = realized_cost(plan, _prices(), Y)["total"].mean()
    assert plan.objective == pytest.approx(cost, rel=1e-8)


def test_battery_physics_and_arbitrage() -> None:
    prices = np.where(np.arange(24) < 6, 20.0, 150.0)
    Y = np.full((1, 24), 800.0)
    b = Battery()
    plan = solve_day(prices, Y, MarketParams(battery=b))
    soc = plan.soc
    assert soc[0] == pytest.approx(b.initial_frac * b.energy_mwh) and soc[-1] == pytest.approx(soc[0])
    assert soc.min() >= -1e-7 and soc.max() <= b.energy_mwh + 1e-7
    np.testing.assert_allclose(
        soc[1:] - soc[:-1], b.eta_charge * plan.charge - plan.discharge / b.eta_discharge, atol=1e-6
    )
    assert np.all(np.minimum(plan.charge, plan.discharge) <= 1e-7)
    assert plan.charge[:6].sum() > 0 and plan.discharge[6:].sum() > 0  # buy cheap, use expensive
    # battery cannot make the day more expensive than doing nothing
    idle = solve_day(prices, Y, MarketParams(battery=NO_BATTERY))
    assert plan.objective <= idle.objective + 1e-6


def test_auto_mode_falls_back_to_milp_when_relaxation_cheats() -> None:
    """With negative prices the LP relaxation 'burns' energy by charging and discharging at once."""
    prices = np.full(24, -50.0)
    Y = np.full((1, 24), 500.0)
    relaxed = solve_day(prices, Y, mip=False)
    assert np.any(np.minimum(relaxed.charge, relaxed.discharge) > 1e-6)
    exact = solve_day(prices, Y)
    assert exact.used_mip and np.all(np.minimum(exact.charge, exact.discharge) <= 1e-6)
    assert exact.objective >= relaxed.objective - 1e-6  # relaxation is a lower bound


def test_auto_equals_milp_on_regular_day() -> None:
    rng = np.random.default_rng(5)
    Y = 900 + 50 * rng.standard_normal((20, 24))
    a, m = solve_day(_prices(), Y), solve_day(_prices(), Y, mip=True)
    assert not a.used_mip
    # the relaxation optimum is MILP-feasible, hence MILP-optimal; the MILP solve stops at a 1e-4 gap
    assert a.objective <= m.objective + 1e-6
    assert m.objective - a.objective <= 1e-4 * abs(m.objective)


def test_stochastic_beats_deterministic_in_sample() -> None:
    """The SAA solution is optimal for its own scenario set, so no other plan can be cheaper there."""
    rng = np.random.default_rng(6)
    Y = 900 + 60 * rng.standard_normal((40, 24))
    stoch = solve_day(_prices(), Y)
    det = solve_day(_prices(), Y.mean(axis=0, keepdims=True))
    c_s = realized_cost(stoch, _prices(), Y)["total"].mean()
    c_d = realized_cost(det, _prices(), Y)["total"].mean()
    assert c_s <= c_d + 1e-6
