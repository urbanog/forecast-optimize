"""The full reproducible experiment behind the README tables and figures.

``python -m forecast_optimize experiment`` runs

1. the synthetic data generator (2020-2024, seed 7),
2. a rolling-origin back-test over 2024 (6 folds x 61 days, 112-day calibration
   block before every origin) for the three forecasters,
3. the day-ahead decision back-test of the four planning strategies on the
   LightGBM forecasts,

and writes ``results/metrics.json``, ``results/decisions.csv`` and the figures in
``docs/figures``.  Runtime is about two minutes on a laptop CPU.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from forecast_optimize.data.synthetic import SyntheticConfig, generate
from forecast_optimize.models.cv import concat, run_backtest, score
from forecast_optimize.optimize.dayahead import MarketParams
from forecast_optimize.pipeline import STRATEGIES, decision_backtest

MODELS = ("seasonal_naive", "fourier_linear", "lightgbm")
BACKTEST = dict(first_origin="2024-01-01", n_folds=6, test_days=61, cal_days=112)
N_SCENARIOS = 50


def run(out_dir: str | Path = "results", fig_dir: str | Path | None = "docs/figures", seed: int = 7) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    data = generate(SyntheticConfig(seed=seed))
    timings["generate_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    folds = run_backtest(data, MODELS, **BACKTEST)  # type: ignore[arg-type]
    timings["backtest_s"] = time.perf_counter() - t0
    forecast_metrics = {m: score(concat(fr)) for m, fr in folds.items()}
    per_fold = {m: [{"origin": str(r.origin.date()), **score(r)} for r in fr] for m, fr in folds.items()}

    params = MarketParams()
    t0 = time.perf_counter()
    dec = decision_backtest(data, folds["lightgbm"], params, n_scenarios=N_SCENARIOS, seed=seed)
    timings["decisions_s"] = time.perf_counter() - t0
    dec.to_csv(out_dir / "decisions.csv")

    n_days = len(dec)
    cost = {s: float(dec[f"cost_{s}"].mean()) for s in STRATEGIES}
    decision = {
        "n_days": n_days,
        "mean_daily_cost_eur": cost,
        "mean_regret_vs_oracle_eur": {s: cost[s] - cost["oracle"] for s in STRATEGIES},
        "vss_realised_eur_per_day": cost["deterministic"] - cost["stochastic"],
        "vss_realised_se": float((dec["cost_deterministic"] - dec["cost_stochastic"]).std(ddof=1) / np.sqrt(n_days)),
        "quantile_minus_stochastic_eur_per_day": float((dec["cost_quantile"] - dec["cost_stochastic"]).mean()),
        "quantile_minus_stochastic_se": float(
            (dec["cost_quantile"] - dec["cost_stochastic"]).std(ddof=1) / np.sqrt(n_days)
        ),
        "vss_in_model_eur_per_day": float((dec["eev_deterministic"] - dec["eev_stochastic"]).mean()),
        "share_days_stochastic_beats_deterministic": float((dec["cost_stochastic"] < dec["cost_deterministic"]).mean()),
        "mean_capacity_mw": {s: float(dec[f"cap_{s}"].mean()) for s in STRATEGIES},
        "mean_actual_peak_mw": float(dec["peak_actual"].mean()),
        "mean_solve_seconds": {s: float(dec[f"solve_s_{s}"].mean()) for s in STRATEGIES},
        "milp_fallbacks": int(sum(dec[f"mip_{s}"].sum() for s in STRATEGIES)),
        "n_solves": int(len(STRATEGIES) * n_days),
        "critical_ratio_energy": params.critical_ratio,
        "critical_ratio_capacity": params.capacity_ratio,
        "n_scenarios": N_SCENARIOS,
    }

    result = {
        "config": {"seed": seed, **BACKTEST, "market": _market_dict(params)},
        "forecast": forecast_metrics,
        "forecast_per_fold": per_fold,
        "decision": decision,
        "timings": timings,
    }
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2))

    if fig_dir is not None:
        from forecast_optimize.plots import make_figures

        make_figures(data, folds, dec, Path(fig_dir))
    return result


def _market_dict(p: MarketParams) -> dict:
    b = p.battery
    return {
        "lambda_up": p.lambda_up,
        "lambda_down": p.lambda_down,
        "capacity_cost": p.capacity_cost,
        "exceed_penalty": p.exceed_penalty,
        "battery": {
            "energy_mwh": b.energy_mwh,
            "power_mw": b.power_mw,
            "eta_charge": b.eta_charge,
            "eta_discharge": b.eta_discharge,
            "degradation_eur_per_mwh": b.degradation_eur_per_mwh,
        },
    }


def markdown_tables(result: dict) -> str:
    """Render the README result tables from ``metrics.json`` (so numbers are never typed by hand)."""
    f = result["forecast"]
    lines = [
        "| model | MAE [MW] | RMSE [MW] | MAPE [%] | pinball raw | pinball conformal | 90% cov. raw | 90% cov. conformal | 90% width [MW] | 80% cov. conformal |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for m in MODELS:
        r = f[m]
        lines.append(
            f"| {m} | {r['MAE']:.1f} | {r['RMSE']:.1f} | {r['MAPE']:.2f} | {r['pinball_raw']:.2f} | "
            f"{r['pinball_conformal']:.2f} | {r['cov90_raw']:.3f} | {r['cov90_conformal']:.3f} | "
            f"{r['width90_conformal']:.1f} | {r['cov80_conformal']:.3f} |"
        )
    d = result["decision"]
    lines += [
        "",
        "| strategy | mean daily cost [EUR] | regret vs oracle [EUR/day] | mean capacity [MW] | mean solve time [s] |",
        "|---|---:|---:|---:|---:|",
    ]
    for s in STRATEGIES:
        lines.append(
            f"| {s} | {d['mean_daily_cost_eur'][s]:,.0f} | {d['mean_regret_vs_oracle_eur'][s]:,.0f} | "
            f"{d['mean_capacity_mw'][s]:.1f} | {d['mean_solve_seconds'][s]:.3f} |"
        )
    lines += [
        "",
        f"* days evaluated: {d['n_days']}; mean realised daily peak: {d['mean_actual_peak_mw']:.1f} MW; "
        f"solves needing the MILP fallback: {d['milp_fallbacks']} of {d['n_solves']}",
        f"* realised VSS (deterministic - stochastic): {d['vss_realised_eur_per_day']:,.0f} EUR/day "
        f"(standard error {d['vss_realised_se']:,.0f}); stochastic cheaper on "
        f"{100 * d['share_days_stochastic_beats_deterministic']:.0f}% of days",
        f"* critical-quantile path minus stochastic: {d['quantile_minus_stochastic_eur_per_day']:,.0f} EUR/day "
        f"(standard error {d['quantile_minus_stochastic_se']:,.0f})",
        f"* in-model VSS (EEV - RP on 1000 fresh scenarios/day): {d['vss_in_model_eur_per_day']:,.0f} EUR/day",
    ]
    return "\n".join(lines)


def describe(result: dict) -> pd.DataFrame:
    return pd.DataFrame(result["forecast"]).T
