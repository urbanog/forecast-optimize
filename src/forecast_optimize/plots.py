"""Figures for the README (matplotlib, static PNG).

Colours: a validated categorical palette (blue, orange; adjacent-pair CVD
separation checked), recessive grid, text in neutral inks.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from forecast_optimize.models.cv import FoldResult

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
BLUE_LIGHT = "#cde2fb"
BLUE_MID = "#86b6ef"


def _style(ax) -> None:  # type: ignore[no-untyped-def]
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def make_figures(
    data: pd.DataFrame, folds: Mapping[str, Sequence[FoldResult]], decisions: pd.DataFrame, fig_dir: Path
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": INK2})
    paths = []

    # 1. one test fortnight with the calibrated predictive bands -------------------
    fr = folds["lightgbm"][0]
    start = pd.Timestamp("2024-01-29")
    m = (fr.index >= start) & (fr.index < start + pd.Timedelta(days=14))
    idx, levels = fr.index[m], list(fr.levels)
    q = fr.q_cal[m]
    j05, j25, j75, j95 = (levels.index(a) for a in (0.05, 0.25, 0.75, 0.95))
    fig, ax = plt.subplots(figsize=(10, 4.3), facecolor=SURFACE)
    _style(ax)
    ax.fill_between(idx, q[:, j05], q[:, j95], color=BLUE_LIGHT, linewidth=0, label="90% interval (conformal)")
    ax.fill_between(idx, q[:, j25], q[:, j75], color=BLUE_MID, linewidth=0, label="50% interval (conformal)")
    ax.plot(idx, fr.point[m], color=BLUE, linewidth=1.6, label="point forecast (LightGBM)")
    ax.plot(idx, fr.y[m], color=INK, linewidth=1.2, label="realised load")
    feb8 = pd.Timestamp("2024-02-08")
    if idx[0] <= feb8 <= idx[-1]:
        ax.axvspan(feb8, feb8 + pd.Timedelta(days=2), color=GRID, alpha=0.5, linewidth=0, zorder=0)
        ax.text(
            feb8 + pd.Timedelta(days=1),
            1385,
            "Thu 8 Feb holiday + Fri bridge day:\nlargest errors (few such days to learn from)",
            ha="center",
            va="top",
            fontsize=8,
            color=INK2,
        )
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d %b"))
    ax.set_ylim(600, 1400)
    ax.set_ylabel("load [MW]")
    ax.set_title("Day-ahead probabilistic forecast, two test weeks (issued daily at 00:00)", loc="left", fontsize=11)
    ax.legend(loc="lower left", ncol=4, fontsize=8, frameon=False, bbox_to_anchor=(0, -0.24))
    ax.margins(x=0)
    fig.tight_layout()
    paths.append(fig_dir / "forecast_bands.png")
    fig.savefig(paths[-1], dpi=150, facecolor=SURFACE)
    plt.close(fig)

    # 2. reliability of the quantiles: empirical P(Y <= q_tau) vs tau -------------
    allf = folds["lightgbm"]
    y = np.concatenate([r.y for r in allf])
    q_raw = np.vstack([r.q_raw for r in allf])
    q_cal = np.vstack([r.q_cal for r in allf])
    lv = np.asarray(levels)
    emp_raw = (y[:, None] <= q_raw).mean(axis=0)
    emp_cal = (y[:, None] <= q_cal).mean(axis=0)
    fig, ax = plt.subplots(figsize=(4.8, 4.4), facecolor=SURFACE)
    _style(ax)
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle="--")
    ax.plot(lv, emp_raw, color=ORANGE, linewidth=2, marker="o", markersize=6, label="raw quantile regression")
    ax.plot(lv, emp_cal, color=BLUE, linewidth=2, marker="s", markersize=6, label="after split conformal")
    ax.set_xlabel(r"nominal level $\tau$")
    ax.set_ylabel(r"empirical $P(Y \leq \hat q_\tau)$ on test folds")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Quantile reliability, LightGBM, 2024", loc="left", fontsize=11)
    ax.legend(loc="upper left", fontsize=8.5, frameon=False)
    fig.tight_layout()
    paths.append(fig_dir / "quantile_reliability.png")
    fig.savefig(paths[-1], dpi=150, facecolor=SURFACE)
    plt.close(fig)

    # 3. regret of the planning strategies -----------------------------------------
    strategies = ["deterministic", "quantile", "stochastic"]
    regret = [float((decisions[f"cost_{s}"] - decisions["cost_oracle"]).mean()) for s in strategies]
    se = [
        float((decisions[f"cost_{s}"] - decisions["cost_oracle"]).std(ddof=1) / np.sqrt(len(decisions)))
        for s in strategies
    ]
    fig, ax = plt.subplots(figsize=(6.2, 3.4), facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="x", visible=False)
    labels = ["deterministic\n(point forecast)", "critical quantile\n(single path)", "stochastic\n(50 scenarios)"]
    bars = ax.bar(labels, regret, width=0.55, color=BLUE, yerr=se, ecolor=INK2, capsize=4, error_kw={"linewidth": 1})
    for b, r in zip(bars, regret, strict=True):
        ax.text(
            b.get_x() + b.get_width() / 2, r * 0.5, f"{r:,.0f} €", ha="center", va="center", color="white", fontsize=10
        )
    ax.set_ylabel("mean regret vs oracle [EUR/day]")
    ax.set_title(f"Cost of forecast uncertainty, {len(decisions)} test days (± 1 s.e.)", loc="left", fontsize=11)
    fig.tight_layout()
    paths.append(fig_dir / "decision_regret.png")
    fig.savefig(paths[-1], dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return paths
