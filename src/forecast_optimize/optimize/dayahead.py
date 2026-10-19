r"""Day-ahead procurement, capacity nomination and battery dispatch as a (MI)LP.

A supplier must, before day D starts, buy energy :math:`x_t` on the day-ahead
market at price :math:`\pi_t` (t = 0..23), nominate a capacity :math:`C` for the
day, and schedule a battery (charge :math:`c_t`, discharge :math:`d_t`, state of
charge :math:`e_t`).  After the day the realised load :math:`y_t` is known and

* every MWh of shortfall :math:`u_t` is bought at :math:`\pi_t + \lambda^+`,
* every MWh of surplus :math:`v_t` is sold back at :math:`\pi_t - \lambda^-`,
* the realised peak grid draw above :math:`C` is charged :math:`p^{ex}` per MW.

Two-stage stochastic program with scenarios :math:`y^s` and weights :math:`w_s`::

    min   sum_t pi_t x_t + k C + delta sum_t (c_t + d_t)
          + sum_s w_s [ sum_t ((pi_t + lam+) u_st - (pi_t - lam-) v_st) + p_ex g_s ]
    s.t.  x_t + d_t - c_t + u_st - v_st = y_st                 (energy balance)
          y_st - d_t + c_t - C <= g_s                          (peak exceedance)
          e_{t+1} = e_t + eta_c c_t - d_t / eta_d,  e_0 = e_24 = e_init
          0 <= e_t <= E,  0 <= c_t <= P z_t,  0 <= d_t <= P (1 - z_t),  z_t in {0,1}
          x, C, u, v, g >= 0

First-stage decisions: x, C, c, d, e, z.  Recourse: u, v, g.  The binary
:math:`z_t` forbids simultaneous charging and discharging.

``mip="auto"`` (default) first solves the LP relaxation (drop z).  The relaxation
is a lower bound for the MILP, so if its optimum already satisfies
:math:`\min(c_t, d_t) = 0` for every t it is feasible - hence optimal - for the
MILP and is returned; otherwise the MILP is solved.  ``mip=True`` always solves
the MILP, ``mip=False`` returns the relaxation.  With one scenario equal to a point forecast this is the
*deterministic* plan; with the realised load it is the *perfect-foresight oracle*.
The model is assembled as sparse matrices and solved with HiGHS (``highspy``).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import highspy
import numpy as np
import scipy.sparse as sp

INF = highspy.kHighsInf
H = 24


@dataclass(frozen=True)
class Battery:
    energy_mwh: float = 100.0
    power_mw: float = 25.0
    eta_charge: float = 0.95
    eta_discharge: float = 0.95
    initial_frac: float = 0.5
    degradation_eur_per_mwh: float = 2.0


@dataclass(frozen=True)
class MarketParams:
    lambda_up: float = 40.0  # EUR/MWh premium for buying a shortfall in real time
    lambda_down: float = 15.0  # EUR/MWh discount when selling a surplus in real time
    capacity_cost: float = 4.0  # EUR per MW of nominated capacity per day
    exceed_penalty: float = 60.0  # EUR per MW of realised peak above the nomination
    battery: Battery = field(default_factory=Battery)

    @property
    def critical_ratio(self) -> float:
        """Newsvendor critical quantile lambda+ / (lambda+ + lambda-) for the energy position."""
        return self.lambda_up / (self.lambda_up + self.lambda_down)

    @property
    def capacity_ratio(self) -> float:
        """Critical quantile 1 - k / p_ex of the daily peak for the capacity nomination (no battery)."""
        return 1.0 - self.capacity_cost / self.exceed_penalty


@dataclass
class Plan:
    buy: np.ndarray
    charge: np.ndarray
    discharge: np.ndarray
    soc: np.ndarray
    capacity: float
    objective: float
    status: str
    solve_seconds: float
    n_scenarios: int
    used_mip: bool = False


_LOCAL = threading.local()


def _highs() -> highspy.Highs:
    """One reusable, silent HiGHS instance per thread (constructing one costs ~5 ms)."""
    h = getattr(_LOCAL, "highs", None)
    if h is None:
        h = highspy.Highs()
        h.silent()
        _LOCAL.highs = h
    h.clearModel()
    return h


class _Builder:
    """Tiny helper that collects variables and sparse rows for a HighsLp."""

    def __init__(self) -> None:
        self.cost: list[np.ndarray] = []
        self.lb: list[np.ndarray] = []
        self.ub: list[np.ndarray] = []
        self.integer: list[np.ndarray] = []
        self.n = 0
        self.rows: list[int] = []
        self.cols: list[int] = []
        self.vals: list[float] = []
        self.rlo: list[float] = []
        self.rhi: list[float] = []

    def var(self, size: int, cost: float | np.ndarray = 0.0, lb: float = 0.0, ub: float = INF, integer: bool = False):
        idx = np.arange(self.n, self.n + size)
        self.n += size
        self.cost.append(np.broadcast_to(np.asarray(cost, float), (size,)).copy())
        self.lb.append(np.full(size, lb))
        self.ub.append(np.full(size, ub))
        self.integer.append(np.full(size, integer))
        return idx

    def rows_block(self, cols: np.ndarray, vals: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> None:
        """Add len(lo) rows; ``cols``/``vals`` have shape (n_rows, k)."""
        r0 = len(self.rlo)
        nr, k = cols.shape
        self.rows.extend(np.repeat(np.arange(r0, r0 + nr), k).tolist())
        self.cols.extend(cols.ravel().tolist())
        self.vals.extend(np.broadcast_to(vals, cols.shape).ravel().tolist())
        self.rlo.extend(np.broadcast_to(lo, (nr,)).tolist())
        self.rhi.extend(np.broadcast_to(hi, (nr,)).tolist())

    def solve(self, time_limit: float, mip_gap: float) -> tuple[highspy.Highs, np.ndarray]:
        lp = highspy.HighsLp()
        lp.num_col_ = self.n
        lp.num_row_ = len(self.rlo)
        lp.col_cost_ = np.concatenate(self.cost)
        lp.col_lower_ = np.concatenate(self.lb)
        lp.col_upper_ = np.concatenate(self.ub)
        lp.row_lower_ = np.asarray(self.rlo, float)
        lp.row_upper_ = np.asarray(self.rhi, float)
        A = sp.csc_matrix((self.vals, (self.rows, self.cols)), shape=(lp.num_row_, lp.num_col_))
        lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.start_ = A.indptr
        lp.a_matrix_.index_ = A.indices
        lp.a_matrix_.value_ = A.data
        integer = np.concatenate(self.integer)
        if integer.any():
            lp.integrality_ = [
                highspy.HighsVarType.kInteger if b else highspy.HighsVarType.kContinuous for b in integer
            ]
        h = _highs()
        h.setOptionValue("time_limit", float(time_limit))
        h.setOptionValue("mip_rel_gap", float(mip_gap))
        h.setOptionValue("threads", 1)
        h.passModel(lp)
        h.run()
        return h, np.asarray(h.getSolution().col_value)


def solve_day(
    prices: np.ndarray,
    scenarios: np.ndarray,
    params: MarketParams | None = None,
    weights: np.ndarray | None = None,
    mip: bool | str = "auto",
    time_limit: float = 30.0,
    mip_gap: float = 1e-4,
    complementarity_tol: float = 1e-7,
) -> Plan:
    """Solve the two-stage program for one day; ``scenarios`` has shape (S, 24).

    If the MILP hits ``time_limit`` with a feasible incumbent, that incumbent is
    returned with ``status == "Time limit reached"``; any other non-optimal status raises.
    """
    p = params or MarketParams()
    b = p.battery
    prices = np.asarray(prices, float)
    Y = np.atleast_2d(np.asarray(scenarios, float))
    S = Y.shape[0]
    if prices.shape != (H,) or Y.shape[1] != H:
        raise ValueError("expected 24 prices and scenarios of shape (S, 24)")
    w = np.full(S, 1.0 / S) if weights is None else np.asarray(weights, float) / np.sum(weights)

    def build_and_solve(with_binaries: bool) -> Plan:
        m = _Builder()
        x = m.var(H, prices)
        c = m.var(H, b.degradation_eur_per_mwh, ub=b.power_mw)
        d = m.var(H, b.degradation_eur_per_mwh, ub=b.power_mw)
        e = m.var(H + 1, 0.0, ub=b.energy_mwh)
        cap = m.var(1, p.capacity_cost)
        e_init = b.initial_frac * b.energy_mwh
        m.lb[3][[0, H]] = e_init
        m.ub[3][[0, H]] = e_init
        u = m.var(S * H, (w[:, None] * (prices + p.lambda_up)[None, :]).ravel())
        v = m.var(S * H, (-w[:, None] * (prices - p.lambda_down)[None, :]).ravel())
        g = m.var(S, w * p.exceed_penalty)
        uu, vv = u.reshape(S, H), v.reshape(S, H)
        xs, cs, ds = (np.broadcast_to(a, (S, H)) for a in (x, c, d))

        # energy balance: x + d - c + u - v = y
        cols = np.stack([xs, ds, cs, uu, vv], axis=-1).reshape(-1, 5)
        m.rows_block(cols, np.array([1.0, 1.0, -1.0, 1.0, -1.0]), Y.ravel(), Y.ravel())
        # peak: g_s + d_t - c_t + C >= y_st
        gs = np.broadcast_to(g[:, None], (S, H))
        caps = np.broadcast_to(cap, (S, H))
        cols = np.stack([gs, ds, cs, caps], axis=-1).reshape(-1, 4)
        m.rows_block(cols, np.array([1.0, 1.0, -1.0, 1.0]), Y.ravel(), np.full(S * H, INF))
        # battery dynamics: e_{t+1} - e_t - eta_c c_t + d_t / eta_d = 0
        cols = np.stack([e[1:], e[:-1], c, d], axis=-1)
        m.rows_block(cols, np.array([1.0, -1.0, -b.eta_charge, 1.0 / b.eta_discharge]), np.zeros(H), np.zeros(H))
        if with_binaries:
            z = m.var(H, 0.0, ub=1.0, integer=True)
            m.rows_block(np.stack([c, z], -1), np.array([1.0, -b.power_mw]), np.full(H, -INF), np.zeros(H))
            m.rows_block(np.stack([d, z], -1), np.array([1.0, b.power_mw]), np.full(H, -INF), np.full(H, b.power_mw))

        t0 = time.perf_counter()
        h, sol = m.solve(time_limit, mip_gap)
        elapsed = time.perf_counter() - t0
        status = h.modelStatusToString(h.getModelStatus())
        has_incumbent = h.getInfo().primal_solution_status == highspy.kSolutionStatusFeasible
        if status != "Optimal" and not (status == "Time limit reached" and has_incumbent):
            raise RuntimeError(f"HiGHS returned status {status!r}")
        return Plan(
            buy=sol[x],
            charge=sol[c],
            discharge=sol[d],
            soc=sol[e],
            capacity=float(sol[cap][0]),
            objective=float(h.getInfo().objective_function_value),
            status=status,
            solve_seconds=elapsed,
            n_scenarios=S,
            used_mip=with_binaries,
        )

    if mip is True:
        return build_and_solve(True)
    relaxed = build_and_solve(False)
    if mip is False or np.all(np.minimum(relaxed.charge, relaxed.discharge) <= complementarity_tol):
        return relaxed
    exact = build_and_solve(True)
    exact.solve_seconds += relaxed.solve_seconds
    return exact


def realized_cost(
    plan: Plan, prices: np.ndarray, load: np.ndarray, params: MarketParams | None = None
) -> dict[str, np.ndarray]:
    """Cost of a first-stage plan under one or many load paths (closed-form optimal recourse).

    ``load`` has shape (24,) or (S, 24); every returned entry has shape (S,).
    """
    p = params or MarketParams()
    Y = np.atleast_2d(np.asarray(load, float))
    prices = np.asarray(prices, float)
    net = Y - plan.discharge + plan.charge  # physical grid draw
    gap = net - plan.buy
    short, surplus = np.maximum(gap, 0.0), np.maximum(-gap, 0.0)
    energy = np.full(len(Y), float(prices @ plan.buy))
    imbalance = short @ (prices + p.lambda_up) - surplus @ (prices - p.lambda_down)
    capacity = np.full(len(Y), p.capacity_cost * plan.capacity)
    exceed = p.exceed_penalty * np.maximum(net.max(axis=1) - plan.capacity, 0.0)
    degradation = np.full(len(Y), p.battery.degradation_eur_per_mwh * float(np.sum(plan.charge + plan.discharge)))
    total = energy + imbalance + capacity + exceed + degradation
    return {
        "energy": energy,
        "imbalance": imbalance,
        "capacity": capacity,
        "exceedance": exceed,
        "degradation": degradation,
        "total": total,
    }
