"""SQL optimisation study on the forecast store.

Builds a realistic forecast table (runs issued every 6 h for two years, horizons
0-47 h, 7 quantile levels: ~1M rows), then times typical read queries

* Q1  latest forecast per target hour (four formulations),
* Q2  one day of forecasts: non-sargable ``date(col) = ...`` vs a range predicate,
* Q3  rolling 7-day MAE of the latest forecasts: correlated subquery vs window function,

before and after migration ``002_forecast_indexes.sql``, printing
``EXPLAIN QUERY PLAN`` and median timings.  All formulations of a query are
checked to return identical rows.  Results are written to
``docs/sql_benchmark_results.md`` (raw) - the discussion lives in
``docs/sql_optimization.md``.

Optionally set ``FO_BENCH_POSTGRES_URL=postgresql://...`` to repeat Q1 on
PostgreSQL (timings + ``EXPLAIN`` plans).

Usage:  python scripts/sql_benchmark.py [--db PATH] [--repeats 5]
"""

from __future__ import annotations

import argparse
import os
import platform
import sqlite3
import statistics
import time
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd

LEVELS = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
RUN_EVERY_H = 6
TIMEOUT_S = 60.0  # queries running longer are aborted and reported as "> 60 s"
PG_TIMEOUT_S = 60.0
HORIZON_H = 48
START, END = "2023-01-01", "2025-01-01"

Q1_RANGE = ("2024-06-01 00:00:00", "2024-07-01 00:00:00")
Q2_DAY = "2024-06-12"
Q3_RANGE = ("2024-04-01 00:00:00", "2024-07-01 00:00:00")

# --------------------------------------------------------------------------- queries
Q1 = {
    "correlated_subquery": """
        SELECT fv.target_ts, fv.value_mw
        FROM forecast_values AS fv
        WHERE fv.quantile = 0.5 AND fv.target_ts >= :lo AND fv.target_ts < :hi
          AND fv.horizon_h = (SELECT MIN(f2.horizon_h) FROM forecast_values AS f2
                              WHERE f2.quantile = fv.quantile AND f2.target_ts = fv.target_ts)
        ORDER BY fv.target_ts""",
    "group_by_join": """
        WITH best AS (
            SELECT target_ts, MIN(horizon_h) AS h FROM forecast_values
            WHERE quantile = 0.5 AND target_ts >= :lo AND target_ts < :hi
            GROUP BY target_ts)
        SELECT fv.target_ts, fv.value_mw
        FROM best JOIN forecast_values AS fv
          ON fv.quantile = 0.5 AND fv.target_ts = best.target_ts AND fv.horizon_h = best.h
        ORDER BY fv.target_ts""",
    "group_by_join_pushdown": """
        WITH best AS (
            SELECT target_ts, MIN(horizon_h) AS h FROM forecast_values
            WHERE quantile = 0.5 AND target_ts >= :lo AND target_ts < :hi
            GROUP BY target_ts)
        SELECT fv.target_ts, fv.value_mw
        FROM best JOIN forecast_values AS fv
          ON fv.target_ts = best.target_ts AND fv.horizon_h = best.h
        WHERE fv.quantile = 0.5 AND fv.target_ts >= :lo AND fv.target_ts < :hi
        ORDER BY fv.target_ts""",
    "window_row_number": """
        SELECT target_ts, value_mw FROM (
            SELECT target_ts, value_mw,
                   ROW_NUMBER() OVER (PARTITION BY target_ts ORDER BY horizon_h) AS rn
            FROM forecast_values
            WHERE quantile = 0.5 AND target_ts >= :lo AND target_ts < :hi)
        WHERE rn = 1 ORDER BY target_ts""",
    "sqlite_bare_min": """
        SELECT target_ts, value_mw FROM (
            SELECT target_ts, MIN(horizon_h), value_mw FROM forecast_values
            WHERE quantile = 0.5 AND target_ts >= :lo AND target_ts < :hi
            GROUP BY target_ts)
        ORDER BY target_ts""",
}

Q2 = {
    "date_function": """
        SELECT target_ts, horizon_h, value_mw FROM forecast_values
        WHERE quantile = 0.9 AND date(target_ts) = :day
        ORDER BY target_ts, horizon_h""",
    "range_predicate": """
        SELECT target_ts, horizon_h, value_mw FROM forecast_values
        WHERE quantile = 0.9 AND target_ts >= :day AND target_ts < date(:day, '+1 day')
        ORDER BY target_ts, horizon_h""",
}

_LATEST = """
    latest AS MATERIALIZED (
        SELECT target_ts, value_mw FROM (
            SELECT target_ts, value_mw,
                   ROW_NUMBER() OVER (PARTITION BY target_ts ORDER BY horizon_h) AS rn
            FROM forecast_values
            WHERE quantile = 0.5 AND target_ts >= datetime(:lo, '-168 hours') AND target_ts < :hi)
        WHERE rn = 1),
    err AS MATERIALIZED (
        SELECT l.target_ts AS ts, ABS(m.load_mw - l.value_mw) AS abs_err
        FROM latest AS l JOIN measurements AS m ON m.ts = l.target_ts)"""

Q3 = {
    "correlated_rolling": f"""
        WITH {_LATEST}
        SELECT e1.ts,
               (SELECT AVG(e2.abs_err) FROM err AS e2
                WHERE e2.ts > datetime(e1.ts, '-168 hours') AND e2.ts <= e1.ts) AS mae_7d
        FROM err AS e1 WHERE e1.ts >= :lo ORDER BY e1.ts""",
    "window_rolling": f"""
        WITH {_LATEST}
        SELECT ts, mae_7d FROM (
            SELECT ts, AVG(abs_err) OVER (ORDER BY ts ROWS BETWEEN 167 PRECEDING AND CURRENT ROW) AS mae_7d
            FROM err)
        WHERE ts >= :lo ORDER BY ts""",
}


# --------------------------------------------------------------------------- data
def migration_sql(version: str) -> str:
    folder = resources.files("forecast_optimize.storage") / "migrations" / "sqlite"
    return (folder / f"{version}.sql").read_text(encoding="utf-8")


def build(path: Path) -> dict[str, int]:
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(path)
    conn.executescript(migration_sql("001_init"))
    rng = np.random.default_rng(0)
    hours = pd.date_range(START, END, freq="h", inclusive="left")
    load = 900 + 150 * np.sin(2 * np.pi * (hours.hour.to_numpy() - 8) / 24) + rng.normal(0, 25, len(hours))
    fmt = "%Y-%m-%d %H:%M:%S"
    ts_str = hours.strftime(fmt).to_numpy()
    conn.executemany("INSERT INTO measurements (ts, load_mw) VALUES (?, ?)", zip(ts_str, load.tolist(), strict=True))
    issues = hours[::RUN_EVERY_H]
    conn.executemany(
        "INSERT INTO forecast_runs (run_id, model, issued_at) VALUES (?, 'lightgbm', ?)",
        ((i + 1, t) for i, t in enumerate(issues.strftime(fmt))),
    )
    rows = []
    pos = {t: i for i, t in enumerate(ts_str)}
    zq = np.array([-1.645, -1.2816, -0.6745, 0.0, 0.6745, 1.2816, 1.645])
    for run_id, t in enumerate(issues.strftime(fmt), start=1):
        i0 = pos[t]
        for h in range(HORIZON_H):
            i = i0 + h
            if i >= len(hours):
                break
            sd = 15 + 0.6 * h
            mean = load[i] + rng.normal(0, sd)
            for a, z in zip(LEVELS, zq, strict=True):
                rows.append((run_id, ts_str[i], h, a, mean + z * sd))
    conn.executemany(
        "INSERT INTO forecast_values (run_id, target_ts, horizon_h, quantile, value_mw) VALUES (?, ?, ?, ?, ?)", rows
    )
    conn.commit()
    conn.execute("ANALYZE")
    n = {
        "measurements": conn.execute("SELECT COUNT(*) FROM measurements").fetchone()[0],
        "forecast_runs": conn.execute("SELECT COUNT(*) FROM forecast_runs").fetchone()[0],
        "forecast_values": conn.execute("SELECT COUNT(*) FROM forecast_values").fetchone()[0],
    }
    conn.close()
    return n


# --------------------------------------------------------------------------- timing
def plan(conn: sqlite3.Connection, sql: str, params: dict) -> list[str]:
    return [r[3] for r in conn.execute("EXPLAIN QUERY PLAN " + sql, params).fetchall()]


def timed(conn: sqlite3.Connection, sql: str, params: dict, repeats: int) -> tuple[float, list | None]:
    """Median wall time in ms; (inf, None) if a run exceeds TIMEOUT_S."""
    out: list | None = []
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        conn.set_progress_handler(lambda t0=t0: int(time.perf_counter() - t0 > TIMEOUT_S), 10_000)
        try:
            out = conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError as exc:
            if "interrupted" not in str(exc):
                raise
            return float("inf"), None
        finally:
            conn.set_progress_handler(None, 0)
        times.append(time.perf_counter() - t0)
    return statistics.median(times) * 1000.0, out


def run_suite(conn: sqlite3.Connection, repeats: int, stage: str) -> list[dict]:
    results = []
    suites = (
        ("Q1 latest forecast per hour (30 days)", Q1, {"lo": Q1_RANGE[0], "hi": Q1_RANGE[1]}),
        ("Q2 one day, all horizons (q=0.9)", Q2, {"day": Q2_DAY}),
        ("Q3 rolling 7-day MAE (91 days)", Q3, {"lo": Q3_RANGE[0], "hi": Q3_RANGE[1]}),
    )
    for title, variants, params in suites:
        reference = None
        for name, sql in variants.items():
            ms, rows = timed(conn, sql, params, repeats if "correlated" not in name or stage == "after" else 1)
            if rows is None:
                print(f"[{stage:6}] {title:42} {name:22}  > {TIMEOUT_S:.0f} s (aborted)")
                results.append(
                    {
                        "stage": stage,
                        "query": title,
                        "variant": name,
                        "ms": ms,
                        "rows": -1,
                        "plan": plan(conn, sql, params),
                    }
                )
                continue
            rounded = [tuple(round(v, 6) if isinstance(v, float) else v for v in r) for r in rows]
            if reference is None:
                reference = rounded
            elif rounded != reference:
                raise AssertionError(f"{title}: variant {name} returned different rows")
            results.append(
                {
                    "stage": stage,
                    "query": title,
                    "variant": name,
                    "ms": ms,
                    "rows": len(rows),
                    "plan": plan(conn, sql, params),
                }
            )
            print(f"[{stage:6}] {title:42} {name:22} {ms:10.2f} ms  ({len(rows)} rows)")
    return results


def postgres_q1(url: str) -> list[dict]:  # pragma: no cover - optional
    import psycopg

    from forecast_optimize.storage.db import connect

    sq = sqlite3.connect(ARGS.db)
    db = connect(url)
    for t in ("forecast_values", "forecast_points", "forecast_runs", "measurements", "schema_migrations"):
        db.execute(f"DROP TABLE IF EXISTS {t} CASCADE")
    db.commit()
    db.conn.execute(migration_sql_pg("001_init"))
    with db.conn.cursor() as cur:
        with cur.copy("COPY forecast_runs (run_id, model, issued_at) FROM STDIN") as cp:
            for r in sq.execute("SELECT run_id, model, issued_at FROM forecast_runs"):
                cp.write_row(r)
        with cur.copy("COPY forecast_values (run_id, target_ts, horizon_h, quantile, value_mw) FROM STDIN") as cp:
            for r in sq.execute("SELECT run_id, target_ts, horizon_h, quantile, value_mw FROM forecast_values"):
                cp.write_row(r)
    db.commit()
    print(f"copied {sq.execute('SELECT COUNT(*) FROM forecast_values').fetchone()[0]:,} rows to PostgreSQL")
    db.conn.execute(f"SET statement_timeout = '{int(PG_TIMEOUT_S * 1000)}ms'")
    out = []
    pg_q1 = {k: v.replace(":lo", "%(lo)s").replace(":hi", "%(hi)s") for k, v in Q1.items() if k != "sqlite_bare_min"}
    params = {"lo": Q1_RANGE[0], "hi": Q1_RANGE[1]}
    for stage in ("before", "after"):
        if stage == "after":
            db.conn.execute(migration_sql_pg("002_forecast_indexes"))
            db.commit()
        db.conn.execute("ANALYZE")
        for name, sql in pg_q1.items():
            times = []
            ex = [r[0] for r in db.conn.execute("EXPLAIN (COSTS OFF) " + sql, params)]
            for _ in range(ARGS.repeats):
                t0 = time.perf_counter()
                try:
                    db.conn.execute(sql, params).fetchall()
                except psycopg.errors.QueryCanceled:
                    db.conn.rollback()
                    db.conn.execute(f"SET statement_timeout = '{int(PG_TIMEOUT_S * 1000)}ms'")
                    times = [float("inf")]
                    break
                times.append(time.perf_counter() - t0)
            ms = statistics.median(times) * 1000
            print(f"[pg {stage:6}] Q1 {name:22} {ms:10.2f} ms")
            out.append({"stage": stage, "variant": name, "ms": ms, "plan": ex})
    db.close()
    return out


def migration_sql_pg(version: str) -> str:  # pragma: no cover - optional
    folder = resources.files("forecast_optimize.storage") / "migrations" / "postgres"
    return (folder / f"{version}.sql").read_text(encoding="utf-8")


def _fmt(ms: float) -> str:
    return f"> {TIMEOUT_S * 1000:,.0f} (aborted)" if ms == float("inf") else f"{ms:,.1f}"


def write_report(path: Path, sizes: dict[str, int], results: list[dict], pg: list[dict] | None) -> None:
    lines = [
        "# SQL benchmark - raw results",
        "",
        "Generated by `python scripts/sql_benchmark.py`; do not edit by hand. Discussion: [sql_optimization.md](sql_optimization.md).",
        "",
        f"* SQLite {sqlite3.sqlite_version}, Python {platform.python_version()}, {platform.machine()}",
        f"* rows: {sizes['forecast_values']:,} forecast values, {sizes['forecast_runs']:,} runs, "
        f"{sizes['measurements']:,} measurements; timings are medians of {ARGS.repeats} runs "
        "(single run for correlated variants without indexes)",
        "",
        "| query | variant | before 002 [ms] | after 002 [ms] | rows |",
        "|---|---|---:|---:|---:|",
    ]
    df = pd.DataFrame(results)
    for (q, v), g in df.groupby(["query", "variant"], sort=False):
        b = g[g.stage == "before"].iloc[0]
        a = g[g.stage == "after"].iloc[0]
        lines.append(f"| {q} | `{v}` | {_fmt(b.ms)} | {_fmt(a.ms)} | {a.rows} |")
    lines += ["", "## Query plans", ""]
    for r in results:
        lines += [f"**{r['stage']} - {r['query']} - `{r['variant']}`**", "", "```", *r["plan"], "```", ""]
    if pg:
        lines += ["## PostgreSQL (Q1 only)", "", "| variant | before [ms] | after [ms] |", "|---|---:|---:|"]
        p = pd.DataFrame(pg)
        for v, g in p.groupby("variant", sort=False):
            lines.append(
                f"| `{v}` | {_fmt(g[g.stage == 'before'].ms.iloc[0])} | {_fmt(g[g.stage == 'after'].ms.iloc[0])} |"
            )
        lines.append("")
        for r in pg:
            lines += [f"**pg {r['stage']} - `{r['variant']}`**", "", "```", *r["plan"], "```", ""]
    path.write_text("\n".join(lines))


def main() -> None:
    t0 = time.perf_counter()
    sizes = build(Path(ARGS.db))
    print(f"built {ARGS.db} in {time.perf_counter() - t0:.1f}s: {sizes}")
    conn = sqlite3.connect(ARGS.db)
    results = run_suite(conn, ARGS.repeats, "before")
    conn.executescript(migration_sql("002_forecast_indexes"))
    conn.execute("ANALYZE")
    results += run_suite(conn, ARGS.repeats, "after")
    conn.close()
    pg = postgres_q1(os.environ["FO_BENCH_POSTGRES_URL"]) if os.environ.get("FO_BENCH_POSTGRES_URL") else None
    write_report(Path(ARGS.out), sizes, results, pg)
    print(f"wrote {ARGS.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--db", default="data/sql_benchmark.db")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--out", default="docs/sql_benchmark_results.md")
    ARGS = parser.parse_args()
    Path(ARGS.db).parent.mkdir(parents=True, exist_ok=True)
    main()
