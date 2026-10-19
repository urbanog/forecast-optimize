"""Persistence of measurements and forecasts (SQLite by default, PostgreSQL optional).

The connection is chosen by ``FO_DATABASE_URL``:

* unset or ``sqlite:///path/to/file.db`` -> stdlib ``sqlite3`` (default ``data/forecast_optimize.db``),
* ``sqlite:///:memory:`` -> in-memory SQLite (used by the tests),
* ``postgresql://user@host/dbname`` -> PostgreSQL via ``psycopg`` (optional extra ``[postgres]``).

Schema changes are plain SQL files in ``migrations/<dialect>/NNN_name.sql``;
:func:`migrate` applies the ones not yet recorded in ``schema_migrations``,
each in its own transaction.
"""

from __future__ import annotations

import datetime as dt
import os
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from importlib import resources
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_SQLITE_PATH = os.path.join("data", "forecast_optimize.db")


@dataclass
class Database:
    conn: Any
    dialect: str  # "sqlite" | "postgres"

    # -- low level -----------------------------------------------------------
    def _sql(self, sql: str) -> str:
        return sql.replace("?", "%s") if self.dialect == "postgres" else sql

    def ts(self, x: pd.Timestamp | dt.datetime | str) -> str | dt.datetime:
        t = pd.Timestamp(x)
        return t.strftime("%Y-%m-%d %H:%M:%S") if self.dialect == "sqlite" else t.to_pydatetime()

    def execute(self, sql: str, params: Sequence[Any] = ()) -> Any:
        cur = self.conn.cursor()
        cur.execute(self._sql(sql), tuple(params))
        return cur

    def executemany(self, sql: str, rows: Iterable[Sequence[Any]]) -> None:
        cur = self.conn.cursor()
        cur.executemany(self._sql(sql), list(rows))

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[tuple[Any, ...]]:
        return list(self.execute(sql, params).fetchall())

    def commit(self) -> None:
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # -- migrations ------------------------------------------------------------
    def applied_migrations(self) -> list[str]:
        self.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations (version TEXT PRIMARY KEY, applied_at TEXT NOT NULL)"
        )
        self.commit()
        return [r[0] for r in self.query("SELECT version FROM schema_migrations ORDER BY version")]

    def migrate(self) -> list[str]:
        """Apply pending migrations in lexicographic order; return the versions applied now."""
        done = set(self.applied_migrations())
        folder = resources.files("forecast_optimize.storage") / "migrations" / self.dialect
        files = sorted((f for f in folder.iterdir() if f.name.endswith(".sql")), key=lambda f: f.name)
        applied = []
        for f in files:
            version = f.name.removesuffix(".sql")
            if version in done:
                continue
            sql = f.read_text(encoding="utf-8")
            now = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
            if self.dialect == "sqlite":
                # executescript commits first; wrap in an explicit transaction for atomicity
                self.conn.executescript(
                    f"BEGIN;\n{sql}\nINSERT INTO schema_migrations VALUES ('{version}', '{now}');\nCOMMIT;"
                )
            else:
                with self.conn.transaction():
                    self.conn.execute(sql)
                    self.conn.execute("INSERT INTO schema_migrations VALUES (%s, %s)", (version, now))
            applied.append(version)
        return applied

    # -- measurements -------------------------------------------------------------
    def upsert_measurements(self, df: pd.DataFrame) -> int:
        cols = ["load_mw", "temp_c", "temp_fc_c", "da_price"]
        rows = [
            (self.ts(t), *(None if pd.isna(v) else float(v) for v in vals))
            for t, vals in zip(df.index, df[cols].to_numpy(), strict=True)
        ]
        self.executemany(
            "INSERT INTO measurements (ts, load_mw, temp_c, temp_fc_c, da_price) VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT (ts) DO UPDATE SET load_mw = excluded.load_mw, temp_c = excluded.temp_c, "
            "temp_fc_c = excluded.temp_fc_c, da_price = excluded.da_price",
            rows,
        )
        self.commit()
        return len(rows)

    def load_measurements(self, start: str | None = None, end: str | None = None) -> pd.DataFrame:
        """Measurements with start <= ts < end as an hourly DataFrame."""
        sql = "SELECT ts, load_mw, temp_c, temp_fc_c, da_price FROM measurements WHERE 1 = 1"
        params: list[Any] = []
        if start is not None:
            sql += " AND ts >= ?"
            params.append(self.ts(start))
        if end is not None:
            sql += " AND ts < ?"
            params.append(self.ts(end))
        rows = self.query(sql + " ORDER BY ts", params)
        df = pd.DataFrame(rows, columns=["ts", "load_mw", "temp_c", "temp_fc_c", "da_price"])
        df["ts"] = pd.to_datetime(df["ts"])
        df = df.set_index("ts").astype(float)
        if len(df):
            df = df.asfreq("h")
        return df

    # -- forecasts ------------------------------------------------------------------
    def save_forecast(
        self,
        model: str,
        issued_at: pd.Timestamp,
        index: pd.DatetimeIndex,
        point: np.ndarray,
        quantiles: np.ndarray,
        levels: Sequence[float],
    ) -> int:
        issued = pd.Timestamp(issued_at)
        cur = self.execute(
            "INSERT INTO forecast_runs (model, issued_at) VALUES (?, ?) RETURNING run_id", (model, self.ts(issued))
        )
        run_id = int(cur.fetchone()[0])
        horizons = ((index - issued) / pd.Timedelta(hours=1)).astype(int)
        self.executemany(
            "INSERT INTO forecast_points (run_id, target_ts, horizon_h, mean_mw) VALUES (?, ?, ?, ?)",
            [(run_id, self.ts(t), int(hz), float(v)) for t, hz, v in zip(index, horizons, point, strict=True)],
        )
        rows = []
        for j, a in enumerate(levels):
            rows.extend(
                (run_id, self.ts(t), int(hz), float(a), float(v))
                for t, hz, v in zip(index, horizons, quantiles[:, j], strict=True)
            )
        self.executemany(
            "INSERT INTO forecast_values (run_id, target_ts, horizon_h, quantile, value_mw) VALUES (?, ?, ?, ?, ?)",
            rows,
        )
        self.commit()
        return run_id

    def latest_forecast(self, start: str, end: str, quantile: float = 0.5) -> pd.DataFrame:
        """Most recent forecast (smallest horizon) of each target hour in [start, end) for one quantile.

        Aggregate + join with the range predicate repeated on the joined table: the
        fastest portable formulation in docs/sql_optimization.md on both SQLite and
        PostgreSQL, served by index ``ix_fv_quantile_target_horizon``.
        """
        lo, hi = self.ts(start), self.ts(end)
        rows = self.query(LATEST_FORECAST_SQL, (quantile, lo, hi, quantile, lo, hi))
        df = pd.DataFrame(rows, columns=["target_ts", "horizon_h", "value_mw"])
        df["target_ts"] = pd.to_datetime(df["target_ts"])
        return df.set_index("target_ts")


# The range on fv repeats the one inside `best`: logically redundant, but it lets the
# planner restrict the joined side by index instead of probing it once per group.
LATEST_FORECAST_SQL = """
WITH best AS (
    SELECT target_ts, MIN(horizon_h) AS horizon_h
    FROM forecast_values
    WHERE quantile = ? AND target_ts >= ? AND target_ts < ?
    GROUP BY target_ts
)
SELECT fv.target_ts, fv.horizon_h, fv.value_mw
FROM best
JOIN forecast_values AS fv
  ON fv.target_ts = best.target_ts AND fv.horizon_h = best.horizon_h
WHERE fv.quantile = ? AND fv.target_ts >= ? AND fv.target_ts < ?
ORDER BY fv.target_ts
"""


def connect(url: str | None = None) -> Database:
    """Open a database from a URL (default: ``FO_DATABASE_URL`` or the local SQLite file)."""
    url = url or os.environ.get("FO_DATABASE_URL") or f"sqlite:///{DEFAULT_SQLITE_PATH}"
    if url.startswith("sqlite:///"):
        path = url.removeprefix("sqlite:///")
        if path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        conn = sqlite3.connect(path, check_same_thread=False)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL") if path != ":memory:" else None
        return Database(conn, "sqlite")
    if url.startswith(("postgresql://", "postgres://")):
        try:
            import psycopg
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("PostgreSQL support needs `pip install forecast-optimize[postgres]`") from exc
        return Database(psycopg.connect(url), "postgres")
    raise ValueError(f"unsupported database URL {url!r}")
