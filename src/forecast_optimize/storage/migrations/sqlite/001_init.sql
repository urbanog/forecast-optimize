-- 001: core tables (SQLite dialect). Time stamps are ISO-8601 text 'YYYY-MM-DD HH:MM:SS'
-- in local standard time; ISO text sorts chronologically, so range predicates work on it.

CREATE TABLE measurements (
    ts          TEXT PRIMARY KEY,
    load_mw     REAL NOT NULL,
    temp_c      REAL,
    temp_fc_c   REAL,
    da_price    REAL
) WITHOUT ROWID;

CREATE TABLE forecast_runs (
    run_id      INTEGER PRIMARY KEY,
    model       TEXT NOT NULL,
    issued_at   TEXT NOT NULL,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

-- one row per (run, target hour, quantile level); horizon_h = target_ts - issued_at in hours
CREATE TABLE forecast_values (
    run_id      INTEGER NOT NULL REFERENCES forecast_runs(run_id) ON DELETE CASCADE,
    target_ts   TEXT    NOT NULL,
    horizon_h   INTEGER NOT NULL,
    quantile    REAL    NOT NULL CHECK (quantile > 0 AND quantile < 1),
    value_mw    REAL    NOT NULL,
    PRIMARY KEY (run_id, target_ts, quantile)
);

-- point (conditional-mean) forecasts
CREATE TABLE forecast_points (
    run_id      INTEGER NOT NULL REFERENCES forecast_runs(run_id) ON DELETE CASCADE,
    target_ts   TEXT    NOT NULL,
    horizon_h   INTEGER NOT NULL,
    mean_mw     REAL    NOT NULL,
    PRIMARY KEY (run_id, target_ts)
);
