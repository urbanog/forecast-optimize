-- 001: core tables (PostgreSQL dialect)

CREATE TABLE measurements (
    ts          TIMESTAMP PRIMARY KEY,
    load_mw     DOUBLE PRECISION NOT NULL,
    temp_c      DOUBLE PRECISION,
    temp_fc_c   DOUBLE PRECISION,
    da_price    DOUBLE PRECISION
);

CREATE TABLE forecast_runs (
    run_id      BIGSERIAL PRIMARY KEY,
    model       TEXT NOT NULL,
    issued_at   TIMESTAMP NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE forecast_values (
    run_id      BIGINT  NOT NULL REFERENCES forecast_runs(run_id) ON DELETE CASCADE,
    target_ts   TIMESTAMP NOT NULL,
    horizon_h   INTEGER NOT NULL,
    quantile    DOUBLE PRECISION NOT NULL CHECK (quantile > 0 AND quantile < 1),
    value_mw    DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (run_id, target_ts, quantile)
);

CREATE TABLE forecast_points (
    run_id      BIGINT  NOT NULL REFERENCES forecast_runs(run_id) ON DELETE CASCADE,
    target_ts   TIMESTAMP NOT NULL,
    horizon_h   INTEGER NOT NULL,
    mean_mw     DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (run_id, target_ts)
);
