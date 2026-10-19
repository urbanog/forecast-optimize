-- 002: read-path indexes (PostgreSQL dialect; INCLUDE makes the index covering)

CREATE INDEX ix_fv_quantile_target_horizon
    ON forecast_values (quantile, target_ts, horizon_h) INCLUDE (value_mw);

CREATE INDEX ix_runs_model_issued ON forecast_runs (model, issued_at);
