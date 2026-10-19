-- 002: indexes that serve the read patterns of the API and the reports
-- (see docs/sql_optimization.md for the measurements that motivated them).

-- "latest forecast for each target hour": equality on quantile, range on target_ts,
-- then the smallest horizon (= the most recent issue) within each target_ts.
-- value_mw is included so the query is answered from the index alone (covering index).
CREATE INDEX ix_fv_quantile_target_horizon
    ON forecast_values (quantile, target_ts, horizon_h, value_mw);

-- runs looked up by model and issue time
CREATE INDEX ix_runs_model_issued ON forecast_runs (model, issued_at);
