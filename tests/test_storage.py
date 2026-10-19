from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from forecast_optimize.storage.db import connect


@pytest.fixture()
def db():
    d = connect("sqlite:///:memory:")
    d.migrate()
    yield d
    d.close()


def test_migrations_are_idempotent(db) -> None:
    assert db.applied_migrations() == ["001_init", "002_forecast_indexes"]
    assert db.migrate() == []
    names = {r[0] for r in db.query("SELECT name FROM sqlite_master WHERE type = 'index'")}
    assert "ix_fv_quantile_target_horizon" in names


def test_measurement_roundtrip_and_upsert(db, small_data: pd.DataFrame) -> None:
    part = small_data.iloc[:72]
    assert db.upsert_measurements(part) == 72
    back = db.load_measurements("2023-01-01", "2023-01-04")
    pd.testing.assert_frame_equal(back, part[back.columns], check_freq=False, check_names=False)
    changed = part.copy()
    changed["load_mw"] += 1.0
    db.upsert_measurements(changed)
    assert db.query("SELECT COUNT(*) FROM measurements")[0][0] == 72
    assert db.load_measurements()["load_mw"].iloc[0] == pytest.approx(part["load_mw"].iloc[0] + 1.0)


def test_latest_forecast_picks_smallest_horizon(db) -> None:
    levels = (0.1, 0.5, 0.9)
    target = pd.date_range("2024-03-02", periods=24, freq="h")
    # two runs cover the same day: issued 24 h earlier and at 00:00 of the day itself
    for issued, offset in ((pd.Timestamp("2024-03-01"), 100.0), (pd.Timestamp("2024-03-02"), 0.0)):
        point = 1000.0 + offset + np.arange(24)
        Q = point[:, None] + np.array([-50.0, 0.0, 50.0])[None, :]
        db.save_forecast("m", issued, target, point, Q, levels)
    latest = db.latest_forecast("2024-03-02", "2024-03-03", quantile=0.5)
    assert len(latest) == 24
    assert (latest["horizon_h"] == np.arange(24)).all()
    np.testing.assert_allclose(latest["value_mw"].to_numpy(), 1000.0 + np.arange(24))


def test_foreign_keys_cascade(db) -> None:
    idx = pd.date_range("2024-01-01", periods=24, freq="h")
    run = db.save_forecast("m", idx[0], idx, np.ones(24), np.ones((24, 1)), (0.5,))
    db.execute("DELETE FROM forecast_runs WHERE run_id = ?", (run,))
    db.commit()
    assert db.query("SELECT COUNT(*) FROM forecast_values")[0][0] == 0


@pytest.mark.skipif("FO_TEST_POSTGRES_URL" not in os.environ, reason="set FO_TEST_POSTGRES_URL to test PostgreSQL")
def test_postgres_roundtrip(small_data: pd.DataFrame) -> None:  # pragma: no cover - optional
    d = connect(os.environ["FO_TEST_POSTGRES_URL"])
    for t in ("forecast_values", "forecast_points", "forecast_runs", "measurements", "schema_migrations"):
        d.execute(f"DROP TABLE IF EXISTS {t} CASCADE")
    d.commit()
    assert d.migrate() == ["001_init", "002_forecast_indexes"]
    d.upsert_measurements(small_data.iloc[:48])
    assert len(d.load_measurements()) == 48
    idx = pd.date_range("2023-01-02", periods=24, freq="h")
    d.save_forecast("m", idx[0], idx, np.arange(24.0), np.arange(24.0)[:, None], (0.5,))
    assert len(d.latest_forecast("2023-01-02", "2023-01-03", 0.5)) == 24
    d.close()
