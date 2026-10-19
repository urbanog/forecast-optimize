"""No-leakage tests: features and fold forecasts must not depend on information after the issue time."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from forecast_optimize.features import make_features
from forecast_optimize.models.cv import fit_calibrate_predict, rolling_origin_folds
from forecast_optimize.models.forecasters import make_forecaster


def _corrupt_after(df: pd.DataFrame, issue: pd.Timestamp) -> pd.DataFrame:
    """Destroy everything not known at ``issue`` (00:00 of day D).

    Observed load and temperature from ``issue`` on are unknown; the day-ahead
    temperature forecast is known for day D but not for later days.
    """
    bad = df.copy()
    rng = np.random.default_rng(0)
    after = bad.index >= issue
    bad.loc[after, "load_mw"] = rng.uniform(-1e4, 1e4, after.sum())
    bad.loc[after, "temp_c"] = rng.uniform(-50, 50, after.sum())
    later = bad.index >= issue + pd.Timedelta(days=1)
    bad.loc[later, "temp_fc_c"] = rng.uniform(-50, 50, later.sum())
    return bad


@pytest.mark.parametrize("issue", ["2023-06-15", "2024-01-02", "2024-02-08"])
def test_features_use_only_information_at_issue_time(small_data: pd.DataFrame, issue: str) -> None:
    t0 = pd.Timestamp(issue)
    day = (small_data.index >= t0) & (small_data.index < t0 + pd.Timedelta(days=1))
    clean = make_features(small_data).loc[day]
    dirty = make_features(_corrupt_after(small_data, t0)).loc[day]
    pd.testing.assert_frame_equal(clean, dirty)


def test_folds_are_disjoint_ordered_and_tiling(small_data: pd.DataFrame) -> None:
    folds = list(rolling_origin_folds(small_data.index, "2024-01-01", 4, 14, 28, "2023-01-16"))
    for f in folds:
        assert f.train.max() < f.cal.min() and f.cal.max() < f.test.min()
        assert len(f.test) == 14 * 24 and len(f.cal) == 28 * 24
    for a, b in itertools.pairwise(folds):
        assert a.test.max() + pd.Timedelta(hours=1) == b.test.min()  # test blocks tile time
        assert len(b.train) > len(a.train)  # expanding window


def test_fold_does_not_fit_raises(small_data: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        list(rolling_origin_folds(small_data.index, "2024-04-20", 2, 14, 28))


@pytest.mark.parametrize("name", ["fourier_linear", "lightgbm"])
def test_first_test_day_forecast_ignores_the_future(
    small_data: pd.DataFrame, fast_lgbm: dict[str, object], name: str
) -> None:
    """Fit + calibrate + predict on clean data and on data destroyed after the origin: day 1 must agree."""
    fold = next(rolling_origin_folds(small_data.index, "2024-02-01", 1, 7, 28, "2023-01-16"))
    kw = fast_lgbm if name == "lightgbm" else {}
    dirty = _corrupt_after(small_data, fold.origin)
    res = []
    for df in (small_data, dirty):
        res.append(fit_calibrate_predict(make_forecaster(name, **kw), df, make_features(df), fold))
    first = slice(0, 24)
    np.testing.assert_allclose(res[0].point[first], res[1].point[first])
    np.testing.assert_allclose(res[0].q_cal[first], res[1].q_cal[first])
