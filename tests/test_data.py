from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from forecast_optimize.calendar_si import bridge_mask, day_type, easter_sunday, slovenian_holidays
from forecast_optimize.data.synthetic import SyntheticConfig, generate


def test_easter_known_dates() -> None:
    known = {2019: (4, 21), 2020: (4, 12), 2021: (4, 4), 2023: (4, 9), 2024: (3, 31), 2025: (4, 20), 2038: (4, 25)}
    for year, (m, d) in known.items():
        assert easter_sunday(year) == dt.date(year, m, d)


def test_holidays_2023_complete() -> None:
    hol = slovenian_holidays([2023])
    expected = {
        dt.date(2023, 1, 1), dt.date(2023, 1, 2), dt.date(2023, 2, 8), dt.date(2023, 4, 9),
        dt.date(2023, 4, 10), dt.date(2023, 4, 27), dt.date(2023, 5, 1), dt.date(2023, 5, 2),
        dt.date(2023, 5, 28), dt.date(2023, 6, 25), dt.date(2023, 8, 14), dt.date(2023, 8, 15),
        dt.date(2023, 10, 31), dt.date(2023, 11, 1), dt.date(2023, 12, 25), dt.date(2023, 12, 26),
    }  # fmt: skip
    assert set(hol) == expected


def test_second_january_was_working_day_2013_2016() -> None:
    assert dt.date(2015, 1, 2) not in slovenian_holidays([2015])
    assert dt.date(2017, 1, 2) in slovenian_holidays([2017])


def test_day_type_and_bridge() -> None:
    idx = pd.date_range("2024-04-26", "2024-05-03 23:00", freq="h")
    dtyp = pd.Series(day_type(idx), index=idx).resample("D").first()
    # Sat 27 Apr is a holiday -> type 2; Wed 1 May, Thu 2 May holidays; Fri 3 May is a bridge day
    assert dtyp[pd.Timestamp("2024-04-27")] == 2
    assert dtyp[pd.Timestamp("2024-05-01")] == 2
    assert dtyp[pd.Timestamp("2024-05-02")] == 2
    assert dtyp[pd.Timestamp("2024-04-30")] == 0


def test_bridge_days() -> None:
    idx = pd.date_range("2024-04-26", "2024-05-06 23:00", freq="h")
    b = pd.Series(bridge_mask(idx), index=idx).resample("D").first()
    # Fri 3 May 2024: holiday (Thu 2 May) before, Saturday after -> bridge day
    assert bool(b[pd.Timestamp("2024-05-03")])
    # Tue 30 Apr 2024: Mon 29 Apr is a working day -> not a bridge
    assert not bool(b[pd.Timestamp("2024-04-30")])
    # holidays themselves are never bridges
    assert not bool(b[pd.Timestamp("2024-05-02")])


def test_generator_reproducible_and_seeded() -> None:
    cfg = SyntheticConfig(start="2023-01-01", end="2023-03-31 23:00", seed=3)
    a, b = generate(cfg), generate(cfg)
    pd.testing.assert_frame_equal(a, b)
    c = generate(SyntheticConfig(start="2023-01-01", end="2023-03-31 23:00", seed=4))
    assert not np.allclose(a["load_mw"], c["load_mw"])


def test_generator_stylised_facts() -> None:
    df = generate(SyntheticConfig(start="2022-01-01", end="2023-12-31 23:00", seed=5))
    assert df.index.freqstr == "h" and df.notna().all().all()
    hourly = df["load_mw"].groupby(df.index.hour).mean()
    assert hourly.idxmin() in range(0, 6) and hourly.idxmax() in range(9, 21)
    dow = df["load_mw"].groupby(df.index.dayofweek).mean()
    assert dow[6] < dow[5] < dow[:5].min()  # Sunday < Saturday < weekdays
    # U-shaped temperature response: load is higher at both cold and hot extremes than at ~16 deg C
    t = df["temp_c"].ewm(halflife=12).mean()
    bins = pd.cut(t, [-30, 0, 14, 18, 40])
    by_t = df["load_mw"].groupby(bins, observed=True).mean()
    assert by_t.iloc[0] > by_t.iloc[2] and by_t.iloc[3] > by_t.iloc[2]
    # heteroskedastic noise: larger in cold weather
    resid = df["load_mw"] - df["load_mean_mw"]
    assert resid[t < 0].std() > resid[(t > 14) & (t < 18)].std()
