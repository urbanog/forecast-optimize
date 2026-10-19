"""Feature engineering for day-ahead load forecasting.

Information set
---------------
The forecast for calendar day D is *issued at 00:00 of D*: every measurement up
to and including 23:00 of D-1 is known, together with the day-ahead temperature
forecast for D.  All features for a row with time stamp t in day D are functions
of that information set only; ``tests/test_leakage.py`` checks this by corrupting
all data after the issue time and verifying that the features do not change.

(A real day-ahead market bid is due around noon of D-1; moving the issue time
earlier only shifts the lags - see the README, "Limitations".)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from forecast_optimize.calendar_si import bridge_mask, day_type, holiday_mask

TARGET = "load_mw"


def _ewm(x: pd.Series, halflife_h: float) -> pd.Series:
    return x.ewm(halflife=halflife_h, adjust=False).mean()


def calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Deterministic calendar features (known arbitrarily far in advance)."""
    doy = np.asarray(index.dayofyear, dtype=float)
    out = pd.DataFrame(index=index)
    out["hour"] = index.hour
    out["dow"] = index.dayofweek
    out["day_type"] = day_type(index)
    out["is_holiday"] = holiday_mask(index).astype(np.int8)
    out["is_bridge"] = bridge_mask(index).astype(np.int8)
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    month = np.asarray(index.month)
    dom = np.asarray(index.day)
    out["xmas_period"] = (((month == 12) & (dom >= 24)) | ((month == 1) & (dom <= 6))).astype(np.int8)
    return out


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the LightGBM feature matrix for every hour in ``df``.

    ``df`` must have an hourly, gap-free index and columns ``load_mw``, ``temp_c``
    (observed temperature) and ``temp_fc_c`` (day-ahead temperature forecast).
    Rows whose lags are not available (the first two weeks) contain NaN.
    """
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex) or (idx.freq is None and pd.infer_freq(idx) != "h"):
        raise ValueError("df must have a regular hourly DatetimeIndex")
    y = df[TARGET]
    feats = calendar_features(idx)

    # Weather: the day-ahead forecast for the target hour and causal smoothings of the
    # *archived forecast* series (at hour t these use forecasts for hours <= t, all of
    # which are known at the issue time 00:00 of D).
    tfc = df["temp_fc_c"]
    feats["temp_fc"] = tfc
    feats["temp_fc_ewm12"] = _ewm(tfc, 12.0)
    feats["temp_fc_ewm48"] = _ewm(tfc, 48.0)
    day = idx.normalize()
    feats["temp_fc_daymean"] = tfc.groupby(day).transform("mean")

    # Load lags.  For hour h of day D, lag k*24 is hour h of day D-k: always observed.
    for k in (1, 2, 7, 14):
        feats[f"lag_{24 * k}"] = y.shift(24 * k)
    # Daily aggregates of the previous day(s), broadcast to all hours of D.
    daily_mean = y.groupby(day).mean()
    daily_max = y.groupby(day).max()
    one_day = pd.Timedelta(days=1)
    feats["prevday_mean"] = daily_mean.reindex(day - one_day).to_numpy()
    feats["prevday_max"] = daily_max.reindex(day - one_day).to_numpy()
    feats["prev7_mean"] = daily_mean.rolling(7).mean().reindex(day - one_day).to_numpy()
    # Last observed value at the issue time (23:00 of D-1).
    last_obs = y[idx.hour == 23]
    last_obs.index = last_obs.index.normalize() + one_day
    feats["last_obs"] = last_obs.reindex(day).to_numpy()
    # Day type of the lagged days, so the model can "undo" a holiday in a lag.
    dt_series = pd.Series(feats["day_type"].to_numpy(), index=idx)
    feats["day_type_lag24"] = dt_series.shift(24)
    feats["day_type_lag168"] = dt_series.shift(168)
    # Weather change w.r.t. the day used by lag_24: observed temperature yesterday.
    feats["temp_lag24"] = df["temp_c"].shift(24)
    feats["temp_delta24"] = feats["temp_fc"] - feats["temp_lag24"]
    return feats


def fourier_design(index: pd.DatetimeIndex, temp_fc: pd.Series, prevday_mean: pd.Series) -> pd.DataFrame:
    """Design matrix of the linear baseline.

    * daily Fourier terms (K=6) interacted with the three day types,
    * annual Fourier terms (K=3),
    * a piecewise-linear (hinge) basis in the 24h-smoothed temperature forecast
      with knots at 6, 12, 15, 18, 21, 24 deg C,
    * holiday / bridge / Christmas-period dummies,
    * the previous day's mean load (an ARX term that tracks slow level shifts).
    """
    cal = calendar_features(index)
    h = np.asarray(index.hour, dtype=float)
    doy = np.asarray(index.dayofyear, dtype=float)
    cols: dict[str, np.ndarray] = {}
    dtyp = cal["day_type"].to_numpy()
    for k in (0, 1, 2):
        ind = (dtyp == k).astype(float)
        cols[f"dt{k}"] = ind
        for j in range(1, 7):
            cols[f"dt{k}_s{j}"] = ind * np.sin(2 * np.pi * j * h / 24)
            cols[f"dt{k}_c{j}"] = ind * np.cos(2 * np.pi * j * h / 24)
    for j in range(1, 4):
        cols[f"ann_s{j}"] = np.sin(2 * np.pi * j * doy / 365.25)
        cols[f"ann_c{j}"] = np.cos(2 * np.pi * j * doy / 365.25)
    tsm = _ewm(temp_fc, 12.0).to_numpy()
    cols["temp"] = tsm
    for knot in (6.0, 12.0, 15.0, 18.0, 21.0, 24.0):
        cols[f"hinge_{knot:g}"] = np.maximum(0.0, tsm - knot)
    cols["is_bridge"] = cal["is_bridge"].to_numpy(float)
    cols["xmas"] = cal["xmas_period"].to_numpy(float)
    cols["prevday_mean"] = prevday_mean.to_numpy(float)
    return pd.DataFrame(cols, index=index)
