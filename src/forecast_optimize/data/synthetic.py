"""Seeded synthetic generator for hourly electricity load, temperature and prices.

The generating model is documented in ``docs/data_model.md``; the symbols used
below match that document.  Summary (t = hour index, h = hour of day,
s = day of year, k = day type):

* temperature  T_t = T0 - A cos(2pi (s - s0)/365.25) + D(s) cos(2pi (h - 15)/24) + Z_t + nu_t,
  with Z a daily AR(1) "weather regime" anomaly (linearly interpolated to hours)
  and nu an hourly AR(1);
* day-ahead temperature forecast  T^fc_t = T_t + e_t,  e an hourly AR(1);
* smoothed (felt) temperature  T~_t = EWMA_lambda(T)_t   (thermal inertia of buildings);
* load  y_t = m_t + sigma_t * eta_t, where

      m_t = mu(t) L_d * P_k(h) * F_t * G_t + beta_H * HDD_t + beta_C * c(h) * CDD_t,
      (L_d a slow daily log-AR(1) activity level, G_t a winter-evening lighting factor),
      HDD_t = max(0, T_H - T~_t),  CDD_t = max(0, T~_t - T_C),
      sigma_t = s0 * m_t * (1 + kappa * (HDD_t + CDD_t)),

  eta a unit-variance AR(1) and F_t the product of special-day factors
  (bridge days, Christmas/New Year period, August holidays, winter lighting).
* day-ahead price  pi_t = a_d + b * shape(h) + c * (m_t - mean m) + noise.

Everything is driven by a single ``numpy.random.Generator`` so a seed fully
determines the output.  Time stamps are naive local standard time (CET, UTC+1);
daylight saving time is ignored so that every day has exactly 24 hours.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from forecast_optimize.calendar_si import bridge_mask, day_type, holiday_mask

HOURS_PER_YEAR = 365.25 * 24


@dataclass(frozen=True)
class SyntheticConfig:
    """Parameters of the generating model (defaults resemble a Slovenian distribution area)."""

    start: str = "2020-01-01"
    end: str = "2024-12-31 23:00"
    seed: int = 7
    # temperature (deg C)
    temp_mean: float = 11.0
    temp_annual_amp: float = 10.5
    temp_coldest_doy: float = 20.0
    temp_diurnal_winter: float = 3.0
    temp_diurnal_summer: float = 6.0
    temp_regime_phi: float = 0.8  # daily AR(1) coefficient
    temp_regime_sd: float = 3.0  # stationary s.d. of daily anomaly
    temp_hourly_phi: float = 0.7
    temp_hourly_sd: float = 0.5
    temp_fc_phi: float = 0.95
    temp_fc_sd: float = 1.2  # stationary s.d. of day-ahead forecast error
    thermal_halflife_h: float = 12.0
    # load (MW)
    base_load: float = 1000.0
    annual_growth: float = 0.01
    heat_threshold: float = 15.0  # T_H
    cool_threshold: float = 21.0  # T_C
    beta_heat: float = 9.0  # MW per heating degree
    beta_cool: float = 22.0  # MW per cooling degree (at the afternoon peak)
    noise_rel_sd: float = 0.022  # s0
    noise_weather_kappa: float = 0.03  # kappa
    noise_phi: float = 0.9
    level_rel_sd: float = 0.025  # s.d. of the slow multiplicative level factor L_d
    level_phi: float = 0.97  # daily AR(1) coefficient of log L_d
    # prices (EUR/MWh)
    price_level: float = 95.0
    price_level_sd: float = 18.0
    price_level_phi: float = 0.85
    price_shape_amp: float = 28.0
    price_load_coef: float = 0.12
    price_noise_sd: float = 6.0


def _ar1(rng: np.random.Generator, n: int, phi: float, sd: float) -> np.ndarray:
    """Stationary Gaussian AR(1) path with marginal standard deviation ``sd``."""
    out = np.empty(n)
    innov_sd = sd * np.sqrt(1.0 - phi**2)
    eps = rng.standard_normal(n)
    out[0] = sd * eps[0]
    for i in range(1, n):
        out[i] = phi * out[i - 1] + innov_sd * eps[i]
    return out


def _bump(h: np.ndarray, centre: float, width: float) -> np.ndarray:
    """Periodic Gaussian bump on the 24-hour circle."""
    d = (h - centre + 12.0) % 24.0 - 12.0
    return np.exp(-0.5 * (d / width) ** 2)


def daily_profile(h: np.ndarray, dtype: np.ndarray) -> np.ndarray:
    """Normalised intraday profile P_k(h) for day types k = 0 (work), 1 (Sat), 2 (Sun/holiday)."""
    h = h.astype(float)
    rise = {0: 6.5, 1: 8.0, 2: 8.5}
    level = {0: 1.0, 1: 0.86, 2: 0.78}
    out = np.empty_like(h)
    for k in (0, 1, 2):
        m = dtype == k
        hk = h[m]
        # occupancy: logistic rise in the morning, fall late in the evening
        occ = 1.0 / (1.0 + np.exp(-(hk - rise[k]) / 0.9)) - 1.0 / (1.0 + np.exp(-(hk - 22.5) / 1.0))
        shape = 0.72 + 0.26 * occ + 0.05 * _bump(hk, 11.0, 2.0) + 0.07 * _bump(hk, 19.5, 1.8)
        out[m] = level[k] * shape
    return out


def generate(config: SyntheticConfig | None = None) -> pd.DataFrame:
    """Generate the synthetic data set.

    Returns a DataFrame indexed by hourly ``ts`` with columns
    ``load_mw``, ``temp_c``, ``temp_fc_c``, ``da_price``, ``load_mean_mw``
    (the deterministic part m_t, useful for diagnostics) and ``load_sd_mw`` (sigma_t).
    """
    cfg = config or SyntheticConfig()
    rng = np.random.default_rng(cfg.seed)
    idx = pd.date_range(cfg.start, cfg.end, freq="h", name="ts")
    n = len(idx)
    h = np.asarray(idx.hour, dtype=float)
    doy = np.asarray(idx.dayofyear, dtype=float)
    t_years = np.arange(n) / HOURS_PER_YEAR
    ann = np.cos(2 * np.pi * (doy - cfg.temp_coldest_doy) / 365.25)  # +1 in late January

    # --- temperature -------------------------------------------------------
    days = idx.normalize().unique()
    z_day = _ar1(rng, len(days) + 1, cfg.temp_regime_phi, cfg.temp_regime_sd)
    # daily anomaly anchored at noon, linearly interpolated to hours
    day_pos = np.arange(len(days) + 1) * 24.0 + 12.0
    z = np.interp(np.arange(n), day_pos, z_day)
    diurnal_amp = (
        0.5 * (cfg.temp_diurnal_winter + cfg.temp_diurnal_summer)
        - 0.5 * (cfg.temp_diurnal_summer - cfg.temp_diurnal_winter) * ann
    )
    temp = (
        cfg.temp_mean
        - cfg.temp_annual_amp * ann
        + diurnal_amp * np.cos(2 * np.pi * (h - 15.0) / 24.0)
        + z
        + _ar1(rng, n, cfg.temp_hourly_phi, cfg.temp_hourly_sd)
    )
    temp_fc = temp + _ar1(rng, n, cfg.temp_fc_phi, cfg.temp_fc_sd)
    lam = 1.0 - 0.5 ** (1.0 / cfg.thermal_halflife_h)
    temp_smooth = pd.Series(temp).ewm(alpha=lam, adjust=False).mean().to_numpy()

    # --- load ----------------------------------------------------------------
    dtype = day_type(idx)
    special = np.ones(n)
    special[bridge_mask(idx)] *= 0.92
    month = np.asarray(idx.month)
    dom = np.asarray(idx.day)
    xmas = ((month == 12) & (dom >= 24)) | ((month == 1) & (dom <= 6))
    special[xmas] *= 0.93
    special[(month == 8) & (dom >= 1) & (dom <= 25)] *= 0.95
    # winter evening lighting peak
    lighting = 1.0 + 0.04 * 0.5 * (1.0 + np.cos(2 * np.pi * (doy - 355.0) / 365.25)) * _bump(h, 18.0, 1.5)
    day_ix = (np.arange(n) // 24).astype(int)
    # slow, unobserved activity level (economy, industry schedules): daily AR(1) in logs
    level = np.exp(_ar1(rng, len(days), cfg.level_phi, cfg.level_rel_sd))[day_ix]
    mu = cfg.base_load * (1.0 + cfg.annual_growth * t_years) * level
    hdd = np.maximum(0.0, cfg.heat_threshold - temp_smooth)
    cdd = np.maximum(0.0, temp_smooth - cfg.cool_threshold)
    cool_shape = 0.6 + 0.4 * _bump(h, 15.0, 4.0)
    m = mu * daily_profile(h, dtype) * special * lighting + cfg.beta_heat * hdd + cfg.beta_cool * cool_shape * cdd
    holiday_like = holiday_mask(idx) | bridge_mask(idx)
    sigma = cfg.noise_rel_sd * m * (1.0 + cfg.noise_weather_kappa * (hdd + cdd)) * (1.0 + 0.5 * holiday_like)
    load = m + sigma * _ar1(rng, n, cfg.noise_phi, 1.0)

    # --- day-ahead prices ------------------------------------------------------
    level_day = cfg.price_level + _ar1(rng, len(days), cfg.price_level_phi, cfg.price_level_sd)
    # morning and evening peaks, midday solar dip (stronger in summer)
    solar = 0.5 * (1.0 - ann)
    shape = _bump(h, 8.0, 1.8) + 1.2 * _bump(h, 19.5, 2.0) - (0.4 + 0.8 * solar) * _bump(h, 13.0, 2.5)
    price = (
        level_day[day_ix]
        + cfg.price_shape_amp * shape
        + cfg.price_load_coef * (m - m.mean())
        + rng.normal(0.0, cfg.price_noise_sd, n)
    )
    price = np.maximum(price, 5.0)

    return pd.DataFrame(
        {
            "load_mw": load,
            "temp_c": temp,
            "temp_fc_c": temp_fc,
            "da_price": price,
            "load_mean_mw": m,
            "load_sd_mw": sigma,
        },
        index=idx,
    )
