"""Slovenian public holidays (dela prosti dnevi) and calendar helpers.

Implemented from the Slovenian Public Holidays and Non-Working Days Act
(Zakon o praznikih in dela prostih dnevih v Republiki Sloveniji):

* fixed-date holidays (1 Jan, 8 Feb, 27 Apr, 1-2 May, 25 Jun, 15 Aug,
  31 Oct, 1 Nov, 25-26 Dec),
* 2 January, which was a working day in 2013-2016 and a holiday otherwise,
* the movable feasts Easter Sunday, Easter Monday and Whit Sunday
  (Pentecost, Easter + 49 days),
* the one-off "day of solidarity" on 14 August 2023 (after the floods).

Easter is computed with the anonymous Gregorian algorithm (Meeus/Jones/Butcher).
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable

import numpy as np
import pandas as pd

_FIXED: tuple[tuple[int, int, str], ...] = (
    (1, 1, "novo leto"),
    (2, 8, "Prešernov dan"),
    (4, 27, "dan upora proti okupatorju"),
    (5, 1, "praznik dela"),
    (5, 2, "praznik dela"),
    (6, 25, "dan državnosti"),
    (8, 15, "Marijino vnebovzetje"),
    (10, 31, "dan reformacije"),
    (11, 1, "dan spomina na mrtve"),
    (12, 25, "božič"),
    (12, 26, "dan samostojnosti in enotnosti"),
)

_ONE_OFF: dict[dt.date, str] = {dt.date(2023, 8, 14): "dan solidarnosti"}


def easter_sunday(year: int) -> dt.date:
    """Gregorian Easter Sunday (anonymous Gregorian / Meeus-Jones-Butcher algorithm)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    ell = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ell) // 451
    month, day = divmod(h + ell - 7 * m + 114, 31)
    return dt.date(year, month, day + 1)


def slovenian_holidays(years: Iterable[int]) -> dict[dt.date, str]:
    """Return ``{date: name}`` for all Slovenian public holidays in ``years``."""
    out: dict[dt.date, str] = {}
    for year in years:
        for month, day, name in _FIXED:
            out[dt.date(year, month, day)] = name
        if not 2013 <= year <= 2016:
            out[dt.date(year, 1, 2)] = "novo leto"
        easter = easter_sunday(year)
        out[easter] = "velikonočna nedelja"
        out[easter + dt.timedelta(days=1)] = "velikonočni ponedeljek"
        out[easter + dt.timedelta(days=49)] = "binkoštna nedelja"
        for date, name in _ONE_OFF.items():
            if date.year == year:
                out[date] = name
    return out


def holiday_mask(index: pd.DatetimeIndex) -> np.ndarray:
    """Boolean array: is the calendar day of each timestamp a public holiday?"""
    days = index.normalize()
    hol = slovenian_holidays(range(days.year.min(), days.year.max() + 1))
    hol_ts = pd.DatetimeIndex([pd.Timestamp(d) for d in hol])
    return np.asarray(days.isin(hol_ts))


def day_type(index: pd.DatetimeIndex) -> np.ndarray:
    """Day type per timestamp: 0 = working day, 1 = Saturday, 2 = Sunday or holiday."""
    dow = np.asarray(index.dayofweek)
    out = np.zeros(len(index), dtype=np.int8)
    out[dow == 5] = 1
    out[(dow == 6) | holiday_mask(index)] = 2
    return out


def bridge_mask(index: pd.DatetimeIndex) -> np.ndarray:
    """Bridge days: working days squeezed between a non-working day on both sides.

    E.g. a Friday after a Thursday holiday, or a Monday before a Tuesday holiday.
    Many people take these days off, which depresses load.
    """
    days = index.normalize()
    span = pd.date_range(days.min() - pd.Timedelta(days=1), days.max() + pd.Timedelta(days=1), freq="D")
    dtype = pd.Series(day_type(span), index=span)
    working = dtype == 0
    prev_off = (dtype.shift(1) != 0).fillna(False)
    next_off = (dtype.shift(-1) != 0).fillna(False)
    bridge = working & prev_off & next_off
    return np.asarray(bridge.reindex(days).to_numpy(), dtype=bool)
