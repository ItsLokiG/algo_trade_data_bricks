"""Intraday technical indicators: session VWAP, ATR, RSI, EMA.

All functions are pure pandas: no network, no state.  Session-scoped
calculations (VWAP, opening range) reset at each trading day so no
information leaks across sessions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def session_vwap(df: pd.DataFrame) -> pd.Series:
    """Volume-weighted average price, resetting each session.

    Uses typical price ((H+L+C)/3).  Bars with zero volume get the
    running VWAP carried forward (they contribute nothing).  Returns a
    Series aligned to *df.index*; the value at bar i uses bars <= i only.
    """
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    volume = df["volume"].astype(float)

    session = df.index.normalize()
    tpv = typical * volume

    grouped_tpv = pd.Series(tpv.values, index=session).groupby(level=0).cumsum()
    grouped_vol = pd.Series(volume.values, index=session).groupby(level=0).cumsum()

    with np.errstate(invalid="ignore", divide="ignore"):
        vwap = grouped_tpv / grouped_vol
    vwap = vwap.replace([np.inf, -np.inf], np.nan).ffill()
    vwap.index = df.index
    return vwap


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range on continuous intraday bars.

    True range spans the overnight gap (uses the previous bar's close even
    across sessions) — deliberately conservative for stop distances.
    Simple rolling mean (research convention; Wilder smoothing is a
    configurable future refinement).
    """
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period, min_periods=1).mean()


def rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder RSI on the intraday close series."""
    delta = df["close"].diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    return out.fillna(50.0)


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average."""
    return series.ewm(span=period, adjust=False).mean()


def opening_range(
    df: pd.DataFrame,
    session_start: str = "09:15",
    or_minutes: int = 30,
) -> pd.DataFrame:
    """Compute the per-session opening range (OR high / low).

    Rule (adapted from day-trader-skill trader-entry): the opening range
    is the high/low of all bars whose OPEN timestamp falls within
    session_start + or_minutes.  Values are forward-filled across the rest
    of that session only (a bar in session S never sees session S-1's OR).

    Returns *df* with added columns ``or_high``, ``or_low``, ``or_mid``.
    """
    start = pd.Timestamp(session_start).time()
    end = (pd.Timestamp(session_start) + pd.Timedelta(minutes=or_minutes)).time()

    or_high = {}
    or_low = {}
    for date, group in df.groupby(df.index.normalize()):
        times = group.index.time
        in_range = (times >= start) & (times < end)
        or_bars = group[in_range]
        or_high[date] = float(or_bars["high"].max()) if len(or_bars) else np.nan
        or_low[date] = float(or_bars["low"].min()) if len(or_bars) else np.nan

    high_series = pd.Series(or_high)
    low_series = pd.Series(or_low)

    df = df.copy()
    session_dates = df.index.normalize()
    df["or_high"] = high_series.reindex(session_dates).values
    df["or_low"] = low_series.reindex(session_dates).values
    df["or_mid"] = (df["or_high"] + df["or_low"]) / 2.0
    return df


def session_equilibrium(df: pd.DataFrame) -> pd.Series:
    """Running 50% level of the session range so far (SMC premium/discount).

    At bar i: (running session high + running session low) / 2, using
    bars <= i only.  Below equilibrium = discount; above = premium.
    """
    session = df.index.normalize()
    running_high = (
        pd.Series(df["high"].values, index=session).groupby(level=0).cummax()
    )
    running_low = (
        pd.Series(df["low"].values, index=session).groupby(level=0).cummin()
    )
    eq = (running_high + running_low) / 2.0
    eq.index = df.index
    return eq
