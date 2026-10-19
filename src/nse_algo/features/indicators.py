"""
Technical indicators for NSE equity research.

All indicators are pure Pandas/NumPy — no Spark, no Databricks dependency.
This keeps strategy logic portable to VS Code.

Implemented:
  - SMA (Simple Moving Average)
  - RSI (Relative Strength Index, Wilder's smoothing)
  - ATR (Average True Range, Wilder's smoothing)
  - Volume average and ratio
  - Momentum / return over a lookback window
  - Relative strength vs benchmark

All functions take a DataFrame with OHLCV columns and return a Series
or DataFrame of indicator values.  They do NOT modify the input.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────
#  Moving averages
# ──────────────────────────────────────────────

def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average.

    Returns NaN for the first ``period - 1`` bars.
    """
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


# ──────────────────────────────────────────────
#  RSI (Wilder's smoothing)
# ──────────────────────────────────────────────

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index using Wilder's smoothing.

    RSI = 100 - (100 / (1 + RS))
    RS = average_gain / average_loss

    Returns a Series bounded [0, 100].
    First ``period`` values are NaN.
    """
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")

    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    # Wilder's smoothing = EMA with alpha = 1/period
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi_val = 100.0 - (100.0 / (1.0 + rs))

    # When avg_loss is 0 (no losses), RSI = 100
    rsi_val = rsi_val.fillna(100.0).where(avg_loss > 0, 100.0)
    # When avg_gain is 0 and avg_loss > 0, RSI = 0
    rsi_val = rsi_val.where(avg_gain > 0, 0.0)
    # First period values should be NaN
    rsi_val.iloc[:period] = np.nan

    return rsi_val


# ──────────────────────────────────────────────
#  ATR (Average True Range, Wilder's smoothing)
# ──────────────────────────────────────────────

def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Average True Range using Wilder's smoothing.

    TR = max(high - low, |high - prev_close|, |low - prev_close|)
    ATR = Wilder's EMA of TR.

    Returns NaN for the first ``period`` bars.
    """
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Wilder's smoothing
    atr_val = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    atr_val.iloc[:period] = np.nan

    return atr_val


# ──────────────────────────────────────────────
#  Volume indicators
# ──────────────────────────────────────────────

def volume_sma(volume: pd.Series, period: int = 20) -> pd.Series:
    """Simple moving average of volume."""
    return sma(volume, period)


def volume_ratio(volume: pd.Series, period: int = 20) -> pd.Series:
    """Current volume / average volume ratio.

    Values > 1 mean above-average volume.
    """
    avg = volume_sma(volume, period)
    return volume / avg.replace(0.0, np.nan)


# ──────────────────────────────────────────────
#  Momentum / returns
# ──────────────────────────────────────────────

def momentum(close: pd.Series, period: int = 20) -> pd.Series:
    """Return over the last *period* bars: (close / close.shift(period)) - 1."""
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    return close.pct_change(periods=period)


def daily_return(close: pd.Series) -> pd.Series:
    """Day-over-day percentage return."""
    return close.pct_change()


# ──────────────────────────────────────────────
#  Relative strength vs benchmark
# ──────────────────────────────────────────────

def relative_strength(
    close: pd.Series,
    benchmark_close: pd.Series,
    period: int = 60,
) -> pd.Series:
    """Relative strength = stock return / benchmark return over *period*.

    Values > 1 mean the stock outperformed the benchmark.
    """
    stock_ret = close.pct_change(periods=period)
    bench_ret = benchmark_close.pct_change(periods=period)
    return stock_ret / bench_ret.replace(0.0, np.nan)


# ──────────────────────────────────────────────
#  All-in-one feature builder
# ──────────────────────────────────────────────

def add_all_indicators(
    df: pd.DataFrame,
    *,
    sma_short: int = 50,
    sma_long: int = 200,
    rsi_period: int = 14,
    atr_period: int = 14,
    volume_avg_period: int = 20,
    momentum_period: int = 20,
    benchmark_close: pd.Series | None = None,
) -> pd.DataFrame:
    """Add all standard indicators to an OHLCV DataFrame.

    Returns a new DataFrame — the input is not modified.
    Indicator columns are prefixed with ``ind_``.
    """
    result = df.copy()

    result["ind_sma_short"] = sma(result["close"], sma_short)
    result["ind_sma_long"] = sma(result["close"], sma_long)
    result["ind_rsi"] = rsi(result["close"], rsi_period)
    result["ind_atr"] = atr(result["high"], result["low"], result["close"], atr_period)
    result["ind_volume_sma"] = volume_sma(result["volume"], volume_avg_period)
    result["ind_volume_ratio"] = volume_ratio(result["volume"], volume_avg_period)
    result["ind_momentum"] = momentum(result["close"], momentum_period)

    if benchmark_close is not None:
        result["ind_rel_strength"] = relative_strength(
            result["close"], benchmark_close, period=60
        )

    return result
