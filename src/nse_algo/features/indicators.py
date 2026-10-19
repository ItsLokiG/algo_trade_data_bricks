"""
Technical indicators for NSE equity research.

All indicators are pure Pandas/NumPy — no Spark, no Databricks dependency.
This keeps strategy logic portable to VS Code.

Implemented:
  - SMA (Simple Moving Average)
  - EMA (Exponential Moving Average)
  - RSI (Relative Strength Index, Wilder's smoothing)
  - ATR (Average True Range, Wilder's smoothing)
  - MACD (Moving Average Convergence Divergence)
  - ADX (Average Directional Index, trend strength)
  - Bollinger Bands (volatility envelope)
  - Stochastic Oscillator
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
#  MACD (Moving Average Convergence Divergence)
# ──────────────────────────────────────────────

def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD line, signal line, and histogram.

    Returns (macd_line, signal_line, histogram).
    MACD line = EMA(fast) - EMA(slow).
    Signal line = EMA(MACD, signal).
    Histogram = MACD - Signal.
    """
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


# ──────────────────────────────────────────────
#  ADX (Average Directional Index)
# ──────────────────────────────────────────────

def adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Average Directional Index — trend strength indicator.

    ADX > 25 indicates a strong trend.
    Returns NaN for the first ``period * 2`` bars.
    """
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")

    prev_high = high.shift(1)
    prev_low = low.shift(1)

    up_move = high - prev_high
    down_move = prev_low - low

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=high.index, dtype=float,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=high.index, dtype=float,
    )

    # True Range
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Wilder's smoothing
    atr_val = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    plus_dm_s = plus_dm.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    minus_dm_s = minus_dm.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    plus_di = 100 * plus_dm_s / atr_val.replace(0.0, np.nan)
    minus_di = 100 * minus_dm_s / atr_val.replace(0.0, np.nan)

    di_sum = (plus_di + minus_di).replace(0.0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / di_sum
    adx_val = dx.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    return adx_val


# ──────────────────────────────────────────────
#  Bollinger Bands
# ──────────────────────────────────────────────

def bollinger_bands(
    close: pd.Series,
    period: int = 20,
    std_dev: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands.

    Returns (upper_band, middle_band, lower_band).
    Middle = SMA(period), Upper/Lower = Middle ± std_dev * rolling_std.
    """
    middle = sma(close, period)
    std = close.rolling(window=period, min_periods=period).std()
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    return upper, middle, lower


# ──────────────────────────────────────────────
#  Stochastic Oscillator
# ──────────────────────────────────────────────

def stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3,
) -> tuple[pd.Series, pd.Series]:
    """Stochastic Oscillator.

    Returns (%K, %D).
    %K = 100 * (close - lowest_low) / (highest_high - lowest_low)
    %D = SMA(%K, d_period)
    """
    lowest_low = low.rolling(window=k_period, min_periods=k_period).min()
    highest_high = high.rolling(window=k_period, min_periods=k_period).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0.0, np.nan)
    d = k.rolling(window=d_period, min_periods=d_period).mean()
    return k, d


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

    # MACD
    macd_line, macd_signal, macd_hist = macd(result["close"])
    result["ind_macd"] = macd_line
    result["ind_macd_signal"] = macd_signal
    result["ind_macd_hist"] = macd_hist

    # ADX
    result["ind_adx"] = adx(result["high"], result["low"], result["close"], 14)

    # Bollinger Bands
    bb_upper, bb_middle, bb_lower = bollinger_bands(result["close"])
    result["ind_bb_upper"] = bb_upper
    result["ind_bb_lower"] = bb_lower

    # Stochastic
    k, d = stochastic(result["high"], result["low"], result["close"])
    result["ind_stoch_k"] = k
    result["ind_stoch_d"] = d

    if benchmark_close is not None:
        result["ind_rel_strength"] = relative_strength(
            result["close"], benchmark_close, period=60
        )

    return result
