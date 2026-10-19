"""
Trading signal generation for the baseline swing strategy.

Strategy hypothesis: liquid stocks in established uptrends that pull back
and show renewed momentum may outperform the benchmark after costs.

This is a RESEARCH HYPOTHESIS, not a proven strategy.

Signals are generated using only data available at decision time.
A signal computed from the close of day T is actionable no earlier than
the next trading session.

Entry conditions (all must be true on the same bar):
  1. Close > SMA(long)  — established uptrend
  2. SMA(short) > SMA(long) — trend confirmation
  3. RSI was below oversold threshold recently (pullback)
  4. RSI has recovered above renewed threshold (momentum resumption)
  5. Volume ratio > minimum (confirmation)

Exit conditions (any one triggers exit):
  1. Close < SMA(short) — trend break
  2. Stop-loss hit (price <= stop)
  3. Trailing stop hit

Signal functions are pure — they take a DataFrame and return a DataFrame.
They do NOT depend on Spark, Databricks, or any external service.
The same functions are reusable by the backtester and paper engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from nse_algo.features.indicators import (
    atr,
    rsi,
    sma,
    volume_ratio,
)


class SignalType(Enum):
    """Signal action."""
    ENTRY = "entry"
    EXIT = "exit"
    HOLD = "hold"
    NONE = "none"


@dataclass(frozen=True)
class SignalConfig:
    """Parameters for signal generation."""
    sma_short_period: int = 50
    sma_long_period: int = 200
    rsi_period: int = 14
    rsi_oversold: float = 40.0
    rsi_renewed: float = 55.0
    rsi_lookback: int = 10  # bars to look back for oversold condition
    min_volume_ratio: float = 1.0
    atr_period: int = 14
    stop_atr_multiple: float = 2.0
    trailing_stop_atr_multiple: float = 2.0


def generate_entry_signals(
    df: pd.DataFrame,
    config: SignalConfig | None = None,
) -> pd.Series:
    """Generate boolean entry signals for each bar.

    Returns a boolean Series: True where all entry conditions are met.
    Uses only data available up to and including the current bar's close.
    """
    cfg = config or SignalConfig()

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # Indicators
    sma_s = sma(close, cfg.sma_short_period)
    sma_l = sma(close, cfg.sma_long_period)
    rsi_val = rsi(close, cfg.rsi_period)
    vol_ratio = volume_ratio(volume, period=20)

    # Condition 1: Close above long SMA (uptrend)
    uptrend = close > sma_l

    # Condition 2: Short SMA above long SMA (trend confirmation)
    trend_confirm = sma_s > sma_l

    # Condition 3: RSI was oversold in the last rsi_lookback bars (pullback)
    rsi_was_oversold = rsi_val.rolling(
        window=cfg.rsi_lookback, min_periods=1
    ).min() <= cfg.rsi_oversold

    # Condition 4: RSI has recovered above renewed threshold
    rsi_renewed = rsi_val >= cfg.rsi_renewed

    # Condition 5: Volume confirmation
    volume_ok = vol_ratio >= cfg.min_volume_ratio

    # All conditions must be true
    entry = uptrend & trend_confirm & rsi_was_oversold & rsi_renewed & volume_ok

    # Don't signal on the first sma_long_period bars (not enough data)
    entry.iloc[:cfg.sma_long_period] = False

    return entry.astype(bool)


def generate_exit_signals(
    df: pd.DataFrame,
    entry_price: float | pd.Series | None = None,
    entry_atr: float | pd.Series | None = None,
    config: SignalConfig | None = None,
) -> pd.Series:
    """Generate boolean exit signals.

    Exit when:
      1. Close < SMA(short) — trend break
      2. Low <= stop price (stop-loss)

    Parameters
    ----------
    df : DataFrame
        OHLCV data.
    entry_price : float or Series, optional
        Entry price(s) for stop calculation. If None, only trend-break exit.
    entry_atr : float or Series, optional
        ATR at entry for stop distance. Required if entry_price is given.
    config : SignalConfig, optional
        Signal parameters.
    """
    cfg = config or SignalConfig()
    close = df["close"]
    low = df["low"]

    sma_s = sma(close, cfg.sma_short_period)

    # Trend-break exit
    trend_break = close < sma_s

    # Stop-loss exit
    if entry_price is not None and entry_atr is not None:
        stop_price = entry_price - (entry_atr * cfg.stop_atr_multiple)
        stop_hit = low <= stop_price
        exit_signal = trend_break | stop_hit
    else:
        exit_signal = trend_break

    # Don't exit before we have enough data for SMA
    exit_signal.iloc[:cfg.sma_short_period] = False

    return exit_signal.astype(bool)


def generate_all_signals(
    df: pd.DataFrame,
    config: SignalConfig | None = None,
) -> pd.DataFrame:
    """Generate a full signal DataFrame with entry, exit, and metadata.

    Returns a DataFrame with columns:
      - signal_entry: bool
      - signal_exit: bool
      - ind_rsi: RSI at signal time
      - ind_atr: ATR at signal time
      - proposed_stop: stop price if entering at this bar's close
      - signal_type: SignalType (entry/exit/hold/none)
    """
    cfg = config or SignalConfig()
    close = df["close"]
    high = df["high"]
    low = df["low"]

    atr_val = atr(high, low, close, cfg.atr_period)
    rsi_val = rsi(close, cfg.rsi_period)

    entries = generate_entry_signals(df, cfg)
    exits = generate_exit_signals(df, config=cfg)

    # Proposed stop for each bar
    proposed_stop = close - (atr_val * cfg.stop_atr_multiple)

    # Signal type: entry takes priority over exit if both fire
    signal_type = pd.Series(SignalType.NONE, index=df.index, dtype=object)
    signal_type[exits] = SignalType.EXIT
    signal_type[entries] = SignalType.ENTRY

    result = df.copy()
    result["signal_entry"] = entries
    result["signal_exit"] = exits
    result["ind_rsi"] = rsi_val
    result["ind_atr"] = atr_val
    result["proposed_stop"] = proposed_stop
    result["signal_type"] = signal_type

    return result
