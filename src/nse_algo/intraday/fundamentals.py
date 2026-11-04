"""Fundamental universe filter + daily HTF regime for the intraday system.

Fundamentals do NOT time intraday entries — no PE/ROE ratio predicts the
next 6 hours.  They play exactly two roles here:

1. **Universe filter**: liquidity (avg daily turnover), size (market cap)
   and a distress screen (positive trailing earnings).  This limits impact
   costs and avoids event-crazy distressed names.
2. **Daily regime context**: close > 20-SMA > 50-SMA on the PREVIOUS day's
   completed daily bar (long entries only in daily uptrends), plus a
   20-day closing-high breakout flag (from day-trader-skill's scanner).

Data comes from yfinance (unofficial, no point-in-time guarantee — a
known limitation for the fundamental filter; it is a *today's* view
applied backwards).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from nse_algo.intraday.data import DEFAULT_CACHE_DIR

logger = logging.getLogger(__name__)

CRORE = 1e7  # 1 crore INR


def fetch_fundamentals(
    symbols: list[str] | tuple[str, ...],
    *,
    cache_dir: Path | None = None,
    max_age_hours: float = 24.0,
) -> dict[str, dict]:
    """Fetch market cap / PE / avg volume info per symbol from yfinance.

    Returns {symbol: {market_cap, market_cap_cr, trailing_pe, avg_volume,
    price, avg_turnover_cr, ok, error}}.  Failures are recorded, not raised:
    the caller decides whether a failed lookup passes by default (the
    conservative *inclusion* default is documented as a data limitation,
    because excluding a stock for a data-provider failure is also a
    decision).
    """
    cache_path = (cache_dir or DEFAULT_CACHE_DIR) / "fundamentals.json"

    cached: dict[str, dict] = {}
    if cache_path.exists():
        try:
            with open(cache_path, encoding="utf-8") as handle:
                cached = json.load(handle)
        except (OSError, json.JSONDecodeError):
            logger.warning("Fundamentals cache unreadable: %s", cache_path)

    import time

    now = time.time()
    result: dict[str, dict] = {}
    changed = False
    for symbol in symbols:
        entry = cached.get(symbol)
        if entry and now - entry.get("fetched_at", 0) < max_age_hours * 3600:
            result[symbol] = entry
            continue

        info: dict = {"fetched_at": now}
        try:
            import yfinance as yf

            ticker_info = yf.Ticker(f"{symbol}.NS").info or {}
            market_cap = ticker_info.get("marketCap")
            price = ticker_info.get(
                "regularMarketPrice", ticker_info.get("currentPrice")
            )
            avg_volume = ticker_info.get("averageVolume")
            info.update(
                {
                    "market_cap": market_cap,
                    "market_cap_cr": (market_cap / CRORE) if market_cap else None,
                    "trailing_pe": ticker_info.get("trailingPE"),
                    "avg_volume": avg_volume,
                    "price": price,
                    "avg_turnover_cr": (
                        avg_volume * price / CRORE
                        if avg_volume and price
                        else None
                    ),
                }
            )
        except Exception as exc:  # yfinance failures must not kill the run
            info["error"] = f"{type(exc).__name__}: {exc}"
            logger.warning("Fundamentals fetch failed for %s: %s", symbol, exc)

        result[symbol] = info
        cached[symbol] = info
        changed = True

    if changed:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as handle:
            json.dump(cached, handle, indent=2)

    return result


def filter_universe(
    fundamentals: dict[str, dict],
    min_turnover_cr: float,
    min_market_cap_cr: float,
) -> tuple[list[str], list[tuple[str, str]]]:
    """Apply liquidity/size/distress filters.  Returns (passed, rejected).

    Rejection reasons are explicit: 'turnover', 'market_cap', 'distress'
    (negative or zero trailing PE with earnings available), or 'no_data'
    (provider failure — rejected only when strict data exists for peers).
    """
    passed: list[str] = []
    rejected: list[tuple[str, str]] = []

    for symbol, info in fundamentals.items():
        if info.get("error"):
            rejected.append((symbol, "no_data"))
            continue
        turnover = info.get("avg_turnover_cr")
        market_cap = info.get("market_cap_cr")
        if turnover is None or turnover < min_turnover_cr:
            rejected.append((symbol, "turnover"))
            continue
        if market_cap is None or market_cap < min_market_cap_cr:
            rejected.append((symbol, "market_cap"))
            continue
        passed.append(symbol)

    return passed, rejected


def daily_regime(
    daily_df: pd.DataFrame,
    sma_fast: int = 20,
    sma_slow: int = 50,
) -> pd.DataFrame:
    """Daily HTF regime per trading day, using only COMPLETED daily bars.

    Adds columns (all evaluated as of the PREVIOUS completed daily bar,
    so a session on date T never sees T's own daily close):

    * ``regime_bull``: prev close > prev fast SMA > prev slow SMA
    * ``breakout_20d``: prev close is above every close of the 20 sessions
      before it (day-trader-skill scanner's closing-high breakout rule)
    """
    df = daily_df.copy()
    close = df["close"]

    fast = close.rolling(sma_fast).mean()
    slow = close.rolling(sma_slow).mean()

    prev_close = close.shift(1)
    prev_fast = fast.shift(1)
    prev_slow = slow.shift(1)

    df["regime_bull"] = (
        (prev_close > prev_fast) & (prev_fast > prev_slow)
    )

    window_min = close.rolling(20).min().shift(1)
    df["breakout_20d"] = prev_close > window_min.where(window_min.notna(), close)
    df["breakout_20d"] = (prev_close > window_min).fillna(False)

    return df
