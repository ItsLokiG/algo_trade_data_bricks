"""Intraday market-data access for NSE equities (yfinance 15m / 1h bars).

yfinance is an unofficial, free source: bars can lag and intraday volumes
are less reliable than daily ones.  Data is cached to CSV under
``data_cache/intraday/`` so backtests are reproducible and re-runnable
without re-downloading.

All timestamps are timezone-aware IST (Asia/Kolkata) bar-OPEN times.
NSE cash session: 09:15-15:30 IST.  Bars outside the session are dropped.

Verified Oct 2026: yfinance 15m data spans ~60 days max, 1h ~730 days.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")

# Repo root: src/nse_algo/intraday/data.py -> parents[3] = repo root
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CACHE_DIR = REPO_ROOT / "data_cache" / "intraday"

# NSE cash session (IST).  Bars with an open timestamp outside this window
# (e.g. a stray pre-open auction bar) are dropped.
SESSION_START_TIME = pd.Timestamp("09:15").time()
SESSION_END_TIME = pd.Timestamp("15:30").time()  # inclusive bound for bar opens


class IntradayDataError(Exception):
    """Raised when intraday data cannot be fetched or is unusable."""


def _to_ns_symbol(symbol: str) -> str:
    """Map an NSE symbol to Yahoo's .NS suffix form."""
    if symbol.startswith("^") or symbol.endswith(".NS"):
        return symbol
    return f"{symbol}.NS"


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten yfinance's MultiIndex (Price, Ticker) columns to lowercase."""
    if isinstance(df.columns, pd.MultiIndex):
        columns: list[str] = []
        for column in df.columns:
            field = next(
                (
                    str(part).lower().strip()
                    for part in column
                    if str(part).lower().strip() in {"open", "high", "low", "close", "volume"}
                ),
                "",
            )
            columns.append(field)
        df.columns = columns
    else:
        df.columns = [str(column).lower().strip() for column in df.columns]
    return df


def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a tz-aware DatetimeIndex in Asia/Kolkata, sorted, in-session."""
    df.index = pd.to_datetime(df.index, utc=True)
    df.index = df.index.tz_convert("Asia/Kolkata")
    df.index.name = "bar_time"
    df = df.sort_index()

    times = df.index.time
    in_session = (times >= SESSION_START_TIME) & (times <= SESSION_END_TIME)
    dropped = int((~in_session).sum())
    if dropped:
        logger.info("Dropping %d bars outside the NSE session", dropped)
        df = df[in_session]
    return df


def _validate_frame(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise IntradayDataError(f"{symbol}: missing columns {sorted(missing)}")
    df = df[list(REQUIRED_COLUMNS)]

    nan_count = int(df.isna().any(axis=1).sum())
    if nan_count:
        logger.warning("%s: dropping %d bars with NaN values", symbol, nan_count)
        df = df.dropna()
    if df.empty:
        raise IntradayDataError(f"{symbol}: no usable intraday bars")
    if not (df["high"] >= df[["open", "close", "low"]].max(axis=1)).all():
        raise IntradayDataError(f"{symbol}: high < max(o,c) in some bars")
    if not (df["low"] <= df[["open", "close", "high"]].min(axis=1)).all():
        raise IntradayDataError(f"{symbol}: low > min(o,c) in some bars")
    if (df["volume"] < 0).any():
        raise IntradayDataError(f"{symbol}: negative volume found")
    return df


def fetch_intraday(
    symbol: str,
    interval: str = "1h",
    period: str = "730d",
    *,
    cache_dir: Path | None = None,
    use_cache: bool = True,
    refresh: bool = False,
) -> pd.DataFrame:
    """Fetch NSE intraday OHLCV bars for *symbol* via yfinance.

    Returns a DataFrame indexed by IST bar-open time with lowercase
    open/high/low/close/volume columns.  Caches to CSV under
    ``data_cache/intraday/{SYMBOL}_{interval}.csv``.

    Raises IntradayDataError on any failure — an empty result is never
    treated as valid empty data.
    """
    cache_path = (cache_dir or DEFAULT_CACHE_DIR) / f"{symbol}_{interval}.csv"
    if use_cache and not refresh and cache_path.exists():
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True).tz_convert("Asia/Kolkata")
        df.index.name = "bar_time"
        return df

    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise IntradayDataError("yfinance is not installed") from exc

    yf_symbol = _to_ns_symbol(symbol)
    logger.info("Fetching %s intraday (%s, %s) from yfinance", yf_symbol, interval, period)
    try:
        raw = yf.download(
            yf_symbol,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
        )
    except Exception as exc:
        raise IntradayDataError(f"yfinance download failed for {symbol}: {exc}") from exc

    if raw is None or raw.empty:
        raise IntradayDataError(f"yfinance returned no intraday data for {symbol}")

    df = _flatten_columns(raw)
    df = _normalize_index(df)
    df = _validate_frame(df, symbol)

    if cache_dir is None:
        DEFAULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path)
    logger.info("Cached %d bars -> %s", len(df), cache_path)
    return df


def fetch_daily_context(
    symbol: str,
    lookback: str = "2y",
    *,
    cache_dir: Path | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch split/dividend-adjusted DAILY bars for HTF regime context.

    Used to compute the daily 20/50-SMA trend filter and the 20-day
    closing-high breakout flag (the latter from the day-trader-skill
    scanner).  Auto-adjusted prices keep SMA comparisons consistent.
    """
    cache_path = (cache_dir or DEFAULT_CACHE_DIR) / f"{symbol}_daily.csv"
    if use_cache and cache_path.exists():
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True).tz_convert("Asia/Kolkata")
        df.index.name = "date"
        return df

    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        raise IntradayDataError("yfinance is not installed") from exc

    try:
        raw = yf.download(
            _to_ns_symbol(symbol), period=lookback, interval="1d",
            auto_adjust=True, progress=False,
        )
    except Exception as exc:
        raise IntradayDataError(f"daily fetch failed for {symbol}: {exc}") from exc

    if raw is None or raw.empty:
        raise IntradayDataError(f"no daily data for {symbol}")

    df = _flatten_columns(raw)
    df = _normalize_index(df)
    df = _validate_frame(df, symbol)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path)
    return df


def session_groups(df: pd.DataFrame) -> dict[pd.Timestamp, pd.DataFrame]:
    """Group an intraday frame by trading date (session).

    Returns an ordered {session_date: bars} mapping.  Sessions with fewer
    than 2 bars are dropped (halts / partial data).
    """
    groups: dict[pd.Timestamp, pd.DataFrame] = {}
    for date, group in df.groupby(df.index.normalize()):
        if len(group) >= 2:
            groups[date] = group.sort_index()
    return groups


def session_dates(df: pd.DataFrame) -> np.ndarray:
    """Unique session (trading-day) dates in an intraday frame."""
    return df.index.normalize().unique()
