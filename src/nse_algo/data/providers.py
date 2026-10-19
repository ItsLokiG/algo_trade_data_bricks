"""
Data provider interface for NSE market data.

Defines a replaceable provider abstraction so the ingestion layer
can swap data sources without changing strategy or backtest code.

yfinance is used for initial prototyping IF available.  It is an
unofficial source and NOT a guaranteed exchange-quality feed.

A SyntheticDataProvider is included for testing — it generates
realistic-looking OHLCV bars with known properties so that tests
have deterministic data without internet access.
"""

from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Protocol

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
#  Provider protocol
# ──────────────────────────────────────────────

class DataProvider(ABC):
    """Abstract base for market-data providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable provider name."""

    @property
    @abstractmethod
    def is_official(self) -> bool:
        """True if the source is an official exchange feed."""

    @abstractmethod
    def fetch_ohlcv(
        self,
        symbol: str,
        start_date: str,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Download daily OHLCV bars for *symbol*.

        Returns a DataFrame indexed by date with columns:
        open, high, low, close, volume — all lowercase.

        Raises ProviderError on failure.  Never returns empty
        or partially-formed data on success.
        """


class ProviderError(Exception):
    """Raised when a data provider fails to return valid data."""


# ──────────────────────────────────────────────
#  yfinance provider (optional)
# ──────────────────────────────────────────────

class YFinanceProvider(DataProvider):
    """yfinance-based provider for NSE equities.

    yfinance is an unofficial source.  Results are NOT guaranteed
to be exchange-quality.  Use only for prototyping.
    """

    def __init__(self) -> None:
        try:
            import yfinance as yf  # noqa: F401
            self._yf = yf
            self._available = True
        except ImportError:
            self._yf = None
            self._available = False
            logger.warning(
                "yfinance is not installed. "
                "Install with: pip install yfinance"
            )

    @property
    def name(self) -> str:
        return "yfinance"

    @property
    def is_official(self) -> bool:
        return False

    def is_available(self) -> bool:
        return self._available

    def fetch_ohlcv(
        self,
        symbol: str,
        start_date: str,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        if not self._available:
            raise ProviderError(
                "yfinance is not installed. Run: pip install yfinance"
            )

        # NSE symbols on yfinance need .NS suffix
        yf_symbol = f"{symbol}.NS" if not symbol.startswith("^") else symbol

        logger.info("Fetching %s from yfinance (%s → %s)", yf_symbol, start_date, end_date)

        try:
            df = self._yf.download(
                yf_symbol,
                start=start_date,
                end=end_date,
                auto_adjust=True,
                progress=False,
            )
        except Exception as exc:
            raise ProviderError(f"yfinance download failed for {symbol}: {exc}") from exc

        if df is None or df.empty:
            raise ProviderError(
                f"yfinance returned empty data for {symbol}. "
                "Do NOT treat this as valid empty data."
            )

        # Normalise column names to lowercase
        # Handle yfinance MultiIndex (yfinance >= 1.7.0 returns (Price, Ticker) tuples)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [str(c).lower().strip() for c in df.columns]

        # Keep only OHLCV columns
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ProviderError(
                f"yfinance data for {symbol} missing columns: {missing}"
            )

        df = df[list(required)]

        # Ensure DatetimeIndex
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"

        # Sort chronologically
        df = df.sort_index()

        # Drop any rows with NaN in any OHLCV column
        nan_count = df.isna().any(axis=1).sum()
        if nan_count > 0:
            logger.warning(
                "Dropping %d rows with NaN values for %s", nan_count, symbol
            )
            df = df.dropna(subset=list(required))

        if df.empty:
            raise ProviderError(f"All rows had NaN for {symbol}")

        return df


# ──────────────────────────────────────────────
#  Synthetic data provider (for testing)
# ──────────────────────────────────────────────

class SyntheticProvider(DataProvider):
    """Generates deterministic synthetic OHLCV bars for testing.

    Uses a seeded random walk so tests are reproducible.
    """

    def __init__(self, seed: int = 42) -> None:
        self._seed = seed

    @property
    def name(self) -> str:
        return "synthetic"

    @property
    def is_official(self) -> bool:
        return False

    def fetch_ohlcv(
        self,
        symbol: str,
        start_date: str,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        rng = np.random.default_rng(self._seed + hash(symbol) % 100000)

        dates = pd.bdate_range(start=start_date, end=end_date or pd.Timestamp.today())
        n = len(dates)
        if n == 0:
            raise ProviderError(f"No business days in range {start_date} → {end_date}")

        # Random walk for close prices — start at ₹100
        returns = rng.normal(0.0005, 0.015, size=n)
        close = 100.0 * np.cumprod(1 + returns)

        # Build OHLC around close
        intraday_range = close * np.abs(rng.normal(0, 0.01, size=n))
        open_ = close * (1 + rng.normal(0, 0.005, size=n))
        high = np.maximum(open_, close) + intraday_range
        low = np.minimum(open_, close) - intraday_range

        # Ensure high >= max(open, close) and low <= min(open, close)
        high = np.maximum(high, np.maximum(open_, close))
        low = np.minimum(low, np.minimum(open_, close))

        # Volume — random around 1M shares
        volume = rng.integers(500_000, 5_000_000, size=n).astype(np.int64)

        df = pd.DataFrame(
            {
                "open": np.round(open_, 2),
                "high": np.round(high, 2),
                "low": np.round(low, 2),
                "close": np.round(close, 2),
                "volume": volume,
            },
            index=dates,
        )
        df.index.name = "date"

        return df


# ──────────────────────────────────────────────
#  Provider registry
# ──────────────────────────────────────────────

def get_provider(provider_name: str, **kwargs) -> DataProvider:
    """Factory to create a provider by name."""
    providers = {
        "yfinance": YFinanceProvider,
        "synthetic": SyntheticProvider,
    }
    cls = providers.get(provider_name)
    if cls is None:
        raise ValueError(
            f"Unknown provider '{provider_name}'. "
            f"Available: {list(providers)}"
        )
    return cls(**kwargs)
