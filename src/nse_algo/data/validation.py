"""
Data-quality validation for NSE OHLCV bars.

Checks for:
  - Missing required columns.
  - Duplicate dates.
  - Gaps in the trading calendar (missing sessions).
  - Negative or zero prices.
  - High < Low or High < max(Open, Close).
  - Volume anomalies.
  - Suspicious price movements (> 20 % daily move flagged).

All checks produce a DataQualityReport — no data is silently deleted.
Suspicious values are *flagged*, not removed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────
#  Data quality report
# ──────────────────────────────────────────────

@dataclass
class DataQualityReport:
    """Result of validating one symbol's OHLCV data."""

    symbol: str
    total_rows: int = 0
    duplicate_dates: int = 0
    missing_sessions: int = 0
    negative_prices: int = 0
    invalid_ohlc: int = 0  # high < low, etc.
    suspicious_moves: int = 0  # > threshold daily move
    zero_volume_days: int = 0
    nan_count: int = 0
    passed: bool = False
    issues: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [
            f"Data Quality Report: {self.symbol} [{status}]",
            f"  Total rows:         {self.total_rows}",
            f"  Duplicate dates:    {self.duplicate_dates}",
            f"  Missing sessions:   {self.missing_sessions}",
            f"  Negative prices:    {self.negative_prices}",
            f"  Invalid OHLC:       {self.invalid_ohlc}",
            f"  Suspicious moves:   {self.suspicious_moves}",
            f"  Zero volume days:   {self.zero_volume_days}",
            f"  NaN count:          {self.nan_count}",
        ]
        if self.issues:
            lines.append("  Issues:")
            for issue in self.issues:
                lines.append(f"    - {issue}")
        return "\n".join(lines)


# ──────────────────────────────────────────────
#  Validation functions
# ──────────────────────────────────────────────

REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")


def validate_ohlcv(
    df: pd.DataFrame,
    symbol: str,
    *,
    max_daily_move_pct: float = 0.20,
    min_trading_days: int = 250,
    max_missing_pct: float = 0.02,
) -> DataQualityReport:
    """Validate OHLCV data and return a quality report.

    Parameters
    ----------
    df : DataFrame
        OHLCV data indexed by date with lowercase columns.
    symbol : str
        Symbol identifier for the report.
    max_daily_move_pct : float
        Flag daily moves (abs(close/open - 1)) above this as suspicious.
    min_trading_days : int
        Minimum number of rows expected.
    max_missing_pct : float
        Maximum fraction of missing business days allowed.

    Returns
    -------
    DataQualityReport
    """
    report = DataQualityReport(symbol=symbol)

    # --- Check required columns ---
    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        report.issues.append(f"Missing columns: {missing_cols}")
        return report  # Can't continue without columns

    report.total_rows = len(df)

    # --- Check for NaN values ---
    nan_mask = df[list(REQUIRED_COLUMNS)].isna().any(axis=1)
    report.nan_count = int(nan_mask.sum())
    if report.nan_count > 0:
        report.issues.append(f"{report.nan_count} rows contain NaN values")

    # --- Check for duplicate dates ---
    report.duplicate_dates = int(df.index.duplicated().sum())
    if report.duplicate_dates > 0:
        report.issues.append(f"{report.duplicate_dates} duplicate dates found")

    # --- Check for missing sessions (business-day gaps) ---
    if len(df) >= 2:
        expected_dates = pd.bdate_range(df.index.min(), df.index.max())
        actual_dates = pd.DatetimeIndex(df.drop_duplicates().index)
        missing = expected_dates.difference(actual_dates)
        report.missing_sessions = len(missing)
        if len(expected_dates) > 0:
            missing_pct = report.missing_sessions / len(expected_dates)
            if missing_pct > max_missing_pct:
                report.issues.append(
                    f"{report.missing_sessions} missing sessions "
                    f"({missing_pct:.1%} > {max_missing_pct:.1%} threshold)"
                )

    # Work on a clean copy for price checks
    clean = df.dropna(subset=list(REQUIRED_COLUMNS)).copy()
    clean = clean[~clean.index.duplicated(keep="first")]

    if clean.empty:
        report.issues.append("No valid rows after dropping NaN and duplicates")
        return report

    # --- Check for negative or zero prices ---
    price_cols = ("open", "high", "low", "close")
    for col in price_cols:
        bad = (clean[col] <= 0).sum()
        if bad > 0:
            report.negative_prices += int(bad)
            report.issues.append(f"{bad} rows with non-positive {col}")

    # --- Check OHLC consistency ---
    # high >= max(open, close), low <= min(open, close), high >= low
    high_ok = clean["high"] >= clean[["open", "close"]].max(axis=1)
    low_ok = clean["low"] <= clean[["open", "close"]].min(axis=1)
    hl_ok = clean["high"] >= clean["low"]

    invalid = (~high_ok) | (~low_ok) | (~hl_ok)
    report.invalid_ohlc = int(invalid.sum())
    if report.invalid_ohlc > 0:
        report.issues.append(f"{report.invalid_ohlc} rows with inconsistent OHLC")

    # --- Check for suspicious price movements ---
    daily_move = (clean["close"] / clean["open"] - 1).abs()
    suspicious = daily_move > max_daily_move_pct
    report.suspicious_moves = int(suspicious.sum())
    if report.suspicious_moves > 0:
        report.issues.append(
            f"{report.suspicious_moves} sessions with > {max_daily_move_pct:.0%} daily move "
            "(flagged, not removed)"
        )

    # --- Check zero volume ---
    report.zero_volume_days = int((clean["volume"] == 0).sum())
    if report.zero_volume_days > 0:
        report.issues.append(
            f"{report.zero_volume_days} sessions with zero volume"
        )

    # --- Check minimum trading history ---
    if report.total_rows < min_trading_days:
        report.issues.append(
            f"Only {report.total_rows} rows (minimum: {min_trading_days})"
        )

    # --- Overall pass/fail ---
    report.passed = (
        report.duplicate_dates == 0
        and report.negative_prices == 0
        and report.invalid_ohlc == 0
        and report.nan_count == 0
        and report.total_rows >= min_trading_days
    )

    return report


def validate_universe(
    data: dict[str, pd.DataFrame],
    **kwargs: Any,
) -> dict[str, DataQualityReport]:
    """Validate OHLCV data for multiple symbols.

    Parameters
    ----------
    data : dict[str, DataFrame]
        Mapping of symbol → OHLCV DataFrame.
    **kwargs
        Passed to ``validate_ohlcv``.

    Returns
    -------
    dict[str, DataQualityReport]
    """
    return {
        symbol: validate_ohlcv(df, symbol, **kwargs)
        for symbol, df in data.items()
    }
