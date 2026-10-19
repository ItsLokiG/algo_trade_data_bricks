"""
Monitoring checks for the trading system.

Checks for:
  - Data quality issues (missing bars, suspicious moves).
  - Portfolio exposure limits.
  - Drawdown breaches.
  - Failed ingestion jobs.
  - Strategy performance anomalies.

All checks return a CheckResult with severity and message.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd


class Severity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass(frozen=True)
class CheckResult:
    """Result of a monitoring check."""
    name: str
    severity: Severity
    message: str
    value: float | None = None

    @property
    def is_error(self) -> bool:
        return self.severity in (Severity.ERROR, Severity.CRITICAL)


def check_drawdown(
    equity_curve: pd.Series,
    max_drawdown_limit: float = 0.20,
) -> CheckResult:
    """Check if drawdown exceeds the limit."""
    if len(equity_curve) < 2:
        return CheckResult("drawdown", Severity.INFO, "Not enough data")
    running_max = equity_curve.cummax()
    drawdown = 1.0 - equity_curve / running_max
    current_dd = float(drawdown.iloc[-1])
    max_dd = float(drawdown.max())

    if max_dd >= max_drawdown_limit:
        return CheckResult(
            "drawdown", Severity.CRITICAL,
            f"Max drawdown {max_dd:.1%} exceeds limit {max_drawdown_limit:.1%}",
            max_dd,
        )
    elif current_dd > max_drawdown_limit * 0.8:
        return CheckResult(
            "drawdown", Severity.WARNING,
            f"Current drawdown {current_dd:.1%} approaching limit",
            current_dd,
        )
    return CheckResult(
        "drawdown", Severity.INFO,
        f"Current drawdown {current_dd:.1%}, max {max_dd:.1%}",
        current_dd,
    )


def check_exposure(
    positions_value: float,
    total_equity: float,
    max_exposure: float = 1.0,
) -> CheckResult:
    """Check portfolio exposure against limit."""
    if total_equity <= 0:
        return CheckResult("exposure", Severity.ERROR, "Equity is zero or negative")
    exposure = positions_value / total_equity
    if exposure > max_exposure:
        return CheckResult(
            "exposure", Severity.ERROR,
            f"Exposure {exposure:.1%} exceeds limit {max_exposure:.1%}",
            exposure,
        )
    return CheckResult(
        "exposure", Severity.INFO,
        f"Exposure {exposure:.1%}",
        exposure,
    )


def check_data_freshness(
    last_data_date: pd.Timestamp,
    max_delay_days: int = 3,
) -> CheckResult:
    """Check if data is stale."""
    if last_data_date is None:
        return CheckResult("data_freshness", Severity.ERROR, "No data available")
    delay = (pd.Timestamp.now() - last_data_date).days
    if delay > max_delay_days:
        return CheckResult(
            "data_freshness", Severity.WARNING,
            f"Data is {delay} days old (max {max_delay_days})",
            float(delay),
        )
    return CheckResult(
        "data_freshness", Severity.INFO,
        f"Data is {delay} days old",
        float(delay),
    )


def check_no_trades(
    n_trades: int,
    min_expected: int = 1,
) -> CheckResult:
    """Check if strategy is generating trades."""
    if n_trades == 0:
        return CheckResult(
            "no_trades", Severity.WARNING,
            "No trades generated — strategy may be inactive or too restrictive",
        )
    return CheckResult(
        "no_trades", Severity.INFO,
        f"{n_trades} trades generated",
    )


def run_all_checks(
    equity_curve: pd.Series | None = None,
    positions_value: float = 0.0,
    total_equity: float = 0.0,
    last_data_date: pd.Timestamp | None = None,
    n_trades: int = 0,
    max_drawdown_limit: float = 0.20,
    max_exposure: float = 1.0,
) -> list[CheckResult]:
    """Run all monitoring checks and return results."""
    results: list[CheckResult] = []

    if equity_curve is not None:
        results.append(check_drawdown(equity_curve, max_drawdown_limit))

    if total_equity > 0:
        results.append(check_exposure(positions_value, total_equity, max_exposure))

    if last_data_date is not None:
        results.append(check_data_freshness(last_data_date))

    results.append(check_no_trades(n_trades))

    return results
