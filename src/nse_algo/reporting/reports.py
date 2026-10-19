"""
Report generation for backtest and paper-trading results.

Generates text and DataFrame reports.  Chart generation is in charts.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from nse_algo.backtest.metrics import PerformanceMetrics, calculate_all_metrics


def generate_trade_table(trades: list) -> pd.DataFrame:
    """Create a DataFrame from trade records."""
    if not trades:
        return pd.DataFrame(columns=[
            "symbol", "entry_date", "exit_date", "entry_price",
            "exit_price", "quantity", "pnl", "pnl_pct", "costs",
            "exit_reason", "holding_days",
        ])
    rows = []
    for t in trades:
        rows.append({
            "symbol": t.symbol,
            "entry_date": t.entry_date,
            "exit_date": t.exit_date,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "quantity": t.quantity,
            "pnl": t.pnl,
            "pnl_pct": t.pnl_pct,
            "costs": t.costs,
            "exit_reason": t.exit_reason.value if hasattr(t.exit_reason, 'value') else str(t.exit_reason),
            "holding_days": t.holding_days,
        })
    return pd.DataFrame(rows)


def generate_text_report(
    metrics: PerformanceMetrics,
    initial_capital: float,
    final_equity: float,
) -> str:
    """Generate a human-readable performance report."""
    lines = [
        "=" * 60,
        "  NSE ALGO TRADING — PERFORMANCE REPORT",
        "=" * 60,
        "",
        str(metrics),
        "",
        f"Initial Capital:       ₹{initial_capital:,.2f}",
        f"Final Equity:          ₹{final_equity:,.2f}",
        f"Net P&L:               ₹{final_equity - initial_capital:,.2f}",
        "",
        "=" * 60,
        "  DISCLAIMER",
        "=" * 60,
        "  Historical performance does not guarantee future results.",
        "  All execution is simulated. Never treat AI-generated signals",
        "  or backtest results as proof of future profitability.",
        "=" * 60,
    ]
    return "\n".join(lines)


def generate_daily_summary(
    equity_curve: pd.Series,
    daily_returns: pd.Series,
) -> pd.DataFrame:
    """Generate a daily summary table."""
    df = pd.DataFrame({
        "equity": equity_curve,
        "daily_return": daily_returns,
    })
    df["cumulative_return"] = (equity_curve / equity_curve.iloc[0] - 1) if len(equity_curve) > 0 else pd.Series(dtype=float)
    return df
