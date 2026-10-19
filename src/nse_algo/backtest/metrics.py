"""
Performance metrics for backtest results.

All formulas are documented with their assumptions.
Annualization uses 252 trading days for daily returns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class PerformanceMetrics:
    """Complete performance metrics for a backtest."""

    total_return: float
    cagr: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    win_rate: float
    avg_winning_trade: float
    avg_losing_trade: float
    expectancy: float
    n_trades: int
    avg_holding_period: float
    total_costs: float
    turnover: float
    volatility_annual: float

    def __str__(self) -> str:
        lines = [
            f"Total Return:          {self.total_return:.2%}",
            f"CAGR:                  {self.cagr:.2%}",
            f"Max Drawdown:          {self.max_drawdown:.2%}",
            f"Sharpe Ratio:          {self.sharpe_ratio:.3f}",
            f"Sortino Ratio:         {self.sortino_ratio:.3f}",
            f"Win Rate:              {self.win_rate:.2%}",
            f"Avg Winning Trade:     ₹{self.avg_winning_trade:,.2f}",
            f"Avg Losing Trade:      ₹{self.avg_losing_trade:,.2f}",
            f"Expectancy:            ₹{self.expectancy:,.2f}",
            f"Number of Trades:     {self.n_trades}",
            f"Avg Holding Period:    {self.avg_holding_period:.1f} bars",
            f"Total Costs:          ₹{self.total_costs:,.2f}",
            f"Annual Volatility:     {self.volatility_annual:.2%}",
        ]
        return "\n".join(lines)


def total_return(equity_curve: pd.Series) -> float:
    """Total return = final / initial - 1."""
    if len(equity_curve) < 2:
        return 0.0
    return float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0)


def cagr(equity_curve: pd.Series) -> float:
    """Compound Annual Growth Rate.

    CAGR = (final/initial)^(252/days) - 1
    """
    if len(equity_curve) < 2:
        return 0.0
    n_days = len(equity_curve)
    if n_days < TRADING_DAYS_PER_YEAR:
        # Not enough data for meaningful annualisation
        return total_return(equity_curve)
    ratio = equity_curve.iloc[-1] / equity_curve.iloc[0]
    if ratio <= 0:
        return -1.0
    return float(ratio ** (TRADING_DAYS_PER_YEAR / n_days) - 1.0)


def max_drawdown(equity_curve: pd.Series) -> float:
    """Maximum drawdown as a fraction (0 to 1).

    MDD = max(1 - equity / running_max(equity))
    """
    if len(equity_curve) < 2:
        return 0.0
    running_max = equity_curve.cummax()
    drawdown = 1.0 - equity_curve / running_max
    return float(drawdown.max())


def sharpe_ratio(
    daily_returns: pd.Series,
    risk_free_rate: float = 0.0,
) -> float:
    """Annualised Sharpe ratio.

    Sharpe = mean(daily_returns - rf_daily) / std(daily_returns) * sqrt(252)

    Returns 0.0 if std is 0 or insufficient data.
    """
    if len(daily_returns) < 2:
        return 0.0

    rf_daily = risk_free_rate / TRADING_DAYS_PER_YEAR
    excess = daily_returns - rf_daily
    std = excess.std()
    if std == 0 or np.isnan(std):
        return 0.0
    return float(excess.mean() / std * np.sqrt(TRADING_DAYS_PER_YEAR))


def sortino_ratio(
    daily_returns: pd.Series,
    risk_free_rate: float = 0.0,
) -> float:
    """Annualised Sortino ratio (uses downside deviation only).

    Sortino = mean(excess) / downside_std * sqrt(252)
    downside_std = std of returns below 0 (or below rf)
    """
    if len(daily_returns) < 2:
        return 0.0

    rf_daily = risk_free_rate / TRADING_DAYS_PER_YEAR
    excess = daily_returns - rf_daily
    downside = excess[excess < 0]
    if len(downside) == 0:
        return float("inf") if excess.mean() > 0 else 0.0
    downside_std = downside.std()
    if downside_std == 0 or np.isnan(downside_std):
        return 0.0
    return float(excess.mean() / downside_std * np.sqrt(TRADING_DAYS_PER_YEAR))


def win_rate(trades: list) -> float:
    """Fraction of trades with positive P&L."""
    if not trades:
        return 0.0
    wins = sum(1 for t in trades if t.pnl > 0)
    return wins / len(trades)


def avg_winning_trade(trades: list) -> float:
    """Average P&L of winning trades."""
    winners = [t.pnl for t in trades if t.pnl > 0]
    if not winners:
        return 0.0
    return float(np.mean(winners))


def avg_losing_trade(trades: list) -> float:
    """Average P&L of losing trades."""
    losers = [t.pnl for t in trades if t.pnl < 0]
    if not losers:
        return 0.0
    return float(np.mean(losers))


def expectancy(trades: list) -> float:
    """Expected P&L per trade.

    E = win_rate * avg_win + (1 - win_rate) * avg_loss
    """
    if not trades:
        return 0.0
    wr = win_rate(trades)
    aw = avg_winning_trade(trades)
    al = avg_losing_trade(trades)
    return wr * aw + (1 - wr) * al


def avg_holding_period(trades: list) -> float:
    """Average holding period in bars."""
    if not trades:
        return 0.0
    return float(np.mean([t.holding_days for t in trades]))


def total_costs(trades: list) -> float:
    """Sum of all transaction costs across trades."""
    return float(sum(t.costs for t in trades))


def volatility_annual(daily_returns: pd.Series) -> float:
    """Annualised volatility = std * sqrt(252)."""
    if len(daily_returns) < 2:
        return 0.0
    return float(daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))


def monthly_returns(equity_curve: pd.Series) -> pd.Series:
    """Monthly returns from equity curve."""
    if len(equity_curve) < 2:
        return pd.Series(dtype=float)
    monthly = equity_curve.resample("ME").last()
    return monthly.pct_change().dropna()


def annual_returns(equity_curve: pd.Series) -> pd.Series:
    """Annual returns from equity curve."""
    if len(equity_curve) < 2:
        return pd.Series(dtype=float)
    annual = equity_curve.resample("YE").last()
    return annual.pct_change().dropna()


def calculate_all_metrics(
    equity_curve: pd.Series,
    daily_returns: pd.Series,
    trades: list,
    initial_capital: float,
    risk_free_rate: float = 0.0,
) -> PerformanceMetrics:
    """Calculate all performance metrics at once."""
    return PerformanceMetrics(
        total_return=total_return(equity_curve),
        cagr=cagr(equity_curve),
        max_drawdown=max_drawdown(equity_curve),
        sharpe_ratio=sharpe_ratio(daily_returns, risk_free_rate),
        sortino_ratio=sortino_ratio(daily_returns, risk_free_rate),
        win_rate=win_rate(trades),
        avg_winning_trade=avg_winning_trade(trades),
        avg_losing_trade=avg_losing_trade(trades),
        expectancy=expectancy(trades),
        n_trades=len(trades),
        avg_holding_period=avg_holding_period(trades),
        total_costs=total_costs(trades),
        turnover=total_costs(trades) / initial_capital if initial_capital > 0 else 0.0,
        volatility_annual=volatility_annual(daily_returns),
    )
