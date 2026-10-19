"""
Tests for reporting and monitoring modules.
"""

import numpy as np
import pandas as pd
import pytest

from nse_algo.backtest.engine import Trade, ExitReason
from nse_algo.backtest.metrics import PerformanceMetrics, calculate_all_metrics
from nse_algo.monitoring.checks import (
    CheckResult,
    Severity,
    check_data_freshness,
    check_drawdown,
    check_exposure,
    check_no_trades,
    run_all_checks,
)
from nse_algo.reporting.reports import (
    generate_daily_summary,
    generate_text_report,
    generate_trade_table,
)


class FakeTrade:
    def __init__(self, pnl=100, costs=10, holding_days=5):
        self.symbol = "TEST"
        self.entry_date = pd.Timestamp("2024-01-15")
        self.exit_date = pd.Timestamp("2024-01-20")
        self.entry_price = 100.0
        self.exit_price = 110.0
        self.quantity = 100
        self.pnl = pnl
        self.pnl_pct = pnl / (100 * 100)
        self.costs = costs
        self.exit_reason = ExitReason.SIGNAL_EXIT
        self.holding_days = holding_days


# ──────────────────────────────────────────────
#  Reporting
# ──────────────────────────────────────────────

class TestTradeTable:
    def test_empty_trades(self) -> None:
        result = generate_trade_table([])
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_with_trades(self) -> None:
        trades = [FakeTrade(), FakeTrade(pnl=-50)]
        result = generate_trade_table(trades)
        assert len(result) == 2
        assert "symbol" in result.columns
        assert "pnl" in result.columns
        assert "exit_reason" in result.columns


class TestTextReport:
    def test_contains_disclaimer(self) -> None:
        metrics = PerformanceMetrics(
            total_return=0.10, cagr=0.08, max_drawdown=0.05,
            sharpe_ratio=1.2, sortino_ratio=1.5, win_rate=0.6,
            avg_winning_trade=200, avg_losing_trade=-100,
            expectancy=80, n_trades=10, avg_holding_period=5,
            total_costs=500, turnover=0.05, volatility_annual=0.15,
        )
        report = generate_text_report(metrics, 100_000, 110_000)
        assert "DISCLAIMER" in report
        assert "does not guarantee" in report

    def test_contains_metrics(self) -> None:
        metrics = PerformanceMetrics(
            total_return=0.10, cagr=0.08, max_drawdown=0.05,
            sharpe_ratio=1.2, sortino_ratio=1.5, win_rate=0.6,
            avg_winning_trade=200, avg_losing_trade=-100,
            expectancy=80, n_trades=10, avg_holding_period=5,
            total_costs=500, turnover=0.05, volatility_annual=0.15,
        )
        report = generate_text_report(metrics, 100_000, 110_000)
        assert "Sharpe" in report
        assert "₹100,000.00" in report


class TestDailySummary:
    def test_generates_dataframe(self) -> None:
        dates = pd.bdate_range("2024-01-01", periods=10)
        eq = pd.Series(np.linspace(100_000, 105_000, 10), index=dates)
        ret = eq.pct_change().dropna()
        result = generate_daily_summary(eq, ret)
        assert "equity" in result.columns
        assert "daily_return" in result.columns


# ──────────────────────────────────────────────
#  Monitoring
# ──────────────────────────────────────────────

class TestCheckDrawdown:
    def test_no_drawdown(self) -> None:
        eq = pd.Series([100, 110, 120])
        result = check_drawdown(eq, max_drawdown_limit=0.20)
        assert result.severity == Severity.INFO

    def test_critical_drawdown(self) -> None:
        eq = pd.Series([100, 80, 75])
        result = check_drawdown(eq, max_drawdown_limit=0.20)
        assert result.severity == Severity.CRITICAL

    def test_warning_near_limit(self) -> None:
        eq = pd.Series([100, 100, 83])  # 16% drawdown, near 20% limit
        result = check_drawdown(eq, max_drawdown_limit=0.20)
        assert result.severity == Severity.WARNING

    def test_short_series(self) -> None:
        eq = pd.Series([100])
        result = check_drawdown(eq)
        assert result.severity == Severity.INFO


class TestCheckExposure:
    def test_ok(self) -> None:
        result = check_exposure(50_000, 100_000, max_exposure=1.0)
        assert result.severity == Severity.INFO

    def test_over_limit(self) -> None:
        result = check_exposure(120_000, 100_000, max_exposure=1.0)
        assert result.severity == Severity.ERROR

    def test_zero_equity(self) -> None:
        result = check_exposure(0, 0)
        assert result.severity == Severity.ERROR


class TestCheckDataFreshness:
    def test_fresh(self) -> None:
        recent = pd.Timestamp.now() - pd.Timedelta(days=1)
        result = check_data_freshness(recent, max_delay_days=3)
        assert result.severity == Severity.INFO

    def test_stale(self) -> None:
        old = pd.Timestamp.now() - pd.Timedelta(days=10)
        result = check_data_freshness(old, max_delay_days=3)
        assert result.severity == Severity.WARNING


class TestCheckNoTrades:
    def test_has_trades(self) -> None:
        result = check_no_trades(5)
        assert result.severity == Severity.INFO

    def test_no_trades(self) -> None:
        result = check_no_trades(0)
        assert result.severity == Severity.WARNING


class TestRunAllChecks:
    def test_returns_list(self) -> None:
        dates = pd.bdate_range("2024-01-01", periods=10)
        eq = pd.Series(np.linspace(100_000, 105_000, 10), index=dates)
        results = run_all_checks(
            equity_curve=eq,
            positions_value=50_000,
            total_equity=100_000,
            last_data_date=dates[-1],
            n_trades=5,
        )
        assert isinstance(results, list)
        assert all(isinstance(r, CheckResult) for r in results)
        assert len(results) >= 4
