"""
Tests for performance metrics and robustness validation.
"""

import numpy as np
import pandas as pd
import pytest

from nse_algo.backtest.metrics import (
    PerformanceMetrics,
    avg_holding_period,
    avg_losing_trade,
    avg_winning_trade,
    calculate_all_metrics,
    cagr,
    expectancy,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    total_costs,
    total_return,
    volatility_annual,
    win_rate,
)
from nse_algo.backtest.validation import (
    DataSplits,
    MonteCarloResult,
    WalkForwardResult,
    monte_carlo_trade_reshuffle,
    parameter_sensitivity,
    split_data_chronologically,
    walk_forward,
)


# ──────────────────────────────────────────────
#  Helper to make synthetic trades
# ──────────────────────────────────────────────

class FakeTrade:
    def __init__(self, pnl: float, costs: float = 10.0, holding_days: int = 5):
        self.pnl = pnl
        self.costs = costs
        self.holding_days = holding_days


# ──────────────────────────────────────────────
#  Return metrics
# ──────────────────────────────────────────────

class TestTotalReturn:
    def test_positive_return(self) -> None:
        eq = pd.Series([100, 110, 120])
        assert total_return(eq) == pytest.approx(0.20)

    def test_negative_return(self) -> None:
        eq = pd.Series([100, 90, 80])
        assert total_return(eq) == pytest.approx(-0.20)

    def test_flat(self) -> None:
        eq = pd.Series([100, 100, 100])
        assert total_return(eq) == 0.0

    def test_single_point(self) -> None:
        eq = pd.Series([100])
        assert total_return(eq) == 0.0


class TestCagr:
    def test_one_year_growth(self) -> None:
        # 252 bars, 100→110 = 10% in one year
        eq = pd.Series(np.linspace(100, 110, 252))
        result = cagr(eq)
        assert 0.05 < result < 0.15  # roughly 10%

    def test_short_period(self) -> None:
        # Less than 252 bars → returns total_return
        eq = pd.Series([100, 120])
        assert cagr(eq) == pytest.approx(0.20)


class TestMaxDrawdown:
    def test_simple_drawdown(self) -> None:
        eq = pd.Series([100, 110, 90, 95])
        # Peak 110, trough 90 → DD = (110-90)/110 = 18.18%
        assert max_drawdown(eq) == pytest.approx(1 - 90/110, rel=1e-3)

    def test_no_drawdown(self) -> None:
        eq = pd.Series([100, 110, 120])
        assert max_drawdown(eq) == 0.0

    def test_single_point(self) -> None:
        eq = pd.Series([100])
        assert max_drawdown(eq) == 0.0


# ──────────────────────────────────────────────
#  Risk-adjusted metrics
# ──────────────────────────────────────────────

class TestSharpeRatio:
    def test_positive_sharpe(self) -> None:
        returns = pd.Series(np.random.default_rng(42).normal(0.001, 0.01, 252))
        sr = sharpe_ratio(returns)
        assert sr > 0  # positive mean return → positive Sharpe

    def test_zero_volatility(self) -> None:
        returns = pd.Series([0.0, 0.0, 0.0])
        assert sharpe_ratio(returns) == 0.0

    def test_short_series(self) -> None:
        returns = pd.Series([0.01])
        assert sharpe_ratio(returns) == 0.0


class TestSortinoRatio:
    def test_positive_sortino(self) -> None:
        returns = pd.Series(np.random.default_rng(42).normal(0.001, 0.01, 252))
        sr = sortino_ratio(returns)
        assert sr > 0

    def test_all_positive_returns(self) -> None:
        returns = pd.Series([0.01, 0.02, 0.01])
        # No downside → infinity
        assert sortino_ratio(returns) == float("inf")


class TestVolatility:
    def test_positive(self) -> None:
        returns = pd.Series([0.01, -0.01, 0.02, -0.02])
        vol = volatility_annual(returns)
        assert vol > 0

    def test_zero(self) -> None:
        returns = pd.Series([0.0, 0.0, 0.0])
        assert volatility_annual(returns) == 0.0


# ──────────────────────────────────────────────
#  Trade metrics
# ──────────────────────────────────────────────

class TestTradeMetrics:
    def test_win_rate(self) -> None:
        trades = [FakeTrade(100), FakeTrade(-50), FakeTrade(200), FakeTrade(-30)]
        assert win_rate(trades) == 0.5

    def test_win_rate_empty(self) -> None:
        assert win_rate([]) == 0.0

    def test_avg_winning_trade(self) -> None:
        trades = [FakeTrade(100), FakeTrade(-50), FakeTrade(200)]
        assert avg_winning_trade(trades) == 150.0

    def test_avg_losing_trade(self) -> None:
        trades = [FakeTrade(100), FakeTrade(-50), FakeTrade(-30)]
        assert avg_losing_trade(trades) == -40.0

    def test_expectancy(self) -> None:
        trades = [FakeTrade(100), FakeTrade(-50), FakeTrade(200), FakeTrade(-30)]
        exp = expectancy(trades)
        # 0.5 * 150 + 0.5 * (-40) = 55
        assert exp == pytest.approx(55.0)

    def test_total_costs(self) -> None:
        trades = [FakeTrade(100, costs=10), FakeTrade(-50, costs=15)]
        assert total_costs(trades) == 25.0

    def test_avg_holding_period(self) -> None:
        trades = [FakeTrade(100, holding_days=3), FakeTrade(-50, holding_days=7)]
        assert avg_holding_period(trades) == 5.0


class TestCalculateAllMetrics:
    def test_returns_metrics_object(self) -> None:
        dates = pd.bdate_range("2022-01-01", periods=300)
        eq = pd.Series(np.linspace(100_000, 110_000, 300), index=dates)
        ret = eq.pct_change().dropna()
        trades = [FakeTrade(500, costs=20, holding_days=5), FakeTrade(-200, costs=15, holding_days=3)]

        metrics = calculate_all_metrics(eq, ret, trades, initial_capital=100_000)
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.total_return > 0
        assert metrics.n_trades == 2
        assert metrics.win_rate == 0.5
        assert metrics.total_costs == 35.0


# ──────────────────────────────────────────────
#  Validation: chronological splits
# ──────────────────────────────────────────────

class TestDataSplits:
    def test_split_sizes(self) -> None:
        df = pd.DataFrame(np.arange(1000), columns=["val"])
        splits = split_data_chronologically(df, 0.6, 0.2)
        assert len(splits.train) == 600
        assert len(splits.validation) == 200
        assert len(splits.test) == 200

    def test_test_fraction(self) -> None:
        df = pd.DataFrame(np.arange(1000), columns=["val"])
        splits = split_data_chronologically(df, 0.6, 0.2)
        assert splits.test_fraction == pytest.approx(0.2)

    def test_overlapping_rejected(self) -> None:
        df = pd.DataFrame(np.arange(100), columns=["val"])
        with pytest.raises(ValueError):
            split_data_chronologically(df, 0.7, 0.4)

    def test_chronological_order(self) -> None:
        df = pd.DataFrame(np.arange(100), columns=["val"])
        splits = split_data_chronologically(df, 0.6, 0.2)
        assert splits.train.iloc[-1, 0] < splits.validation.iloc[0, 0]
        assert splits.validation.iloc[-1, 0] < splits.test.iloc[0, 0]


# ──────────────────────────────────────────────
#  Validation: walk-forward
# ──────────────────────────────────────────────

class TestWalkForward:
    def test_returns_results(self) -> None:
        dates = pd.bdate_range("2020-01-01", periods=500)
        eq = pd.Series(np.linspace(100, 120, 500), index=dates)
        results = walk_forward(eq, n_windows=5, train_fraction=0.7)
        assert len(results) > 0
        for r in results:
            assert isinstance(r, WalkForwardResult)
            assert r.test_start > r.train_end

    def test_invalid_windows(self) -> None:
        with pytest.raises(ValueError):
            walk_forward(pd.Series([1, 2, 3]), n_windows=0)


# ──────────────────────────────────────────────
#  Validation: Monte Carlo
# ──────────────────────────────────────────────

class TestMonteCarlo:
    def test_returns_result(self) -> None:
        returns = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
        result = monte_carlo_trade_reshuffle(returns, n_runs=100, seed=42)
        assert isinstance(result, MonteCarloResult)
        assert result.n_runs == 100
        assert -1 < result.mean_return < 1

    def test_deterministic_with_seed(self) -> None:
        returns = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
        r1 = monte_carlo_trade_reshuffle(returns, n_runs=100, seed=42)
        r2 = monte_carlo_trade_reshuffle(returns, n_runs=100, seed=42)
        assert r1.mean_return == r2.mean_return

    def test_empty_returns(self) -> None:
        result = monte_carlo_trade_reshuffle(np.array([]), n_runs=100)
        assert result.n_runs == 0

    def test_all_positive_returns(self) -> None:
        returns = np.array([0.01, 0.02, 0.03])
        result = monte_carlo_trade_reshuffle(returns, n_runs=100, seed=42)
        assert result.pct_profitable == 1.0

    def test_all_negative_returns(self) -> None:
        returns = np.array([-0.01, -0.02, -0.03])
        result = monte_carlo_trade_reshuffle(returns, n_runs=100, seed=42)
        assert result.pct_profitable == 0.0


# ──────────────────────────────────────────────
#  Validation: parameter sensitivity
# ──────────────────────────────────────────────

class TestParameterSensitivity:
    def test_returns_dataframe(self) -> None:
        def mock_func(data, multiplier):
            return sum(data.values()) * multiplier

        data = {"a": 1, "b": 2}
        result = parameter_sensitivity(data, "multiplier", [1, 2, 3], mock_func)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert "param_value" in result.columns
        assert "metric" in result.columns
