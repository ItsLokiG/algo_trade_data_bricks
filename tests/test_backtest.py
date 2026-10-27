"""
Tests for transaction cost model and backtest engine.
"""

import numpy as np
import pandas as pd
import pytest

from nse_algo.backtest.costs import (
    CostBreakdown,
    calculate_transaction_cost,
    round_to_lot_size,
)
from nse_algo.backtest.engine import (
    BacktestResult,
    ExitReason,
    Position,
    Trade,
    calculate_position_size,
    run_backtest,
)
from nse_algo.config import ProjectConfig, RiskConfig, CostConfig, CapitalConfig
from nse_algo.data.providers import SyntheticProvider


# ──────────────────────────────────────────────
#  Cost model tests
# ──────────────────────────────────────────────

class TestTransactionCost:
    def test_buy_cost_breakdown(self) -> None:
        result = calculate_transaction_cost(100, 500, is_buy=True)
        assert isinstance(result, CostBreakdown)
        assert result.traded_value == 50_000
        assert result.brokerage > 0
        assert result.stt > 0
        assert result.stamp_duty > 0  # buy-side
        assert result.total_cost > 0

    def test_sell_no_stamp_duty(self) -> None:
        result = calculate_transaction_cost(100, 500, is_buy=False)
        assert result.stamp_duty == 0

    def test_brokerage_capped(self) -> None:
        # Large trade: brokerage should be capped
        result = calculate_transaction_cost(
            10_000, 1000, is_buy=True,
            brokerage_pct=0.0003, brokerage_cap=20.0,
        )
        assert result.brokerage == 20.0  # capped

    def test_small_trade_brokerage_not_capped(self) -> None:
        result = calculate_transaction_cost(10, 100, is_buy=True)
        # 10 * 100 * 0.0003 = 0.30, below cap
        assert result.brokerage == 0.30

    def test_zero_quantity_zero_cost(self) -> None:
        result = calculate_transaction_cost(0, 100, is_buy=True)
        assert result.total_cost == 0

    def test_negative_price_raises(self) -> None:
        with pytest.raises(ValueError):
            calculate_transaction_cost(100, -1, is_buy=True)

    def test_gst_on_brokerage_plus_exchange(self) -> None:
        result = calculate_transaction_cost(
            100, 500, is_buy=True,
            brokerage_pct=0.0003, brokerage_cap=20.0,
            exchange_txn_pct=0.0000297,
            gst_pct=0.18,
            sebi_pct=0.000001,
            slippage_pct=0.0,  # disable for clean test
            stamp_duty_pct=0.0,
            stt_pct=0.0,
        )
        expected_gst = (result.brokerage + result.exchange_charges + result.sebi_charges) * 0.18
        assert np.isclose(result.gst, expected_gst)

    def test_str_representation(self) -> None:
        result = calculate_transaction_cost(100, 500, is_buy=True)
        s = str(result)
        assert "Brokerage" in s
        assert "TOTAL" in s


class TestRoundToLotSize:
    def test_round_down(self) -> None:
        assert round_to_lot_size(157, lot_size=50) == 150

    def test_exact_multiple(self) -> None:
        assert round_to_lot_size(100, lot_size=50) == 100

    def test_below_lot_size(self) -> None:
        assert round_to_lot_size(49, lot_size=50) == 0

    def test_invalid_lot_size(self) -> None:
        with pytest.raises(ValueError):
            round_to_lot_size(100, lot_size=0)


# ──────────────────────────────────────────────
#  Position sizing tests
# ──────────────────────────────────────────────

class TestPositionSizing:
    def test_basic_calculation(self) -> None:
        # equity=100k, risk=1%, entry=100, stop=95 → risk_amt=1000, stop_dist=5
        qty = calculate_position_size(
            equity=100_000, entry_price=100, stop_price=95,
            risk_per_trade=0.01, max_position_value=20_000, available_cash=100_000,
        )
        assert qty == 200  # 1000 / 5 = 200

    def test_capped_by_max_position(self) -> None:
        # max_position_value = 5000, price = 100 → max 50 shares
        qty = calculate_position_size(
            equity=100_000, entry_price=100, stop_price=99,  # tiny stop = huge size
            risk_per_trade=0.01, max_position_value=5_000, available_cash=100_000,
        )
        assert qty == 50

    def test_capped_by_cash(self) -> None:
        qty = calculate_position_size(
            equity=100_000, entry_price=100, stop_price=90,
            risk_per_trade=0.01, max_position_value=500_000, available_cash=5_000,
        )
        assert qty == 50  # 5000 / 100

    def test_no_valid_stop(self) -> None:
        qty = calculate_position_size(
            equity=100_000, entry_price=100, stop_price=100,
            risk_per_trade=0.01, max_position_value=20_000, available_cash=100_000,
        )
        assert qty == 0

    def test_stop_above_entry(self) -> None:
        qty = calculate_position_size(
            equity=100_000, entry_price=100, stop_price=105,
            risk_per_trade=0.01, max_position_value=20_000, available_cash=100_000,
        )
        assert qty == 0


# ──────────────────────────────────────────────
#  Backtest engine tests
# ──────────────────────────────────────────────

class TestBacktestEngine:
    def _make_data(self, n_symbols: int = 3) -> dict[str, pd.DataFrame]:
        p = SyntheticProvider(seed=42)
        data = {}
        for i in range(n_symbols):
            data[f"STK{i}"] = p.fetch_ohlcv(f"STK{i}", "2020-01-01", "2023-12-31")
        return data

    def test_runs_without_error(self) -> None:
        data = self._make_data(1)
        config = ProjectConfig(capital=CapitalConfig(initial_capital=100_000))
        result = run_backtest(data, config=config)
        assert isinstance(result, BacktestResult)

    def test_equity_curve_length(self) -> None:
        data = self._make_data(1)
        result = run_backtest(data)
        assert len(result.equity_curve) > 0
        assert result.equity_curve.iloc[0] > 0  # starts with some equity

    def test_final_equity_positive(self) -> None:
        data = self._make_data(1)
        result = run_backtest(data)
        assert result.final_equity > 0

    def test_initial_capital_preserved(self) -> None:
        data = self._make_data(1)
        result = run_backtest(data)
        assert result.initial_capital == 100_000

    def test_trades_recorded(self) -> None:
        data = self._make_data(3)
        result = run_backtest(data)
        # May or may not have trades depending on signals
        assert isinstance(result.trades, list)
        for t in result.trades:
            assert isinstance(t, Trade)
            assert t.symbol in {"STK0", "STK1", "STK2"}
            assert t.quantity > 0

    def test_winning_losing_counts(self) -> None:
        data = self._make_data(3)
        result = run_backtest(data)
        assert result.n_winning + result.n_losing == result.n_trades

    def test_daily_returns_length_matches_equity(self) -> None:
        data = self._make_data(1)
        result = run_backtest(data)
        assert len(result.daily_returns) == len(result.equity_curve)

    def test_positions_history_has_columns(self) -> None:
        data = self._make_data(1)
        result = run_backtest(data)
        if not result.positions_history.empty:
            for col in ("date", "cash", "n_positions", "equity"):
                assert col in result.positions_history.columns

    def test_exit_reasons_valid(self) -> None:
        data = self._make_data(3)
        result = run_backtest(data)
        valid_reasons = {e for e in ExitReason}
        for t in result.trades:
            assert t.exit_reason in valid_reasons

    def test_total_return_calculated(self) -> None:
        data = self._make_data(1)
        result = run_backtest(data)
        # total_return = final/initial - 1
        expected = (result.final_equity / result.initial_capital) - 1.0
        assert np.isclose(result.total_return, expected, rtol=1e-6)

    def test_trade_costs_include_both_sides_and_atr_is_lagged(self, monkeypatch) -> None:
        dates = pd.bdate_range("2024-01-02", periods=3)
        bars = pd.DataFrame(
            {
                "open": [100.0, 100.0, 102.0],
                "high": [101.0, 110.0, 112.0],
                "low": [99.0, 99.0, 100.0],
                "close": [100.0, 105.0, 110.0],
                "volume": [1000, 1000, 1000],
            },
            index=dates,
        )

        def fake_signals(df, config):
            signals = df.copy()
            signals["signal_entry"] = [True, False, False]
            signals["signal_exit"] = False
            signals["proposed_stop"] = [90.0, 90.0, 90.0]
            signals["ind_atr"] = [2.0, 10.0, 10.0]
            return signals

        monkeypatch.setattr(
            "nse_algo.backtest.engine.generate_all_signals", fake_signals
        )
        config = ProjectConfig(
            capital=CapitalConfig(initial_capital=100_000),
            risk=RiskConfig(trailing_stop_enabled=True),
        )
        result = run_backtest({"TEST": bars}, config=config)

        assert result.n_trades == 1
        trade = result.trades[0]
        assert trade.entry_date == dates[1]
        assert trade.exit_date == dates[2]
        assert trade.exit_reason == ExitReason.TRAILING_STOP
        assert trade.exit_price == 101.0
        expected_buy_costs = calculate_transaction_cost(
            trade.quantity, trade.entry_price, is_buy=True
        ).total_cost
        expected_sell_costs = calculate_transaction_cost(
            trade.quantity, trade.exit_price, is_buy=False
        ).total_cost
        assert np.isclose(trade.costs, expected_buy_costs + expected_sell_costs)
        assert np.isclose(
            trade.pnl,
            trade.quantity * (trade.exit_price - trade.entry_price) - trade.costs,
        )
        assert np.isclose(result.final_equity, config.capital.initial_capital + trade.pnl)
        assert np.isclose(result.equity_curve.iloc[-1], result.final_equity)

    def test_end_of_data_liquidation_updates_final_equity(self, monkeypatch) -> None:
        dates = pd.bdate_range("2024-01-02", periods=2)
        bars = pd.DataFrame(
            {
                "open": [100.0, 100.0],
                "high": [101.0, 111.0],
                "low": [99.0, 99.0],
                "close": [100.0, 110.0],
                "volume": [1000, 1000],
            },
            index=dates,
        )

        def fake_signals(df, config):
            signals = df.copy()
            signals["signal_entry"] = [True, False]
            signals["signal_exit"] = False
            signals["proposed_stop"] = [90.0, 90.0]
            signals["ind_atr"] = [2.0, 2.0]
            return signals

        monkeypatch.setattr(
            "nse_algo.backtest.engine.generate_all_signals", fake_signals
        )
        config = ProjectConfig(
            capital=CapitalConfig(initial_capital=100_000),
            risk=RiskConfig(trailing_stop_enabled=False),
        )
        result = run_backtest({"TEST": bars}, config=config)

        assert result.n_trades == 1
        trade = result.trades[0]
        assert trade.exit_reason == ExitReason.END_OF_DATA
        assert trade.exit_price == 110.0
        assert np.isclose(result.final_equity, config.capital.initial_capital + trade.pnl)
        assert np.isclose(result.equity_curve.iloc[-1], result.final_equity)

    def test_entry_day_stop_is_not_ignored(self, monkeypatch) -> None:
        dates = pd.bdate_range("2024-01-02", periods=2)
        bars = pd.DataFrame(
            {
                "open": [100.0, 100.0],
                "high": [101.0, 105.0],
                "low": [99.0, 80.0],
                "close": [100.0, 95.0],
                "volume": [1000, 1000],
            },
            index=dates,
        )

        def fake_signals(df, config):
            signals = df.copy()
            signals["signal_entry"] = [True, False]
            signals["signal_exit"] = False
            signals["proposed_stop"] = [90.0, 90.0]
            signals["ind_atr"] = [2.0, 2.0]
            return signals

        monkeypatch.setattr(
            "nse_algo.backtest.engine.generate_all_signals", fake_signals
        )
        result = run_backtest({"TEST": bars})

        assert result.n_trades == 1
        trade = result.trades[0]
        assert trade.exit_reason == ExitReason.STOP_LOSS
        assert trade.entry_date == trade.exit_date == dates[1]
        assert trade.exit_price == 90.0
        assert np.isclose(
            result.final_equity, result.initial_capital + trade.pnl
        )
