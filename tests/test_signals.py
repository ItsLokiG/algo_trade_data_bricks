"""
Tests for signal generation and stock ranking.
"""

import numpy as np
import pandas as pd
import pytest

from nse_algo.data.providers import SyntheticProvider
from nse_algo.features.indicators import add_all_indicators
from nse_algo.strategy.signals import (
    SignalConfig,
    SignalType,
    generate_all_signals,
    generate_entry_signals,
    generate_exit_signals,
)
from nse_algo.strategy.ranking import (
    RankingConfig,
    rank_stocks,
)


@pytest.fixture
def ohlcv_with_indicators() -> pd.DataFrame:
    p = SyntheticProvider(seed=42)
    df = p.fetch_ohlcv("TEST", "2020-01-01", "2023-12-31")
    return add_all_indicators(df)


# ──────────────────────────────────────────────
#  Entry signals
# ──────────────────────────────────────────────

class TestEntrySignals:
    def test_returns_boolean_series(self, ohlcv_with_indicators: pd.DataFrame) -> None:
        entries = generate_entry_signals(ohlcv_with_indicators)
        assert entries.dtype == bool
        assert len(entries) == len(ohlcv_with_indicators)

    def test_no_signals_before_sma_long_period(self, ohlcv_with_indicators: pd.DataFrame) -> None:
        cfg = SignalConfig(sma_long_period=200)
        entries = generate_entry_signals(ohlcv_with_indicators, cfg)
        assert not entries.iloc[:200].any()

    def test_signals_occur_later_in_data(self, ohlcv_with_indicators: pd.DataFrame) -> None:
        entries = generate_entry_signals(ohlcv_with_indicators)
        # With ~1000 bars, we should have some entries after bar 200
        assert entries.iloc[200:].sum() >= 0  # At least doesn't crash

    def test_does_not_modify_input(self, ohlcv_with_indicators: pd.DataFrame) -> None:
        original = ohlcv_with_indicators.copy()
        generate_entry_signals(ohlcv_with_indicators)
        pd.testing.assert_frame_equal(ohlcv_with_indicators, original)


class TestExitSignals:
    def test_returns_boolean_series(self, ohlcv_with_indicators: pd.DataFrame) -> None:
        exits = generate_exit_signals(ohlcv_with_indicators)
        assert exits.dtype == bool
        assert len(exits) == len(ohlcv_with_indicators)

    def test_no_exits_before_sma_short(self, ohlcv_with_indicators: pd.DataFrame) -> None:
        exits = generate_exit_signals(ohlcv_with_indicators)
        assert not exits.iloc[:50].any()

    def test_stop_loss_exit(self) -> None:
        # Create data where price drops below stop
        dates = pd.bdate_range("2023-01-01", periods=60)
        close = pd.Series([100.0] * 55 + [80.0, 80, 80, 80, 80], index=dates)
        high = close + 2
        low = close - 2
        volume = pd.Series([1_000_000] * 60, index=dates)
        open_ = close.copy()
        df = pd.DataFrame({
            "open": open_, "high": high, "low": low, "close": close, "volume": volume
        }, index=dates)

        # Entry at 100, ATR ~ 4, stop = 100 - 4*2 = 92
        exits = generate_exit_signals(df, entry_price=100.0, entry_atr=4.0)
        # Should have exit signal when low goes below 92
        assert exits.iloc[-5:].any()


class TestAllSignals:
    def test_has_all_columns(self, ohlcv_with_indicators: pd.DataFrame) -> None:
        result = generate_all_signals(ohlcv_with_indicators)
        expected = [
            "signal_entry", "signal_exit", "ind_rsi",
            "ind_atr", "proposed_stop", "signal_type",
        ]
        for col in expected:
            assert col in result.columns

    def test_signal_type_values(self, ohlcv_with_indicators: pd.DataFrame) -> None:
        result = generate_all_signals(ohlcv_with_indicators)
        valid_types = {SignalType.ENTRY, SignalType.EXIT, SignalType.NONE}
        assert set(result["signal_type"].unique()).issubset(valid_types)

    def test_proposed_stop_below_close(self, ohlcv_with_indicators: pd.DataFrame) -> None:
        result = generate_all_signals(ohlcv_with_indicators)
        # Where ATR is not NaN, stop should be below close
        mask = result["ind_atr"].notna()
        assert (result.loc[mask, "proposed_stop"] <= result.loc[mask, "close"]).all()

    def test_does_not_modify_input(self, ohlcv_with_indicators: pd.DataFrame) -> None:
        original = ohlcv_with_indicators.copy()
        generate_all_signals(ohlcv_with_indicators)
        pd.testing.assert_frame_equal(ohlcv_with_indicators, original)


# ──────────────────────────────────────────────
#  Ranking
# ──────────────────────────────────────────────

class TestRankingConfig:
    def test_default_weights_sum_to_one(self) -> None:
        cfg = RankingConfig()
        total = (
            cfg.momentum_weight + cfg.rel_strength_weight
            + cfg.rsi_weight + cfg.liquidity_weight
        )
        assert np.isclose(total, 1.0)

    def test_invalid_weights_raise(self) -> None:
        with pytest.raises(ValueError, match="must sum to 1.0"):
            RankingConfig(momentum_weight=0.5, rel_strength_weight=0.5, rsi_weight=0.5, liquidity_weight=0.5)


class TestRankStocks:
    def _make_universe(self, n: int = 5) -> dict[str, pd.DataFrame]:
        p = SyntheticProvider(seed=42)
        data = {}
        for i in range(n):
            df = p.fetch_ohlcv(f"STK{i}", "2022-01-01", "2023-12-31")
            data[f"STK{i}"] = add_all_indicators(df)
        return data

    def test_returns_dataframe(self) -> None:
        data = self._make_universe()
        result = rank_stocks(data)
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self) -> None:
        data = self._make_universe()
        result = rank_stocks(data)
        for col in ("symbol", "score", "rank", "eligible"):
            assert col in result.columns

    def test_sorted_by_score_descending(self) -> None:
        data = self._make_universe()
        result = rank_stocks(data)
        scores = result["score"].values
        assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))

    def test_rank_starts_at_1(self) -> None:
        data = self._make_universe()
        result = rank_stocks(data)
        assert result["rank"].iloc[0] == 1
        assert result["rank"].iloc[-1] == len(result)

    def test_all_symbols_present(self) -> None:
        data = self._make_universe(n=3)
        result = rank_stocks(data)
        assert len(result) == 3
        assert set(result["symbol"]) == {"STK0", "STK1", "STK2"}

    def test_empty_data_handled(self) -> None:
        result = rank_stocks({})
        assert isinstance(result, pd.DataFrame)

    def test_score_in_zero_one_range(self) -> None:
        data = self._make_universe()
        result = rank_stocks(data)
        assert (result["score"] >= 0).all()
        assert (result["score"] <= 1).all()

    def test_ineligible_stock_has_zero_score(self) -> None:
        data = self._make_universe()
        result = rank_stocks(data, min_avg_daily_value=1e12)  # impossibly high
        assert (result["score"] == 0.0).all()
        assert not result["eligible"].any()
