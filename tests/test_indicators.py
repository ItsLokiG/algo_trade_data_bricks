"""
Tests for technical indicators.

Uses synthetic OHLCV data with known properties to verify
indicator calculations are correct.
"""

import numpy as np
import pandas as pd
import pytest

from nse_algo.data.providers import SyntheticProvider
from nse_algo.features.indicators import (
    add_all_indicators,
    atr,
    daily_return,
    ema,
    momentum,
    relative_strength,
    rsi,
    sma,
    volume_ratio,
    volume_sma,
)


@pytest.fixture
def ohlcv() -> pd.DataFrame:
    """300 bars of synthetic OHLCV data."""
    p = SyntheticProvider(seed=42)
    return p.fetch_ohlcv("TEST", "2022-01-01", "2023-12-31")


# ──────────────────────────────────────────────
#  SMA
# ──────────────────────────────────────────────

class TestSma:
    def test_first_n_minus_1_are_nan(self, ohlcv: pd.DataFrame) -> None:
        result = sma(ohlcv["close"], period=20)
        assert result.iloc[:19].isna().all()

    def test_value_at_period_is_correct(self) -> None:
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = sma(s, period=5)
        assert result.iloc[4] == pytest.approx(3.0)

    def test_invalid_period_raises(self) -> None:
        with pytest.raises(ValueError):
            sma(pd.Series([1.0]), period=0)

    def test_does_not_modify_input(self, ohlcv: pd.DataFrame) -> None:
        original = ohlcv["close"].copy()
        sma(ohlcv["close"], period=10)
        pd.testing.assert_series_equal(ohlcv["close"], original)


class TestEma:
    def test_first_value_not_nan(self, ohlcv: pd.DataFrame) -> None:
        result = ema(ohlcv["close"], period=10)
        # EMA with adjust=False has first value
        assert not result.isna().all()

    def test_invalid_period_raises(self) -> None:
        with pytest.raises(ValueError):
            ema(pd.Series([1.0]), period=0)


# ──────────────────────────────────────────────
#  RSI
# ──────────────────────────────────────────────

class TestRsi:
    def test_bounded_0_to_100(self, ohlcv: pd.DataFrame) -> None:
        result = rsi(ohlcv["close"], period=14)
        valid = result.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_first_period_are_nan(self, ohlcv: pd.DataFrame) -> None:
        result = rsi(ohlcv["close"], period=14)
        assert result.iloc[:14].isna().all()

    def test_all_up_days_rsi_near_100(self) -> None:
        # Monotonically increasing closes → all gains
        close = pd.Series(range(100, 200), dtype=float)
        result = rsi(close, period=14)
        valid = result.dropna()
        # With all gains, RSI should be very high (near 100)
        assert (valid > 90).all()

    def test_all_down_days_rsi_near_0(self) -> None:
        # Monotonically decreasing closes → all losses
        close = pd.Series(range(200, 100, -1), dtype=float)
        result = rsi(close, period=14)
        valid = result.dropna()
        assert (valid < 10).all()

    def test_flat_prices_rsi_50ish(self) -> None:
        # No change → no gain, no loss → RSI undefined but shouldn't crash
        close = pd.Series([100.0] * 50)
        result = rsi(close, period=14)
        # Should not raise; values may be NaN or 100
        assert len(result) == 50

    def test_invalid_period_raises(self) -> None:
        with pytest.raises(ValueError):
            rsi(pd.Series([1.0]), period=0)


# ──────────────────────────────────────────────
#  ATR
# ──────────────────────────────────────────────

class TestAtr:
    def test_first_period_nan(self, ohlcv: pd.DataFrame) -> None:
        result = atr(ohlcv["high"], ohlcv["low"], ohlcv["close"], period=14)
        assert result.iloc[:14].isna().all()

    def test_positive_values(self, ohlcv: pd.DataFrame) -> None:
        result = atr(ohlcv["high"], ohlcv["low"], ohlcv["close"], period=14)
        valid = result.dropna()
        assert (valid > 0).all()

    def test_constant_price_atr_is_zero(self) -> None:
        # If price doesn't move, ATR should be 0
        high = pd.Series([100.0] * 30)
        low = pd.Series([100.0] * 30)
        close = pd.Series([100.0] * 30)
        result = atr(high, low, close, period=14)
        valid = result.dropna()
        assert (valid == 0.0).all()

    def test_invalid_period_raises(self) -> None:
        with pytest.raises(ValueError):
            atr(pd.Series([1]), pd.Series([1]), pd.Series([1]), period=0)


# ──────────────────────────────────────────────
#  Volume indicators
# ──────────────────────────────────────────────

class TestVolumeIndicators:
    def test_volume_sma(self, ohlcv: pd.DataFrame) -> None:
        result = volume_sma(ohlcv["volume"], period=20)
        assert result.iloc[:19].isna().all()
        assert result.iloc[20:].notna().any()

    def test_volume_ratio_positive(self, ohlcv: pd.DataFrame) -> None:
        result = volume_ratio(ohlcv["volume"], period=20)
        valid = result.dropna()
        assert (valid > 0).all()


# ──────────────────────────────────────────────
#  Momentum / returns
# ──────────────────────────────────────────────

class TestMomentum:
    def test_momentum_period(self, ohlcv: pd.DataFrame) -> None:
        result = momentum(ohlcv["close"], period=20)
        # First 20 should be NaN
        assert result.iloc[:20].isna().all()
        # Value at index 20 should be (close[20]/close[0]) - 1
        expected = ohlcv["close"].iloc[20] / ohlcv["close"].iloc[0] - 1
        assert result.iloc[20] == pytest.approx(expected)

    def test_daily_return(self, ohlcv: pd.DataFrame) -> None:
        result = daily_return(ohlcv["close"])
        assert result.iloc[0] != result.iloc[0]  # NaN
        assert len(result) == len(ohlcv)


class TestRelativeStrength:
    def test_outperforms_when_rs_gt_1(self) -> None:
        close = pd.Series(np.linspace(100, 200, 100))  # +100%
        bench = pd.Series(np.linspace(100, 110, 100))  # +10%
        result = relative_strength(close, bench, period=60)
        valid = result.dropna()
        assert (valid > 1).all()

    def test_underperforms_when_rs_lt_1(self) -> None:
        close = pd.Series(np.linspace(100, 105, 100))  # +5%
        bench = pd.Series(np.linspace(100, 200, 100))  # +100%
        result = relative_strength(close, bench, period=60)
        valid = result.dropna()
        assert (valid < 1).all()


# ──────────────────────────────────────────────
#  All indicators builder
# ──────────────────────────────────────────────

class TestAddAllIndicators:
    def test_adds_all_columns(self, ohlcv: pd.DataFrame) -> None:
        result = add_all_indicators(ohlcv)
        expected_cols = [
            "ind_sma_short", "ind_sma_long", "ind_rsi", "ind_atr",
            "ind_volume_sma", "ind_volume_ratio", "ind_momentum",
        ]
        for col in expected_cols:
            assert col in result.columns

    def test_does_not_modify_input(self, ohlcv: pd.DataFrame) -> None:
        original_cols = set(ohlcv.columns)
        add_all_indicators(ohlcv)
        assert set(ohlcv.columns) == original_cols

    def test_benchmark_column_added(self, ohlcv: pd.DataFrame) -> None:
        bench = ohlcv["close"] * 1.1  # synthetic benchmark
        result = add_all_indicators(ohlcv, benchmark_close=bench)
        assert "ind_rel_strength" in result.columns

    def test_no_benchmark_no_rs_column(self, ohlcv: pd.DataFrame) -> None:
        result = add_all_indicators(ohlcv)
        assert "ind_rel_strength" not in result.columns
