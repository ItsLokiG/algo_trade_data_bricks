"""
Tests for the central configuration module.

These tests validate that:
  - The default config loads without error.
  - All risk limits are within sane ranges.
  - Invalid values are rejected.
  - Leverage and short selling are disabled by default.
  - Transaction-cost rates are non-negative.
  - Data splits leave room for an untouched test period.
"""

import pytest

from nse_algo.config import (
    BacktestConfig,
    CapitalConfig,
    ConfigValidationError,
    CostConfig,
    DataConfig,
    MarketConfig,
    ProjectConfig,
    RiskConfig,
    StrategyConfig,
    get_config,
)


# ──────────────────────────────────────────────
#  Default config loads
# ──────────────────────────────────────────────

class TestDefaultConfig:
    def test_default_config_loads(self) -> None:
        cfg = get_config()
        assert cfg is not None
        assert isinstance(cfg, ProjectConfig)

    def test_project_metadata(self) -> None:
        cfg = get_config()
        assert cfg.project_name == "nse_algo_trading"
        assert cfg.version == "0.1.0"
        assert cfg.phase == 1

    def test_default_config_is_singleton(self) -> None:
        cfg1 = get_config()
        cfg2 = get_config()
        assert cfg1 is cfg2


# ──────────────────────────────────────────────
#  Market scope
# ──────────────────────────────────────────────

class TestMarketConfig:
    def test_defaults(self) -> None:
        m = MarketConfig()
        assert m.exchange == "NSE"
        assert m.instrument == "cash_equity"

    def test_non_nse_rejected(self) -> None:
        with pytest.raises(ConfigValidationError):
            MarketConfig(exchange="BSE")

    def test_derivatives_rejected(self) -> None:
        with pytest.raises(ConfigValidationError):
            MarketConfig(instrument="futures")


# ──────────────────────────────────────────────
#  Capital and leverage
# ──────────────────────────────────────────────

class TestCapitalConfig:
    def test_default_capital(self) -> None:
        c = CapitalConfig()
        assert c.initial_capital == 100_000.0
        assert c.currency == "INR"

    def test_leverage_disabled(self) -> None:
        c = CapitalConfig()
        assert c.leverage_enabled is False

    def test_short_selling_disabled(self) -> None:
        c = CapitalConfig()
        assert c.short_selling_enabled is False

    def test_negative_capital_rejected(self) -> None:
        with pytest.raises(ConfigValidationError):
            CapitalConfig(initial_capital=-1)

    def test_leverage_rejected(self) -> None:
        with pytest.raises(ConfigValidationError):
            CapitalConfig(leverage_enabled=True)


# ──────────────────────────────────────────────
#  Risk limits
# ──────────────────────────────────────────────

class TestRiskConfig:
    def test_default_risk_per_trade(self) -> None:
        r = RiskConfig()
        assert r.risk_per_trade == 0.005  # 0.5 %
        assert r.max_risk_per_trade == 0.01  # 1 % cap

    def test_risk_within_limits(self) -> None:
        r = RiskConfig()
        assert r.risk_per_trade <= r.max_risk_per_trade

    def test_no_leverage(self) -> None:
        r = RiskConfig()
        assert r.max_portfolio_exposure <= 1.0

    def test_risk_exceeding_max_rejected(self) -> None:
        with pytest.raises(ConfigValidationError):
            RiskConfig(risk_per_trade=0.02, max_risk_per_trade=0.01)

    def test_negative_risk_rejected(self) -> None:
        with pytest.raises(ConfigValidationError):
            RiskConfig(risk_per_trade=-0.01)

    def test_exposure_over_100pct_rejected(self) -> None:
        with pytest.raises(ConfigValidationError):
            RiskConfig(max_portfolio_exposure=1.5)

    def test_max_positions_at_least_1(self) -> None:
        with pytest.raises(ConfigValidationError):
            RiskConfig(max_positions=0)

    def test_drawdown_limit_in_range(self) -> None:
        r = RiskConfig()
        assert 0 < r.max_drawdown_pct <= 1.0


# ──────────────────────────────────────────────
#  Transaction costs
# ──────────────────────────────────────────────

class TestCostConfig:
    def test_all_rates_non_negative(self) -> None:
        c = CostConfig()
        assert c.brokerage_pct >= 0
        assert c.stt_pct >= 0
        assert c.exchange_txn_pct >= 0
        assert c.sebi_pct >= 0
        assert c.stamp_duty_pct >= 0
        assert c.slippage_pct >= 0

    def test_rates_not_verified_by_default(self) -> None:
        c = CostConfig()
        assert c.rates_verified is False
        assert c.rates_verified_date is None

    def test_negative_brokerage_rejected(self) -> None:
        with pytest.raises(ConfigValidationError):
            CostConfig(brokerage_pct=-0.001)


# ──────────────────────────────────────────────
#  Data config
# ──────────────────────────────────────────────

class TestDataConfig:
    def test_universe_not_empty(self) -> None:
        d = DataConfig()
        assert len(d.universe) > 0

    def test_benchmark_is_nifty(self) -> None:
        d = DataConfig()
        assert d.benchmark_symbol == "^NSEI"

    def test_empty_universe_rejected(self) -> None:
        with pytest.raises(ConfigValidationError):
            DataConfig(universe=())


# ──────────────────────────────────────────────
#  Strategy config
# ──────────────────────────────────────────────

class TestStrategyConfig:
    def test_sma_short_less_than_long(self) -> None:
        s = StrategyConfig()
        assert s.sma_short_period < s.sma_long_period

    def test_rsi_oversold_less_than_renewed(self) -> None:
        s = StrategyConfig()
        assert s.rsi_oversold < s.rsi_renewed

    def test_short_sma_above_long_rejected(self) -> None:
        with pytest.raises(ConfigValidationError):
            StrategyConfig(sma_short_period=200, sma_long_period=50)

    def test_oversold_above_renewed_rejected(self) -> None:
        with pytest.raises(ConfigValidationError):
            StrategyConfig(rsi_oversold=60, rsi_renewed=50)


# ──────────────────────────────────────────────
#  Backtest splits
# ──────────────────────────────────────────────

class TestBacktestConfig:
    def test_splits_leave_test_room(self) -> None:
        b = BacktestConfig()
        assert b.train_fraction + b.validation_fraction < 1.0

    def test_test_fraction_is_positive(self) -> None:
        b = BacktestConfig()
        test_fraction = 1.0 - b.train_fraction - b.validation_fraction
        assert test_fraction > 0

    def test_overlapping_splits_rejected(self) -> None:
        with pytest.raises(ConfigValidationError):
            BacktestConfig(train_fraction=0.7, validation_fraction=0.4)
