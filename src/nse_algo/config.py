"""
Central configuration for the NSE Algorithmic Trading Research System.

All trading, risk, data, cost, and backtest parameters are defined here.
No hard-coded credentials, tokens, or machine-specific paths.

This module is intentionally pure-Python with no Databricks or Spark
dependencies so it can be imported in both Databricks notebooks and a
local VS Code environment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ──────────────────────────────────────────────
#  Validation helpers
# ──────────────────────────────────────────────

class ConfigValidationError(ValueError):
    """Raised when a configuration value is invalid."""


def _validate_positive(value: float, name: str, *, allow_zero: bool = False) -> float:
    """Ensure *value* is a positive (or non-negative) number."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ConfigValidationError(f"{name} must be a number, got {type(value).__name__}")
    if allow_zero:
        if value < 0:
            raise ConfigValidationError(f"{name} must be >= 0, got {value}")
    else:
        if value <= 0:
            raise ConfigValidationError(f"{name} must be > 0, got {value}")
    return float(value)


def _validate_range(
    value: float, name: str, low: float, high: float, *, inclusive: bool = True
) -> float:
    """Ensure *value* falls within ``[low, high]`` (or ``(low, high)``)."""
    if inclusive:
        ok = low <= value <= high
    else:
        ok = low < value < high
    if not ok:
        bracket = "[]" if inclusive else "()"
        raise ConfigValidationError(
            f"{name} must be in {bracket[0]}{low}, {high}{bracket[1]}, got {value}"
        )
    return float(value)


# ──────────────────────────────────────────────
#  Dataclasses for each config section
# ──────────────────────────────────────────────

@dataclass(frozen=True)
class MarketConfig:
    """Market and instrument scope."""

    exchange: str = "NSE"
    instrument: str = "cash_equity"
    trading_style: str = "swing"
    data_frequency: str = "daily"
    timezone: str = "Asia/Kolkata"

    def __post_init__(self) -> None:
        if self.exchange != "NSE":
            raise ConfigValidationError("Only NSE is supported in the initial scope.")
        if self.instrument != "cash_equity":
            raise ConfigValidationError(
                "Only cash equities are supported in the initial scope. "
                "Futures, options, and leverage are excluded."
            )


@dataclass(frozen=True)
class CapitalConfig:
    """Initial capital and leverage settings."""

    initial_capital: float = 100_000.0  # ₹100,000 default research capital
    currency: str = "INR"
    leverage_enabled: bool = False
    short_selling_enabled: bool = False

    def __post_init__(self) -> None:
        _validate_positive(self.initial_capital, "initial_capital")
        if self.leverage_enabled:
            raise ConfigValidationError("Leverage is disabled in the initial scope.")
        if self.short_selling_enabled:
            raise ConfigValidationError("Short selling is disabled in the initial scope.")


@dataclass(frozen=True)
class RiskConfig:
    """Position sizing and portfolio-level risk limits.

    All risk-per-trade values are *planned* risk.  Gaps, illiquidity, and
    execution failures can produce larger actual losses.
    """

    # Planned risk per trade as a fraction of current equity.
    risk_per_trade: float = 0.005  # 0.5 % default
    max_risk_per_trade: float = 0.01  # configurable upper limit 1 %

    # Portfolio-level limits
    max_positions: int = 10
    max_portfolio_exposure: float = 1.0  # fraction of equity (1.0 = 100 %, no leverage)
    max_single_position_pct: float = 0.20  # max 20 % equity in one stock
    max_sector_concentration: float = 0.40  # max 40 % in one sector (future use)

    # Drawdown controls
    max_drawdown_pct: float = 0.20  # halt new entries after 20 % drawdown
    daily_loss_limit_pct: float = 0.03  # halt new entries after 3 % daily loss

    # Stop-loss
    default_stop_atr_multiple: float = 2.0  # stop = entry - (ATR × multiple)
    trailing_stop_enabled: bool = True
    trailing_stop_atr_multiple: float = 2.0

    def __post_init__(self) -> None:
        _validate_range(self.risk_per_trade, "risk_per_trade", 0, 0.05)
        _validate_range(self.max_risk_per_trade, "max_risk_per_trade", 0, 0.05)
        if self.risk_per_trade > self.max_risk_per_trade:
            raise ConfigValidationError(
                f"risk_per_trade ({self.risk_per_trade}) cannot exceed "
                f"max_risk_per_trade ({self.max_risk_per_trade})"
            )
        _validate_range(self.max_portfolio_exposure, "max_portfolio_exposure", 0, 1.0)
        _validate_range(self.max_single_position_pct, "max_single_position_pct", 0, 1.0)
        _validate_range(self.max_sector_concentration, "max_sector_concentration", 0, 1.0)
        _validate_range(self.max_drawdown_pct, "max_drawdown_pct", 0, 1.0)
        _validate_range(self.daily_loss_limit_pct, "daily_loss_limit_pct", 0, 1.0)
        if self.max_positions < 1:
            raise ConfigValidationError("max_positions must be at least 1.")
        _validate_positive(self.default_stop_atr_multiple, "default_stop_atr_multiple")


@dataclass(frozen=True)
class CostConfig:
    """NSE cash-equity transaction-cost model.

    Rates below are *research defaults* and must be verified against
    authoritative sources before any live-use calculation.
    See: https://www.nseindia.com/get-started/charges-fees-taxes
    """

    # Brokerage
    brokerage_pct: float = 0.0003  # 0.03 % of traded value (discount broker)
    brokerage_cap_per_trade: float = 20.0  # ₹20 cap per trade (discount broker)

    # Securities Transaction Tax (STT) — on the *turn* (buy + sell)
    stt_pct: float = 0.001  # 0.1 % on buy and sell

    # Exchange transaction charges (NSE)
    exchange_txn_pct: float = 0.0000297  # 0.00297 % of traded value

    # GST on (brokerage + exchange charges + SEBI charges)
    gst_pct: float = 0.18  # 18 %

    # SEBI turnover fee
    sebi_pct: float = 0.000001  # ₹10 per crore of traded value

    # Stamp duty (on buy side only)
    stamp_duty_pct: float = 0.00003  # 0.003 % on buy-side value

    # Slippage (assumed price impact per trade, as fraction of price)
    slippage_pct: float = 0.0005  # 0.05 % per side

    # Verification metadata
    rates_verified: bool = False
    rates_verified_date: str | None = None
    rates_source: str = "NSE website (to be verified)"

    def __post_init__(self) -> None:
        for attr in [
            "brokerage_pct", "stt_pct", "exchange_txn_pct",
            "sebi_pct", "stamp_duty_pct", "slippage_pct",
        ]:
            _validate_range(getattr(self, attr), attr, 0, 0.1, inclusive=True)
        _validate_range(self.gst_pct, "gst_pct", 0, 1.0)
        _validate_positive(self.brokerage_cap_per_trade, "brokerage_cap_per_trade")


@dataclass(frozen=True)
class DataConfig:
    """Market-data ingestion and storage settings."""

    # Data provider
    provider: str = "yfinance"  # replaceable via provider interface
    provider_type: str = "unofficial"  # NOT exchange-quality

    # Date range for historical data
    start_date: str = "2018-01-01"
    end_date: str | None = None  # None = today

    # Symbol universe — a small, documented list of liquid NSE stocks
    # These are illustrative defaults, not investment recommendations.
    universe: tuple[str, ...] = (
        "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
        "HINDUNILVR", "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK",
    )

    # Benchmark
    benchmark_symbol: str = "^NSEI"  # NIFTY 50 index

    # Data-quality thresholds
    min_trading_days: int = 250  # at least 1 year of history
    max_missing_pct: float = 0.02  # max 2 % missing sessions
    duplicate_tolerance: int = 0  # zero tolerance for duplicate dates

    # Storage
    catalog: str = "dev_catalog"
    schema: str = "algo"
    bronze_table: str = "bronze_prices"
    silver_table: str = "silver_prices"
    data_quality_table: str = "data_quality_results"

    def __post_init__(self) -> None:
        if not self.universe:
            raise ConfigValidationError("universe must contain at least one symbol.")
        if self.min_trading_days < 1:
            raise ConfigValidationError("min_trading_days must be >= 1.")
        _validate_range(self.max_missing_pct, "max_missing_pct", 0, 1.0)


@dataclass(frozen=True)
class StrategyConfig:
    """Strategy parameters for the initial baseline.

    The baseline hypothesis: liquid stocks in established uptrends that
    pull back and show renewed momentum may outperform the benchmark.
    This is a research hypothesis, not a proven strategy.
    """

    # Moving averages
    sma_short_period: int = 50
    sma_long_period: int = 200

    # RSI
    rsi_period: int = 14
    rsi_oversold: float = 40.0  # pullback threshold
    rsi_renewed: float = 55.0  # momentum resumption threshold

    # ATR
    atr_period: int = 14

    # Volume filter
    min_volume_multiplier: float = 1.5  # vs 20-day average volume

    # Liquidity filter
    min_avg_daily_value: float = 5_000_000.0  # ₹50 lakh avg daily traded value

    # Execution timing
    signal_after_close: bool = True  # signal generated using day-T close
    execution_session: str = "next_open"  # execute at next session's open

    def __post_init__(self) -> None:
        if self.sma_short_period >= self.sma_long_period:
            raise ConfigValidationError(
                f"sma_short_period ({self.sma_short_period}) must be < "
                f"sma_long_period ({self.sma_long_period})"
            )
        _validate_range(self.rsi_oversold, "rsi_oversold", 0, 100)
        _validate_range(self.rsi_renewed, "rsi_renewed", 0, 100)
        if self.rsi_oversold >= self.rsi_renewed:
            raise ConfigValidationError(
                f"rsi_oversold ({self.rsi_oversold}) must be < "
                f"rsi_renewed ({self.rsi_renewed})"
            )


@dataclass(frozen=True)
class BacktestConfig:
    """Backtesting engine configuration."""

    # Chronological data splits (fractions of total date range)
    train_fraction: float = 0.60
    validation_fraction: float = 0.20
    # test_fraction = remaining 20 % (untouched until final evaluation)

    # Walk-forward
    walk_forward_windows: int = 5

    # Monte Carlo (trade-order resampling)
    monte_carlo_runs: int = 1000

    # Execution assumptions
    fill_at: str = "next_open"  # open of next session
    allow_partial_fills: bool = False  # initial version assumes full fills
    slippage_assumption: str = "conservative"  # conservative | none | aggressive

    # Experiment tracking
    experiment_id: str | None = None  # set per run
    record_code_revision: bool = True

    def __post_init__(self) -> None:
        _validate_range(self.train_fraction, "train_fraction", 0, 1.0)
        _validate_range(self.validation_fraction, "validation_fraction", 0, 1.0)
        if self.train_fraction + self.validation_fraction >= 1.0:
            raise ConfigValidationError(
                "train_fraction + validation_fraction must be < 1.0 "
                "to leave room for the untouched test period."
            )
        if self.walk_forward_windows < 1:
            raise ConfigValidationError("walk_forward_windows must be >= 1.")
        if self.monte_carlo_runs < 1:
            raise ConfigValidationError("monte_carlo_runs must be >= 1.")


@dataclass(frozen=True)
class ProjectConfig:
    """Top-level project configuration aggregating all sections."""

    project_name: str = "nse_algo_trading"
    version: str = "0.1.0"
    phase: int = 1

    market: MarketConfig = field(default_factory=MarketConfig)
    capital: CapitalConfig = field(default_factory=CapitalConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    data: DataConfig = field(default_factory=DataConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)

    # Environment flags (set at runtime, not in config files)
    is_databricks: bool = False
    is_local: bool = False

    def __post_init__(self) -> None:
        if self.phase < 1:
            raise ConfigValidationError("phase must be >= 1.")


# ──────────────────────────────────────────────
#  Default singleton
# ──────────────────────────────────────────────

DEFAULT_CONFIG = ProjectConfig()


def get_config(**overrides: Any) -> ProjectConfig:
    """Return a validated ``ProjectConfig``.

    Pass keyword overrides to customise any top-level field, e.g.::

        cfg = get_config()
        cfg = get_config(phase=2)

    For nested overrides, construct the dataclass directly.
    """
    if not overrides:
        return DEFAULT_CONFIG

    current = DEFAULT_CONFIG.__dict__.copy()
    current.update(overrides)
    return ProjectConfig(**current)
