"""Intraday (day-trading) configuration — explicit scope extension.

The original project scope EXCLUDED intraday strategies.  This module is
part of the user-approved scope change documented in
``docs/intraday_smc_strategy.md``.  It is deliberately separate from the
daily/weekly swing configuration so the original system and its tests are
unaffected.

All monetary thresholds are research defaults, not recommendations.
Cost rates must be re-verified against authoritative sources before any
live calculation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from nse_algo.config import (
    ConfigValidationError,
    _validate_positive,
    _validate_range,
)


def _validate_time(value: str, name: str) -> str:
    """Validate an HH:MM 24-hour session time string."""
    if not isinstance(value, str) or len(value) != 5 or value[2] != ":":
        raise ConfigValidationError(f"{name} must be 'HH:MM', got {value!r}")
    hh, mm = value[:2], value[3:]
    if not (hh.isdigit() and mm.isdigit()):
        raise ConfigValidationError(f"{name} must be 'HH:MM', got {value!r}")
    if not (0 <= int(hh) <= 23 and 0 <= int(mm) <= 59):
        raise ConfigValidationError(f"{name} out of range: {value!r}")
    return value


@dataclass(frozen=True)
class IntradayCostConfig:
    """NSE intraday (MIS-style, cash equity) transaction-cost model.

    Research defaults — verify against broker/exchange schedules before
    any live calculation.  Brokerage is charged per executed ORDER
    (a round trip = 2 orders).  STT intraday applies on the SELL side only.
    """

    # Brokerage per order: min(brokerage_pct * value, cap)
    brokerage_pct: float = 0.0003  # 0.03 %
    brokerage_cap_per_order: float = 20.0  # Rs 20

    # STT on intraday equity sells
    stt_sell_pct: float = 0.00025  # 0.025 % on sell side

    # Exchange transaction charges (NSE)
    exchange_txn_pct: float = 0.0000297  # 0.00297 %

    # SEBI turnover fee (Rs 10 per crore)
    sebi_pct: float = 0.000001

    # GST on (brokerage + exchange + SEBI)
    gst_pct: float = 0.18

    # Stamp duty on buy side
    stamp_duty_buy_pct: float = 0.00003  # 0.003 %

    # Assumed price impact per side, as fraction of price
    slippage_pct: float = 0.0005  # 0.05 %

    def __post_init__(self) -> None:
        for attr in (
            "brokerage_pct", "stt_sell_pct", "exchange_txn_pct",
            "sebi_pct", "stamp_duty_buy_pct", "slippage_pct",
        ):
            _validate_range(getattr(self, attr), attr, 0, 0.1, inclusive=True)
        _validate_range(self.gst_pct, "gst_pct", 0, 1.0)
        _validate_positive(self.brokerage_cap_per_order, "brokerage_cap_per_order")


@dataclass(frozen=True)
class IntradayConfig:
    """Parameters for the intraday SMC strategy and backtest.

    Times are IST (Asia/Kolkata) 'HH:MM' strings matched against the
    *bar open timestamps* of NSE sessions (09:15-15:30).
    """

    # ── Data ─────────────────────────────────
    interval: str = "1h"  # '1h' (primary) or '15m' (recent-window check)
    lookback_period: str = "730d"  # yfinance max for 1h; 15m max is '60d'

    # ── SMC structure ────────────────────────
    swing_k: int = 2  # fractal width: k bars each side of a swing
    ob_max_lookback_bars: int = 10  # how far back to find the OB candle
    sweep_max_age_bars: int = 12  # sweep must precede the break by <= this
    choch_max_age_bars: int = 6  # bullish break must be within this many bars

    # ── Session times (IST) ──────────────────
    session_start: str = "09:15"
    entry_start: str = "09:45"  # after opening range completes
    entry_end: str = "14:15"  # leaves >= 1h before square-off
    squareoff_bar_time: str = "15:15"  # final session bar (close = square-off)

    # ── Indicators ────────────────────────────
    atr_period: int = 14  # on continuous intraday bars
    daily_sma_fast: int = 20
    daily_sma_slow: int = 50

    # ── Exits ────────────────────────────────
    stop_atr_buffer: float = 0.1  # stop below zone/sweep by 0.1 * ATR
    target_r_multiple: float = 2.0  # fallback target if no liquidity above

    # ── Opening-range breakout (ORB) setup ── rules adapted from
    # github.com/jakeschincariol/day-trader-skill (trader-entry/-exit/-risk).
    # NSE adaptation: the skill uses a 15-minute range on 5-minute bars on
    # NYSE; with 15m NSE bars the range is the first 30 minutes (2 bars),
    # with 1h bars it degenerates to the single 09:15 bar.
    or_minutes: int = 30  # opening-range length in minutes
    orb_min_volume_x: float = 1.5  # trigger bar volume >= x * avg OR-bar volume
    orb_stop_rule: str = "or_mid"  # 'or_mid' (skill default) or 'or_low'
    orb_enabled: bool = True

    # ── Risk gate (trader-risk / trader-budget rules) ──
    min_reward_risk: float = 1.0  # (target-entry)/(entry-stop) must be >= this
    min_risk_pct_of_price: float = 0.0005  # stop floor: risk < 0.05% of price
    # is too tight to survive the spread (trader-exit refusal rule)
    max_single_position_pct: float = 0.20  # trader-budget position cap analog
    one_entry_per_symbol_per_day: bool = True  # duplicate guard (trader-risk)

    # ── Risk ──────────────────────────────────
    initial_capital: float = 100_000.0
    risk_per_trade: float = 0.005  # 0.5 % of equity per trade (planned)
    max_risk_per_trade: float = 0.01
    max_concurrent_positions: int = 2
    max_entries_per_day: int = 4
    daily_loss_limit_pct: float = 0.02  # halt new entries at -2 % day P&L
    max_volume_participation: float = 0.01  # qty <= 1 % of bar volume

    # ── Universe filters ──────────────────────
    min_avg_daily_turnover_cr: float = 100.0  # 1 crore = 1e7 INR
    min_market_cap_cr: float = 50_000.0
    universe: tuple[str, ...] = (
        "RELIANCE", "HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK",
        "AXISBANK", "INFY", "TCS", "HCLTECH", "LT",
        "ITC", "BHARTIARTL", "HINDUNILVR", "MARUTI", "SUNPHARMA",
    )

    # ── Costs ────────────────────────────────
    costs: IntradayCostConfig = field(default_factory=IntradayCostConfig)

    def __post_init__(self) -> None:
        if self.interval not in ("1h", "15m"):
            raise ConfigValidationError(
                f"interval must be '1h' or '15m', got {self.interval!r}"
            )
        if self.interval == "15m" and self.lookback_period not in ("60d", "59d"):
            raise ConfigValidationError(
                "yfinance 15m data is limited to ~60 days; use lookback_period='60d'."
            )
        for name in ("session_start", "entry_start", "entry_end", "squareoff_bar_time"):
            _validate_time(getattr(self, name), name)
        if self.entry_start >= self.entry_end:
            raise ConfigValidationError("entry_start must be before entry_end")
        if self.entry_start < self.session_start:
            raise ConfigValidationError("entry_start cannot precede session_start")

        if self.swing_k < 1:
            raise ConfigValidationError("swing_k must be >= 1")
        if self.ob_max_lookback_bars < 1:
            raise ConfigValidationError("ob_max_lookback_bars must be >= 1")

        _validate_range(self.risk_per_trade, "risk_per_trade", 0, 0.05)
        _validate_range(self.max_risk_per_trade, "max_risk_per_trade", 0, 0.05)
        if self.risk_per_trade > self.max_risk_per_trade:
            raise ConfigValidationError("risk_per_trade cannot exceed max_risk_per_trade")
        _validate_range(self.daily_loss_limit_pct, "daily_loss_limit_pct", 0, 0.2)
        _validate_range(self.max_volume_participation, "max_volume_participation", 0, 0.5)
        _validate_range(self.stop_atr_buffer, "stop_atr_buffer", 0, 1.0)
        _validate_positive(self.target_r_multiple, "target_r_multiple")
        _validate_positive(self.initial_capital, "initial_capital")

        if self.max_concurrent_positions < 1:
            raise ConfigValidationError("max_concurrent_positions must be >= 1")
        if self.max_entries_per_day < 1:
            raise ConfigValidationError("max_entries_per_day must be >= 1")
        if self.atr_period < 2:
            raise ConfigValidationError("atr_period must be >= 2")
        if self.daily_sma_fast >= self.daily_sma_slow:
            raise ConfigValidationError("daily_sma_fast must be < daily_sma_slow")
        if not self.universe:
            raise ConfigValidationError("universe must not be empty")

        # ORB / risk-gate validation
        if self.or_minutes < 15:
            raise ConfigValidationError("or_minutes must be >= 15")
        _validate_range(self.orb_min_volume_x, "orb_min_volume_x", 0.0, 10.0)  # 0.0 disables the volume filter
        if self.orb_stop_rule not in ("or_mid", "or_low"):
            raise ConfigValidationError("orb_stop_rule must be 'or_mid' or 'or_low'")
        _validate_range(self.min_reward_risk, "min_reward_risk", 0.0, 10.0)
        _validate_range(
            self.min_risk_pct_of_price, "min_risk_pct_of_price", 0.0, 0.05
        )
        _validate_range(
            self.max_single_position_pct, "max_single_position_pct", 0, 1.0
        )
