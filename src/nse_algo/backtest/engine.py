"""
Event-driven backtesting engine for NSE swing trading.

Processes daily bars chronologically.  A signal generated from the
close of day T is executed no earlier than the next session's open.

Implements:
  - Cash and position accounting.
  - Entry/exit order generation.
  - Simulated fills at next-session open.
  - Transaction cost calculation.
  - Stop-loss and trailing stop.
  - Position limits.
  - Daily portfolio valuation.
  - Realised and unrealised P&L.
  - Equity curve.
  - Trade history with exit reasons.

The engine is pure Python (Pandas/NumPy) — no Spark dependency.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from nse_algo.backtest.costs import CostBreakdown, calculate_transaction_cost
from nse_algo.config import ProjectConfig, RiskConfig
from nse_algo.strategy.signals import SignalConfig, generate_all_signals

logger = logging.getLogger(__name__)


class ExitReason(Enum):
    """Reason a position was closed."""
    SIGNAL_EXIT = "signal_exit"
    STOP_LOSS = "stop_loss"
    TRAILING_STOP = "trailing_stop"
    END_OF_DATA = "end_of_data"


@dataclass
class Position:
    """Open position for a single symbol."""
    symbol: str
    quantity: int = 0
    entry_price: float = 0.0
    entry_date: pd.Timestamp | None = None
    stop_price: float = 0.0
    atr_at_entry: float = 0.0
    highest_close: float = 0.0  # for trailing stop

    @property
    def market_value(self) -> float:
        return self.quantity * self.entry_price  # updated externally

    @property
    def is_open(self) -> bool:
        return self.quantity > 0


@dataclass
class Trade:
    """Completed round-trip trade."""
    symbol: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    quantity: int
    pnl: float
    pnl_pct: float
    costs: float
    exit_reason: ExitReason
    holding_days: int


@dataclass
class BacktestResult:
    """Full result of a backtest run."""
    equity_curve: pd.Series
    trades: list[Trade]
    positions_history: pd.DataFrame
    daily_returns: pd.Series
    final_equity: float
    initial_capital: float
    n_trades: int
    n_winning: int
    n_losing: int
    config: ProjectConfig

    @property
    def total_return(self) -> float:
        if self.initial_capital <= 0:
            return 0.0
        return (self.final_equity / self.initial_capital) - 1.0


def calculate_position_size(
    equity: float,
    entry_price: float,
    stop_price: float,
    risk_per_trade: float,
    max_position_value: float,
    available_cash: float,
) -> int:
    """Calculate position size based on risk budget.

    size = (equity * risk_per_trade) / (entry_price - stop_price)

    Constraints:
      - Cannot exceed max_position_value (as shares).
      - Cannot exceed available_cash.
      - Minimum 1 share.

    Does NOT guarantee losses are limited to risk_per_trade — gaps
    and slippage can cause larger losses.
    """
    if entry_price <= 0 or stop_price <= 0:
        return 0
    if entry_price <= stop_price:
        return 0  # no valid stop

    risk_amount = equity * risk_per_trade
    stop_distance = entry_price - stop_price

    if stop_distance <= 0:
        return 0

    size_by_risk = risk_amount / stop_distance
    size_by_max_value = max_position_value / entry_price
    size_by_cash = available_cash / entry_price

    quantity = int(min(size_by_risk, size_by_max_value, size_by_cash))

    return max(quantity, 0)


def run_backtest(
    data: dict[str, pd.DataFrame],
    config: ProjectConfig | None = None,
    signal_config: SignalConfig | None = None,
) -> BacktestResult:
    """Run an event-driven backtest.

    Parameters
    ----------
    data : dict[str, DataFrame]
        Mapping of symbol → OHLCV DataFrame.
    config : ProjectConfig, optional
        Project configuration (capital, risk, costs).
    signal_config : SignalConfig, optional
        Signal generation parameters.

    Returns
    -------
    BacktestResult
    """
    cfg = config or ProjectConfig()
    sig_cfg = signal_config or SignalConfig()

    initial_capital = cfg.capital.initial_capital
    risk_cfg = cfg.risk
    cost_cfg = cfg.costs

    # Generate signals for each symbol
    signal_data: dict[str, pd.DataFrame] = {}
    for symbol, df in data.items():
        signal_data[symbol] = generate_all_signals(df, sig_cfg)

    # Collect all trading dates
    all_dates: pd.DatetimeIndex = pd.DatetimeIndex([])
    for df in signal_data.values():
        all_dates = all_dates.union(df.index)
    all_dates = all_dates.sort_values()

    # State tracking
    cash = initial_capital
    positions: dict[str, Position] = {
        symbol: Position(symbol=symbol) for symbol in data
    }
    trades: list[Trade] = []
    equity_history: list[float] = []
    daily_returns: list[float] = []
    dates_list: list[pd.Timestamp] = []
    positions_records: list[dict[str, Any]] = []
    prev_equity = initial_capital
    new_entry_suspended = False
    max_equity = initial_capital

    for i, date in enumerate(all_dates):
        # Skip if no data for any symbol at this date — process what's available
        # Process exits first, then entries

        # --- Check exits for open positions ---
        symbols_to_close: list[tuple[str, ExitReason, float]] = []
        for symbol, pos in positions.items():
            if not pos.is_open:
                continue
            df = signal_data.get(symbol)
            if df is None or date not in df.index:
                continue

            row = df.loc[date]
            exit_reason: ExitReason | None = None
            exit_price = 0.0

            # Signal exit (close < SMA short)
            if row.get("signal_exit", False):
                exit_reason = ExitReason.SIGNAL_EXIT
                exit_price = row["open"]  # fill at next open (this bar)

            # Stop-loss check (low <= stop)
            elif pos.stop_price > 0 and row["low"] <= pos.stop_price:
                exit_reason = ExitReason.STOP_LOSS
                # Conservative: assume fill at stop price (could be worse with gaps)
                exit_price = pos.stop_price

            # Trailing stop check
            elif risk_cfg.trailing_stop_enabled and pos.highest_close > 0:
                trailing_stop = pos.highest_close - (
                    pos.atr_at_entry * risk_cfg.trailing_stop_atr_multiple
                )
                if row["low"] <= trailing_stop:
                    exit_reason = ExitReason.TRAILING_STOP
                    exit_price = trailing_stop

            if exit_reason is not None:
                symbols_to_close.append((symbol, exit_reason, exit_price))

        # Execute exits
        for symbol, reason, price in symbols_to_close:
            pos = positions[symbol]
            if not pos.is_open:
                continue

            # Calculate costs for sell
            costs = calculate_transaction_cost(
                pos.quantity, price, is_buy=False,
                brokerage_pct=cost_cfg.brokerage_pct,
                brokerage_cap=cost_cfg.brokerage_cap_per_trade,
                stt_pct=cost_cfg.stt_pct,
                exchange_txn_pct=cost_cfg.exchange_txn_pct,
                gst_pct=cost_cfg.gst_pct,
                sebi_pct=cost_cfg.sebi_pct,
                stamp_duty_pct=cost_cfg.stamp_duty_pct,
                slippage_pct=cost_cfg.slippage_pct,
            )

            proceeds = pos.quantity * price - costs.total_cost
            cash += proceeds

            # Record trade
            entry_value = pos.quantity * pos.entry_price
            exit_value = pos.quantity * price
            pnl = exit_value - entry_value - costs.total_cost
            pnl_pct = pnl / entry_value if entry_value > 0 else 0.0

            holding_days = i - 0  # approximate
            # Calculate actual holding days
            if pos.entry_date is not None:
                holding_days = len(all_dates[all_dates >= pos.entry_date]) - len(all_dates[all_dates > date])

            trades.append(Trade(
                symbol=symbol,
                entry_date=pos.entry_date,
                exit_date=date,
                entry_price=pos.entry_price,
                exit_price=price,
                quantity=pos.quantity,
                pnl=pnl,
                pnl_pct=pnl_pct,
                costs=costs.total_cost,
                exit_reason=reason,
                holding_days=holding_days,
            ))

            positions[symbol] = Position(symbol=symbol)  # reset
            logger.debug("Exited %s at %.2f (%s)", symbol, price, reason.value)

        # --- Check entries ---
        current_equity = cash + sum(
            p.quantity * signal_data[s].loc[date, "close"]
            for s, p in positions.items()
            if p.is_open and s in signal_data and date in signal_data[s].index
        )

        # Drawdown check
        if current_equity < max_equity:
            drawdown = (max_equity - current_equity) / max_equity if max_equity > 0 else 0
            if drawdown >= risk_cfg.max_drawdown_pct:
                new_entry_suspended = True
                logger.info("Drawdown %.1f%% — new entries suspended", drawdown * 100)

        # Daily loss check (simplified)
        daily_ret = (current_equity / prev_equity - 1) if prev_equity > 0 else 0.0
        if daily_ret < -risk_cfg.daily_loss_limit_pct:
            new_entry_suspended = True

        max_equity = max(max_equity, current_equity)

        n_open = sum(1 for p in positions.values() if p.is_open)

        if not new_entry_suspended and n_open < risk_cfg.max_positions:
            for symbol, df in signal_data.items():
                if date not in df.index:
                    continue
                row = df.loc[date]

                if not row.get("signal_entry", False):
                    continue
                if positions[symbol].is_open:
                    continue  # already holding

                # Next bar's open is the fill price
                # For simplicity, we fill at the current bar's open (signal from prior close)
                fill_price = row["open"]
                if fill_price <= 0:
                    continue

                stop = row.get("proposed_stop", fill_price * 0.95)
                if stop <= 0:
                    stop = fill_price * 0.95

                atr_val = row.get("ind_atr", 0)
                if atr_val <= 0 or np.isnan(atr_val):
                    atr_val = fill_price * 0.02  # fallback 2%

                max_position_value = current_equity * risk_cfg.max_single_position_pct
                quantity = calculate_position_size(
                    equity=current_equity,
                    entry_price=fill_price,
                    stop_price=stop,
                    risk_per_trade=risk_cfg.risk_per_trade,
                    max_position_value=max_position_value,
                    available_cash=cash,
                )

                if quantity <= 0:
                    continue

                # Calculate buy costs
                buy_costs = calculate_transaction_cost(
                    quantity, fill_price, is_buy=True,
                    brokerage_pct=cost_cfg.brokerage_pct,
                    brokerage_cap=cost_cfg.brokerage_cap_per_trade,
                    stt_pct=cost_cfg.stt_pct,
                    exchange_txn_pct=cost_cfg.exchange_txn_pct,
                    gst_pct=cost_cfg.gst_pct,
                    sebi_pct=cost_cfg.sebi_pct,
                    stamp_duty_pct=cost_cfg.stamp_duty_pct,
                    slippage_pct=cost_cfg.slippage_pct,
                )

                total_cost = quantity * fill_price + buy_costs.total_cost
                if total_cost > cash:
                    # Reduce quantity
                    quantity = int(cash / (fill_price * (1 + cost_cfg.brokerage_pct + cost_cfg.stt_pct + cost_cfg.slippage_pct)))
                    if quantity <= 0:
                        continue
                    buy_costs = calculate_transaction_cost(
                        quantity, fill_price, is_buy=True,
                        brokerage_pct=cost_cfg.brokerage_pct,
                        brokerage_cap=cost_cfg.brokerage_cap_per_trade,
                        stt_pct=cost_cfg.stt_pct,
                        exchange_txn_pct=cost_cfg.exchange_txn_pct,
                        gst_pct=cost_cfg.gst_pct,
                        sebi_pct=cost_cfg.sebi_pct,
                        stamp_duty_pct=cost_cfg.stamp_duty_pct,
                        slippage_pct=cost_cfg.slippage_pct,
                    )
                    total_cost = quantity * fill_price + buy_costs.total_cost

                cash -= total_cost

                positions[symbol] = Position(
                    symbol=symbol,
                    quantity=quantity,
                    entry_price=fill_price,
                    entry_date=date,
                    stop_price=stop,
                    atr_at_entry=atr_val,
                    highest_close=row["close"],
                )
                logger.debug("Entered %s: %d shares @ %.2f, stop %.2f", symbol, quantity, fill_price, stop)
                n_open += 1
                if n_open >= risk_cfg.max_positions:
                    break

        # --- End of day: update trailing stops and record equity ---
        for symbol, pos in positions.items():
            if not pos.is_open:
                continue
            df = signal_data.get(symbol)
            if df is None or date not in df.index:
                continue
            close = df.loc[date, "close"]
            pos.highest_close = max(pos.highest_close, close)

        # Calculate end-of-day equity
        eod_equity = cash
        for symbol, pos in positions.items():
            if not pos.is_open:
                continue
            df = signal_data.get(symbol)
            if df is None or date not in df.index:
                continue
            eod_equity += pos.quantity * df.loc[date, "close"]

        equity_history.append(eod_equity)
        daily_returns.append((eod_equity / prev_equity - 1) if prev_equity > 0 else 0.0)
        dates_list.append(date)
        prev_equity = eod_equity

        positions_records.append({
            "date": date,
            "cash": cash,
            "n_positions": n_open,
            "equity": eod_equity,
        })

    # --- Close remaining positions at end of data ---
    last_date = all_dates[-1] if len(all_dates) > 0 else None
    if last_date is not None:
        for symbol, pos in positions.items():
            if not pos.is_open:
                continue
            df = signal_data.get(symbol)
            if df is None or last_date not in df.index:
                continue
            price = df.loc[last_date, "close"]
            costs = calculate_transaction_cost(
                pos.quantity, price, is_buy=False,
                brokerage_pct=cost_cfg.brokerage_pct,
                brokerage_cap=cost_cfg.brokerage_cap_per_trade,
                stt_pct=cost_cfg.stt_pct,
                exchange_txn_pct=cost_cfg.exchange_txn_pct,
                gst_pct=cost_cfg.gst_pct,
                sebi_pct=cost_cfg.sebi_pct,
                stamp_duty_pct=cost_cfg.stamp_duty_pct,
                slippage_pct=cost_cfg.slippage_pct,
            )
            proceeds = pos.quantity * price - costs.total_cost
            cash += proceeds
            entry_value = pos.quantity * pos.entry_price
            exit_value = pos.quantity * price
            pnl = exit_value - entry_value - costs.total_cost
            pnl_pct = pnl / entry_value if entry_value > 0 else 0.0
            trades.append(Trade(
                symbol=symbol,
                entry_date=pos.entry_date,
                exit_date=last_date,
                entry_price=pos.entry_price,
                exit_price=price,
                quantity=pos.quantity,
                pnl=pnl,
                pnl_pct=pnl_pct,
                costs=costs.total_cost,
                exit_reason=ExitReason.END_OF_DATA,
                holding_days=0,
            ))

    equity_series = pd.Series(equity_history, index=dates_list, name="equity")
    returns_series = pd.Series(daily_returns, index=dates_list, name="daily_return")
    positions_df = pd.DataFrame(positions_records)

    n_winning = sum(1 for t in trades if t.pnl > 0)
    n_losing = sum(1 for t in trades if t.pnl < 0)

    return BacktestResult(
        equity_curve=equity_series,
        trades=trades,
        positions_history=positions_df,
        daily_returns=returns_series,
        final_equity=equity_series.iloc[-1] if len(equity_series) > 0 else initial_capital,
        initial_capital=initial_capital,
        n_trades=len(trades),
        n_winning=n_winning,
        n_losing=n_losing,
        config=cfg,
    )
