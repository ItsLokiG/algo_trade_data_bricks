"""
Paper-trading engine — processes daily data, generates signals,
records intended orders, simulates fills, tracks state.

Does NOT submit real broker orders.
Supports pause/resume and a drawdown kill-switch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from nse_algo.backtest.costs import calculate_transaction_cost
from nse_algo.config import ProjectConfig
from nse_algo.paper.database import PaperDatabase
from nse_algo.strategy.signals import SignalConfig, generate_all_signals

logger = logging.getLogger(__name__)


@dataclass
class PaperEngine:
    """Paper-trading engine with SQLite persistence."""

    config: ProjectConfig
    signal_config: SignalConfig = field(default_factory=SignalConfig)
    db: PaperDatabase = field(default_factory=lambda: PaperDatabase(":memory:"))
    paused: bool = False

    def process_day(
        self,
        date: pd.Timestamp,
        data: dict[str, pd.DataFrame],
    ) -> dict[str, Any]:
        """Process an execution session using only the preceding close for signals."""
        if self.paused:
            return {"status": "paused", "date": str(date)}

        date = pd.Timestamp(date)
        cfg = self.config
        cost_cfg = cfg.costs

        latest_equity = self.db.get_latest_equity()
        cash = latest_equity["cash"] if latest_equity else cfg.capital.initial_capital
        max_equity = self.db.get_state("max_equity", cfg.capital.initial_capital)
        prior_equity = (
            float(latest_equity["total_equity"])
            if latest_equity
            else cfg.capital.initial_capital
        )
        drawdown = (
            (max_equity - prior_equity) / max_equity if max_equity > 0 else 0.0
        )
        entries_enabled = drawdown < cfg.risk.max_drawdown_pct
        if not entries_enabled:
            logger.warning(
                "Drawdown %.1f%% — new paper entries suspended", drawdown * 100
            )

        def transaction_cost(quantity: int, price: float, is_buy: bool):
            return calculate_transaction_cost(
                quantity,
                price,
                is_buy=is_buy,
                brokerage_pct=cost_cfg.brokerage_pct,
                brokerage_cap=cost_cfg.brokerage_cap_per_trade,
                stt_pct=cost_cfg.stt_pct,
                exchange_txn_pct=cost_cfg.exchange_txn_pct,
                gst_pct=cost_cfg.gst_pct,
                sebi_pct=cost_cfg.sebi_pct,
                stamp_duty_pct=cost_cfg.stamp_duty_pct,
                slippage_pct=cost_cfg.slippage_pct,
            )

        new_orders = 0
        for symbol, df in data.items():
            if date not in df.index:
                continue

            prior_bars = df.loc[df.index < date]
            if prior_bars.empty:
                continue
            signal_row = generate_all_signals(
                prior_bars, self.signal_config
            ).iloc[-1]
            row = df.loc[date]

            pos_row = self.db.get_position(symbol)
            has_position = pos_row is not None and pos_row["quantity"] > 0
            if has_position:
                stop_price = float(pos_row["stop_price"])
                stop_hit = stop_price > 0 and row["low"] <= stop_price
                signal_exit = bool(signal_row.get("signal_exit", False))
                if stop_hit or signal_exit:
                    order_id = f"SELL_{symbol}_{date.strftime('%Y%m%d')}"
                    if not self.db.order_exists(order_id):
                        fill_price = (
                            min(float(row["open"]), stop_price)
                            if stop_hit
                            else float(row["open"])
                        )
                        self.db.insert_order(order_id, symbol, "sell", pos_row["quantity"], fill_price, str(date))
                        costs = transaction_cost(pos_row["quantity"], fill_price, False)
                        self.db.insert_fill(order_id, symbol, fill_price, pos_row["quantity"], str(date), costs.total_cost)
                        self.db.update_order_status(order_id, "filled")
                        cash += pos_row["quantity"] * fill_price - costs.total_cost
                        self.db.upsert_position(symbol, 0)
                        new_orders += 1
                else:
                    highest_close = max(
                        float(pos_row["highest_close"]), float(row["close"])
                    )
                    atr_value = signal_row.get("ind_atr", 0.0)
                    if (
                        cfg.risk.trailing_stop_enabled
                        and pd.notna(atr_value)
                        and atr_value > 0
                    ):
                        trailing_stop = (
                            highest_close
                            - atr_value * cfg.risk.trailing_stop_atr_multiple
                        )
                        stop_price = max(stop_price, trailing_stop)
                    self.db.upsert_position(
                        symbol,
                        pos_row["quantity"],
                        pos_row["entry_price"],
                        pos_row["entry_date"],
                        stop_price,
                        highest_close,
                    )

            elif entries_enabled and bool(signal_row.get("signal_entry", False)):
                n_open = len(self.db.get_all_positions())
                if n_open >= cfg.risk.max_positions:
                    continue

                fill_price = float(row["open"])
                stop = signal_row.get("proposed_stop", fill_price * 0.95)
                if pd.isna(stop) or stop <= 0 or stop >= fill_price:
                    stop = fill_price * 0.95

                from nse_algo.backtest.engine import calculate_position_size
                max_pos_val = prior_equity * cfg.risk.max_single_position_pct
                qty = calculate_position_size(
                    equity=prior_equity, entry_price=fill_price,
                    stop_price=stop, risk_per_trade=cfg.risk.risk_per_trade,
                    max_position_value=max_pos_val, available_cash=cash,
                )
                if qty <= 0:
                    continue

                buy_costs = transaction_cost(qty, fill_price, True)
                while qty > 0 and qty * fill_price + buy_costs.total_cost > cash:
                    qty -= 1
                    buy_costs = transaction_cost(qty, fill_price, True)
                if qty <= 0:
                    continue

                order_id = f"BUY_{symbol}_{date.strftime('%Y%m%d')}"
                if not self.db.order_exists(order_id):
                    self.db.insert_order(order_id, symbol, "buy", qty, fill_price, str(date))
                    self.db.insert_fill(order_id, symbol, fill_price, qty, str(date), buy_costs.total_cost)
                    self.db.update_order_status(order_id, "filled")
                    cash -= qty * fill_price + buy_costs.total_cost
                    self.db.upsert_position(
                        symbol, qty, fill_price, str(date), stop, float(row["close"])
                    )
                    new_orders += 1
                    if row["low"] <= stop:
                        exit_price = min(float(row["open"]), stop)
                        exit_order_id = f"SELL_{symbol}_{date.strftime('%Y%m%d')}"
                        exit_costs = transaction_cost(qty, exit_price, False)
                        self.db.insert_order(
                            exit_order_id, symbol, "sell", qty, exit_price, str(date)
                        )
                        self.db.insert_fill(
                            exit_order_id,
                            symbol,
                            exit_price,
                            qty,
                            str(date),
                            exit_costs.total_cost,
                        )
                        self.db.update_order_status(exit_order_id, "filled")
                        cash += qty * exit_price - exit_costs.total_cost
                        self.db.upsert_position(symbol, 0)
                        new_orders += 1

        positions_value = 0.0
        for pos in self.db.get_all_positions():
            df = data.get(pos["symbol"])
            if df is None:
                mark_price = float(pos["entry_price"])
            elif date in df.index:
                mark_price = float(df.loc[date, "close"])
            else:
                past_prices = df.loc[df.index < date, "close"]
                mark_price = (
                    float(past_prices.iloc[-1])
                    if not past_prices.empty
                    else float(pos["entry_price"])
                )
            positions_value += pos["quantity"] * mark_price

        total_equity = cash + positions_value
        max_equity = max(float(max_equity), total_equity)
        self.db.set_state("max_equity", max_equity)
        self.db.insert_equity(str(date), cash, positions_value, total_equity)

        return {
            "status": "drawdown_halt" if not entries_enabled else "ok",
            "date": str(date),
            "cash": cash,
            "positions_value": positions_value,
            "total_equity": total_equity,
            "new_orders": new_orders,
        }

    def pause(self) -> None:
        self.paused = True
        self.db.set_state("paused", True)

    def resume(self) -> None:
        self.paused = False
        self.db.set_state("paused", False)
