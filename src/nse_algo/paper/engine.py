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
        """Process one trading day.

        1. Check and execute pending fills.
        2. Generate signals from completed bars.
        3. Record new orders (not yet filled).
        4. Update positions and equity.

        Returns a summary dict.
        """
        if self.paused:
            return {"status": "paused", "date": str(date)}

        cfg = self.config
        cost_cfg = cfg.costs

        # Get current state
        latest_equity = self.db.get_latest_equity()
        cash = latest_equity["cash"] if latest_equity else cfg.capital.initial_capital
        max_equity = self.db.get_state("max_equity", cfg.capital.initial_capital)

        # Check drawdown kill-switch
        positions = self.db.get_all_positions()
        positions_value = 0.0
        for pos in positions:
            df = data.get(pos["symbol"])
            if df is not None and date in df.index:
                positions_value += pos["quantity"] * df.loc[date, "close"]

        total_equity = cash + positions_value
        if max_equity > 0:
            drawdown = (max_equity - total_equity) / max_equity
            if drawdown >= cfg.risk.max_drawdown_pct:
                self.paused = True
                logger.warning("Drawdown %.1f%% — paper trading paused", drawdown * 100)
                return {"status": "drawdown_halt", "date": str(date), "drawdown": drawdown}

        max_equity = max(max_equity, total_equity)
        self.db.set_state("max_equity", max_equity)

        # Generate signals and orders
        new_orders = 0
        for symbol, df in data.items():
            if date not in df.index:
                continue

            signal_df = generate_all_signals(df, self.signal_config)
            row = signal_df.loc[date]

            # Check exits for open positions
            pos_row = self.db.get_position(symbol)
            if pos_row and pos_row["quantity"] > 0:
                if row.get("signal_exit", False) or (
                    pos_row["stop_price"] > 0 and row["low"] <= pos_row["stop_price"]
                ):
                    order_id = f"SELL_{symbol}_{date.strftime('%Y%m%d')}"
                    if not self.db.order_exists(order_id):
                        fill_price = row["open"] if row.get("signal_exit", False) else pos_row["stop_price"]
                        self.db.insert_order(order_id, symbol, "sell", pos_row["quantity"], fill_price, str(date))
                        costs = calculate_transaction_cost(
                            pos_row["quantity"], fill_price, is_buy=False,
                            brokerage_pct=cost_cfg.brokerage_pct,
                            brokerage_cap=cost_cfg.brokerage_cap_per_trade,
                            stt_pct=cost_cfg.stt_pct,
                        )
                        self.db.insert_fill(order_id, symbol, fill_price, pos_row["quantity"], str(date), costs.total_cost)
                        self.db.update_order_status(order_id, "filled")
                        cash += pos_row["quantity"] * fill_price - costs.total_cost
                        self.db.upsert_position(symbol, 0)
                        new_orders += 1

            # Check entries
            elif row.get("signal_entry", False) and not pos_row:
                n_open = len(self.db.get_all_positions())
                if n_open >= cfg.risk.max_positions:
                    continue

                fill_price = row["open"]
                stop = row.get("proposed_stop", fill_price * 0.95)
                if stop <= 0:
                    stop = fill_price * 0.95

                from nse_algo.backtest.engine import calculate_position_size
                max_pos_val = total_equity * cfg.risk.max_single_position_pct
                qty = calculate_position_size(
                    equity=total_equity, entry_price=fill_price,
                    stop_price=stop, risk_per_trade=cfg.risk.risk_per_trade,
                    max_position_value=max_pos_val, available_cash=cash,
                )
                if qty <= 0:
                    continue

                buy_costs = calculate_transaction_cost(
                    qty, fill_price, is_buy=True,
                    brokerage_pct=cost_cfg.brokerage_pct,
                    brokerage_cap=cost_cfg.brokerage_cap_per_trade,
                    stt_pct=cost_cfg.stt_pct,
                )
                total_cost = qty * fill_price + buy_costs.total_cost
                if total_cost > cash:
                    continue

                order_id = f"BUY_{symbol}_{date.strftime('%Y%m%d')}"
                if not self.db.order_exists(order_id):
                    self.db.insert_order(order_id, symbol, "buy", qty, fill_price, str(date))
                    self.db.insert_fill(order_id, symbol, fill_price, qty, str(date), buy_costs.total_cost)
                    self.db.update_order_status(order_id, "filled")
                    cash -= total_cost
                    self.db.upsert_position(symbol, qty, fill_price, str(date), stop, row["close"])
                    new_orders += 1

        # Update equity snapshot
        self.db.insert_equity(str(date), cash, positions_value, cash + positions_value)

        return {
            "status": "ok",
            "date": str(date),
            "cash": cash,
            "positions_value": positions_value,
            "total_equity": cash + positions_value,
            "new_orders": new_orders,
        }

    def pause(self) -> None:
        self.paused = True
        self.db.set_state("paused", True)

    def resume(self) -> None:
        self.paused = False
        self.db.set_state("paused", False)
