"""Intraday backtest engine for the NSE SMC + ORB day-trading strategy.

Anti-look-ahead execution, modelled after the lessons of the swing-system
look-ahead bug (Oct 2026) and the day-trader-skill execution rules:

* a signal decided at the CLOSE of bar t fills at the OPEN of bar t+1
  (plus slippage) — never on the signal bar;
* intrabar stop/target checks are CONSERVATIVE: if both the stop and the
  target are touchable within one bar, the stop is assumed to hit first;
* every position is force-squared-off on the final bar of its session —
  no overnight holds, ever;
* trader-risk rules at fill time: stop required and below the fill,
  reward:risk >= ``min_reward_risk`` measured at ACTUAL fill prices,
  risk-per-share floor, one entry per symbol per day, max entries/day,
  max concurrent positions, and a daily loss limit that counts the FULL
  planned max loss of every entry placed today (win or lose);
* trader-budget sizing: shares = min(max-loss budget, position cap,
  volume participation cap, available cash) — the stop decides the size.

The engine simulates.  It never places real orders, and no result here
should be read as a promise of live performance.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from nse_algo.intraday.config import IntradayConfig, IntradayCostConfig

logger = logging.getLogger(__name__)


@dataclass
class _Position:
    symbol: str
    qty: int
    entry_price: float
    stop: float
    target: float
    entry_time: pd.Timestamp
    entry_bar: int
    planned_risk: float  # qty * (entry - stop), the full max loss
    setup: str


@dataclass
class IntradayResult:
    trades: pd.DataFrame
    equity_curve: pd.Series
    config_summary: dict

    def summary(self) -> dict:
        """Honest performance metrics.  Small samples are labelled as such."""
        trades = self.trades
        n = len(trades)
        out = {
            "n_trades": n,
            "warning": (
                "Sample too small for statistical conclusions (< 30 trades)"
                if n < 30
                else ""
            ),
        }
        if n == 0:
            out.update(
                {
                    "total_return_pct": 0.0,
                    "win_rate": float("nan"),
                    "profit_factor": float("nan"),
                    "expectancy_inr": float("nan"),
                }
            )
            return out

        wins = trades[trades["pnl_net"] > 0]
        losses = trades[trades["pnl_net"] <= 0]
        gross_profit = float(wins["pnl_net"].sum())
        gross_loss = abs(float(losses["pnl_net"].sum()))

        out.update(
            {
                "total_return_pct": (
                    float(self.equity_curve.iloc[-1] / self.equity_curve.iloc[0] - 1.0) * 100.0
                    if len(self.equity_curve) > 1
                    else 0.0
                ),
                "win_rate": len(wins) / n * 100.0,
                "profit_factor": (
                    gross_profit / gross_loss if gross_loss > 0 else float("inf")
                ),
                "expectancy_inr": float(trades["pnl_net"].mean()),
                "avg_win_inr": float(wins["pnl_net"].mean()) if len(wins) else 0.0,
                "avg_loss_inr": float(losses["pnl_net"].mean()) if len(losses) else 0.0,
                "total_costs_inr": float(trades["costs"].sum()),
                "total_gross_pnl_inr": float(trades["pnl_gross"].sum()),
                "cost_share_of_gross": (
                    float(trades["costs"].sum() / trades["pnl_gross"].abs().sum() * 100.0)
                    if trades["pnl_gross"].abs().sum() > 0
                    else float("nan")
                ),
                "avg_r_multiple": float(trades["r_multiple"].mean()),
                "avg_holding_bars": float(trades["bars_held"].mean()),
                "exit_reasons": trades["exit_reason"].value_counts().to_dict(),
                "by_setup": {
                    setup: {
                        "n": int(len(group)),
                        "net_pnl": float(group["pnl_net"].sum()),
                        "win_rate": (
                            float((group["pnl_net"] > 0).mean() * 100.0)
                            if len(group)
                            else float("nan")
                        ),
                    }
                    for setup, group in trades.groupby("setup")
                },
            }
        )

        curve = self.equity_curve
        if len(curve) > 1:
            running_max = curve.cummax()
            dd = (curve / running_max - 1.0).min()
            out["max_drawdown_pct"] = float(dd) * 100.0
        return out


class IntradayBacktester:
    """Bar-driven intraday portfolio simulator (long-only, cash only)."""

    def __init__(self, config: IntradayConfig) -> None:
        self.config = config
        self.costs: IntradayCostConfig = config.costs

    # ── Cost model ─────────────────────────────────────────
    def _buy_costs(self, value: float) -> float:
        brokerage = min(
            self.costs.brokerage_pct * value, self.costs.brokerage_cap_per_order
        )
        exchange = self.costs.exchange_txn_pct * value
        sebi = self.costs.sebi_pct * value
        gst = self.costs.gst_pct * (brokerage + exchange + sebi)
        stamp = self.costs.stamp_duty_buy_pct * value
        return brokerage + exchange + sebi + gst + stamp

    def _sell_costs(self, value: float) -> float:
        brokerage = min(
            self.costs.brokerage_pct * value, self.costs.brokerage_cap_per_order
        )
        exchange = self.costs.exchange_txn_pct * value
        sebi = self.costs.sebi_pct * value
        gst = self.costs.gst_pct * (brokerage + exchange + sebi)
        stt = self.costs.stt_sell_pct * value
        return brokerage + exchange + sebi + gst + stt

    # ── Main loop ────────────────────────────────────────
    def run(self, signals_by_symbol: dict[str, pd.DataFrame]) -> IntradayResult:
        cfg = self.config
        slip = self.costs.slippage_pct

        cash = cfg.initial_capital
        positions: dict[str, _Position] = {}
        pending: dict[str, dict] = {}  # symbol -> {stop, target, setup}

        trades: list[dict] = []
        equity_points: list[tuple] = []

        symbols = sorted(signals_by_symbol)

        # Chronology: timestamp -> {symbol: (bar_index, row)}.
        timeline: dict[pd.Timestamp, dict[str, tuple[int, pd.Series]]] = {}
        for symbol in symbols:
            df = signals_by_symbol[symbol]
            for idx, (ts, row) in enumerate(df.iterrows()):
                timeline.setdefault(ts, {})[symbol] = (idx, row)

        last_bar_of_session: dict[str, set] = {s: set() for s in symbols}
        for symbol in symbols:
            df = signals_by_symbol[symbol]
            for date in df.index.normalize().unique():
                session = df[df.index.normalize() == date]
                last_bar_of_session[symbol].add(session.index[-1])

        current_day: pd.Timestamp | None = None
        day_start_equity = cfg.initial_capital
        entries_today = 0
        symbols_entered_today: set = set()
        committed_risk_today = 0.0
        equity = cfg.initial_capital

        for ts in sorted(timeline):
            day = ts.normalize()
            if day != current_day:
                current_day = day
                day_start_equity = equity
                entries_today = 0
                symbols_entered_today = set()
                committed_risk_today = 0.0

            bars_here = timeline[ts]

            for symbol in symbols:
                if symbol not in bars_here:
                    continue
                bar_idx, row = bars_here[symbol]
                o = float(row["open"])
                h = float(row["high"])
                l = float(row["low"])
                c = float(row["close"])

                # ── 1) Fill a pending entry at this bar's open ──
                if symbol in pending and symbol not in positions:
                    order = pending.pop(symbol)
                    stop, target = order["stop"], order["target"]
                    fill_price = o * (1.0 + slip)

                    risk_share = fill_price - stop
                    reward_share = target - fill_price
                    daily_headroom = (
                        cfg.daily_loss_limit_pct * day_start_equity
                        - committed_risk_today
                    )
                    risk_budget = min(equity * cfg.risk_per_trade, daily_headroom)

                    qty = 0
                    if risk_share > 0:
                        qty_candidates = [
                            math.floor(risk_budget / risk_share),
                            math.floor(
                                cfg.max_single_position_pct * equity / fill_price
                            ),
                            math.floor(cash / fill_price),
                        ]
                        bar_volume = float(row["volume"])
                        if bar_volume > 0:
                            qty_candidates.append(
                                math.floor(cfg.max_volume_participation * bar_volume)
                            )
                        qty = max(0, min(qty_candidates))

                    gates_ok = (
                        qty >= 1
                        and risk_share > 0
                        and risk_share >= cfg.min_risk_pct_of_price * fill_price
                        and reward_share / risk_share >= cfg.min_reward_risk
                        and entries_today < cfg.max_entries_per_day
                        and len(positions) < cfg.max_concurrent_positions
                        and (
                            not cfg.one_entry_per_symbol_per_day
                            or symbol not in symbols_entered_today
                        )
                    )

                    if gates_ok:
                        buy_value = qty * fill_price
                        buy_cost = self._buy_costs(buy_value)
                        cash -= buy_value + buy_cost
                        planned_risk = qty * risk_share
                        committed_risk_today += planned_risk
                        entries_today += 1
                        symbols_entered_today.add(symbol)
                        positions[symbol] = _Position(
                            symbol=symbol,
                            qty=qty,
                            entry_price=fill_price,
                            stop=stop,
                            target=target,
                            entry_time=ts,
                            entry_bar=bar_idx,
                            planned_risk=planned_risk,
                            setup=order["setup"],
                        )
                    else:
                        logger.debug("Entry gate blocked for %s at %s", symbol, ts)

                # ── 2) Manage an open position on this bar ──
                if symbol in positions:
                    pos = positions[symbol]
                    is_last_bar = ts in last_bar_of_session[symbol]
                    exit_price = None
                    exit_reason = ""

                    # Conservative: stop before target within the same bar.
                    if l <= pos.stop:
                        exit_price = pos.stop * (1.0 - slip)
                        exit_reason = "STOP"
                    elif h >= pos.target:
                        exit_price = pos.target * (1.0 - slip)
                        exit_reason = "TARGET"
                    elif is_last_bar:
                        exit_price = c * (1.0 - slip)
                        exit_reason = "SQUAREOFF"

                    if exit_price is not None:
                        sell_value = pos.qty * exit_price
                        sell_cost = self._sell_costs(sell_value)
                        cash += sell_value - sell_cost

                        pnl_gross = pos.qty * (exit_price - pos.entry_price)
                        buy_cost = self._buy_costs(pos.qty * pos.entry_price)
                        pnl_net = pnl_gross - buy_cost - sell_cost

                        risk_share = pos.entry_price - pos.stop
                        r_multiple = (
                            (exit_price - pos.entry_price) / risk_share
                            if risk_share > 0
                            else float("nan")
                        )

                        trades.append(
                            {
                                "symbol": symbol,
                                "setup": pos.setup,
                                "entry_time": pos.entry_time,
                                "exit_time": ts,
                                "qty": pos.qty,
                                "entry_price": pos.entry_price,
                                "exit_price": exit_price,
                                "stop": pos.stop,
                                "target": pos.target,
                                "pnl_gross": pnl_gross,
                                "costs": buy_cost + sell_cost,
                                "pnl_net": pnl_net,
                                "r_multiple": r_multiple,
                                "bars_held": bar_idx - pos.entry_bar,
                                "exit_reason": exit_reason,
                            }
                        )
                        del positions[symbol]

                # ── 3) New signal at this bar's close → pending for next bar ──
                if (
                    bool(row.get("entry", False))
                    and symbol not in positions
                    and symbol not in pending
                    and pd.notna(row.get("stop", np.nan))
                    and pd.notna(row.get("target", np.nan))
                    and ts not in last_bar_of_session[symbol]
                ):
                    pending[symbol] = {
                        "stop": float(row["stop"]),
                        "target": float(row["target"]),
                        "setup": str(row.get("setup", "")),
                    }

            # Mark equity at this timestamp.
            marked = cash
            for symbol, pos in positions.items():
                if symbol in bars_here:
                    marked += pos.qty * float(bars_here[symbol][1]["close"])
                else:
                    df = signals_by_symbol[symbol]
                    prior = df[df.index < ts]
                    if len(prior):
                        marked += pos.qty * float(prior["close"].iloc[-1])
            equity = marked
            equity_points.append((ts, equity))

        for symbol in list(positions):
            logger.error(
                "Position in %s left open at end of data — data gap?", symbol
            )

        trades_df = pd.DataFrame(trades)
        equity_curve = (
            pd.Series(
                [p[1] for p in equity_points],
                index=pd.DatetimeIndex([p[0] for p in equity_points]),
            )
            .groupby(level=0)
            .last()
        )
        equity_curve.index.name = "bar_time"

        return IntradayResult(
            trades=trades_df,
            equity_curve=equity_curve,
            config_summary={
                "interval": cfg.interval,
                "risk_per_trade": cfg.risk_per_trade,
                "max_entries_per_day": cfg.max_entries_per_day,
                "orb_enabled": cfg.orb_enabled,
                "slippage_pct": slip,
            },
        )
