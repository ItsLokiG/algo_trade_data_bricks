"""Monthly cross-sectional momentum rotation — second strategy book (built Nov 2026).

Long-only top-N momentum rotation over the large-cap universe (same data cache
and cost model as daily_smc). Decision at the month-end CLOSE, fills at the
NEXT trading day's OPEN. Equal weight, full NSE delivery costs, daily marks.

Validation (per-year fresh runs, walk-forward chained OOS, honest):
* Walk-forward OOS 2022-26 (select variant each Jan by trailing 3y return
  over 10 candidates): +58.1% chained vs NIFTY +29.1% and C10 +47.0%.
* Fixed t10/ungated/12-1 (this default config): +62.6% OOS 2022-26.
* Correlation with C10 daily returns ≈ 0.51 → genuine diversification.

Honest limitations (do not remove):
* SURVIVORSHIP INFLATION IS SEVERE HERE: ranking by past returns on TODAY'S
  constituents systematically picks the survivors/winners of the last 8
  years. Absolute returns (+462% full 2019-26 for the default config) are
  upper bounds, not expectations. The 2022-26 OOS window is less inflated
  but still inflated.
* 2026: ALL 10 tested variants were negative (-3% to -20%); the momentum
  edge is currently in drawdown. 2025 was flat (+3.4% default).
* The walk-forward selector picks the most aggressive variants (6-1 / 12-0
  lookbacks) which are also the most survivorship-inflated; the default
  config here is the more conservative fixed 12-1, ungated.
* The NIFTY>SMA40 regime gate HURTS this strategy (misses recoveries):
  ungated +462% vs gated +133% full period. Default use_gate=False.
* This is research/paper infrastructure, not investment advice.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from daily_smc import (LARGE_CAPS, load_universe, load_nifty, trade_costs,
                       summary, yearly_returns)  # shared infrastructure


@dataclass(frozen=True)
class MomentumConfig:
    """Validated default: top-10, ungated, 12-1 lookback."""
    top_n: int = 10
    use_gate: bool = False          # NIFTY>SMA40 gate — validated to HURT here
    lb_months: int = 12             # lookback span in months (21 bars/month)
    skip_months: int = 1           # skip most recent month(s)
    weight_pct: float = 0.10        # per-position fraction of equity at each rebalance
    capital: float = 1_000_000.0


def month_end_pairs(cal: pd.DatetimeIndex) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """(month-end decision day T, next trading day T2) pairs."""
    ends = [d for d in cal if d.month != (d + pd.Timedelta(days=1)).month]
    return [(T, cal[cal.get_loc(T) + 1]) for T in ends[:-1]]


def momentum_rank(uni: dict[str, pd.DataFrame], T: pd.Timestamp, cfg: MomentumConfig) -> list[str]:
    """Rank large caps by lb-skip momentum decided at close of T. No look-ahead."""
    span, skipb = cfg.lb_months * 21, cfg.skip_months * 21
    cands = []
    for s, df in uni.items():
        if s not in LARGE_CAPS or T not in df.index:
            continue
        i = df.index.get_loc(T)
        if i < span:
            continue
        m = df["close"].iloc[i - skipb] / df["close"].iloc[i - span] - 1.0
        cands.append((m, s))
    cands.sort(reverse=True)
    return [s for _, s in cands[:cfg.top_n]]


def run_momentum_backtest(uni: dict[str, pd.DataFrame], nifty: pd.DataFrame,
                          cfg: MomentumConfig, start: str = "2018-01-01",
                          end: str = "2026-12-31") -> tuple[pd.Series, list[dict]]:
    """Anti-look-ahead monthly rotation backtest. Returns (equity, trades)."""
    cal = nifty.index
    bull = nifty["close"] > nifty["close"].rolling(40).mean()
    pairs = [(T, T2) for T, T2 in month_end_pairs(cal) if start <= str(T2.date()) <= end]
    days = [d for d in cal if start <= str(d.date()) <= end]

    targets = {}
    for T, T2 in pairs:
        ranked = momentum_rank(uni, T, cfg)
        targets[T2] = ranked if (not cfg.use_gate or bool(bull.loc[T])) else []

    holdings: dict[str, int] = {}
    cash, last_close = cfg.capital, {}
    trades: list[dict] = []
    equity: dict[pd.Timestamp, float] = {}

    for d in days:
        if d in targets:
            tgt = set(targets[d])
            for s in list(holdings):
                if s not in tgt and d in uni[s].index:
                    px = uni[s].loc[d, "open"]
                    cash += holdings[s] * px - trade_costs(holdings[s] * px, False)
                    trades.append(dict(symbol=s, side="SELL", date=str(d.date()),
                                       qty=holdings[s], price=float(px)))
                    del holdings[s]
            eq_sz = cash + sum(q * last_close.get(s, np.nan) for s, q in holdings.items())
            budget = cfg.weight_pct * eq_sz
            for s in targets[d]:
                if s in holdings or d not in uni[s].index:
                    continue
                px = uni[s].loc[d, "open"]
                qty = int(min(budget / px, cash / (px * 1.01)))
                if qty <= 0 or qty * px > cash:
                    continue
                cash -= qty * px + trade_costs(qty * px, True)
                holdings[s] = qty
                trades.append(dict(symbol=s, side="BUY", date=str(d.date()),
                                   qty=qty, price=float(px)))
        for s, q in holdings.items():
            if d in uni[s].index:
                last_close[s] = uni[s].loc[d, "close"]
        equity[d] = cash + sum(q * last_close.get(s, np.nan) for s, q in holdings.items())

    return pd.Series(equity).sort_index().dropna(), trades
