"""
Stock ranking for the NSE trading universe.

Ranks eligible stocks using a documented formula based on:
  - Momentum (recent return)
  - Relative strength vs benchmark
  - RSI position (momentum quality)
  - Liquidity (traded value)

Rankings are calculated only from information available at the decision time.
No future data is used.

The ranking function takes a dict of {symbol: DataFrame with indicators}
and returns a ranked DataFrame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RankingConfig:
    """Parameters for stock ranking."""
    momentum_period: int = 20
    momentum_weight: float = 0.40
    rel_strength_weight: float = 0.30
    rsi_weight: float = 0.15
    liquidity_weight: float = 0.15

    def __post_init__(self) -> None:
        total = (
            self.momentum_weight
            + self.rel_strength_weight
            + self.rsi_weight
            + self.liquidity_weight
        )
        if not np.isclose(total, 1.0):
            raise ValueError(
                f"Ranking weights must sum to 1.0, got {total:.4f}"
            )


def rank_stocks(
    data: dict[str, pd.DataFrame],
    date: pd.Timestamp | str | None = None,
    config: RankingConfig | None = None,
    min_avg_daily_value: float = 5_000_000.0,
) -> pd.DataFrame:
    """Rank stocks by composite score as of a given date.

    Parameters
    ----------
    data : dict[str, DataFrame]
        Mapping of symbol → DataFrame with indicator columns
        (ind_momentum, ind_rel_strength, ind_rsi, volume, close).
    date : Timestamp or str, optional
        Date to evaluate at. If None, uses the last available date.
    config : RankingConfig, optional
        Ranking parameters.
    min_avg_daily_value : float
        Minimum 20-day average traded value to be eligible.

    Returns
    -------
    DataFrame
        Sorted by score descending. Columns: symbol, score,
        momentum, rel_strength, rsi, avg_traded_value, eligible.
    """
    cfg = config or RankingConfig()

    rows: list[dict[str, Any]] = []

    for symbol, df in data.items():
        if df.empty:
            rows.append({
                "symbol": symbol,
                "score": 0.0,
                "eligible": False,
            })
            continue

        # Restrict every feature, including liquidity, to information available
        # by the requested decision date.
        if date is not None:
            target = pd.Timestamp(date)
            valid = df.index[df.index <= target]
            if len(valid) == 0:
                rows.append({
                    "symbol": symbol,
                    "score": 0.0,
                    "eligible": False,
                })
                continue
            as_of = valid[-1]
        else:
            as_of = df.index[-1]
        row = df.loc[as_of]

        # Extract features
        momentum = row.get("ind_momentum", np.nan)
        rel_str = row.get("ind_rel_strength", np.nan)
        rsi_val = row.get("ind_rsi", np.nan)

        # Liquidity: 20-day average traded value
        close = df["close"] if "close" in df.columns else pd.Series(dtype=float)
        volume = df["volume"] if "volume" in df.columns else pd.Series(dtype=float)
        if len(close) > 0 and len(volume) > 0:
            traded_value = (close * volume).loc[:as_of].tail(20)
            avg_traded_value = float(traded_value.mean()) if not traded_value.empty else 0.0
        else:
            avg_traded_value = 0.0

        eligible = avg_traded_value >= min_avg_daily_value

        # Normalise components to [0, 1] range for comparability
        # Momentum: clip to [-0.5, 0.5] then rescale to [0, 1]
        mom_norm = np.clip((momentum + 0.5) / 1.0, 0, 1) if not np.isnan(momentum) else 0

        # Relative strength: clip to [0, 2] then rescale to [0, 1]
        rs_norm = np.clip(rel_str / 2.0, 0, 1) if not np.isnan(rel_str) else 0

        # RSI: already [0, 100], rescale to [0, 1]
        rsi_norm = rsi_val / 100.0 if not np.isnan(rsi_val) else 0

        # Liquidity: log-scale then normalise
        if avg_traded_value > 0:
            liq_norm = np.clip(np.log10(avg_traded_value) / 9.0, 0, 1)  # ₹10^9 cap
        else:
            liq_norm = 0.0

        # Composite score
        score = (
            cfg.momentum_weight * mom_norm
            + cfg.rel_strength_weight * rs_norm
            + cfg.rsi_weight * rsi_norm
            + cfg.liquidity_weight * liq_norm
        )

        if not eligible:
            score = 0.0

        rows.append({
            "symbol": symbol,
            "score": float(score),
            "momentum": float(momentum) if not np.isnan(momentum) else None,
            "rel_strength": float(rel_str) if not np.isnan(rel_str) else None,
            "rsi": float(rsi_val) if not np.isnan(rsi_val) else None,
            "avg_traded_value": avg_traded_value,
            "eligible": eligible,
        })

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("score", ascending=False).reset_index(drop=True)
    result["rank"] = result.index + 1

    return result
