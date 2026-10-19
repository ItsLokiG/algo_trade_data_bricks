"""
NSE cash-equity transaction cost model.

Models all applicable costs for NSE cash-equity delivery trades:
  - Brokerage (with cap)
  - Securities Transaction Tax (STT)
  - Exchange transaction charges
  - GST on (brokerage + exchange + SEBI charges)
  - SEBI turnover fee
  - Stamp duty (buy-side only)
  - Slippage (price impact)

IMPORTANT: Rates are RESEARCH DEFAULTS and must be verified against
authoritative sources before any live-use calculation.
Source: https://www.nseindia.com/get-started/charges-fees-taxes
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CostBreakdown:
    """Detailed cost breakdown for a single transaction."""

    traded_value: float
    brokerage: float
    stt: float
    exchange_charges: float
    gst: float
    sebi_charges: float
    stamp_duty: float
    slippage_cost: float
    total_cost: float

    def __str__(self) -> str:
        lines = [
            f"  Traded value:      ₹{self.traded_value:,.2f}",
            f"  Brokerage:         ₹{self.brokerage:,.2f}",
            f"  STT:               ₹{self.stt:,.2f}",
            f"  Exchange charges:  ₹{self.exchange_charges:,.2f}",
            f"  GST:               ₹{self.gst:,.2f}",
            f"  SEBI charges:      ₹{self.sebi_charges:,.2f}",
            f"  Stamp duty:        ₹{self.stamp_duty:,.2f}",
            f"  Slippage cost:     ₹{self.slippage_cost:,.2f}",
            f"  TOTAL:             ₹{self.total_cost:,.2f}",
        ]
        return "\n".join(lines)


def calculate_transaction_cost(
    quantity: float,
    price: float,
    is_buy: bool,
    *,
    brokerage_pct: float = 0.0003,
    brokerage_cap: float = 20.0,
    stt_pct: float = 0.001,
    exchange_txn_pct: float = 0.0000297,
    gst_pct: float = 0.18,
    sebi_pct: float = 0.000001,
    stamp_duty_pct: float = 0.00003,
    slippage_pct: float = 0.0005,
) -> CostBreakdown:
    """Calculate all transaction costs for a single NSE cash-equity trade.

    Parameters
    ----------
    quantity : float
        Number of shares.
    price : float
        Execution price per share.
    is_buy : bool
        True for buy, False for sell.
    brokerage_pct : float
        Brokerage as fraction of traded value.
    brokerage_cap : float
        Maximum brokerage per trade (₹).
    stt_pct : float
        STT as fraction of traded value (applied on both buy and sell).
    exchange_txn_pct : float
        Exchange transaction charges as fraction of traded value.
    gst_pct : float
        GST rate on (brokerage + exchange charges + SEBI charges).
    sebi_pct : float
        SEBI turnover fee as fraction of traded value.
    stamp_duty_pct : float
        Stamp duty as fraction of traded value (buy-side only).
    slippage_pct : float
        Assumed slippage as fraction of price.

    Returns
    -------
    CostBreakdown
        Detailed cost breakdown.
    """
    if quantity <= 0:
        return CostBreakdown(0, 0, 0, 0, 0, 0, 0, 0, 0)
    if price <= 0:
        raise ValueError(f"price must be > 0, got {price}")

    traded_value = quantity * price

    # Brokerage (capped)
    brokerage = min(traded_value * brokerage_pct, brokerage_cap)

    # STT (both buy and sell)
    stt = traded_value * stt_pct

    # Exchange transaction charges
    exchange_charges = traded_value * exchange_txn_pct

    # SEBI turnover fee
    sebi_charges = traded_value * sebi_pct

    # GST on (brokerage + exchange + SEBI)
    gst = (brokerage + exchange_charges + sebi_charges) * gst_pct

    # Stamp duty (buy-side only)
    stamp_duty = traded_value * stamp_duty_pct if is_buy else 0.0

    # Slippage cost
    slippage_cost = traded_value * slippage_pct

    total = (
        brokerage + stt + exchange_charges + gst
        + sebi_charges + stamp_duty + slippage_cost
    )

    return CostBreakdown(
        traded_value=traded_value,
        brokerage=brokerage,
        stt=stt,
        exchange_charges=exchange_charges,
        gst=gst,
        sebi_charges=sebi_charges,
        stamp_duty=stamp_duty,
        slippage_cost=slippage_cost,
        total_cost=total,
    )


def round_to_lot_size(quantity: float, lot_size: int = 1) -> int:
    """Round quantity down to the nearest valid lot size."""
    if lot_size < 1:
        raise ValueError(f"lot_size must be >= 1, got {lot_size}")
    return int(quantity // lot_size) * lot_size
