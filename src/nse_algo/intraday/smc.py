"""Smart Money Concepts (SMC) — mechanical, session-scoped implementations.

Covers the four SMC building blocks used by the intraday strategy:

* **Swing points** — fractal highs/lows (k bars each side).
* **BOS / CHoCH** — break-of-structure and change-of-character events.
* **Liquidity sweeps** — wicks beyond a prior swing level that close back
  inside (failed stop-hunts).
* **Bullish order blocks (OB)** — the last down-close candle before a
  bullish structure break; **bullish fair value gaps (FVG)** — 3-candle
  imbalances.

Everything is computed from COMPLETED bars only: a swing at bar j needs k
bars on each side, so it is only usable from bar j + k onward and never
influences earlier decisions.  This module produces features; entry
decisions live in ``signals.py``.

Note: SMC is a popular-retail framework with weak academic validation.
These are deliberate mechanical formalizations of discretionary rules —
they may not capture what every practitioner means by the terms.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class SwingPoint:
    bar: int  # positional index of the swing candle
    price: float
    kind: str  # "HIGH" or "LOW"
    confirmed_at: int  # first bar at which this swing is usable (bar + k)


@dataclass
class StructureEvent:
    bar: int
    kind: str  # "BOS_UP", "CHOCH_UP", "BOS_DOWN", "CHOCH_DOWN"
    level: float  # the broken swing level


@dataclass
class Sweep:
    bar: int
    kind: str  # "LOW" (swept sell-side liquidity) or "HIGH"
    level: float  # the swing level that was wicked through


@dataclass
class Zone:
    kind: str  # "OB" or "FVG"
    bottom: float
    top: float
    created_at: int  # bar at which the zone became actionable
    source_bar: int  # candle that formed the zone

    def retested_at(self, low: float, close: float) -> bool:
        """Price returned into the zone: wick touched it, close held above."""
        return low <= self.top and close > self.bottom


class SessionSMC:
    """SMC analysis of ONE trading session's bars.

    Parameters
    ----------
    df : pd.DataFrame
        Session bars (lowercase OHLCV), chronologically ordered.
    swing_k : int
        Fractal width: a swing candle must exceed its k neighbours on each
        side.  Usable only k bars later (confirmation).
    ob_max_lookback_bars : int
        How far before a structure break to search for the order-block candle.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        swing_k: int = 2,
        ob_max_lookback_bars: int = 10,
    ) -> None:
        required = ("open", "high", "low", "close")
        missing = set(required) - set(df.columns)
        if missing:
            raise ValueError(f"SessionSMC requires columns {sorted(missing)}")
        if len(df) == 0:
            raise ValueError("SessionSMC requires a non-empty session")

        self.df = df.reset_index(drop=True)
        self.n = len(df)
        self.k = swing_k
        self.ob_lookback = ob_max_lookback_bars

        self.highs = self.df["high"].to_numpy(dtype=float)
        self.lows = self.df["low"].to_numpy(dtype=float)
        self.opens = self.df["open"].to_numpy(dtype=float)
        self.closes = self.df["close"].to_numpy(dtype=float)

        # Populated by the state machine below.
        self.swings: list[SwingPoint] = []
        self.events: list[StructureEvent] = []
        self.sweeps: list[Sweep] = []
        self.zones: list[Zone] = []

        self._run()

    # ── Fractal swings ───────────────────────────────────────────
    def _is_fractal_high(self, j: int) -> bool:
        lo, hi = max(0, j - self.k), min(self.n, j + self.k + 1)
        window = self.highs[lo:hi]
        center = self.highs[j]
        strictly_greater = all(center > h for idx, h in enumerate(window) if lo + idx != j)
        # Ties on the same level don't count as clean fractals.
        return strictly_greater

    def _is_fractal_low(self, j: int) -> bool:
        lo, hi = max(0, j - self.k), min(self.n, j + self.k + 1)
        window = self.lows[lo:hi]
        center = self.lows[j]
        strictly_lower = all(center < l for idx, l in enumerate(window) if lo + idx != j)
        return strictly_lower

    # ── State machine ────────────────────────────────────────
    def _run(self) -> None:
        direction: str | None = None
        # Most recent confirmed-but-unbroken swing levels.
        active_high: SwingPoint | None = None
        active_low: SwingPoint | None = None
        # Snapshot of the level that WAS active when a sweep happened,
        # tracked for zone-based stops in signals.py.
        self._sweep_levels: dict[int, float] = {}

        for i in range(self.n):
            # 1) Confirm the fractal whose right window completed at bar i.
            j = i - self.k
            if j >= self.k:
                if self._is_fractal_high(j):
                    swing = SwingPoint(bar=j, price=float(self.highs[j]), kind="HIGH", confirmed_at=i)
                    self.swings.append(swing)
                    # A new swing high further out replaces the active one.
                    if active_high is None or swing.price >= active_high.price:
                        active_high = swing
                if self._is_fractal_low(j):
                    swing = SwingPoint(bar=j, price=float(self.lows[j]), kind="LOW", confirmed_at=i)
                    self.swings.append(swing)
                    if active_low is None or swing.price <= active_low.price:
                        active_low = swing

            # 2) Liquidity sweep of the active low (wick below, close back
            #    above).  Checked BEFORE the break: a sweep is a failed break.
            if active_low is not None and i > active_low.confirmed_at:
                level = active_low.price
                if self.lows[i] < level and self.closes[i] > level:
                    self.sweeps.append(Sweep(bar=i, kind="LOW", level=level))

            # 3) Structure breaks by close.
            if active_high is not None and i > active_high.confirmed_at:
                if self.closes[i] > active_high.price:
                    kind = "BOS_UP" if direction == "bull" else "CHOCH_UP"
                    self.events.append(
                        StructureEvent(bar=i, kind=kind, level=active_high.price)
                    )
                    direction = "bull"
                    active_high = None  # consumed; needs a new swing high
                    self._make_bullish_ob(i)

            if active_low is not None and i > active_low.confirmed_at:
                if self.closes[i] < active_low.price:
                    kind = "BOS_DOWN" if direction == "bear" else "CHOCH_DOWN"
                    self.events.append(
                        StructureEvent(bar=i, kind=kind, level=active_low.price)
                    )
                    direction = "bear"
                    active_low = None

            # 4) Bullish FVG: low[i] > high[i-2] (3-candle imbalance).
            if i >= 2 and self.lows[i] > self.highs[i - 2]:
                self.zones.append(
                    Zone(
                        kind="FVG",
                        bottom=float(self.highs[i - 2]),
                        top=float(self.lows[i]),
                        created_at=i,
                        source_bar=i - 1,
                    )
                )

        self.direction = direction

    def _make_bullish_ob(self, break_bar: int) -> None:
        """Order block: last down-close candle before a bullish break."""
        start = max(0, break_bar - self.ob_lookback)
        for m in range(break_bar - 1, start - 1, -1):
            if self.closes[m] < self.opens[m]:
                self.zones.append(
                    Zone(
                        kind="OB",
                        bottom=float(self.lows[m]),
                        top=float(self.highs[m]),
                        created_at=break_bar,
                        source_bar=m,
                    )
                )
                return

    # ── Queries used by signals.py ────────────────────────────
    def last_bullish_event(self, i: int, max_age: int) -> StructureEvent | None:
        """Most recent BOS_UP/CHOCH_UP within *max_age* bars of *i*."""
        best: StructureEvent | None = None
        for event in self.events:
            if event.bar > i:
                break
            if event.kind in ("BOS_UP", "CHOCH_UP"):
                best = event
        if best is not None and 1 <= i - best.bar <= max_age:
            return best
        return None

    def sweep_before(self, event_bar: int, max_age: int) -> Sweep | None:
        """A LOW sweep that happened at most *max_age* bars before the event."""
        for sweep in reversed(self.sweeps):
            if sweep.bar > event_bar:
                continue
            if event_bar - sweep.bar <= max_age:
                return sweep
            return None
        return None

    def zone_retests_at(self, i: int) -> list[Zone]:
        """Zones whose retest condition is true AT bar *i*.

        A zone is usable only from the bar AFTER its creation (the creation
        bar itself always overlaps the zone) and is invalidated by any close
        below its bottom in between.
        """
        hits: list[Zone] = []
        for zone in self.zones:
            if zone.created_at >= i:
                continue
            # Invalidated if any close between creation and now fell through.
            if self.closes[zone.created_at + 1 : i + 1].min() < zone.bottom:
                continue
            if zone.retested_at(float(self.lows[i]), float(self.closes[i])):
                hits.append(zone)
        return hits

    def swing_highs_upto(self, i: int) -> list[SwingPoint]:
        """Confirmed swing highs with bar <= i (confirmed at or before i)."""
        return [s for s in self.swings if s.kind == "HIGH" and s.confirmed_at <= i]

    def nearest_liquidity_above(self, i: int, price: float) -> float | None:
        """Closest confirmed swing-high level above *price* at bar *i*."""
        levels = [s.price for s in self.swing_highs_upto(i) if s.price > price]
        return min(levels) if levels else None
