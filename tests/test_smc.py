"""Unit tests for the session-scoped SMC primitives.

All fixtures are synthetic bars with KNOWN geometry so expected swings,
breaks, order blocks, FVGs and sweeps can be asserted exactly.
"""

from __future__ import annotations

import pandas as pd
import pytest

from nse_algo.intraday.smc import SessionSMC


def make_session(rows, date="2026-01-05", freq="1h"):
    """rows: list of (open, high, low, close) tuples starting 09:15 IST."""
    start = pd.Timestamp(f"{date} 09:15", tz="Asia/Kolkata")
    index = pd.date_range(start, periods=len(rows), freq=freq)
    df = pd.DataFrame(
        rows, columns=["open", "high", "low", "close"], index=index
    )
    df["volume"] = 1_000_000
    return df


class TestSwings:
    def test_fractal_high_detected_with_confirmation_delay(self):
        # Bar 2 is a clear fractal high; k=2 → confirmed only at bar 4.
        rows = [
            (100, 101, 99, 100),
            (100, 102, 99, 101),
            (101, 105, 100, 104),   # swing high at bar 2
            (104, 104, 100, 101),
            (101, 102, 100, 100),
            (100, 101, 99, 100),
        ]
        smc = SessionSMC(make_session(rows), swing_k=2)
        highs = [s for s in smc.swings if s.kind == "HIGH"]
        assert any(s.bar == 2 and s.price == 105 for s in highs)
        confirmed = [s for s in highs if s.bar == 2][0]
        assert confirmed.confirmed_at == 4  # bar 2 + k

    def test_flat_top_is_not_a_swing(self):
        # Two equal highs: not a strict fractal.
        rows = [
            (100, 102, 99, 100),
            (100, 103, 99, 101),
            (101, 103, 100, 102),
            (102, 103, 100, 101),
            (101, 102, 99, 100),
            (100, 101, 99, 100),
        ]
        smc = SessionSMC(make_session(rows), swing_k=2)
        assert not [s for s in smc.swings if s.kind == "HIGH" and s.price == 103]


class TestStructure:
    def test_choch_up_after_downtrend(self):
        # Swing high 105 at bar 2 (confirmed at bar 4 — k bars later),
        # then price closes above it while direction was not bullish → CHOCH_UP.
        rows = [
            (100, 101, 99, 100),
            (100, 102, 99, 101),
            (101, 105, 100, 104),    # swing high 105 at bar 2
            (104, 104, 100, 101),
            (101, 102, 96, 97),
            (97, 98, 94, 95),
            (95, 96, 93, 94),
            (94, 95, 91, 92),       # swing low 91 at bar 7 (confirmed at 9)
            (92, 99, 91.5, 98),
            (98, 108, 97, 107),     # close 107 > 105 → CHOCH_UP at bar 9
            (107, 108, 100, 101),
            (101, 102, 99, 100),
        ]
        smc = SessionSMC(make_session(rows), swing_k=2)
        ups = [e for e in smc.events if e.kind in ("BOS_UP", "CHOCH_UP")]
        assert ups, "expected an upward structure break"
        assert ups[0].kind == "CHOCH_UP"  # direction was not bullish before
        assert smc.direction == "bull"

    def test_sweep_of_swing_low_detected(self):
        # Swing low 95 at bar 2 (confirmed at bar 4); the sweep bar must come
        # AFTER confirmation, so at bar 6 the wick pierces 95 and closes back
        # above it.
        rows = [
            (100, 101, 99, 100),
            (100, 102, 96, 101),
            (101, 102, 95, 100),   # swing low 95 at bar 2
            (100, 101, 96, 100),
            (100, 100, 96, 99),    # confirmation bar
            (99, 100, 96, 98),
            (98, 99, 94, 97),      # wick below 95, close 97 > 95 → sweep
            (97, 98, 96, 97),
        ]
        smc = SessionSMC(make_session(rows), swing_k=2)
        low_sweeps = [s for s in smc.sweeps if s.kind == "LOW"]
        assert any(s.bar == 6 and s.level == 95 for s in low_sweeps)


class TestZones:
    def test_order_block_before_bullish_break(self):
        rows = [
            (100, 101, 99, 100),
            (100, 102, 99, 101),
            (101, 105, 100, 104),   # swing high 105 at bar 2 (confirmed at 4)
            (104, 104, 100, 101),
            (101, 102, 96, 97),
            (97, 98, 94, 95),
            (95, 96, 93, 94),
            (94, 95, 91, 92),      # down candle — the order block
            (92, 99, 91.5, 98),    # up candle (not the OB)
            (98, 108, 97, 107),   # close above 105 → OB created at bar 9
            (107, 108, 100, 101),
            (101, 102, 99, 100),
        ]
        smc = SessionSMC(make_session(rows), swing_k=2)
        obs = [z for z in smc.zones if z.kind == "OB"]
        assert obs, "expected a bullish order block"
        ob = obs[-1]
        assert ob.source_bar == 7
        assert ob.bottom == 91 and ob.top == 95
        assert ob.created_at == 9

    def test_fvg_created_on_three_candle_gap(self):
        rows = [
            (100, 101, 99, 100),
            (101, 102, 100, 101),   # bar 1: high = 102
            (101, 105, 103, 104),  # bar 2: low 103 > high 101? need bar0 high
            (104, 106, 103, 105),
            (105, 106, 100, 101),
        ]
        # Fix: make the gap explicit — high[0]=101, low[2]=103
        rows = [
            (100, 101, 99, 100),   # high 101
            (100, 102, 99.5, 101),
            (103, 105, 103, 104),  # low 103 > high 101 → FVG [101, 103]
            (104, 106, 103, 105),
            (105, 106, 100, 101),
        ]
        smc = SessionSMC(make_session(rows), swing_k=2)
        fvgs = [z for z in smc.zones if z.kind == "FVG"]
        assert any(z.bottom == 101 and z.top == 103 for z in fvgs)

    def test_zone_invalidated_by_close_below(self):
        rows = [
            (100, 101, 99, 100),
            (100, 102, 99, 101),
            (103, 105, 103, 104),  # FVG [101, 103] created at bar 2
            (104, 105, 103, 104),
            (104, 104.5, 99, 99.5),  # close below 101 → invalidated
            (99, 100, 98, 99),
        ]
        smc = SessionSMC(make_session(rows), swing_k=2)
        hits = smc.zone_retests_at(3)
        assert any(z.kind == "FVG" for z in hits)
        hits_after = smc.zone_retests_at(5)
        assert not any(z.kind == "FVG" and z.bottom == 101 for z in hits_after)

    def test_zone_not_retested_on_creation_bar(self):
        rows = [
            (100, 101, 99, 100),
            (100, 102, 99, 101),
            (103, 105, 103, 104),  # FVG created here
            (104, 105, 103, 104),
        ]
        smc = SessionSMC(make_session(rows), swing_k=2)
        assert smc.zone_retests_at(2) == []  # creation bar itself: no retest


class TestQueries:
    def test_nearest_liquidity_above(self):
        rows = [
            (100, 101, 99, 100),
            (100, 102, 99, 101),
            (101, 103, 100, 102),   # swing high 103 at bar 2 (confirmed at 4)
            (102, 102.5, 96, 97),
            (97, 98, 94, 95),
            (95, 96, 93, 94),
            (94, 95, 91, 92),       # ask at bar 6: only 103 is confirmed
            (92, 99, 91, 98),
            (98, 108, 97, 107),    # swing high 108 at bar 8 (confirmed at 10)
            (107, 107.5, 100, 101),  # below 108 (no fractal tie)
            (101, 102, 99, 100),
        ]
        smc = SessionSMC(make_session(rows), swing_k=2)
        # At bar 6, the 103 swing is confirmed; 108 is not yet.
        assert smc.nearest_liquidity_above(6, 98.0) == 103

    def test_last_bullish_event_respects_max_age(self):
        rows = [
            (100, 101, 99, 100),
            (100, 102, 99, 101),
            (101, 105, 100, 104),   # swing high 105 at bar 2 (confirmed at 4)
            (104, 104, 100, 101),
            (101, 102, 96, 97),
            (97, 98, 94, 95),
            (95, 96, 93, 94),
            (94, 95, 91, 92),
            (98, 108, 97, 107),     # bullish break at bar 8
        ] + [(101, 102, 99, 100)] * 8
        smc = SessionSMC(make_session(rows), swing_k=2)
        assert smc.last_bullish_event(9, max_age=6) is not None
        assert smc.last_bullish_event(15, max_age=6) is None  # too old
