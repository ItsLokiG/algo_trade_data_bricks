"""Tests for the intraday backtest engine and the day-trader-skill rules
(ORB trigger semantics, refusal gates, risk accounting, square-off).

Engine fixtures are hand-crafted frames with entry/stop/target columns so
execution mechanics are tested in isolation from signal generation.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from nse_algo.intraday.config import IntradayConfig
from nse_algo.intraday.engine import IntradayBacktester
from nse_algo.intraday.fundamentals import daily_regime
from nse_algo.intraday.signals import generate_signals


def make_bars(rows, date="2026-01-05", freq="1h", volume=1_000_000):
    """rows: (open, high, low, close) starting 09:15 IST."""
    start = pd.Timestamp(f"{date} 09:15", tz="Asia/Kolkata")
    index = pd.date_range(start, periods=len(rows), freq=freq)
    df = pd.DataFrame(
        rows, columns=["open", "high", "low", "close"], index=index
    )
    df["volume"] = volume
    return df


def add_signal(df, bar_time, stop, target, setup="SMC"):
    df["entry"] = False
    df["stop"] = np.nan
    df["target"] = np.nan
    df["setup"] = ""
    df.loc[bar_time, "entry"] = True
    df.loc[bar_time, "stop"] = stop
    df.loc[bar_time, "target"] = target
    df.loc[bar_time, "setup"] = setup
    return df


class TestExecutionMechanics:
    def test_fill_at_next_bar_open_no_lookahead(self):
        # Signal at the first bar's CLOSE → fills at the SECOND bar's open.
        # Target is 3.5 above the signal close so reward:risk stays >= 1 even
        # at the (higher) fill price — the trader-risk gate is measured at
        # ACTUAL fill prices, not the signal close.
        bars = make_bars([
            (100, 101, 99, 100),          # signal bar
            (100.5, 104, 100.2, 102.5),   # fill + target hit
            (102, 103, 101, 102),
            (102, 103, 101, 101),
            (101, 102, 100, 101),
            (101, 102, 100, 100),
            (100, 101, 99, 100),
        ])
        bars = add_signal(bars, bars.index[0], stop=99.0, target=103.5)

        result = IntradayBacktester(IntradayConfig()).run({"TEST": bars})
        assert len(result.trades) == 1
        trade = result.trades.iloc[0]

        slip = IntradayConfig().costs.slippage_pct
        assert trade["entry_time"] == bars.index[1]  # next bar, not signal bar
        assert trade["entry_price"] == pytest.approx(100.5 * (1 + slip))
        assert trade["exit_reason"] == "TARGET"

    def test_stop_hits_before_target_in_same_bar(self):
        # Bar 1 touches BOTH the stop and the target → conservative: STOP.
        bars = make_bars([
            (100, 101, 99, 100),
            (100.5, 105, 98, 100),   # low <= stop, high >= target
            (100, 101, 99, 100),
            (100, 101, 99, 100),
            (100, 101, 99, 100),
            (100, 101, 99, 100),
            (100, 101, 99, 100),
        ])
        bars = add_signal(bars, bars.index[0], stop=99.0, target=103.0)

        result = IntradayBacktester(IntradayConfig()).run({"TEST": bars})
        trade = result.trades.iloc[0]
        assert trade["exit_reason"] == "STOP"
        slip = IntradayConfig().costs.slippage_pct
        assert trade["exit_price"] == pytest.approx(99.0 * (1 - slip))

    def test_squareoff_on_last_bar_of_session(self):
        # No stop or target touched → forced square-off at the final bar.
        bars = make_bars([
            (100, 101, 99, 100),
            (100.5, 101, 100, 100.6),
            (100.6, 101, 100, 100.5),
            (100.5, 101, 100, 100.4),
            (100.4, 101, 100, 100.3),
            (100.3, 101, 100, 100.2),
            (100.2, 101, 100, 100.1),  # last session bar
        ])
        bars = add_signal(bars, bars.index[0], stop=95.0, target=110.0)

        result = IntradayBacktester(IntradayConfig()).run({"TEST": bars})
        trade = result.trades.iloc[0]
        assert trade["exit_reason"] == "SQUAREOFF"
        assert trade["exit_time"] == bars.index[-1]

    def test_signal_on_last_bar_never_fills(self):
        bars = make_bars([
            (100, 101, 99, 100),
            (100, 101, 99, 100),
            (100, 101, 99, 100),
            (100, 101, 99, 100),
            (100, 101, 99, 100),
            (100, 101, 99, 100),
            (100, 101, 99, 100),
        ])
        bars = add_signal(bars, bars.index[-1], stop=99.0, target=102.0)

        result = IntradayBacktester(IntradayConfig()).run({"TEST": bars})
        assert len(result.trades) == 0

    def test_no_position_carried_across_sessions(self):
        # Two sessions; a position from session 1 must be flat by its end.
        day1 = make_bars([
            (100, 101, 99, 100),
            (100.5, 101, 100, 100.6),
            (100.6, 101, 100, 100.5),
            (100.5, 101, 100, 100.4),
            (100.4, 101, 100, 100.3),
            (100.3, 101, 100, 100.2),
            (100.2, 101, 100, 100.1),
        ], date="2026-01-05")
        day2 = make_bars([(100, 101, 99, 100)] * 7, date="2026-01-06")
        bars = pd.concat([day1, day2])
        bars = add_signal(bars, day1.index[0], stop=95.0, target=110.0)

        result = IntradayBacktester(IntradayConfig()).run({"TEST": bars})
        assert len(result.trades) == 1
        assert result.trades.iloc[0]["exit_time"] < day2.index[0]


class TestCostsAndSizing:
    def test_cost_model_matches_hand_calculation(self):
        cfg = IntradayConfig()
        bt = IntradayBacktester(cfg)

        buy_value = 200 * 100.0
        expected_buy = (
            min(cfg.costs.brokerage_pct * buy_value, cfg.costs.brokerage_cap_per_order)
            + cfg.costs.exchange_txn_pct * buy_value
            + cfg.costs.sebi_pct * buy_value
            + cfg.costs.gst_pct * (
                min(cfg.costs.brokerage_pct * buy_value, cfg.costs.brokerage_cap_per_order)
                + cfg.costs.exchange_txn_pct * buy_value
                + cfg.costs.sebi_pct * buy_value
            )
            + cfg.costs.stamp_duty_buy_pct * buy_value
        )
        assert bt._buy_costs(buy_value) == pytest.approx(expected_buy)

        sell_value = 200 * 102.0
        expected_sell = (
            min(cfg.costs.brokerage_pct * sell_value, cfg.costs.brokerage_cap_per_order)
            + cfg.costs.exchange_txn_pct * sell_value
            + cfg.costs.sebi_pct * sell_value
            + cfg.costs.gst_pct * (
                min(cfg.costs.brokerage_pct * sell_value, cfg.costs.brokerage_cap_per_order)
                + cfg.costs.exchange_txn_pct * sell_value
                + cfg.costs.sebi_pct * sell_value
            )
            + cfg.costs.stt_sell_pct * sell_value
        )
        assert bt._sell_costs(sell_value) == pytest.approx(expected_sell)

    def test_sizing_is_min_of_limits(self):
        # qty = min(risk budget, position cap, volume cap, cash).
        # fill ≈ 100.05; risk 2.05 → risk-based 243; position cap
        # 0.2*100k/100.05 = 199 → the position cap decides.
        bars = make_bars([
            (100, 101, 99, 100),
            (100, 103, 99.5, 102),
            (102, 103, 101, 102),
            (102, 103, 101, 101),
            (101, 102, 100, 101),
            (101, 102, 100, 100),
            (100, 101, 99, 100),
        ], volume=1_000_000)
        bars = add_signal(bars, bars.index[0], stop=98.0, target=105.0)

        cfg = IntradayConfig()
        result = IntradayBacktester(cfg).run({"TEST": bars})
        trade = result.trades.iloc[0]
        assert trade["qty"] == 199  # position cap (20 % of equity)


class TestRiskGate:
    def test_max_entries_per_day_enforced(self):
        day = "2026-01-05"
        a = add_signal(
            make_bars([(100, 101, 99, 100)] * 7, date=day),
            pd.Timestamp(f"{day} 09:15", tz="Asia/Kolkata"), 99, 102,
        )
        b = add_signal(
            make_bars([(200, 201, 199, 200)] * 7, date=day),
            pd.Timestamp(f"{day} 09:15", tz="Asia/Kolkata"), 199, 202,
        )
        cfg = IntradayConfig(max_entries_per_day=1)
        result = IntradayBacktester(cfg).run({"AAA": a, "BBB": b})
        assert len(result.trades) == 1  # second entry blocked

    def test_one_entry_per_symbol_per_day(self):
        # Same symbol signals twice in one session → only the first fills.
        day = "2026-01-05"
        bars = make_bars([
            (100, 101, 99, 100),
            (100, 101, 99.5, 100),
            (100, 101, 99.5, 100),
            (100, 101, 99.5, 100),
            (100, 101, 99.5, 100),
            (100, 101, 99.5, 100),
            (100, 101, 99.5, 100),
        ], date=day)
        bars = add_signal(bars, bars.index[0], 99, 102)
        bars.loc[bars.index[2], "entry"] = True
        bars.loc[bars.index[2], "stop"] = 99.0
        bars.loc[bars.index[2], "target"] = 102.0

        result = IntradayBacktester(IntradayConfig()).run({"TEST": bars})
        assert len(result.trades) == 1

    def test_daily_loss_limit_blocks_second_entry(self):
        # Daily loss limit 0.5 % of equity = 500 INR. The first symbol's
        # wide stop (risk ~10/share) lets the planned max loss consume most
        # of the budget (49 shares × 10.05 ≈ 492), so the second symbol has
        # < 8 INR of headroom → risk-per-share exceeds it → qty floors to 0.
        day = "2026-01-05"
        a = add_signal(
            make_bars([(100, 101, 99, 100)] * 7, date=day),
            pd.Timestamp(f"{day} 09:15", tz="Asia/Kolkata"), 90, 111,
        )
        b = add_signal(
            make_bars([(200, 201, 199, 200)] * 7, date=day),
            pd.Timestamp(f"{day} 09:15", tz="Asia/Kolkata"), 190, 221,
        )
        cfg = IntradayConfig(daily_loss_limit_pct=0.005)
        result = IntradayBacktester(cfg).run({"AAA": a, "BBB": b})
        assert len(result.trades) == 1
        assert result.trades.iloc[0]["symbol"] == "AAA"
        # Planned max loss of the taken trade ~ the whole daily budget.
        trade = result.trades.iloc[0]
        assert trade["qty"] * (trade["entry_price"] - trade["stop"]) < 500


class TestOrbSignals:
    """ORB rules from day-trader-skill trader-entry, via generate_signals."""

    @staticmethod
    def _regime(dates, bull=True):
        idx = pd.DatetimeIndex(
            [pd.Timestamp(d, tz="Asia/Kolkata") for d in dates], name="date"
        )
        return pd.DataFrame(
            {"regime_bull": [bull] * len(idx), "close": [100.0] * len(idx)},
            index=idx,
        )

    def test_orb_triggers_on_close_not_wick(self):
        # OR = first 30 min = the single 09:15 bar (high 101).
        # Bar 2 WICKS above 101 but closes under → no trigger.
        # Bar 3 closes above → trigger there.
        bars = make_bars([
            (100, 101, 99, 100),       # OR bar
            (100, 101.5, 99.5, 100.2),  # wick above 101, close under → no
            (100.5, 102, 100, 101.8),  # closes above 101 → trigger
            (102, 103, 101, 102),
            (102, 103, 101, 101),
            (101, 102, 100, 101),
            (101, 102, 100, 100),
        ])
        bars["volume"] = [1_000_000] * 7
        cfg = IntradayConfig(orb_min_volume_x=0.0)
        out = generate_signals(bars, self._regime(["2026-01-05"]), cfg)

        orb_bars = out[(out["setup"] == "ORB")]
        assert len(orb_bars) == 1
        assert orb_bars.index[0] == bars.index[2]
        # Stop at OR midpoint: (101 + 99) / 2 = 100
        assert orb_bars.iloc[0]["stop"] == pytest.approx(100.0)
        # Target 2R: 101.8 + 2 * (101.8 - 100) = 105.4
        assert orb_bars.iloc[0]["target"] == pytest.approx(105.4)

    def test_orb_volume_filter_blocks_thin_trigger(self):
        bars = make_bars([
            (100, 101, 99, 100),
            (100.5, 102, 100, 101.8),  # closes above 101 …
            (102, 103, 101, 102),
            (102, 103, 101, 101),
            (101, 102, 100, 101),
            (101, 102, 100, 100),
            (101, 102, 100, 100),
        ])
        bars["volume"] = [1_000_000, 100, 100, 100, 100, 100, 100]
        cfg = IntradayConfig(orb_min_volume_x=1.5)
        out = generate_signals(bars, self._regime(["2026-01-05"]), cfg)
        assert (out["setup"] == "ORB").sum() == 0  # trigger bar too thin

    def test_orb_one_trigger_per_day(self):
        bars = make_bars([
            (100, 101, 99, 100),
            (100.5, 102, 100, 101.8),
            (101.8, 103, 101, 102.5),
            (102.5, 104, 102, 103.5),
            (103.5, 105, 103, 104.5),
            (104.5, 106, 104, 105.5),
            (105.5, 107, 105, 106.5),
        ])
        bars["volume"] = [1_000_000] * 7
        cfg = IntradayConfig(orb_min_volume_x=0.0)
        out = generate_signals(bars, self._regime(["2026-01-05"]), cfg)
        assert (out["setup"] == "ORB").sum() == 1

    def test_orb_blocked_when_regime_bearish(self):
        bars = make_bars([
            (100, 101, 99, 100),
            (100.5, 102, 100, 101.8),
            (102, 103, 101, 102),
            (102, 103, 101, 101),
            (101, 102, 100, 101),
            (101, 102, 100, 100),
            (101, 102, 100, 100),
        ])
        bars["volume"] = [1_000_000] * 7
        cfg = IntradayConfig(orb_min_volume_x=0.0)
        out = generate_signals(bars, self._regime(["2026-01-05"], bull=False), cfg)
        assert (out["entry"] == True).sum() == 0  # noqa: E712

    def test_stop_floor_refuses_too_tight_stop(self):
        # ORB risk per share = close - or_mid, which is at least half the OR
        # range. With a ~10,000 INR stock and a tight 8-point OR (or_mid
        # 10,000, or_high 10,004), a trigger closing at 10,004.01 risks only
        # 4.01/share — below the 0.05 % floor (5.00) → trader-exit refusal.
        bars = make_bars([
            (10000, 10004, 9996, 10000),       # OR bar: tight range
            (10003, 10004.5, 10002, 10004.01),  # closes above or_high, barely
            (10004, 10005, 10003, 10004),
            (10004, 10005, 10003, 10004),
            (10004, 10005, 10003, 10004),
            (10004, 10005, 10003, 10004),
            (10004, 10005, 10003, 10004),
        ])
        bars["volume"] = [1_000_000] * 7
        cfg = IntradayConfig(orb_min_volume_x=0.0)
        out = generate_signals(bars, self._regime(["2026-01-05"]), cfg)
        assert (out["entry"] == True).sum() == 0  # noqa: E712

        # Sanity: with a wider OR (16 points → risk 8+ > 5 floor) the same
        # close DOES trigger — proving the floor is what blocked above.
        bars2 = make_bars([
            (10000, 10008, 9992, 10000),      # OR bar: wider range
            (10003, 10010, 10002, 10008.01),
            (10008, 10009, 10007, 10008),
            (10008, 10009, 10007, 10008),
            (10008, 10009, 10007, 10008),
            (10008, 10009, 10007, 10008),
            (10008, 10009, 10007, 10008),
        ])
        bars2["volume"] = [1_000_000] * 7
        out2 = generate_signals(bars2, self._regime(["2026-01-05"]), cfg)
        assert (out2["entry"] == True).sum() == 1  # noqa: E712


class TestDailyRegime:
    def test_regime_uses_previous_completed_bar_only(self):
        dates = pd.date_range("2026-01-01", periods=10, freq="D")
        daily = pd.DataFrame(
            {
                "close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
                "open": 100.0, "high": 101.0, "low": 99.0, "volume": 1,
            },
            index=dates.tz_localize("Asia/Kolkata"),
        )
        out = daily_regime(daily, sma_fast=3, sma_slow=5)
        # First bars have NaN SMAs → regime False.
        assert not out["regime_bull"].iloc[:5].any()
        assert bool(out["regime_bull"].iloc[-1])
