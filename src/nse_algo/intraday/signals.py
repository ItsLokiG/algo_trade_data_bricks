"""Confluence signal generation for the intraday NSE strategy.

Two independent long-only entry setups, both evaluated ONLY on completed
bars (decision at bar-t close, execution at bar-(t+1) open in the engine):

**Setup 1 — SMC confluence** (design doc section 6):
1. Daily regime bullish (prev daily close > 20-SMA > 50-SMA).
2. A liquidity sweep of a prior swing low, then a bullish CHoCH/BOS
   within ``choch_max_age_bars``, sweep within ``sweep_max_age_bars``.
3. Price retested a valid bullish order block or FVG this bar.
4. Zone in discount (close below session equilibrium).
5. Close >= session VWAP.
6. Time window 09:45-14:15 IST.

**Setup 2 — Opening-range breakout (ORB)** — rules from
github.com/jakeschincariol/day-trader-skill (trader-entry):
1. Daily regime bullish.
2. First COMPLETED bar after the opening range that CLOSES above the
   range high (a wick is not a trigger; a close exactly at the level is
   not above it).
3. Trigger-bar volume >= ``orb_min_volume_x`` x the OR bars' average.
4. One trigger per symbol per day.  Stop = OR midpoint (skill default);
   target = 2R.

Shared refusal gates (trader-exit / trader-risk):
* risk per share < ``min_risk_pct_of_price`` of price -> too tight to
  survive the spread;
* reward:risk < ``min_reward_risk`` -> skip.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from nse_algo.intraday.config import IntradayConfig
from nse_algo.intraday.smc import SessionSMC
from nse_algo.intraday.technicals import (
    atr,
    opening_range,
    session_equilibrium,
    session_vwap,
)

logger = logging.getLogger(__name__)


def _parse_hhmm(value: str) -> pd.Timedelta:
    hours, minutes = value.split(":")
    return pd.Timedelta(hours=int(hours), minutes=int(minutes))


def generate_signals(
    intraday_df: pd.DataFrame,
    daily_regime_df: pd.DataFrame,
    config: IntradayConfig,
) -> pd.DataFrame:
    """Generate per-bar entry/stop/target signals for one symbol.

    Parameters
    ----------
    intraday_df : tz-aware intraday OHLCV bars (IST, bar-open index).
    daily_regime_df : output of ``fundamentals.daily_regime`` — must have a
        DatetimeIndex (dates) and a ``regime_bull`` boolean column.

    Returns
    -------
    DataFrame with the original OHLCV columns plus ``vwap``, ``atr``,
    ``or_high``/``or_low``/``or_mid``, and signal columns ``entry`` (bool),
    ``stop``, ``target``, ``setup`` ('SMC' or 'ORB'), ``signal_reason``.
    """
    df = intraday_df.copy()

    # Continuous-series indicators (ATR spans the overnight gap on purpose).
    df["atr"] = atr(df, period=config.atr_period)

    # Session-scoped features.
    df = opening_range(
        df, session_start=config.session_start, or_minutes=config.or_minutes
    )
    df["vwap"] = np.nan
    df["equilibrium"] = np.nan

    df["entry"] = False
    df["stop"] = np.nan
    df["target"] = np.nan
    df["setup"] = ""
    df["signal_reason"] = ""

    # Map session date -> regime/breakout flags (evaluated on the previous
    # completed daily bar, never the current day's close).
    regime_by_date: dict[pd.Timestamp, bool] = {}
    if "regime_bull" in getattr(daily_regime_df, "columns", []):
        for date, row in daily_regime_df.iterrows():
            regime_by_date[pd.Timestamp(date).normalize()] = bool(row["regime_bull"])

    entry_start_t = _parse_hhmm(config.entry_start)
    entry_end_t = _parse_hhmm(config.entry_end)
    # Opening range covers bars whose OPEN time is within
    # [session_start, session_start + or_minutes).
    or_end_time = (
        pd.Timestamp(config.session_start) + pd.Timedelta(minutes=config.or_minutes)
    ).time()
    or_end_t = _parse_hhmm(config.session_start) + pd.Timedelta(
        minutes=config.or_minutes
    )

    for date, session in df.groupby(df.index.normalize()):
        positions = np.arange(len(session))
        pos_by_time = {ts: int(p) for p, ts in zip(positions, session.index)}

        # Session-scoped indicator series.
        vwap_s = session_vwap(session)
        eq_s = session_equilibrium(session)

        smc = SessionSMC(
            session, swing_k=config.swing_k,
            ob_max_lookback_bars=config.ob_max_lookback_bars,
        )

        regime_bull = regime_by_date.get(date, False)

        # ORB bars: those whose open time is inside the opening range.
        or_mask = np.array([t < or_end_time for t in session.index.time])
        or_indices = positions[or_mask]
        or_avg_volume = (
            float(session.iloc[or_indices]["volume"].mean())
            if len(or_indices)
            else 0.0
        )

        orb_triggered = False

        for ts, row in session.iterrows():
            i = pos_by_time[ts]
            time_of_day = ts - ts.normalize()

            if not (entry_start_t <= time_of_day <= entry_end_t):
                continue

            close = float(row["close"])
            low = float(row["low"])
            atr_val = float(row["atr"])

            signal_setup: str | None = None
            stop: float = np.nan
            target: float = np.nan

            # ── Setup 1: SMC confluence ──
            if regime_bull and atr_val > 0:
                event = smc.last_bullish_event(i, config.choch_max_age_bars)
                if event is not None:
                    sweep = smc.sweep_before(event.bar, config.sweep_max_age_bars)
                    zones = smc.zone_retests_at(i)
                    if sweep is not None and zones:
                        # Prefer the zone with the highest bottom (nearest).
                        zone = max(zones, key=lambda z: z.bottom)
                        vwap_val = float(vwap_s.iloc[i])
                        eq_val = float(eq_s.iloc[i])
                        if (
                            close >= vwap_val
                            and close < eq_val  # discount
                        ):
                            cand_stop = (
                                min(zone.bottom, sweep.level)
                                - config.stop_atr_buffer * atr_val
                            )
                            if cand_stop < close:
                                liquidity = smc.nearest_liquidity_above(i, close)
                                risk = close - cand_stop
                                r_target = close + config.target_r_multiple * risk
                                cand_target = (
                                    liquidity if liquidity is not None else r_target
                                )
                                if liquidity is not None and liquidity > r_target:
                                    cand_target = r_target
                                signal_setup = "SMC"
                                stop = cand_stop
                                target = cand_target

            # ── Setup 2: ORB (trader-entry rules) ──
            if (
                signal_setup is None
                and config.orb_enabled
                and not orb_triggered
                and regime_bull
                and time_of_day >= or_end_t
                and pd.notna(row["or_high"])
                and close > float(row["or_high"])  # strictly above
                and or_avg_volume > 0
                and float(row["volume"]) >= config.orb_min_volume_x * or_avg_volume
            ):
                orb_stop = (
                    float(row["or_mid"])
                    if config.orb_stop_rule == "or_mid"
                    else float(row["or_low"])
                )
                if orb_stop < close:
                    signal_setup = "ORB"
                    stop = orb_stop
                    target = close + 2.0 * (close - orb_stop)  # 2R
                    orb_triggered = True  # one trigger per setup per day

            # ── Shared refusal gates (trader-exit / trader-risk) ──
            if signal_setup is not None:
                risk = close - stop
                if risk < config.min_risk_pct_of_price * close:
                    signal_setup = None  # too tight to survive the spread
                elif risk <= 0 or (target - close) / risk < config.min_reward_risk:
                    signal_setup = None  # reward:risk below minimum

            if signal_setup is not None:
                df.loc[ts, "entry"] = True
                df.loc[ts, "stop"] = stop
                df.loc[ts, "target"] = target
                df.loc[ts, "setup"] = signal_setup
                df.loc[ts, "signal_reason"] = (
                    f"{signal_setup}: close={close:.2f} stop={stop:.2f} "
                    f"target={target:.2f}"
                )

        # Store session-scoped indicator values for diagnostics.
        df.loc[session.index, "vwap"] = vwap_s.values
        df.loc[session.index, "equilibrium"] = eq_s.values

    return df
