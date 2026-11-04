# Intraday SMC Day-Trading Strategy — Design Document

**Scope change (Oct 2026):** The original project scope excluded intraday
strategies. This module is an explicit, user-requested scope change. It adds a
**separate** intraday research layer — it does not alter the daily/weekly swing
system or its existing tests.

## 1. Hypothesis

Institutional order flow leaves detectable footprints on intraday price
action: liquidity sweeps (stop-hunts beyond obvious swing levels) followed by
market-structure breaks, then reactions at order blocks (the last opposing
candle before the break) and fair value gaps (3-candle imbalances). Trading
**in the direction of the daily trend**, entering at these institutional
zones in the session's discount range, may capture a positive-expectancy
intraday move to the next liquidity pool.

**Why it might work:** post-sweep reversion + trend continuation reflects
forced liquidation (stopped-out traders) followed by informed re-entry.

**Why it might NOT work:** SMC/ICT concepts are popular-retail frameworks with
weak academic validation. Intraday edges are small relative to costs.
This is a hypothesis to test, not a belief.

## 2. Universe & filters

* NSE cash equities only. Long-only. No leverage. No short selling.
* Liquidity filter: average daily turnover above a configurable floor
  (default ~Rs 100 crore) to limit impact costs.
* Fundamental filter (yfinance): large-cap market cap floor, positive
  trailing earnings (no distressed names). Fundamentals are a *universe
  filter* here — they do not time intraday entries.
* No entries in the first 30 minutes of a session (opening-range completion,
  also avoids earnings-gap chaos).

## 3. Timeframes

* **Daily bars** — HTF context: close > 20-SMA and > 50-SMA (uptrend regime).
* **1h bars (primary backtest)** — SMC structure: swings, BOS/CHoCH, OBs, FVGs.
* **15m bars (recent-60d validation)** — finer entry/stop granularity.

yfinance intraday limits (verified Oct 2026): 15m ≈ 60 days, 1h ≈ 730 days.

## 4. Smart Money Concepts (mechanical, rule-based)

All computed **session-scoped** (09:15–15:30 IST) to avoid cross-day leakage:

* **Swing points** — fractal highs/lows with `k` bars on each side (default k=2).
* **BOS (Break of Structure)** — close above the last swing high while the
  prevailing structure is bullish.
* **CHoCH (Change of Character)** — structure break *against* the prevailing
  direction (e.g., close above last swing high while structure was bearish).
* **Liquidity sweep** — bar wick exceeds a prior swing high/low but the bar
  **closes back inside** the prior range (failed stop-hunt).
* **Order block (bullish)** — last down-close candle before a bullish
  BOS/CHoCH; zone = that candle's low→high. Invalidated once price closes
  below the zone.
* **Fair value gap (bullish)** — 3-candle imbalance: low[t] > high[t-2];
  zone = [high[t-2], low[t]]. Entry on first retest of the gap.
* **Premium/Discount** — equilibrium = 50% of the session range to date.
  Long entries only in **discount** (below equilibrium).

## 5. Technical confluence

* **VWAP** (session, volume-weighted) — close must be >= VWAP at entry.
* **ATR(14, intraday bars)** — stop distances and volatility sanity checks.
* **Opening range (first 30 min)** — no entries before it completes; OR high
  used as a liquidity reference for targets.
* **Time filters** — no entries before 09:45, none after 14:15 (leaves ~1h to
  manage the trade before square-off).

## 6. Entry rule (all must be true at bar close)

1. Daily regime bullish (close > 20-SMA > 50-SMA, using only data up to the
   previous completed daily bar).
2. Intraday: a liquidity sweep of a prior swing low occurred, then a bullish
   CHoCH/BOS confirmed within the last N bars (default 6).
3. Price has returned into a **valid bullish order block or bullish FVG**.
4. Zone is in **discount** (below session equilibrium).
5. Close >= session VWAP.
6. Time filter passes (09:45–14:15).

## 7. Exits (defined before entry)

* **Stop-loss** — min(OB/FVG zone low, sweep low) − 0.1×ATR buffer.
* **Target** — nearest liquidity pool above (last intraday swing high) or 2R,
  whichever is closer.
* **Time exit** — forced square-off at the 15:15 bar close. **No overnight
  positions. Ever.**
* **Daily circuit breaker** — stop all entries for the day at −2% day P&L.

## 8. Risk & sizing

* Planned risk 0.5% of equity per trade (cap 1%).
* Max 2 concurrent intraday positions; max 4 entries/day.
* Fixed-fractional sizing: qty = (equity × risk%) / (entry − stop).
* Position size capped by liquidity: <= 1% of that bar's volume.

## 9. Cost model (intraday, research defaults — verify before live)

* Brokerage: min(0.03% of value, Rs 20) per order (2 orders per round trip).
* STT intraday: 0.025% on **sell** side only.
* Exchange txn 0.00297%, SEBI Rs 10/crore, GST 18% on brokerage + charges,
  stamp duty 0.003% buy side, slippage 0.05% per side.

## 10. Execution realism (anti-look-ahead)

* Signal computed at close of bar `t` → fill at **open of bar t+1**.
* If stop and target are both touchable within one bar → **assume stop hit
  first** (conservative).
* Regime filters use only completed bars.

## 11. Validation plan

1. Unit tests on synthetic bars with known SMC geometry.
2. 1h backtest, ~2 years, 10–15 liquid large caps.
3. 15m backtest on the most recent 60 days (finer granularity check).
4. In-sample / out-of-sample split (first 70% / last 30% by time).
5. Parameter sensitivity: k (fractal width), max sweep age, entry window.
6. Honest reporting: trade count, win rate, PF, expectancy, max DD, total
   costs, and cost share of gross P&L. Label results "historically positive
   under these assumptions" at best — never "profitable, production-ready".

## 12. Known weaknesses

* 60-day 15m window is far too small to conclude anything.
* yfinance intraday data is unofficial; bar volumes can be unreliable.
* SMC is discretionary-retail theory formalized mechanically here; the
  mechanical rules may not capture what practitioners mean.
* Intraday strategies fail silently on latency — this system is research /
  paper only. No broker integration is in scope.

## 13. Progression (never skipped)

Research backtest → paper signals (state-file based, like the swing system) →
manual small-size live verification (out of scope for this repo) only after
months of positive paper results vs. costs.
