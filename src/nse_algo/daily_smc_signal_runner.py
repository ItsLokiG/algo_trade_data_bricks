# Databricks notebook source
# DBTITLE 1,Overview
# MAGIC %md
# MAGIC # Daily SMC Signal Runner — Paper Trading
# MAGIC
# MAGIC Runs the validated **C10 daily SMC + price-action strategy** (large caps, full NSE delivery costs, no look-ahead) and maintains a paper-trading portfolio.
# MAGIC
# MAGIC **How to use:** run all cells after market close on any trading day (~16:15 IST, once yfinance has the day's bar). The runner will:
# MAGIC
# MAGIC 1. Refresh the data cache (universe + NIFTY, ~2 min if stale)
# MAGIC 2. Catch up on any missed days automatically (state persisted in `data_cache/paper/daily_smc_paper_state.json`)
# MAGIC 3. Fill yesterday's signals at today's open, check stops/targets/time exits intrabar (stop-first), mark equity at close
# MAGIC 4. Emit **BUY orders for the next session** (fill at next open; skip if the open gaps at/below the stop)
# MAGIC
# MAGIC Paper account: ₹1,000,000, 1% equity risk per trade, max 20 positions, 6 new entries/day, NIFTY>SMA40 regime gate.

# COMMAND ----------

# DBTITLE 1,Setup
# Setup: install yfinance if missing, load the strategy module from a /tmp copy
import importlib.util, shutil, subprocess, sys

REPO = "/Workspace/Repos/lokigovindasamy@gmail.com/algo_trade_data_bricks_with_vs"
MODULE = f"{REPO}/src/nse_algo/strategy/daily_smc.py"
STATE_PATH = f"{REPO}/data_cache/paper/daily_smc_paper_state.json"

if importlib.util.find_spec("yfinance") is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "yfinance"], check=True)

shutil.copy(MODULE, "/tmp/daily_smc.py")
sys.modules.pop("daily_smc", None)
sys.path.insert(0, "/tmp")
import daily_smc as dsm
print("module loaded from", dsm.__file__)

# Validated C10 config. Conservative option (trades return for drawdown):
#   cfg = dsm.DailySMCConfig(breaker_dd_pct=0.15, breaker_resume_pct=0.075)
cfg = dsm.DailySMCConfig()
print(f"config: setups={cfg.setups}, large_only={cfg.large_only}, "
      f"breaker={'OFF' if cfg.breaker_dd_pct == 0 else f'{cfg.breaker_dd_pct:.0%}/{cfg.breaker_resume_pct:.0%}'}")

# COMMAND ----------

# DBTITLE 1,Data refresh
# Data refresh: universe (182 tickers) + NIFTY benchmark, then rebuild signals
import time
from pathlib import Path

import yfinance as yf

refreshed = dsm.update_universe_cache(max_age_hours=20)  # ~1.5 min when stale

nifty_path = Path(dsm.NIFTY_CACHE)
nifty_path.parent.mkdir(parents=True, exist_ok=True)
if not nifty_path.exists() or (time.time() - nifty_path.stat().st_mtime) > 20 * 3600:
    ndf = yf.Ticker("^NSEI").history(start="2018-01-01", interval="1d")
    ndf = ndf.rename(columns={"Open": "open", "High": "high", "Low": "low",
                              "Close": "close", "Volume": "volume"})
    ndf[["open", "high", "low", "close", "volume"]].to_csv(nifty_path)
    print("NIFTY cache refreshed")

uni = dsm.load_universe()
nifty = dsm.load_nifty()  # tolerant of capital-case columns too
assert len(nifty) > 2000 and len(nifty.columns) == 5, "NIFTY cache bad"
sig = {s: dsm.build_daily_signals(df) for s, df in uni.items()}
latest = max(d for d in nifty.index)
print(f"universe: {len(uni)} symbols ({len(refreshed)} re-downloaded) | NIFTY bars: {len(nifty)} | latest bar: {latest.date()}")

# COMMAND ----------

# DBTITLE 1,Daily tick logic
# The daily tick — an exact mirror of the validated engine's per-day logic.
# Fills at open (skip if open <= stop), stop-first intrabar exits, time exit
# at close on the max-hold day, new signals decided at close, equity marked at close.


def apply_day(t, st, sig, nifty_bull, cfg):
    eq_hist = st["equity"]
    last_eq = eq_hist[-1][1] if eq_hist else st["capital"]

    for sym in sorted(sig):
        s = sig[sym]
        if t not in s.index:
            continue
        o, h, l, c = (s.loc[t, x] for x in ("open", "high", "low", "close"))

        if sym in st["pending"] and sym not in st["positions"]:
            stp, tgt, setup = st["pending"].pop(sym)
            if o <= stp:
                continue  # gapped below stop — entry invalid
            dist = o - stp
            qty = min(int(last_eq * cfg.risk_pct / dist),
                      int(last_eq * cfg.pos_cap_pct / o), int(st["cash"] / o))
            if qty <= 0:
                continue
            st["cash"] -= qty * o + dsm.trade_costs(qty * o, True)
            st["positions"][sym] = dict(qty=qty, entry=float(o), stop=float(stp),
                                        target=float(tgt), setup=setup, bars=0,
                                        entry_date=str(t.date()))
            continue  # no exit checks on the entry bar

        if sym in st["positions"]:
            p = st["positions"][sym]
            p["bars"] += 1
            xp, rs = None, None
            if o <= p["stop"]:
                xp, rs = o, "STOP_GAP"
            elif o >= p["target"]:
                xp, rs = o, "TARGET_GAP"
            elif l <= p["stop"]:
                xp, rs = p["stop"], "STOP"
            elif h >= p["target"]:
                xp, rs = p["target"], "TARGET"
            elif p["bars"] >= cfg.max_hold[p["setup"]]:
                xp, rs = c, "TIME"
            if xp is not None:
                val = p["qty"] * xp
                st["cash"] += val - dsm.trade_costs(val, False)
                st["trades"].append(dict(
                    symbol=sym, setup=p["setup"], entry_date=p["entry_date"],
                    exit_date=str(t.date()), qty=p["qty"], entry=p["entry"],
                    exit=float(xp), stop=p["stop"], target=p["target"],
                    pnl_gross=(xp - p["entry"]) * p["qty"], reason=rs))
                del st["positions"][sym]

    # circuit breaker (OFF in the validated config)
    if cfg.breaker_dd_pct > 0:
        st["peak_eq"] = max(st.get("peak_eq", st["capital"]), last_eq)
        if not st.get("halted") and last_eq < st["peak_eq"] * (1 - cfg.breaker_dd_pct):
            st["halted"] = True
        elif st.get("halted") and last_eq >= st["peak_eq"] * (1 - cfg.breaker_resume_pct):
            st["halted"] = False

    # new entries decided at the close of t (fill next session)
    if (t in nifty_bull.index and bool(nifty_bull.loc[t])
            and not st.get("halted") and len(st["positions"]) < cfg.max_positions):
        count = 0
        for sym in sorted(sig):
            if count >= cfg.max_new_per_day or len(st["positions"]) >= cfg.max_positions:
                break
            if sym in st["positions"] or sym in st["pending"]:
                continue
            if cfg.large_only and sym not in dsm.LARGE_CAPS:
                continue
            s = sig[sym]
            if t not in s.index:
                continue
            setup = s.loc[t, "setup"]
            if setup not in cfg.setups:
                continue
            sref, atr = s.loc[t, "stop_ref"], s.loc[t, "atr"]
            if not (sref == sref and atr == atr):
                continue
            ccl = s.loc[t, "close"]
            stp = sref - cfg.stop_buffer_atr[setup] * atr
            r_ref = ccl - stp
            if r_ref <= 0 or r_ref < cfg.stop_floor_pct * ccl:
                continue
            tgt = ccl + cfg.target_r[setup] * r_ref
            st["pending"][sym] = [float(stp), float(tgt), setup]
            count += 1

    eq = st["cash"]
    for sym, p in st["positions"].items():
        s = sig[sym]
        eq += p["qty"] * (s.loc[t, "close"] if t in s.index else p["entry"])
    st["equity"].append([str(t.date()), round(float(eq), 2)])
    st["last_date"] = str(t.date())


print("apply_day defined")

# COMMAND ----------

# DBTITLE 1,Run + persist state
# Load (or initialize) paper state, replay missed days, persist.
import json
from pathlib import Path

st_path = Path(STATE_PATH)
if st_path.exists():
    st = json.loads(st_path.read_text())
    print(f"loaded paper state (last processed: {st['last_date']})")
else:
    st = dict(capital=cfg.capital, cash=cfg.capital, positions={}, pending={},
              trades=[], equity=[], halted=False, peak_eq=cfg.capital, last_date=None)
    print("initialized fresh paper state at ₹1,000,000")

nifty_bull = nifty["close"] > nifty["close"].rolling(cfg.nifty_regime_sma).mean()
all_dates = sorted({d for s in sig.values() for d in s.index})
if st["last_date"] is None:
    # fresh start: only process the latest available bar
    uniq = sorted(set(str(d.date()) for d in all_dates))
    st["last_date"] = uniq[-2] if len(uniq) > 1 else uniq[-1]
todo = [d for d in all_dates
        if str(d.date()) > st["last_date"] and d <= latest]
print(f"processing {len(todo)} day(s) up to {latest.date()}")
for t in todo:
    apply_day(t, st, sig, nifty_bull, cfg)

st_path.parent.mkdir(parents=True, exist_ok=True)
st_path.write_text(json.dumps(st, indent=2))
print("state saved:", st_path)

# COMMAND ----------

# DBTITLE 1,Next-session orders
# Report: portfolio + NEXT SESSION orders
eq = st["equity"][-1][1]
print(f"=== PAPER PORTFOLIO — processed through {st['last_date']} ===")
print(f"NIFTY regime: {'BULL (above SMA40) — entries allowed' if bool(nifty_bull.iloc[-1]) else 'BEAR (below SMA40) — entries blocked'} (NIFTY close {nifty['close'].iloc[-1]:,.0f} vs SMA40 {nifty['close'].rolling(40).mean().iloc[-1]:,.0f})")
print(f"equity ₹{eq:,.0f} ({(eq / st['capital'] - 1) * 100:+.2f}%) | cash ₹{st['cash']:,.0f} | positions {len(st['positions'])} | closed paper trades {len(st['trades'])}")

print("\nOpen positions:")
if st["positions"]:
    for sym, p in sorted(st["positions"].items()):
        print(f"  {sym:12s} {p['setup']:3s} qty {p['qty']:>5d}  entry ₹{p['entry']:>9.2f}  stop ₹{p['stop']:>9.2f}  target ₹{p['target']:>9.2f}  bars {p['bars']}")
else:
    print("  (none)")

print("\nNEXT SESSION BUY orders (fill at open; skip if open <= stop):")
if st["pending"]:
    for sym, (stp, tgt, setup) in sorted(st["pending"].items()):
        print(f"  BUY {sym:12s} {setup:3s}  stop ₹{stp:.2f}  target ₹{tgt:.2f}")
else:
    print("  (none — no signals passed the filters at the last close)")

if st.get("halted"):
    print("\n!! circuit breaker HALTED — no new entries until recovery")

# COMMAND ----------

# DBTITLE 1,Honest notes
# MAGIC %md
# MAGIC ## Honest notes (do not remove)
# MAGIC
# MAGIC * Validated profile (C10, this config): walk-forward OOS 2021-26 +79% vs NIFTY +62% — but **2025 (-7.9%) and 2026 so far (-2.8%) were negative**. The edge is carried by 2021/2023/2024; treat paper results accordingly.
# MAGIC * Survivorship bias: the universe is today's constituents. OOS selection is partially contaminated (large-cap restriction chosen after observing 2024 midcap failures).
# MAGIC * Circuit breaker: tested AFTER seeing 2025-26 weakness — no variant improved the validated profile; default OFF. `dd15/resume7.5` is available as a conservative option.
# MAGIC * Costs assume full NSE delivery charges + 0.05% slippage per side. Data from yfinance (revised for corporate actions — pre-2021 backtest numbers can shift slightly on re-download).
# MAGIC * This is research/paper-trading infrastructure, not investment advice.