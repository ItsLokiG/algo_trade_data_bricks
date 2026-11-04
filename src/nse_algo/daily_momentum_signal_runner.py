# Databricks notebook source
# DBTITLE 1,Overview
# MAGIC %md
# MAGIC # Daily Momentum Rotation Runner — Paper Trading (2nd book)
# MAGIC
# MAGIC Runs the validated **monthly top-10 momentum rotation** (large caps, 12-1 lookback, ungated, full NSE delivery costs, no look-ahead) as a second, diversified paper book alongside the C10 SMC book.
# MAGIC
# MAGIC **How to use:** run all cells after market close on any trading day (~16:15 IST). The runner:
# MAGIC
# MAGIC 1. Refreshes the data cache (shared with the SMC runner)
# MAGIC 2. Catches up on missed days automatically (state in `data_cache/paper/daily_momentum_paper_state.json`)
# MAGIC 3. At each month-end close: ranks the large-cap universe by 12-1 momentum → rebalance targets; fills at the NEXT session's open
# MAGIC 4. Marks equity daily and prints a **combined view** with the C10 book
# MAGIC
# MAGIC Paper account: ₹1,000,000, 10% of equity per position, ~10 positions, monthly turnover.

# COMMAND ----------

# DBTITLE 1,Setup
# Setup: install yfinance if missing, load both strategy modules from /tmp copies
import importlib.util, shutil, subprocess, sys

REPO = "/Workspace/Repos/lokigovindasamy@gmail.com/algo_trade_data_bricks_with_vs"
STRAT = f"{REPO}/src/nse_algo/strategy"
STATE_PATH = f"{REPO}/data_cache/paper/daily_momentum_paper_state.json"
SMC_STATE = f"{REPO}/data_cache/paper/daily_smc_paper_state.json"

if importlib.util.find_spec("yfinance") is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "yfinance"], check=True)

shutil.copy(f"{STRAT}/daily_smc.py", "/tmp/daily_smc.py")
shutil.copy(f"{STRAT}/daily_momentum.py", "/tmp/daily_momentum.py")
for m in ("daily_smc", "daily_momentum"):
    sys.modules.pop(m, None)
sys.path.insert(0, "/tmp")
import daily_smc as dsm
import daily_momentum as dm
cfg = dm.MomentumConfig()  # top10, ungated, 12-1 — the validated default
print("modules loaded | cfg:", cfg)

# COMMAND ----------

# DBTITLE 1,Data refresh
# Data refresh: universe + NIFTY (shared cache with the SMC runner), then load raw frames
import time
from pathlib import Path

import yfinance as yf

refreshed = dsm.update_universe_cache(max_age_hours=20)
nifty_path = Path(dsm.NIFTY_CACHE)
nifty_path.parent.mkdir(parents=True, exist_ok=True)
if not nifty_path.exists() or (time.time() - nifty_path.stat().st_mtime) > 20 * 3600:
    ndf = yf.Ticker("^NSEI").history(start="2018-01-01", interval="1d")
    ndf = ndf.rename(columns={"Open": "open", "High": "high", "Low": "low",
                              "Close": "close", "Volume": "volume"})
    ndf[["open", "high", "low", "close", "volume"]].to_csv(nifty_path)
    print("NIFTY cache refreshed")

uni = dsm.load_universe()
nifty = dsm.load_nifty()
cal = nifty.index
print(f"universe: {len(uni)} symbols ({len(refreshed)} re-downloaded) | calendar latest: {cal[-1].date()}")

# COMMAND ----------

# DBTITLE 1,Daily tick logic
# The daily tick — exact mirror of run_momentum_backtest's per-day logic:
# rebalance fills at open of the day after month-end, daily equity marks.


def apply_day_mom(d, st, uni, cal, cfg):
    # 1) execute pending monthly rebalance at the open
    if st.get("pending") and (st["pending"].get("exec_date") is None
                            or str(d.date()) == st["pending"]["exec_date"]):
        # exec_date=None (inception) fires at the first session AFTER the state
        # was initialized — decided at a prior close, so no look-ahead.
        tgt = set(st["pending"]["syms"])
        for s in list(st["holdings"]):
            if s not in tgt and d in uni[s].index:
                px = uni[s].loc[d, "open"]
                st["cash"] += st["holdings"][s] * px - dsm.trade_costs(st["holdings"][s] * px, False)
                st["trades"].append(dict(symbol=s, side="SELL", date=str(d.date()),
                                        qty=st["holdings"][s], price=float(px)))
                del st["holdings"][s]
        eq_sz = st["cash"] + sum(q * st["last_close"].get(s, 0.0) for s, q in st["holdings"].items())
        budget = cfg.weight_pct * eq_sz
        for s in st["pending"]["syms"]:
            if s in st["holdings"] or d not in uni[s].index:
                continue
            px = uni[s].loc[d, "open"]
            qty = int(min(budget / px, st["cash"] / (px * 1.01)))
            if qty <= 0 or qty * px > st["cash"]:
                continue
            st["cash"] -= qty * px + dsm.trade_costs(qty * px, True)
            st["holdings"][s] = qty
            st["trades"].append(dict(symbol=s, side="BUY", date=str(d.date()),
                                    qty=qty, price=float(px)))
        st["pending"] = None

    # 2) mark equity at close
    for s, q in st["holdings"].items():
        if d in uni[s].index:
            st["last_close"][s] = float(uni[s].loc[d, "close"])
    eq = st["cash"] + sum(q * st["last_close"].get(s, 0.0) for s, q in st["holdings"].items())
    st["equity"].append([str(d.date()), round(float(eq), 2)])

    # 3) month-end decision (rank at close of the last trading day of the month)
    if d in cal:
        i = cal.get_loc(d)
        if i + 1 < len(cal) and d.month != cal[i + 1].month:
            ranked = dm.momentum_rank(uni, d, cfg)
            if cfg.use_gate:
                bull = nifty["close"].loc[d] > nifty["close"].rolling(40).mean().loc[d]
                ranked = ranked if bull else []
            st["pending"] = {"exec_date": str(cal[i + 1].date()), "syms": ranked}
    st["last_date"] = str(d.date())


print("apply_day_mom defined")

# COMMAND ----------

# DBTITLE 1,Run + persist state
# Load (or initialize) paper state, replay missed days, persist.
import json
from pathlib import Path

st_path = Path(STATE_PATH)
if st_path.exists():
    st = json.loads(st_path.read_text())
    print(f"loaded paper state (last processed: {st['last_date']})")
    FRESH = False
else:
    st = dict(capital=cfg.capital, cash=cfg.capital, holdings={}, last_close={},
              pending=None, trades=[], equity=[], last_date=None)
    print("initialized fresh paper state at ₹1,000,000")
    FRESH = True

all_dates = sorted({d for df in uni.values() for d in df.index})
if st["last_date"] is None:
    uniq = sorted(set(str(d.date()) for d in all_dates))
    st["last_date"] = uniq[-2] if len(uniq) > 1 else uniq[-1]
todo = [d for d in all_dates if str(d.date()) > st["last_date"] and d <= cal[-1]]
print(f"processing {len(todo)} day(s) up to {cal[-1].date()}")
for d in todo:
    apply_day_mom(d, st, uni, cal, cfg)

# inception: a fresh book deploys at the NEXT session's open using ranks decided
# at the latest close (standard fund inception — not a look-ahead case).
if FRESH and st.get("pending") is None:
    st["pending"] = {"exec_date": None, "syms": dm.momentum_rank(uni, cal[-1], cfg),
                     "inception": True}
    print(f"inception deployment queued: {st['pending']['syms']}")

st_path.parent.mkdir(parents=True, exist_ok=True)
st_path.write_text(json.dumps(st, indent=2))
print("state saved:", st_path)

# COMMAND ----------

# DBTITLE 1,Report + combined view
# Report: momentum book + COMBINED view with the C10 SMC book
eq = st["equity"][-1][1]
print(f"=== MOMENTUM PAPER BOOK — processed through {st['last_date']} ===")
print(f"equity ₹{eq:,.0f} ({(eq / st['capital'] - 1) * 100:+.2f}%) | cash ₹{st['cash']:,.0f} | positions {len(st['holdings'])} | orders {len(st['trades'])}")

print("\nCurrent holdings:")
if st["holdings"]:
    for s, q in sorted(st["holdings"].items()):
        px = st["last_close"].get(s, float("nan"))
        print(f"  {s:12s} qty {q:>6d}  last ₹{px:>9.2f}")
else:
    print("  (none)")

print("\nPending rebalance:" + (f" execute {st['pending']['exec_date']} -> {st['pending']['syms']}" if st.get("pending") else " (none — waiting for next month-end)"))

# combined view with C10 book (separate ₹1M accounts each)
print("\n=== COMBINED VIEW (equal-weight of the two books) ===")
try:
    smc = json.loads(Path(SMC_STATE).read_text())
    smc_eq = smc["equity"][-1][1] if smc["equity"] else smc["capital"]
    r_smc = smc_eq / smc["capital"] - 1
    r_mom = eq / st["capital"] - 1
    print(f"C10 SMC book:  ₹{smc_eq:,.0f} ({r_smc*100:+.2f}%) | through {smc['last_date']}")
    print(f"Momentum book: ₹{eq:,.0f} ({r_mom*100:+.2f}%) | through {st['last_date']}")
    print(f"50/50 combined: {( (r_smc + r_mom) / 2 * 100):+.2f}%")
except FileNotFoundError:
    print("(C10 SMC state file not found — run the SMC runner first)")

# COMMAND ----------

# DBTITLE 1,Honest notes
# MAGIC %md
# MAGIC ## Honest notes (do not remove)
# MAGIC
# MAGIC * Walk-forward OOS 2022-26: +58.1% (selector) / +62.6% (this fixed config) vs NIFTY +29.1% — validated, but with severe survivorship inflation: ranking past returns over TODAY'S large-cap list systematically favors past winners. Absolute return levels are upper bounds.
# MAGIC * **2026: every momentum variant tested was negative (-3% to -20%).** The momentum edge is currently in drawdown — do not extrapolate the 2019-23 run rates.
# MAGIC * The NIFTY regime gate HURTS this strategy (misses recoveries) — deliberately ungated, unlike the C10 book.
# MAGIC * Correlation with C10 ≈ 0.51 → real diversification; the combined book is the actual product being tested.
# MAGIC * Costs: full NSE delivery charges + 0.05% slippage per side, ~448 orders over 8 years (~₹35/order avg).
# MAGIC * This is research/paper-trading infrastructure, not investment advice.