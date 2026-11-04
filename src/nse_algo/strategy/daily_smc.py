"""Daily SMC + Price Action swing strategy (built Oct 2026).

Validated on real yfinance daily data, 182-stock NIFTY large+midcap
universe, 2018-2026, full NSE delivery costs (~0.35% round trip):

* Selected config ("C10"): three long-only setups on LARGE CAPS only —
  midcap breakouts failed in the 2024-26 rotation regime.
* Full period 2018-2026: +161.5%, Sharpe 0.85, MaxDD -24.9%, 2,268 trades
  (~1.04/day), win 49.1%, PF 1.29.
* Walk-forward OOS 2021-26: +79% vs NIFTY +62% (selection by trailing
  return; trailing-Sharpe selection gives +64% — robust to metric).

Honest limitations (do not remove):
* Survivorship bias: universe is today's constituents.
* OOS is partially contaminated: the large-cap restriction was motivated
  by observing 2024 midcap failure during research.
* 2025 (-7.9%) and 2026 (-2.8%) OOS years were negative; the edge is
  carried by 2021/2023/2024.
* NIFTY benchmark is price-only (no TRI).
* Circuit breakers were tested AFTER observing 2025-26 weakness and NONE
  improved the validated profile — trailing-P&L deadlocks (never reactivates);
  peak-drawdown variants (10/15/20% halt, half-width resume) all REDUCE
  full-period return (+125.9% OFF vs -2.2%/-10.2%/-13.4% with breakers)
  because the strategy's edge comes from the crash-recovery legs the
  breaker sits out. In OOS 2021-26, dd15/resume7.5 trades return for
  drawdown (+68.7% vs +81.7%, MaxDD -16.9% vs -21.2%, Sharpe 0.78 vs 0.74)
  — available as a conservative option, default OFF.
* Universe cache re-downloads can shift pre-2021 results by a few points
  (yfinance revises split/dividend adjustments); 2021+ results are stable.

Execution model (anti-look-ahead):
* A signal decided at the CLOSE of day T fills at the OPEN of T+1.
* Stops/targets checked intrabar, stop-first when both touchable.
* Time exits at the close of the max-hold day (MOC order).
* Gap-aware: gap-through-stop fills at the open, gap-through-target too.

This module is self-contained (numpy + pandas only) so it can be imported
from any path — including /tmp copies that bypass workspace-mount lag.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# ── Universe ────────────────────────────────────────────────────────
# Large caps (NIFTY 100 members as of the research universe build, Oct 2026).
LARGE_CAPS = frozenset("""
HDFCBANK ICICIBANK SBIN KOTAKBANK AXISBANK INDUSINDBK BANDHANBNK AUBANK PNB CANBK
FEDERALBNK BANKBARODA IDFCFIRSTB BAJFINANCE BAJAJFINSV CHOLAFIN MUTHOOTFIN PFC RECLTD SBILIFE
HDFCLIFE ICICIGI LICI TCS INFY WIPRO HCLTECH TECHM PERSISTENT COFORGE MPHASIS OFSS
RELIANCE ONGC IOC BPCL GAIL PETRONET POWERGRID NTPC NHPC TATAPOWER ADANIPOWER ADANIGREEN
TATASTEEL JSWSTEEL HINDALCO VEDL NMDC SAIL COALINDIA JINDALSTEL NATIONALUM HINDCOPPER MARUTI
M&M BAJAJ-AUTO EICHERMOT HEROMOTOCO ASHOKLEY TVSMOTOR BOSCHLTD MRF HINDUNILVR ITC
NESTLEIND BRITANNIA TATACONSUM DABUR GODREJCP COLPAL UBL MARICO PGHH HINDPETRO SUNPHARMA DRREDDY
CIPLA DIVISLAB AUROPHARMA APOLLOHOSP LALPATHLAB ZYDUSLIFE BIOCON TORNTPHARM ULTRACEMCO GRASIM
AMBUJACEM SHREECEM ACC DALBHARAT RAMCOCEM DLF PRESTIGE OBEROIRLTY BHARTIARTL IDEA SUNTV NAUKRI
PVRINOX JUBLFOOD LT SIEMENS ABB BHEL BEL CUMMINSIND THERMAX HAVELLS POLYCAB ADANIPORTS ADANIENT
TITAN ASIANPAINT BERGEPAINT DMART PAGEIND IRCTC IGL CONCOR TATACOMM ABCAPITAL RBLBANK YESBANK
ADANIENSOL TATAELXSI CDSL INDIAMART INDIGO
""".split())

# Midcaps kept in the data cache (NOT traded by the selected config — kept
# for research candidates).
MIDCAPS = frozenset("""
SHRIRAMFIN KPITTECH SONACOMS EXIDEIND MAHLOG RITES RVNL IRFC MAZDOCK BDL
CESC SJVN NLCINDIA SUZLON METROPOLIS TRENT BSOFT DEVYANI KALYANKJIL PATANJALI ASTERDM MAXHEALTH
RAINBOW KIMS ASTRAL PIDILITIND CEATLTD APOLLOTYRE JKTYRE ESCORTS BRIGADE GODREJPROP LODHA
APLAPOLLO WELCORP GRAVITA GLENMARK LAURUSLABS AARTIIND ALKEM
LUPIN TORNTPOWER IPCALAB ABBOTINDIA COROMANDEL J&KBANK UCOBANK IOB MAHABANK
NYKAA PAYTM POLICYBZR EASEMYTRIP
""".split())

def _repo_root(here: Path) -> Path:
    """Find the repo root (contains src/ + data_cache/) from this file's path.
    Falls back to the file's own directory for /tmp copies."""
    for p in here.parents:
        if p.name == "strategy" and p.parent.name == "nse_algo" and p.parent.parent.name == "src":
            return p.parent.parent.parent
    return here.parent


_HERE = Path(__file__).resolve()
DEFAULT_CACHE_DIR = _repo_root(_HERE) / "data_cache" / "daily_universe"
NIFTY_CACHE = _repo_root(_HERE) / "data_cache" / "NIFTY_daily.csv"

# ── Cost model (NSE cash equity, DELIVERY — research defaults) ───────
COSTS = dict(
    brokerage_pct=0.0003, brokerage_cap=20.0,   # per ORDER, min(pct*value, cap)
    stt_pct=0.001,                               # 0.1% BOTH sides (delivery)
    exchange_pct=0.0000297,
    gst_pct=0.18,                                # on brokerage+exchange+SEBI
    sebi_pct=0.000001,
    stamp_buy_pct=0.00003,
    slippage_pct=0.0005,                        # per side
)


def trade_costs(value: float, is_buy: bool) -> float:
    """Total transaction cost for one order of `value` rupees."""
    c = COSTS
    brk = min(value * c["brokerage_pct"], c["brokerage_cap"])
    ex = value * c["exchange_pct"]
    se = value * c["sebi_pct"]
    gst = (brk + ex + se) * c["gst_pct"]
    stt = value * c["stt_pct"]
    stamp = value * c["stamp_buy_pct"] if is_buy else 0.0
    return brk + ex + se + gst + stt + stamp + value * c["slippage_pct"]


# ── Indicators ──────────────────────────────────────────────────────
def wilder_atr(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    prev_c = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    atr = np.full_like(tr, np.nan)
    if len(tr) > period:
        atr[period] = tr[1 : period + 1].mean()
        for i in range(period + 1, len(tr)):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def fractal_extrema(high: np.ndarray, low: np.ndarray, k: int):
    """Fractal swing highs/lows; a swing at i is confirmed at i+k."""
    n = len(high)
    sw_hi, sw_lo = np.full(n, np.nan), np.full(n, np.nan)
    for i in range(k, n - k):
        win_h, win_l = high[i - k : i + k + 1], low[i - k : i + k + 1]
        if high[i] >= win_h.max() and win_h.argmax() == k:
            sw_hi[i] = high[i]
        if low[i] <= win_l.min() and win_l.argmin() == k:
            sw_lo[i] = low[i]
    return sw_hi, sw_lo


# ── Signal generation ────────────────────────────────────────────────
def build_daily_signals(df: pd.DataFrame, warmup: int = 60) -> pd.DataFrame:
    """Per-day entry signals for one symbol. Decision at close T,
    execution at open T+1 (handled by the engine).

    Setups (priority order):
    SMC — uptrend + liquidity sweep of prior fractal swing low (k=3,
          close rejects back above) within 5 bars + close breaks the last
          confirmed minor swing high (k=2, CHoCH) + bullish candle.
          Stop ref: sweep low. 
    PA  — hammer/pin bar (lower wick >= 50% of range, upper wick <= 35%,
          close > SMA20, low touched the SMA20 zone) in an uptrend.
          Stop ref: signal-bar low.
          NOTE: the validated research runs are PIN-ONLY. A bullish-engulfing
          variant existed in the prototype but its condition was
          self-contradictory (required prev bar red AND green) and therefore
          never fired — engulfing was never actually evaluated.
    BRK — close above the prior 20-day closing high in an uptrend.
          Stop ref: close.

    Returns the OHLCV frame plus: setup, stop_ref (RAW level — the engine
    applies the ATR buffer), atr.
    """
    n = len(df)
    o, h, l, c = (df[x].values for x in ("open", "high", "low", "close"))
    sma20 = df["close"].rolling(20).mean().values
    sma50 = df["close"].rolling(50).mean().values
    atr = wilder_atr(df, 14)
    hi20_max = df["close"].rolling(20).max().shift(1).values

    sw_hi2, _ = fractal_extrema(h, l, 2)   # minor swing highs
    _, sw_lo3 = fractal_extrema(h, l, 3)    # structural swing lows

    last_swl = np.full(n, np.nan)
    last_min_hi = np.full(n, np.nan)
    cur_swl, cur_min_hi = np.nan, np.nan
    for i in range(n):
        if i - 3 >= 0 and not np.isnan(sw_lo3[i - 3]):
            cur_swl = sw_lo3[i - 3]
        if i - 2 >= 0 and not np.isnan(sw_hi2[i - 2]):
            cur_min_hi = sw_hi2[i - 2]
        last_swl[i], last_min_hi[i] = cur_swl, cur_min_hi

    swept = (l < last_swl) & (c > last_swl)
    sweep_age = np.full(n, np.inf)
    sweep_low = np.full(n, np.nan)
    for i in range(n):
        j = i
        while j >= 0 and (i - j) <= 5:
            if swept[j]:
                sweep_age[i] = i - j
                sweep_low[i] = l[j]
                break
            j -= 1

    rng = np.maximum(h - l, 1e-9)
    body_bot = np.minimum(o, c)
    lower_wick = body_bot - l
    upper_wick = h - np.maximum(o, c)
    bull = c > o
    touch20 = l <= sma20 * 1.005
    pin = bull & touch20 & (c > sma20) & (lower_wick >= 0.5 * rng) & (upper_wick <= 0.35 * rng)

    regime = (c > sma50) & (sma20 > sma50)
    choch = c > last_min_hi
    breakout = c > hi20_max

    setup = np.array([""] * n, dtype=object)
    stop_ref = np.full(n, np.nan)

    m_smc = regime & (sweep_age <= 5) & choch & bull & np.isfinite(atr)
    setup[m_smc] = "SMC"
    stop_ref[m_smc] = sweep_low[m_smc]

    m_pa = regime & pin & ~m_smc
    setup[m_pa] = "PA"
    stop_ref[m_pa] = l[m_pa]

    m_brk = regime & breakout & ~m_smc & ~m_pa & np.isfinite(atr)
    setup[m_brk] = "BRK"
    stop_ref[m_brk] = c[m_brk]

    # Warmup + NaN guards. NOTE: do NOT reject stop_ref >= close here —
    # BRK's raw ref IS the close (the ATR buffer is applied by the engine,
    # which also rejects any final stop >= close).
    bad = ~np.isfinite(stop_ref)
    setup[bad] = ""
    if warmup:
        setup[:warmup] = ""

    out = df[["open", "high", "low", "close", "volume"]].copy()
    out["setup"] = setup
    out["stop_ref"] = stop_ref
    out["atr"] = atr
    return out


# ── Data loading / refresh ───────────────────────────────────────────
def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    if df.index.tz is None:
        df.index = df.index.tz_localize("Asia/Kolkata")
    df.index = df.index.tz_convert("Asia/Kolkata").normalize()
    return df[~df.index.duplicated()].sort_index()


def load_universe(cache_dir: Path | None = None, min_bars: int = 1000) -> dict[str, pd.DataFrame]:
    cache = Path(cache_dir or DEFAULT_CACHE_DIR)
    out = {}
    for f in sorted(cache.glob("*.csv")):
        df = pd.read_csv(f, index_col=0, parse_dates=True)
        if len(df) < min_bars:
            continue
        df = _normalize_index(df)
        out[f.stem] = df[["open", "high", "low", "close", "volume"]].astype(float)
    return out


def load_nifty(path: Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(path or NIFTY_CACHE, index_col=0, parse_dates=True)
    df = df.rename(columns={c: str(c).lower() for c in df.columns})
    return _normalize_index(df)[["open", "high", "low", "close", "volume"]].astype(float)


def update_universe_cache(cache_dir: Path | None = None, *,
                          max_age_hours: float = 20.0,
                          sleep_s: float = 0.35) -> list[str]:
    """Re-download any symbol whose cached CSV is missing or stale.
    Returns refreshed symbols. yfinance must be installed (pip)."""
    import yfinance as yf

    cache = Path(cache_dir or DEFAULT_CACHE_DIR)
    cache.mkdir(parents=True, exist_ok=True)
    refreshed = []
    for sym in sorted(LARGE_CAPS | MIDCAPS):
        path = cache / f"{sym}.csv"
        if path.exists() and (time.time() - path.stat().st_mtime) < max_age_hours * 3600:
            continue
        try:
            df = yf.Ticker(f"{sym}.NS").history(start="2018-01-01", interval="1d")
            if len(df) >= 1000:
                df = df.rename(columns={"Open": "open", "High": "high", "Low": "low",
                                        "Close": "close", "Volume": "volume"})
                df[["open", "high", "low", "close", "volume"]].to_csv(path)
                refreshed.append(sym)
        except Exception:
            pass  # delisted/renamed tickers are expected failures
        time.sleep(sleep_s)
    return refreshed


# ── Backtest engine ──────────────────────────────────────────────────
@dataclass(frozen=True)
class DailySMCConfig:
    """Validated 'C10' configuration (large caps)."""

    setups: tuple = ("SMC", "PA", "BRK")
    large_only: bool = True

    # Effective stops = stop_ref − buffer×ATR. These reproduce the
    # validated research runs EXACTLY (the prototype applied a hidden
    # double buffer; totals: SMC −1.0 ATR below sweep, PA −1.0 below bar
    # low, BRK −4.0 below close — a disaster stop; exits are mostly TIME).
    stop_buffer_atr: dict = field(default_factory=lambda: {"SMC": 1.0, "PA": 1.0, "BRK": 4.0})
    target_r: dict = field(default_factory=lambda: {"SMC": 2.5, "PA": 2.5, "BRK": 3.0})
    max_hold: dict = field(default_factory=lambda: {"SMC": 10, "PA": 8, "BRK": 12})
    stop_floor_pct: float = 0.01  # planned risk must be >= 1% of price

    nifty_regime_sma: int = 40   # NIFTY close > SMA(40) gate for entries

    capital: float = 1_000_000.0
    risk_pct: float = 0.01        # 1% of equity planned risk per trade
    max_positions: int = 20
    max_new_per_day: int = 6
    pos_cap_pct: float = 0.15

    # Circuit breaker (peak-drawdown, self-reactivating): halt NEW entries
    # when equity drops more than `breaker_dd_pct` below its running peak,
    # resume when it recovers to within `breaker_resume_pct` of the peak.
    # 0.0 disables (DEFAULT — validated config). Tested dd10/15/20 with
    # half-width resume: see docstring. Conservative option: 0.15/0.075.
    breaker_dd_pct: float = 0.0
    breaker_resume_pct: float = 0.0


def run_backtest(sig_by_sym: dict[str, pd.DataFrame], nifty: pd.DataFrame,
                 cfg: DailySMCConfig, start: str = "2018-01-01",
                 end: str = "2026-12-31") -> tuple[pd.Series, pd.DataFrame]:
    """Anti-look-ahead portfolio backtest. Returns (equity, trades)."""
    syms = sorted(sig_by_sym)
    arrs = {}
    for sym in syms:
        s = sig_by_sym[sym]
        arrs[sym] = dict(
            pos={ts: i for i, ts in enumerate(s.index)},
            o=s["open"].values, h=s["high"].values, l=s["low"].values,
            c=s["close"].values, setup=s["setup"].values,
            sref=s["stop_ref"].values, atr=s["atr"].values,
        )
    all_dates = sorted({d for sym in syms for d in sig_by_sym[sym].index})

    nifty_bull = nifty["close"] > nifty["close"].rolling(cfg.nifty_regime_sma).mean()

    cash = cfg.capital
    positions: dict[str, dict] = {}
    pending: dict[str, tuple] = {}
    trades: list[dict] = []
    equity: dict[pd.Timestamp, float] = {}
    eq_marks: list[float] = []
    last_eq = cfg.capital
    days_halted = 0
    halted = False
    peak_eq = cfg.capital

    for t in all_dates:
        if not (start <= str(t.date()) <= end):
            continue

        # ── 1) bars: fills, exits ──
        for sym in syms:
            a = arrs[sym]
            i = a["pos"].get(t)
            if i is None:
                continue
            o, h, l, c = a["o"][i], a["h"][i], a["l"][i], a["c"][i]

            if sym in pending and sym not in positions:
                stp, tgt, setup = pending.pop(sym)
                fill = o
                if fill <= stp:
                    continue  # gapped below stop — entry invalid
                dist = fill - stp
                qty = int((last_eq * cfg.risk_pct) / dist)
                qty = min(qty, int(last_eq * cfg.pos_cap_pct / fill), int(cash / fill))
                if qty <= 0:
                    continue
                cash -= qty * fill + trade_costs(qty * fill, True)
                positions[sym] = dict(qty=qty, entry=fill, stop=stp, target=tgt,
                                     setup=setup, bars=0)
                continue  # no exit checks on the entry bar

            if sym in positions:
                p = positions[sym]
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
                    cash += val - trade_costs(val, False)
                    trades.append(dict(
                        symbol=sym, setup=p["setup"], entry_date=t, exit_date=t,
                        qty=p["qty"], entry=p["entry"], exit=xp, stop=p["stop"],
                        pnl_gross=(xp - p["entry"]) * p["qty"], reason=rs))
                    del positions[sym]

        # ── 2) new entries decided at close of t ──
        if cfg.breaker_dd_pct > 0:
            peak_eq = max(peak_eq, last_eq)
            if not halted and last_eq < peak_eq * (1 - cfg.breaker_dd_pct):
                halted = True
            elif halted and last_eq >= peak_eq * (1 - cfg.breaker_resume_pct):
                halted = False
        days_halted += int(halted)

        if len(positions) < cfg.max_positions:
            breaker_off = halted
            nbull = (t in nifty_bull.index) and bool(nifty_bull.loc[t])
            if nbull and not breaker_off:
                count = 0
                for sym in syms:
                    if count >= cfg.max_new_per_day or len(positions) >= cfg.max_positions:
                        break
                    if sym in positions or sym in pending:
                        continue
                    if cfg.large_only and sym not in LARGE_CAPS:
                        continue
                    a = arrs[sym]
                    i = a["pos"].get(t)
                    if i is None:
                        continue
                    setup = a["setup"][i]
                    if setup not in cfg.setups:
                        continue
                    sref, atr = a["sref"][i], a["atr"][i]
                    if not (sref == sref and atr == atr):
                        continue
                    ccl = a["c"][i]
                    stp = sref - cfg.stop_buffer_atr[setup] * atr
                    r_ref = ccl - stp
                    if r_ref <= 0 or r_ref < cfg.stop_floor_pct * ccl:
                        continue
                    tgt = ccl + cfg.target_r[setup] * r_ref
                    pending[sym] = (stp, tgt, setup)
                    count += 1

        # ── 3) mark equity at close ──
        eq = cash
        for sym, p in positions.items():
            a = arrs[sym]
            i = a["pos"].get(t)
            if i is not None:
                eq += p["qty"] * a["c"][i]
        equity[t] = eq
        eq_marks.append(eq)
        last_eq = eq

    eqs = pd.Series(equity).sort_index()
    trd = pd.DataFrame(trades)
    if len(trd):
        trd["costs"] = [trade_costs(r.qty * r.entry, True) + trade_costs(r.qty * r.exit, False)
                        for r in trd.itertuples()]
        trd["pnl_net"] = trd["pnl_gross"] - trd["costs"]
    return eqs, trd


def yearly_returns(eqs: pd.Series) -> pd.Series:
    """Calendar-year returns from a continuous equity curve."""
    return eqs.resample("YE").last().pct_change().dropna() * 100.0


def summary(eqs: pd.Series, trd: pd.DataFrame) -> dict:
    """Standard honest metrics."""
    if len(trd) == 0:
        return dict(return_pct=0.0, cagr_pct=0.0, sharpe=float("nan"), maxdd_pct=0.0,
                    n_trades=0, trades_per_day=0.0, win_pct=float("nan"),
                    profit_factor=float("nan"), expectancy_inr=0.0)
    yrs = (eqs.index[-1] - eqs.index[0]).days / 365.25
    dr = eqs.pct_change().dropna()
    dd = (eqs / eqs.cummax() - 1).min()
    pf = (trd.loc[trd.pnl_net > 0, "pnl_net"].sum()
          / abs(trd.loc[trd.pnl_net <= 0, "pnl_net"].sum())) if (trd.pnl_net <= 0).any() else float("inf")
    return dict(
        return_pct=round((eqs.iloc[-1] / eqs.iloc[0] - 1) * 100, 1),
        cagr_pct=round(((eqs.iloc[-1] / eqs.iloc[0]) ** (1 / yrs) - 1) * 100, 1),
        sharpe=round(dr.mean() / dr.std() * np.sqrt(252), 2) if dr.std() > 0 else float("nan"),
        maxdd_pct=round(dd * 100, 1),
        n_trades=len(trd),
        trades_per_day=round(len(trd) / len(eqs), 2),
        win_pct=round((trd.pnl_net > 0).mean() * 100, 1) if len(trd) else float("nan"),
        profit_factor=round(pf, 2),
        expectancy_inr=round(trd.pnl_net.mean(), 0) if len(trd) else float("nan"),
    )
