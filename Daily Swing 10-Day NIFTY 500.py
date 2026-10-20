# Databricks notebook source
# DBTITLE 1,Daily Swing Strategy Title
# MAGIC %md
# MAGIC # Daily Swing Strategy — 10-Day Max Holding | NIFTY 500
# MAGIC
# MAGIC ## Strategy Overview
# MAGIC
# MAGIC **Goal**: Maximum profitability short-term swing trading on daily bars with 1-10 day holding period.
# MAGIC
# MAGIC ### Key Differences from Weekly Strategy
# MAGIC | Component | Weekly (Existing) | Daily (This Notebook) |
# MAGIC |---|---|---|
# MAGIC | Bars | Weekly (Friday close) | Daily (each trading day) |
# MAGIC | Universe | 130 NIFTY 200 | 250+ NIFTY 500 (incl. midcaps) |
# MAGIC | Max holding | 4 weeks | 10 days |
# MAGIC | SMA trend | SMA(50) weekly | SMA(50) daily |
# MAGIC | Stop loss | 1.5x ATR(14) weekly | 1.5x ATR(14) daily (tighter) |
# MAGIC | Take profit | 3.0x ATR(14) weekly | 2.0x ATR(14) daily (faster) |
# MAGIC | NIFTY regime | SMA(40) weekly | SMA(50) daily |
# MAGIC | Rebalance | Weekly | Daily |
# MAGIC | ML data | ~36K rows | ~250K rows (7x more) |
# MAGIC
# MAGIC ### Advantage of Daily Bars
# MAGIC - 7x more data points for ML training (250 bars/yr vs 52 weeks/yr)
# MAGIC - Tighter stops = smaller risk per trade
# MAGIC - Faster exits = quicker capital recycling
# MAGIC - More trades = more statistical confidence
# MAGIC - Midcap stocks = more opportunities, higher volatility = bigger swings

# COMMAND ----------

# DBTITLE 1,Install Dependencies
# MAGIC %pip install lightgbm vaderSentiment yfinance -q
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Imports Config Universe
import sys, os, json, time, warnings, requests
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET

REPO_ROOT = "/Workspace/Repos/lokigovindasamy@gmail.com/algo_trade_data_bricks"
SRC_PATH = os.path.join(REPO_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

import pandas as pd
import numpy as np
import yfinance as yf
import lightgbm as lgb
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
from nse_algo.features.indicators import sma, ema, rsi, atr, macd, bollinger_bands, stochastic
warnings.filterwarnings("ignore")
analyzer = SentimentIntensityAnalyzer()

# Daily Swing Config
CONFIG = {
    "max_positions": 12, "initial_capital": 100_000,
    "stop_atr_mult": 1.5, "target_atr_mult": 2.0,
    "max_holding_days": 10, "rsi_min": 35, "rsi_max": 72,
    "roce_min": 12.0, "nifty_regime_sma": 50,
    "sma_trend": 50, "ml_train_fraction": 0.60,
    "ml_gap_days": 10, "vol_lookback": 20,
    "barrier_upper": 2.0, "barrier_lower": 1.0,
}

# NIFTY 500 Universe: Large caps + Midcaps + selected Small caps
UNIVERSE = [
    # Large Cap (NIFTY 100)
    "HDFCBANK","ICICIBANK","SBIN","KOTAKBANK","AXISBANK","INDUSINDBK","BANDHANBNK","AUBANK","PNB","CANBK",
    "FEDERALBNK","BANKBARODA","IDFCFIRSTB","BAJFINANCE","BAJAJFINSV","CHOLAFIN","MUTHOOTFIN","PFC","RECLTD","SBILIFE",
    "HDFCLIFE","ICICIPRULI","ICICIGI","LICI","TCS","INFY","WIPRO","HCLTECH","TECHM","PERSISTENT",
    "COFORGE","MPHASIS","OFSS","RELIANCE","ONGC","IOC","BPCL","GAIL","PETRONET","POWERGRID",
    "NTPC","NHPC","TATAPOWER","ADANIPOWER","ADANIGREEN","TATASTEEL","JSWSTEEL","HINDALCO","VEDL","NMDC",
    "SAIL","COALINDIA","JINDALSTEL","NATIONALUM","HINDCOPPER","MARUTI","M&M","TATAMOTORS","BAJAJ-AUTO","EICHERMOT",
    "HEROMOTOCO","ASHOKLEY","TVSMOTOR","BOSCHLTD","MRF","HINDUNILVR","ITC","NESTLEIND","BRITANNIA","TATACONSUM",
    "DABUR","GODREJCP","COLPAL","UBL","MARICO","PGHH","HINDPETRO","SUNPHARMA","DRREDDY","CIPLA",
    "DIVISLAB","AUROPHARMA","APOLLOHOSP","LALPATHLAB","ZYDUSLIFE","BIOCON","TORNTPHARM","ULTRACEMCO","GRASIM","AMBUJACEM",
    "SHREECEM","ACC","DALBHARAT","RAMCOCEM","DLF","PRESTIGE","OBEROIRLTY","BHARTIARTL","IDEA","SUNTV",
    "NAUKRI","PVRINOX","JUBLFOOD","LT","SIEMENS","ABB","BHEL","BEL","CUMMINSIND","THERMAX",
    "HAVELLS","POLYCAB","ADANIPORTS","ADANIENT","TITAN","ASIANPAINT","BERGEPAINT","DMART","PAGEIND","IRCTC",
    "IGL","CONCOR","TATACOMM","ABCAPITAL","RBLBANK","YESBANK","ADANIENSOL","TATAELXSI","CDSL","INDIAMART",
    "FSL","ZOMATO","MCDOWELL-N","GUJGASLTD","GMRINFRA",
    # Midcap (NIFTY Midcap 150)
    "ABCAPITAL","AUBANK","BANDHANBNK","BANKBARODA","CANBK","FEDERALBNK","IDFCFIRSTB","PNB","RBLBANK","YESBANK",
    "ABCAPITAL","CHOLAFIN","MUTHOOTFIN","PFC","RECLTD","SHRIRAMFIN","SBICARDS","HDFCAMC","UTIAMC","LICI",
    "PHILIPSB","AVANTIFEED","MASTEK","TATAELXSI","KPITTECH","LTIM","OFSS","PERSISTENT","SONACOMS","CMRSL",
    "EXIDEIND","AMARAJABAT","BCLR","GMRINFRA","MAHLOG","RITES","RVNL","IRFC","MAZDOCK","BDL",
    "BEL","BHEL","CESC","GMRINFRA","TATAPOWER","NHPC","SJVN","NLCINDIA","GMMPHARMA","SUZLON",
    "ADANIENSOL","ADANIGREEN","ADANIPOWER","ONGC","BPCL","GAIL","IGL","MGL","GUJGASLTD","HINDPETRO",
    "METROPOLIS","LALPATHLAB","FHL","KOKUYO","HLVLTD","AGI","GLAXO","SANOFI","PROcter","ECLERX",
    "JUBLFOOD","DEVYANI","SAPSTAR","INOXWIND","KALYANKJIL","TRENT","VMART","BSOFT","CDSL","INDIAMART",
    "PATANJALI","TATAINVEST","KFCONSUMER","BBFC","ASTERDM","MAXHEALTH","RAINBOW","KIMS","ARSHIA","LALPATHLAB",
    # Additional Midcaps + Small Caps (high liquidity)
    "GMRINFRA","MFSL","PEL","PIDIILIND","TIINDIA","KSB","ADVENCO","BAYERCROP","PIDILITIND","SIEMENS",
    "ASTRAL","FINPIPE","PRADIP","SUPREMEIND","KINGFA","MASTEK","CEATLTD","MRF","APOLLOTYRE","JKTYRE",
    "ESCORTS","BRIGADE","PRESTIGE","GODREJPROP","OBEROIRLTY","LODHA","PHOENIXMILL","OBEROI","DLF","DBREAL",
    "ANURAS","ASTRAZEN","BLUEDART","TCIEXP","MAHSCOOTER","SAIL","JSWSTEEL","TATASTEEL","APLAPOLLO","WELCORP",
    "RNRL","GALLANTT","GRAVITA","MAGADI","NATCOPHARMA","GLENMARK","LAURUSLABS","AJIPHARMA","AARTIDRUGS","AARTIIND",
    "ALKEM","LUPIN","TORNTPOWER","IPCALAB","ABBOTINDIA","GODFRIPRO","CHOLAHD","PRINCEPIPE","WELCORP","SUPRIYA",
    "BALAKRISHNA","BALAMINES","TATVA","ANANTRAJ","GHCL","DCM SHRIRAM","PIIND","COROMANDEL","CHAMBALFERT","FERTILISERS",
    "NFL","DFMFOODS","KARURVYSYA","J&KBANK","SBIN","UCOBANK","IOB","MAHABANK","DENA","VijAYABANK",
    "INDIGO","SPICEJET","HONASA","NYKAA","ZOMATO","PAYTM","POLICYBZR","CARTRADE","EASEMYTRIP","IXIGO",
    "FSL","RBLBANK","YESBANK","ADANIENSOL","TATAELXSI","CDSL","INDIAMART","ABSLNIFTYETF","HDFCSMLCAP",
]
seen = set()
UNIVERSE = [x for x in UNIVERSE if not (x in seen or seen.add(x))]
print(f"✅ Universe: {len(UNIVERSE)} unique stocks (Large + Mid + Small cap)")
print(f"✅ Config: {CONFIG['max_holding_days']}-day max hold, stop={CONFIG['stop_atr_mult']}x ATR, target={CONFIG['target_atr_mult']}x ATR")
print(f"✅ All imports ready | LightGBM {lgb.__version__} | yfinance {yf.__version__}")

# COMMAND ----------

# DBTITLE 1,Fetch Daily Data
# Fetch DAILY bars for all stocks (8+ years)
start_date = "2018-01-01"
end_date = pd.Timestamp.now().strftime("%Y-%m-%d")
print(f"Fetching DAILY data from {start_date} to {end_date} for {len(UNIVERSE)} stocks...")

daily_data = {}
failed = []
for i, sym in enumerate(UNIVERSE):
    try:
        df = yf.Ticker(f"{sym}.NS").history(start=start_date, end=end_date, interval="1d")
        if len(df) < 250:
            failed.append(sym)
            continue
        df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
        if len(df) >= 250:
            daily_data[sym] = df.dropna()
    except:
        failed.append(sym)
    if (i + 1) % 30 == 0:
        time.sleep(0.5)
        print(f"  ... {i+1}/{len(UNIVERSE)} processed, {len(daily_data)} successful")

# NIFTY 50 daily
nifty_daily = yf.Ticker("^NSEI").history(start=start_date, end=end_date, interval="1d")
nifty_daily = nifty_daily.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})

print(f"\n✅ Daily data loaded for {len(daily_data)}/{len(UNIVERSE)} stocks")
print(f"   NIFTY 50: {len(nifty_daily)} daily bars")
print(f"   Avg bars per stock: {np.mean([len(v) for v in daily_data.values()]):.0f}")
print(f"   Period: {list(daily_data.values())[0].index[0].date()} to {list(daily_data.values())[0].index[-1].date()}")
if failed:
    print(f"   Failed ({len(failed)}): {failed[:15]}")

# COMMAND ----------

# DBTITLE 1,Features + Labels + ML Training
# Build Daily Features + Triple-Barrier Labels (10-day max) + Train LightGBM
all_feats = []
for sym, df in daily_data.items():
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    f = pd.DataFrame(index=df.index)
    f["symbol"] = sym
    # Momentum (daily)
    for p in [1, 2, 3, 5, 10, 20, 50]:
        f[f"mom_{p}d"] = c.pct_change(p)
    # SMA ratios
    for p in [10, 20, 50]:
        s = sma(c, p)
        f[f"sma_ratio_{p}"] = c / s
        f[f"price_vs_sma{p}"] = (c / s - 1) * 100
    f["above_sma50"] = (c > sma(c, 50)).astype(int)
    f["above_sma20"] = (c > sma(c, 20)).astype(int)
    # Indicators
    f["rsi_14"] = rsi(c, 14)
    _, _, f["macd_hist"] = macd(c)
    bb_u, bb_m, bb_l = bollinger_bands(c, 20, 2.0)
    f["bb_pctb"] = (c - bb_l) / (bb_u - bb_l)
    f["atr_pct"] = (atr(h, l, c, 14) / c) * 100
    low_14, high_14 = l.rolling(14).min(), h.rolling(14).max()
    f["williams_r"] = ((high_14 - c) / (high_14 - low_14)) * -100
    f["stoch_k"], f["stoch_d"] = stochastic(h, l, c, 14, 3)
    # Volume
    f["vol_ratio_5"] = v / v.rolling(5).mean()
    f["vol_mom_5"] = v.pct_change(5)
    # Volatility (daily)
    ret = c.pct_change()
    f["realized_vol_5d"] = ret.rolling(5).std() * np.sqrt(252)
    f["realized_vol_20d"] = ret.rolling(20).std() * np.sqrt(252)
    f["roc_10d"] = ((c - c.shift(10)) / c.shift(10)) * 100
    # ATR for position sizing
    f["atr_14"] = atr(h, l, c, 14)
    all_feats.append(f)

feat_df = pd.concat(all_feats).sort_index()
tech_cols = [c for c in feat_df.columns if c not in ["symbol", "atr_14"]]

# Cross-sectional features (per day)
for col, rn in [("mom_10d", "rank_mom10"), ("rsi_14", "rank_rsi"), ("vol_ratio_5", "rank_vol")]:
    feat_df[rn] = feat_df.groupby(level=0)[col].rank(pct=True)
for col, zn in [("mom_10d", "zscore_mom10"), ("rsi_14", "zscore_rsi"), ("mom_5d", "zscore_mom5")]:
    grp = feat_df.groupby(level=0)[col]
    feat_df[zn] = (grp.transform(lambda x: x - x.mean()) / grp.transform(lambda x: x.std() + 1e-8))
cross_cols = ["rank_mom10", "rank_rsi", "rank_vol", "zscore_mom10", "zscore_rsi", "zscore_mom5"]

# NIFTY regime (daily)
nc = nifty_daily["close"]
ns = sma(nc, CONFIG["nifty_regime_sma"])
nf = pd.DataFrame(index=nifty_daily.index)
nf["nifty_regime"] = (nc > ns).astype(int)
nf["nifty_mom_5d"] = nc.pct_change(5)
nf["nifty_mom_20d"] = nc.pct_change(20)
nf["nifty_vol_20d"] = nc.pct_change().rolling(20).std() * np.sqrt(252)
for col in nf.columns:
    feat_df[col] = feat_df.index.map(nf[col])
regime_cols = list(nf.columns)

# Triple-barrier labels (10-day max, daily volatility)
def daily_triple_barrier(df, vol_lb=20, up=2.0, lo=1.0, max_d=10):
    close, high, low = df["close"].values, df["high"].values, df["low"].values
    n = len(close)
    labels = np.full(n, np.nan)
    rets = np.diff(np.log(close))
    for i in range(n - 1):
        if i < vol_lb:
            continue
        vol = np.std(rets[i-vol_lb:i])
        if vol <= 0 or np.isnan(vol):
            continue
        up_b = close[i] * (1 + up * vol)
        lo_b = close[i] * (1 - lo * vol)
        for j in range(1, min(max_d + 1, n - i)):
            if low[i + j] <= lo_b:
                labels[i] = -1
                break
            elif high[i + j] >= up_b:
                labels[i] = 1
                break
        else:
            labels[i] = 0
    return pd.Series(labels, index=df.index)

all_labels = []
for sym, df in daily_data.items():
    lbls = daily_triple_barrier(df, CONFIG["vol_lookback"], CONFIG["barrier_upper"], CONFIG["barrier_lower"], CONFIG["max_holding_days"])
    all_labels.append(pd.DataFrame({"symbol": sym, "label": lbls}, index=df.index))
label_df = pd.concat(all_labels).sort_index()
feat_df["label"] = label_df["label"].values
feat_df["target"] = (feat_df["label"] == 1).astype(int)

all_feat_cols = tech_cols + cross_cols + regime_cols
print(f"✅ Features: {len(all_feat_cols)} total")
print(f"   Technical: {len(tech_cols)} | Cross: {len(cross_cols)} | Regime: {len(regime_cols)}")
print(f"   Rows: {len(feat_df):,} | Stocks: {feat_df['symbol'].nunique()}")
print(f"   Target positive rate: {feat_df['target'].mean():.1%}")

# Train LightGBM
model_df = feat_df.dropna(subset=["target"]).copy()
model_df[all_feat_cols] = model_df[all_feat_cols].replace([np.inf, -np.inf], np.nan)
imp = SimpleImputer(strategy="median")
model_df[all_feat_cols] = imp.fit_transform(model_df[all_feat_cols])

all_dates = sorted(model_df.index.unique())
n_tr = int(len(all_dates) * CONFIG["ml_train_fraction"])
tr_end = all_dates[n_tr]
gap_end = all_dates[min(n_tr + CONFIG["ml_gap_days"], len(all_dates) - 1)]
train = model_df[model_df.index < tr_end].copy()
test = model_df[model_df.index >= gap_end].copy()
print(f"\nTrain: {train.index[0].date()} to {train.index[-1].date()} ({len(train):,} rows)")
print(f"Test:  {test.index[0].date()} to {test.index[-1].date()} ({len(test):,} rows)")

lgb_model = lgb.LGBMClassifier(n_estimators=500, max_depth=6, learning_rate=0.03, num_leaves=31, min_child_samples=200, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbose=-1)
lgb_model.fit(train[all_feat_cols], train["target"].values, eval_set=[(test[all_feat_cols], test["target"].values)], callbacks=[lgb.early_stopping(50, verbose=False)])
train_auc = roc_auc_score(train["target"].values, lgb_model.predict_proba(train[all_feat_cols])[:, 1])
test_auc = roc_auc_score(test["target"].values, lgb_model.predict_proba(test[all_feat_cols])[:, 1])
test["ml_prob"] = lgb_model.predict_proba(test[all_feat_cols])[:, 1]
test["pred_rank"] = test.groupby(level=0)["ml_prob"].rank(pct=True, ascending=True)
top_dec = test[test["pred_rank"] >= 0.9]
print(f"\n✅ LightGBM: Train AUC={train_auc:.3f}, Test AUC={test_auc:.3f}")
print(f"   Top decile positive rate: {top_dec['target'].mean():.1%} vs baseline {test['target'].mean():.1%}")
print(f"\n─── Top 15 Features ───")
imp_df = pd.DataFrame({"feature": all_feat_cols, "importance": lgb_model.feature_importances_}).sort_values("importance", ascending=False)
display(imp_df.head(15))

# COMMAND ----------

# DBTITLE 1,Daily Swing Backtest
# Daily Swing Trading Backtest
SWING = {"max_positions": 12, "initial_capital": 100_000, "stop_atr_mult": 1.5, "target_atr_mult": 2.0, "max_holding_days": 10, "rsi_min": 35, "rsi_max": 72}

def get_ind(sym, dt):
    if sym not in daily_data or dt not in daily_data[sym].index:
        return None
    df = daily_data[sym]
    idx = df.index.get_loc(dt)
    if idx < 50:
        return None
    c, h, l = df["close"], df["high"], df["low"]
    return {"close": c.iloc[idx], "high": h.iloc[idx], "low": l.iloc[idx], "rsi": rsi(c, 14).iloc[idx], "sma50": sma(c, 50).iloc[idx], "atr": atr(h, l, c, 14).iloc[idx]}

def nifty_bull(dt):
    if dt not in nifty_daily.index:
        return False
    idx = nifty_daily.index.get_loc(dt)
    if idx < 50:
        return False
    return nifty_daily["close"].iloc[idx] > sma(nifty_daily["close"], 50).iloc[idx]

test_dates = sorted(test.index.unique())
print(f"Backtest: {test_dates[0].date()} to {test_dates[-1].date()} ({len(test_dates)} trading days)")

capital = SWING["initial_capital"]
positions = {}
pv_history = []
daily_returns = []
trade_log = []

for di, dt in enumerate(test_dates):
    # 1. Check exits
    to_close = []
    for sym, pos in positions.items():
        if sym not in daily_data or dt not in daily_data[sym].index:
            to_close.append((sym, pos["entry_price"], "data_missing"))
            continue
        row = daily_data[sym].loc[dt]
        cp, hp, lp = row["close"], row["high"], row["low"]
        days_held = di - test_dates.index(pos["entry_date"])
        reason = None
        ep = cp
        if lp <= pos["stop"]:
            reason = "stop_loss"
            ep = pos["stop"]
        elif hp >= pos["target"]:
            reason = "take_profit"
            ep = pos["target"]
        elif days_held >= SWING["max_holding_days"]:
            reason = "time_exit"
        elif cp < pos.get("sma50_entry", float('inf')) * 0.97:
            reason = "trend_exit"
        if reason:
            to_close.append((sym, ep, reason))
    for sym, ep, reason in to_close:
        pos = positions.pop(sym)
        pnl = (ep - pos["entry_price"]) / pos["entry_price"]
        capital += pos["shares"] * ep
        trade_log.append({"symbol": sym, "entry_date": pos["entry_date"].strftime("%Y-%m-%d"), "exit_date": dt.strftime("%Y-%m-%d"), "entry_price": pos["entry_price"], "exit_price": ep, "return_pct": round(pnl * 100, 2), "exit_reason": reason, "days_held": di - test_dates.index(pos["entry_date"])})
    # 2. New entries
    if nifty_bull(dt) and len(positions) < SWING["max_positions"]:
        preds = test[test.index == dt].sort_values("ml_prob", ascending=False)
        cands = []
        for _, row in preds.iterrows():
            sym = row["symbol"]
            if sym in positions or len(positions) + len(cands) >= SWING["max_positions"]:
                continue
            ind = get_ind(sym, dt)
            if ind is None:
                continue
            if ind["rsi"] < SWING["rsi_min"] or ind["rsi"] > SWING["rsi_max"]:
                continue
            if ind["close"] <= ind["sma50"]:
                continue
            cands.append({"symbol": sym, "ml_prob": row["ml_prob"], "close": ind["close"], "atr": ind["atr"], "sma50": ind["sma50"]})
        cands.sort(key=lambda x: x["ml_prob"], reverse=True)
        n_slots = SWING["max_positions"] - len(positions)
        for cand in cands[:n_slots]:
            if capital < 500:
                break
            size = capital / max(n_slots, 1)
            shares = size / cand["close"]
            capital -= shares * cand["close"]
            positions[cand["symbol"]] = {"entry_price": cand["close"], "entry_date": dt, "shares": shares, "stop": cand["close"] - SWING["stop_atr_mult"] * cand["atr"], "target": cand["close"] + SWING["target_atr_mult"] * cand["atr"], "sma50_entry": cand["sma50"]}
    # 3. Portfolio value
    pv = capital
    for sym, pos in positions.items():
        if sym in daily_data and dt in daily_data[sym].index:
            pv += pos["shares"] * daily_data[sym]["close"].loc[dt]
        else:
            pv += pos["shares"] * pos["entry_price"]
    pv_history.append({"date": dt.strftime("%Y-%m-%d"), "value": pv, "n_pos": len(positions), "nifty_bull": nifty_bull(dt)})
    if len(pv_history) > 1:
        daily_returns.append(pv / pv_history[-2]["value"] - 1)

# Close remaining
for sym, pos in list(positions.items()):
    ld = test_dates[-1]
    ep = daily_data[sym]["close"].loc[ld] if sym in daily_data and ld in daily_data[sym].index else pos["entry_price"]
    pnl = (ep - pos["entry_price"]) / pos["entry_price"]
    capital += pos["shares"] * ep
    trade_log.append({"symbol": sym, "entry_date": pos["entry_date"].strftime("%Y-%m-%d"), "exit_date": ld.strftime("%Y-%m-%d"), "entry_price": pos["entry_price"], "exit_price": ep, "return_pct": round(pnl * 100, 2), "exit_reason": "backtest_end", "days_held": len(test_dates) - 1 - test_dates.index(pos["entry_date"])})

# Metrics
pv_df = pd.DataFrame(pv_history)
tdf = pd.DataFrame(trade_log)
total_ret = (pv_df["value"].iloc[-1] / SWING["initial_capital"] - 1) * 100
n_yrs = len(pv_df) / 252
cagr = ((pv_df["value"].iloc[-1] / SWING["initial_capital"]) ** (1 / n_yrs) - 1) * 100 if n_yrs > 0 else 0
dr = np.array(daily_returns)
sharpe = (dr.mean() / (dr.std() + 1e-8)) * np.sqrt(252) if len(dr) > 1 else 0
mdd = ((pv_df["value"] / pv_df["value"].cummax() - 1) * 100).min()
# NIFTY benchmark
nt = nifty_daily[nifty_daily.index >= test_dates[0]]
n_ret = (nt["close"].iloc[-1] / nt["close"].iloc[0] - 1) * 100
n_cagr = ((nt["close"].iloc[-1] / nt["close"].iloc[0]) ** (1 / n_yrs) - 1) * 100 if n_yrs > 0 else 0
n_dr = nt["close"].pct_change().dropna()
n_sharpe = (n_dr.mean() / (n_dr.std() + 1e-8)) * np.sqrt(252)
n_mdd = ((nt["close"] / nt["close"].cummax() - 1) * 100).min()

if len(tdf) > 0:
    wins = tdf[tdf["return_pct"] > 0]
    losses = tdf[tdf["return_pct"] <= 0]
    wr = len(wins) / len(tdf) * 100
    aw = wins["return_pct"].mean() if len(wins) > 0 else 0
    al = losses["return_pct"].mean() if len(losses) > 0 else 0
    pf = abs(wins["return_pct"].sum() / losses["return_pct"].sum()) if len(losses) > 0 and losses["return_pct"].sum() != 0 else float('inf')
else:
    wr, aw, al, pf = 0, 0, 0, 0

print(f"\n{'='*70}")
print(f"  DAILY SWING STRATEGY - 10-Day Max Holding")
print(f"  Period: {test_dates[0].date()} to {test_dates[-1].date()} ({len(test_dates)} days, {n_yrs:.1f} years)")
print(f"{'='*70}")
print(f"\n  {'Metric':<25} {'Daily Swing':>15} {'NIFTY 50':>15} {'Diff':>10}")
print(f"  {'-'*65}")
print(f"  {'Total Return':<25} {total_ret:>14.1f}% {n_ret:>14.1f}% {total_ret - n_ret:>+9.1f}%")
print(f"  {'CAGR':<25} {cagr:>14.1f}% {n_cagr:>14.1f}% {cagr - n_cagr:>+9.1f}%")
print(f"  {'Sharpe Ratio':<25} {sharpe:>15.2f} {n_sharpe:>15.2f} {sharpe - n_sharpe:>+10.2f}")
print(f"  {'Max Drawdown':<25} {mdd:>14.1f}% {n_mdd:>14.1f}% {mdd - n_mdd:>+9.1f}%")
print(f"  {'-'*65}")
print(f"\n  Trade Statistics:")
print(f"    Total trades: {len(tdf)}")
print(f"    Win rate: {wr:.1f}%")
print(f"    Avg win: +{aw:.2f}%")
print(f"    Avg loss: {al:.2f}%")
print(f"    Profit factor: {pf:.2f}")
if len(tdf) > 0:
    print(f"    Avg days held: {tdf['days_held'].mean():.1f}")
    print(f"\n  Exit Breakdown:")
    for reason, count in tdf["exit_reason"].value_counts().items():
        ar = tdf[tdf["exit_reason"] == reason]["return_pct"].mean()
        print(f"    {reason}: {count} trades, avg {ar:+.2f}%")
if len(tdf) > 0 and len(wins) > 0:
    print(f"\n  Top 10 Winning Trades:")
    display(wins.nlargest(10, "return_pct")[["symbol", "entry_date", "exit_date", "entry_price", "exit_price", "return_pct", "exit_reason"]].round(2))
if len(tdf) > 0 and len(losses) > 0:
    print(f"\n  Top 10 Losing Trades:")
    display(losses.nsmallest(10, "return_pct")[["symbol", "entry_date", "exit_date", "entry_price", "exit_price", "return_pct", "exit_reason"]].round(2))
print(f"\n  Equity Curve (first/last 5):")
display(pv_df.head(5))
print("...")
display(pv_df.tail(5))
print(f"\n{'='*70}")
verdict = 'PROFITABLE - BEATS NIFTY' if total_ret > n_ret else 'NOT BEATING NIFTY'
print(f"  VERDICT: {verdict}")
print(f"  Daily Swing: {total_ret:.1f}% return, Sharpe {sharpe:.2f}, MaxDD {mdd:.1f}%")
print(f"  NIFTY B&H: {n_ret:.1f}% return, Sharpe {n_sharpe:.2f}, MaxDD {n_mdd:.1f}%")
print(f"  Alpha: {total_ret - n_ret:+.1f}% | Win Rate: {wr:.1f}% | PF: {pf:.2f} | Trades: {len(tdf)}")
print(f"{'='*70}")

# COMMAND ----------

# DBTITLE 1,Paper Trading Overview
# MAGIC %md
# MAGIC ## Paper Trading System — Daily Swing (10-Day)
# MAGIC
# MAGIC This cell runs the daily swing strategy in paper trading mode:
# MAGIC - Loads portfolio state from workspace JSON (persists between runs)
# MAGIC - Checks exits for open positions using latest daily prices
# MAGIC - Generates new BUY signals using ML + technical filters
# MAGIC - Saves updated state
# MAGIC - Shows current portfolio status and today's signals
# MAGIC
# MAGIC **Run this cell daily (after market close) to update signals.**

# COMMAND ----------

# DBTITLE 1,Paper Trading System
# Paper Trading - Daily Swing Strategy
import json
from datetime import datetime

STATE_PATH = "/Workspace/Repos/lokigovindasamy@gmail.com/algo_trade_data_bricks/daily_swing_paper_trade_state.json"
SWING = {"max_positions": 12, "initial_capital": 100_000, "stop_atr_mult": 1.5, "target_atr_mult": 2.0, "max_holding_days": 10, "rsi_min": 35, "rsi_max": 72}

# Load state
try:
    with open(STATE_PATH, "r") as f:
        state = json.load(f)
    print(f"📂 Loaded state from {STATE_PATH}")
except:
    state = {"capital": SWING["initial_capital"], "positions": {}, "trade_log": [], "start_date": datetime.now().isoformat()}
    print("🆕 No previous state - starting fresh daily swing paper trading")

print(f"   Capital: Rs {state['capital']:,.0f}")
print(f"   Open positions: {len(state['positions'])}")
print(f"   Total trades: {len(state['trade_log'])}")

now = datetime.now()
today_str = now.strftime("%Y-%m-%d")

# 1. NIFTY regime check (daily SMA50)
nc = nifty_daily["close"].iloc[-1]
ns = sma(nifty_daily["close"], 50).iloc[-1]
nb = nc > ns if not pd.isna(ns) else False
print(f"\n📊 NIFTY Regime: {'🟢 BULL' if nb else '🔴 BEAR'} (close={nc:.0f}, SMA50={ns:.0f})")

# 2. Check exits for open positions
exits = []
for sym, pos in list(state["positions"].items()):
    if sym not in daily_data:
        exits.append((sym, pos["entry_price"], "data_missing"))
        continue
    df = daily_data[sym]
    cp = df["close"].iloc[-1]
    hp = df["high"].iloc[-1]
    lp = df["low"].iloc[-1]
    days_held = 0
    try:
        entry_dt = datetime.fromisoformat(pos["entry_date"])
        days_held = (now - entry_dt).days
    except:
        days_held = 0
    reason = None
    ep = cp
    if lp <= pos["stop"]:
        reason = "stop_loss"
        ep = pos["stop"]
    elif hp >= pos["target"]:
        reason = "take_profit"
        ep = pos["target"]
    elif days_held >= SWING["max_holding_days"]:
        reason = "time_exit"
    elif cp < pos.get("sma50_entry", float('inf')) * 0.97:
        reason = "trend_exit"
    if reason:
        exits.append((sym, ep, reason))

for sym, ep, reason in exits:
    pos = state["positions"].pop(sym)
    pnl = (ep - pos["entry_price"]) / pos["entry_price"]
    state["capital"] += pos["shares"] * ep
    state["trade_log"].append({"symbol": sym, "entry_date": pos["entry_date"], "exit_date": today_str, "entry_price": pos["entry_price"], "exit_price": ep, "return_pct": round(pnl * 100, 2), "exit_reason": reason})
    emoji = "✅" if pnl > 0 else "❌"
    print(f"  {emoji} EXIT {sym}: {pos['entry_price']:.0f} → {ep:.0f} ({pnl*100:+.1f}%) - {reason}")

# 3. Generate new BUY signals
new_signals = []
if nb and len(state["positions"]) < SWING["max_positions"] and 'test' in dir() and test is not None:
    latest_date = test.index.max()
    preds = test[test.index == latest_date].sort_values("ml_prob", ascending=False)
    for _, row in preds.iterrows():
        sym = row["symbol"]
        if sym in state["positions"] or len(state["positions"]) + len(new_signals) >= SWING["max_positions"]:
            continue
        if sym not in daily_data:
            continue
        df = daily_data[sym]
        c, h, l = df["close"], df["high"], df["low"]
        r = rsi(c, 14).iloc[-1]
        s50 = sma(c, 50).iloc[-1]
        a14 = atr(h, l, c, 14).iloc[-1]
        cp = c.iloc[-1]
        if pd.isna(r) or pd.isna(s50) or pd.isna(a14):
            continue
        if r < SWING["rsi_min"] or r > SWING["rsi_max"]:
            continue
        if cp <= s50:
            continue
        new_signals.append({"symbol": sym, "ml_prob": row["ml_prob"], "close": cp, "rsi": r, "atr": a14, "sma50": s50, "stop": cp - SWING["stop_atr_mult"] * a14, "target": cp + SWING["target_atr_mult"] * a14})
    new_signals.sort(key=lambda x: x["ml_prob"], reverse=True)
    n_slots = SWING["max_positions"] - len(state["positions"])
    for sig in new_signals[:n_slots]:
        if state["capital"] < 500:
            break
        size = state["capital"] / max(n_slots, 1)
        shares = size / sig["close"]
        state["capital"] -= shares * sig["close"]
        state["positions"][sig["symbol"]] = {"entry_price": sig["close"], "entry_date": today_str, "shares": shares, "stop": sig["stop"], "target": sig["target"], "sma50_entry": sig["sma50"]}
        print(f"  🟢 BUY {sig['symbol']}: Rs {sig['close']:.0f} | Stop Rs {sig['stop']:.0f} | Target Rs {sig['target']:.0f} | ML {sig['ml_prob']:.2f} | RSI {sig['rsi']:.0f}")
else:
    if not nb:
        print("\n⚠️ NIFTY in BEAR regime - no new entries. Strategy in cash.")
    elif 'test' not in dir() or test is None:
        print("\n⚠️ ML model not available - run earlier cells for full signals.")
    elif len(state["positions"]) >= SWING["max_positions"]:
        print(f"\n📊 Portfolio full ({len(state['positions'])} positions) - no new entries.")

# 4. Save state
state["last_updated"] = now.isoformat()
with open(STATE_PATH, "w") as f:
    json.dump(state, f, indent=2)
print(f"\n💾 State saved to {STATE_PATH}")

# 5. Portfolio summary
print(f"\n{'='*70}")
print(f"  PAPER TRADING - Daily Swing (10-Day) | {today_str}")
print(f"{'='*70}")
pv = state["capital"]
pos_vals = []
for sym, pos in state["positions"].items():
    cp = daily_data[sym]["close"].iloc[-1] if sym in daily_data else pos["entry_price"]
    val = pos["shares"] * cp
    pv += val
    pnl_pct = (cp - pos["entry_price"]) / pos["entry_price"] * 100
    pos_vals.append({"symbol": sym, "entry": pos["entry_price"], "current": cp, "pnl_pct": pnl_pct, "value": val, "stop": pos["stop"], "target": pos["target"]})
tr = (pv / SWING["initial_capital"] - 1) * 100
print(f"\n  Portfolio Value: Rs {pv:,.0f} (Return: {tr:+.1f}%)")
print(f"  Cash: Rs {state['capital']:,.0f}")
print(f"  Open Positions: {len(state['positions'])}")
if pos_vals:
    print(f"\n─── Open Positions ───")
    display(pd.DataFrame(pos_vals).sort_values("pnl_pct", ascending=False).round(2))
if state["trade_log"]:
    print(f"\n─── Recent Trades (last 10) ───")
    tdf = pd.DataFrame(state["trade_log"])
    display(tdf.tail(10))
    w = (tdf["return_pct"] > 0).sum()
    print(f"\n  Win Rate: {w}/{len(tdf)} ({w/len(tdf)*100:.1f}%)")
    print(f"  Avg Return: {tdf['return_pct'].mean():.2f}%")
if new_signals:
    print(f"\n─── Today's BUY Signals ({len(new_signals)} stocks) ───")
    display(pd.DataFrame(new_signals).sort_values("ml_prob", ascending=False).round(2))
print(f"\n{'='*70}")
print(f"  Run daily (after market close) to update signals")
print(f"{'='*70}")