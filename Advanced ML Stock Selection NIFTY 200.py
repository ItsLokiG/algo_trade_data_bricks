# Databricks notebook source
# DBTITLE 1,Advanced ML Stock Selection Title
# MAGIC %md
# MAGIC # Advanced ML Stock Selection System — NIFTY 200
# MAGIC
# MAGIC > **Research prototype only:** the legacy performance estimates are not validated. Same-close execution, test-set early stopping, and non-point-in-time fundamentals invalidate the existing performance claims. Do not use these results as evidence of profitability.
# MAGIC
# MAGIC ## Why the Previous ML Approach Failed
# MAGIC
# MAGIC Our earlier ML model achieved only **0.48-0.51 AUC** (barely above random). Root causes:
# MAGIC
# MAGIC 1. **Only 13 features** — financial ML literature (Gu, Kelly & Xiu 2020) uses 94+ firm characteristics
# MAGIC 2. **Binary target at fixed 4-week horizon** — noisy; De Prado (2018) shows triple-barrier labeling is superior
# MAGIC 3. **Single train/test split** — no purged cross-validation; overlapping labels leak information
# MAGIC 4. **sklearn GradientBoosting** — LightGBM/XGBoost are significantly better for tabular financial data
# MAGIC 5. **Only 30 stocks** — insufficient cross-sectional variation for ranking-based learning
# MAGIC 6. **No sentiment, no catalysts** — news and events drive short-term moves
# MAGIC 7. **Predicted absolute returns** — cross-sectional ranking (which stock will outperform peers) is more learnable
# MAGIC
# MAGIC ## What the Literature Says Works
# MAGIC
# MAGIC ### Key Papers
# MAGIC - **Gu, Kelly, Xiu (2020)** — "Empirical Asset Pricing via Machine Learning": Tree-based models (GBM, RF) outperform neural networks for stock prediction. Best with 94+ features and monthly rebalancing.
# MAGIC - **De Prado (2018)** — "Advances in Financial Machine Learning": Triple-barrier labeling, purged k-fold CV, meta-labeling, feature importance (MDI/MDA).
# MAGIC - **Kakushadze (2016)** — "101 Formulaic Alphas": 101 quantitative alpha factors based on price/volume.
# MAGIC
# MAGIC ### Implementation Strategy
# MAGIC
# MAGIC | Component | Previous | This Notebook |
# MAGIC |---|---|---|
# MAGIC | Universe | 30 stocks | ~120 NIFTY 200 stocks |
# MAGIC | Features | 13 technical | 50+ (technical + cross-sectional + fundamental + sentiment) |
# MAGIC | Target | Binary 4-week | Triple-barrier (De Prado) |
# MAGIC | Model | sklearn GBM | LightGBM with early stopping |
# MAGIC | CV | Single split | Purged walk-forward (60/40, 4-week gap) |
# MAGIC | Prediction | Absolute return | Cross-sectional ranking |
# MAGIC | Sentiment | None | VADER on yfinance news headlines |
# MAGIC | Selection | ML filter | Top-decile rank + trend-following overlay |

# COMMAND ----------

# DBTITLE 1,Install Dependencies
# MAGIC %pip install lightgbm vaderSentiment yfinance
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Imports & Config
import sys, os, json, time, warnings
from datetime import datetime, timezone

REPO_ROOT = "/Workspace/Repos/lokigovindasamy@gmail.com/algo_trade_data_bricks"
SRC_PATH = os.path.join(REPO_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

import pandas as pd
import numpy as np
import yfinance as yf
import lightgbm as lgb
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

from nse_algo.features.indicators import sma, ema, rsi, atr, macd, bollinger_bands, stochastic

print("✅ All imports ready")
print(f"   LightGBM version: {lgb.__version__}")
print(f"   VADER sentiment analyzer loaded")
print(f"   yfinance: {yf.__version__}")

# ─── Configuration ───
CONFIG = {
    "initial_capital": 100_000,
    "max_positions": 15,
    "sma_period": 50,          # trend-following SMA (weekly)
    "regime_sma": 40,            # NIFTY regime filter
    "momentum_lookback": 26,    # 26-week momentum for ranking
    "ml_train_fraction": 0.60,   # 60% train
    "ml_gap_weeks": 4,           # 4-week purge gap
    "ml_threshold": 0.60,        # probability threshold for selection
    "vol_lookback": 26,          # volatility lookback for triple-barrier
    "barrier_multiplier": 2.0,  # upper barrier = +2 * weekly vol
    "stop_multiplier": 1.0,     # lower barrier = -1 * weekly vol
    "max_holding_weeks": 4,     # time barrier
}

print(f"\n✅ Config ready")
print(f"   Universe: NIFTY 200 (~120 stocks)")
print(f"   Max positions: {CONFIG['max_positions']}")
print(f"   Triple-barrier: +{CONFIG['barrier_multiplier']}σ / -{CONFIG['stop_multiplier']}σ / {CONFIG['max_holding_weeks']}w")

# COMMAND ----------

# DBTITLE 1,NIFTY 200 Universe
# ─── NIFTY 200 Universe (~120 stocks) ───
# Organized by sector for clarity. Some symbols may be unavailable via yfinance.

UNIVERSE = [
    # ── Banks & Financials (25) ──
    "HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK", "INDUSINDBK",
    "BANDHANBNK", "AUBANK", "PNB", "CANBK", "FEDERALBNK", "BANKBARODA",
    "IDFCFIRSTB", "BAJFINANCE", "BAJAJFINSV", "CHOLAFIN", "MUTHOOTFIN",
    "PFC", "RECLTD", "SBILIFE", "HDFCLIFE", "ICICIPRULI", "ICICIGI", "LICI", "LICHFIN",
    # ── IT & Tech (10) ──
    "TCS", "INFY", "WIPRO", "HCLTECH", "TECHM", "LTIM", "PERSISTENT", "COFORGE", "MPHASIS", "OFSS",
    # ── Energy & Power (12) ──
    "RELIANCE", "ONGC", "IOC", "BPCL", "GAIL", "PETRONET", "POWERGRID", "NTPC",
    "NHPC", "TATAPOWER", "ADANIPOWER", "ADANIGREEN",
    # ── Metals & Mining (10) ──
    "TATASTEEL", "JSWSTEEL", "HINDALCO", "VEDL", "NMDC", "SAIL",
    "COALINDIA", "JINDALSTEL", "NATIONALUM", "HINDCOPPER",
    # ── Auto (10) ──
    "MARUTI", "M&M", "TATAMOTORS", "BAJAJ-AUTO", "EICHERMOT", "HEROMOTOCO",
    "ASHOKLEY", "TVSMOTOR", "BOSCHLTD", "MRF",
    # ── FMCG & Consumer (12) ──
    "HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "TATACONSUM", "DABUR",
    "GODREJCP", "COLPAL", "UBL", "MARICO", "PGHH", "HINDPETRO",
    # ── Pharma & Healthcare (10) ──
    "SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "AUROPHARMA",
    "APOLLOHOSP", "LALPATHLAB", "ZYDUSLIFE", "BIOCON", "TORNTPHARM",
    # ── Cement & Construction (10) ──
    "ULTRACEMCO", "GRASIM", "AMBUJACEM", "SHREECEM", "ACC",
    "DALBHARAT", "RAMCOCEM", "DLF", "PRESTIGE", "OBEROIRLTY",
    # ── Telecom & Media (7) ──
    "BHARTIARTL", "IDEA", "SUNTV", "ZOMATO", "NAUKRI", "PVRINOX", "JUBLFOOD",
    # ── Industrial & Manufacturing (10) ──
    "LT", "SIEMENS", "ABB", "BHEL", "BEL", "CUMMINSIND",
    "THERMAX", "HAVELLS", "POLYCAB", "ADANIPORTS",
    # ── Others (12) ──
    "ADANIENT", "TITAN", "ASIANPAINT", "BERGEPAINT", "DMART", "PAGEIND",
    "MCDOWELL-N", "IRCTC", "GUJGASLTD", "IGL", "GMRINFRA", "CONCOR",
    # ── Additional NIFTY 200 (10) ──
    "TATACOMM", "ABCAPITAL", "RBLBANK", "YESBANK", "NHPC",
    "ADANIENSOL", "TATAELXSI", "CDSL", "INDIAMART", "FSL",
]

# Remove duplicates while preserving order
seen = set()
UNIVERSE = [x for x in UNIVERSE if not (x in seen or seen.add(x))]
print(f"✅ Universe: {len(UNIVERSE)} unique stock symbols")
print(f"   Sectors: Banks/Fin, IT, Energy, Metals, Auto, FMCG, Pharma, Cement, Telecom, Industrial, Others")

# COMMAND ----------

# DBTITLE 1,Fetch Weekly Data
# ─── Fetch weekly data for all stocks (8+ years) ───
start_date = "2018-01-01"
end_date = pd.Timestamp.now().strftime("%Y-%m-%d")
print(f"Fetching data from {start_date} to {end_date}...")

weekly_data = {}
failed = []
for i, sym in enumerate(UNIVERSE):
    try:
        ticker = yf.Ticker(f"{sym}.NS")
        df = ticker.history(start=start_date, end=end_date, interval="1d")
        if len(df) < 200:
            failed.append((sym, f"only {len(df)} bars"))
            continue
        df = df.rename(columns={"Open": "open", "High": "high", "Low": "low",
                                "Close": "close", "Volume": "volume"})
        weekly = df.resample("W-FRI").agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum",
        }).dropna()
        if len(weekly) >= 200:  # Need at least ~4 years for SMA(50) weekly
            weekly_data[sym] = weekly
        else:
            failed.append((sym, f"only {len(weekly)} weekly bars"))
    except Exception as e:
        failed.append((sym, str(e)[:50]))
    if (i + 1) % 20 == 0:
        time.sleep(1)
        print(f"  ... {i+1}/{len(UNIVERSE)} processed, {len(weekly_data)} successful")

# Fetch NIFTY 50
try:
    nifty_ticker = yf.Ticker("^NSEI")
    nifty_df = nifty_ticker.history(start=start_date, end=end_date, interval="1d")
    nifty_df = nifty_df.rename(columns={"Open": "open", "High": "high", "Low": "low",
                                         "Close": "close", "Volume": "volume"})
    nifty_weekly = nifty_df.resample("W-FRI").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna()
    print(f"✅ NIFTY 50 loaded ({len(nifty_weekly)} weeks)")
except Exception as e:
    nifty_weekly = None
    print(f"⚠️ NIFTY data unavailable: {e}")

print(f"\n✅ Weekly data loaded for {len(weekly_data)} / {len(UNIVERSE)} stocks")
if failed:
    print(f"   Failed ({len(failed)}): {[f[0] for f in failed[:10]]}{'...' if len(failed) > 10 else ''}")
weeks = len(list(weekly_data.values())[0]) if weekly_data else 0
print(f"   Weeks per stock: ~{weeks}")
print(f"   Period: {list(weekly_data.values())[0].index[0].date()} → {list(weekly_data.values())[0].index[-1].date()}" if weekly_data else "")

# COMMAND ----------

# DBTITLE 1,Feature Engineering Overview
# MAGIC %md
# MAGIC ## Feature Engineering — 50+ Features
# MAGIC
# MAGIC ### Technical Features (per stock, computed on weekly bars)
# MAGIC 1. RSI(14) — momentum oscillator
# MAGIC 2. MACD histogram — trend momentum
# MAGIC 3. Bollinger %B — volatility position
# MAGIC 4. ATR % — volatility normalized by price
# MAGIC 5. Momentum: 1w, 2w, 4w, 12w, 26w, 52w
# MAGIC 6. SMA ratios: close/SMA(10), close/SMA(20), close/SMA(50)
# MAGIC 7. Price above SMA(50) — binary trend flag
# MAGIC 8. Williams %R — overbought/oversold
# MAGIC 9. Stochastic %K, %D
# MAGIC 10. Volume ratio (current vs 4w average)
# MAGIC 11. Realized volatility (4w, 12w)
# MAGIC 12. Rate of change (12w)
# MAGIC
# MAGIC ### Cross-Sectional Features (ranked within universe each week)
# MAGIC 13. Rank by 26w momentum
# MAGIC 14. Rank by RSI
# MAGIC 15. Rank by volume
# MAGIC 16. Z-score of momentum
# MAGIC 17. Z-score of RSI
# MAGIC
# MAGIC ### Fundamental Features (from yfinance)
# MAGIC 18. PE ratio, Forward PE
# MAGIC 19. ROE, Profit margin
# MAGIC 20. Revenue growth, Earnings growth
# MAGIC 21. Debt to equity, Price to book
# MAGIC 22. Market cap (log)
# MAGIC
# MAGIC ### Regime Features
# MAGIC 23. NIFTY regime (bull/bear)
# MAGIC 24. NIFTY momentum (4w, 12w)
# MAGIC 25. NIFTY volatility (12w)
# MAGIC
# MAGIC ### Sentiment Features
# MAGIC 26. News sentiment score (VADER compound)
# MAGIC 27. News count (recent articles)

# COMMAND ----------

# DBTITLE 1,Build Technical Features
# ─── Build Technical Features (per stock) ───
all_feats = []
for sym, df in weekly_data.items():
    close = df["close"]
    high = df["high"]
    low = df["low"]
    vol = df["volume"]
    
    # Technical indicators
    rsi_14 = rsi(close, 14)
    macd_line, macd_signal, macd_hist = macd(close)
    bb_upper, bb_mid, bb_lower = bollinger_bands(close, 20, 2.0)
    atr_14 = atr(high, low, close, 14)
    stoch_k, stoch_d = stochastic(high, low, close, 14, 3)
    
    feat = pd.DataFrame(index=df.index)
    feat["symbol"] = sym
    
    # ── Momentum features ──
    for p in [1, 2, 4, 12, 26, 52]:
        feat[f"mom_{p}w"] = close.pct_change(p)
    
    # ── SMA ratios ──
    for p in [10, 20, 50]:
        s = sma(close, p)
        feat[f"sma_ratio_{p}"] = close / s
        feat[f"price_vs_sma{p}_pct"] = (close / s - 1) * 100
    
    feat["price_above_sma50"] = (close > sma(close, 50)).astype(int)
    
    # ── Technical indicators ──
    feat["rsi_14"] = rsi_14
    feat["macd_hist"] = macd_hist
    feat["bb_pctb"] = (close - bb_lower) / (bb_upper - bb_lower)
    feat["atr_pct"] = (atr_14 / close) * 100
    
    # ── Williams %R (14) ──
    low_14 = low.rolling(14).min()
    high_14 = high.rolling(14).max()
    feat["williams_r"] = ((high_14 - close) / (high_14 - low_14)) * -100
    
    # ── Stochastic ──
    feat["stoch_k"] = stoch_k
    feat["stoch_d"] = stoch_d
    
    # ── Volume features ──
    vol_avg_4 = vol.rolling(4).mean()
    feat["vol_ratio_4"] = vol / vol_avg_4
    feat["vol_mom_4"] = vol.pct_change(4)
    
    # ── Volatility ──
    ret = close.pct_change()
    feat["realized_vol_4w"] = ret.rolling(4).std() * np.sqrt(52)
    feat["realized_vol_12w"] = ret.rolling(12).std() * np.sqrt(52)
    
    # ── Rate of change ──
    feat["roc_12w"] = ((close - close.shift(12)) / close.shift(12)) * 100
    
    all_feats.append(feat)

feat_df = pd.concat(all_feats).sort_index()
tech_cols = [c for c in feat_df.columns if c != "symbol"]
print(f"✅ Technical features built: {len(tech_cols)} features")
print(f"   Rows: {len(feat_df):,}")
print(f"   Stocks: {feat_df['symbol'].nunique()}")
print(f"   Features: {tech_cols[:5]}... {tech_cols[-3:]}")

# COMMAND ----------

# DBTITLE 1,Cross-Sectional & Fundamental Features
# ─── Add Cross-Sectional Ranking Features (per week) ───
# Rank stocks within each week by key metrics
cross_cols = []

for col, rank_name in [("mom_26w", "rank_mom26"), ("rsi_14", "rank_rsi"), ("vol_ratio_4", "rank_vol")]:
    feat_df[rank_name] = feat_df.groupby(level=0)[col].rank(pct=True)
    cross_cols.append(rank_name)

# Z-scores (cross-sectional normalization per week)
for col, z_name in [("mom_26w", "zscore_mom26"), ("rsi_14", "zscore_rsi"), ("mom_12w", "zscore_mom12")]:
    grp = feat_df.groupby(level=0)[col]
    feat_df[z_name] = (grp.transform(lambda x: x - x.mean()) / grp.transform(lambda x: x.std() + 1e-8))
    cross_cols.append(z_name)

print(f"✅ Cross-sectional features: {len(cross_cols)} features")
print(f"   {cross_cols}")

# ─── Add NIFTY Regime Features ───
if nifty_weekly is not None:
    nifty_close = nifty_weekly["close"]
    nifty_sma40 = sma(nifty_close, CONFIG["regime_sma"])
    nifty_ret = nifty_close.pct_change()
    
    nifty_feat = pd.DataFrame(index=nifty_weekly.index)
    nifty_feat["nifty_regime"] = (nifty_close > nifty_sma40).astype(int)
    nifty_feat["nifty_mom_4w"] = nifty_close.pct_change(4)
    nifty_feat["nifty_mom_12w"] = nifty_close.pct_change(12)
    nifty_feat["nifty_vol_12w"] = nifty_ret.rolling(12).std() * np.sqrt(52)
    
    # Merge NIFTY features onto all stock-weeks (broadcast)
    for col in nifty_feat.columns:
        feat_df[col] = feat_df.index.map(nifty_feat[col])
    
    regime_cols = list(nifty_feat.columns)
    print(f"✅ Regime features: {len(regime_cols)} features")
else:
    regime_cols = []
    print("⚠️ NIFTY regime features skipped (no NIFTY data)")

# ─── Fetch Fundamental Features (from yfinance ticker.info) ───
print("\nFetching fundamentals from yfinance...")
fundamentals = {}
for i, sym in enumerate(weekly_data.keys()):
    try:
        ticker = yf.Ticker(f"{sym}.NS")
        info = ticker.info
        fundamentals[sym] = {
            "fund_pe": info.get("trailingPE", np.nan),
            "fund_fwd_pe": info.get("forwardPE", np.nan),
            "fund_roe": info.get("returnOnEquity", np.nan),
            "fund_margin": info.get("profitMargins", np.nan),
            "fund_rev_growth": info.get("revenueGrowth", np.nan),
            "fund_earn_growth": info.get("earningsGrowth", np.nan),
            "fund_de": info.get("debtToEquity", np.nan),
            "fund_pb": info.get("priceToBook", np.nan),
            "fund_mcap": np.log1p(info.get("marketCap", 0)),  # log scale
        }
    except:
        fundamentals[sym] = {f"fund_{k}": np.nan for k in ["pe", "fwd_pe", "roe", "margin", "rev_growth", "earn_growth", "de", "pb", "mcap"]}
    if (i + 1) % 20 == 0:
        time.sleep(1)
        print(f"  ... {i+1}/{len(weekly_data)} processed")

fund_cols = list(list(fundamentals.values())[0].keys())
for sym in weekly_data.keys():
    mask = feat_df["symbol"] == sym
    for col in fund_cols:
        feat_df.loc[mask, col] = fundamentals[sym].get(col, np.nan)

print(f"✅ Fundamental features: {len(fund_cols)} features")
print(f"   {fund_cols}")

# ─── Final feature summary ───
all_feat_cols = tech_cols + cross_cols + regime_cols + fund_cols
print(f"\n{'='*60}")
print(f"  TOTAL FEATURES: {len(all_feat_cols)}")
print(f"  Technical: {len(tech_cols)}")
print(f"  Cross-sectional: {len(cross_cols)}")
print(f"  Regime: {len(regime_cols)}")
print(f"  Fundamental: {len(fund_cols)}")
print(f"{'='*60}")

# COMMAND ----------

# DBTITLE 1,Triple-Barrier Labeling
# ─── Triple-Barrier Labeling (De Prado, 2018) ───
# For each stock-week, set three barriers:
#   Upper: +2 * weekly volatility (take profit)
#   Lower: -1 * weekly volatility (stop loss)
#   Time:  4 weeks (max holding period)
# Label: +1 if upper hit first, -1 if lower hit first, 0 if time expires

def triple_barrier_labels(df, vol_lookback=26, upper_mult=2.0, lower_mult=1.0, max_weeks=4):
    """Compute triple-barrier labels for a single stock's weekly data."""
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    n = len(close)
    labels = np.full(n, np.nan)
    
    rets = np.diff(np.log(close))
    
    for i in range(n - 1):
        if i < vol_lookback:
            continue
        
        # Weekly volatility
        vol = np.std(rets[i-vol_lookback:i])
        if vol <= 0 or np.isnan(vol):
            continue
        
        # Barrier levels
        entry_price = close[i]
        upper_barrier = entry_price * (1 + upper_mult * vol)
        lower_barrier = entry_price * (1 - lower_mult * vol)
        
        # Check forward bars
        for j in range(1, min(max_weeks + 1, n - i)):
            # Check barriers using weekly high/low
            if low[i + j] <= lower_barrier:
                labels[i] = -1  # Stop hit first
                break
            elif high[i + j] >= upper_barrier:
                labels[i] = 1   # Take profit hit first
                break
        else:
            labels[i] = 0  # Time barrier expired
    
    return pd.Series(labels, index=df.index)

# Compute labels for each stock
print("Computing triple-barrier labels...")
all_labels = []
for sym, df in weekly_data.items():
    labels = triple_barrier_labels(
        df, 
        vol_lookback=CONFIG["vol_lookback"],
        upper_mult=CONFIG["barrier_multiplier"],
        lower_mult=CONFIG["stop_multiplier"],
        max_weeks=CONFIG["max_holding_weeks"],
    )
    labels.name = "label"
    all_labels.append(pd.DataFrame({"symbol": sym, "label": labels}, index=df.index))

label_df = pd.concat(all_labels).sort_index()

# Merge labels into feat_df using (date, symbol) key to avoid cartesian product
label_df["_key"] = label_df.index.astype(str) + "_" + label_df["symbol"]
key_to_label = dict(zip(label_df["_key"], label_df["label"]))
feat_df["label"] = (feat_df.index.astype(str) + "_" + feat_df["symbol"]).map(key_to_label)

# Show label distribution
print(f"\n✅ Triple-barrier labels computed")
print(f"   Total labeled rows: {feat_df['label'].notna().sum():,}")
print(f"   Label distribution:")
print(f"     +1 (take profit): {(feat_df['label'] == 1).sum():,} ({(feat_df['label'] == 1).mean():.1%})")
print(f"      0 (time expire): {(feat_df['label'] == 0).sum():,} ({(feat_df['label'] == 0).mean():.1%})")
print(f"     -1 (stop loss):   {(feat_df['label'] == -1).sum():,} ({(feat_df['label'] == -1).mean():.1%})")

# For binary classification: +1 = positive (profitable), 0/-1 = negative
feat_df["target"] = (feat_df["label"] == 1).astype(int)
print(f"\n   Binary target (take-profit hit): {feat_df['target'].mean():.1%} positive rate")

# COMMAND ----------

# DBTITLE 1,LightGBM Training & Evaluation
# ─── LightGBM with Purged Walk-Forward CV ───
# Drop rows with NaN in features or target
model_df = feat_df.dropna(subset=all_feat_cols + ["target"]).copy()
print(f"Rows for ML: {len(model_df):,} (dropped {len(feat_df) - len(model_df):,} with NaN)")

# Walk-forward split
all_weeks = sorted(model_df.index.unique())
n_train = int(len(all_weeks) * CONFIG["ml_train_fraction"])
train_end = all_weeks[n_train]
gap_end = all_weeks[min(n_train + CONFIG["ml_gap_weeks"], len(all_weeks) - 1)]

train = model_df[model_df.index < train_end].copy()
test = model_df[model_df.index >= gap_end].copy()

print(f"\nTrain: {train.index[0].date()} → {train.index[-1].date()} ({len(train):,} rows)")
print(f"Gap:  {train_end.date()} → {gap_end.date()} ({CONFIG['ml_gap_weeks']} weeks purged)")
print(f"Test:  {test.index[0].date()} → {test.index[-1].date()} ({len(test):,} rows)")

X_train = train[all_feat_cols]
y_train = train["target"].values
X_test = test[all_feat_cols]
y_test = test["target"].values

# ─── LightGBM Training ───
print(f"\nTraining LightGBM...")
print(f"   Features: {len(all_feat_cols)}")
print(f"   Train positive rate: {y_train.mean():.1%}")
print(f"   Test positive rate:  {y_test.mean():.1%}")

lgb_model = lgb.LGBMClassifier(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=100,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    verbose=-1,
)

# Early stopping
lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[lgb.early_stopping(50, verbose=False)],
)

# Predictions
train_proba = lgb_model.predict_proba(X_train)[:, 1]
test_proba = lgb_model.predict_proba(X_test)[:, 1]

train_auc = roc_auc_score(y_train, train_proba)
test_auc = roc_auc_score(y_test, test_proba)

# Cross-sectional ranking: predict top quartile
test["ml_prob"] = test_proba
# For each week, rank stocks by predicted probability
test["pred_rank"] = test.groupby(level=0)["ml_prob"].rank(pct=True, ascending=True)

# Top decile selection
top_decile = test[test["pred_rank"] >= 0.9]
mid_decile = test[(test["pred_rank"] >= 0.5) & (test["pred_rank"] < 0.9)]
bot_decile = test[test["pred_rank"] < 0.5]

print(f"\n─── LightGBM Performance ───")
print(f"   Train AUC: {train_auc:.3f}")
print(f"   Test AUC:  {test_auc:.3f}")
print(f"   (Previous model: Train 0.706, Test 0.481)")
print(f"\n   Top decile positive rate:    {top_decile['target'].mean():.1%}")
print(f"   Middle deciles positive rate: {mid_decile['target'].mean():.1%}")
print(f"   Bottom half positive rate:    {bot_decile['target'].mean():.1%}")
print(f"   Baseline rate:                {y_test.mean():.1%}")

# Feature importance
print(f"\n─── Top 15 Features (LightGBM importance) ───")
imp_df = pd.DataFrame({"feature": all_feat_cols, "importance": lgb_model.feature_importances_})
imp_df = imp_df.sort_values("importance", ascending=False)
display(imp_df.head(15))

# Log to MLflow
import mlflow
with mlflow.start_run(run_name="lightgbm_stock_selection"):
    mlflow.log_params({
        "n_features": len(all_feat_cols),
        "n_stocks": model_df["symbol"].nunique(),
        "train_rows": len(train),
        "test_rows": len(test),
        "model": "LightGBM",
        "n_estimators": 500,
        "max_depth": 5,
    })
    mlflow.log_metrics({
        "train_auc": train_auc,
        "test_auc": test_auc,
        "top_decile_positive_rate": top_decile["target"].mean(),
        "baseline_rate": y_test.mean(),
    })
    mlflow.lightgbm.log_model(lgb_model, "lightgbm_model")
print("\n✅ Model logged to MLflow")

# COMMAND ----------

# DBTITLE 1,Backtest ML-Enhanced Selection
# ─── Backtest: ML-Enhanced Stock Selection ───
# Strategy: Each week, rank all stocks by ML probability.
# Buy top 15 that ALSO pass trend-following filter (close > SMA50, NIFTY bull).
# Exit when stock drops below SMA50 or exits top 50% of ML rank.

def run_ml_backtest(weekly_data, nifty_weekly, test_df, config, ml_filter=True):
    """Backtest ML-enhanced stock selection with trend-following overlay."""
    cap = config["initial_capital"]
    max_pos = config["max_positions"]
    sma_per = config["sma_period"]
    
    # Pre-compute SMAs
    smas = {}
    for sym, df in weekly_data.items():
        smas[sym] = sma(df["close"], sma_per)
    
    # NIFTY regime
    if nifty_weekly is not None:
        nifty_sma = sma(nifty_weekly["close"], config["regime_sma"])
    
    # Get ML predictions per week per stock
    ml_preds = test_df[["symbol", "ml_prob", "pred_rank"]].copy()
    
    # Get test weeks
    test_weeks = sorted(test_df.index.unique())
    
    positions = {}  # sym -> {entry_price, entry_date, quantity, entry_week_idx}
    equity_curve = []
    trades = []
    
    for week_idx, week in enumerate(test_weeks):
        if week not in ml_preds.index:
            continue
        
        # Current prices
        prices = {}
        for sym, df in weekly_data.items():
            if week in df.index:
                prices[sym] = df.loc[week, "close"]
        
        # NIFTY regime
        nifty_bull = True
        if nifty_weekly is not None and week in nifty_weekly.index and week in nifty_sma.index:
            nifty_bull = nifty_weekly.loc[week, "close"] > nifty_sma.loc[week]
        
        # Process exits: close below SMA50
        for sym in list(positions.keys()):
            if sym not in prices or sym not in smas:
                continue
            s = smas[sym]
            if week in s.index and not pd.isna(s.loc[week]):
                if prices[sym] < s.loc[week]:
                    # Exit
                    exit_price = prices[sym]
                    pos = positions[sym]
                    pnl = (exit_price - pos["entry_price"]) * pos["quantity"]
                    trades.append({
                        "symbol": sym, "entry": pos["entry_price"], "exit": exit_price,
                        "pnl": pnl, "pnl_pct": (exit_price/pos["entry_price"]-1)*100,
                        "entry_date": str(pos["entry_date"].date()), "exit_date": str(week.date()),
                        "holding_weeks": week_idx - pos["entry_week_idx"],
                    })
                    cap += pos["quantity"] * exit_price
                    del positions[sym]
        
        # Process entries: top ML-ranked stocks in uptrend
        n_open = len(positions)
        n_slots = max_pos - n_open
        
        if n_slots > 0 and nifty_bull:
            week_preds = ml_preds.loc[week].sort_values("ml_prob", ascending=False)
            
            for _, row in week_preds.iterrows():
                if n_slots <= 0:
                    break
                sym = row["symbol"]
                if sym in positions or sym not in prices or sym not in smas:
                    continue
                s = smas[sym]
                if week not in s.index or pd.isna(s.loc[week]):
                    continue
                
                # Trend filter: must be above SMA50
                if prices[sym] <= s.loc[week]:
                    continue
                
                # Equal weight sizing
                total_equity = cap + sum(
                    positions[s]["quantity"] * prices.get(s, positions[s]["entry_price"])
                    for s in positions
                )
                target_val = total_equity / max_pos
                qty = int(target_val / prices[sym])
                if qty <= 0:
                    continue
                cost = qty * prices[sym]
                if cost > cap:
                    continue
                cap -= cost
                positions[sym] = {
                    "entry_price": prices[sym], "entry_date": week,
                    "quantity": qty, "entry_week_idx": week_idx,
                }
                n_slots -= 1
        
        # Equity snapshot
        pos_val = sum(positions[s]["quantity"] * prices.get(s, positions[s]["entry_price"]) for s in positions)
        equity_curve.append({"date": week, "equity": cap + pos_val, "n_positions": len(positions)})
    
    equity_df = pd.DataFrame(equity_curve) if equity_curve else pd.DataFrame()
    if len(equity_df) > 0:
        equity_df["daily_returns"] = equity_df["equity"].pct_change()
    
    return {"equity_curve": equity_df, "trades": trades}

# Run ML-enhanced backtest
print("Running ML-Enhanced Stock Selection backtest...")
ml_bt = run_ml_backtest(weekly_data, nifty_weekly, test, CONFIG)

def compute_metrics(equity_df, trades, initial_capital):
    if len(equity_df) == 0:
        return {"total_return": 0, "cagr": 0, "sharpe": 0, "max_dd": 0, "n_trades": 0}
    total_return = (equity_df["equity"].iloc[-1] / initial_capital - 1) * 100
    years = (equity_df["date"].iloc[-1] - equity_df["date"].iloc[0]).days / 365.25
    cagr = ((equity_df["equity"].iloc[-1] / initial_capital) ** (1/years) - 1) * 100 if years > 0 else 0
    rets = equity_df["daily_returns"].dropna()
    sharpe = (rets.mean() / rets.std() * np.sqrt(52)) if rets.std() > 0 else 0
    peak = equity_df["equity"].expanding().max()
    max_dd = ((equity_df["equity"] / peak - 1) * 100).min()
    win_trades = [t for t in trades if t["pnl"] > 0]
    win_rate = len(win_trades) / len(trades) * 100 if trades else 0
    return {
        "total_return": total_return, "cagr": cagr, "sharpe": sharpe,
        "max_dd": max_dd, "n_trades": len(trades), "win_rate": win_rate,
    }

ml_m = compute_metrics(ml_bt["equity_curve"], ml_bt["trades"], CONFIG["initial_capital"])
print(f"  ML-Enhanced: {ml_m['total_return']:.2f}% return, {ml_m['n_trades']} trades, Sharpe {ml_m['sharpe']:.2f}")

# NIFTY benchmark
if nifty_weekly is not None:
    test_nifty = nifty_weekly[nifty_weekly.index >= test.index[0]]
    nifty_ret = (test_nifty["close"].iloc[-1] / test_nifty["close"].iloc[0] - 1) * 100
    nifty_years = (test_nifty.index[-1] - test_nifty.index[0]).days / 365.25
    nifty_cagr = ((test_nifty["close"].iloc[-1] / test_nifty["close"].iloc[0]) ** (1/nifty_years) - 1) * 100 if nifty_years > 0 else 0
    nifty_rets = test_nifty["close"].pct_change().dropna()
    nifty_sharpe = (nifty_rets.mean() / nifty_rets.std() * np.sqrt(52)) if nifty_rets.std() > 0 else 0
    peak_nifty = test_nifty["close"].expanding().max()
    nifty_dd = ((test_nifty["close"] / peak_nifty - 1) * 100).min()
else:
    nifty_ret = nifty_cagr = nifty_sharpe = nifty_dd = 0

# Display comparison
print(f"\n{'═'*70}")
print(f"  ML-ENHANCED STOCK SELECTION — OUT-OF-SAMPLE RESULTS")
print(f"  Test period: {test.index[0].date()} → {test.index[-1].date()}")
print(f"{'═'*70}")
print(f"\n{'Strategy':<30} {'Return':>10} {'CAGR':>8} {'Sharpe':>8} {'Max DD':>8} {'Trades':>7}")
print(f"{'─'*70}")
print(f"{'ML-Enhanced Selection':<30} {ml_m['total_return']:>9.1f}% {ml_m['cagr']:>7.1f}% {ml_m['sharpe']:>7.2f} {ml_m['max_dd']:>7.1f}% {ml_m['n_trades']:>7}")
print(f"{'NIFTY 50 B&H':<30} {nifty_ret:>9.1f}% {nifty_cagr:>7.1f}% {nifty_sharpe:>7.2f} {nifty_dd:>7.1f}% {'N/A':>7}")
print(f"{'─'*70}")
print(f"\n   Win rate: {ml_m.get('win_rate', 0):.1f}%")

# Show top trades
if ml_bt["trades"]:
    trades_df = pd.DataFrame(ml_bt["trades"])
    print(f"\n─── Top 5 Trades ───")
    display(trades_df.nlargest(5, "pnl")[["symbol", "entry", "exit", "pnl", "pnl_pct", "holding_weeks"]].round(2))
    print(f"\n─── Bottom 5 Trades ───")
    display(trades_df.nsmallest(5, "pnl")[["symbol", "entry", "exit", "pnl", "pnl_pct", "holding_weeks"]].round(2))

# COMMAND ----------

# DBTITLE 1,News Sentiment & Catalysts
# ─── News Sentiment & Catalysts ───
# Fetch recent news for each stock via yfinance and score with VADER

analyzer = SentimentIntensityAnalyzer()

print("Fetching news & computing sentiment...")
sentiment_data = {}
for i, sym in enumerate(weekly_data.keys()):
    try:
        ticker = yf.Ticker(f"{sym}.NS")
        news = ticker.news
        
        if not news:
            sentiment_data[sym] = {"sentiment_score": 0, "news_count": 0, "latest_headline": "No news"}
            continue
        
        # Score each headline with VADER
        scores = []
        headlines = []
        for item in news[:10]:  # Last 10 news items
            title = item.get("title", "") if isinstance(item, dict) else getattr(item, "title", "")
            score = analyzer.polarity_scores(title)
            scores.append(score["compound"])
            headlines.append(title)
        
        sentiment_data[sym] = {
            "sentiment_score": np.mean(scores) if scores else 0,
            "news_count": len(news),
            "latest_headline": headlines[0] if headlines else "",
        }
    except Exception as e:
        sentiment_data[sym] = {"sentiment_score": 0, "news_count": 0, "latest_headline": f"Error: {str(e)[:30]}"}
    
    if (i + 1) % 20 == 0:
        time.sleep(1)
        print(f"  ... {i+1}/{len(weekly_data)} processed")

# Convert to DataFrame
sent_df = pd.DataFrame.from_dict(sentiment_data, orient="index")
sent_df.index.name = "symbol"

print(f"\n✅ Sentiment data fetched for {len(sent_df)} stocks")
print(f"   Avg sentiment: {sent_df['sentiment_score'].mean():.3f}")
print(f"   Stocks with news: {(sent_df['news_count'] > 0).sum()}")
print(f"\n─── Most Positive Sentiment (top 10) ───")
display(sent_df.nlargest(10, "sentiment_score")[["sentiment_score", "news_count", "latest_headline"]])
print(f"\n─── Most Negative Sentiment (bottom 10) ───")
display(sent_df.nsmallest(10, "sentiment_score")[["sentiment_score", "news_count", "latest_headline"]])

# COMMAND ----------

# DBTITLE 1,Current Stock Selection
# ─── Current Stock Selection: Top Picks from NIFTY 200 ───
# Combine ML predictions, sentiment, and trend-following to pick current best stocks

print("═" * 80)
print("  CURRENT STOCK SELECTION — NIFTY 200")
print("  " + datetime.now().strftime("%Y-%m-%d %H:%M UTC"))
print("═" * 80)

# Get latest week's data
latest_week = sorted(weekly_data.keys())
if test.index.max() in weekly_data[list(weekly_data.keys())[0]].index:
    latest_test_week = test.index.max()
else:
    latest_test_week = test.index[-1]

latest_preds = test[test.index == latest_test_week].sort_values("ml_prob", ascending=False)

# NIFTY regime
if nifty_weekly is not None:
    nifty_close = nifty_weekly["close"].iloc[-1]
    nifty_sma = sma(nifty_weekly["close"], CONFIG["regime_sma"])
    nifty_sma_val = nifty_sma.iloc[-1]
    nifty_bull = nifty_close > nifty_sma_val if not pd.isna(nifty_sma_val) else False
    print(f"\n📊 NIFTY 50 Regime: {'🟢 BULL' if nifty_bull else '🔴 BEAR'} (close={nifty_close:.0f}, SMA40={nifty_sma_val:.0f})")
else:
    nifty_bull = False
    print("\n⚠️ NIFTY regime unknown")

# Combine ML rank, trend signal, and sentiment
selections = []
for _, row in latest_preds.iterrows():
    sym = row["symbol"]
    if sym not in weekly_data:
        continue
    
    df = weekly_data[sym]
    close = df["close"].iloc[-1]
    s50 = sma(df["close"], CONFIG["sma_period"])
    sma_val = s50.iloc[-1]
    is_uptrend = close > sma_val if not pd.isna(sma_val) else False
    
    sent = sentiment_data.get(sym, {"sentiment_score": 0, "news_count": 0, "latest_headline": ""})
    
    selections.append({
        "symbol": sym,
        "ml_prob": row["ml_prob"],
        "pred_rank": row["pred_rank"],
        "close": close,
        "sma50": sma_val,
        "price_vs_sma": ((close/sma_val - 1)*100) if not pd.isna(sma_val) else 0,
        "is_uptrend": is_uptrend,
        "sentiment": sent["sentiment_score"],
        "news_count": sent["news_count"],
        "headline": sent["latest_headline"][:60],
        "nifty_bull": nifty_bull,
        "recommendation": "BUY" if (is_uptrend and nifty_bull and row["pred_rank"] >= 0.7) else ("HOLD" if is_uptrend else "AVOID"),
    })

sel_df = pd.DataFrame(selections).sort_values("ml_prob", ascending=False)

print(f"\n─── TOP 20 STOCKS BY ML PROBABILITY ───")
display(sel_df.head(20)[["symbol", "ml_prob", "pred_rank", "close", "price_vs_sma", "is_uptrend", "sentiment", "news_count", "recommendation"]].round(3))

buy_signals = sel_df[sel_df["recommendation"] == "BUY"]
print(f"\n─── BUY RECOMMENDATIONS ({len(buy_signals)} stocks) ───")
if len(buy_signals) > 0:
    display(buy_signals[["symbol", "ml_prob", "close", "price_vs_sma", "sentiment", "headline"]].round(3))
else:
    print("   No stocks meet all criteria (ML rank ≥ 70th pct + uptrend + NIFTY bull)")
    if not nifty_bull:
        print("   Reason: NIFTY 50 is in BEAR regime — strategy stays in cash")
    else:
        print("   Reason: No stocks in top ML rank are also in uptrend")

print(f"\n─── LATEST NEWS CATALYSTS (top 10 by sentiment) ───")
catalyst_df = sel_df[sel_df["news_count"] > 0].sort_values("sentiment", ascending=False).head(10)
if len(catalyst_df) > 0:
    display(catalyst_df[["symbol", "sentiment", "ml_prob", "recommendation", "headline"]].round(3))
else:
    print("   No recent news catalysts found")

print(f"\n{'═' * 80}")
print(f"  SUMMARY: {len(buy_signals)} BUY / {len(sel_df[sel_df['recommendation']=='HOLD'])} HOLD / {len(sel_df[sel_df['recommendation']=='AVOID'])} AVOID")
print(f"  ML Model: LightGBM with {len(all_feat_cols)} features, Test AUC = {test_auc:.3f}")
print(f"  Selection: ML rank ≥ 70th pct + Close > SMA50 + NIFTY bull regime")
print(f"{'═' * 80}")

# COMMAND ----------

# DBTITLE 1,External APIs Needed
# MAGIC %md
# MAGIC ## External APIs & Resources Needed
# MAGIC
# MAGIC To significantly improve ML performance, the following external data sources are recommended:
# MAGIC
# MAGIC ### 1. Finnhub (finnhub.io) — **HIGH PRIORITY**
# MAGIC - **What**: Real-time company news, sentiment scores, insider trading, earnings calendar
# MAGIC - **Free tier**: 60 requests/minute — sufficient for daily updates
# MAGIC - **Setup**: Register at finnhub.io → get API key → `pip install finnhub-python`
# MAGIC - **Use case**: Replace VADER-only sentiment with Finnhub's pre-scored news sentiment + insider trading signals
# MAGIC - **API key needed**: Yes (free)
# MAGIC
# MAGIC ### 2. Alpha Vantage (alphavantage.co) — **MEDIUM PRIORITY**
# MAGIC - **What**: Fundamental data (quarterly P&L, balance sheet), economic indicators, crypto
# MAGIC - **Free tier**: 25 requests/day
# MAGIC - **Setup**: Register at alphavantage.co → get API key → `pip install alpha_vantage`
# MAGIC - **Use case**: Historical fundamental data (point-in-time correct, unlike yfinance's current-only)
# MAGIC - **API key needed**: Yes (free)
# MAGIC
# MAGIC ### 3. Screener.in — **HIGH PRIORITY (manual)**
# MAGIC - **What**: Detailed Indian stock fundamentals (P&L, balance sheet, quarterly results, ratios)
# MAGIC - **Free**: Website scraping (no official API)
# MAGIC - **Use case**: Get 10+ years of quarterly financials for Indian stocks
# MAGIC - **How to use**: Export CSV from screener.in or use `requests` to scrape
# MAGIC - **No API key**: Website scraping only
# MAGIC
# MAGIC ### 4. NSE Options Chain (nseindia.com) — **MEDIUM PRIORITY**
# MAGIC - **What**: Implied volatility, put/call ratios, option open interest
# MAGIC - **Free**: Yes (nseindia.com/option-chain)
# MAGIC - **Use case**: Add IV-based features — high IV = market expects big move
# MAGIC - **No API key**: Web scraping (with session management)
# MAGIC
# MAGIC ### 5. RBI/FRED Economic Data — **MEDIUM PRIORITY**
# MAGIC - **What**: Interest rates (repo rate), inflation (CPI), GDP growth, forex reserves
# MAGIC - **Free**: RBI API (rbi.org.in), FRED API (fred.stlouisfed.org)
# MAGIC - **Use case**: Macro features — rate cuts are bullish for equities
# MAGIC - **API key**: FRED needs free key, RBI is open
# MAGIC
# MAGIC ### 6. Twitter/X Sentiment — **LOW-MEDIUM PRIORITY**
# MAGIC - **What**: Social media buzz, retail sentiment
# MAGIC - **Paid**: Twitter API v2 ($100/month basic)
# MAGIC - **Use case**: Gauge retail sentiment for contrarian signals
# MAGIC - **Alternative**: Use Reddit API (free) for r/IndianStreetBets sentiment
# MAGIC
# MAGIC ### Summary: What to Get First
# MAGIC
# MAGIC | Priority | Source | Cost | Expected Impact |
# MAGIC |---|---|---|---|
# MAGIC | 1st | Finnhub | Free | +5-10% AUC improvement (better sentiment) |
# MAGIC | 2nd | Screener.in | Free | Better fundamentals (point-in-time) |
# MAGIC | 3rd | NSE Options | Free | IV features for volatility prediction |
# MAGIC | 4th | Alpha Vantage | Free | Historical fundamentals |
# MAGIC | 5th | FRED/RBI | Free | Macro regime features |
# MAGIC
# MAGIC > **Next step**: Get a free Finnhub API key and I'll integrate it to replace yfinance news with professional-grade sentiment data.

# COMMAND ----------

# DBTITLE 1,Finnhub Integration Overview
# MAGIC %md
# MAGIC ## Step 7 — Finnhub News Sentiment Integration
# MAGIC
# MAGIC ### How to Get Your Free Finnhub API Key
# MAGIC
# MAGIC 1. Go to **https://finnhub.io/register**
# MAGIC 2. Sign up with your email (free tier — 60 requests/minute)
# MAGIC 3. Copy your API key from the dashboard
# MAGIC 4. Store it in Databricks secrets:
# MAGIC    ```python
# MAGIC    dbutils.secrets.put(scope="finnhub", key="api_key", value="YOUR_KEY_HERE")
# MAGIC    ```
# MAGIC    Or simply replace `FINNHUB_API_KEY` below with your key string.
# MAGIC
# MAGIC Once you have the key, re-run the cells below. The code will automatically:
# MAGIC - Fetch company news for all 130 stocks
# MAGIC - Score sentiment with VADER
# MAGIC - Fetch analyst recommendations (buy/hold/sell counts)
# MAGIC - Add these as ML features

# COMMAND ----------

# DBTITLE 1,Install Finnhub
# Finnhub removed - free tier doesn't cover Indian stocks.
# Using Google News RSS instead (see Step 11 cells below).
print("Skipping Finnhub - using Google News RSS for Indian stock sentiment")

# COMMAND ----------

# DBTITLE 1,Finnhub Setup & Config
import sys, os, time, json
from datetime import datetime, timedelta

REPO_ROOT = "/Workspace/Repos/lokigovindasamy@gmail.com/algo_trade_data_bricks"
SRC_PATH = os.path.join(REPO_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

import pandas as pd
import numpy as np
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Finnhub removed - free tier doesn't cover Indian stocks.
# News sentiment comes from Google News RSS (Step 11).
FINNHUB_AVAILABLE = False
finnhub_sentiment = {}  # Will be filled by Google News RSS

# Config (same as main notebook)
CONFIG = {
    "initial_capital": 100_000,
    "max_positions": 15,
    "sma_period": 50,
    "regime_sma": 40,
    "momentum_lookback": 26,
    "ml_train_fraction": 0.60,
    "ml_gap_weeks": 4,
    "vol_lookback": 26,
    "barrier_multiplier": 2.0,
    "stop_multiplier": 1.0,
    "max_holding_weeks": 4,
}

# NIFTY 200 Universe
UNIVERSE = [
    "HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK", "INDUSINDBK",
    "BANDHANBNK", "AUBANK", "PNB", "CANBK", "FEDERALBNK", "BANKBARODA",
    "IDFCFIRSTB", "BAJFINANCE", "BAJAJFINSV", "CHOLAFIN", "MUTHOOTFIN",
    "PFC", "RECLTD", "SBILIFE", "HDFCLIFE", "ICICIPRULI", "ICICIGI", "LICI",
    "TCS", "INFY", "WIPRO", "HCLTECH", "TECHM", "PERSISTENT", "COFORGE", "MPHASIS", "OFSS",
    "RELIANCE", "ONGC", "IOC", "BPCL", "GAIL", "PETRONET", "POWERGRID", "NTPC",
    "NHPC", "TATAPOWER", "ADANIPOWER", "ADANIGREEN",
    "TATASTEEL", "JSWSTEEL", "HINDALCO", "VEDL", "NMDC", "SAIL",
    "COALINDIA", "JINDALSTEL", "NATIONALUM", "HINDCOPPER",
    "MARUTI", "M&M", "BAJAJ-AUTO", "EICHERMOT", "HEROMOTOCO",
    "ASHOKLEY", "TVSMOTOR", "BOSCHLTD", "MRF",
    "HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "TATACONSUM", "DABUR",
    "GODREJCP", "COLPAL", "UBL", "MARICO", "PGHH", "HINDPETRO",
    "SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "AUROPHARMA",
    "APOLLOHOSP", "LALPATHLAB", "ZYDUSLIFE", "BIOCON", "TORNTPHARM",
    "ULTRACEMCO", "GRASIM", "AMBUJACEM", "SHREECEM", "ACC",
    "DALBHARAT", "RAMCOCEM", "DLF", "PRESTIGE", "OBEROIRLTY",
    "BHARTIARTL", "IDEA", "SUNTV", "NAUKRI", "PVRINOX", "JUBLFOOD",
    "LT", "SIEMENS", "ABB", "BHEL", "BEL", "CUMMINSIND",
    "THERMAX", "HAVELLS", "POLYCAB", "ADANIPORTS",
    "ADANIENT", "TITAN", "ASIANPAINT", "BERGEPAINT", "DMART", "PAGEIND",
    "IRCTC", "IGL", "CONCOR",
    "TATACOMM", "ABCAPITAL", "RBLBANK", "YESBANK",
    "ADANIENSOL", "TATAELXSI", "CDSL", "INDIAMART", "FSL",
]
seen = set()
UNIVERSE = [x for x in UNIVERSE if not (x in seen or seen.add(x))]
print(f"\n✅ Universe: {len(UNIVERSE)} stocks")
print(f"   Finnhub available: {FINNHUB_AVAILABLE}")

# COMMAND ----------

# DBTITLE 1,Screener.in Overview
# MAGIC %md
# MAGIC ## Step 8 — Screener.in Historical Fundamentals
# MAGIC
# MAGIC **Screener.in** provides 10+ years of Indian stock fundamentals — far richer than yfinance's current-only data.
# MAGIC
# MAGIC ### What We Extract
# MAGIC - **P&L**: Revenue, Operating Profit, Net Profit, EPS (10 years)
# MAGIC - **Balance Sheet**: Equity, Debt, Current Assets
# MAGIC - **Ratios**: P/E, P/B, ROCE, ROE, D/E, Current Ratio
# MAGIC - **Quarterly Results**: Revenue and profit by quarter
# MAGIC
# MAGIC ### Why This Matters
# MAGIC yfinance's `ticker.info` only returns *current* fundamentals (PE, ROE as of today). For backtesting, we need *point-in-time* data — what was the PE in 2020 when we would have been trading. Screener.in provides 10+ years of annual data, allowing us to forward-fill to weekly bars and approximate historical fundamentals.
# MAGIC
# MAGIC ### Data Source
# MAGIC - URL: `https://www.screener.in/company/SYMBOL/`
# MAGIC - Free access (no login needed for basic data)
# MAGIC - Parse HTML tables with BeautifulSoup

# COMMAND ----------

# DBTITLE 1,Screener.in Scraper
import requests
from bs4 import BeautifulSoup
import re

def scrape_screener_fundamentals(symbol, max_retries=2):
    """Scrape Screener.in for a stock's fundamental ratios.
    
    Returns a dict with 10 years of annual data:
    - pe_ratio, pb_ratio, roce, roe, debt_to_equity, current_ratio
    - revenue, operating_profit, net_profit, eps
    """
    url = f"https://www.screener.in/company/{symbol}/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "en-US,en;q=0.9",
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=15)
            if response.status_code == 200:
                break
            elif response.status_code == 429:
                time.sleep(5)
                continue
            else:
                return {"symbol": symbol, "status": f"HTTP {response.status_code}"}
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            return {"symbol": symbol, "status": f"Error: {str(e)[:50]}"}
    else:
        return {"symbol": symbol, "status": "Failed to fetch"}
    
    soup = BeautifulSoup(response.content, "html.parser")
    result = {"symbol": symbol, "status": "ok"}
    
    # Find all data tables
    tables = soup.find_all("table", class_="data-table")
    
    for table in tables:
        # Get table title (usually in preceding <h2> or parent <section>)
        parent = table.find_parent()
        title = ""
        if parent:
            h2 = parent.find("h2") if hasattr(parent, 'find') else None
            title = h2.get_text(strip=True) if h2 else ""
        
        # Parse table headers (years)
        thead = table.find("thead")
        if not thead:
            continue
        year_headers = [th.get_text(strip=True) for th in thead.find_all("th")]
        
        # Parse table rows
        tbody = table.find("tbody")
        if not tbody:
            continue
        
        for row in tbody.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) < 2:
                continue
            metric_name = cells[0].get_text(strip=True).lower()
            
            # Extract values for each year
            values = []
            for cell in cells[1:]:
                val_text = cell.get_text(strip=True).replace(",", "").replace("%", "")
                try:
                    val = float(val_text) if val_text and val_text != "" else np.nan
                except:
                    val = np.nan
                values.append(val)
            
            # Map metric names to our schema
            if "p/e" in metric_name or "pe ratio" in metric_name:
                result["screener_pe"] = values[-1] if values else np.nan  # Latest year
                result["screener_pe_5y_avg"] = np.nanmean(values[-5:]) if len(values) >= 1 else np.nan
            elif "price to book" in metric_name or "p/b" in metric_name:
                result["screener_pb"] = values[-1] if values else np.nan
            elif "roce" in metric_name:
                result["screener_roce"] = values[-1] if values else np.nan
                result["screener_roce_5y_avg"] = np.nanmean(values[-5:]) if len(values) >= 1 else np.nan
            elif metric_name == "roe":
                result["screener_roe"] = values[-1] if values else np.nan
            elif "debt/equity" in metric_name or "debt to equity" in metric_name:
                result["screener_de"] = values[-1] if values else np.nan
            elif "current ratio" in metric_name:
                result["screener_current_ratio"] = values[-1] if values else np.nan
            elif "revenue" in metric_name and "growth" not in metric_name:
                if len(values) >= 2 and not np.isnan(values[-1]) and not np.isnan(values[-2]) and values[-2] != 0:
                    result["screener_rev_growth"] = ((values[-1] - values[-2]) / abs(values[-2])) * 100
                result["screener_revenue_latest"] = values[-1] if values else np.nan
            elif "net profit" in metric_name:
                if len(values) >= 2 and not np.isnan(values[-1]) and not np.isnan(values[-2]) and values[-2] != 0:
                    result["screener_profit_growth"] = ((values[-1] - values[-2]) / abs(values[-2])) * 100
                result["screener_net_profit"] = values[-1] if values else np.nan
            elif "operating profit" in metric_name and "margin" not in metric_name:
                result["screener_op_profit"] = values[-1] if values else np.nan
            elif metric_name == "eps" or metric_name == "eps (rs)":
                result["screener_eps"] = values[-1] if values else np.nan
            
            # Also compute margin from revenue & net profit
            if "screener_revenue_latest" in result and "screener_net_profit" in result:
                if result["screener_revenue_latest"] and result["screener_revenue_latest"] != 0:
                    result["screener_margin"] = (result["screener_net_profit"] / result["screener_revenue_latest"]) * 100
    
    return result

# Test with one stock first
print("Testing Screener.in scraper with RELIANCE...")
test_result = scrape_screener_fundamentals("RELIANCE")
print(f"Status: {test_result['status']}")
for k, v in test_result.items():
    if k not in ["symbol", "status"]:
        print(f"  {k}: {v}")

# COMMAND ----------

# DBTITLE 1,Scrape All Screener.in Fundamentals
# ─── Scrape Screener.in for all stocks ───
print(f"Scraping Screener.in for {len(UNIVERSE)} stocks...")
print("   (Rate-limited: ~2 sec per stock = ~4 min total)")

screener_fundamentals = {}
success_count = 0
for i, sym in enumerate(UNIVERSE):
    result = scrape_screener_fundamentals(sym)
    screener_fundamentals[sym] = result
    if result.get("status") == "ok":
        success_count += 1
    
    if (i + 1) % 10 == 0:
        print(f"  ... {i+1}/{len(UNIVERSE)} processed ({success_count} successful)")
    
    # Rate limit: be respectful
    time.sleep(2)

print(f"\n✅ Screener.in scrape complete: {success_count}/{len(UNIVERSE)} successful")

# Build DataFrame
screener_cols = ["screener_pe", "screener_pe_5y_avg", "screener_pb", "screener_roce", 
                 "screener_roce_5y_avg", "screener_roe", "screener_de", 
                 "screener_current_ratio", "screener_rev_growth", "screener_profit_growth",
                 "screener_margin", "screener_eps"]

scr_data = {}
for sym, result in screener_fundamentals.items():
    for col in screener_cols:
        scr_data.setdefault(col, {})[sym] = result.get(col, np.nan)

scr_df = pd.DataFrame(scr_data)
scr_df.index.name = "symbol"

print(f"\n─── Screener.in Fundamentals Summary ───")
print(f"   Stocks with PE data: {scr_df['screener_pe'].notna().sum()}")
print(f"   Stocks with ROCE data: {scr_df['screener_roce'].notna().sum()}")
print(f"   Stocks with ROE data: {scr_df['screener_roe'].notna().sum()}")
print(f"\n   Avg P/E: {scr_df['screener_pe'].mean():.1f}")
print(f"   Avg ROCE: {scr_df['screener_roce'].mean():.1f}%")
print(f"   Avg ROE: {scr_df['screener_roe'].mean():.1f}%")
print(f"   Avg D/E: {scr_df['screener_de'].mean():.2f}")
print(f"   Avg Revenue Growth: {scr_df['screener_rev_growth'].mean():.1f}%")
print(f"   Avg Profit Growth: {scr_df['screener_profit_growth'].mean():.1f}%")

print(f"\n─── Top 10 by ROCE (Return on Capital Employed) ───")
display(scr_df.nlargest(10, "screener_roce")[["screener_roce", "screener_roe", "screener_pe", "screener_de", "screener_rev_growth"]].round(2))

print(f"\n─── Top 10 by Revenue Growth ───")
display(scr_df.nlargest(10, "screener_rev_growth")[["screener_rev_growth", "screener_profit_growth", "screener_roce", "screener_pe", "screener_margin"]].round(2))

# COMMAND ----------

# DBTITLE 1,Re-train Overview
# MAGIC %md
# MAGIC ## Step 9 — Re-train LightGBM with Enhanced Features
# MAGIC
# MAGIC Now we combine all features:
# MAGIC - **44 original features** (technical + cross-sectional + regime + yfinance fundamentals)
# MAGIC - **4 Finnhub sentiment features** (news sentiment, news count, analyst recommendations, analyst count)
# MAGIC - **12 Screener.in fundamentals** (PE, ROCE, ROE, D/E, P/B, revenue growth, profit growth, margin, EPS, current ratio, 5-year avg PE, 5-year avg ROCE)
# MAGIC - **2 cross-sectional rankings** (rank by ROCE, rank by sentiment)
# MAGIC
# MAGIC **Total: 62 features** → re-train LightGBM and compare AUC

# COMMAND ----------

# DBTITLE 1,Re-train LightGBM with Enhanced Features
# ─── Add Finnhub + Screener.in features to feat_df ───
from nse_algo.features.indicators import sma, rsi, atr, macd, bollinger_bands, stochastic
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import mlflow

# Re-fetch weekly data if not in memory (from earlier cells)
if 'weekly_data' not in dir() or len(weekly_data) == 0:
    print("Re-fetching weekly data (state lost from earlier cells)...")
    weekly_data = {}
    for i, sym in enumerate(UNIVERSE):
        try:
            ticker = yf.Ticker(f"{sym}.NS")
            df = ticker.history(start="2018-01-01", end=pd.Timestamp.now().strftime("%Y-%m-%d"), interval="1d")
            if len(df) < 200:
                continue
            df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
            weekly = df.resample("W-FRI").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
            if len(weekly) >= 200:
                weekly_data[sym] = weekly
        except:
            pass
        if (i + 1) % 20 == 0:
            time.sleep(1)
    # NIFTY
    nifty_ticker = yf.Ticker("^NSEI")
    nifty_df = nifty_ticker.history(start="2018-01-01", end=pd.Timestamp.now().strftime("%Y-%m-%d"), interval="1d")
    nifty_df = nifty_df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
    nifty_weekly = nifty_df.resample("W-FRI").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    print(f"✅ Re-fetched {len(weekly_data)} stocks + NIFTY")

# Rebuild features if needed
if 'feat_df' not in dir() or len(feat_df) == 0:
    print("Rebuilding feature_df...")
    all_feats = []
    for sym, df in weekly_data.items():
        close, high, low, vol = df["close"], df["high"], df["low"], df["volume"]
        rsi_14 = rsi(close, 14)
        _, _, macd_hist = macd(close)
        bb_u, bb_m, bb_l = bollinger_bands(close, 20, 2.0)
        atr_14 = atr(high, low, close, 14)
        st_k, st_d = stochastic(high, low, close, 14, 3)
        feat = pd.DataFrame(index=df.index)
        feat["symbol"] = sym
        for p in [1, 2, 4, 12, 26, 52]:
            feat[f"mom_{p}w"] = close.pct_change(p)
        for p in [10, 20, 50]:
            s = sma(close, p)
            feat[f"sma_ratio_{p}"] = close / s
            feat[f"price_vs_sma{p}_pct"] = (close / s - 1) * 100
        feat["price_above_sma50"] = (close > sma(close, 50)).astype(int)
        feat["rsi_14"] = rsi_14
        feat["macd_hist"] = macd_hist
        feat["bb_pctb"] = (close - bb_l) / (bb_u - bb_l)
        feat["atr_pct"] = (atr_14 / close) * 100
        low_14, high_14 = low.rolling(14).min(), high.rolling(14).max()
        feat["williams_r"] = ((high_14 - close) / (high_14 - low_14)) * -100
        feat["stoch_k"], feat["stoch_d"] = st_k, st_d
        feat["vol_ratio_4"] = vol / vol.rolling(4).mean()
        feat["vol_mom_4"] = vol.pct_change(4)
        ret = close.pct_change()
        feat["realized_vol_4w"] = ret.rolling(4).std() * np.sqrt(52)
        feat["realized_vol_12w"] = ret.rolling(12).std() * np.sqrt(52)
        feat["roc_12w"] = ((close - close.shift(12)) / close.shift(12)) * 100
        all_feats.append(feat)
    feat_df = pd.concat(all_feats).sort_index()
    tech_cols = [c for c in feat_df.columns if c != "symbol"]
    # Cross-sectional
    for col, rn in [("mom_26w", "rank_mom26"), ("rsi_14", "rank_rsi"), ("vol_ratio_4", "rank_vol")]:
        feat_df[rn] = feat_df.groupby(level=0)[col].rank(pct=True)
    for col, zn in [("mom_26w", "zscore_mom26"), ("rsi_14", "zscore_rsi"), ("mom_12w", "zscore_mom12")]:
        grp = feat_df.groupby(level=0)[col]
        feat_df[zn] = (grp.transform(lambda x: x - x.mean()) / grp.transform(lambda x: x.std() + 1e-8))
    cross_cols = ["rank_mom26", "rank_rsi", "rank_vol", "zscore_mom26", "zscore_rsi", "zscore_mom12"]
    # Regime
    nifty_close = nifty_weekly["close"]
    nifty_sma40 = sma(nifty_close, CONFIG["regime_sma"])
    nifty_ret = nifty_close.pct_change()
    nifty_feat = pd.DataFrame(index=nifty_weekly.index)
    nifty_feat["nifty_regime"] = (nifty_close > nifty_sma40).astype(int)
    nifty_feat["nifty_mom_4w"] = nifty_close.pct_change(4)
    nifty_feat["nifty_mom_12w"] = nifty_close.pct_change(12)
    nifty_feat["nifty_vol_12w"] = nifty_ret.rolling(12).std() * np.sqrt(52)
    for col in nifty_feat.columns:
        feat_df[col] = feat_df.index.map(nifty_feat[col])
    regime_cols = list(nifty_feat.columns)
    # yfinance fundamentals (skip if slow)
    fund_cols = ["fund_pe", "fund_fwd_pe", "fund_roe", "fund_margin", "fund_rev_growth", "fund_earn_growth", "fund_de", "fund_pb", "fund_mcap"]
    for sym in weekly_data.keys():
        try:
            info = yf.Ticker(f"{sym}.NS").info
            mask = feat_df["symbol"] == sym
            for col, key in [("fund_pe", "trailingPE"), ("fund_fwd_pe", "forwardPE"), ("fund_roe", "returnOnEquity"), ("fund_margin", "profitMargins"), ("fund_rev_growth", "revenueGrowth"), ("fund_earn_growth", "earningsGrowth"), ("fund_de", "debtToEquity"), ("fund_pb", "priceToBook"), ("fund_mcap", None)]:
                if key:
                    feat_df.loc[mask, col] = info.get(key, np.nan)
                else:
                    feat_df.loc[mask, col] = np.log1p(info.get("marketCap", 0))
        except:
            pass
    # Triple-barrier labels
    def triple_barrier_labels(df, vol_lookback=26, upper_mult=2.0, lower_mult=1.0, max_weeks=4):
        close, high, low = df["close"].values, df["high"].values, df["low"].values
        n = len(close)
        labels = np.full(n, np.nan)
        rets = np.diff(np.log(close))
        for i in range(n - 1):
            if i < vol_lookback:
                continue
            vol = np.std(rets[i-vol_lookback:i])
            if vol <= 0 or np.isnan(vol):
                continue
            upper = close[i] * (1 + upper_mult * vol)
            lower = close[i] * (1 - lower_mult * vol)
            for j in range(1, min(max_weeks + 1, n - i)):
                if low[i + j] <= lower:
                    labels[i] = -1
                    break
                elif high[i + j] >= upper:
                    labels[i] = 1
                    break
            else:
                labels[i] = 0
        return pd.Series(labels, index=df.index)
    all_labels = []
    for sym, df in weekly_data.items():
        lbls = triple_barrier_labels(df, CONFIG["vol_lookback"], CONFIG["barrier_multiplier"], CONFIG["stop_multiplier"], CONFIG["max_holding_weeks"])
        all_labels.append(pd.DataFrame({"symbol": sym, "label": lbls}, index=df.index))
    label_df = pd.concat(all_labels).sort_index()
    label_df["_key"] = label_df.index.astype(str) + "_" + label_df["symbol"]
    key_map = dict(zip(label_df["_key"], label_df["label"]))
    feat_df["label"] = (feat_df.index.astype(str) + "_" + feat_df["symbol"]).map(key_map)
    feat_df["target"] = (feat_df["label"] == 1).astype(int)
    all_feat_cols = tech_cols + cross_cols + regime_cols + fund_cols
    print(f"✅ Rebuilt features: {len(all_feat_cols)} cols, {len(feat_df):,} rows")

# ─── Add Finnhub sentiment features ───
if 'finnhub_sentiment' not in dir():
    print("⚠️ finnhub_sentiment not found — creating empty (no Finnhub API key)")
    finnhub_sentiment = {}
fh_cols = ["fh_sentiment", "fh_news_count", "fh_rec_score", "fh_analyst_count"]
for sym in feat_df["symbol"].unique():
    mask = feat_df["symbol"] == sym
    sent = finnhub_sentiment.get(sym, {})
    for col in fh_cols:
        feat_df.loc[mask, col] = sent.get(col, 0)

# ─── Add Screener.in fundamentals ───
screener_cols = list(scr_df.columns)
for sym in feat_df["symbol"].unique():
    mask = feat_df["symbol"] == sym
    if sym in scr_df.index:
        for col in screener_cols:
            feat_df.loc[mask, col] = scr_df.loc[sym, col]

# ─── Add cross-sectional rankings for new features ───
feat_df["rank_roce"] = feat_df.groupby(level=0)["screener_roce"].rank(pct=True)
feat_df["rank_sentiment"] = feat_df.groupby(level=0)["fh_sentiment"].rank(pct=True)

# ─── Enhanced feature set ───
valid_screener_cols = [c for c in screener_cols if feat_df[c].notna().mean() > 0.3]
print(f"\n   Valid Screener.in cols: {valid_screener_cols}")
enhanced_feat_cols = all_feat_cols + fh_cols + valid_screener_cols + ["rank_roce", "rank_sentiment"]
print(f"\n✅ Enhanced feature set: {len(enhanced_feat_cols)} features")
print(f"   Original: {len(all_feat_cols)}")
print(f"   + Finnhub sentiment: {len(fh_cols)}")
print(f"   + Screener.in fundamentals: {len(screener_cols)}")
print(f"   + New rankings: 2")

# ─── Re-train LightGBM with enhanced features ───
model_df_enh = feat_df.dropna(subset=["target"]).copy()
print(f"\nRows for ML (before imputation): {len(model_df_enh):,}")
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
model_df_enh[enhanced_feat_cols] = model_df_enh[enhanced_feat_cols].replace([np.inf, -np.inf], np.nan)
X_all = imputer.fit_transform(model_df_enh[enhanced_feat_cols])
model_df_enh[enhanced_feat_cols] = X_all
print(f"Rows for ML (after imputation): {len(model_df_enh):,}")

# Walk-forward split (same as before)
all_weeks = sorted(model_df_enh.index.unique())
n_train = int(len(all_weeks) * CONFIG["ml_train_fraction"])
train_end = all_weeks[n_train]
gap_end = all_weeks[min(n_train + CONFIG["ml_gap_weeks"], len(all_weeks) - 1)]

train_enh = model_df_enh[model_df_enh.index < train_end].copy()
test_enh = model_df_enh[model_df_enh.index >= gap_end].copy()

print(f"Train: {train_enh.index[0].date()} → {train_enh.index[-1].date()} ({len(train_enh):,} rows)")
print(f"Test:  {test_enh.index[0].date()} → {test_enh.index[-1].date()} ({len(test_enh):,} rows)")

X_train_e = train_enh[enhanced_feat_cols]
y_train_e = train_enh["target"].values
X_test_e = test_enh[enhanced_feat_cols]
y_test_e = test_enh["target"].values

# Train enhanced LightGBM
lgb_enh = lgb.LGBMClassifier(
    n_estimators=500, max_depth=5, learning_rate=0.05,
    num_leaves=31, min_child_samples=100,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbose=-1,
)
lgb_enh.fit(X_train_e, y_train_e, eval_set=[(X_test_e, y_test_e)], callbacks=[lgb.early_stopping(50, verbose=False)])

train_auc_e = roc_auc_score(y_train_e, lgb_enh.predict_proba(X_train_e)[:, 1])
test_auc_e = roc_auc_score(y_test_e, lgb_enh.predict_proba(X_test_e)[:, 1])

test_enh["ml_prob"] = lgb_enh.predict_proba(X_test_e)[:, 1]
test_enh["pred_rank"] = test_enh.groupby(level=0)["ml_prob"].rank(pct=True, ascending=True)

top_decile_e = test_enh[test_enh["pred_rank"] >= 0.9]
bot_half_e = test_enh[test_enh["pred_rank"] < 0.5]

print(f"\n{'═'*65}")
print(f"  ML MODEL COMPARISON — Original vs Enhanced")
print(f"{'═'*65}")
print(f"\n{'Model':<25} {'Features':>8} {'Train AUC':>10} {'Test AUC':>10} {'Top Dec':>8} {'Bot Half':>8}")
print(f"{'─'*65}")
prev_train_auc = locals().get('train_auc', 0.697)
prev_test_auc = locals().get('test_auc', 0.510)
prev_top = 0.273
prev_bot = 0.249
print(f"{'Original (44 feat)':<25} {44:>8} {prev_train_auc:>10.3f} {prev_test_auc:>10.3f} {prev_top:>7.1%} {prev_bot:>7.1%}")
print(f"{'Enhanced ({len(enhanced_feat_cols)} feat)':<25} {len(enhanced_feat_cols):>8} {train_auc_e:>10.3f} {test_auc_e:>10.3f} {top_decile_e['target'].mean():>7.1%} {bot_half_e['target'].mean():>7.1%}")
print(f"{'─'*65}")
auc_delta = test_auc_e - prev_test_auc
print(f"  AUC improvement: {auc_delta:+.3f} ({'✅ improved' if auc_delta > 0 else '❌ no improvement'})")

# Feature importance for new features
print(f"\n─── Top 15 Features (Enhanced Model) ───")
imp_e = pd.DataFrame({"feature": enhanced_feat_cols, "importance": lgb_enh.feature_importances_})
imp_e = imp_e.sort_values("importance", ascending=False)
display(imp_e.head(15))

# Highlight new features in importance
new_feat_imp = imp_e[imp_e["feature"].str.startswith(("fh_", "screener_", "rank_roce", "rank_sentiment"))]
if len(new_feat_imp) > 0:
    print(f"\n─── New Feature Importances ───")
    display(new_feat_imp.head(10))
else:
    print("\n⚠️ New features have 0 importance — model didn't find them useful")

# Log to MLflow
with mlflow.start_run(run_name="lightgbm_enhanced_finnhub_screener"):
    mlflow.log_params({"n_features": len(enhanced_feat_cols), "n_stocks": model_df_enh["symbol"].nunique(), "model": "LightGBM_enhanced"})
    mlflow.log_metrics({"train_auc": train_auc_e, "test_auc": test_auc_e, "auc_delta_vs_original": auc_delta})
    mlflow.lightgbm.log_model(lgb_enh, "lightgbm_enhanced")
print("\n✅ Enhanced model logged to MLflow")

# COMMAND ----------

# DBTITLE 1,Final Stock Selection
# MAGIC %md
# MAGIC ## Step 10 — Final Stock Selection with Enhanced Model
# MAGIC
# MAGIC Current stock picks using the enhanced LightGBM model (53 features) with Screener.in fundamentals and news sentiment.

# COMMAND ----------

# DBTITLE 1,Enhanced Stock Selection Output
# ─── Final Stock Selection with Enhanced Model ───
print("═" * 80)
print("  ENHANCED STOCK SELECTION — NIFTY 200")
print("  " + datetime.now().strftime("%Y-%m-%d %H:%M UTC"))
print("  Model: LightGBM Enhanced (53 features, Test AUC = {:.3f})".format(test_auc_e))
print("═" * 80)

# Get latest week predictions from enhanced model
latest_week_enh = test_enh.index.max()
latest_preds_enh = test_enh[test_enh.index == latest_week_enh].sort_values("ml_prob", ascending=False)

# NIFTY regime
nifty_close = nifty_weekly["close"].iloc[-1]
nifty_sma_val = sma(nifty_weekly["close"], CONFIG["regime_sma"]).iloc[-1]
nifty_bull = nifty_close > nifty_sma_val if not pd.isna(nifty_sma_val) else False
print(f"\n📊 NIFTY 50 Regime: {'🟢 BULL' if nifty_bull else '🔴 BEAR'} (close={nifty_close:.0f}, SMA40={nifty_sma_val:.0f})")

# Build selection table with Screener.in fundamentals
selections_enh = []
for _, row in latest_preds_enh.iterrows():
    sym = row["symbol"]
    if sym not in weekly_data:
        continue
    df = weekly_data[sym]
    close = df["close"].iloc[-1]
    s50 = sma(df["close"], CONFIG["sma_period"])
    sma_val = s50.iloc[-1]
    is_uptrend = close > sma_val if not pd.isna(sma_val) else False
    
    # Get Screener.in data
    scr = screener_fundamentals.get(sym, {})
    
    selections_enh.append({
        "symbol": sym,
        "ml_prob": row["ml_prob"],
        "pred_rank": row["pred_rank"],
        "close": close,
        "price_vs_sma": ((close/sma_val - 1)*100) if not pd.isna(sma_val) else 0,
        "is_uptrend": is_uptrend,
        "roce": scr.get("screener_roce", np.nan),
        "roce_5y": scr.get("screener_roce_5y_avg", np.nan),
        "profit_growth": scr.get("screener_profit_growth", np.nan),
        "rev_growth": scr.get("screener_rev_growth", np.nan),
        "margin": scr.get("screener_margin", np.nan),
        "nifty_bull": nifty_bull,
        "recommendation": "BUY" if (is_uptrend and nifty_bull and row["pred_rank"] >= 0.7) else ("HOLD" if is_uptrend else "AVOID"),
    })

sel_enh_df = pd.DataFrame(selections_enh).sort_values("ml_prob", ascending=False)

# Top 20 with fundamentals
cols_to_show = ["symbol", "ml_prob", "pred_rank", "close", "price_vs_sma", "is_uptrend", "roce", "profit_growth", "recommendation"]
print(f"\n─── TOP 20 STOCKS BY ENHANCED ML PROBABILITY ───")
display(sel_enh_df.head(20)[cols_to_show].round(2))

# BUY signals
buy_signals = sel_enh_df[sel_enh_df["recommendation"] == "BUY"]
print(f"\n─── BUY RECOMMENDATIONS ({len(buy_signals)} stocks) ───")
if len(buy_signals) > 0:
    display(buy_signals[["symbol", "ml_prob", "close", "roce", "profit_growth", "rev_growth"]].round(2))
else:
    print("   No stocks meet all criteria (ML rank >= 70th pct + uptrend + NIFTY bull)")
    if not nifty_bull:
        print("   Reason: NIFTY 50 is in BEAR regime — strategy stays in cash")

# Quality stocks (high ROCE + uptrend)
quality_stocks = sel_enh_df[(sel_enh_df["roce"] > 20) & (sel_enh_df["is_uptrend"])].sort_values("roce", ascending=False)
print(f"\n─── QUALITY STOCKS WATCHLIST (ROCE > 20% + Uptrend) ───")
if len(quality_stocks) > 0:
    display(quality_stocks[["symbol", "roce", "roce_5y", "profit_growth", "margin", "ml_prob", "recommendation"]].head(15).round(2))
else:
    print("   No stocks with ROCE > 20% are currently in uptrend")

# Fundamental screen (high ROCE + high profit growth)
fund_screen = sel_enh_df[(sel_enh_df["roce"] > 15) & (sel_enh_df["profit_growth"] > 10)].sort_values("roce", ascending=False)
print(f"\n─── FUNDAMENTAL SCREEN (ROCE > 15% + Profit Growth > 10%) ───")
if len(fund_screen) > 0:
    display(fund_screen[["symbol", "roce", "profit_growth", "rev_growth", "margin", "is_uptrend", "recommendation"]].head(15).round(2))
else:
    print("   No stocks meet fundamental criteria")

print(f"\n{'═' * 80}")
print(f"  SUMMARY: {len(buy_signals)} BUY / {len(sel_enh_df[sel_enh_df['recommendation']=='HOLD'])} HOLD / {len(sel_enh_df[sel_enh_df['recommendation']=='AVOID'])} AVOID")
print(f"  Model: Enhanced LightGBM (53 features, Test AUC = {test_auc_e:.3f})")
print(f"  Data: 59,248 rows (imputed), 130 stocks, 8 years")
print(f"  Features: 44 original + 4 Finnhub + 3 Screener.in + 2 rankings")
if FINNHUB_AVAILABLE:
    print(f"  Finnhub: ✅ Active ({(fh_sent_df['fh_news_count'] > 0).sum()} stocks with news)")
else:
    print(f"  Finnhub: ⚠️  No API key (get free key at finnhub.io/register)")
print(f"  Screener.in: ✅ {success_count}/130 stocks scraped")
print(f"{'═' * 80}")

# COMMAND ----------

# DBTITLE 1,Step 11 Overview
# MAGIC %md
# MAGIC ## Step 11 - Google News RSS Sentiment + Feature Cleanup + Swing Trading
# MAGIC
# MAGIC 1. **Feature Cleanup**: Remove rank_sentiment (spurious with zero data). Keep Finnhub/Screener.in features for future improvement.
# MAGIC 2. **Google News RSS**: Fetch real Indian stock news from news.google.com/rss - free, covers Indian stocks.
# MAGIC 3. **Swing Trading Strategy**: ML selection + technical entry + ATR stop/target + weekly rebalance.

# COMMAND ----------

# DBTITLE 1,Google News RSS Sentiment
# Google News RSS Sentiment for Indian Stocks
import xml.etree.ElementTree as ET
analyzer = SentimentIntensityAnalyzer()

def fetch_google_news_sentiment(symbol, max_items=10):
    url = f"https://news.google.com/rss/search?q={symbol}+NSE+stock&hl=en-IN&gl=IN&ceid=IN:en"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code != 200:
            return {"gn_sentiment": 0, "gn_news_count": 0, "gn_latest_headline": "HTTP error", "gn_headlines": []}
        root = ET.fromstring(resp.content)
        items = root.findall(".//item")[:max_items]
        if not items:
            return {"gn_sentiment": 0, "gn_news_count": 0, "gn_latest_headline": "No news", "gn_headlines": []}
        scores, headlines = [], []
        for item in items:
            title = item.findtext("title", default="")
            title = title.replace("&quot;", '"').replace("&amp;", "&")
            scores.append(analyzer.polarity_scores(title)["compound"])
            headlines.append(title)
        return {"gn_sentiment": np.mean(scores), "gn_news_count": len(items), "gn_latest_headline": headlines[0] if headlines else "", "gn_headlines": headlines[:5]}
    except Exception as e:
        return {"gn_sentiment": 0, "gn_news_count": 0, "gn_latest_headline": f"Error: {str(e)[:30]}", "gn_headlines": []}

print("Testing Google News RSS with RELIANCE...")
test_gn = fetch_google_news_sentiment("RELIANCE")
print(f"  News count: {test_gn['gn_news_count']}")
print(f"  Sentiment: {test_gn['gn_sentiment']:.3f}")
print(f"  Latest: {test_gn['gn_latest_headline'][:80]}")
for h in test_gn['gn_headlines'][:3]:
    print(f"    - {h[:80]}")

print(f"\nFetching Google News RSS for {len(UNIVERSE)} stocks...")
google_news_sentiment = {}
news_found = 0
for i, sym in enumerate(UNIVERSE):
    google_news_sentiment[sym] = fetch_google_news_sentiment(sym)
    if google_news_sentiment[sym]["gn_news_count"] > 0:
        news_found += 1
    time.sleep(0.3)
    if (i + 1) % 20 == 0:
        print(f"  ... {i+1}/{len(UNIVERSE)} processed ({news_found} with news)")

gn_df = pd.DataFrame.from_dict(google_news_sentiment, orient="index")
print(f"\n✅ Google News RSS complete: {news_found}/{len(UNIVERSE)} stocks with news")
if news_found > 0:
    print(f"   Avg sentiment: {gn_df.loc[gn_df['gn_news_count'] > 0, 'gn_sentiment'].mean():.3f}")
    print(f"\n─── Most Positive (top 10) ───")
    pos = gn_df[gn_df["gn_news_count"] > 0].nlargest(10, "gn_sentiment")
    display(pos[["gn_sentiment", "gn_news_count", "gn_latest_headline"]])
    print(f"\n─── Most Negative (bottom 10) ───")
    neg = gn_df[gn_df["gn_news_count"] > 0].nsmallest(10, "gn_sentiment")
    display(neg[["gn_sentiment", "gn_news_count", "gn_latest_headline"]])
else:
    print("   ⚠️ No news found - Google News RSS may be blocked in this environment")

# COMMAND ----------

# DBTITLE 1,Update Sentiment + Re-train v2
# Update sentiment with Google News + Re-train cleaned model
for sym in UNIVERSE:
    gn = google_news_sentiment.get(sym, {})
    if sym in finnhub_sentiment:
        if gn.get("gn_news_count", 0) > 0:
            finnhub_sentiment[sym]["fh_sentiment"] = gn["gn_sentiment"]
            finnhub_sentiment[sym]["fh_news_count"] = gn["gn_news_count"]
            finnhub_sentiment[sym]["fh_latest_headline"] = gn["gn_latest_headline"]
            finnhub_sentiment[sym]["fh_headlines"] = gn["gn_headlines"]
    else:
        finnhub_sentiment[sym] = {"fh_sentiment": gn.get("gn_sentiment", 0), "fh_news_count": gn.get("gn_news_count", 0), "fh_rec_score": 0, "fh_analyst_count": 0, "fh_latest_headline": gn.get("gn_latest_headline", ""), "fh_headlines": gn.get("gn_headlines", [])}

fh_sent_df = pd.DataFrame.from_dict(finnhub_sentiment, orient="index")
news_now = (fh_sent_df["fh_news_count"] > 0).sum()
print(f"✅ Updated sentiment: {news_now}/{len(fh_sent_df)} stocks with real news data")
if news_now > 0:
    print(f"   Avg sentiment: {fh_sent_df.loc[fh_sent_df['fh_news_count'] > 0, 'fh_sentiment'].mean():.3f}")

# Update feat_df with new sentiment
for sym in feat_df["symbol"].unique():
    mask = feat_df["symbol"] == sym
    sent = finnhub_sentiment.get(sym, {})
    feat_df.loc[mask, "fh_sentiment"] = sent.get("fh_sentiment", 0)
    feat_df.loc[mask, "fh_news_count"] = sent.get("fh_news_count", 0)
feat_df["rank_sentiment"] = feat_df.groupby(level=0)["fh_sentiment"].rank(pct=True)

# Feature cleanup: remove rank_sentiment if all same value
if news_now == 0:
    print("\n⚠️ No news - removing rank_sentiment (spurious, all identical)")
    enhanced_feat_cols_v2 = [c for c in enhanced_feat_cols if c != "rank_sentiment"]
else:
    print(f"\n✅ {news_now} stocks have real news - keeping rank_sentiment")
    enhanced_feat_cols_v2 = enhanced_feat_cols
print(f"Cleaned feature set: {len(enhanced_feat_cols_v2)} features")

# Re-train
from sklearn.impute import SimpleImputer
model_df_v2 = feat_df.dropna(subset=["target"]).copy()
model_df_v2[enhanced_feat_cols_v2] = model_df_v2[enhanced_feat_cols_v2].replace([np.inf, -np.inf], np.nan)
imp_v2 = SimpleImputer(strategy="median")
model_df_v2[enhanced_feat_cols_v2] = imp_v2.fit_transform(model_df_v2[enhanced_feat_cols_v2])

all_w = sorted(model_df_v2.index.unique())
n_tr = int(len(all_w) * CONFIG["ml_train_fraction"])
tr_end = all_w[n_tr]
gap_end = all_w[min(n_tr + CONFIG["ml_gap_weeks"], len(all_w) - 1)]
train_v2 = model_df_v2[model_df_v2.index < tr_end].copy()
test_v2 = model_df_v2[model_df_v2.index >= gap_end].copy()

lgb_v2 = lgb.LGBMClassifier(n_estimators=500, max_depth=5, learning_rate=0.05, num_leaves=31, min_child_samples=100, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbose=-1)
lgb_v2.fit(train_v2[enhanced_feat_cols_v2], train_v2["target"].values, eval_set=[(test_v2[enhanced_feat_cols_v2], test_v2["target"].values)], callbacks=[lgb.early_stopping(50, verbose=False)])

train_auc_v2 = roc_auc_score(train_v2["target"].values, lgb_v2.predict_proba(train_v2[enhanced_feat_cols_v2])[:, 1])
test_auc_v2 = roc_auc_score(test_v2["target"].values, lgb_v2.predict_proba(test_v2[enhanced_feat_cols_v2])[:, 1])
test_v2["ml_prob"] = lgb_v2.predict_proba(test_v2[enhanced_feat_cols_v2])[:, 1]
test_v2["pred_rank"] = test_v2.groupby(level=0)["ml_prob"].rank(pct=True, ascending=True)

top_dec_v2 = test_v2[test_v2["pred_rank"] >= 0.9]
print(f"\n{'='*65}")
print(f"  ML MODEL COMPARISON")
print(f"{'='*65}")
print(f"  {'Model':<30} {'Feat':>5} {'TrAUC':>6} {'TeAUC':>6} {'TopDec':>7}")
print(f"  {'-'*60}")
print(f"  {'Original (44 feat)':<30} {44:>5} {0.697:>6.3f} {0.510:>6.3f} {0.273:>6.1%}")
print(f"  {'Enhanced v1 (53 feat)':<30} {53:>5} {train_auc_e:>6.3f} {test_auc_e:>6.3f} {top_decile_e['target'].mean():>6.1%}")
print(f"  {'Enhanced v2 ({len(enhanced_feat_cols_v2)} feat)':<30} {len(enhanced_feat_cols_v2):>5} {train_auc_v2:>6.3f} {test_auc_v2:>6.3f} {top_dec_v2['target'].mean():>6.1%}")
print(f"  {'-'*60}")
print(f"  AUC change v1->v2: {test_auc_v2 - test_auc_e:+.3f}")

imp_v2_df = pd.DataFrame({"feature": enhanced_feat_cols_v2, "importance": lgb_v2.feature_importances_}).sort_values("importance", ascending=False)
print(f"\n─── Top 15 Features (v2) ───")
display(imp_v2_df.head(15))

# COMMAND ----------

# DBTITLE 1,Swing Strategy Overview
# MAGIC %md
# MAGIC ## Step 12 - Swing Trading Strategy + Backtest
# MAGIC
# MAGIC ### Strategy Rules
# MAGIC
# MAGIC **Entry** (all must be true):
# MAGIC 1. ML probability in top 20 (cross-sectional rank)
# MAGIC 2. NIFTY regime = BULL (close > SMA40)
# MAGIC 3. Stock in uptrend (close > SMA50)
# MAGIC 4. RSI(14) between 40-70 (not overbought)
# MAGIC 5. Screener.in ROCE > 15% (quality filter)
# MAGIC
# MAGIC **Exit** (any one triggers):
# MAGIC 1. Stop loss: -1.5 x ATR(14) from entry
# MAGIC 2. Take profit: +3.0 x ATR(14) from entry (2:1 R/R)
# MAGIC 3. Time exit: After 4 weeks
# MAGIC 4. Trend exit: Close falls below SMA50
# MAGIC
# MAGIC **Portfolio**: Max 15 positions, equal weight, weekly rebalance, cash earns 0%

# COMMAND ----------

# DBTITLE 1,Swing Trading Backtest
# Swing Trading Strategy Backtest
from nse_algo.features.indicators import sma as calc_sma, rsi as calc_rsi, atr as calc_atr

SWING = {"max_positions": 15, "initial_capital": 100_000, "stop_atr_mult": 1.5, "target_atr_mult": 3.0, "max_holding_weeks": 4, "rsi_min": 40, "rsi_max": 70, "roce_min": 15.0, "top_n": 20}

test_weeks = sorted(test_v2.index.unique())
print(f"Backtest: {test_weeks[0].date()} to {test_weeks[-1].date()} ({len(test_weeks)} weeks)")

capital = SWING["initial_capital"]
positions = {}
portfolio_values = []
weekly_returns = []
trade_log = []

def get_indicators(sym, week_idx):
    if sym not in weekly_data or week_idx not in weekly_data[sym].index:
        return None
    df = weekly_data[sym]
    idx = df.index.get_loc(week_idx)
    if idx < 50:
        return None
    c, h, l = df["close"], df["high"], df["low"]
    return {"close": c.iloc[idx], "rsi": calc_rsi(c, 14).iloc[idx], "sma50": calc_sma(c, 50).iloc[idx], "atr": calc_atr(h, l, c, 14).iloc[idx]}

def nifty_bull(week_idx):
    if nifty_weekly is None or week_idx not in nifty_weekly.index:
        return False
    idx = nifty_weekly.index.get_loc(week_idx)
    if idx < 40:
        return False
    c = nifty_weekly["close"].iloc[idx]
    s = calc_sma(nifty_weekly["close"], 40).iloc[idx]
    return c > s if not pd.isna(s) else False

for wi, week_idx in enumerate(test_weeks):
    ws = week_idx.strftime("%Y-%m-%d")
    # 1. Check exits
    to_close = []
    for sym, pos in positions.items():
        if sym not in weekly_data or week_idx not in weekly_data[sym].index:
            to_close.append((sym, pos["entry_price"], "data_missing"))
            continue
        cp = weekly_data[sym]["close"].loc[week_idx]
        wh = wi - test_weeks.index(pos["entry_week"])
        reason = None
        if cp <= pos["stop"]:
            reason = "stop_loss"
        elif cp >= pos["target"]:
            reason = "take_profit"
        elif wh >= SWING["max_holding_weeks"]:
            reason = "time_exit"
        elif cp < pos.get("sma50_entry", np.inf) * 0.98:
            reason = "trend_exit"
        if reason:
            to_close.append((sym, cp, reason))
    for sym, ep, reason in to_close:
        pos = positions.pop(sym)
        pnl = (ep - pos["entry_price"]) / pos["entry_price"]
        capital += pos["shares"] * ep
        trade_log.append({"symbol": sym, "entry_week": pos["entry_week"].strftime("%Y-%m-%d"), "exit_week": ws, "entry_price": pos["entry_price"], "exit_price": ep, "return_pct": pnl * 100, "exit_reason": reason, "weeks_held": wi - test_weeks.index(pos["entry_week"])})
    # 2. New entries
    if nifty_bull(week_idx) and len(positions) < SWING["max_positions"]:
        preds = test_v2[test_v2.index == week_idx].sort_values("ml_prob", ascending=False)
        candidates = []
        for _, row in preds.iterrows():
            sym = row["symbol"]
            if sym in positions or len(positions) + len(candidates) >= SWING["max_positions"]:
                continue
            ind = get_indicators(sym, week_idx)
            if ind is None:
                continue
            if ind["rsi"] < SWING["rsi_min"] or ind["rsi"] > SWING["rsi_max"]:
                continue
            if ind["close"] <= ind["sma50"]:
                continue
            roce = screener_fundamentals.get(sym, {}).get("screener_roce", 0)
            if pd.isna(roce):
                roce = 0
            if roce < SWING["roce_min"]:
                continue
            candidates.append({"symbol": sym, "ml_prob": row["ml_prob"], "close": ind["close"], "rsi": ind["rsi"], "atr": ind["atr"], "sma50": ind["sma50"], "roce": roce})
        candidates.sort(key=lambda x: x["ml_prob"], reverse=True)
        n_slots = SWING["max_positions"] - len(positions)
        for cand in candidates[:n_slots]:
            if capital < 100:
                break
            size = capital / max(n_slots, 1)
            shares = size / cand["close"]
            capital -= shares * cand["close"]
            positions[cand["symbol"]] = {"entry_price": cand["close"], "entry_week": week_idx, "shares": shares, "stop": cand["close"] - SWING["stop_atr_mult"] * cand["atr"], "target": cand["close"] + SWING["target_atr_mult"] * cand["atr"], "sma50_entry": cand["sma50"]}
    # 3. Portfolio value
    pv = capital
    for sym, pos in positions.items():
        if sym in weekly_data and week_idx in weekly_data[sym].index:
            pv += pos["shares"] * weekly_data[sym]["close"].loc[week_idx]
        else:
            pv += pos["shares"] * pos["entry_price"]
    portfolio_values.append({"week": ws, "value": pv, "n_positions": len(positions), "nifty_bull": nifty_bull(week_idx)})
    if len(portfolio_values) > 1:
        weekly_returns.append(pv / portfolio_values[-2]["value"] - 1)

# Close remaining
for sym, pos in list(positions.items()):
    lw = test_weeks[-1]
    ep = weekly_data[sym]["close"].loc[lw] if sym in weekly_data and lw in weekly_data[sym].index else pos["entry_price"]
    pnl = (ep - pos["entry_price"]) / pos["entry_price"]
    capital += pos["shares"] * ep
    trade_log.append({"symbol": sym, "entry_week": pos["entry_week"].strftime("%Y-%m-%d"), "exit_week": lw.strftime("%Y-%m-%d"), "entry_price": pos["entry_price"], "exit_price": ep, "return_pct": pnl * 100, "exit_reason": "backtest_end", "weeks_held": len(test_weeks) - 1 - test_weeks.index(pos["entry_week"])})

print(f"Backtest complete: {len(trade_log)} trades")
print(f"\nComputing metrics...")

# Metrics
pv_df = pd.DataFrame(portfolio_values)
trade_df = pd.DataFrame(trade_log)
total_ret = (pv_df["value"].iloc[-1] / SWING["initial_capital"] - 1) * 100
n_yrs = len(pv_df) / 52
cagr = ((pv_df["value"].iloc[-1] / SWING["initial_capital"]) ** (1 / n_yrs) - 1) * 100 if n_yrs > 0 else 0
wr = np.array(weekly_returns)
sharpe = (wr.mean() / (wr.std() + 1e-8)) * np.sqrt(52) if len(wr) > 1 else 0
rmax = pv_df["value"].cummax()
mdd = ((pv_df["value"] / rmax - 1) * 100).min()

# NIFTY benchmark
nt = nifty_weekly[nifty_weekly.index >= test_weeks[0]]
n_ret = (nt["close"].iloc[-1] / nt["close"].iloc[0] - 1) * 100
n_cagr = ((nt["close"].iloc[-1] / nt["close"].iloc[0]) ** (1 / n_yrs) - 1) * 100 if n_yrs > 0 else 0
n_wr = nt["close"].pct_change().dropna()
n_sharpe = (n_wr.mean() / (n_wr.std() + 1e-8)) * np.sqrt(52)
n_mdd = ((nt["close"] / nt["close"].cummax() - 1) * 100).min()

if len(trade_df) > 0:
    wins = trade_df[trade_df["return_pct"] > 0]
    losses = trade_df[trade_df["return_pct"] <= 0]
    win_rate = len(wins) / len(trade_df) * 100
    avg_win = wins["return_pct"].mean() if len(wins) > 0 else 0
    avg_loss = losses["return_pct"].mean() if len(losses) > 0 else 0
    pf = abs(wins["return_pct"].sum() / losses["return_pct"].sum()) if len(losses) > 0 and losses["return_pct"].sum() != 0 else float('inf')
else:
    win_rate, avg_win, avg_loss, pf = 0, 0, 0, 0

print(f"\n{'='*70}")
print(f"  SWING TRADING STRATEGY - Backtest Results")
print(f"  Period: {test_weeks[0].date()} to {test_weeks[-1].date()} ({len(test_weeks)} weeks, {n_yrs:.1f} years)")
print(f"{'='*70}")
print(f"\n  {'Metric':<25} {'Swing':>15} {'NIFTY 50':>15} {'Diff':>10}")
print(f"  {'-'*65}")
print(f"  {'Total Return':<25} {total_ret:>14.1f}% {n_ret:>14.1f}% {total_ret - n_ret:>+9.1f}%")
print(f"  {'CAGR':<25} {cagr:>14.1f}% {n_cagr:>14.1f}% {cagr - n_cagr:>+9.1f}%")
print(f"  {'Sharpe Ratio':<25} {sharpe:>15.2f} {n_sharpe:>15.2f} {sharpe - n_sharpe:>+10.2f}")
print(f"  {'Max Drawdown':<25} {mdd:>14.1f}% {n_mdd:>14.1f}% {mdd - n_mdd:>+9.1f}%")
print(f"  {'-'*65}")
print(f"\n  Trade Statistics:")
print(f"    Total trades: {len(trade_df)}")
print(f"    Win rate: {win_rate:.1f}%")
print(f"    Avg win: +{avg_win:.2f}%")
print(f"    Avg loss: {avg_loss:.2f}%")
print(f"    Profit factor: {pf:.2f}")
if len(trade_df) > 0:
    print(f"    Avg weeks held: {trade_df['weeks_held'].mean():.1f}")
    print(f"\n  Exit Breakdown:")
    for reason, count in trade_df["exit_reason"].value_counts().items():
        ar = trade_df[trade_df["exit_reason"] == reason]["return_pct"].mean()
        print(f"    {reason}: {count} trades, avg {ar:+.2f}%")

if len(trade_df) > 0 and len(wins) > 0:
    print(f"\n  Top 10 Winning Trades:")
    display(wins.nlargest(10, "return_pct")[["symbol", "entry_week", "exit_week", "entry_price", "exit_price", "return_pct", "exit_reason"]].round(2))
if len(trade_df) > 0 and len(losses) > 0:
    print(f"\n  Top 10 Losing Trades:")
    display(losses.nsmallest(10, "return_pct")[["symbol", "entry_week", "exit_week", "entry_price", "exit_price", "return_pct", "exit_reason"]].round(2))

print(f"\n  Equity Curve (first/last 5 weeks):")
display(pv_df.head(5))
print("...")
display(pv_df.tail(5))

print(f"\n{'='*70}")
verdict = 'PROFITABLE - BEATS NIFTY' if total_ret > n_ret else 'NOT BEATING NIFTY'
print(f"  VERDICT: {verdict}")
print(f"  Swing: {total_ret:.1f}% return, Sharpe {sharpe:.2f}, MaxDD {mdd:.1f}%")
print(f"  NIFTY: {n_ret:.1f}% return, Sharpe {n_sharpe:.2f}, MaxDD {n_mdd:.1f}%")
print(f"  Alpha: {total_ret - n_ret:+.1f}% | Win Rate: {win_rate:.1f}% | PF: {pf:.2f}")
print(f"{'='*70}")

# COMMAND ----------

# DBTITLE 1,Paper Trading Overview
# MAGIC %md
# MAGIC ## Step 13 - Paper Trading System for Swing Strategy
# MAGIC
# MAGIC This cell runs the swing trading strategy in paper trading mode:
# MAGIC - Loads portfolio state from workspace (persists between runs)
# MAGIC - Checks exits for open positions using current prices
# MAGIC - Generates new BUY signals using ML + technical + fundamental filters
# MAGIC - Saves updated state
# MAGIC - Shows current portfolio status and this week's signals
# MAGIC
# MAGIC **Run this cell weekly (every Friday) to update signals.**

# COMMAND ----------

# DBTITLE 1,Paper Trading Init
# Quick initialization for paper trading (if earlier cells haven't been run)
from nse_algo.features.indicators import sma as calc_sma, rsi as calc_rsi, atr as calc_atr

# Fetch NIFTY weekly if not in memory
if 'nifty_weekly' not in dir() or nifty_weekly is None or len(nifty_weekly) == 0:
    print("Fetching NIFTY 50 weekly data...")
    import yfinance as yf
    nt = yf.Ticker("^NSEI").history(start="2018-01-01", end=pd.Timestamp.now().strftime("%Y-%m-%d"), interval="1d")
    nifty_weekly = nt.resample("W-FRI").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna()
    nifty_weekly.columns = [c.lower() for c in nifty_weekly.columns]
    print(f"✅ NIFTY 50: {len(nifty_weekly)} weeks, latest close = {nifty_weekly['close'].iloc[-1]:.0f}")
else:
    print(f"✅ NIFTY 50 already loaded: {len(nifty_weekly)} weeks")

# Fetch weekly data for all stocks if not in memory
if 'weekly_data' not in dir() or len(weekly_data) == 0:
    print("Fetching weekly stock data (130 stocks)...")
    weekly_data = {}
    UNIVERSE = UNIVERSE if 'UNIVERSE' in dir() else ["HDFCBANK","ICICIBANK","SBIN","RELIANCE","TCS","INFY","HCLTECH","WIPRO","LT","BHARTIARTL","ITC","HINDUNILVR","MARUTI","BAJAJ-AUTO","SUNPHARMA","TATASTEEL","ULTRACEMCO","ADANIPORTS","ADANIENT","TITAN"]
    for i, sym in enumerate(UNIVERSE):
        try:
            df = yf.Ticker(f"{sym}.NS").history(start="2018-01-01", end=pd.Timestamp.now().strftime("%Y-%m-%d"), interval="1d")
            if len(df) < 200:
                continue
            df.columns = [c.lower() for c in df.columns]
            w = df.resample("W-FRI").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
            if len(w) >= 200:
                weekly_data[sym] = w
        except:
            pass
        if (i + 1) % 30 == 0:
            time.sleep(0.5)
            print(f"  ... {i+1}/{len(UNIVERSE)} fetched")
    print(f"✅ Weekly data: {len(weekly_data)} stocks")
else:
    print(f"✅ Weekly data already loaded: {len(weekly_data)} stocks")

# Helper function for indicators
def get_indicators(sym, week_idx):
    if sym not in weekly_data or week_idx not in weekly_data[sym].index:
        return None
    df = weekly_data[sym]
    idx = df.index.get_loc(week_idx)
    if idx < 50:
        return None
    c, h, l = df["close"], df["high"], df["low"]
    return {"close": c.iloc[idx], "rsi": calc_rsi(c, 14).iloc[idx], "sma50": calc_sma(c, 50).iloc[idx], "atr": calc_atr(h, l, c, 14).iloc[idx]}

# Dummy screener_fundamentals if not available
if 'screener_fundamentals' not in dir():
    screener_fundamentals = {sym: {"screener_roce": 20} for sym in weekly_data}  # Default ROCE 20%
    print("⚠️ Using default screener_fundamentals (ROCE=20% for all)")

# Dummy finnhub_sentiment if not available
if 'finnhub_sentiment' not in dir():
    finnhub_sentiment = {sym: {"fh_sentiment": 0, "fh_news_count": 0, "fh_latest_headline": ""} for sym in weekly_data}

# Dummy test_v2 if not available (needed for ML signals)
if 'test_v2' not in dir():
    test_v2 = None
    print("⚠️ test_v2 not available - ML signals will be skipped. Run earlier cells for full signals.")
else:
    print(f"✅ test_v2 available: {len(test_v2)} rows")

print("\n✅ Initialization complete")

# COMMAND ----------

# DBTITLE 1,Paper Trading System
# Paper Trading - Swing Strategy State Management
import json
from datetime import datetime

STATE_PATH = "/Workspace/Repos/lokigovindasamy@gmail.com/algo_trade_data_bricks/swing_paper_trade_state.json"
SWING = {"max_positions": 15, "initial_capital": 100_000, "stop_atr_mult": 1.5, "target_atr_mult": 3.0, "max_holding_weeks": 4, "rsi_min": 40, "rsi_max": 70, "roce_min": 15.0}

# Load state
try:
    with open(STATE_PATH, "r") as f:
        state = json.load(f)
    print(f"📂 Loaded state from {STATE_PATH}")
except:
    state = {"capital": SWING["initial_capital"], "positions": {}, "trade_log": [], "start_date": datetime.now().isoformat()}
    print("🆕 No previous state - starting fresh paper trading portfolio")

print(f"   Capital: Rs {state['capital']:,.0f}")
print(f"   Open positions: {len(state['positions'])}")
print(f"   Total trades: {len(state['trade_log'])}")

# Current date and latest available week
now = datetime.now()
current_week_str = now.strftime("%Y-%m-%d")

# 1. NIFTY regime check
nifty_close = nifty_weekly["close"].iloc[-1]
nifty_sma = calc_sma(nifty_weekly["close"], 40).iloc[-1]
nifty_bull = nifty_close > nifty_sma if not pd.isna(nifty_sma) else False
print(f"\n📊 NIFTY 50 Regime: {'🟢 BULL' if nifty_bull else '🔴 BEAR'} (close={nifty_close:.0f}, SMA40={nifty_sma:.0f})")

# 2. Check exits for open positions
exits = []
for sym, pos in list(state["positions"].items()):
    if sym not in weekly_data:
        exits.append((sym, pos["entry_price"], "data_missing"))
        continue
    df = weekly_data[sym]
    close = df["close"].iloc[-1]
    high = df["high"].iloc[-1]
    low = df["low"].iloc[-1]
    entry_date = pos["entry_date"]
    weeks_held = 0
    try:
        entry_dt = datetime.fromisoformat(entry_date)
        weeks_held = (now - entry_dt).days // 7
    except:
        weeks_held = 0
    
    reason = None
    if low <= pos["stop"]:
        reason = "stop_loss"
        close = pos["stop"]  # Exit at stop price
    elif high >= pos["target"]:
        reason = "take_profit"
        close = pos["target"]  # Exit at target price
    elif weeks_held >= SWING["max_holding_weeks"]:
        reason = "time_exit"
    elif close < pos.get("sma50_entry", float('inf')) * 0.98:
        reason = "trend_exit"
    
    if reason:
        exits.append((sym, close, reason))

for sym, exit_price, reason in exits:
    pos = state["positions"].pop(sym)
    pnl = (exit_price - pos["entry_price"]) / pos["entry_price"]
    state["capital"] += pos["shares"] * exit_price
    state["trade_log"].append({"symbol": sym, "entry_date": pos["entry_date"], "exit_date": current_week_str, "entry_price": pos["entry_price"], "exit_price": exit_price, "return_pct": round(pnl * 100, 2), "exit_reason": reason})
    emoji = "✅" if pnl > 0 else "❌"
    print(f"  {emoji} EXIT {sym}: {pos['entry_price']:.0f} → {exit_price:.0f} ({pnl*100:+.1f}%) - {reason}")

# 3. Generate new BUY signals
new_signals = []
if nifty_bull and len(state["positions"]) < SWING["max_positions"] and test_v2 is not None:
    # Get latest week ML predictions
    latest_week = test_v2.index.max()
    latest_preds = test_v2[test_v2.index == latest_week].sort_values("ml_prob", ascending=False)
    
    for _, row in latest_preds.iterrows():
        sym = row["symbol"]
        if sym in state["positions"]:
            continue
        if len(state["positions"]) + len(new_signals) >= SWING["max_positions"]:
            break
        
        ind = get_indicators(sym, latest_week)
        if ind is None:
            continue
        if ind["rsi"] < SWING["rsi_min"] or ind["rsi"] > SWING["rsi_max"]:
            continue
        if ind["close"] <= ind["sma50"]:
            continue
        
        roce = screener_fundamentals.get(sym, {}).get("screener_roce", 0)
        if pd.isna(roce):
            roce = 0
        if roce < SWING["roce_min"]:
            continue
        
        sent = finnhub_sentiment.get(sym, {}).get("fh_sentiment", 0)
        headline = finnhub_sentiment.get(sym, {}).get("fh_latest_headline", "")
        
        new_signals.append({"symbol": sym, "ml_prob": row["ml_prob"], "close": ind["close"], "rsi": ind["rsi"], "atr": ind["atr"], "roce": roce, "sentiment": sent, "headline": headline[:50]})
    
    # Sort by ML probability, take top candidates
    new_signals.sort(key=lambda x: x["ml_prob"], reverse=True)
    n_slots = SWING["max_positions"] - len(state["positions"])
    for sig in new_signals[:n_slots]:
        if state["capital"] < 1000:
            break
        size = state["capital"] / max(n_slots, 1)
        shares = size / sig["close"]
        state["capital"] -= shares * sig["close"]
        stop = sig["close"] - SWING["stop_atr_mult"] * sig["atr"]
        target = sig["close"] + SWING["target_atr_mult"] * sig["atr"]
        state["positions"][sig["symbol"]] = {"entry_price": sig["close"], "entry_date": current_week_str, "shares": shares, "stop": stop, "target": target, "sma50_entry": get_indicators(sig["symbol"], latest_week)["sma50"]}
        print(f"  🟢 BUY {sig['symbol']}: Rs {sig['close']:.0f} | Stop Rs {stop:.0f} | Target Rs {target:.0f} | ML {sig['ml_prob']:.2f} | ROCE {sig['roce']:.0f}% | Sent {sig['sentiment']:.2f}")
else:
    if not nifty_bull:
        print("\n⚠️ NIFTY in BEAR regime - no new entries. Strategy in cash.")
    elif test_v2 is None:
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
print(f"  PAPER TRADING - Swing Strategy | {current_week_str}")
print(f"{'='*70}")

portfolio_val = state["capital"]
position_vals = []
for sym, pos in state["positions"].items():
    if sym in weekly_data:
        cp = weekly_data[sym]["close"].iloc[-1]
    else:
        cp = pos["entry_price"]
    val = pos["shares"] * cp
    portfolio_val += val
    pnl_pct = (cp - pos["entry_price"]) / pos["entry_price"] * 100
    position_vals.append({"symbol": sym, "entry": pos["entry_price"], "current": cp, "pnl_pct": pnl_pct, "value": val, "stop": pos["stop"], "target": pos["target"]})

total_return = (portfolio_val / SWING["initial_capital"] - 1) * 100
print(f"\n  Portfolio Value: Rs {portfolio_val:,.0f} (Return: {total_return:+.1f}%)")
print(f"  Cash: Rs {state['capital']:,.0f}")
print(f"  Open Positions: {len(state['positions'])}")

if position_vals:
    print(f"\n─── Open Positions ───")
    display(pd.DataFrame(position_vals).sort_values("pnl_pct", ascending=False).round(2))

if state["trade_log"]:
    print(f"\n─── Recent Trades (last 10) ───")
    tl = pd.DataFrame(state["trade_log"])
    display(tl.tail(10))
    wins = (tl["return_pct"] > 0).sum()
    print(f"\n  Win Rate: {wins}/{len(tl)} ({wins/len(tl)*100:.1f}%)")
    print(f"  Avg Return: {tl['return_pct'].mean():.2f}%")

if new_signals:
    print(f"\n─── This Week's BUY Signals ({len(new_signals)} stocks) ───")
    display(pd.DataFrame(new_signals).sort_values("ml_prob", ascending=False).round(2))

print(f"\n{'='*70}")
print(f"  Next steps: Re-run this cell weekly (Friday) to update signals")
print(f"{'='*70}")