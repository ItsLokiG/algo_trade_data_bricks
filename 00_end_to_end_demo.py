# Databricks notebook source
# DBTITLE 1,NSE Algo Trading — End-to-End Demo
# MAGIC %md
# MAGIC """
# MAGIC NSE Algorithmic Trading — End-to-End Pipeline Demo
# MAGIC
# MAGIC This notebook orchestrates the full research pipeline:
# MAGIC   1. Data ingestion (yfinance or synthetic fallback)
# MAGIC   2. Data-quality checks
# MAGIC   3. Technical indicators
# MAGIC   4. Signal generation
# MAGIC   5. Stock ranking
# MAGIC   6. Event-driven backtest with NSE transaction costs
# MAGIC   7. Performance metrics
# MAGIC   8. Walk-forward + Monte Carlo robustness validation
# MAGIC   9. Paper-trading engine demo
# MAGIC
# MAGIC All strategy logic lives in the nse_algo package (src/nse_algo/).
# MAGIC This notebook only orchestrates and displays results.
# MAGIC
# MAGIC DISCLAIMER: Historical performance does not guarantee future results.
# MAGIC All execution is simulated. This is a research tool, not investment advice.
# MAGIC """

# COMMAND ----------

# DBTITLE 1,Install yfinance
# MAGIC %pip install yfinance

# COMMAND ----------

# DBTITLE 1,Step 1: Setup
# MAGIC %md
# MAGIC ## Step 1 — Setup & Imports
# MAGIC
# MAGIC Import the `nse_algo` package and configure the project. All strategy logic lives in the package; this notebook only orchestrates and displays results.

# COMMAND ----------

# DBTITLE 1,Imports & Config
import sys, os

# Add the repo's src/ directory to the Python path
REPO_ROOT = "/Workspace/Repos/lokigovindasamy@gmail.com/algo_trade_data_bricks"
SRC_PATH = os.path.join(REPO_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from nse_algo.config import ProjectConfig, get_config
from nse_algo.data.providers import get_provider, YFinanceProvider, SyntheticProvider
from nse_algo.features.indicators import add_all_indicators
from nse_algo.strategy.signals import SignalConfig, generate_all_signals, SignalType
from nse_algo.strategy.ranking import RankingConfig, rank_stocks
from nse_algo.backtest.engine import run_backtest, BacktestResult, calculate_position_size
from nse_algo.backtest.metrics import calculate_all_metrics, PerformanceMetrics
from nse_algo.backtest.validation import (
    split_data_chronologically, walk_forward,
    monte_carlo_trade_reshuffle, parameter_sensitivity,
)
from nse_algo.reporting.reports import generate_trade_table, generate_text_report, generate_daily_summary
from nse_algo.paper.engine import PaperEngine

print("✅ All nse_algo modules imported successfully")

# Build the project configuration
cfg = get_config(phase=10)
print(f"Project: {cfg.project_name} v{cfg.version}, Phase {cfg.phase}")
print(f"Universe: {cfg.data.universe}")
print(f"Initial capital: ₹{cfg.capital.initial_capital:,.0f}")
print(f"Risk per trade: {cfg.risk.risk_per_trade:.1%}")
print(f"Max positions: {cfg.risk.max_positions}")
print(f"Cost rates verified: {cfg.costs.rates_verified}")

# COMMAND ----------

# DBTITLE 1,Step 2: Data Ingestion
# MAGIC %md
# MAGIC ## Step 2 — Data Ingestion
# MAGIC
# MAGIC Fetch historical OHLCV data for the NSE stock universe. We try `yfinance` first (now installed), and fall back to the `SyntheticProvider` if the network is unavailable.
# MAGIC
# MAGIC **⚠️ yfinance is an unofficial source. Results are NOT exchange-quality. Use only for prototyping.**

# COMMAND ----------

# DBTITLE 1,Fetch OHLCV Data
# Try yfinance first, fall back to synthetic
use_synthetic = False
provider = get_provider("yfinance")

if not provider.is_available():
    print("⚠️ yfinance not available — using SyntheticProvider")
    provider = get_provider("synthetic", seed=42)
    use_synthetic = True

symbols = list(cfg.data.universe)
start_date = cfg.data.start_date
end_date = cfg.data.end_date or pd.Timestamp.today().strftime("%Y-%m-%d")

price_data: dict[str, pd.DataFrame] = {}

for sym in symbols:
    try:
        df = provider.fetch_ohlcv(sym, start_date, end_date)
        price_data[sym] = df
        print(f"  {sym}: {len(df)} bars ({df.index[0].date()} → {df.index[-1].date()})")
    except Exception as e:
        print(f"  {sym}: FAILED — {e}")
        if not use_synthetic:
            # Fall back to synthetic for this symbol
            synth = get_provider("synthetic", seed=42)
            df = synth.fetch_ohlcv(sym, start_date, end_date)
            price_data[sym] = df
            print(f"  {sym}: using synthetic ({len(df)} bars)")

# Fetch benchmark (NIFTY 50)
benchmark_symbol = cfg.data.benchmark_symbol
try:
    benchmark_df = provider.fetch_ohlcv(benchmark_symbol, start_date, end_date)
    benchmark_close = benchmark_df["close"]
    print(f"\nBenchmark {benchmark_symbol}: {len(benchmark_df)} bars")
except Exception:
    benchmark_close = None
    print(f"\n⚠️ Benchmark {benchmark_symbol} unavailable — relative strength will be NaN")

print(f"\n✅ Data loaded for {len(price_data)} symbols")
if use_synthetic:
    print("⚠️ Using SYNTHETIC data — results are for pipeline validation only, NOT market analysis.")

# COMMAND ----------

# DBTITLE 1,Step 3: Data Quality
# MAGIC %md
# MAGIC ## Step 3 — Data Quality Checks
# MAGIC
# MAGIC Basic data-quality validation: check for missing values, duplicate dates, and sufficient history.

# COMMAND ----------

# DBTITLE 1,Data Quality Checks
quality_report = []

for sym, df in price_data.items():
    n_bars = len(df)
    n_duplicates = df.index.duplicated().sum()
    n_missing = df.isna().any(axis=1).sum()
    min_date = df.index[0].date()
    max_date = df.index[-1].date()
    
    # Check for gaps (missing trading days)
    all_dates = pd.bdate_range(start=min_date, end=max_date)
    n_gap_days = len(all_dates.difference(df.index))
    gap_pct = n_gap_days / len(all_dates) if len(all_dates) > 0 else 0
    
    passes = (
        n_bars >= cfg.data.min_trading_days
        and n_duplicates <= cfg.data.duplicate_tolerance
        and gap_pct <= cfg.data.max_missing_pct
    )
    
    quality_report.append({
        "symbol": sym,
        "bars": n_bars,
        "duplicates": n_duplicates,
        "missing_rows": n_missing,
        "gap_days": n_gap_days,
        "gap_pct": f"{gap_pct:.2%}",
        "date_range": f"{min_date} → {max_date}",
        "passes": passes,
    })

quality_df = pd.DataFrame(quality_report)
display(quality_df)

n_pass = quality_df["passes"].sum()
print(f"\n✅ {n_pass}/{len(quality_df)} symbols pass quality checks")
if n_pass < len(quality_df):
    print("⚠️ Some symbols failed — they may produce unreliable signals.")

# COMMAND ----------

# DBTITLE 1,Step 4: Indicators
# MAGIC %md
# MAGIC ## Step 4 — Technical Indicators
# MAGIC
# MAGIC Compute SMA, EMA, RSI, ATR, volume ratio, momentum, and relative strength for each stock using the `add_all_indicators` function.

# COMMAND ----------

# DBTITLE 1,Compute Indicators
indicator_data: dict[str, pd.DataFrame] = {}

for sym, df in price_data.items():
    enriched = add_all_indicators(
        df,
        sma_short=cfg.strategy.sma_short_period,
        sma_long=cfg.strategy.sma_long_period,
        rsi_period=cfg.strategy.rsi_period,
        atr_period=cfg.strategy.atr_period,
        benchmark_close=benchmark_close,
    )
    indicator_data[sym] = enriched

# Show a sample
sample_sym = symbols[0]
sample_df = indicator_data[sample_sym]
indicator_cols = [c for c in sample_df.columns if c.startswith("ind_")]
print(f"Sample indicators for {sample_sym} (last 5 bars):")
display(sample_df[["close", "volume"] + indicator_cols].tail(5))

print(f"\n✅ Indicators computed for {len(indicator_data)} symbols")
print(f"Indicator columns: {indicator_cols}")

# COMMAND ----------

# DBTITLE 1,Step 5: Signals
# MAGIC %md
# MAGIC ## Step 5 — Signal Generation
# MAGIC
# MAGIC Generate entry and exit signals using the baseline swing-trading strategy:
# MAGIC - **Entry**: close > SMA(long), SMA(short) > SMA(long), RSI was oversold then recovered, volume confirmation
# MAGIC - **Exit**: close < SMA(short), or stop-loss hit
# MAGIC
# MAGIC Signals use only data available at decision time (no look-ahead bias).

# COMMAND ----------

# DBTITLE 1,Generate Signals
# Build signal config from project config
sig_cfg = SignalConfig(
    sma_short_period=cfg.strategy.sma_short_period,
    sma_long_period=cfg.strategy.sma_long_period,
    rsi_period=cfg.strategy.rsi_period,
    rsi_oversold=cfg.strategy.rsi_oversold,
    rsi_renewed=cfg.strategy.rsi_renewed,
    atr_period=cfg.strategy.atr_period,
    stop_atr_multiple=cfg.risk.default_stop_atr_multiple,
    trailing_stop_atr_multiple=cfg.risk.trailing_stop_atr_multiple,
)

signal_data: dict[str, pd.DataFrame] = {}

for sym, df in price_data.items():
    sig_df = generate_all_signals(df, sig_cfg)
    signal_data[sym] = sig_df

# Count signals
total_entries = sum(df["signal_entry"].sum() for df in signal_data.values())
total_exits = sum(df["signal_exit"].sum() for df in signal_data.values())

print(f"Total entry signals: {total_entries}")
print(f"Total exit signals:  {total_exits}")

# Show signal summary per symbol
signal_summary = []
for sym, df in signal_data.items():
    n_entry = df["signal_entry"].sum()
    n_exit = df["signal_exit"].sum()
    last_signal = df[df["signal_type"] != SignalType.NONE]["signal_type"].tail(1)
    last_signal_str = last_signal.values[0].value if len(last_signal) > 0 else "none"
    signal_summary.append({
        "symbol": sym,
        "entry_signals": n_entry,
        "exit_signals": n_exit,
        "last_signal": last_signal_str,
    })

display(pd.DataFrame(signal_summary))

# COMMAND ----------

# DBTITLE 1,Step 6: Ranking
# MAGIC %md
# MAGIC ## Step 6 — Stock Ranking
# MAGIC
# MAGIC Rank the universe by a composite score combining momentum, relative strength, RSI position, and liquidity.

# COMMAND ----------

# DBTITLE 1,Rank Stocks
rank_cfg = RankingConfig()
rankings = rank_stocks(indicator_data, date=None, config=rank_cfg)

print("Stock Rankings (by composite score):")
display(rankings[["rank", "symbol", "score", "momentum", "rel_strength", "rsi", "avg_traded_value", "eligible"]])

# Select top-ranked eligible stocks for backtest focus
top_stocks = rankings[rankings["eligible"]]["symbol"].head(5).tolist()
print(f"\nTop 5 eligible stocks: {top_stocks}")

# COMMAND ----------

# DBTITLE 1,Step 7: Backtest
# MAGIC %md
# MAGIC ## Step 7 — Event-Driven Backtest
# MAGIC
# MAGIC Run the full event-driven backtest with realistic NSE transaction costs (brokerage, STT, exchange charges, GST, SEBI, stamp duty, slippage), ATR-based position sizing, stop-losses, trailing stops, and portfolio risk limits.

# COMMAND ----------

# DBTITLE 1,Run Backtest
result = run_backtest(price_data, config=cfg, signal_config=sig_cfg)

print(f"Initial capital:  ₹{result.initial_capital:,.2f}")
print(f"Final equity:     ₹{result.final_equity:,.2f}")
print(f"Total return:     {result.total_return:.2%}")
print(f"Number of trades:  {result.n_trades}")
print(f"Winning trades:    {result.n_winning}")
print(f"Losing trades:     {result.n_losing}")

# Show equity curve
print("\nEquity curve (first 5 and last 5 bars):")
equity_df = pd.DataFrame({"equity": result.equity_curve})
display(pd.concat([equity_df.head(5), equity_df.tail(5)]))

# Show trade table
trade_table = generate_trade_table(result.trades)
if not trade_table.empty:
    print(f"\nTrade history ({len(trade_table)} trades):")
    display(trade_table.head(10))
else:
    print("\nNo trades executed — signals may be rare with synthetic data.")

# COMMAND ----------

# DBTITLE 1,Step 8: Metrics
# MAGIC %md
# MAGIC ## Step 8 — Performance Metrics
# MAGIC
# MAGIC Calculate comprehensive risk-adjusted performance metrics: total return, CAGR, max drawdown, Sharpe ratio, Sortino ratio, win rate, expectancy, and more.

# COMMAND ----------

# DBTITLE 1,Calculate Metrics
metrics = calculate_all_metrics(
    equity_curve=result.equity_curve,
    daily_returns=result.daily_returns,
    trades=result.trades,
    initial_capital=result.initial_capital,
)

# Print the text report
report = generate_text_report(metrics, result.initial_capital, result.final_equity)
print(report)

# Also show as DataFrame
metrics_df = pd.DataFrame([
    {"metric": k, "value": v} for k, v in metrics.__dict__.items()
])
print("\nMetrics as DataFrame:")
display(metrics_df)

# COMMAND ----------

# DBTITLE 1,Step 9: Validation
# MAGIC %md
# MAGIC ## Step 9 — Robustness Validation
# MAGIC
# MAGIC ### Walk-Forward Analysis
# MAGIC Split the equity curve chronologically and compare in-sample vs out-of-sample returns.
# MAGIC
# MAGIC ### Monte Carlo Trade-Order Resampling
# MAGIC Reshuffle trade order 1,000 times to see the distribution of outcomes due to trade sequencing alone.

# COMMAND ----------

# DBTITLE 1,Walk-Forward & Monte Carlo
# --- Walk-Forward ---
wf_results = walk_forward(
    result.equity_curve,
    n_windows=cfg.backtest.walk_forward_windows,
    train_fraction=0.70,
)

wf_df = pd.DataFrame([
    {
        "window": r.window,
        "train_period": f"{r.train_start.date()} → {r.train_end.date()}",
        "test_period": f"{r.test_start.date()} → {r.test_end.date()}",
        "train_return": f"{r.train_return:.2%}",
        "test_return": f"{r.test_return:.2%}",
    }
    for r in wf_results
])
print("Walk-Forward Results:")
display(wf_df)

# --- Monte Carlo ---
if result.trades:
    trade_returns = np.array([t.pnl_pct for t in result.trades])
    mc_result = monte_carlo_trade_reshuffle(
        trade_returns,
        n_runs=cfg.backtest.monte_carlo_runs,
        seed=42,
    )
    print(f"\nMonte Carlo Trade-Order Resampling ({mc_result.n_runs} runs):")
    print(f"  Mean return:      {mc_result.mean_return:.2%}")
    print(f"  Median return:    {mc_result.median_return:.2%}")
    print(f"  Std of returns:  {mc_result.std_return:.2%}")
    print(f"  5th percentile:  {mc_result.percentile_5:.2%}")
    print(f"  95th percentile: {mc_result.percentile_95:.2%}")
    print(f"  % profitable:     {mc_result.pct_profitable:.1%}")
else:
    print("\n⚠️ No trades to run Monte Carlo on — signals may be rare with synthetic data.")

# COMMAND ----------

# DBTITLE 1,Step 10: Paper Trading
# MAGIC %md
# MAGIC ## Step 10 — Paper Trading Demo
# MAGIC
# MAGIC The paper-trading engine processes daily bars one at a time, generates signals, records orders, simulates fills, and tracks state — without submitting real broker orders.
# MAGIC
# MAGIC This demo runs the paper engine on the first 50 trading days of data.

# COMMAND ----------

# DBTITLE 1,Paper Trading Demo
paper_engine = PaperEngine(config=cfg, signal_config=sig_cfg)

# Collect all trading dates
all_dates = sorted(set().union(*[df.index for df in price_data.values()]))

# Run paper trading on first 50 days
n_days = min(50, len(all_dates))
print(f"Running paper trading on first {n_days} trading days...\n")

daily_results = []
for date in all_dates[:n_days]:
    summary = paper_engine.process_day(date, price_data)
    daily_results.append(summary)
    if summary["status"] == "drawdown_halt":
        print(f"  ⚠️ {date.date()}: Drawdown halt — {summary.get('drawdown', 0):.1%}")
        break
    if summary.get("new_orders", 0) > 0:
        print(f"  {date.date()}: {summary['new_orders']} new order(s), equity ₹{summary['total_equity']:,.0f}")

print(f"\n✅ Paper trading complete — {len(daily_results)} days processed")
print(f"Final status: {daily_results[-1]['status']}")

# Show equity snapshots
paper_equity_df = pd.DataFrame(daily_results)
print("\nPaper trading equity snapshots (last 5 days):")
display(paper_equity_df.tail(5))

# COMMAND ----------

# DBTITLE 1,Summary
# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC This notebook demonstrated the **full end-to-end pipeline** of the NSE Algorithmic Trading Research System:
# MAGIC
# MAGIC 1. **Data ingestion** — yfinance with synthetic fallback
# MAGIC 2. **Data quality** — duplicate, gap, and sufficiency checks
# MAGIC 3. **Technical indicators** — SMA, EMA, RSI, ATR, volume, momentum, relative strength
# MAGIC 4. **Signal generation** — trend-following pullback entry, trend-break/stop exit
# MAGIC 5. **Stock ranking** — composite momentum + relative strength + RSI + liquidity
# MAGIC 6. **Backtest** — event-driven, NSE transaction costs, ATR position sizing, trailing stops, drawdown controls
# MAGIC 7. **Performance metrics** — Sharpe, Sortino, CAGR, max drawdown, expectancy, win rate
# MAGIC 8. **Robustness** — walk-forward analysis, Monte Carlo trade-order resampling
# MAGIC 9. **Paper trading** — restart-safe simulated order execution with SQLite persistence
# MAGIC
# MAGIC ### Key Limitations
# MAGIC
# MAGIC - yfinance is an **unofficial** data source — results are NOT exchange-quality
# MAGIC - NSE transaction cost rates are **unverified** research defaults (`rates_verified: False`)
# MAGIC - Synthetic data produces **pipeline validation only**, not market analysis
# MAGIC - Historical performance does **not** guarantee future results
# MAGIC - This is a **research tool**, not investment advice
# MAGIC
# MAGIC ### Next Steps
# MAGIC
# MAGIC - Verify NSE cost rates against authoritative sources
# MAGIC - Expand the stock universe with proper historical membership data
# MAGIC - Run parameter sensitivity analysis to detect overfitting
# MAGIC - Test with different market regimes (bull, bear, sideways)
# MAGIC - Consider MLflow experiment tracking for systematic comparison