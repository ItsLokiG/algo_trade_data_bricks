# NSE Algorithmic Trading Research System

A portable, testable algorithmic trading research platform for **NSE cash-equity swing trading** on daily and weekly bars.

**Profitability is a hypothesis to test, not an assumption or guarantee.**

## Current Phase

**Phases 1–9 complete (197 tests). Two active swing trading strategies deployed in paper trading.**

See the [development workflow](#development-phases) below for the full roadmap.

## Project Structure

```
algo_trade_data_bricks/
├── Advanced ML Stock Selection NIFTY 200.py   # Weekly swing strategy (4-week hold)
├── Daily Swing 10-Day NIFTY 500.py            # Daily swing strategy (10-day hold)
├── src/
│   └── nse_algo/
│       ├── __init__.py             # Package init, v0.1.0
│       ├── config.py               # Central configuration (all parameters)
│       ├── data/
│       │   ├── __init__.py
│       │   ├── providers.py         # DataProvider interface, YFinance, Synthetic
│       │   └── validation.py        # Data-quality checks (OHLCV validation)
│       ├── features/
│       │   ├── __init__.py
│       │   └── indicators.py       # SMA, EMA, RSI, ATR, volume, momentum
│       ├── strategy/
│       │   ├── __init__.py
│       │   ├── signals.py          # Entry/exit signal generation
│       │   └── ranking.py           # Stock ranking by composite score
│       ├── backtest/
│       │   ├── __init__.py
│       │   ├── costs.py             # NSE transaction cost model
│       │   ├── engine.py            # Event-driven backtester
│       │   ├── metrics.py           # Sharpe, Sortino, drawdown, CAGR, etc.
│       │   └── validation.py        # Walk-forward, Monte Carlo, data splits
│       ├── paper/
│       │   ├── __init__.py
│       │   ├── database.py          # SQLite persistence (orders, fills, equity)
│       │   └── engine.py            # Paper-trading engine (no real orders)
│       ├── reporting/
│       │   ├── __init__.py
│       │   └── reports.py           # Text reports, trade tables, summaries
│       └── monitoring/
│           ├── __init__.py
│           └── checks.py            # Drawdown, exposure, freshness checks
├── tests/
│   ├── test_config.py              # 32 tests — config validation
│   ├── test_validation.py          # 25 tests — providers + data quality
│   ├── test_indicators.py          # 26 tests — technical indicators
│   ├── test_signals.py             # 21 tests — signals + ranking
│   ├── test_backtest.py            # 27 tests — costs + backtest engine
│   ├── test_metrics.py             # 36 tests — metrics + robustness
│   ├── test_paper.py               # 17 tests — paper DB + engine
│   └── test_reporting.py           # 13 tests — reporting + monitoring
├── README.md
├── .gitignore
└── requirements.txt
```

## Environment Inspection (Phase 1)

| Component | Status |
| --- | --- |
| Python | 3.12.3 |
| pandas | 2.3.3 |
| numpy | 2.3.4 |
| pytest | 8.4.2 |
| pyspark | 4.3.0.dev0 (Spark 4.2.0 via Spark Connect) |
| delta | 4.4.0 |
| mlflow | 3.12.0 |
| matplotlib | 3.10.6 |
| yfinance | NOT INSTALLED (Phase 2 will verify availability) |
| duckdb | NOT INSTALLED (future local portability) |

### Unity Catalog

| Catalog | Schemas |
| --- | --- |
| `dev_catalog` | bronze, silver, gold, default |
| `medallian_catalog` | default, table_for_all |

Schema creation permission confirmed on both catalogs. The project will use
`dev_catalog.algo` for trading tables.

## Configuration

All parameters are defined in `src/nse_algo/config.py` using frozen dataclasses
with validation. Key defaults:

| Parameter | Default | Notes |
| --- | --- | --- |
| Initial capital | ₹100,000 | Research default, configurable |
| Leverage | Disabled | Hard-rejected by validation |
| Short selling | Disabled | Hard-rejected by validation |
| Risk per trade | 0.5% | Planned risk, not guaranteed max loss |
| Max risk per trade | 1.0% | Configurable upper limit |
| Max positions | 10 | |
| Max portfolio exposure | 100% | No leverage |
| Max drawdown | 20% | Halts new entries |
| Brokerage | 0.03% | ₹20 cap/trade — **NOT VERIFIED** |
| STT | 0.1% | **NOT VERIFIED** |
| Slippage | 0.05% per side | Conservative assumption |

> **Cost rates are research defaults and MUST be verified against authoritative
> sources before any live-use calculation.**

## Active Trading Strategies

### 1. Weekly Swing — Advanced ML Stock Selection NIFTY 200

| Metric | Weekly Swing | NIFTY 50 |
|---|---|---|
| Total Return (3.5y) | 96.8% | 25.7% |
| CAGR | 21.6% | 6.8% |
| Sharpe | 1.60 | 0.61 |
| Max Drawdown | -12.3% | -15.5% |
| Win Rate | 61.5% | — |
| Profit Factor | 2.16 | — |
| Trades | 504 | — |
| Avg Hold | 3.8 weeks | — |

130 NIFTY 200 stocks, 53 features (technical + Google News RSS sentiment + Screener.in fundamentals), LightGBM, NIFTY regime filter (weekly SMA40), ATR stops 1.5x, targets 3x.

### 2. Daily Swing — 10-Day Max Holding NIFTY 500

| Metric | Daily Swing | NIFTY 50 |
|---|---|---|
| Total Return (3.4y) | 71.4% | 27.8% |
| CAGR | 17.1% | 7.4% |
| Sharpe | 1.39 | 0.62 |
| Max Drawdown | -13.5% | -15.8% |
| Win Rate | 51.5% | — |
| Profit Factor | 1.32 | — |
| Trades | 1,204 | — |
| Avg Hold | 6.1 days | — |

224 stocks (large + midcap), 37 features, LightGBM, 10-day triple-barrier, NIFTY regime (daily SMA50), ATR stops 1.5x, targets 2x, daily rebalancing.

### Paper Trading

Both strategies have paper trading with JSON state persistence (gitignored):
- Weekly: run every Friday
- Daily: run daily after market close

## Running Tests

### In Databricks

```python
import sys
sys.path.insert(0, "/Workspace/Repos/lokigovindasamy@gmail.com/algo_trade_data_bricks/src")

import pytest
pytest.main(["/Workspace/Repos/lokigovindasamy@gmail.com/algo_trade_data_bricks/tests", "-v"])
```

### Locally (VS Code)

```bash
cd algo_trade_data_bricks
pip install -r requirements.txt
python -m pytest tests/ -v
```

## Development Phases

| Phase | Description | Status |
| --- | --- | --- |
| 1 | Environment, config, structure, initial tests | ✅ Complete |
| 2 | Market data ingestion, caching, quality checks | ✅ Complete |
| 3 | Technical indicators and feature engineering | ✅ Complete |
| 4 | Strategy rules and stock ranking | ✅ Complete |
| 5 | Backtester, execution simulation, accounting | ✅ Complete |
| 6 | Performance metrics, walk-forward, robustness | ✅ Complete |
| 7 | Optional point-in-time fundamentals | ⏭️ Skipped (optional) |
| 8 | Paper-trading engine (no real orders) | ✅ Complete |
| 9 | Reporting, monitoring, scheduling | ✅ Complete |
| 10 | GitHub migration and VS Code execution | ✅ See below |
| 11 | Optional MLflow and ML enhancements | ⏭️ Future |

## VS Code Migration Guide

### 1. Export to GitHub

In Databricks, use the Repos folder to commit and push:

```bash
# From Databricks Repos, commit all changes and push to GitHub
# Or use the Databricks CLI:
databricks workspace export-dir /Repos/<email>/algo_trade_data_bricks ./algo_trade_data_bricks
cd algo_trade_data_bricks
git add .
git commit -m "Phases 1-9 complete"
git push
```

### 2. Clone in VS Code (Windows 11)

```bash
git clone https://github.com/<username>/algo_trade_data_bricks.git
cd algo_trade_data_bricks
code .
```

### 3. Create Virtual Environment

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Run Tests

```bash
python -m pytest tests/ -v
```

### 5. Run the Backtester

```python
from nse_algo.config import ProjectConfig
from nse_algo.data.providers import SyntheticProvider
from nse_algo.backtest.engine import run_backtest

# Generate synthetic data for testing
p = SyntheticProvider(seed=42)
data = {"STK0": p.fetch_ohlcv("STK0", "2020-01-01", "2023-12-31")}

# Run backtest
result = run_backtest(data, config=ProjectConfig())
print(f"Trades: {result.n_trades}")
print(f"Final equity: ₹{result.final_equity:,.2f}")
```

### 6. Run the Paper Trading Engine

```python
from nse_algo.config import ProjectConfig
from nse_algo.paper.engine import PaperEngine
from nse_algo.paper.database import PaperDatabase
from nse_algo.data.providers import SyntheticProvider

db = PaperDatabase("paper_trading.db")
engine = PaperEngine(config=ProjectConfig(), db=db)

p = SyntheticProvider(seed=42)
data = {"STK0": p.fetch_ohlcv("STK0", "2020-01-01", "2023-12-31")}

for date in data["STK0"].index[200:210]:
    result = engine.process_day(date, data)
    print(f"{date.date()}: {result['status']}, equity: ₹{result.get('total_equity', 0):,.2f}")

db.close()
```

### 7. Local Storage Alternatives

- **Delta tables → Parquet**: Use `df.to_parquet()` and `pd.read_parquet()` instead of Delta Lake.
- **DuckDB**: Install `duckdb` for local analytical queries on Parquet files.
- **SQLite**: Already used for paper-trading state — no changes needed.

## Key Constraints

- **No real broker orders** — simulation and paper trading only.
- **No leverage, short selling, futures, or options** in the initial scope.
- **No hard-coded credentials or machine-specific paths.**
- **Core strategy logic is pure Python** (Pandas/NumPy) — no Spark dependency.
- **PySpark/Delta Lake** used only for data engineering where beneficial.

## Disclaimer

This is a research system. Historical performance does not guarantee future
results. All execution is simulated. Never treat AI-generated signals or
backtest results as proof of future profitability.
