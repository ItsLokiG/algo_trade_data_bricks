# NSE Algorithmic Trading Research System

A portable, testable algorithmic trading research platform for **NSE cash-equity swing trading** on daily and weekly bars.

**Profitability is a hypothesis to test, not an assumption or guarantee.**

## Current Phase

**Research prototype:** core modules and 210 tests are available. Strategy profitability has not been established; the legacy strategy notebooks and their historical performance figures are not validated for trading decisions.

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
│       │   ├── collection.py        # Timestamped download snapshots and charts
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
│   ├── test_backtest.py            # 29 tests — costs + backtest engine
│   ├── test_metrics.py             # 36 tests — metrics + robustness
│   ├── test_paper.py               # 19 tests — paper DB + engine
│   └── test_reporting.py           # 13 tests — reporting + monitoring
├── README.md
├── SECURITY.md
├── collect_daily_data.py            # Free exploratory daily-data collector
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
| yfinance | Optional; installed locally as 1.7.0 on 2026-10-27 |
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

## Candidate Strategies (Unverified)

The two Databricks notebooks are research prototypes, not verified profitable
strategies. Their previously displayed headline returns have been removed:
the backtests use same-bar closing prices for some decisions, the ML test
period is used for early stopping, the weekly model applies current
fundamentals to historical observations, and trading costs are not fully
modeled. Those results should not be used as evidence of an edge.

### Weekly Swing — Advanced ML Stock Selection NIFTY 200

Research design: weekly stock selection, technical/cross-sectional features,
LightGBM, a NIFTY regime filter, and ATR-based exits. The notebook requires
point-in-time data and an execution-aware out-of-sample rebuild before
performance can be assessed.

### Daily Swing — 10-Day Max Holding NIFTY 500

Research design: daily stock selection, technical/cross-sectional features,
LightGBM, a NIFTY regime filter, and ATR-based exits. The notebook requires
execution-aware signals, leakage-safe model selection, and full transaction
cost modeling before its backtest can support a conclusion.

### Paper Trading

The notebook/dashboard simulations do not submit broker orders. The reusable
Python paper engine is separate and simulates close-based signals at the next
session's open. It processes open-gap exits before entries, intraday stop
triggers after open fills, and constrains aggregate open market value to the
configured portfolio-exposure cap. These are simulation conventions, not a
guarantee of broker settlement or execution behavior. Its accounting and
execution behavior is covered by tests, but this is not evidence that either
candidate strategy is profitable.

### Market-data MCP (optional)

The VS Code workspace includes an optional Tapetide MCP connection in
`.vscode/mcp.json`. It uses the pinned Python bridge (`tapetide-mcp` 1.0.1)
through `uvx`, and can provide NSE/BSE quotes, financials, technical
screening, daily/weekly price history, and point-in-time/corporate-action
lookup tools to an MCP-compatible assistant. On first use, VS Code prompts
for a Tapetide token; it is not stored in this repository. The configured
MCP package sends requests and that token to Tapetide's remote service.

This is a third-party research/data service, not an official NSE/BSE feed.
The pinned bridge forwards MCP requests to its remote service, so queries
and the token are sent to Tapetide. Do not send brokerage credentials,
personal data, or strategy information you consider confidential. Its
advertised historical coverage and point-in-time tools still need independent
validation before use in performance claims. The MCP connection is for
assistant research only; it is **not** wired as a Python backtest provider,
does not submit orders, and does not by itself establish profitability.
For reproducible backtests, persist the exact historical data used, its
retrieval time, adjustment factors, instrument identifiers, and historical
index membership alongside each run. Review the provider's current data
licensing, methodology, and usage limits before relying on it.

Use only the optional MCP you have explicitly reviewed; installing multiple
Claude/Codex MCPs does not improve the evidence quality. For authorized
historical OHLCV, prefer a licensed exchange or broker feed and independently
validate its corrections and adjustment policy. For fundamental inputs, use
dated exchange filings or a licensed point-in-time dataset. See
[Financial-analysis methodology](docs/financial-analysis.md) and
[Security policy](SECURITY.md).

The bridge is not a substitute for the local backtest engine. Do not treat
assistant-generated calculations, MCP screening output, or vendor claims as
audited financial statements or validated strategy results.

### Free daily-data collection (exploratory)

The local collector downloads daily bars for a small NSE sample and NIFTY 50
from Yahoo Finance, saves timestamped CSV snapshots plus SHA-256 manifests,
validates basic OHLCV quality, and plots close/adjusted-close, volume, 50/200
session averages, and RSI. Yahoo Finance is unofficial; this is for pipeline
and visualization checks, not investment decisions or a backtest-ready data
set. The collector preserves provider-returned `Adj Close`, dividends, and
splits and does not apply its own adjustment.

Install the optional data dependency if it is missing:

```powershell
python -m pip install yfinance
```

From the repository root, collect the default sample from 2019 onward:

```powershell
python .\collect_daily_data.py
```

The default symbols are RELIANCE, TCS, HDFCBANK, INFY, ITC, and NIFTY 50. To
choose other symbols or dates:

```powershell
python .\collect_daily_data.py --symbols RELIANCE.NS TCS.NS ^NSEI --start 2020-01-01
```

PowerShell treats `^` literally in this argument. Each run creates a new
snapshot under `data/raw/yfinance/` with a `manifest.json` and a `charts/`
subdirectory. These generated datasets are ignored by Git. The manifest
records provider, retrieval time, requested range, validation results,
adjustment policy, and hashes. The `end` argument, when specified, is
exclusive.

Do not pass these files directly to the backtester yet: the backtester does
not model split/dividend cashflows, and the Yahoo history may be adjusted or
revised. The manifest deliberately marks each snapshot as not backtest-ready.
The approximate missing-session check uses weekdays rather than the NSE
holiday calendar.

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
