# Project assessment: profitability and backtest readiness

**Assessment date:** 2026-10-27
**Conclusion:** profitability is **unknown**. A small Yahoo Finance daily-data snapshot is available for exploratory checks, but it is explicitly not backtest-ready, and this review did not produce a valid historical backtest. The strategy must not be treated as profitable or ready for live orders.

## Evidence available in this checkout

- The project contains two legacy ML strategy notebooks exported as Python files, a newer reusable Python research package, a paper-trading engine, and a Streamlit dashboard.
- A timestamped Yahoo Finance snapshot is stored under `data/raw/yfinance/`. It covers a small set of symbols, contains data-quality warnings, and is marked `backtest_ready: false`; it is not a suitable historical universe for strategy evaluation. The optional `yfinance` package is installed in the local environment.
- The repository documentation itself labels the prior notebook returns unvalidated and identifies same-bar execution, test-set early stopping, incomplete costs, and current fundamentals used historically as invalidating issues.
- The workspace MCP configuration describes an optional Tapetide service. It requests a user token and sends requests and that token to Tapetide. No Tapetide tool or authenticated market-data connection was available in this session. Plugin discovery found no connected finance or market-data provider suitable for this assessment.
- Synthetic data is present only as a deterministic test utility. It cannot establish whether a trading signal has an edge.

## Findings that block a profitability claim

### Critical: historical results are not independently reproducible

The legacy strategies download market data on demand and rely on a current symbol universe. There is no persisted snapshot of raw bars, corporate-action policy, historical index membership, retrieval time, or source version. A universe formed from today's NIFTY constituents introduces survivorship bias if applied to earlier years.

The ML notebooks also use the final test partition as LightGBM's early-stopping evaluation set. That makes the test set part of model selection, so the reported test performance is not out of sample. Their backtests include same-close entries while signals use that close, and the model's historical fundamentals are fetched from current ticker metadata rather than point-in-time filings. The daily notebook and weekly notebook therefore cannot support a profitability conclusion.

### Addressed: core package had a time leak in liquidity ranking

`rank_stocks(data, date=...)` now restricts liquidity history to the latest bar available at the requested decision time. Regression tests cover future bars and non-session decision dates. This removes that identified leak; it does not validate the overall strategy.

### Addressed: configured portfolio exposure limit was not enforced

`run_backtest` now sizes new entries against remaining aggregate exposure and processes known open-gap exits before entries, with intraday stop checks after open fills. Regression tests cover exposure caps and prevent intraday stop proceeds from financing earlier open entries. Daily OHLC bars still cannot resolve the exact intraday ordering of a stop and other events, and this engine remains a research simulation.

### High: cost assumptions are not verified

The cost model is useful as a research approximation, but its defaults are explicitly unverified in the README and cost module. It needs current authoritative NSE/broker charges and realistic spread/market-impact assumptions by liquidity and order size. A flat slippage percentage is unlikely to model a broad NIFTY 500 universe well.

### Medium: benchmarks and candidate baselines are missing from the reusable backtest

The package calculates portfolio metrics, but the run path does not automatically compare against a matching total-return benchmark, cash baseline, or simple non-ML strategy under the same dates, universe, and costs. Ranking and indicator modules alone do not demonstrate an investment edge.

### Medium: fundamentals are not part of the reusable package

The current ranking inputs are technical. The legacy ML files fetch current fundamentals and news, which are not historical point-in-time records. The financial-analysis methodology in `docs/financial-analysis.md` correctly describes the data needed before adding fundamentals to a historical test.

## What the system does well

- The reusable backtest is event-driven and generally shifts close-derived entry/exit signals to the next session's open.
- Position sizing, cash accounting, stops, transaction-cost components, metrics, validation utilities, and a separate paper engine provide a stronger foundation than the legacy notebook simulators.
- The repository clearly warns that the existing legacy results are not validated and documents appropriate point-in-time data requirements.

These are software and process strengths, not evidence of profitability.

## Required work before an honest strategy verdict

1. Obtain licensed, reproducible daily OHLCV with delisted names, corporate actions, point-in-time index constituents, and a total-return benchmark. Save immutable raw inputs and a manifest for each run. The available small Yahoo snapshot is only for collection and visualization checks.
2. Replace legacy notebook performance claims with one execution model: signal after close, fill no earlier than next tradable open, explicit gap/limit handling, costs, liquidity constraints, and corporate-action-aware holdings.
3. Use chronological train/validation/test periods. Tune and early-stop only on training/validation windows; preserve a final untouched test period. Refit in walk-forward windows and include delisted constituents to reduce survivorship bias.
4. Compare net results with buy-and-hold total return, cash, and simple technical baselines. Report CAGR, drawdown, volatility, Sharpe/Sortino, turnover, exposure, trade count, concentration, cost sensitivity, and results by period and liquidity bucket.
5. Freeze strategy rules before the untouched test, then paper trade prospectively without retuning against that record.

Do not optimize parameters to the historical sample until the data and execution model are sound; that can improve the backtest while making live performance worse. No change can guarantee profitability.
