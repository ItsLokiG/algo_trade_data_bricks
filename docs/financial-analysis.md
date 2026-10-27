# Financial Analysis for the Research System

## Current capability

The current Python package does not ingest or score point-in-time financial
statements. Its stock-ranking inputs are technical. An MCP financial summary
or a current screener result is not a historical fundamental dataset and must
not be fed into a historical backtest as if it were known at that time.

## Analysis workflow

1. **Start from primary, dated evidence.** Prefer audited annual reports and
   exchange-filed quarterly results. Record the company identifier, reporting
   period, publication timestamp, source URL or document ID, retrieval time,
   currency, units, and whether values are consolidated or standalone.
2. **Respect information availability.** A result is usable only after its
   actual public filing timestamp. Keep `period_end` separate from
   `published_at` and `available_at`; use the earliest time the strategy could
   reasonably have acted on the information. Never backfill restated or
   corrected values into earlier dates without preserving the original
   vintage.
3. **Check completeness and comparability.** Reconcile units, fiscal periods,
   corporate actions, consolidated versus standalone accounts, exceptional
   items, and missing values. Flag restatements and auditor qualifications.
   Do not silently substitute a current ratio for a missing historical value.
4. **Assess business quality, growth, balance sheet, and valuation separately.**
   Candidate measures include:
   - quality: return on invested capital, operating margin, and cash flow from
     operations relative to reported profit;
   - growth: multi-year sales and earnings growth, consistency, and dilution;
   - balance sheet: net debt, interest coverage, liquidity, and cash conversion;
   - valuation: sector-aware price/earnings, enterprise-value multiples, and
     price/book where economically meaningful;
   - governance risks: auditor qualifications, pledged promoter holdings,
     related-party disclosures, and material regulatory or litigation items.
5. **Compare like with like.** Use sector-aware peer groups and historical
   sector context. Do not rank banks using industrial-company leverage ratios
   or treat a single valuation multiple as a complete intrinsic-value model.
   Explicitly document exclusions and the reason for every factor.
6. **Keep fundamental and technical hypotheses distinct.** Fundamentals can
   define quality, solvency, or valuation eligibility; technical indicators
   can define timing. Freeze the factor definitions and weights before an
   out-of-sample evaluation. Do not tune them against the final test period.
7. **Evaluate after realistic implementation costs.** Use point-in-time
   constituents, delisted securities where available, corporate actions,
   filing lags, next-tradable-session execution, liquidity constraints,
   slippage, brokerage, taxes, and exchange charges. Compare against an
   appropriate total-return benchmark and simple baselines.
8. **Report uncertainty, not just a headline return.** Include net and gross
   returns, drawdown, volatility, turnover, exposure, concentration, trade
   counts, data coverage, missingness, sensitivity to costs, walk-forward
   results, and the untouched test period. A strong backtest is a reason to
   continue prospective paper trading, not a promise of future profit.

## Use of MCPs and AI assistants

An MCP can help locate filings or retrieve vendor ratios, but its output is
external data—not independent verification. Preserve raw responses and source
references, reconcile important figures with primary filings, and record
provider/version/retrieval time. Confirm licensing and point-in-time semantics
before storing or backtesting the data. Treat model-generated conclusions as
hypotheses to check, never as audited facts or trade instructions.

The optional Tapetide connection is not currently integrated with the Python
provider interface or backtester. No results from it have been validated by
this repository.

## Promotion gates

Do not describe a strategy as reliable or profitable until its data and code
are reproducible, its accounting and execution assumptions are independently
reviewed, its out-of-sample results survive robustness checks, and prospective
paper-trading results are recorded without changing the model in response to
the test. Even then, results do not guarantee future performance.
