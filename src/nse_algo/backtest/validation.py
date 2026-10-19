"""
Robustness and anti-overfitting validation.

Implements:
  - Chronological data splits (train / validation / test).
  - Walk-forward testing.
  - Monte Carlo trade-order resampling.
  - Parameter sensitivity analysis.

These tests do NOT prove future profitability.  They help detect
overfitting, look-ahead bias, and unstable parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DataSplits:
    """Chronological train/validation/test splits."""
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame

    @property
    def train_fraction(self) -> float:
        total = len(self.train) + len(self.validation) + len(self.test)
        return len(self.train) / total if total > 0 else 0.0

    @property
    def test_fraction(self) -> float:
        total = len(self.train) + len(self.validation) + len(self.test)
        return len(self.test) / total if total > 0 else 0.0


def split_data_chronologically(
    df: pd.DataFrame,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
) -> DataSplits:
    """Split data chronologically into train, validation, test.

    The test set is the remaining fraction after train + validation.
    The test set is NEVER used for optimisation.
    """
    if not 0 < train_fraction < 1:
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}")
    if not 0 < validation_fraction < 1:
        raise ValueError(f"validation_fraction must be in (0, 1), got {validation_fraction}")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train_fraction + validation_fraction must be < 1.0")

    n = len(df)
    train_end = int(n * train_fraction)
    val_end = int(n * (train_fraction + validation_fraction))

    return DataSplits(
        train=df.iloc[:train_end],
        validation=df.iloc[train_end:val_end],
        test=df.iloc[val_end:],
    )


@dataclass(frozen=True)
class WalkForwardResult:
    """Result of one walk-forward window."""
    window: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_return: float
    test_return: float


def walk_forward(
    df: pd.DataFrame,
    n_windows: int = 5,
    train_fraction: float = 0.70,
    run_func: Callable[[pd.DataFrame], float] | None = None,
) -> list[WalkForwardResult]:
    """Walk-forward analysis.

    For each window, trains on a rolling period and tests on the next.
    Compares in-sample vs out-of-sample returns.

    Parameters
    ----------
    df : DataFrame
        Full dataset (e.g., equity curve or price data).
    n_windows : int
        Number of walk-forward windows.
    train_fraction : float
        Fraction of each window used for training.
    run_func : callable, optional
        Function that takes a DataFrame and returns a performance metric
        (e.g., total return). If None, uses simple total return.
    """
    if n_windows < 1:
        raise ValueError("n_windows must be >= 1")
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be in (0, 1)")

    if run_func is None:
        def run_func(d: pd.DataFrame) -> float:
            if len(d) < 2:
                return 0.0
            return float(d.iloc[-1] / d.iloc[0] - 1.0) if isinstance(d, pd.Series) else 0.0

    n = len(df)
    window_size = n // n_windows
    results: list[WalkForwardResult] = []

    for w in range(n_windows):
        start = w * window_size
        end = min(start + window_size, n)
        if end - start < 10:
            continue

        train_end = start + int((end - start) * train_fraction)
        if train_end >= end:
            continue

        train_data = df.iloc[start:train_end]
        test_data = df.iloc[train_end:end]

        train_ret = run_func(train_data)
        test_ret = run_func(test_data)

        results.append(WalkForwardResult(
            window=w,
            train_start=df.index[start],
            train_end=df.index[train_end - 1],
            test_start=df.index[train_end],
            test_end=df.index[end - 1],
            train_return=train_ret,
            test_return=test_ret,
        ))

    return results


@dataclass(frozen=True)
class MonteCarloResult:
    """Result of Monte Carlo trade-order resampling."""
    n_runs: int
    mean_return: float
    median_return: float
    std_return: float
    percentile_5: float
    percentile_95: float
    pct_profitable: float


def monte_carlo_trade_reshuffle(
    trade_returns: np.ndarray | pd.Series,
    n_runs: int = 1000,
    seed: int = 42,
) -> MonteCarloResult:
    """Monte Carlo analysis by reshuffling trade order.

    Reshuffles the order of trade returns and recalculates the
    cumulative return for each run.  This shows the distribution of
    outcomes due to trade ordering alone.

    LIMITATION: This does not account for regime changes, correlation
    between trades, or position sizing effects.  It is a simplified
    test of trade-order sensitivity, not a full risk model.

    Parameters
    ----------
    trade_returns : array-like
        Array of per-trade returns (not P&L in rupees — use pct returns).
    n_runs : int
        Number of Monte Carlo runs.
    seed : int
        Random seed for reproducibility.
    """
    if n_runs < 1:
        raise ValueError("n_runs must be >= 1")

    returns = np.asarray(trade_returns, dtype=float)
    if len(returns) == 0:
        return MonteCarloResult(0, 0, 0, 0, 0, 0, 0)

    rng = np.random.default_rng(seed)
    cumulative_returns = np.empty(n_runs)

    for i in range(n_runs):
        shuffled = rng.permutation(returns)
        equity = np.cumprod(1 + shuffled)
        cumulative_returns[i] = equity[-1] - 1.0

    return MonteCarloResult(
        n_runs=n_runs,
        mean_return=float(np.mean(cumulative_returns)),
        median_return=float(np.median(cumulative_returns)),
        std_return=float(np.std(cumulative_returns)),
        percentile_5=float(np.percentile(cumulative_returns, 5)),
        percentile_95=float(np.percentile(cumulative_returns, 95)),
        pct_profitable=float(np.mean(cumulative_returns > 0)),
    )


def parameter_sensitivity(
    data: dict[str, pd.DataFrame],
    param_name: str,
    param_values: list,
    run_func: Callable[..., Any],
) -> pd.DataFrame:
    """Run the backtest with different parameter values.

    Returns a DataFrame with one row per parameter value and the
    corresponding performance metric.  Helps detect overfitting to
    a specific parameter value.

    Parameters
    ----------
    data : dict[str, DataFrame]
        Input data for the backtest.
    param_name : str
        Name of the parameter to vary (passed as kwarg to run_func).
    param_values : list
        List of parameter values to test.
    run_func : callable
        Function that takes data and **kwargs and returns a metric (float).
    """
    rows = []
    for val in param_values:
        metric = run_func(data, **{param_name: val})
        rows.append({"param_value": val, "metric": metric})
    return pd.DataFrame(rows)
