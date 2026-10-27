"""
Tests for data providers and data-quality validation.

Uses the SyntheticProvider to generate deterministic test data without
needing internet access or yfinance.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nse_algo.data.providers import (
    DataProvider,
    ProviderError,
    SyntheticProvider,
    YFinanceProvider,
    get_provider,
)
from nse_algo.data.validation import (
    DataQualityReport,
    validate_ohlcv,
    validate_universe,
)


# ──────────────────────────────────────────────
#  Provider tests
# ──────────────────────────────────────────────

class TestSyntheticProvider:
    def test_returns_dataframe(self) -> None:
        p = SyntheticProvider(seed=42)
        df = p.fetch_ohlcv("TEST", "2023-01-01", "2023-12-31")
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_has_required_columns(self) -> None:
        p = SyntheticProvider(seed=42)
        df = p.fetch_ohlcv("TEST", "2023-01-01", "2023-12-31")
        for col in ("open", "high", "low", "close", "volume"):
            assert col in df.columns

    def test_ohlc_consistency(self) -> None:
        """high >= max(open, close), low <= min(open, close), high >= low."""
        p = SyntheticProvider(seed=42)
        df = p.fetch_ohlcv("TEST", "2023-01-01", "2023-12-31")
        assert (df["high"] >= df[["open", "close"]].max(axis=1)).all()
        assert (df["low"] <= df[["open", "close"]].min(axis=1)).all()
        assert (df["high"] >= df["low"]).all()

    def test_deterministic_with_same_seed(self) -> None:
        p1 = SyntheticProvider(seed=42)
        p2 = SyntheticProvider(seed=42)
        df1 = p1.fetch_ohlcv("TEST", "2023-01-01", "2023-12-31")
        df2 = p2.fetch_ohlcv("TEST", "2023-01-01", "2023-12-31")
        pd.testing.assert_frame_equal(df1, df2)

    def test_deterministic_across_processes(self) -> None:
        script = (
            "from nse_algo.data.providers import SyntheticProvider; "
            "print(SyntheticProvider(seed=42).fetch_ohlcv("
            "'TEST', '2023-01-01', '2023-01-10').to_json())"
        )
        env = os.environ.copy()
        source_root = str(Path(__file__).resolve().parents[1] / "src")
        existing_path = env.get("PYTHONPATH")
        env["PYTHONPATH"] = os.pathsep.join(
            part for part in (source_root, existing_path) if part
        )

        outputs = [
            subprocess.run(
                [sys.executable, "-c", script],
                check=True,
                capture_output=True,
                text=True,
                env=env,
            ).stdout
            for _ in range(2)
        ]

        assert outputs[0] == outputs[1]

    def test_different_symbols_different_data(self) -> None:
        p = SyntheticProvider(seed=42)
        df1 = p.fetch_ohlcv("AAA", "2023-01-01", "2023-12-31")
        df2 = p.fetch_ohlcv("BBB", "2023-01-01", "2023-12-31")
        assert not df1["close"].equals(df2["close"])

    def test_prices_positive(self) -> None:
        p = SyntheticProvider(seed=42)
        df = p.fetch_ohlcv("TEST", "2023-01-01", "2023-12-31")
        assert (df[["open", "high", "low", "close"]] > 0).all().all()

    def test_volume_positive(self) -> None:
        p = SyntheticProvider(seed=42)
        df = p.fetch_ohlcv("TEST", "2023-01-01", "2023-12-31")
        assert (df["volume"] > 0).all()

    def test_is_not_official(self) -> None:
        p = SyntheticProvider(seed=42)
        assert p.is_official is False
        assert p.name == "synthetic"

    def test_empty_date_range_raises(self) -> None:
        p = SyntheticProvider(seed=42)
        with pytest.raises(ProviderError):
            p.fetch_ohlcv("TEST", "2023-01-01", "2023-01-01")


class TestYFinanceProvider:
    def test_not_official(self) -> None:
        p = YFinanceProvider()
        assert p.is_official is False
        assert p.name == "yfinance"

    def test_unavailable_raises_on_fetch(self) -> None:
        p = YFinanceProvider()
        if not p.is_available():
            with pytest.raises(ProviderError, match="not installed"):
                p.fetch_ohlcv("RELIANCE", "2023-01-01", "2023-12-31")


class TestProviderFactory:
    def test_get_synthetic(self) -> None:
        p = get_provider("synthetic", seed=99)
        assert isinstance(p, SyntheticProvider)

    def test_get_yfinance(self) -> None:
        p = get_provider("yfinance")
        assert isinstance(p, YFinanceProvider)

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown provider"):
            get_provider("bloomberg")


# ──────────────────────────────────────────────
#  Validation tests
# ──────────────────────────────────────────────

class TestValidateOhlcv:
    def _good_data(self, n: int = 300) -> pd.DataFrame:
        """Generate clean OHLCV data."""
        p = SyntheticProvider(seed=42)
        df = p.fetch_ohlcv("TEST", "2022-01-01", "2023-12-31")
        return df

    def test_good_data_passes(self) -> None:
        df = self._good_data()
        report = validate_ohlcv(df, "TEST", min_trading_days=250)
        assert report.passed
        assert report.duplicate_dates == 0
        assert report.negative_prices == 0
        assert report.invalid_ohlc == 0
        assert report.nan_count == 0

    def test_missing_columns_fails(self) -> None:
        df = pd.DataFrame({"open": [1], "close": [2]})
        report = validate_ohlcv(df, "BAD")
        assert not report.passed
        assert any("Missing columns" in i for i in report.issues)

    def test_duplicate_dates_detected(self) -> None:
        df = self._good_data()
        # Duplicate the first row
        df = pd.concat([df, df.iloc[[0]]])
        df = df.sort_index()
        report = validate_ohlcv(df, "TEST", min_trading_days=250)
        assert report.duplicate_dates == 1
        assert not report.passed

    def test_negative_prices_detected(self) -> None:
        df = self._good_data()
        df.iloc[0, df.columns.get_loc("close")] = -1.0
        report = validate_ohlcv(df, "TEST", min_trading_days=250)
        assert report.negative_prices > 0
        assert not report.passed

    def test_invalid_ohlc_detected(self) -> None:
        df = self._good_data()
        # Make high < low on one row
        df.iloc[0, df.columns.get_loc("high")] = 1.0
        df.iloc[0, df.columns.get_loc("low")] = 100.0
        report = validate_ohlcv(df, "TEST", min_trading_days=250)
        assert report.invalid_ohlc > 0
        assert not report.passed

    def test_nan_detected(self) -> None:
        df = self._good_data()
        df.iloc[0, df.columns.get_loc("close")] = np.nan
        report = validate_ohlcv(df, "TEST", min_trading_days=250)
        assert report.nan_count == 1
        assert not report.passed

    def test_too_few_rows_fails(self) -> None:
        df = self._good_data()
        report = validate_ohlcv(df.iloc[:10], "TEST", min_trading_days=250)
        assert not report.passed
        assert any("minimum" in i for i in report.issues)

    def test_suspicious_move_flagged(self) -> None:
        df = self._good_data()
        # Create a 50% daily move
        idx = 10
        df.iloc[idx, df.columns.get_loc("open")] = 100.0
        df.iloc[idx, df.columns.get_loc("close")] = 150.0
        df.iloc[idx, df.columns.get_loc("high")] = 155.0
        df.iloc[idx, df.columns.get_loc("low")] = 99.0
        report = validate_ohlcv(df, "TEST", min_trading_days=250, max_daily_move_pct=0.20)
        assert report.suspicious_moves >= 1

    def test_report_has_symbol(self) -> None:
        df = self._good_data()
        report = validate_ohlcv(df, "MYSYMBOL", min_trading_days=250)
        assert report.symbol == "MYSYMBOL"

    def test_report_str(self) -> None:
        df = self._good_data()
        report = validate_ohlcv(df, "TEST", min_trading_days=250)
        s = str(report)
        assert "TEST" in s
        assert "PASS" in s or "FAIL" in s


class TestValidateUniverse:
    def test_multiple_symbols(self) -> None:
        p = SyntheticProvider(seed=42)
        data = {
            "AAA": p.fetch_ohlcv("AAA", "2022-01-01", "2023-12-31"),
            "BBB": p.fetch_ohlcv("BBB", "2022-01-01", "2023-12-31"),
        }
        reports = validate_universe(data, min_trading_days=250)
        assert len(reports) == 2
        assert all(isinstance(r, DataQualityReport) for r in reports.values())
        assert all(r.passed for r in reports.values())
