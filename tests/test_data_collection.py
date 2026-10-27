"""Tests for timestamped daily-data collection and chart generation."""

import json

import pandas as pd
import pytest

from nse_algo.data.collection import collect_daily_snapshot
from nse_algo.data.providers import DataProvider, ProviderError, SyntheticProvider


class FailingProvider(DataProvider):
    @property
    def name(self) -> str:
        return "failing-test-provider"

    @property
    def is_official(self) -> bool:
        return False

    def fetch_ohlcv(
        self, symbol: str, start_date: str, end_date: str | None = None
    ) -> pd.DataFrame:
        raise ProviderError(f"unavailable: {symbol}")


class EmptyProvider(DataProvider):
    @property
    def name(self) -> str:
        return "empty-test-provider"

    @property
    def is_official(self) -> bool:
        return False

    def fetch_ohlcv(
        self, symbol: str, start_date: str, end_date: str | None = None
    ) -> pd.DataFrame:
        return pd.DataFrame()


class TestCollectDailySnapshot:
    def test_saves_csv_manifest_and_chart(self, tmp_path) -> None:
        result = collect_daily_snapshot(
            ["TEST"],
            "2022-01-03",
            tmp_path,
            end_date="2024-01-01",
            provider=SyntheticProvider(seed=42),
        )

        assert result.successful
        outcome = result.symbols[0]
        csv_path = result.snapshot_dir / outcome["csv"]
        chart_path = result.snapshot_dir / outcome["chart"]
        manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        saved = pd.read_csv(csv_path)

        assert {"date", "open", "high", "low", "close", "volume"} <= set(saved.columns)
        assert outcome["rows"] > 250
        assert len(outcome["sha256"]) == 64
        assert chart_path.is_file()
        assert manifest["backtest_ready"] is False
        assert manifest["provider"] == "synthetic"
        assert manifest["provider_package_version"] is None

    def test_records_empty_provider_result_as_error(self, tmp_path) -> None:
        result = collect_daily_snapshot(
            ["TEST"],
            "2022-01-03",
            tmp_path,
            provider=EmptyProvider(),
            make_charts=False,
        )

        assert not result.successful
        assert result.symbols[0]["status"] == "error"
        assert result.symbols[0]["error"] == "Provider returned no rows."

    def test_records_provider_failure_in_manifest(self, tmp_path) -> None:
        result = collect_daily_snapshot(
            ["TEST"],
            "2022-01-03",
            tmp_path,
            provider=FailingProvider(),
            make_charts=False,
        )

        manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        assert not result.successful
        assert manifest["symbols"][0]["status"] == "error"
        assert "unavailable: TEST" in manifest["symbols"][0]["error"]

    def test_rejects_duplicate_symbols(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="unique"):
            collect_daily_snapshot(
                ["TEST", "TEST"],
                "2022-01-03",
                tmp_path,
                provider=SyntheticProvider(),
            )

    def test_rejects_path_like_symbol(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="Invalid symbol"):
            collect_daily_snapshot(
                ["../TEST"],
                "2022-01-03",
                tmp_path,
                provider=SyntheticProvider(),
            )
