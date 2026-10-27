"""
Tests for paper-trading database and engine.
"""

import pandas as pd
import pytest

from nse_algo.config import ProjectConfig
from nse_algo.data.providers import SyntheticProvider
from nse_algo.paper.database import PaperDatabase
from nse_algo.paper.engine import PaperEngine


class TestPaperDatabase:
    def test_create_in_memory(self) -> None:
        db = PaperDatabase(":memory:")
        assert db.order_exists("nonexistent") is False
        db.close()

    def test_insert_and_check_order(self) -> None:
        db = PaperDatabase(":memory:")
        db.insert_order("ORDER_001", "RELIANCE", "buy", 100, 500.0, "2024-01-15")
        assert db.order_exists("ORDER_001") is True
        assert db.order_exists("ORDER_002") is False
        db.close()

    def test_duplicate_order_ignored(self) -> None:
        db = PaperDatabase(":memory:")
        db.insert_order("ORDER_001", "RELIANCE", "buy", 100, 500.0, "2024-01-15")
        db.insert_order("ORDER_001", "RELIANCE", "buy", 100, 500.0, "2024-01-15")
        # Should only have one order
        pending = db.get_pending_orders()
        assert len(pending) == 1
        db.close()

    def test_update_order_status(self) -> None:
        db = PaperDatabase(":memory:")
        db.insert_order("ORDER_001", "RELIANCE", "buy", 100, 500.0, "2024-01-15")
        db.update_order_status("ORDER_001", "filled")
        pending = db.get_pending_orders()
        assert len(pending) == 0
        db.close()

    def test_insert_and_get_fill(self) -> None:
        db = PaperDatabase(":memory:")
        db.insert_order("ORDER_001", "RELIANCE", "buy", 100, 500.0, "2024-01-15")
        db.insert_fill("ORDER_001", "RELIANCE", 500.0, 100, "2024-01-15", 25.0)
        fills = db.get_fills()
        assert len(fills) == 1
        assert fills[0]["fill_price"] == 500.0
        db.close()

    def test_position_crud(self) -> None:
        db = PaperDatabase(":memory:")
        db.upsert_position("RELIANCE", 100, 500.0, "2024-01-15", 480.0, 510.0)
        pos = db.get_position("RELIANCE")
        assert pos is not None
        assert pos["quantity"] == 100
        db.upsert_position("RELIANCE", 0)
        pos = db.get_position("RELIANCE")
        assert pos["quantity"] == 0
        db.close()

    def test_equity_snapshot(self) -> None:
        db = PaperDatabase(":memory:")
        db.insert_equity("2024-01-15", 100_000, 0, 100_000)
        db.insert_equity("2024-01-16", 95_000, 5_000, 100_000)
        latest = db.get_latest_equity()
        assert latest is not None
        assert latest["date"] == "2024-01-16"
        db.close()

    def test_state_key_value(self) -> None:
        db = PaperDatabase(":memory:")
        db.set_state("paused", True)
        db.set_state("max_equity", 150000)
        assert db.get_state("paused") is True
        assert db.get_state("max_equity") == 150000
        assert db.get_state("nonexistent", "default") == "default"
        db.close()

    def test_context_manager(self) -> None:
        with PaperDatabase(":memory:") as db:
            db.insert_order("ORDER_001", "RELIANCE", "buy", 100, 500.0, "2024-01-15")
            assert db.order_exists("ORDER_001")
        # db should be closed after context


class TestPaperEngine:
    def _make_data(self, n: int = 1) -> dict[str, pd.DataFrame]:
        p = SyntheticProvider(seed=42)
        return {f"STK{i}": p.fetch_ohlcv(f"STK{i}", "2020-01-01", "2023-12-31") for i in range(n)}

    def test_process_day_no_crash(self) -> None:
        data = self._make_data(1)
        config = ProjectConfig()
        engine = PaperEngine(config=config)
        dates = list(data["STK0"].index)
        result = engine.process_day(dates[250], data)
        assert "status" in result
        engine.db.close()

    def test_pause_resume(self) -> None:
        engine = PaperEngine(config=ProjectConfig())
        engine.pause()
        assert engine.paused is True
        data = self._make_data(1)
        dates = list(data["STK0"].index)
        result = engine.process_day(dates[250], data)
        assert result["status"] == "paused"
        engine.resume()
        assert engine.paused is False
        engine.db.close()

    def test_equity_persisted(self) -> None:
        data = self._make_data(1)
        engine = PaperEngine(config=ProjectConfig())
        dates = list(data["STK0"].index)
        engine.process_day(dates[250], data)
        latest = engine.db.get_latest_equity()
        assert latest is not None
        assert latest["total_equity"] > 0
        engine.db.close()

    def test_entry_signal_uses_prior_close_and_fills_current_open(self, monkeypatch) -> None:
        dates = pd.bdate_range("2024-01-02", periods=2)
        data = {
            "TEST": pd.DataFrame(
                {
                    "open": [95.0, 100.0],
                    "high": [101.0, 112.0],
                    "low": [94.0, 99.0],
                    "close": [100.0, 110.0],
                    "volume": [1000, 1000],
                },
                index=dates,
            )
        }
        signal_dates = []

        def fake_signals(bars, config):
            signal_dates.append(bars.index[-1])
            return pd.DataFrame(
                {
                    "signal_entry": [True],
                    "signal_exit": [False],
                    "proposed_stop": [90.0],
                    "ind_atr": [2.0],
                },
                index=[bars.index[-1]],
            )

        monkeypatch.setattr("nse_algo.paper.engine.generate_all_signals", fake_signals)
        engine = PaperEngine(config=ProjectConfig())
        result = engine.process_day(dates[1], data)

        assert signal_dates == [dates[0]]
        assert result["new_orders"] == 1
        fill = engine.db.get_fills()[0]
        assert fill["fill_price"] == 100.0
        assert engine.db.get_position("TEST")["entry_price"] == 100.0
        assert result["positions_value"] == 50 * 110.0
        engine.db.close()

    def test_gap_through_stop_fills_at_open_and_marks_equity(self, monkeypatch) -> None:
        dates = pd.bdate_range("2024-01-02", periods=2)
        data = {
            "TEST": pd.DataFrame(
                {
                    "open": [100.0, 80.0],
                    "high": [101.0, 86.0],
                    "low": [99.0, 75.0],
                    "close": [100.0, 85.0],
                    "volume": [1000, 1000],
                },
                index=dates,
            )
        }

        def fake_signals(bars, config):
            return pd.DataFrame(
                {
                    "signal_entry": [False],
                    "signal_exit": [False],
                    "proposed_stop": [95.0],
                    "ind_atr": [2.0],
                },
                index=[bars.index[-1]],
            )

        monkeypatch.setattr("nse_algo.paper.engine.generate_all_signals", fake_signals)
        engine = PaperEngine(config=ProjectConfig())
        engine.db.upsert_position("TEST", 10, 100.0, str(dates[0]), 90.0, 100.0)
        engine.db.insert_equity(str(dates[0]), 99_000.0, 1_000.0, 100_000.0)
        result = engine.process_day(dates[1], data)

        fill = engine.db.get_fills()[0]
        assert fill["fill_price"] == 80.0
        assert engine.db.get_position("TEST")["quantity"] == 0
        assert result["positions_value"] == 0.0
        assert result["total_equity"] == result["cash"]
        engine.db.close()

    def test_entry_day_stop_is_recorded_immediately(self, monkeypatch) -> None:
        dates = pd.bdate_range("2024-01-02", periods=2)
        data = {
            "TEST": pd.DataFrame(
                {
                    "open": [100.0, 100.0],
                    "high": [101.0, 105.0],
                    "low": [99.0, 80.0],
                    "close": [100.0, 95.0],
                    "volume": [1000, 1000],
                },
                index=dates,
            )
        }

        def fake_signals(bars, config):
            return pd.DataFrame(
                {
                    "signal_entry": [True],
                    "signal_exit": [False],
                    "proposed_stop": [90.0],
                    "ind_atr": [2.0],
                },
                index=[bars.index[-1]],
            )

        monkeypatch.setattr("nse_algo.paper.engine.generate_all_signals", fake_signals)
        engine = PaperEngine(config=ProjectConfig())
        result = engine.process_day(dates[1], data)

        fills = engine.db.get_fills()
        assert result["new_orders"] == 2
        assert len(fills) == 2
        assert [fill["fill_price"] for fill in fills] == [100.0, 90.0]
        assert engine.db.get_position("TEST")["quantity"] == 0
        assert result["positions_value"] == 0.0
        engine.db.close()

    def test_restart_recovery(self) -> None:
        """Test that state persists across restarts."""
        import tempfile, os
        db_path = tempfile.mktemp(suffix=".db")
        try:
            data = self._make_data(1)
            config = ProjectConfig()

            # First run
            db1 = PaperDatabase(db_path)
            db1.set_state("test_key", "test_value")
            db1.insert_equity("2024-01-15", 100_000, 0, 100_000)
            db1.close()

            # Second run — should recover state
            db2 = PaperDatabase(db_path)
            assert db2.get_state("test_key") == "test_value"
            latest = db2.get_latest_equity()
            assert latest is not None
            assert latest["date"] == "2024-01-15"
            db2.close()
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)
