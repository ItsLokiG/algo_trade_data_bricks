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
