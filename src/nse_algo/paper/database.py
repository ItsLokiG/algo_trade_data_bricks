"""
SQLite-based persistence for the paper-trading engine.

Stores orders, fills, positions, and equity snapshots.
Supports restart recovery and duplicate-order prevention.

Uses Python's built-in sqlite3 — no external dependency.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS paper_orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id TEXT UNIQUE NOT NULL,
    symbol TEXT NOT NULL,
    order_type TEXT NOT NULL,  -- 'buy' or 'sell'
    quantity INTEGER NOT NULL,
    price REAL NOT NULL,
    signal_date TEXT NOT NULL,
    status TEXT DEFAULT 'pending',  -- pending, filled, cancelled
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS paper_fills (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    fill_price REAL NOT NULL,
    fill_quantity INTEGER NOT NULL,
    fill_date TEXT NOT NULL,
    costs REAL NOT NULL,
    FOREIGN KEY (order_id) REFERENCES paper_orders(order_id)
);

CREATE TABLE IF NOT EXISTS paper_positions (
    symbol TEXT PRIMARY KEY,
    quantity INTEGER DEFAULT 0,
    entry_price REAL DEFAULT 0,
    entry_date TEXT,
    stop_price REAL DEFAULT 0,
    highest_close REAL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS paper_equity (
    date TEXT PRIMARY KEY,
    cash REAL NOT NULL,
    positions_value REAL NOT NULL,
    total_equity REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS paper_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class PaperDatabase:
    """SQLite database for paper-trading state."""

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        self._conn: sqlite3.Connection | None = None
        self._connect()

    def _connect(self) -> None:
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA_SQL)
        self._conn.commit()

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "PaperDatabase":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # ── Orders ──

    def insert_order(
        self, order_id: str, symbol: str, order_type: str,
        quantity: int, price: float, signal_date: str,
    ) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO paper_orders (order_id, symbol, order_type, quantity, price, signal_date, status, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)",
            (order_id, symbol, order_type, quantity, price, signal_date, datetime.now().isoformat()),
        )
        self._conn.commit()

    def order_exists(self, order_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM paper_orders WHERE order_id = ?", (order_id,)
        ).fetchone()
        return row is not None

    def update_order_status(self, order_id: str, status: str) -> None:
        self._conn.execute(
            "UPDATE paper_orders SET status = ? WHERE order_id = ?", (status, order_id)
        )
        self._conn.commit()

    def get_pending_orders(self) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM paper_orders WHERE status = 'pending'"
        ).fetchall()

    # ── Fills ──

    def insert_fill(
        self, order_id: str, symbol: str, fill_price: float,
        fill_quantity: int, fill_date: str, costs: float,
    ) -> None:
        self._conn.execute(
            "INSERT INTO paper_fills (order_id, symbol, fill_price, fill_quantity, fill_date, costs) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (order_id, symbol, fill_price, fill_quantity, fill_date, costs),
        )
        self._conn.commit()

    def get_fills(self) -> list[sqlite3.Row]:
        return self._conn.execute("SELECT * FROM paper_fills").fetchall()

    # ── Positions ──

    def get_position(self, symbol: str) -> sqlite3.Row | None:
        return self._conn.execute(
            "SELECT * FROM paper_positions WHERE symbol = ?", (symbol,)
        ).fetchone()

    def upsert_position(
        self, symbol: str, quantity: int, entry_price: float = 0,
        entry_date: str | None = None, stop_price: float = 0,
        highest_close: float = 0,
    ) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO paper_positions (symbol, quantity, entry_price, entry_date, stop_price, highest_close) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (symbol, quantity, entry_price, entry_date, stop_price, highest_close),
        )
        self._conn.commit()

    def get_all_positions(self) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM paper_positions WHERE quantity > 0"
        ).fetchall()

    # ── Equity ──

    def insert_equity(self, date: str, cash: float, positions_value: float, total_equity: float) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO paper_equity (date, cash, positions_value, total_equity) "
            "VALUES (?, ?, ?, ?)",
            (date, cash, positions_value, total_equity),
        )
        self._conn.commit()

    def get_latest_equity(self) -> sqlite3.Row | None:
        return self._conn.execute(
            "SELECT * FROM paper_equity ORDER BY date DESC LIMIT 1"
        ).fetchone()

    # ── State ──

    def set_state(self, key: str, value: Any) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO paper_state (key, value) VALUES (?, ?)",
            (key, json.dumps(value)),
        )
        self._conn.commit()

    def get_state(self, key: str, default: Any = None) -> Any:
        row = self._conn.execute(
            "SELECT value FROM paper_state WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return default
        return json.loads(row["value"])
