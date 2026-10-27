"""Download, validate, and chart local daily market-data snapshots."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import pandas as pd

from nse_algo.data.providers import DataProvider, ProviderError, get_provider
from nse_algo.data.validation import validate_ohlcv

logger = logging.getLogger(__name__)


def _package_version(package_name: str) -> str | None:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


DEFAULT_SYMBOLS = (
    "RELIANCE.NS",
    "TCS.NS",
    "HDFCBANK.NS",
    "INFY.NS",
    "ITC.NS",
    "^NSEI",
)
_SYMBOL_PATTERN = re.compile(r"^[A-Za-z0-9.^_-]+$")


@dataclass(frozen=True)
class SnapshotResult:
    """Files and per-symbol outcomes produced by a collection run."""

    snapshot_dir: Path
    manifest_path: Path
    symbols: tuple[dict[str, Any], ...]

    @property
    def successful(self) -> bool:
        return all(
            item["status"] in {"ok", "ok_with_warnings"} for item in self.symbols
        )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _plot_symbol(data: pd.DataFrame, symbol: str, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from nse_algo.features.indicators import rsi, sma

    price = data["adj_close"] if "adj_close" in data.columns else data["close"]
    price_label = "Adjusted close (provider)" if "adj_close" in data.columns else "Close"

    figure, (price_axis, volume_axis, rsi_axis) = plt.subplots(
        3,
        1,
        figsize=(13, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1, 1]},
    )
    price_axis.plot(data.index, data["close"], alpha=0.45, label="Raw close")
    if "adj_close" in data.columns:
        price_axis.plot(data.index, data["adj_close"], alpha=0.8, label=price_label)
    price_axis.plot(data.index, sma(price, 50), label="50-session SMA")
    price_axis.plot(data.index, sma(price, 200), label="200-session SMA")
    price_axis.set_title(f"{symbol} — daily data, technical visualization only")
    price_axis.set_ylabel("Price")
    price_axis.legend(loc="best")
    price_axis.grid(alpha=0.2)

    volume_axis.bar(data.index, data["volume"], width=1.0, color="slategray")
    volume_axis.set_ylabel("Volume")
    volume_axis.grid(alpha=0.2)

    rsi_axis.plot(data.index, rsi(price, 14), color="darkorange", label="RSI (14)")
    rsi_axis.axhline(70, color="gray", linestyle="--", linewidth=0.8)
    rsi_axis.axhline(30, color="gray", linestyle="--", linewidth=0.8)
    rsi_axis.set_ylabel("RSI")
    rsi_axis.set_ylim(0, 100)
    rsi_axis.grid(alpha=0.2)

    figure.supxlabel("Date")
    figure.tight_layout()
    figure.savefig(output_path, dpi=140)
    plt.close(figure)


def collect_daily_snapshot(
    symbols: tuple[str, ...] | list[str],
    start_date: str,
    output_root: str | Path,
    *,
    end_date: str | None = None,
    provider: DataProvider | None = None,
    make_charts: bool = True,
) -> SnapshotResult:
    """Fetch symbols into a timestamped immutable-ish local snapshot.

    Raw OHLCV and provider adjustment/action fields are retained. Data is
    deliberately not converted into backtest-ready prices by this function.
    """
    if not symbols:
        raise ValueError("At least one symbol is required.")
    if len(set(symbols)) != len(symbols):
        raise ValueError("Symbols must be unique.")
    invalid_symbols = [symbol for symbol in symbols if not _SYMBOL_PATTERN.fullmatch(symbol)]
    if invalid_symbols:
        raise ValueError(f"Invalid symbol values: {invalid_symbols}")

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date) if end_date else None
    if end is not None and end <= start:
        raise ValueError("end_date must be later than start_date.")

    data_provider = provider or get_provider("yfinance")
    retrieved_at = datetime.now(timezone.utc)
    snapshot_id = retrieved_at.strftime("%Y%m%dT%H%M%S%fZ")
    snapshot_dir = Path(output_root) / snapshot_id
    snapshot_dir.mkdir(parents=True, exist_ok=False)
    chart_dir = snapshot_dir / "charts"
    if make_charts:
        chart_dir.mkdir()

    outcomes: list[dict[str, Any]] = []
    for symbol in symbols:
        try:
            bars = data_provider.fetch_ohlcv(
                symbol, start.strftime("%Y-%m-%d"),
                end.strftime("%Y-%m-%d") if end is not None else None,
            )
        except ProviderError as exc:
            logger.error("Data download failed for %s: %s", symbol, exc)
            outcomes.append({"symbol": symbol, "status": "error", "error": str(exc)})
            continue

        if bars.empty:
            message = "Provider returned no rows."
            logger.error("%s for %s", message, symbol)
            outcomes.append({"symbol": symbol, "status": "error", "error": message})
            continue

        bars = bars.sort_index()
        report = validate_ohlcv(bars, symbol, min_trading_days=1, max_missing_pct=0.10)
        csv_path = snapshot_dir / f"{symbol}.csv"
        export = bars.rename_axis("date").reset_index()
        export.to_csv(csv_path, index=False, date_format="%Y-%m-%d")

        outcome: dict[str, Any] = {
            "symbol": symbol,
            "status": (
                "invalid_data"
                if not report.passed
                else "ok_with_warnings"
                if report.issues or report.suspicious_moves
                else "ok"
            ),
            "rows": len(bars),
            "first_date": bars.index.min().strftime("%Y-%m-%d"),
            "last_date": bars.index.max().strftime("%Y-%m-%d"),
            "zero_volume_dates": [
                timestamp.strftime("%Y-%m-%d")
                for timestamp in bars.index[bars["volume"] == 0]
            ],
            "csv": csv_path.name,
            "sha256": _file_sha256(csv_path),
            "quality": asdict(report),
        }
        if report.passed and make_charts:
            chart_path = chart_dir / f"{symbol}.png"
            _plot_symbol(bars, symbol, chart_path)
            outcome["chart"] = str(Path("charts") / chart_path.name)
        outcomes.append(outcome)

    manifest_path = snapshot_dir / "manifest.json"
    manifest = {
        "snapshot_id": snapshot_id,
        "retrieved_at_utc": retrieved_at.isoformat(),
        "provider": data_provider.name,
        "provider_package_version": (
            _package_version("yfinance")
            if data_provider.name == "yfinance"
            else None
        ),
        "provider_is_official": data_provider.is_official,
        "interval": "1d",
        "adjustment_policy": (
            "Provider-returned OHLCV, adjusted close, dividends, and stock splits "
            "are retained where available; no local adjustment is applied."
        ),
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date_exclusive": end.strftime("%Y-%m-%d") if end is not None else None,
        "symbols": outcomes,
        "backtest_ready": False,
        "limitations": [
            "Data is for exploratory research and may be incomplete or corrected later.",
            "Yahoo Finance is unofficial; source data rights and availability may vary.",
            "Business-day gap checks are approximate and do not use the NSE holiday calendar.",
            "No historical index membership, delisted securities, or point-in-time fundamentals are included.",
            "These files are not validated for backtesting or trading use.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return SnapshotResult(
        snapshot_dir=snapshot_dir,
        manifest_path=manifest_path,
        symbols=tuple(outcomes),
    )
