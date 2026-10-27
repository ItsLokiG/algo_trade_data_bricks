"""Command-line entry point for exploratory daily-data collection."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from nse_algo.data.collection import DEFAULT_SYMBOLS, collect_daily_snapshot
from nse_algo.data.providers import ProviderError


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download, validate, and chart daily NSE sample data."
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=list(DEFAULT_SYMBOLS),
        help="Yahoo symbols; use .NS for NSE equities and ^NSEI for NIFTY 50.",
    )
    parser.add_argument("--start", default="2019-01-01", help="Start date (YYYY-MM-DD).")
    parser.add_argument(
        "--end",
        default=None,
        help="End date, exclusive (YYYY-MM-DD); defaults to the provider's latest data.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/raw/yfinance"),
        help="Root directory for timestamped snapshots.",
    )
    parser.add_argument(
        "--no-charts",
        action="store_true",
        help="Skip local price/volume/RSI charts.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        result = collect_daily_snapshot(
            args.symbols,
            args.start,
            args.output_root,
            end_date=args.end,
            make_charts=not args.no_charts,
        )
    except (ProviderError, ValueError) as exc:
        parser.error(str(exc))

    for item in result.symbols:
        if item["status"] == "error":
            print(f"{item['symbol']}: ERROR — {item['error']}")
        else:
            quality = item["quality"]
            print(
                f"{item['symbol']}: {item['status']} — {item['rows']} bars "
                f"({item['first_date']} to {item['last_date']}), "
                f"{quality['suspicious_moves']} suspicious moves"
            )
            for issue in quality["issues"]:
                print(f"  quality warning: {issue}")
            if item.get("chart"):
                print(f"  chart: {result.snapshot_dir / item['chart']}")

    print(f"Snapshot: {result.snapshot_dir.resolve()}")
    print(f"Manifest: {result.manifest_path.resolve()}")
    return 0 if result.successful else 1


if __name__ == "__main__":
    raise SystemExit(main())
