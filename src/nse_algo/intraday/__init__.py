"""Intraday SMC day-trading research module (explicit scope extension).

Separate from the daily/weekly swing system.  See
``docs/intraday_smc_strategy.md`` for the design document and the
scope-change note.
"""

from nse_algo.intraday.config import IntradayConfig, IntradayCostConfig

__all__ = ["IntradayConfig", "IntradayCostConfig"]
