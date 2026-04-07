"""Data Service — wraps src/data_loader.py, src/universe.py, src/eulerpool_fundamentals.py.

Provides OHLCV download/refresh, yfinance fundamentals, Eulerpool PIT
quarterly data, and universe management with in-process caching so the
webapp doesn't hit Yahoo / Eulerpool on every request.
"""

from __future__ import annotations

import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.data_loader import default_cache_dir, download_ohlcv, load_fundamentals
from src.features import MACRO_BENCHMARK_TICKER
from src.universe import SPI_TICKERS, filter_by_min_volume, get_spi_tickers

logger = logging.getLogger(__name__)


class DataService:
    """Stateful data gateway with in-memory caches for OHLCV, fundamentals, and Eulerpool PIT data.

    All heavy I/O (Yahoo downloads, Parquet reads, Eulerpool API) runs
    through here so callers (routers, ModelService) don't duplicate
    loading logic.
    """

    def __init__(self) -> None:
        self._ohlcv: dict[str, pd.DataFrame] = {}
        self._fundamentals: dict[str, dict[str, Any]] = {}
        self._eulerpool_quarterly: dict[str, list[dict]] = {}
        self._eulerpool_profiles: dict[str, dict] = {}
        self._cache_dir: Path = default_cache_dir()
        self._loaded: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self._loaded and len(self._ohlcv) > 0

    @property
    def ohlcv(self) -> dict[str, pd.DataFrame]:
        if not self._loaded:
            raise RuntimeError("Data not loaded — call refresh_ohlcv() or load_cached() first")
        return self._ohlcv

    @property
    def fundamentals(self) -> dict[str, dict[str, Any]]:
        if not self._loaded:
            raise RuntimeError("Data not loaded — call refresh_ohlcv() or load_cached() first")
        return self._fundamentals

    @property
    def eulerpool_quarterly(self) -> dict[str, list[dict]]:
        if not self._loaded:
            raise RuntimeError("Data not loaded — call refresh_ohlcv() or load_cached() first")
        return self._eulerpool_quarterly

    @property
    def eulerpool_profiles(self) -> dict[str, dict]:
        if not self._loaded:
            raise RuntimeError("Data not loaded — call refresh_ohlcv() or load_cached() first")
        return self._eulerpool_profiles

    def load_cached(self) -> int:
        """Load OHLCV + fundamentals from local Parquet/JSON cache without downloading.

        Returns the number of tickers loaded.
        """
        if self._loaded:
            return len(self._ohlcv)

        return self._do_load(force_refresh=False)

    def refresh_ohlcv(self, *, force_download: bool = False) -> int:
        """Download or refresh OHLCV data for the full SPI universe.

        Parameters
        ----------
        force_download
            If True, re-download from Yahoo even when cached Parquet exists.

        Returns the number of tickers loaded after filtering.
        """
        return self._do_load(force_refresh=force_download)

    def get_ticker_price(
        self,
        ticker: str,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        """Return OHLCV slice for a single ticker.

        Parameters
        ----------
        ticker
            Yahoo Finance symbol (e.g. ``NESN.SW``).
        start, end
            Optional ISO date bounds (inclusive). When omitted the full
            cached range is returned.
        """
        if not self._loaded:
            self.load_cached()

        df = self._ohlcv.get(ticker)
        if df is None or df.empty:
            raise KeyError(f"No OHLCV data for ticker {ticker!r}")

        if start is not None:
            df = df.loc[start:]
        if end is not None:
            df = df.loc[:end]

        return df

    def get_universe(self) -> list[str]:
        """Return the full SPI ticker list (pre-liquidity-filter)."""
        return get_spi_tickers()

    def get_liquid_tickers(self) -> list[str]:
        """Return only the tickers that passed the liquidity filter."""
        if not self._loaded:
            self.load_cached()
        return [t for t in self._ohlcv if t != MACRO_BENCHMARK_TICKER]

    def get_smi_ohlcv(self) -> pd.DataFrame:
        """Return the ^SSMI (macro benchmark) OHLCV frame.

        Raises ``KeyError`` if benchmark data is missing.
        """
        if not self._loaded:
            self.load_cached()

        smi = self._ohlcv.get(MACRO_BENCHMARK_TICKER)
        if smi is None or smi.empty:
            raise KeyError(f"Macro benchmark {MACRO_BENCHMARK_TICKER} not in OHLCV cache")
        return smi

    def data_end_date(self) -> date | None:
        """Return the last available date across loaded OHLCV data, or None."""
        if not self._loaded or not self._ohlcv:
            return None
        smi = self._ohlcv.get(MACRO_BENCHMARK_TICKER)
        if smi is not None and not smi.empty:
            return pd.Timestamp(smi.index[-1]).date()
        first = next(iter(self._ohlcv.values()), None)
        if first is not None and not first.empty:
            return pd.Timestamp(first.index[-1]).date()
        return None

    def ensure_data_covers(self, target_date: str) -> None:
        """Re-download OHLCV if the cached data doesn't reach *target_date*.

        Allows a 3-day grace window for weekends / holidays (e.g. requesting
        Monday when the last data point is Friday is fine).
        """
        if not self._loaded:
            self.load_cached()

        target = date.fromisoformat(target_date)
        end = self.data_end_date()
        grace = timedelta(days=3)

        if end is not None and end >= target - grace:
            return

        new_yf_end = (target + timedelta(days=5)).isoformat()
        logger.info(
            "Data ends at %s but need %s — refreshing OHLCV (YF_END=%s)",
            end, target, new_yf_end,
        )
        self._do_load(force_refresh=True, yf_end_override=new_yf_end)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _do_load(
        self,
        *,
        force_refresh: bool,
        yf_end_override: str | None = None,
    ) -> int:
        """Unified load path — download or cache-read, filter, load fundamentals."""
        tickers_with_macro = list(dict.fromkeys([*SPI_TICKERS, MACRO_BENCHMARK_TICKER]))

        yf_end = yf_end_override or config.YF_END

        logger.info(
            "Loading OHLCV (%d tickers, force_refresh=%s, end=%s)",
            len(tickers_with_macro),
            force_refresh,
            yf_end,
        )

        ohlcv = download_ohlcv(
            tickers_with_macro,
            config.YF_START,
            yf_end,
            self._cache_dir,
            force_refresh=force_refresh,
        )

        smi_ohlcv = ohlcv.get(MACRO_BENCHMARK_TICKER)
        liquid = filter_by_min_volume(ohlcv, config.MIN_DAILY_VOLUME_CHF)
        filtered: dict[str, pd.DataFrame] = {t: ohlcv[t] for t in liquid}

        if smi_ohlcv is not None and not smi_ohlcv.empty:
            filtered[MACRO_BENCHMARK_TICKER] = smi_ohlcv

        logger.info("Universe after liquidity filter: %d tickers", len(filtered))

        fundamentals: dict[str, dict[str, Any]] = {}
        for t in filtered:
            fundamentals[t] = load_fundamentals(t, cache_dir=self._cache_dir)

        # Eulerpool PIT quarterly fundamentals (cached on disk by EulerpoolClient)
        eulerpool_q: dict[str, list[dict]] = {}
        eulerpool_p: dict[str, dict] = {}
        try:
            from src.eulerpool_fundamentals import fetch_all_quarterly

            stock_tickers = [t for t in filtered if t != MACRO_BENCHMARK_TICKER]
            eulerpool_q, eulerpool_p = fetch_all_quarterly(stock_tickers)
            n_with_data = sum(1 for v in eulerpool_q.values() if v)
            logger.info(
                "Eulerpool PIT coverage: %d/%d tickers with quarterly data",
                n_with_data,
                len(stock_tickers),
            )
        except Exception:
            logger.warning(
                "Eulerpool PIT data unavailable — falling back to yfinance fundamentals",
                exc_info=True,
            )

        self._ohlcv = filtered
        self._fundamentals = fundamentals
        self._eulerpool_quarterly = eulerpool_q
        self._eulerpool_profiles = eulerpool_p
        self._loaded = True

        return len(filtered)
