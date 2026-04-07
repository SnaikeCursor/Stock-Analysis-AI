"""Eulerpool Financial Data API client for point-in-time fundamentals.

Provides historical quarterly/annual financial statements with filing dates,
enabling look-ahead-bias-free backtesting.

Usage:
    from src.eulerpool_client import EulerpoolClient
    client = EulerpoolClient()  # reads EULERPOOL_API_KEY from env / config
    income = client.income_statement("LOGN.SW")
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import requests

import config

logger = logging.getLogger(__name__)

_CACHE_DIR = config.DATA_DIR / "eulerpool"


class EulerpoolClient:
    """Thin REST wrapper around Eulerpool's equity endpoints."""

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or config.EULERPOOL_API_KEY
        if not self.api_key:
            raise ValueError(
                "EULERPOOL_API_KEY not set. "
                "Export it or add to .env (see .env.example)."
            )
        self.base = config.EULERPOOL_BASE_URL
        self._session = requests.Session()
        self._session.headers["Accept"] = "application/json"

    # ── Low-level ────────────────────────────────────────────────────────

    def _get(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        retries: int = 2,
    ) -> Any:
        url = f"{self.base}{path}"
        qp = dict(params or {})
        qp["token"] = self.api_key

        for attempt in range(retries + 1):
            try:
                resp = self._session.get(url, params=qp, timeout=30)
                if resp.status_code == 429:
                    wait = float(resp.headers.get("Retry-After", 5))
                    logger.warning("Rate-limited, waiting %.0fs …", wait)
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as exc:
                safe_msg = str(exc).split("?token=")[0]
                logger.warning(
                    "Eulerpool %s attempt %d failed: %s",
                    path, attempt + 1, safe_msg,
                )
                if attempt < retries:
                    time.sleep(1.5 * (attempt + 1))
        return None

    # ── Caching ──────────────────────────────────────────────────────────

    @staticmethod
    def _cache_path(category: str, identifier: str) -> Path:
        d = _CACHE_DIR / category
        d.mkdir(parents=True, exist_ok=True)
        safe = identifier.replace("/", "_").replace("\\", "_")
        return d / f"{safe}.json"

    def _cached_or_fetch(
        self,
        category: str,
        identifier: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        force: bool = False,
    ) -> Any:
        cp = self._cache_path(category, identifier)
        if not force and cp.exists():
            try:
                return json.loads(cp.read_text(encoding="utf-8"))
            except Exception:
                pass

        data = self._get(path, params=params)
        if data is not None:
            cp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        return data

    # ── Equity: Financial Statements (annual) ────────────────────────────

    def income_statement(
        self, identifier: str, *, force: bool = False,
    ) -> list[dict[str, Any]] | None:
        """Annual income statement history."""
        return self._cached_or_fetch(
            "income_statement", identifier,
            f"/equity/incomestatement/{identifier}",
            force=force,
        )

    def balance_sheet(
        self, identifier: str, *, force: bool = False,
    ) -> list[dict[str, Any]] | None:
        """Annual balance sheet history."""
        return self._cached_or_fetch(
            "balance_sheet", identifier,
            f"/equity/balancesheet/{identifier}",
            force=force,
        )

    def cash_flow(
        self, identifier: str, *, force: bool = False,
    ) -> list[dict[str, Any]] | None:
        """Annual cash flow statement history."""
        return self._cached_or_fetch(
            "cash_flow", identifier,
            f"/equity/cashflowstatement/{identifier}",
            force=force,
        )

    # ── Equity: Financial Statements (quarterly) ─────────────────────────

    def income_statement_quarterly(
        self, identifier: str, *, force: bool = False,
    ) -> list[dict[str, Any]] | None:
        """Quarterly income statement history."""
        return self._cached_or_fetch(
            "income_statement_q", identifier,
            f"/equity/income-statement-quarterly/{identifier}",
            force=force,
        )

    def fundamentals_quarterly(
        self, identifier: str, *, force: bool = False,
    ) -> list[dict[str, Any]] | None:
        """Quarterly consolidated fundamentals (revenue, EPS, margins)."""
        return self._cached_or_fetch(
            "fundamentals_q", identifier,
            f"/equity/fundamentals-quarterly/{identifier}",
            force=force,
        )

    # ── Equity: Ratios & Metrics ─────────────────────────────────────────

    def metrics(
        self, identifier: str, *, force: bool = False,
    ) -> list[dict[str, Any]] | None:
        """Financial ratios (P/E, P/B, ROE, …)."""
        return self._cached_or_fetch(
            "metrics", identifier,
            f"/equity/metrics/{identifier}",
            force=force,
        )

    def valuation_history(
        self, identifier: str, *, force: bool = False,
    ) -> list[dict[str, Any]] | None:
        """Historical valuation multiples."""
        return self._cached_or_fetch(
            "valuation_history", identifier,
            f"/equity/valuation-history/{identifier}",
            force=force,
        )

    # ── Equity: Analyst ──────────────────────────────────────────────────

    def estimates(
        self, identifier: str, *, force: bool = False,
    ) -> list[dict[str, Any]] | None:
        """Analyst consensus estimates."""
        return self._cached_or_fetch(
            "estimates", identifier,
            f"/equity/estimates/{identifier}",
            force=force,
        )

    def pit_estimates(
        self, identifier: str, *, force: bool = False,
    ) -> list[dict[str, Any]] | None:
        """Point-in-time analyst estimates (no look-ahead bias)."""
        return self._cached_or_fetch(
            "pit_estimates", identifier,
            f"/equity/pit/estimates/{identifier}",
            force=force,
        )

    # ── Equity: Profile ──────────────────────────────────────────────────

    def profile(
        self, identifier: str, *, force: bool = False,
    ) -> dict[str, Any] | None:
        """Company profile (sector, industry, market cap, …)."""
        return self._cached_or_fetch(
            "profile", identifier,
            f"/equity/profile/{identifier}",
            force=force,
        )

    def pit_profile(
        self, identifier: str, *, force: bool = False,
    ) -> dict[str, Any] | None:
        """Point-in-time company profile."""
        return self._cached_or_fetch(
            "pit_profile", identifier,
            f"/equity/pit/profile/{identifier}",
            force=force,
        )

    # ── Equity: Coverage check ───────────────────────────────────────────

    def coverage(
        self, identifier: str, *, force: bool = False,
    ) -> dict[str, Any] | None:
        """Data coverage info for a security."""
        return self._cached_or_fetch(
            "coverage", identifier,
            f"/equity/coverage/{identifier}",
            force=force,
        )

    # ── Index: Constituents ──────────────────────────────────────────────

    def index_constituents(
        self,
        index_id: str,
        *,
        start: int = 0,
        end: int = 500,
        force: bool = False,
    ) -> list[dict[str, Any]] | None:
        """Current/historical constituents of a stock index (e.g. 'sp500', 'spi')."""
        return self._cached_or_fetch(
            "index", index_id,
            f"/equity-extended/index-history/{index_id}",
            params={"start": start, "end": end},
            force=force,
        )

    # ── Equity: List / Search ────────────────────────────────────────────

    def list_stocks(
        self,
        *,
        exchange: str | None = None,
        force: bool = False,
    ) -> list[dict[str, Any]] | None:
        """List all available stocks, optionally filtered by exchange."""
        params: dict[str, Any] = {}
        if exchange:
            params["exchange"] = exchange
        cache_key = exchange or "all"
        return self._cached_or_fetch(
            "stock_list", cache_key,
            "/equity/list",
            params=params,
            force=force,
        )

    def search(self, query: str) -> list[dict[str, Any]] | None:
        """Search for stocks by name or ticker."""
        return self._get(f"/equity/search/{query}")
