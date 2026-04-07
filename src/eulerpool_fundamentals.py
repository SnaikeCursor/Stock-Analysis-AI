"""Eulerpool point-in-time fundamental feature extraction.

Wraps :class:`EulerpoolClient` to produce look-ahead-bias-free fundamental
features for each cutoff date using ``fundamentals_quarterly`` and ``profile``
endpoints.  Historical quarterly records are anchored by their ``period`` field,
enabling true point-in-time feature construction.
"""
from __future__ import annotations

import bisect
import logging
from typing import Any

import numpy as np
import pandas as pd

from src.eulerpool_client import EulerpoolClient

logger = logging.getLogger(__name__)

__all__ = [
    "extract_pit_features",
    "fetch_all_quarterly",
    "get_pit_record",
    "get_price_at_cutoff",
    "trailing_4q",
]

# Fields expected in each Eulerpool fundamentals_quarterly record:
#   period, revenue, grossIncome, ebit, earnings, revenue_ps, ebit_ps,
#   earnings_ps, net_debt, net_debt_by_ebit, gross_margin, ebit_margin,
#   earnings_margin, shares, dividend, payout_ratio


# ---------------------------------------------------------------------------
# Batch fetch
# ---------------------------------------------------------------------------

def fetch_all_quarterly(
    tickers: list[str],
    *,
    force: bool = False,
) -> tuple[dict[str, list[dict]], dict[str, dict]]:
    """Batch-fetch quarterly fundamentals and company profiles for *tickers*.

    Returns ``(quarterly_by_ticker, profiles_by_ticker)``.  Leverages
    :class:`EulerpoolClient`'s on-disk JSON cache so only the first
    invocation hits the API (~2 requests per ticker).

    Parameters
    ----------
    tickers
        Yahoo-style identifiers (e.g. ``LOGN.SW``).
    force
        If *True*, bypass cache and re-fetch from the API.
    """
    client = EulerpoolClient()
    quarterly: dict[str, list[dict]] = {}
    profiles: dict[str, dict] = {}

    n_q_ok = 0
    n_p_ok = 0

    for i, ticker in enumerate(tickers):
        q = client.fundamentals_quarterly(ticker, force=force)
        if q and isinstance(q, list):
            quarterly[ticker] = sorted(q, key=lambda r: r.get("period", ""))
            n_q_ok += 1
        else:
            quarterly[ticker] = []

        p = client.profile(ticker, force=force)
        if p and isinstance(p, dict):
            profiles[ticker] = p
            n_p_ok += 1
        else:
            profiles[ticker] = {}

        if (i + 1) % 50 == 0:
            logger.info("Eulerpool fetch: %d/%d tickers done", i + 1, len(tickers))

    logger.info(
        "Eulerpool fetch complete: %d/%d quarterly, %d/%d profiles",
        n_q_ok, len(tickers), n_p_ok, len(tickers),
    )
    return quarterly, profiles


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_actual(record: dict[str, Any]) -> bool:
    """True when the record is an actual (not estimated) quarter.

    Eulerpool marks forward estimates with a trailing ``'e'`` on the period.
    """
    period = record.get("period", "")
    if not period:
        return False
    return not str(period).rstrip().endswith("e")


def _period_str(record: dict[str, Any]) -> str:
    """Period date string stripped of any estimate suffix."""
    return str(record.get("period", "")).rstrip("e").strip()


def _safe_float(record: dict[str, Any], field: str) -> float:
    val = record.get(field)
    if val is None:
        return float("nan")
    try:
        v = float(val)
        return v if np.isfinite(v) else float("nan")
    except (ValueError, TypeError):
        return float("nan")


# ---------------------------------------------------------------------------
# Point-in-time lookup
# ---------------------------------------------------------------------------

def get_pit_record(
    quarterly_records: list[dict],
    cutoff_date: str,
) -> dict | None:
    """Latest actual quarterly record whose ``period <= cutoff_date``.

    *quarterly_records* must be pre-sorted by period ascending (as returned
    by :func:`fetch_all_quarterly`).  Uses :func:`bisect.bisect_right` for
    O(log n) lookup.
    """
    if not quarterly_records:
        return None

    actuals = [r for r in quarterly_records if _is_actual(r)]
    if not actuals:
        return None

    periods = [_period_str(r) for r in actuals]
    idx = bisect.bisect_right(periods, cutoff_date) - 1
    if idx < 0:
        return None
    return actuals[idx]


def trailing_4q(
    quarterly_records: list[dict],
    cutoff_date: str,
    field: str,
) -> float:
    """Sum of the last 4 actual quarter values for *field* on or before *cutoff_date*.

    Returns ``NaN`` when fewer than 4 valid values exist.
    """
    actuals = [r for r in quarterly_records if _is_actual(r)]
    if not actuals:
        return float("nan")

    periods = [_period_str(r) for r in actuals]
    idx = bisect.bisect_right(periods, cutoff_date) - 1
    if idx < 0:
        return float("nan")

    start = max(0, idx - 3)
    window = actuals[start : idx + 1]
    if len(window) < 4:
        return float("nan")

    total = 0.0
    n_valid = 0
    for r in window:
        val = r.get(field)
        if val is not None:
            try:
                v = float(val)
                if np.isfinite(v):
                    total += v
                    n_valid += 1
            except (ValueError, TypeError):
                pass

    return total if n_valid == 4 else float("nan")


# ---------------------------------------------------------------------------
# YoY growth helper
# ---------------------------------------------------------------------------

def _yoy_growth(
    quarterly_records: list[dict],
    cutoff_date: str,
    field: str,
) -> float:
    """Year-over-year growth: ``value_q / value_q_minus_4 − 1``."""
    actuals = [r for r in quarterly_records if _is_actual(r)]
    if not actuals:
        return float("nan")

    periods = [_period_str(r) for r in actuals]
    idx = bisect.bisect_right(periods, cutoff_date) - 1
    if idx < 4:
        return float("nan")

    current = _safe_float(actuals[idx], field)
    prior = _safe_float(actuals[idx - 4], field)

    if not np.isfinite(current) or not np.isfinite(prior) or prior == 0:
        return float("nan")
    return (current / prior) - 1.0


# ---------------------------------------------------------------------------
# Price lookup
# ---------------------------------------------------------------------------

def get_price_at_cutoff(
    ohlcv: pd.DataFrame,
    cutoff_date: str,
) -> float:
    """Last available close price on or before *cutoff_date*.

    Returns ``NaN`` when no valid price is available.
    """
    if ohlcv is None or ohlcv.empty or "Close" not in ohlcv.columns:
        return float("nan")

    df = ohlcv.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    trunc = df.loc[: pd.Timestamp(cutoff_date)]
    if trunc.empty:
        return float("nan")

    close = trunc["Close"].dropna()
    if close.empty:
        return float("nan")
    return float(close.iloc[-1])


# ---------------------------------------------------------------------------
# Feature extraction — single ticker
# ---------------------------------------------------------------------------

_PIT_FEATURE_KEYS: list[str] = [
    "pe_ratio",
    "ev_ebitda",
    "dividend_yield",
    "revenue_growth",
    "earnings_growth",
    "profit_margin",
    "market_cap_log",
    "ebit_margin",
    "gross_margin",
    "net_debt_by_ebit",
]


def extract_pit_features(
    quarterly_records: list[dict],
    cutoff_date: str,
    price_at_cutoff: float,
    publication_lag_days: int = 0,
) -> dict[str, float]:
    """Produce the PIT fundamental feature dict for one ticker at one cutoff.

    *price_at_cutoff* uses the real *cutoff_date* (market price at rebalance).
    Quarterly selection and TTM windows use *cutoff_date* minus
    *publication_lag_days* when ``publication_lag_days > 0``, so fundamentals
    are treated as available only after a typical reporting delay.

    Mapping from Eulerpool ``fundamentals_quarterly`` fields:

    =============================  ==========================================
    Feature                        Derivation
    =============================  ==========================================
    ``pe_ratio``                   ``price / trailing_4q(earnings_ps)``
    ``profit_margin``              ``earnings_margin`` from latest quarter
    ``revenue_growth``             YoY ``revenue`` growth
    ``earnings_growth``            YoY ``earnings`` growth
    ``market_cap_log``             ``log(price × shares)``
    ``dividend_yield``             ``trailing_4q(dividend) / price``
    ``ev_ebitda``                  ``(price×shares + net_debt) / trailing_4q(ebit)``
    ``ebit_margin``                direct from latest quarter
    ``gross_margin``               direct from latest quarter
    ``net_debt_by_ebit``           direct from latest quarter
    =============================  ==========================================
    """
    nan_dict: dict[str, float] = {k: float("nan") for k in _PIT_FEATURE_KEYS}

    if (
        not quarterly_records
        or not np.isfinite(price_at_cutoff)
        or price_at_cutoff <= 0
    ):
        return nan_dict

    if publication_lag_days > 0:
        adj = (
            pd.Timestamp(cutoff_date)
            - pd.Timedelta(days=publication_lag_days)
        ).strftime("%Y-%m-%d")
    else:
        adj = cutoff_date

    rec = get_pit_record(quarterly_records, adj)
    if rec is None:
        return nan_dict

    feats = dict(nan_dict)

    # ── Trailing P/E ──
    ttm_eps = trailing_4q(quarterly_records, adj, "earnings_ps")
    if np.isfinite(ttm_eps) and ttm_eps > 0:
        feats["pe_ratio"] = price_at_cutoff / ttm_eps

    # ── Profit margin = earnings_margin ──
    feats["profit_margin"] = _safe_float(rec, "earnings_margin")

    # ── YoY growth ──
    feats["revenue_growth"] = _yoy_growth(quarterly_records, adj, "revenue")
    feats["earnings_growth"] = _yoy_growth(quarterly_records, adj, "earnings")

    # ── Market cap ──
    shares = _safe_float(rec, "shares")
    if np.isfinite(shares) and shares > 0:
        mcap = price_at_cutoff * shares
        feats["market_cap_log"] = float(np.log(mcap)) if mcap > 0 else float("nan")

    # ── Dividend yield (TTM) ──
    ttm_div = trailing_4q(quarterly_records, adj, "dividend")
    if np.isfinite(ttm_div):
        feats["dividend_yield"] = ttm_div / price_at_cutoff

    # ── EV / EBITDA (EBIT as proxy) ──
    ttm_ebit = trailing_4q(quarterly_records, adj, "ebit")
    net_debt = _safe_float(rec, "net_debt")
    if np.isfinite(shares) and shares > 0 and np.isfinite(ttm_ebit) and ttm_ebit > 0:
        ev = price_at_cutoff * shares + (net_debt if np.isfinite(net_debt) else 0.0)
        if ev > 0:
            feats["ev_ebitda"] = ev / ttm_ebit

    # ── Direct margin / leverage fields ──
    feats["ebit_margin"] = _safe_float(rec, "ebit_margin")
    feats["gross_margin"] = _safe_float(rec, "gross_margin")
    feats["net_debt_by_ebit"] = _safe_float(rec, "net_debt_by_ebit")

    return feats
