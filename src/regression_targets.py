"""Monthly forward return targets for regression (no lookahead).

For each month-end snapshot *t*, the 1-month forward return uses only prices at or
before the last trading day of month *t* (features) and at the last trading day
of month *t+1* (realized close). No future information enters the feature side.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["compute_monthly_forward_returns"]


def _month_iter(start_month: str, end_month: str) -> list[tuple[int, int]]:
    """Expand ``YYYY-MM`` bounds to (year, month) pairs inclusive."""
    start = pd.Period(start_month, freq="M")
    end = pd.Period(end_month, freq="M")
    if start > end:
        raise ValueError(f"start_month {start_month!r} must be <= end_month {end_month!r}")
    return [(p.year, p.month) for p in pd.period_range(start=start, end=end, freq="M")]


def _last_trading_day_in_month(
    idx: pd.DatetimeIndex,
    year: int,
    month: int,
) -> pd.Timestamp | None:
    """Last timestamp in *idx* that falls in the given calendar month."""
    start = pd.Timestamp(year=year, month=month, day=1)
    end = start + pd.offsets.MonthEnd(0)
    mask = (idx >= start) & (idx <= end)
    if not mask.any():
        return None
    return idx[mask].max()


def _close_on_trading_day(close: pd.Series, ts: pd.Timestamp) -> float | None:
    """Close at *ts* if that timestamp exists in the index and is valid."""
    if ts not in close.index:
        return None
    val = close.loc[ts]
    if isinstance(val, pd.Series):
        val = val.iloc[-1]
    v = float(val)
    return v if np.isfinite(v) and v > 0 else None


def compute_monthly_forward_returns(
    ohlcv_by_ticker: dict[str, pd.DataFrame],
    start_month: str,
    end_month: str,
) -> pd.DataFrame:
    """Compute 1-month forward returns from month-end snapshots.

    For each calendar month in ``[start_month, end_month]`` and each ticker:

    - ``cutoff_date`` is the last trading day in that month (features use data
      only through this date — callers must align feature engineering the same way).
    - ``forward_1m_return`` is ``Close(last_trading_day_{t+1}) / Close(cutoff_date) - 1``.

    Rows with missing prices at either end are omitted (no imputation).

    Parameters
    ----------
    ohlcv_by_ticker
        Mapping ticker -> OHLCV DataFrame with DatetimeIndex and ``Close`` column.
    start_month, end_month
        Inclusive month bounds, e.g. ``\"2022-09\"`` and ``\"2024-12\"``.

    Returns
    -------
    pd.DataFrame
        Columns: ``ticker``, ``date``, ``cutoff_date``, ``forward_1m_return``.

        ``date`` is the calendar month-end (Timestamp) for the snapshot month *t*
        (label for the period). ``cutoff_date`` is the actual last trading day in *t*.
    """
    months = _month_iter(start_month, end_month)
    rows: list[dict[str, Any]] = []

    for year, month in months:
        month_end_cal = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
        if month == 12:
            ny, nm = year + 1, 1
        else:
            ny, nm = year, month + 1

        for ticker, df in ohlcv_by_ticker.items():
            if df is None or df.empty or "Close" not in df.columns:
                continue
            close = df["Close"].copy()
            close.index = pd.to_datetime(close.index)
            close = close.sort_index()
            idx = close.index
            cutoff_ts = _last_trading_day_in_month(idx, year, month)
            if cutoff_ts is None:
                continue
            fwd_ts = _last_trading_day_in_month(idx, ny, nm)
            if fwd_ts is None:
                continue
            if fwd_ts <= cutoff_ts:
                continue

            px0 = _close_on_trading_day(close, cutoff_ts)
            px1 = _close_on_trading_day(close, fwd_ts)
            if px0 is None or px1 is None:
                continue

            fwd_ret = (px1 / px0) - 1.0
            if not np.isfinite(fwd_ret):
                continue

            rows.append({
                "ticker": ticker,
                "date": month_end_cal,
                "cutoff_date": cutoff_ts,
                "forward_1m_return": float(fwd_ret),
            })

    out = pd.DataFrame(rows)
    if out.empty:
        logger.warning(
            "compute_monthly_forward_returns: no rows for months %s–%s (%d tickers)",
            start_month,
            end_month,
            len(ohlcv_by_ticker),
        )
        return pd.DataFrame(
            columns=["ticker", "date", "cutoff_date", "forward_1m_return"],
        )

    out = out.sort_values(["date", "ticker"]).reset_index(drop=True)
    logger.info(
        "compute_monthly_forward_returns: %d rows (%d months × ~%d tickers/row)",
        len(out),
        out["date"].nunique(),
        len(out) // max(out["date"].nunique(), 1),
    )
    return out
