"""Monthly forward return targets for regression (no lookahead).

For each month-end snapshot *t*, the 1-month forward return uses only prices at or
before the last trading day of month *t* (features) and at the last trading day
of month *t+1* (realized close). No future information enters the feature side.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "compute_monthly_forward_returns",
    "compute_quarterly_forward_returns",
    "compute_annual_forward_returns",
    "normalize_forward_returns_cs",
]


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


def _last_trading_day_on_or_before(
    idx: pd.DatetimeIndex,
    dt: pd.Timestamp,
) -> pd.Timestamp | None:
    """Last timestamp in *idx* on or before *dt*."""
    mask = idx <= dt
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


_QUARTER_END_MONTHS = (3, 6, 9, 12)


def compute_quarterly_forward_returns(
    ohlcv_by_ticker: dict[str, pd.DataFrame],
    start_quarter: str,
    end_quarter: str,
    cutoff_shift_days: int = 0,
) -> pd.DataFrame:
    """Compute 3-month forward returns from quarter-end snapshots.

    For each quarter in ``[start_quarter, end_quarter]`` and each ticker:

    - ``cutoff_date`` is the last trading day at the end of the quarter
      (features use data only through this date).
    - ``forward_1q_return`` is
      ``Close(last_trading_day_end_of_next_quarter) / Close(cutoff_date) - 1``.

    When *cutoff_shift_days* > 0, both the cutoff and forward timestamps are
    pushed forward by that many calendar days (then snapped to the last
    trading day on or before the shifted date).  This eliminates look-ahead
    bias by aligning cutoffs to post-publication dates.

    Parameters
    ----------
    ohlcv_by_ticker
        Mapping ticker -> OHLCV DataFrame with DatetimeIndex and ``Close``.
    start_quarter, end_quarter
        Inclusive quarter bounds as ``"YYYY-QN"``, e.g. ``"2012-Q1"`` and
        ``"2024-Q4"``.
    cutoff_shift_days
        Calendar days to shift both cutoff and forward dates forward
        (default ``0`` — no shift, backward-compatible).

    Returns
    -------
    pd.DataFrame
        Columns: ``ticker``, ``date``, ``cutoff_date``, ``forward_1q_return``.
    """
    quarters = _quarter_iter(start_quarter, end_quarter)
    rows: list[dict[str, Any]] = []

    for year, qtr in quarters:
        end_month = _QUARTER_END_MONTHS[qtr - 1]
        nxt_year, nxt_qtr = (year, qtr + 1) if qtr < 4 else (year + 1, 1)
        nxt_end_month = _QUARTER_END_MONTHS[nxt_qtr - 1]
        quarter_end_cal = (
            pd.Timestamp(year=year, month=end_month, day=1)
            + pd.offsets.MonthEnd(0)
        )

        for ticker, df in ohlcv_by_ticker.items():
            if df is None or df.empty or "Close" not in df.columns:
                continue
            close = df["Close"].copy()
            close.index = pd.to_datetime(close.index)
            close = close.sort_index()
            idx = close.index

            cutoff_ts = _last_trading_day_in_month(idx, year, end_month)
            if cutoff_ts is None:
                continue
            if cutoff_shift_days > 0:
                shifted = cutoff_ts + pd.Timedelta(days=cutoff_shift_days)
                cutoff_ts = _last_trading_day_on_or_before(idx, shifted)
                if cutoff_ts is None:
                    continue

            fwd_ts = _last_trading_day_in_month(idx, nxt_year, nxt_end_month)
            if fwd_ts is None:
                continue
            if cutoff_shift_days > 0:
                shifted = fwd_ts + pd.Timedelta(days=cutoff_shift_days)
                fwd_ts = _last_trading_day_on_or_before(idx, shifted)
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
                "date": quarter_end_cal,
                "cutoff_date": cutoff_ts,
                "forward_1q_return": float(fwd_ret),
            })

    out = pd.DataFrame(rows)
    if out.empty:
        logger.warning(
            "compute_quarterly_forward_returns: no rows for %s–%s (%d tickers)",
            start_quarter,
            end_quarter,
            len(ohlcv_by_ticker),
        )
        return pd.DataFrame(
            columns=["ticker", "date", "cutoff_date", "forward_1q_return"],
        )

    out = out.sort_values(["date", "ticker"]).reset_index(drop=True)
    logger.info(
        "compute_quarterly_forward_returns: %d rows (%d quarters × ~%d tickers)",
        len(out),
        out["date"].nunique(),
        len(out) // max(out["date"].nunique(), 1),
    )
    return out


def compute_annual_forward_returns(
    ohlcv_by_ticker: dict[str, pd.DataFrame],
    start_quarter: str,
    end_quarter: str,
    cutoff_shift_days: int = 0,
) -> pd.DataFrame:
    """Compute 12-month forward returns from quarter-end snapshots.

    For each quarter in ``[start_quarter, end_quarter]`` and each ticker:

    - ``cutoff_date`` is the last trading day at the end of the quarter
      (features use data only through this date).
    - ``forward_1y_return`` is
      ``Close(last_trading_day_end_of_same_quarter_next_year)
      / Close(cutoff_date) - 1`` (four calendar quarters forward).

    When *cutoff_shift_days* > 0, both the cutoff and forward timestamps are
    pushed forward by that many calendar days (then snapped to the last
    trading day on or before the shifted date).

    Parameters
    ----------
    ohlcv_by_ticker
        Mapping ticker -> OHLCV DataFrame with DatetimeIndex and ``Close``.
    start_quarter, end_quarter
        Inclusive quarter bounds as ``"YYYY-QN"``, e.g. ``"2012-Q1"`` and
        ``"2024-Q4"``.
    cutoff_shift_days
        Calendar days to shift both cutoff and forward dates forward
        (default ``0`` — no shift, backward-compatible).

    Returns
    -------
    pd.DataFrame
        Columns: ``ticker``, ``date``, ``cutoff_date``, ``forward_1y_return``.
    """
    quarters = _quarter_iter(start_quarter, end_quarter)
    rows: list[dict[str, Any]] = []

    for year, qtr in quarters:
        end_month = _QUARTER_END_MONTHS[qtr - 1]
        fwd_year = year + 1
        fwd_end_month = _QUARTER_END_MONTHS[qtr - 1]
        quarter_end_cal = (
            pd.Timestamp(year=year, month=end_month, day=1)
            + pd.offsets.MonthEnd(0)
        )

        for ticker, df in ohlcv_by_ticker.items():
            if df is None or df.empty or "Close" not in df.columns:
                continue
            close = df["Close"].copy()
            close.index = pd.to_datetime(close.index)
            close = close.sort_index()
            idx = close.index

            cutoff_ts = _last_trading_day_in_month(idx, year, end_month)
            if cutoff_ts is None:
                continue
            if cutoff_shift_days > 0:
                shifted = cutoff_ts + pd.Timedelta(days=cutoff_shift_days)
                cutoff_ts = _last_trading_day_on_or_before(idx, shifted)
                if cutoff_ts is None:
                    continue

            fwd_ts = _last_trading_day_in_month(idx, fwd_year, fwd_end_month)
            if fwd_ts is None:
                continue
            if cutoff_shift_days > 0:
                shifted = fwd_ts + pd.Timedelta(days=cutoff_shift_days)
                fwd_ts = _last_trading_day_on_or_before(idx, shifted)
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
                "date": quarter_end_cal,
                "cutoff_date": cutoff_ts,
                "forward_1y_return": float(fwd_ret),
            })

    out = pd.DataFrame(rows)
    if out.empty:
        logger.warning(
            "compute_annual_forward_returns: no rows for %s–%s (%d tickers)",
            start_quarter,
            end_quarter,
            len(ohlcv_by_ticker),
        )
        return pd.DataFrame(
            columns=["ticker", "date", "cutoff_date", "forward_1y_return"],
        )

    out = out.sort_values(["date", "ticker"]).reset_index(drop=True)
    logger.info(
        "compute_annual_forward_returns: %d rows (%d quarters × ~%d tickers)",
        len(out),
        out["date"].nunique(),
        len(out) // max(out["date"].nunique(), 1),
    )
    return out


def _quarter_iter(
    start_quarter: str, end_quarter: str,
) -> list[tuple[int, int]]:
    """Expand ``YYYY-QN`` bounds to ``(year, quarter_number)`` pairs."""
    start = pd.Period(start_quarter, freq="Q")
    end_p = pd.Period(end_quarter, freq="Q")
    if start > end_p:
        raise ValueError(
            f"start_quarter {start_quarter!r} must be <= end_quarter {end_quarter!r}",
        )
    return [
        (p.year, p.quarter)
        for p in pd.period_range(start=start, end=end_p, freq="Q")
    ]


def _cs_zscore_series(s: pd.Series) -> pd.Series:
    st = s.std(ddof=0)
    if not np.isfinite(st) or st < 1e-15:
        return pd.Series(0.0, index=s.index, dtype=float)
    return (s - s.mean()) / st


def _cs_rank_pct_series(s: pd.Series) -> pd.Series:
    return s.rank(pct=True, method="average")


def normalize_forward_returns_cs(
    df: pd.DataFrame,
    return_col: str,
    *,
    method: Literal["zscore", "rank"] = "zscore",
) -> pd.DataFrame:
    """Cross-sectional normalization of forward returns per ``cutoff_date``.

    For each ``cutoff_date`` group, applies either a z-score
    ``(ret - mean) / std`` (population std, ddof=0) or percentile rank in
    ``(0, 1]`` (ties averaged). Constant groups yield zero z-scores.

    Parameters
    ----------
    df
        Output of :func:`compute_quarterly_forward_returns`,
        :func:`compute_annual_forward_returns`, or similar, with columns
        ``ticker``, ``cutoff_date``, and *return_col*.
    return_col
        Column name of the raw forward return (e.g. ``forward_1q_return``).
    method
        ``"zscore"`` (default) or ``"rank"`` for percentile rank.

    Returns
    -------
    pd.DataFrame
        *df* with an extra column ``{return_col}_cs``.
    """
    if return_col not in df.columns:
        raise ValueError(
            f"normalize_forward_returns_cs: missing column {return_col!r}",
        )
    if "cutoff_date" not in df.columns:
        raise ValueError("normalize_forward_returns_cs: missing column 'cutoff_date'")

    out = df.copy()
    new_col = f"{return_col}_cs"
    if method == "zscore":
        out[new_col] = out.groupby("cutoff_date", sort=False)[return_col].transform(
            _cs_zscore_series,
        )
    elif method == "rank":
        out[new_col] = out.groupby("cutoff_date", sort=False)[return_col].transform(
            _cs_rank_pct_series,
        )
    else:
        raise ValueError(f"normalize_forward_returns_cs: unknown method {method!r}")
    return out
