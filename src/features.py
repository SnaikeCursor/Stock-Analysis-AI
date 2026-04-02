"""Feature engineering on pre-label data (no lookahead).

All features are point-in-time snapshots computed strictly on data up to and
including the *cutoff_date*.  No information from the classification window
(Q1 2024 for training, 2025 for OOS) leaks into features.

Technical features (23): momentum, trend, volatility, volume, liquidity, mean-reversion.
Fundamental features (10): valuation, growth, quality, size.
Seasonality (2): cyclic month encoding from the feature cutoff (``month_sin``, ``month_cos``).
Derived features: interaction terms (momentum × fundamentals), sector-relative
valuation (P/E vs sector median), and Swiss market regime (SMI momentum as SPI proxy).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from ta.momentum import ROCIndicator, RSIIndicator
from ta.trend import ADXIndicator, MACD, SMAIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import OnBalanceVolumeIndicator

try:
    from config import FEATURE_CUTOFF_DATE
except ImportError:
    FEATURE_CUTOFF_DATE = "2023-12-31"

logger = logging.getLogger(__name__)

__all__ = [
    "DERIVED_FEATURE_NAMES",
    "FUNDAMENTAL_FEATURE_NAMES",
    "INTERACTION_FEATURE_NAMES",
    "MACRO_BENCHMARK_TICKER",
    "MACRO_FEATURE_NAMES",
    "SEASONALITY_FEATURE_NAMES",
    "SECTOR_RELATIVE_FEATURE_NAMES",
    "TECHNICAL_FEATURE_NAMES",
    "audit_fundamental_coverage",
    "build_derived_features",
    "build_feature_matrix",
    "build_fundamental_features",
    "build_multi_period_feature_matrix",
    "build_rank_features",
    "build_sector_dummies",
    "build_technical_features",
    "compute_macro_momentum",
    "drop_correlated_features",
]

_MIN_BARS: int = 63

TECHNICAL_FEATURE_NAMES: list[str] = [
    # Momentum (5)
    "mom_1m",
    "mom_3m",
    "mom_6m",
    "rsi_14",
    "roc_10",
    # Trend (3)
    "sma_ratio_50_200",
    "macd_diff_norm",
    "adx_14",
    # Volatility (4)
    "hvol_20d",
    "hvol_60d",
    "atr_14_pct",
    "bb_width",
    # Volume (3)
    "obv_slope_20d",
    "volume_ratio_20_60",
    "rel_volume_5d",
    # Liquidity (3)
    "amihud_illiq",
    "volume_trend_60d",
    "spread_proxy",
    # Mean reversion (5)
    "dist_52w_high",
    "dist_52w_low",
    "zscore_20d",
    "return_skew_60d",
    "max_drawdown_60d",
]

FUNDAMENTAL_FEATURE_NAMES: list[str] = [
    "pe_ratio",
    "pb_ratio",
    "ev_ebitda",
    "dividend_yield",
    "revenue_growth",
    "earnings_growth",
    "roe",
    "profit_margin",
    "debt_equity",
    "market_cap_log",
    # Analyst consensus (yfinance .info)
    "analyst_rating",
    "analyst_count",
    "analyst_target_upside",
]

SEASONALITY_FEATURE_NAMES: list[str] = [
    "month_sin",
    "month_cos",
]

# Swiss Market Index (yfinance) — broad CH equity regime; proxies SPI for macro momentum.
MACRO_BENCHMARK_TICKER: str = "^SSMI"

INTERACTION_FEATURE_NAMES: list[str] = [
    "mom_3m_x_roe",
    "inv_pe_x_mom_6m",
]

SECTOR_RELATIVE_FEATURE_NAMES: list[str] = [
    "pe_vs_sector_median",
]

MACRO_FEATURE_NAMES: list[str] = [
    "spi_mom_3m",
    "spi_mom_6m",
]

DERIVED_FEATURE_NAMES: list[str] = (
    INTERACTION_FEATURE_NAMES + SECTOR_RELATIVE_FEATURE_NAMES + MACRO_FEATURE_NAMES
)

_YF_FUNDAMENTAL_MAP: dict[str, str] = {
    "trailingPE": "pe_ratio",
    "priceToBook": "pb_ratio",
    "enterpriseToEbitda": "ev_ebitda",
    "dividendYield": "dividend_yield",
    "revenueGrowth": "revenue_growth",
    "earningsGrowth": "earnings_growth",
    "returnOnEquity": "roe",
    "profitMargins": "profit_margin",
    "debtToEquity": "debt_equity",
    "recommendationMean": "analyst_rating",
    "numberOfAnalystOpinions": "analyst_count",
    "targetMeanPrice": "analyst_target_upside",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _truncate(df: pd.DataFrame, cutoff: str) -> pd.DataFrame:
    """Rows with index <= cutoff (strict lookahead prevention)."""
    out = df.copy()
    out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    return out.loc[: pd.Timestamp(cutoff)]


def _last_valid(s: pd.Series) -> float:
    """Last non-NaN value, or NaN if the series is empty/all-NaN."""
    vals = s.dropna()
    return float(vals.iloc[-1]) if not vals.empty else float("nan")


def _period_return(close: pd.Series, periods: int) -> float:
    """Simple return over *periods* trading days back from the last close."""
    if len(close) < periods + 1:
        return float("nan")
    p0 = float(close.iloc[-(periods + 1)])
    p1 = float(close.iloc[-1])
    if not np.isfinite(p0) or p0 <= 0:
        return float("nan")
    return (p1 / p0) - 1.0


# ---------------------------------------------------------------------------
# Technical features — single ticker
# ---------------------------------------------------------------------------

def _compute_technicals(df: pd.DataFrame) -> dict[str, float]:
    """All technical features from one ticker's OHLCV (already truncated).

    Returns a dict with exactly the keys in :data:`TECHNICAL_FEATURE_NAMES`.
    Missing/uncomputable features are ``NaN``.
    """
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    volume = df["Volume"].astype(float)

    close_clean = close.dropna()
    n = len(close_clean)

    if n < _MIN_BARS:
        logger.debug("Skipping technicals: only %d bars (need %d)", n, _MIN_BARS)
        return {name: float("nan") for name in TECHNICAL_FEATURE_NAMES}

    last_close = float(close_clean.iloc[-1])
    feats: dict[str, float] = {}

    # ── Momentum ──────────────────────────────────────────────────────────
    feats["mom_1m"] = _period_return(close_clean, 21)
    feats["mom_3m"] = _period_return(close_clean, 63)
    feats["mom_6m"] = _period_return(close_clean, 126)
    feats["rsi_14"] = _last_valid(RSIIndicator(close, window=14).rsi())
    feats["roc_10"] = _last_valid(ROCIndicator(close, window=10).roc())

    # ── Trend ─────────────────────────────────────────────────────────────
    sma50 = _last_valid(SMAIndicator(close, window=50).sma_indicator())
    sma200 = _last_valid(SMAIndicator(close, window=200).sma_indicator())
    if np.isfinite(sma200) and sma200 != 0:
        feats["sma_ratio_50_200"] = sma50 / sma200
    else:
        feats["sma_ratio_50_200"] = float("nan")

    macd_hist = _last_valid(MACD(close).macd_diff())
    if np.isfinite(macd_hist) and last_close > 0:
        feats["macd_diff_norm"] = macd_hist / last_close
    else:
        feats["macd_diff_norm"] = float("nan")

    feats["adx_14"] = _last_valid(
        ADXIndicator(high, low, close, window=14).adx()
    )

    # ── Volatility ────────────────────────────────────────────────────────
    log_ret = np.log(close / close.shift(1)).dropna()
    feats["hvol_20d"] = (
        float(log_ret.iloc[-20:].std() * np.sqrt(252))
        if len(log_ret) >= 20
        else float("nan")
    )
    feats["hvol_60d"] = (
        float(log_ret.iloc[-60:].std() * np.sqrt(252))
        if len(log_ret) >= 60
        else float("nan")
    )

    atr_val = _last_valid(
        AverageTrueRange(high, low, close, window=14).average_true_range()
    )
    if np.isfinite(atr_val) and last_close > 0:
        feats["atr_14_pct"] = atr_val / last_close
    else:
        feats["atr_14_pct"] = float("nan")

    feats["bb_width"] = _last_valid(
        BollingerBands(close, window=20, window_dev=2).bollinger_wband()
    )

    # ── Volume ────────────────────────────────────────────────────────────
    obv = OnBalanceVolumeIndicator(close, volume).on_balance_volume().dropna()
    obv_tail = obv.iloc[-20:]
    if len(obv_tail) >= 10:
        x = np.arange(len(obv_tail), dtype=float)
        y = obv_tail.values.astype(float)
        valid = np.isfinite(y)
        if valid.sum() >= 5:
            slope = float(np.polyfit(x[valid], y[valid], 1)[0])
            denom = float(np.mean(np.abs(y[valid])))
            feats["obv_slope_20d"] = slope / denom if denom > 0 else float("nan")
        else:
            feats["obv_slope_20d"] = float("nan")
    else:
        feats["obv_slope_20d"] = float("nan")

    vol_clean = volume.dropna()
    if len(vol_clean) >= 60:
        v20 = float(vol_clean.iloc[-20:].mean())
        v60 = float(vol_clean.iloc[-60:].mean())
        feats["volume_ratio_20_60"] = v20 / v60 if v60 > 0 else float("nan")
    else:
        feats["volume_ratio_20_60"] = float("nan")

    if len(vol_clean) >= 20:
        v5 = float(vol_clean.iloc[-5:].mean())
        v20 = float(vol_clean.iloc[-20:].mean())
        feats["rel_volume_5d"] = v5 / v20 if v20 > 0 else float("nan")
    else:
        feats["rel_volume_5d"] = float("nan")

    # ── Liquidity ─────────────────────────────────────────────────────────
    # Amihud (2002) illiquidity: mean(|daily return| / dollar volume) × 1e6
    # over 20 trading days.  Higher values ⇒ more illiquid.
    ret_abs = close.pct_change().abs()
    amihud_tail = (ret_abs / volume).iloc[-20:]
    amihud_valid = amihud_tail.replace([np.inf, -np.inf], np.nan).dropna()
    feats["amihud_illiq"] = (
        float(amihud_valid.mean()) * 1e6 if len(amihud_valid) >= 10 else float("nan")
    )

    # Log-volume trend: OLS slope of ln(volume) over the last 60 trading days.
    # Positive slope ⇒ increasing participation.
    vol_tail_60 = volume.iloc[-60:]
    vol_pos = vol_tail_60[vol_tail_60 > 0]
    if len(vol_pos) >= 30:
        log_vol = np.log(vol_pos.values.astype(float))
        x_idx = np.arange(len(log_vol), dtype=float)
        feats["volume_trend_60d"] = float(np.polyfit(x_idx, log_vol, 1)[0])
    else:
        feats["volume_trend_60d"] = float("nan")

    # Spread proxy: mean((High − Low) / Close) over 20 days.
    hl_spread = ((high - low) / close).iloc[-20:]
    hl_valid = hl_spread.replace([np.inf, -np.inf], np.nan).dropna()
    feats["spread_proxy"] = (
        float(hl_valid.mean()) if len(hl_valid) >= 10 else float("nan")
    )

    # ── Mean reversion ────────────────────────────────────────────────────
    window_52w = close_clean.iloc[-252:] if n >= 252 else close_clean
    hi_52w = float(window_52w.max())
    lo_52w = float(window_52w.min())
    feats["dist_52w_high"] = (
        (last_close / hi_52w) - 1.0 if hi_52w > 0 else float("nan")
    )
    feats["dist_52w_low"] = (
        (last_close / lo_52w) - 1.0 if lo_52w > 0 else float("nan")
    )

    tail_20 = close_clean.iloc[-20:]
    std_20 = float(tail_20.std())
    if std_20 > 0:
        feats["zscore_20d"] = (last_close - float(tail_20.mean())) / std_20
    else:
        feats["zscore_20d"] = float("nan")

    ret_simple = close_clean.pct_change().dropna()
    tail_ret_60 = ret_simple.iloc[-60:]
    if len(tail_ret_60) >= 60:
        feats["return_skew_60d"] = float(tail_ret_60.skew())
    else:
        feats["return_skew_60d"] = float("nan")

    tail_60_close = close_clean.iloc[-60:]
    if len(tail_60_close) >= 60:
        cummax = tail_60_close.cummax()
        dd = (tail_60_close / cummax) - 1.0
        feats["max_drawdown_60d"] = float(dd.min())
    else:
        feats["max_drawdown_60d"] = float("nan")

    return feats


# ---------------------------------------------------------------------------
# Fundamental features — single ticker
# ---------------------------------------------------------------------------

def _extract_fundamentals(info: dict[str, Any]) -> dict[str, float]:
    """Extract fundamental features from a yfinance ``.info`` dict."""
    feats: dict[str, float] = {}
    for yf_key, feat_name in _YF_FUNDAMENTAL_MAP.items():
        if feat_name == "analyst_target_upside":
            # Derived from targetMeanPrice and current/regular price (see below)
            continue
        raw = info.get(yf_key)
        if raw is None or (isinstance(raw, float) and not np.isfinite(raw)):
            feats[feat_name] = float("nan")
        else:
            try:
                feats[feat_name] = float(raw)
            except (ValueError, TypeError):
                feats[feat_name] = float("nan")

    mcap = info.get("marketCap")
    if mcap is not None:
        try:
            val = float(mcap)
            feats["market_cap_log"] = float(np.log(val)) if val > 0 else float("nan")
        except (ValueError, TypeError):
            feats["market_cap_log"] = float("nan")
    else:
        feats["market_cap_log"] = float("nan")

    # Target upside vs current price (not a single yfinance field)
    target_raw = info.get("targetMeanPrice")
    price_raw = info.get("currentPrice")
    if price_raw is None:
        price_raw = info.get("regularMarketPrice")
    if target_raw is None or price_raw is None:
        feats["analyst_target_upside"] = float("nan")
    elif isinstance(target_raw, float) and not np.isfinite(target_raw):
        feats["analyst_target_upside"] = float("nan")
    elif isinstance(price_raw, float) and not np.isfinite(price_raw):
        feats["analyst_target_upside"] = float("nan")
    else:
        try:
            tgt = float(target_raw)
            px = float(price_raw)
            feats["analyst_target_upside"] = (tgt / px) - 1.0 if px > 0 else float("nan")
        except (ValueError, TypeError):
            feats["analyst_target_upside"] = float("nan")

    return feats


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_technical_features(
    ohlcv_by_ticker: dict[str, pd.DataFrame],
    cutoff_date: str | None = None,
) -> pd.DataFrame:
    """Compute technical feature matrix (one row per ticker).

    Parameters
    ----------
    ohlcv_by_ticker
        Mapping ticker -> OHLCV DataFrame (DatetimeIndex, columns include
        Open / High / Low / Close / Volume).
    cutoff_date
        ISO date string.  All OHLCV rows *after* this date are dropped before
        any computation (lookahead prevention).  Defaults to
        :data:`config.FEATURE_CUTOFF_DATE`.
    """
    cutoff = cutoff_date if cutoff_date is not None else FEATURE_CUTOFF_DATE
    rows: dict[str, dict[str, float]] = {}
    nan_row = {name: float("nan") for name in TECHNICAL_FEATURE_NAMES}

    for ticker, df in ohlcv_by_ticker.items():
        if df is None or df.empty:
            rows[ticker] = nan_row.copy()
            continue
        try:
            trunc = _truncate(df, cutoff)
            if trunc.empty:
                rows[ticker] = nan_row.copy()
                continue
            rows[ticker] = _compute_technicals(trunc)
        except Exception:
            logger.warning(
                "Technical feature computation failed for %s", ticker, exc_info=True
            )
            rows[ticker] = nan_row.copy()

    result = pd.DataFrame.from_dict(rows, orient="index")
    result.index.name = "ticker"
    return result[TECHNICAL_FEATURE_NAMES]


def build_fundamental_features(
    fundamentals_by_ticker: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Compute fundamental feature matrix (one row per ticker).

    Parameters
    ----------
    fundamentals_by_ticker
        Mapping ticker -> yfinance ``.info`` dict (as returned by
        :func:`data_loader.load_fundamentals`).
    """
    rows: dict[str, dict[str, float]] = {}
    nan_row = {name: float("nan") for name in FUNDAMENTAL_FEATURE_NAMES}

    for ticker, info in fundamentals_by_ticker.items():
        if not info:
            rows[ticker] = nan_row.copy()
            continue
        try:
            rows[ticker] = _extract_fundamentals(info)
        except Exception:
            logger.warning(
                "Fundamental extraction failed for %s", ticker, exc_info=True
            )
            rows[ticker] = nan_row.copy()

    result = pd.DataFrame.from_dict(rows, orient="index")
    result.index.name = "ticker"
    return result[FUNDAMENTAL_FEATURE_NAMES]


def compute_macro_momentum(
    ohlcv: pd.DataFrame,
    cutoff_date: str,
) -> dict[str, float]:
    """SMI-based market regime: simple returns of the benchmark index up to *cutoff_date*.

    Uses the same bar counts as stock momentum (21 / 63 trading days).  Missing or
    insufficient history yields NaNs for those keys.

    Parameters
    ----------
    ohlcv
        Index OHLCV with ``Close`` (and DatetimeIndex).
    cutoff_date
        Last date included (inclusive), ISO string.

    Returns
    -------
    dict
        Keys ``spi_mom_3m``, ``spi_mom_6m`` (aligned with :data:`MACRO_FEATURE_NAMES`).
    """
    out: dict[str, float] = {k: float("nan") for k in MACRO_FEATURE_NAMES}
    if ohlcv is None or ohlcv.empty or "Close" not in ohlcv.columns:
        return out
    try:
        trunc = _truncate(ohlcv, cutoff_date)
    except Exception:
        return out
    close = trunc["Close"].astype(float).dropna()
    if len(close) < _MIN_BARS:
        return out
    out["spi_mom_3m"] = _period_return(close, 63)
    out["spi_mom_6m"] = _period_return(close, 126)
    return out


def build_derived_features(
    feature_matrix: pd.DataFrame,
    fundamentals_by_ticker: dict[str, dict[str, Any]] | None,
    *,
    cutoff_date: str,
    macro_ohlcv: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Interaction, sector-relative valuation, and macro regime columns.

    Expects *feature_matrix* to contain the columns needed for interactions
    (``mom_3m``, ``roe``, ``pe_ratio``, ``mom_6m``) when fundamentals were merged.

    Parameters
    ----------
    feature_matrix
        Technical (+ optional fundamental) rows, index = ticker.
    fundamentals_by_ticker
        Same mapping as :func:`build_fundamental_features`; required for sector-relative
        P/E.  If None, sector column is all NaN.
    cutoff_date
        Passed to :func:`compute_macro_momentum` for the macro series.
    macro_ohlcv
        Optional OHLCV for the benchmark index.  If omitted, macro columns are NaN
        unless pre-computed values are injected by the caller.

    Returns
    -------
    pd.DataFrame
        One row per ticker, columns :data:`DERIVED_FEATURE_NAMES`.
    """
    idx = feature_matrix.index
    derived = pd.DataFrame(
        np.nan,
        index=idx,
        columns=DERIVED_FEATURE_NAMES,
        dtype=float,
    )

    # ── Interactions (plan: mom_3m * roe; (1/pe) * mom_6m) ─────────────────
    if all(c in feature_matrix.columns for c in ("mom_3m", "roe")):
        m3 = feature_matrix["mom_3m"].astype(float)
        roe = feature_matrix["roe"].astype(float)
        derived["mom_3m_x_roe"] = m3 * roe
    if all(c in feature_matrix.columns for c in ("pe_ratio", "mom_6m")):
        pe = feature_matrix["pe_ratio"].astype(float)
        m6 = feature_matrix["mom_6m"].astype(float)
        inv_pe = 1.0 / pe.where((pe > 0) & np.isfinite(pe))
        derived["inv_pe_x_mom_6m"] = (inv_pe * m6).astype(float)

    # ── Sector-relative P/E ───────────────────────────────────────────────
    if fundamentals_by_ticker is not None and "pe_ratio" in feature_matrix.columns:
        pe_s = feature_matrix["pe_ratio"].astype(float)
        med_by_ticker = pd.Series(np.nan, index=idx, dtype=float)
        sectors: list[str | None] = []
        for t in idx:
            info = fundamentals_by_ticker.get(t) or {}
            raw = info.get("sector") if info else None
            if raw and isinstance(raw, str) and raw.strip():
                sectors.append(raw.strip())
            else:
                sectors.append(None)
        df_sec = pd.DataFrame({"pe": pe_s.values, "sector": sectors}, index=idx)
        for sec, grp in df_sec.groupby("sector", dropna=True):
            if sec is None:
                continue
            pe_col = grp["pe"].replace([np.inf, -np.inf], np.nan)
            pos = pe_col[np.isfinite(pe_col) & (pe_col > 0)]
            med = float(pos.median()) if len(pos) else float("nan")
            if np.isfinite(med):
                med_by_ticker.loc[grp.index] = med
        derived["pe_vs_sector_median"] = pe_s - med_by_ticker

    # ── Macro (same value broadcast to all tickers) ─────────────────────────
    if macro_ohlcv is not None:
        macro_vals = compute_macro_momentum(macro_ohlcv, cutoff_date)
        for name in MACRO_FEATURE_NAMES:
            v = macro_vals.get(name, float("nan"))
            derived[name] = v

    return derived


def audit_fundamental_coverage(
    fundamentals_by_ticker: dict[str, dict[str, Any]],
    *,
    min_non_nan_fraction: float = 0.5,
) -> pd.DataFrame:
    """Per-ticker counts of non-NaN fundamental features (likely stale data flags).

    Uses the same extraction rules as :func:`build_fundamental_features`.
    Tickers with empty ``info`` dicts or with fewer than
    ``min_non_nan_fraction`` of :data:`FUNDAMENTAL_FEATURE_NAMES` populated are
    flagged ``likely_stale``.

    Parameters
    ----------
    fundamentals_by_ticker
        Ticker → yfinance ``.info`` dict (or ``{}`` if fetch failed).
    min_non_nan_fraction
        Below this fraction of valid fundamentals, set ``likely_stale`` True.

    Returns
    -------
    pd.DataFrame
        Index ``ticker``; columns ``n_non_nan``, ``n_total``, ``fraction_non_nan``,
        ``empty_info``, ``likely_stale``.
    """
    n_total = len(FUNDAMENTAL_FEATURE_NAMES)
    rows: list[dict[str, Any]] = []

    for ticker, info in fundamentals_by_ticker.items():
        if not info:
            rows.append(
                {
                    "ticker": ticker,
                    "n_non_nan": 0,
                    "n_total": n_total,
                    "fraction_non_nan": 0.0,
                    "empty_info": True,
                    "likely_stale": True,
                },
            )
            continue
        feats = _extract_fundamentals(info)
        fund_row = pd.Series(feats)
        nn = int(fund_row.notna().sum())
        frac = nn / n_total if n_total else 0.0
        stale = frac < min_non_nan_fraction
        rows.append(
            {
                "ticker": ticker,
                "n_non_nan": nn,
                "n_total": n_total,
                "fraction_non_nan": frac,
                "empty_info": False,
                "likely_stale": stale,
            },
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(
            columns=[
                "n_non_nan",
                "n_total",
                "fraction_non_nan",
                "empty_info",
                "likely_stale",
            ],
        )

    out = out.set_index("ticker")
    n_stale = int(out["likely_stale"].sum())
    logger.info(
        "Fundamental coverage audit: %d tickers, %d likely stale (frac < %.0f%%)",
        len(out),
        n_stale,
        100 * min_non_nan_fraction,
    )
    if n_stale:
        bad = out.loc[out["likely_stale"]].index.tolist()
        preview = bad[:30]
        logger.info(
            "Likely stale / sparse fundamentals (%d): %s%s",
            n_stale,
            ", ".join(preview),
            " …" if len(bad) > 30 else "",
        )
    return out


def build_feature_matrix(
    ohlcv_by_ticker: dict[str, pd.DataFrame],
    cutoff_date: str,
    fundamentals_by_ticker: dict[str, dict[str, Any]] | None = None,
    *,
    macro_ohlcv: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Full feature matrix: technicals (+ optionally fundamentals), one row per ticker.

    Parameters
    ----------
    ohlcv_by_ticker
        Ticker -> OHLCV DataFrame.  May include :data:`MACRO_BENCHMARK_TICKER`
        (``^SSMI``) for macro regime features; that key is excluded from per-stock
        technicals and only used for ``spi_mom_*`` columns.
    cutoff_date
        Features use only data on or before this date (lookahead prevention).
    fundamentals_by_ticker
        Optional ticker -> yfinance ``.info`` dict.  When provided, fundamental
        columns are appended to the technical features.
    macro_ohlcv
        Optional OHLCV for the Swiss benchmark index.  Overrides lookup of
        ``ohlcv_by_ticker[MACRO_BENCHMARK_TICKER]``.

    Returns
    -------
    pd.DataFrame
        Index = ticker, columns = feature names.  ``NaN`` for features that
        could not be computed (insufficient history, missing data, …).
        Appends :data:`SEASONALITY_FEATURE_NAMES` from the cutoff calendar month
        and :data:`DERIVED_FEATURE_NAMES` (interactions, sector-relative P/E, macro).
    """
    stock_ohlcv = {
        k: v for k, v in ohlcv_by_ticker.items() if k != MACRO_BENCHMARK_TICKER
    }
    tech = build_technical_features(stock_ohlcv, cutoff_date=cutoff_date)

    if fundamentals_by_ticker is not None:
        fund = build_fundamental_features(fundamentals_by_ticker)
        tech = tech.join(fund, how="left")

    _month = int(pd.Timestamp(cutoff_date).month)
    _ang = 2 * np.pi * _month / 12.0
    tech["month_sin"] = np.sin(_ang)
    tech["month_cos"] = np.cos(_ang)

    macro_df = macro_ohlcv
    if macro_df is None:
        macro_df = ohlcv_by_ticker.get(MACRO_BENCHMARK_TICKER)
    derived = build_derived_features(
        tech,
        fundamentals_by_ticker,
        cutoff_date=cutoff_date,
        macro_ohlcv=macro_df,
    )
    tech = pd.concat([tech, derived], axis=1)

    logger.info(
        "Feature matrix: %d tickers x %d features (NaN rate %.1f%%)",
        tech.shape[0],
        tech.shape[1],
        tech.isna().mean().mean() * 100,
    )
    return tech


def build_multi_period_feature_matrix(
    ohlcv_by_ticker: dict[str, pd.DataFrame],
    multi_period_returns: pd.DataFrame,
    fundamentals_by_ticker: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Build a stacked feature matrix aligned with multi-period return observations.

    For each unique ``feature_cutoff`` in *multi_period_returns*, computes the
    full feature matrix once, then selects the rows corresponding to tickers
    in that period.  The result uses the same ``obs_id`` (``ticker__period``)
    index as the multi-period returns DataFrame.

    Parameters
    ----------
    ohlcv_by_ticker
        Ticker -> OHLCV DataFrame (full history, not truncated).
    multi_period_returns
        Output of :func:`classifier.compute_multi_period_returns` — must have
        columns ``ticker``, ``period``, ``feature_cutoff``.
    fundamentals_by_ticker
        Optional ticker -> yfinance ``.info`` dict.

    Returns
    -------
    pd.DataFrame
        Feature matrix indexed by ``obs_id``, one row per stock-period.
    """
    if multi_period_returns.empty:
        return pd.DataFrame()

    cutoff_groups = multi_period_returns.groupby("feature_cutoff")
    parts: list[pd.DataFrame] = []

    for cutoff, group in cutoff_groups:
        feat = build_feature_matrix(
            ohlcv_by_ticker,
            cutoff_date=str(cutoff),
            fundamentals_by_ticker=fundamentals_by_ticker,
        )

        for obs_id, row in group.iterrows():
            ticker = row["ticker"]
            if ticker in feat.index:
                row_feat = feat.loc[[ticker]].copy()
                row_feat.index = pd.Index([obs_id], name="obs_id")
                parts.append(row_feat)

    if not parts:
        return pd.DataFrame()

    result = pd.concat(parts)
    logger.info(
        "Multi-period feature matrix: %d observations x %d features (NaN rate %.1f%%)",
        result.shape[0],
        result.shape[1],
        result.isna().mean().mean() * 100,
    )
    return result


# ---------------------------------------------------------------------------
# Cross-sectional rank features
# ---------------------------------------------------------------------------

def build_rank_features(feature_matrix: pd.DataFrame) -> pd.DataFrame:
    """Convert raw feature values to cross-sectional percentile ranks.

    Each column is independently ranked across the universe of tickers for the
    given snapshot, producing values in ``[0, 1]`` (0 = lowest, 1 = highest).
    Tree models often benefit from relative positioning (e.g. "this stock's
    momentum is top-20%") rather than raw values.

    Parameters
    ----------
    feature_matrix
        Feature DataFrame (ticker index, feature columns).  NaN values remain
        NaN in the output.

    Returns
    -------
    pd.DataFrame
        Same shape as *feature_matrix*, columns renamed with a ``rank_`` prefix.
    """
    ranked = feature_matrix.rank(pct=True, na_option="keep")
    ranked.columns = [f"rank_{c}" for c in ranked.columns]
    return ranked


# ---------------------------------------------------------------------------
# Sector dummies
# ---------------------------------------------------------------------------

def build_sector_dummies(
    fundamentals_by_ticker: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """One-hot encode sector membership from yfinance ``.info`` dicts.

    The ``sector`` field in yfinance ``info`` (e.g. "Industrials",
    "Healthcare") is extracted per ticker and converted to binary columns
    prefixed with ``sector_``.

    Parameters
    ----------
    fundamentals_by_ticker
        Mapping ticker -> yfinance ``.info`` dict.

    Returns
    -------
    pd.DataFrame
        Index = ticker, columns = ``sector_<Name>`` (binary 0/1).
        Tickers with no sector information get 0 across all columns.
    """
    sectors: dict[str, str | None] = {}
    for ticker, info in fundamentals_by_ticker.items():
        raw = info.get("sector") if info else None
        if raw and isinstance(raw, str) and raw.strip():
            sectors[ticker] = raw.strip()
        else:
            sectors[ticker] = None

    sector_series = pd.Series(sectors, name="sector")
    dummies = pd.get_dummies(sector_series, prefix="sector", dtype=float)
    dummies.index.name = "ticker"
    return dummies


# ---------------------------------------------------------------------------
# Auto-drop correlated features
# ---------------------------------------------------------------------------

def drop_correlated_features(
    feature_matrix: pd.DataFrame,
    threshold: float = 0.85,
) -> tuple[pd.DataFrame, list[str]]:
    """Drop one of each feature pair with |Pearson rho| > *threshold*.

    When two features are highly correlated, the one appearing later in column
    order is dropped.  This reduces multicollinearity without requiring VIF
    (which is O(p²) and slower for wide matrices).

    Parameters
    ----------
    feature_matrix
        Feature DataFrame (ticker/obs index, feature columns).
    threshold
        Absolute correlation cutoff (default 0.85).

    Returns
    -------
    reduced : pd.DataFrame
        Feature matrix with redundant columns removed.
    dropped : list[str]
        Names of columns that were dropped.
    """
    corr = feature_matrix.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))

    to_drop: list[str] = []
    for col in upper.columns:
        if (upper[col] > threshold).any():
            to_drop.append(col)

    to_drop = list(dict.fromkeys(to_drop))
    reduced = feature_matrix.drop(columns=to_drop)

    if to_drop:
        logger.info(
            "Dropped %d correlated features (|rho| > %.2f): %s",
            len(to_drop),
            threshold,
            to_drop,
        )

    return reduced, to_drop
