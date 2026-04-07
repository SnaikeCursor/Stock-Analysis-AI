"""Feature engineering on pre-label data (no lookahead).

All features are point-in-time snapshots computed strictly on data up to and
including the *cutoff_date*.  No information from the classification window
(Q1 2024 for training, 2025 for OOS) leaks into features.

Technical features (23): momentum, trend, volatility, volume, liquidity, mean-reversion.
Fundamental features (10): valuation, growth, quality, size, margins, leverage.
Seasonality (2): cyclic month encoding from the feature cutoff (``month_sin``, ``month_cos``).
Derived features: interaction terms (momentum × fundamentals), sector-relative
valuation (P/E vs sector median), and Swiss market regime (SMI momentum as SPI proxy).

Fundamental features support two data sources:

* **yfinance** (``.info`` snapshot) — static, single-date; used as fallback.
* **Eulerpool** (``fundamentals_quarterly``) — historical quarterly records with
  ``period`` dates, enabling true point-in-time feature construction per cutoff.
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
    "FEATURE_REGISTRY",
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
    "build_fundamental_features_pit",
    "build_multi_period_feature_matrix",
    "build_rank_features",
    "build_sector_dummies",
    "build_technical_features",
    "compute_macro_momentum",
    "drop_correlated_features",
    "drop_correlated_features_train_test",
    "get_feature_registry_df",
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
    "ev_ebitda",
    "dividend_yield",
    "revenue_growth",
    "earnings_growth",
    "profit_margin",
    "market_cap_log",
    # PIT-sourced margins & leverage (Eulerpool); NaN when yfinance fallback
    "ebit_margin",
    "gross_margin",
    "net_debt_by_ebit",
]

SEASONALITY_FEATURE_NAMES: list[str] = [
    "month_sin",
    "month_cos",
]

# Swiss Market Index (yfinance) — broad CH equity regime; proxies SPI for macro momentum.
MACRO_BENCHMARK_TICKER: str = "^SSMI"

INTERACTION_FEATURE_NAMES: list[str] = [
    "mom_3m_x_ebit_margin",
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

# ---------------------------------------------------------------------------
# Feature registry — category, lookback, human-readable description per feature
# ---------------------------------------------------------------------------

FEATURE_REGISTRY: dict[str, dict[str, str]] = {
    # ── Momentum ──
    "mom_1m": {
        "category": "momentum",
        "lookback": "21d",
        "description": "1-month price momentum (simple return over 21 trading days)",
    },
    "mom_3m": {
        "category": "momentum",
        "lookback": "63d",
        "description": "3-month price momentum (simple return over 63 trading days)",
    },
    "mom_6m": {
        "category": "momentum",
        "lookback": "126d",
        "description": "6-month price momentum (simple return over 126 trading days)",
    },
    "rsi_14": {
        "category": "momentum",
        "lookback": "14d",
        "description": "Relative Strength Index (14-day window)",
    },
    "roc_10": {
        "category": "momentum",
        "lookback": "10d",
        "description": "Rate of Change (10-day window)",
    },
    # ── Trend ──
    "sma_ratio_50_200": {
        "category": "trend",
        "lookback": "200d",
        "description": "SMA(50) / SMA(200) — golden/death cross proximity",
    },
    "macd_diff_norm": {
        "category": "trend",
        "lookback": "26d",
        "description": "MACD histogram normalised by close price",
    },
    "adx_14": {
        "category": "trend",
        "lookback": "14d",
        "description": "Average Directional Index (14-day) — trend strength",
    },
    # ── Volatility ──
    "hvol_20d": {
        "category": "volatility",
        "lookback": "20d",
        "description": "20-day historical volatility (annualised std of log returns)",
    },
    "hvol_60d": {
        "category": "volatility",
        "lookback": "60d",
        "description": "60-day historical volatility (annualised std of log returns)",
    },
    "atr_14_pct": {
        "category": "volatility",
        "lookback": "14d",
        "description": "Average True Range (14-day) as percentage of close",
    },
    "bb_width": {
        "category": "volatility",
        "lookback": "20d",
        "description": "Bollinger Band width (upper − lower) / middle",
    },
    # ── Volume ──
    "obv_slope_20d": {
        "category": "volume",
        "lookback": "20d",
        "description": "OBV slope (linear regression of On-Balance Volume over 20 days)",
    },
    "volume_ratio_20_60": {
        "category": "volume",
        "lookback": "60d",
        "description": "Mean volume (20d) / mean volume (60d) — short-term volume momentum",
    },
    "rel_volume_5d": {
        "category": "volume",
        "lookback": "60d",
        "description": "Mean volume (5d) / mean volume (60d) — very-short-term activity spike",
    },
    # ── Liquidity ──
    "amihud_illiq": {
        "category": "liquidity",
        "lookback": "20d",
        "description": "Amihud illiquidity ratio (mean |return| / volume over 20 days)",
    },
    "volume_trend_60d": {
        "category": "liquidity",
        "lookback": "60d",
        "description": "Volume trend (linear regression slope of log-volume over 60 days)",
    },
    "spread_proxy": {
        "category": "liquidity",
        "lookback": "20d",
        "description": "Bid-ask spread proxy: mean (High−Low)/Close over 20 days",
    },
    # ── Mean reversion ──
    "dist_52w_high": {
        "category": "mean_reversion",
        "lookback": "252d",
        "description": "Distance from 52-week high (current close / 252d max − 1)",
    },
    "dist_52w_low": {
        "category": "mean_reversion",
        "lookback": "252d",
        "description": "Distance from 52-week low (current close / 252d min − 1)",
    },
    "zscore_20d": {
        "category": "mean_reversion",
        "lookback": "20d",
        "description": "Z-score of close price vs. 20-day SMA and std",
    },
    "return_skew_60d": {
        "category": "mean_reversion",
        "lookback": "60d",
        "description": "Skewness of daily returns over 60 days",
    },
    "max_drawdown_60d": {
        "category": "mean_reversion",
        "lookback": "60d",
        "description": "Maximum drawdown over the last 60 trading days",
    },
    # ── Fundamentals (PIT via Eulerpool; yfinance fallback) ──
    "pe_ratio": {
        "category": "fundamental_valuation",
        "lookback": "point-in-time",
        "description": "Trailing P/E ratio (price / TTM EPS from Eulerpool quarterly)",
    },
    "ev_ebitda": {
        "category": "fundamental_valuation",
        "lookback": "point-in-time",
        "description": "EV / EBIT (EBIT proxy for EBITDA; Eulerpool quarterly)",
    },
    "dividend_yield": {
        "category": "fundamental_valuation",
        "lookback": "point-in-time",
        "description": "Trailing dividend yield (TTM dividend / price)",
    },
    "revenue_growth": {
        "category": "fundamental_growth",
        "lookback": "point-in-time",
        "description": "Year-over-year revenue growth (quarterly vs 4Q prior)",
    },
    "earnings_growth": {
        "category": "fundamental_growth",
        "lookback": "point-in-time",
        "description": "Year-over-year earnings growth (quarterly vs 4Q prior)",
    },
    "profit_margin": {
        "category": "fundamental_quality",
        "lookback": "point-in-time",
        "description": "Net profit margin (earnings_margin from Eulerpool quarterly)",
    },
    "market_cap_log": {
        "category": "fundamental_size",
        "lookback": "point-in-time",
        "description": "Log of market cap (price × shares at cutoff)",
    },
    "ebit_margin": {
        "category": "fundamental_quality",
        "lookback": "point-in-time",
        "description": "EBIT margin (Eulerpool quarterly)",
    },
    "gross_margin": {
        "category": "fundamental_quality",
        "lookback": "point-in-time",
        "description": "Gross margin (Eulerpool quarterly; yfinance grossMargins fallback)",
    },
    "net_debt_by_ebit": {
        "category": "fundamental_leverage",
        "lookback": "point-in-time",
        "description": "Net debt / EBIT leverage ratio (replaces debt_equity)",
    },
    # ── Seasonality ──
    "month_sin": {
        "category": "seasonality",
        "lookback": "0d",
        "description": "Cyclic month encoding: sin(2π × month / 12)",
    },
    "month_cos": {
        "category": "seasonality",
        "lookback": "0d",
        "description": "Cyclic month encoding: cos(2π × month / 12)",
    },
    # ── Interaction / derived ──
    "mom_3m_x_ebit_margin": {
        "category": "interaction",
        "lookback": "63d + PIT",
        "description": "Interaction: 3-month momentum × EBIT margin (quality-momentum blend)",
    },
    "inv_pe_x_mom_6m": {
        "category": "interaction",
        "lookback": "126d + PIT",
        "description": "Interaction: inverse P/E × 6-month momentum (value-momentum blend)",
    },
    "pe_vs_sector_median": {
        "category": "sector_relative",
        "lookback": "point-in-time",
        "description": "P/E ratio relative to sector median (sector-normalised valuation)",
    },
    # ── Macro ──
    "spi_mom_3m": {
        "category": "macro",
        "lookback": "63d",
        "description": "Swiss Market Index (^SSMI) 3-month momentum (SPI proxy)",
    },
    "spi_mom_6m": {
        "category": "macro",
        "lookback": "126d",
        "description": "Swiss Market Index (^SSMI) 6-month momentum (SPI proxy)",
    },
}


def get_feature_registry_df() -> pd.DataFrame:
    """Return the feature registry as a DataFrame for display and export."""
    rows = []
    for name, info in FEATURE_REGISTRY.items():
        rows.append({"feature": name, **info})
    return pd.DataFrame(rows).set_index("feature")

_YF_FUNDAMENTAL_MAP: dict[str, str] = {
    "trailingPE": "pe_ratio",
    "enterpriseToEbitda": "ev_ebitda",
    "dividendYield": "dividend_yield",
    "revenueGrowth": "revenue_growth",
    "earningsGrowth": "earnings_growth",
    "profitMargins": "profit_margin",
    "grossMargins": "gross_margin",
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
    """Extract fundamental features from a yfinance ``.info`` dict.

    This is the **fallback** path used when Eulerpool PIT data is unavailable.
    Features that require Eulerpool-only fields (``ebit_margin``,
    ``net_debt_by_ebit``) are set to ``NaN``.
    """
    feats: dict[str, float] = {}
    for yf_key, feat_name in _YF_FUNDAMENTAL_MAP.items():
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

    feats["ebit_margin"] = float("nan")
    feats["net_debt_by_ebit"] = float("nan")

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


def build_fundamental_features_pit(
    eulerpool_quarterly: dict[str, list[dict]],
    eulerpool_profiles: dict[str, dict],
    cutoff_date: str,
    ohlcv_by_ticker: dict[str, pd.DataFrame],
    publication_lag_days: int = 0,
) -> pd.DataFrame:
    """Point-in-time fundamental features from Eulerpool quarterly data.

    For each ticker, retrieves the close price at *cutoff_date* from OHLCV and
    calls :func:`~src.eulerpool_fundamentals.extract_pit_features` to produce
    features anchored to the latest quarterly filing before the cutoff.

    Parameters
    ----------
    eulerpool_quarterly
        Ticker -> list of quarterly records (sorted by ``period`` ascending).
    eulerpool_profiles
        Ticker -> profile dict (used for sector info elsewhere, passed through
        for interface consistency).
    cutoff_date
        Features are derived from the latest quarter with ``period <= cutoff_date``
        (after subtracting *publication_lag_days* when that is positive).
    ohlcv_by_ticker
        Ticker -> OHLCV DataFrame; used to get the close price at *cutoff_date*.
    publication_lag_days
        Days to shift the fundamental cutoff earlier (reporting delay).  Price
        still uses *cutoff_date*.

    Returns
    -------
    pd.DataFrame
        One row per ticker, columns :data:`FUNDAMENTAL_FEATURE_NAMES`.
    """
    from src.eulerpool_fundamentals import extract_pit_features, get_price_at_cutoff

    nan_row = {name: float("nan") for name in FUNDAMENTAL_FEATURE_NAMES}
    rows: dict[str, dict[str, float]] = {}

    for ticker in eulerpool_quarterly:
        qr = eulerpool_quarterly.get(ticker, [])
        ohlcv = ohlcv_by_ticker.get(ticker)

        if not qr or ohlcv is None or ohlcv.empty:
            rows[ticker] = nan_row.copy()
            continue

        try:
            price = get_price_at_cutoff(ohlcv, cutoff_date)
            rows[ticker] = extract_pit_features(
                qr, cutoff_date, price, publication_lag_days=publication_lag_days
            )
        except Exception:
            logger.warning(
                "PIT fundamental extraction failed for %s at %s",
                ticker, cutoff_date, exc_info=True,
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
    eulerpool_profiles: dict[str, dict] | None = None,
) -> pd.DataFrame:
    """Interaction, sector-relative valuation, and macro regime columns.

    Expects *feature_matrix* to contain the columns needed for interactions
    (``mom_3m``, ``ebit_margin``, ``pe_ratio``, ``mom_6m``) when fundamentals
    were merged.

    Parameters
    ----------
    feature_matrix
        Technical (+ optional fundamental) rows, index = ticker.
    fundamentals_by_ticker
        Ticker -> yfinance ``.info`` dict; used for sector-relative P/E when
        *eulerpool_profiles* is not provided.
    cutoff_date
        Passed to :func:`compute_macro_momentum` for the macro series.
    macro_ohlcv
        Optional OHLCV for the benchmark index.  If omitted, macro columns are NaN.
    eulerpool_profiles
        Ticker -> Eulerpool profile dict.  When provided, takes precedence over
        *fundamentals_by_ticker* for sector lookup.

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

    # ── Interactions: mom_3m × ebit_margin; (1/pe) × mom_6m ────────────────
    if all(c in feature_matrix.columns for c in ("mom_3m", "ebit_margin")):
        m3 = feature_matrix["mom_3m"].astype(float)
        em = feature_matrix["ebit_margin"].astype(float)
        derived["mom_3m_x_ebit_margin"] = m3 * em
    if all(c in feature_matrix.columns for c in ("pe_ratio", "mom_6m")):
        pe = feature_matrix["pe_ratio"].astype(float)
        m6 = feature_matrix["mom_6m"].astype(float)
        inv_pe = 1.0 / pe.where((pe > 0) & np.isfinite(pe))
        derived["inv_pe_x_mom_6m"] = (inv_pe * m6).astype(float)

    # ── Sector-relative P/E ───────────────────────────────────────────────
    sector_source = eulerpool_profiles or fundamentals_by_ticker
    if sector_source is not None and "pe_ratio" in feature_matrix.columns:
        pe_s = feature_matrix["pe_ratio"].astype(float)
        med_by_ticker = pd.Series(np.nan, index=idx, dtype=float)
        sectors: list[str | None] = []
        for t in idx:
            info = sector_source.get(t) or {}
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
    eulerpool_quarterly: dict[str, list[dict]] | None = None,
    eulerpool_profiles: dict[str, dict] | None = None,
    publication_lag_days: int = 0,
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
        Optional ticker -> yfinance ``.info`` dict.  Used as **fallback** when
        *eulerpool_quarterly* is not provided.
    macro_ohlcv
        Optional OHLCV for the Swiss benchmark index.  Overrides lookup of
        ``ohlcv_by_ticker[MACRO_BENCHMARK_TICKER]``.
    eulerpool_quarterly
        Optional ticker -> list of Eulerpool quarterly records (sorted by
        ``period``).  When provided, fundamentals are computed point-in-time
        via :func:`build_fundamental_features_pit`.
    eulerpool_profiles
        Optional ticker -> Eulerpool company profile dict.  Used for sector
        lookup in derived features when Eulerpool data is active.
    publication_lag_days
        Passed to :func:`build_fundamental_features_pit` when Eulerpool
        quarterly data is used.

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

    if eulerpool_quarterly is not None:
        fund = build_fundamental_features_pit(
            eulerpool_quarterly,
            eulerpool_profiles or {},
            cutoff_date,
            stock_ohlcv,
            publication_lag_days=publication_lag_days,
        )
        tech = tech.join(fund, how="left")
    elif fundamentals_by_ticker is not None:
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
        eulerpool_profiles=eulerpool_profiles,
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


def drop_correlated_features_train_test(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    threshold: float = 0.85,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Drop correlated columns using *X_train* only; apply the same drops to *X_test*.

    Pearson correlations and the drop decision are computed on the training
    matrix only, avoiding holdout leakage from a full-panel correlation matrix.
    """
    reduced_train, dropped = drop_correlated_features(X_train, threshold=threshold)
    cols = list(reduced_train.columns)
    X_test_out = X_test.reindex(columns=cols)
    return reduced_train, X_test_out, dropped
