"""Hybrid L/S strategy: Classification Long + Regression Short.

Classification selects the annual Winner pool (long, rebalanced yearly),
while Regression identifies monthly Bottom-N stocks (short, rebalanced
monthly).  This asymmetric rebalancing exploits each model's comparative
advantage — classification is stronger at identifying long-term Winners,
regression is better at ranking short-term Losers (Sharpe 0.93 on short
side in robustness tests).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

try:
    from config import OOS_FEATURE_CUTOFF_DATE, OOS_YEAR
except ImportError:
    OOS_FEATURE_CUTOFF_DATE = "2024-12-31"
    OOS_YEAR = 2025

logger = logging.getLogger(__name__)

__all__ = [
    "HybridBacktestResult",
    "MonthlyShortSnapshot",
    "hybrid_summary_table",
    "plot_hybrid_cumulative",
    "plot_hybrid_short_ic",
    "plot_hybrid_short_turnover",
    "predict_monthly_returns_for_year",
    "run_hybrid_backtest",
    "run_hybrid_strategy",
]


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class MonthlyShortSnapshot:
    """Single short-leg rebalancing period."""

    cutoff_date: pd.Timestamp
    holding_end: pd.Timestamp
    predicted_returns: pd.Series
    short_tickers: list[str]
    ic: float
    n_universe: int
    turnover: float


@dataclass
class HybridBacktestResult:
    """Full hybrid strategy backtest result.

    Attributes
    ----------
    long_tickers
        Predicted Winners held for the full OOS year.
    long_weights
        Portfolio weights for the long leg (normalised to 1.0).
    short_snapshots
        Per-month short-leg rebalancing data.
    daily_returns
        Strategy name → daily return Series (``long``, ``short``,
        ``long_short``, ``benchmark``).
    strategy_metrics
        Strategy name → performance dict (cumulative, Sharpe, …).
    oos_year
        Calendar year of the backtest.
    costs_bps_long
        One-way transaction costs applied to the long leg (bps).
    costs_bps_short
        One-way transaction costs applied to the short leg (bps).
    bottom_n
        Number of short positions per month.
    classification_cutoff
        Feature cutoff date used for classification.
    """

    long_tickers: list[str]
    long_weights: pd.Series
    short_snapshots: list[MonthlyShortSnapshot]
    daily_returns: dict[str, pd.Series]
    strategy_metrics: dict[str, dict[str, float]]
    oos_year: int
    costs_bps_long: float
    costs_bps_short: float
    bottom_n: int
    classification_cutoff: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _spearman_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman rank correlation; NaN when < 5 finite pairs."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(yt) & np.isfinite(yp)
    if mask.sum() < 5:
        return float("nan")
    corr, _ = spearmanr(yt[mask], yp[mask])
    return float(corr)


def _compute_metrics(daily_returns: pd.Series) -> dict[str, float]:
    """Cumulative return, annualised Sharpe, max drawdown."""
    nan_result: dict[str, float] = {
        "cumulative_return": float("nan"),
        "annualized_return": float("nan"),
        "volatility": float("nan"),
        "sharpe_ratio": float("nan"),
        "max_drawdown": float("nan"),
        "n_trading_days": 0,
    }
    dr = daily_returns.dropna()
    if dr.empty:
        return nan_result

    n = len(dr)
    cumulative = float((1 + dr).prod() - 1)
    ann_factor = 252 / n if n > 0 else 1.0
    ann_return = float((1 + cumulative) ** ann_factor - 1)
    volatility = float(dr.std() * np.sqrt(252))
    sharpe = ann_return / volatility if volatility > 0 else float("nan")

    wealth = (1 + dr).cumprod()
    running_max = wealth.cummax()
    drawdown = (wealth - running_max) / running_max
    max_dd = float(drawdown.min())

    return {
        "cumulative_return": cumulative,
        "annualized_return": ann_return,
        "volatility": volatility,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "n_trading_days": n,
    }


def _one_way_turnover(old_tickers: list[str], new_tickers: list[str]) -> float:
    """Fraction of new positions absent in the previous portfolio."""
    if not new_tickers:
        return 0.0
    if not old_tickers:
        return 1.0
    return len(set(new_tickers) - set(old_tickers)) / len(new_tickers)


def _build_close_matrix(
    ohlcv_by_ticker: dict[str, pd.DataFrame],
    tickers: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Aligned daily close prices for *tickers* in ``[start, end]``."""
    series: list[pd.Series] = []
    for t in tickers:
        df = ohlcv_by_ticker.get(t)
        if df is None or df.empty or "Close" not in df.columns:
            continue
        close = df["Close"].copy()
        close.index = pd.to_datetime(close.index)
        close = close.sort_index().loc[start:end].dropna()
        if len(close) < 2:
            continue
        close.name = t
        series.append(close)
    if not series:
        return pd.DataFrame()
    return pd.concat(series, axis=1)


def _normalize_cutoff_key(key: Any) -> str:
    """Convert cutoff to ISO date string ``YYYY-MM-DD``."""
    if isinstance(key, pd.Timestamp):
        return key.strftime("%Y-%m-%d")
    return str(key)[:10]


def _month_end_cutoffs_for_year(
    ohlcv_by_ticker: dict[str, pd.DataFrame],
    year: int,
) -> list[pd.Timestamp]:
    """Last trading day per month from Dec(*year*-1) through Nov(*year*).

    Dec(year-1) cutoff → hold Jan(year), …, Nov(year) cutoff → hold Dec(year).
    """
    all_dates: set[pd.Timestamp] = set()
    for df in ohlcv_by_ticker.values():
        if df is not None and not df.empty:
            idx = pd.to_datetime(df.index)
            all_dates.update(idx.tolist())
    if not all_dates:
        return []

    idx = pd.DatetimeIndex(sorted(all_dates))
    cutoffs: list[pd.Timestamp] = []
    months = [(year - 1, 12)] + [(year, m) for m in range(1, 12)]

    for y, m in months:
        start = pd.Timestamp(year=y, month=m, day=1)
        end = start + pd.offsets.MonthEnd(0)
        mask = (idx >= start) & (idx <= end)
        if mask.any():
            cutoffs.append(idx[mask].max())

    return cutoffs


# ---------------------------------------------------------------------------
# Convenience: predict monthly returns for an OOS year
# ---------------------------------------------------------------------------


def predict_monthly_returns_for_year(
    regression_result: Any,
    ohlcv_by_ticker: dict[str, pd.DataFrame],
    year: int,
    fundamentals_by_ticker: dict[str, dict[str, Any]] | None = None,
) -> dict[str, pd.Series]:
    """Predict 1-month forward returns at each month-end cutoff for *year*.

    Generates cutoffs from Dec(*year*-1) through Nov(*year*), builds features
    at each cutoff, and applies the regression model.

    Parameters
    ----------
    regression_result
        :class:`~regression_model.RegressionTrainResult`.
    ohlcv_by_ticker
        Ticker → OHLCV DataFrame.
    year
        Calendar year to predict.
    fundamentals_by_ticker
        Optional ticker → yfinance ``.info`` dict.

    Returns
    -------
    dict
        ``cutoff_date`` (ISO string) → pd.Series of predicted returns
        indexed by ticker.
    """
    try:
        from src.features import build_feature_matrix
        from src.regression_model import predict_returns
    except ImportError:
        from features import build_feature_matrix
        from regression_model import predict_returns

    cutoffs = _month_end_cutoffs_for_year(ohlcv_by_ticker, year)
    result: dict[str, pd.Series] = {}

    for cd in cutoffs:
        cd_str = cd.strftime("%Y-%m-%d")
        X = build_feature_matrix(ohlcv_by_ticker, cd_str, fundamentals_by_ticker)
        if X.empty:
            logger.warning(
                "predict_monthly_returns_for_year: empty features for %s", cd_str,
            )
            continue
        pred = predict_returns(regression_result, X)
        result[cd_str] = pred

    logger.info(
        "predict_monthly_returns_for_year(%d): %d/%d cutoffs produced predictions",
        year,
        len(result),
        len(cutoffs),
    )
    return result


# ---------------------------------------------------------------------------
# Core hybrid backtest
# ---------------------------------------------------------------------------


def run_hybrid_backtest(
    ohlcv_by_ticker: dict[str, pd.DataFrame],
    long_tickers: list[str],
    predicted_returns_by_month: dict[str, pd.Series],
    *,
    long_weights: pd.Series | None = None,
    bottom_n: int = 10,
    oos_year: int | None = None,
    costs_bps_long: float = 50.0,
    costs_bps_short: float = 30.0,
    forward_returns_df: pd.DataFrame | None = None,
    classification_cutoff: str | None = None,
) -> HybridBacktestResult:
    """Run hybrid L/S backtest: classification Long + regression Short.

    The long leg holds *long_tickers* for the full OOS year (annual
    rebalancing).  The short leg picks Bottom-*bottom_n* from monthly
    regression predictions (monthly rebalancing).

    Parameters
    ----------
    ohlcv_by_ticker
        Ticker → OHLCV DataFrame.
    long_tickers
        Predicted Winners from classification (held all year).
    predicted_returns_by_month
        ``cutoff_date`` (ISO string) → pd.Series of predicted 1-month
        forward returns (regression output).
    long_weights
        Optional confidence weights for the long leg (e.g. P(Winner)).
        Equal-weight when ``None``.
    bottom_n
        Number of bottom-ranked stocks to short each month.
    oos_year
        OOS calendar year; defaults to ``config.OOS_YEAR``.
    costs_bps_long
        One-way transaction costs for the long leg (applied on entry and
        exit, suitable for annual holding with wide small-cap spreads).
    costs_bps_short
        One-way costs for the short leg (turnover-adjusted per month).
    forward_returns_df
        Optional, for IC calculation on the short side.  Output of
        :func:`~regression_targets.compute_monthly_forward_returns`.
    classification_cutoff
        Cutoff date used for classification features (for metadata).

    Returns
    -------
    HybridBacktestResult
    """
    yr = OOS_YEAR if oos_year is None else oos_year
    cls_cutoff = classification_cutoff or f"{yr - 1}-12-31"

    if not long_tickers:
        logger.warning("run_hybrid_backtest: empty long_tickers list")

    # -- Normalise long weights --
    if long_weights is not None and long_tickers:
        lw = long_weights.reindex(long_tickers).fillna(0.0)
        lw_sum = lw.sum()
        lw = lw / lw_sum if lw_sum > 0 else pd.Series(
            1.0 / len(long_tickers), index=long_tickers,
        )
    elif long_tickers:
        lw = pd.Series(1.0 / len(long_tickers), index=long_tickers)
    else:
        lw = pd.Series(dtype=float)
    lw.name = "weight"

    # -- Sort monthly predictions chronologically --
    pred_by_cd: dict[str, pd.Series] = {
        _normalize_cutoff_key(k): v
        for k, v in sorted(predicted_returns_by_month.items())
    }
    cutoff_strs = list(pred_by_cd.keys())
    if not cutoff_strs:
        raise ValueError("predicted_returns_by_month is empty")

    # -- Forward-return lookup for IC --
    fwd_lookup: dict[str, pd.Series] = {}
    if forward_returns_df is not None:
        for cd_ts, grp in forward_returns_df.groupby("cutoff_date"):
            key = _normalize_cutoff_key(cd_ts)
            fwd_lookup[key] = grp.set_index("ticker")["forward_1m_return"]

    # -- Holding-period boundaries for the short leg --
    cutoff_ts = [pd.Timestamp(c) for c in cutoff_strs]
    holding_ends: list[pd.Timestamp] = []
    for i in range(len(cutoff_ts)):
        if i + 1 < len(cutoff_ts):
            holding_ends.append(cutoff_ts[i + 1])
        else:
            nxt = cutoff_ts[i] + pd.offsets.MonthEnd(1)
            if nxt <= cutoff_ts[i]:
                nxt = cutoff_ts[i] + pd.offsets.MonthEnd(2)
            holding_ends.append(nxt)

    # -- Build global close matrix --
    year_start = pd.Timestamp(f"{yr}-01-01")
    year_end = pd.Timestamp(f"{yr}-12-31")
    # Include late-Dec of prior year so pct_change covers the first OOS day
    safe_start = pd.Timestamp(f"{yr - 1}-12-15")
    span_start = min(cutoff_ts[0], safe_start)
    span_end = max(holding_ends[-1], year_end)

    all_tickers = sorted(
        set(long_tickers)
        | {t for ps in pred_by_cd.values() for t in ps.dropna().index}
    )
    close_matrix = _build_close_matrix(
        ohlcv_by_ticker, all_tickers, span_start, span_end,
    )
    daily_ret_matrix = (
        close_matrix.pct_change() if not close_matrix.empty else pd.DataFrame()
    )

    # ---- Long leg (annual, full year) ----
    long_daily = pd.Series(dtype=float, name="long")
    if long_tickers and not daily_ret_matrix.empty:
        yr_mask = (
            (daily_ret_matrix.index > year_start)
            & (daily_ret_matrix.index <= year_end)
        )
        yr_rets = daily_ret_matrix.loc[yr_mask]
        la = [t for t in long_tickers if t in yr_rets.columns]
        if la and not yr_rets.empty:
            w = lw.reindex(la).fillna(0.0)
            w_sum = w.sum()
            if w_sum > 0:
                w = w / w_sum
            else:
                w = pd.Series(1.0 / len(la), index=la)
            long_daily = (yr_rets[la] * w).sum(axis=1)

            if costs_bps_long > 0 and not long_daily.empty:
                cost_frac = costs_bps_long / 10_000.0
                long_daily = long_daily.copy()
                long_daily.iloc[0] -= cost_frac
                long_daily.iloc[-1] -= cost_frac
            long_daily.name = "long"

    # ---- Short leg (monthly rebalancing) ----
    short_snapshots: list[MonthlyShortSnapshot] = []
    short_parts: list[pd.Series] = []
    prev_short: list[str] = []
    cost_frac_short = costs_bps_short / 10_000.0
    long_set = set(long_tickers)

    for i, cd_str in enumerate(cutoff_strs):
        ct = cutoff_ts[i]
        he = holding_ends[i]
        pred = pred_by_cd[cd_str].dropna().sort_values(ascending=False)

        short_tk = list(pred.tail(bottom_n).index)
        # Exclude long-held tickers to avoid being both long and short
        short_tk = [t for t in short_tk if t not in long_set]

        # IC
        actual_ret = fwd_lookup.get(cd_str)
        ic = float("nan")
        if actual_ret is not None:
            common = pred.index.intersection(actual_ret.index)
            if len(common) >= 5:
                ic = _spearman_ic(
                    actual_ret.reindex(common).values,
                    pred.reindex(common).values,
                )

        to_s = _one_way_turnover(prev_short, short_tk)

        if not daily_ret_matrix.empty:
            mask = (daily_ret_matrix.index > ct) & (daily_ret_matrix.index <= he)
            pr = daily_ret_matrix.loc[mask]
            sa = [t for t in short_tk if t in pr.columns]
            if sa and not pr.empty:
                short_dr = -pr[sa].mean(axis=1)
                if cost_frac_short > 0 and not short_dr.empty:
                    short_dr = short_dr.copy()
                    short_dr.iloc[0] -= cost_frac_short * to_s
                short_parts.append(short_dr)

        short_snapshots.append(MonthlyShortSnapshot(
            cutoff_date=ct,
            holding_end=he,
            predicted_returns=pred,
            short_tickers=short_tk,
            ic=ic,
            n_universe=len(pred),
            turnover=to_s,
        ))
        prev_short = short_tk

    short_daily = pd.Series(dtype=float, name="short")
    if short_parts:
        short_daily = pd.concat(short_parts).sort_index()
        short_daily = short_daily[~short_daily.index.duplicated(keep="last")]
        short_daily.name = "short"

    # ---- Long/Short combined (50/50 allocation) ----
    ls_daily = pd.Series(dtype=float, name="long_short")
    if not long_daily.empty and not short_daily.empty:
        aligned = pd.concat(
            [long_daily.rename("l"), short_daily.rename("s")], axis=1,
        ).dropna()
        if not aligned.empty:
            ls_daily = (aligned["l"] + aligned["s"]) / 2.0
            ls_daily.name = "long_short"
    elif not long_daily.empty:
        ls_daily = long_daily.copy()
        ls_daily.name = "long_short"
    elif not short_daily.empty:
        ls_daily = short_daily.copy()
        ls_daily.name = "long_short"

    # ---- Benchmark (equal-weight universe) ----
    benchmark_daily = pd.Series(dtype=float, name="benchmark")
    if not daily_ret_matrix.empty:
        yr_mask = (
            (daily_ret_matrix.index > year_start)
            & (daily_ret_matrix.index <= year_end)
        )
        yr_rets = daily_ret_matrix.loc[yr_mask]
        ba = [t for t in all_tickers if t in yr_rets.columns]
        if ba:
            benchmark_daily = yr_rets[ba].mean(axis=1)
            benchmark_daily.name = "benchmark"

    # ---- Aggregate ----
    daily_returns = {
        "long": long_daily,
        "short": short_daily,
        "long_short": ls_daily,
        "benchmark": benchmark_daily,
    }
    strategy_metrics = {n: _compute_metrics(dr) for n, dr in daily_returns.items()}

    if short_snapshots:
        turnovers = [s.turnover for s in short_snapshots]
        strategy_metrics["short"]["avg_monthly_turnover"] = float(
            np.nanmean(turnovers),
        )

    logger.info(
        "Hybrid backtest %d: %d long tickers (annual), "
        "Bottom-%d short (monthly, %d months)",
        yr,
        len(long_tickers),
        bottom_n,
        len(short_snapshots),
    )
    for name, m in strategy_metrics.items():
        logger.info(
            "  %-12s cum=%.3f  ann=%.3f  sharpe=%.2f  maxDD=%.3f  (%d days)",
            name,
            m.get("cumulative_return", float("nan")),
            m.get("annualized_return", float("nan")),
            m.get("sharpe_ratio", float("nan")),
            m.get("max_drawdown", float("nan")),
            m.get("n_trading_days", 0),
        )
    if short_snapshots:
        ics = [s.ic for s in short_snapshots if np.isfinite(s.ic)]
        if ics:
            logger.info(
                "  Short IC: mean=%.4f  std=%.4f  positive=%.0f%%",
                float(np.mean(ics)),
                float(np.std(ics)),
                100.0 * sum(1 for v in ics if v > 0) / len(ics),
            )

    return HybridBacktestResult(
        long_tickers=long_tickers,
        long_weights=lw,
        short_snapshots=short_snapshots,
        daily_returns=daily_returns,
        strategy_metrics=strategy_metrics,
        oos_year=yr,
        costs_bps_long=costs_bps_long,
        costs_bps_short=costs_bps_short,
        bottom_n=bottom_n,
        classification_cutoff=cls_cutoff,
    )


# ---------------------------------------------------------------------------
# High-level convenience: models → backtest
# ---------------------------------------------------------------------------


def run_hybrid_strategy(
    ohlcv_by_ticker: dict[str, pd.DataFrame],
    classification_result: Any,
    regression_result: Any,
    fundamentals_by_ticker: dict[str, dict[str, Any]] | None = None,
    *,
    oos_year: int | None = None,
    classification_cutoff: str | None = None,
    bottom_n: int = 10,
    costs_bps_long: float = 50.0,
    costs_bps_short: float = 30.0,
    top_n: int | None = None,
    sector_map: dict[str, str] | None = None,
    max_sector_weight: float | None = None,
    forward_returns_df: pd.DataFrame | None = None,
) -> HybridBacktestResult:
    """End-to-end hybrid strategy from trained models to backtest result.

    1. Build classification features at annual cutoff → predict Winners → Long.
    2. Generate monthly regression predictions → Bottom-N → Short.
    3. Simulate daily returns with asymmetric rebalancing.

    Parameters
    ----------
    ohlcv_by_ticker
        Ticker → OHLCV DataFrame.
    classification_result
        :class:`~model.TrainResult` or :class:`~model.MultiQuarterEnsembleResult`.
    regression_result
        :class:`~regression_model.RegressionTrainResult`.
    fundamentals_by_ticker
        Optional ticker → yfinance ``.info`` dict for feature building.
    oos_year
        Calendar year for the backtest; defaults to ``config.OOS_YEAR``.
    classification_cutoff
        Feature cutoff for classification; defaults to ``<oos_year-1>-12-31``.
    bottom_n
        Monthly short positions from regression.
    costs_bps_long
        Long leg transaction costs (annual entry/exit).
    costs_bps_short
        Short leg transaction costs (monthly, turnover-adjusted).
    top_n
        Only go long on top-N Winners by probability.
    sector_map
        Ticker → sector for diversification constraints.
    max_sector_weight
        Max portfolio weight per sector in the long leg.
    forward_returns_df
        Optional, for short-side IC tracking.

    Returns
    -------
    HybridBacktestResult
    """
    try:
        from src.backtest import compute_portfolio_weights
        from src.features import build_feature_matrix
        from src.model import predict as cls_predict
        from src.model import predict_proba as cls_predict_proba
    except ImportError:
        from backtest import compute_portfolio_weights
        from features import build_feature_matrix
        from model import predict as cls_predict
        from model import predict_proba as cls_predict_proba

    yr = OOS_YEAR if oos_year is None else oos_year
    cutoff = classification_cutoff or f"{yr - 1}-12-31"

    # 1. Classification → Long tickers
    X_cls = build_feature_matrix(ohlcv_by_ticker, cutoff, fundamentals_by_ticker)
    preds = cls_predict(classification_result, X_cls)
    winners = list(preds[preds == "Winners"].index)

    proba_weights: pd.Series | None = None
    try:
        proba_df = cls_predict_proba(classification_result, X_cls)
        if "Winners" in proba_df.columns:
            proba_weights = proba_df["Winners"]
    except Exception:
        logger.debug("Could not extract proba_weights from classification model")

    long_w = compute_portfolio_weights(
        winners,
        proba_weights=proba_weights,
        sector_map=sector_map,
        max_sector_weight=max_sector_weight,
        top_n=top_n,
    )
    long_tickers = list(long_w.index)

    logger.info(
        "Hybrid Long: %d Winners predicted (cutoff %s), %d after filters",
        len(winners),
        cutoff,
        len(long_tickers),
    )

    # 2. Regression → monthly predictions for Short
    monthly_preds = predict_monthly_returns_for_year(
        regression_result,
        ohlcv_by_ticker,
        yr,
        fundamentals_by_ticker,
    )

    # 3. Run backtest
    return run_hybrid_backtest(
        ohlcv_by_ticker,
        long_tickers,
        monthly_preds,
        long_weights=long_w,
        bottom_n=bottom_n,
        oos_year=yr,
        costs_bps_long=costs_bps_long,
        costs_bps_short=costs_bps_short,
        forward_returns_df=forward_returns_df,
        classification_cutoff=cutoff,
    )


# ---------------------------------------------------------------------------
# Summary & display
# ---------------------------------------------------------------------------


def hybrid_summary_table(result: HybridBacktestResult) -> pd.DataFrame:
    """Tabular comparison of all strategy legs vs benchmark.

    Parameters
    ----------
    result
        A :class:`HybridBacktestResult`.

    Returns
    -------
    pd.DataFrame
        One row per strategy with performance metrics.
    """
    labels = {
        "long": f"Long (Classification, {len(result.long_tickers)} Winners)",
        "short": f"Short (Regression, Bottom-{result.bottom_n})",
        "long_short": "Hybrid Long/Short",
        "benchmark": "Benchmark (EW)",
    }
    rows: list[dict[str, Any]] = []
    for key, label in labels.items():
        m = result.strategy_metrics.get(key, {})
        row: dict[str, Any] = {"Strategy": label}
        row.update(m)
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Strategy")
    float_cols = df.select_dtypes(include="number").columns
    for c in float_cols:
        if c != "n_trading_days":
            df[c] = df[c].map(lambda v: f"{v:.4f}" if np.isfinite(v) else "–")
    return df


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def plot_hybrid_cumulative(
    result: HybridBacktestResult,
    *,
    title: str | None = None,
    ax: Any | None = None,
) -> Any:
    """Equity curves for all strategy legs and benchmark.

    Parameters
    ----------
    result
        :class:`HybridBacktestResult`.
    """
    import matplotlib.pyplot as plt

    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(12, 5))

    if title is None:
        title = (
            f"Hybrid Strategy — {result.oos_year}  "
            f"(Long: {len(result.long_tickers)} Winners, "
            f"Short: Bottom-{result.bottom_n})"
        )

    colours = {
        "long": "#2ecc71",
        "short": "#e74c3c",
        "long_short": "#3498db",
        "benchmark": "#7f8c8d",
    }
    labels = {
        "long": f"Long (Classification, {len(result.long_tickers)})",
        "short": f"Short (Regression, B-{result.bottom_n})",
        "long_short": "Hybrid L/S",
        "benchmark": "Benchmark (EW)",
    }

    for name, dr in result.daily_returns.items():
        if dr.empty:
            continue
        wealth = (1 + dr).cumprod()
        ax.plot(
            wealth.index,
            wealth.values,
            label=labels.get(name, name),
            color=colours.get(name),
            linewidth=1.5 if name != "benchmark" else 1.2,
            linestyle="-" if name != "benchmark" else "--",
        )

    ax.axhline(1.0, color="gray", linewidth=0.7, linestyle=":")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of 1 CHF")
    ax.legend()

    if created:
        plt.tight_layout()
    return ax


def plot_hybrid_short_turnover(
    result: HybridBacktestResult,
    *,
    title: str = "Monthly Short-Leg Turnover",
    ax: Any | None = None,
) -> Any:
    """Bar chart of short-leg turnover per month.

    Parameters
    ----------
    result
        :class:`HybridBacktestResult`.
    """
    import matplotlib.pyplot as plt

    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(12, 4))

    month_labels = [
        s.cutoff_date.strftime("%Y-%m") for s in result.short_snapshots
    ]
    turnovers = [s.turnover for s in result.short_snapshots]

    colors = ["#e74c3c" if t > 0.5 else "#f39c12" for t in turnovers]
    ax.bar(
        range(len(turnovers)),
        turnovers,
        color=colors,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_xticks(range(len(month_labels)))
    ax.set_xticklabels(month_labels, rotation=45, ha="right", fontsize=8)

    if turnovers:
        avg = float(np.nanmean(turnovers))
        ax.axhline(
            avg,
            color="#3498db",
            linestyle="--",
            linewidth=1.5,
            label=f"Avg = {avg:.0%}",
        )

    ax.set_ylabel("Turnover")
    ax.set_ylim(0, 1.1)
    ax.set_title(title)
    ax.legend(loc="upper right")

    if created:
        plt.tight_layout()
    return ax


def plot_hybrid_short_ic(
    result: HybridBacktestResult,
    *,
    title: str = "Monthly Short-Leg IC (Spearman)",
    ax: Any | None = None,
) -> Any:
    """Bar chart of per-month IC for the short leg.

    Parameters
    ----------
    result
        :class:`HybridBacktestResult`.
    """
    import matplotlib.pyplot as plt

    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(12, 4))

    month_labels = [
        s.cutoff_date.strftime("%Y-%m") for s in result.short_snapshots
    ]
    ic_values = [s.ic for s in result.short_snapshots]
    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in ic_values]

    ax.bar(
        range(len(ic_values)),
        ic_values,
        color=colors,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_xticks(range(len(month_labels)))
    ax.set_xticklabels(month_labels, rotation=45, ha="right", fontsize=8)

    finite_ics = [v for v in ic_values if np.isfinite(v)]
    if finite_ics:
        ic_mean = float(np.mean(finite_ics))
        ax.axhline(
            ic_mean,
            color="#3498db",
            linestyle="--",
            linewidth=1.5,
            label=f"Mean IC = {ic_mean:.3f}",
        )
    ax.axhline(0, color="grey", linestyle="-", linewidth=0.5)
    ax.set_ylabel("IC (Spearman)")
    ax.set_title(title)
    ax.legend(loc="upper right")

    if created:
        plt.tight_layout()
    return ax
