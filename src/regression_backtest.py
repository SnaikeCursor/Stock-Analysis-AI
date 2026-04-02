"""Monthly portfolio backtest for regression-based return prediction.

Each rebalancing date (month-end cutoff), the model's predicted returns rank
all stocks in the universe.  Top-N go long, Bottom-N go short, and an
equal-weight benchmark holds everything.  Daily portfolio returns are tracked
with turnover-adjusted transaction costs (default 30 bps one-way for the
regression leg; use ``hysteresis_buffer`` to keep names until rank falls outside
``top_n + buffer`` / ``bottom_n + buffer``, reducing churn).

Phase 4 portfolio construction (optional): predicted returns can drive
confidence weights (long: higher predicted return → higher weight; short:
lower predicted return → higher short weight), each name capped at
``max_position_weight`` (default 15% when confidence weighting is on), then
sector caps via ``max_sector_weight`` (e.g. 30%) using the same iterative
redistribution as :func:`backtest.compute_portfolio_weights`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

try:
    from config import RANDOM_SEED
except ImportError:
    RANDOM_SEED = 42

try:
    from src.backtest import _apply_sector_cap
except ImportError:
    from backtest import _apply_sector_cap

logger = logging.getLogger(__name__)

# Phase 4 defaults (plan): max single-name weight when confidence weighting is enabled.
DEFAULT_MAX_POSITION_WEIGHT: float = 0.15

# Realistic one-way costs for monthly small-cap regression backtests (plan Phase 2.2).
DEFAULT_REGRESSION_COSTS_BPS: float = 30.0
# Keep names until rank worsens beyond top_n + buffer (1-based rank; best = 1).
DEFAULT_HYSTERESIS_BUFFER: int = 5

__all__ = [
    "MonthlySnapshot",
    "RegressionBacktestResult",
    "backtest_summary_table",
    "plot_backtest_cumulative",
    "plot_monthly_ic",
    "plot_monthly_turnover",
    "plot_quantile_returns",
    "predict_all_months",
    "run_regression_backtest",
    "DEFAULT_REGRESSION_COSTS_BPS",
    "DEFAULT_HYSTERESIS_BUFFER",
    "DEFAULT_MAX_POSITION_WEIGHT",
]


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class MonthlySnapshot:
    """Single rebalancing period's data and diagnostics."""

    cutoff_date: pd.Timestamp
    holding_end: pd.Timestamp
    predicted_returns: pd.Series
    actual_returns: pd.Series | None
    long_tickers: list[str]
    short_tickers: list[str]
    ic: float
    n_universe: int
    turnover_long: float
    turnover_short: float


@dataclass
class RegressionBacktestResult:
    """Full monthly-rebalanced backtest result."""

    snapshots: list[MonthlySnapshot]
    daily_returns: dict[str, pd.Series]
    strategy_metrics: dict[str, dict[str, float]]
    quantile_monthly: pd.DataFrame
    summary_df: pd.DataFrame
    costs_bps: float
    top_n: int
    bottom_n: int
    hysteresis_buffer: int
    use_confidence_weights: bool = False
    max_position_weight: float | None = None
    max_sector_weight: float | None = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _spearman_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman rank correlation.  Returns NaN when < 5 finite pairs."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(yt) & np.isfinite(yp)
    if mask.sum() < 5:
        return float("nan")
    corr, _ = spearmanr(yt[mask], yp[mask])
    return float(corr)


def _compute_metrics(daily_returns: pd.Series) -> dict[str, float]:
    """Cumulative return, annualised Sharpe, max drawdown from daily returns."""
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
    """Fraction of new positions not present in the previous portfolio."""
    if not new_tickers:
        return 0.0
    if not old_tickers:
        return 1.0
    return len(set(new_tickers) - set(old_tickers)) / len(new_tickers)


def _apply_position_cap(weights: pd.Series, cap: float) -> pd.Series:
    """Cap each position at *cap* and redistribute excess among uncapped names."""
    w = weights.copy().astype(float)
    if w.empty:
        return w
    s = float(w.sum())
    if s <= 1e-15:
        return w
    w = w / s
    cap = float(cap)
    if cap <= 0 or cap >= 1.0:
        return w
    n = len(w)
    if cap * n < 1.0 - 1e-9:
        logger.warning(
            "Position cap %.0f%% × %d names = %.0f%% < 100%%; relaxing to equal weight",
            cap * 100,
            n,
            cap * n * 100,
        )
        return pd.Series(1.0 / n, index=w.index, dtype=float)
    for _ in range(100):
        over = w > cap + 1e-12
        if not over.any():
            break
        excess = float((w[over] - cap).sum())
        w.loc[over] = cap
        under = ~over
        if not under.any():
            break
        u_sum = float(w[under].sum())
        if u_sum <= 1e-15:
            break
        w.loc[under] = w[under] + (w[under] / u_sum) * excess
    s2 = float(w.sum())
    if s2 > 1e-15:
        w = w / s2
    return w


def _compute_leg_weights(
    pred: pd.Series,
    tickers: list[str],
    *,
    leg: str,
    use_confidence_weights: bool,
    max_position_weight: float | None,
    sector_map: dict[str, str] | None,
    max_sector_weight: float | None,
) -> pd.Series:
    """Normalised weights for one leg (long or short). Order: EW/confidence → position cap → sector cap."""
    if not tickers:
        return pd.Series(dtype=float, name="weight")
    valid = [t for t in tickers if t in pred.index and np.isfinite(float(pred.loc[t]))]
    if not valid:
        return pd.Series(dtype=float, name="weight")
    if use_confidence_weights:
        p = pred.reindex(valid).astype(float)
        if leg == "long":
            w_raw = p - p.min() + 1e-12
        elif leg == "short":
            w_raw = p.max() - p + 1e-12
        else:
            raise ValueError("leg must be 'long' or 'short'")
        if float(w_raw.sum()) <= 1e-15:
            w = pd.Series(1.0 / len(valid), index=valid)
        else:
            w = w_raw / float(w_raw.sum())
    else:
        w = pd.Series(1.0 / len(valid), index=valid)

    if max_position_weight is not None:
        w = _apply_position_cap(w, max_position_weight)

    if max_sector_weight is not None and sector_map is not None:
        w = _apply_sector_cap(w, sector_map, max_sector_weight)

    w.name = "weight"
    return w


def _daily_weighted_returns(
    price_returns: pd.DataFrame,
    weights: pd.Series,
) -> pd.Series:
    """Daily portfolio return with column weights (must sum to 1)."""
    cols = [t for t in weights.index if t in price_returns.columns]
    if not cols:
        return pd.Series(dtype=float)
    w = weights.reindex(cols).fillna(0.0).astype(float)
    s = float(w.sum())
    if s <= 1e-15:
        return pd.Series(dtype=float)
    w = w / s
    return (price_returns[cols] * w).sum(axis=1)


def _select_long_with_hysteresis(
    pred: pd.Series,
    prev_long: list[str],
    top_n: int,
    hysteresis_buffer: int,
) -> list[str]:
    """Top-*top_n* long names; retain prior holdings while rank stays ≤ top_n + buffer.

    Rank 1 = highest predicted return.  When *hysteresis_buffer* is 0, this matches
    pure ``pred.head(top_n)``.
    """
    pred = pred.dropna().sort_values(ascending=False)
    if pred.empty:
        return []
    if hysteresis_buffer <= 0 or not prev_long:
        return list(pred.head(top_n).index)
    r = pred.rank(ascending=False, method="first")
    zone = top_n + hysteresis_buffer
    retained = [t for t in prev_long if t in r.index and float(r.loc[t]) <= zone]
    retained.sort(key=lambda t: float(r.loc[t]))
    if len(retained) >= top_n:
        return retained[:top_n]
    picked: list[str] = list(retained)
    picked_set = set(picked)
    for t in pred.index:
        if len(picked) >= top_n:
            break
        if t not in picked_set:
            picked.append(t)
            picked_set.add(t)
    return picked[:top_n]


def _select_short_with_hysteresis(
    pred: pd.Series,
    prev_short: list[str],
    bottom_n: int,
    hysteresis_buffer: int,
) -> list[str]:
    """Bottom-*bottom_n* short names; retain prior shorts while worst-rank stays in zone.

    Rank 1 (ascending rank) = lowest predicted return.  When *hysteresis_buffer*
    is 0, this matches pure ``pred.tail(bottom_n)``.
    """
    pred = pred.dropna().sort_values(ascending=False)
    if pred.empty:
        return []
    if hysteresis_buffer <= 0 or not prev_short:
        return list(pred.tail(bottom_n).index)
    r_worst = pred.rank(ascending=True, method="first")
    zone = bottom_n + hysteresis_buffer
    retained = [
        t for t in prev_short
        if t in r_worst.index and float(r_worst.loc[t]) <= zone
    ]
    retained.sort(key=lambda t: float(r_worst.loc[t]))
    if len(retained) >= bottom_n:
        return retained[:bottom_n]
    picked: list[str] = list(retained)
    picked_set = set(picked)
    for t in pred.sort_values(ascending=True).index:
        if len(picked) >= bottom_n:
            break
        if t not in picked_set:
            picked.append(t)
            picked_set.add(t)
    return picked[:bottom_n]


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


# ---------------------------------------------------------------------------
# Convenience: predict all months
# ---------------------------------------------------------------------------


def predict_all_months(
    train_result: Any,
    ohlcv_by_ticker: dict[str, pd.DataFrame],
    forward_returns_df: pd.DataFrame,
    fundamentals_by_ticker: dict[str, dict[str, Any]] | None = None,
) -> dict[str, pd.Series]:
    """Predict 1-month forward returns for every cutoff date.

    Builds features at each unique ``cutoff_date`` in *forward_returns_df*
    and applies the trained regression model.

    Parameters
    ----------
    train_result
        :class:`~regression_model.RegressionTrainResult` from training.
    ohlcv_by_ticker
        Ticker → OHLCV DataFrame.
    forward_returns_df
        Output of :func:`~regression_targets.compute_monthly_forward_returns`.
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

    cutoff_dates = sorted(forward_returns_df["cutoff_date"].unique())
    result: dict[str, pd.Series] = {}

    for cd in cutoff_dates:
        cd_str = _normalize_cutoff_key(cd)
        X = build_feature_matrix(ohlcv_by_ticker, cd_str, fundamentals_by_ticker)
        if X.empty:
            logger.warning("predict_all_months: empty features for cutoff %s", cd_str)
            continue
        pred = predict_returns(train_result, X)
        result[cd_str] = pred

    logger.info(
        "predict_all_months: %d/%d cutoffs produced predictions",
        len(result),
        len(cutoff_dates),
    )
    return result


# ---------------------------------------------------------------------------
# Core backtest
# ---------------------------------------------------------------------------


def run_regression_backtest(
    ohlcv_by_ticker: dict[str, pd.DataFrame],
    predicted_returns_by_month: dict[str, pd.Series],
    forward_returns_df: pd.DataFrame,
    *,
    top_n: int = 10,
    bottom_n: int | None = None,
    costs_bps: float = DEFAULT_REGRESSION_COSTS_BPS,
    hysteresis_buffer: int = DEFAULT_HYSTERESIS_BUFFER,
    n_quantiles: int = 5,
    sector_map: dict[str, str] | None = None,
    max_sector_weight: float | None = None,
    use_confidence_weights: bool = False,
    max_position_weight: float | None = None,
) -> RegressionBacktestResult:
    """Run a monthly-rebalanced portfolio backtest.

    Each month, stocks are ranked by predicted return.  The long-only strategy
    holds the top-*top_n*, the short leg sells the bottom-*bottom_n*, and the
    benchmark holds everything equal-weight.

    Parameters
    ----------
    ohlcv_by_ticker
        Ticker → OHLCV DataFrame (DatetimeIndex, ``Close`` column).
    predicted_returns_by_month
        ``cutoff_date`` (ISO string ``"YYYY-MM-DD"``) → pd.Series of
        predicted 1-month forward returns indexed by ticker.  Produced by
        :func:`predict_all_months` or manually.
    forward_returns_df
        Output of :func:`~regression_targets.compute_monthly_forward_returns`.
        Used for IC calculation and quantile analysis.
    top_n
        Number of top-ranked stocks to go long.
    bottom_n
        Number of bottom-ranked stocks to go short (defaults to *top_n*).
    costs_bps
        One-way transaction costs in basis points (default 30; classification
        backtests often use ~50 bps).  Deducted proportionally to turnover at
        each rebalancing (entry only — turnover = 0 means no cost for
        unchanged positions).
    hysteresis_buffer
        Non-negative slack for turnover reduction: a name already in the long
        (short) book stays while its rank remains among the best
        ``top_n + hysteresis_buffer`` (worst ``bottom_n + hysteresis_buffer``
        for shorts).  Set to ``0`` to rebalance to strict top-/bottom-*N* each
        month (no hysteresis).
    n_quantiles
        Quantile buckets for spread analysis.
    sector_map
        Ticker → sector string (e.g. from yfinance ``.info["sector"]``).  Used
        with *max_sector_weight* to cap aggregate weight per sector on each leg.
    max_sector_weight
        Maximum fraction of the leg in any one sector (e.g. ``0.30`` for 30%).
        Ignored if *sector_map* is ``None``.
    use_confidence_weights
        If ``True``, long weights increase with predicted return and short
        weights increase with more negative predicted return (within the
        selected top-/bottom-*N*).  Otherwise each leg is equal-weighted before
        caps.
    max_position_weight
        Maximum weight per name after normalisation (e.g. ``0.15`` for 15%).
        When *use_confidence_weights* is ``True`` and this is ``None``,
        defaults to :data:`DEFAULT_MAX_POSITION_WEIGHT` (15%).

    Returns
    -------
    RegressionBacktestResult
    """
    if bottom_n is None:
        bottom_n = top_n

    eff_max_position_weight = max_position_weight
    if use_confidence_weights and eff_max_position_weight is None:
        eff_max_position_weight = DEFAULT_MAX_POSITION_WEIGHT

    if max_sector_weight is not None and sector_map is None:
        logger.warning(
            "max_sector_weight=%s ignored because sector_map is None",
            max_sector_weight,
        )

    pred_by_cd: dict[str, pd.Series] = {
        _normalize_cutoff_key(k): v for k, v in predicted_returns_by_month.items()
    }
    cutoff_strs = sorted(pred_by_cd.keys())
    if not cutoff_strs:
        raise ValueError("predicted_returns_by_month is empty")

    # Forward-return lookup: cutoff_date str → Series(ticker → return)
    fwd_lookup: dict[str, pd.Series] = {}
    for cd_ts, grp in forward_returns_df.groupby("cutoff_date"):
        fwd_lookup[_normalize_cutoff_key(cd_ts)] = (
            grp.set_index("ticker")["forward_1m_return"]
        )

    # Holding-period boundaries: hold from cutoff_i to cutoff_{i+1}
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

    # Global close matrix for the entire backtest span
    all_tickers = sorted(
        {t for ps in pred_by_cd.values() for t in ps.dropna().index}
    )
    close_matrix = _build_close_matrix(
        ohlcv_by_ticker, all_tickers, cutoff_ts[0], holding_ends[-1],
    )
    daily_ret_matrix = (
        close_matrix.pct_change() if not close_matrix.empty else pd.DataFrame()
    )

    # ---------- monthly loop ----------
    snapshots: list[MonthlySnapshot] = []
    parts: dict[str, list[pd.Series]] = {
        "long_only": [], "short_only": [], "long_short": [], "benchmark": [],
    }
    quantile_rows: list[dict[str, Any]] = []
    prev_long: list[str] = []
    prev_short: list[str] = []
    cost_frac = costs_bps / 10_000.0

    for i, cd_str in enumerate(cutoff_strs):
        ct = cutoff_ts[i]
        he = holding_ends[i]
        pred = pred_by_cd[cd_str].dropna().sort_values(ascending=False)

        long_tk = _select_long_with_hysteresis(
            pred, prev_long, top_n, hysteresis_buffer,
        )
        short_tk = _select_short_with_hysteresis(
            pred, prev_short, bottom_n, hysteresis_buffer,
        )
        # Prevent overlap when universe is small
        long_set = set(long_tk)
        short_tk = [t for t in short_tk if t not in long_set]
        all_tk = list(pred.index)

        # IC for this month
        actual_ret = fwd_lookup.get(cd_str)
        ic = float("nan")
        if actual_ret is not None:
            common = pred.index.intersection(actual_ret.index)
            if len(common) >= 5:
                ic = _spearman_ic(
                    actual_ret.reindex(common).values,
                    pred.reindex(common).values,
                )

        # Turnover vs previous month
        to_l = _one_way_turnover(prev_long, long_tk)
        to_s = _one_way_turnover(prev_short, short_tk)

        # Daily returns for holding period (ct, he]
        if not daily_ret_matrix.empty:
            mask = (daily_ret_matrix.index > ct) & (daily_ret_matrix.index <= he)
            pr = daily_ret_matrix.loc[mask]

            w_long = (
                _compute_leg_weights(
                    pred,
                    long_tk,
                    leg="long",
                    use_confidence_weights=use_confidence_weights,
                    max_position_weight=eff_max_position_weight,
                    sector_map=sector_map,
                    max_sector_weight=max_sector_weight,
                )
                if long_tk
                else pd.Series(dtype=float)
            )
            w_short = (
                _compute_leg_weights(
                    pred,
                    short_tk,
                    leg="short",
                    use_confidence_weights=use_confidence_weights,
                    max_position_weight=eff_max_position_weight,
                    sector_map=sector_map,
                    max_sector_weight=max_sector_weight,
                )
                if short_tk
                else pd.Series(dtype=float)
            )

            long_part = (
                _daily_weighted_returns(pr, w_long)
                if not w_long.empty
                else pd.Series(dtype=float)
            )
            short_part_raw = (
                _daily_weighted_returns(pr, w_short)
                if not w_short.empty
                else pd.Series(dtype=float)
            )

            # Long leg
            if not long_part.empty:
                long_dr = long_part.copy()
                if cost_frac > 0:
                    long_dr.iloc[0] -= cost_frac * to_l
                parts["long_only"].append(long_dr)

            # Short leg (returns negated → positive when shorted stocks fall)
            if not short_part_raw.empty:
                short_dr = -short_part_raw.copy()
                if cost_frac > 0:
                    short_dr.iloc[0] -= cost_frac * to_s
                parts["short_only"].append(short_dr)

            # Long/Short combined (50/50 allocation on notionals)
            if not long_part.empty and not short_part_raw.empty:
                ls_dr = (long_part - short_part_raw) / 2.0
                if cost_frac > 0 and not ls_dr.empty:
                    ls_dr = ls_dr.copy()
                    ls_dr.iloc[0] -= cost_frac * (to_l + to_s) / 2.0
                parts["long_short"].append(ls_dr)
            elif not long_part.empty:
                parts["long_short"].append(long_part.copy())

            # Benchmark (equal weight entire universe)
            ba = [t for t in all_tk if t in pr.columns]
            if ba:
                parts["benchmark"].append(pr[ba].mean(axis=1))

        # Quantile analysis
        if actual_ret is not None:
            _add_quantile_row(pred, actual_ret, cd_str, n_quantiles, quantile_rows)

        snapshots.append(MonthlySnapshot(
            cutoff_date=ct,
            holding_end=he,
            predicted_returns=pred,
            actual_returns=actual_ret,
            long_tickers=long_tk,
            short_tickers=short_tk,
            ic=ic,
            n_universe=len(pred),
            turnover_long=to_l,
            turnover_short=to_s,
        ))
        prev_long = long_tk
        prev_short = short_tk

    # ---------- aggregate ----------
    daily_returns: dict[str, pd.Series] = {}
    for name, ps in parts.items():
        if ps:
            combined = pd.concat(ps).sort_index()
            combined = combined[~combined.index.duplicated(keep="last")]
            combined.name = name
            daily_returns[name] = combined
        else:
            daily_returns[name] = pd.Series(dtype=float, name=name)

    strategy_metrics = {n: _compute_metrics(dr) for n, dr in daily_returns.items()}

    if snapshots:
        avg_to_l = float(np.nanmean([s.turnover_long for s in snapshots]))
        avg_to_s = float(np.nanmean([s.turnover_short for s in snapshots]))
        strategy_metrics["long_only"]["avg_monthly_turnover"] = avg_to_l
        strategy_metrics["short_only"]["avg_monthly_turnover"] = avg_to_s
        strategy_metrics["long_short"]["avg_monthly_turnover"] = (
            avg_to_l + avg_to_s
        ) / 2.0

    quantile_df = pd.DataFrame(quantile_rows)
    if not quantile_df.empty:
        quantile_df = quantile_df.set_index("cutoff_date")

    summary_df = _build_monthly_summary(snapshots)

    logger.info(
        "Backtest: %d months | Top-%d Long, Bottom-%d Short | costs=%dbps | "
        "hysteresis_buffer=%d | conf_w=%s max_pos=%s sector_cap=%s",
        len(snapshots),
        top_n,
        bottom_n,
        int(costs_bps),
        hysteresis_buffer,
        use_confidence_weights,
        eff_max_position_weight,
        max_sector_weight,
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
    if snapshots:
        ics = [s.ic for s in snapshots if np.isfinite(s.ic)]
        if ics:
            logger.info(
                "  IC: mean=%.4f  std=%.4f  IR=%.2f  positive=%.0f%%",
                float(np.mean(ics)),
                float(np.std(ics)),
                float(np.mean(ics) / np.std(ics)) if np.std(ics) > 0 else float("nan"),
                100.0 * sum(1 for v in ics if v > 0) / len(ics),
            )

    return RegressionBacktestResult(
        snapshots=snapshots,
        daily_returns=daily_returns,
        strategy_metrics=strategy_metrics,
        quantile_monthly=quantile_df,
        summary_df=summary_df,
        costs_bps=costs_bps,
        top_n=top_n,
        bottom_n=bottom_n,
        hysteresis_buffer=hysteresis_buffer,
        use_confidence_weights=use_confidence_weights,
        max_position_weight=eff_max_position_weight,
        max_sector_weight=max_sector_weight,
    )


# ---------------------------------------------------------------------------
# Quantile helper
# ---------------------------------------------------------------------------


def _add_quantile_row(
    pred: pd.Series,
    actual_ret: pd.Series,
    cd_str: str,
    n_quantiles: int,
    rows: list[dict[str, Any]],
) -> None:
    """Append one row of quantile-return data (Q1 = top predicted)."""
    common = pred.index.intersection(actual_ret.index)
    if len(common) < n_quantiles:
        return

    pred_c = pred.reindex(common)
    actual_c = actual_ret.reindex(common)

    try:
        qlabels = [f"Q{q + 1}" for q in range(n_quantiles)]
        # rank descending so Q1 = highest predicted return
        qbins = pd.qcut(
            pred_c.rank(ascending=False, method="first"),
            n_quantiles,
            labels=qlabels,
        )
        q_rets = actual_c.groupby(qbins).mean()
        row: dict[str, Any] = {"cutoff_date": cd_str}
        for ql in qlabels:
            row[ql] = float(q_rets[ql]) if ql in q_rets.index else float("nan")
        row["spread"] = row.get("Q1", 0.0) - row.get(f"Q{n_quantiles}", 0.0)
        rows.append(row)
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# Monthly summary
# ---------------------------------------------------------------------------


def _build_monthly_summary(snapshots: list[MonthlySnapshot]) -> pd.DataFrame:
    """Per-month metrics table from snapshots."""
    rows: list[dict[str, Any]] = []
    for s in snapshots:
        row: dict[str, Any] = {
            "cutoff_date": s.cutoff_date,
            "ic": s.ic,
            "n_universe": s.n_universe,
            "n_long": len(s.long_tickers),
            "n_short": len(s.short_tickers),
            "turnover_long": s.turnover_long,
            "turnover_short": s.turnover_short,
        }
        if s.actual_returns is not None:
            long_rets = s.actual_returns.reindex(s.long_tickers)
            short_rets = s.actual_returns.reindex(s.short_tickers)
            long_mean = float(long_rets.mean()) if not long_rets.empty else float("nan")
            short_mean = float(short_rets.mean()) if not short_rets.empty else float("nan")
            bench_mean = float(s.actual_returns.mean())
            row["long_return"] = long_mean
            row["short_return"] = short_mean
            if np.isfinite(long_mean) and np.isfinite(short_mean):
                row["ls_return"] = (long_mean - short_mean) / 2.0
            else:
                row["ls_return"] = float("nan")
            row["benchmark_return"] = bench_mean
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summary & display
# ---------------------------------------------------------------------------


def backtest_summary_table(result: RegressionBacktestResult) -> pd.DataFrame:
    """Tabular comparison of all strategy legs vs benchmark.

    Parameters
    ----------
    result
        A :class:`RegressionBacktestResult` from :func:`run_regression_backtest`.

    Returns
    -------
    pd.DataFrame
        One row per strategy with performance metrics.
    """
    labels = {
        "long_only": f"Long-Only (Top-{result.top_n})",
        "short_only": f"Short-Only (Bottom-{result.bottom_n})",
        "long_short": "Long/Short",
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


def plot_backtest_cumulative(
    result: RegressionBacktestResult,
    *,
    title: str | None = None,
    ax: Any | None = None,
) -> Any:
    """Equity curves for all strategy legs and benchmark.

    Parameters
    ----------
    result
        :class:`RegressionBacktestResult` from :func:`run_regression_backtest`.
    """
    import matplotlib.pyplot as plt

    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(12, 5))

    if title is None:
        title = f"Cumulative Returns — Top-{result.top_n} / Bottom-{result.bottom_n}"

    colours = {
        "long_only": "#2ecc71",
        "short_only": "#e74c3c",
        "long_short": "#3498db",
        "benchmark": "#7f8c8d",
    }
    labels = {
        "long_only": f"Long-Only (Top-{result.top_n})",
        "short_only": f"Short-Only (Bottom-{result.bottom_n})",
        "long_short": "Long/Short",
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


def plot_monthly_ic(
    result: RegressionBacktestResult,
    *,
    title: str = "Monthly IC (Spearman)",
    ax: Any | None = None,
) -> Any:
    """Bar chart of per-month IC with mean reference line.

    Parameters
    ----------
    result
        :class:`RegressionBacktestResult` from :func:`run_regression_backtest`.
    """
    import matplotlib.pyplot as plt

    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(12, 4))

    ic_values = [s.ic for s in result.snapshots]
    labels = [s.cutoff_date.strftime("%Y-%m") for s in result.snapshots]
    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in ic_values]

    ax.bar(range(len(ic_values)), ic_values, color=colors, edgecolor="white",
           linewidth=0.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)

    finite_ics = [v for v in ic_values if np.isfinite(v)]
    if finite_ics:
        ic_mean = float(np.mean(finite_ics))
        ax.axhline(
            ic_mean, color="#3498db", linestyle="--", linewidth=1.5,
            label=f"Mean IC = {ic_mean:.3f}",
        )
    ax.axhline(0, color="grey", linestyle="-", linewidth=0.5)
    ax.set_ylabel("IC (Spearman)")
    ax.set_title(title)
    ax.legend(loc="upper right")

    if created:
        plt.tight_layout()
    return ax


def plot_monthly_turnover(
    result: RegressionBacktestResult,
    *,
    title: str = "Monthly Portfolio Turnover",
    ax: Any | None = None,
) -> Any:
    """Side-by-side bars of long and short turnover per month.

    Parameters
    ----------
    result
        :class:`RegressionBacktestResult` from :func:`run_regression_backtest`.
    """
    import matplotlib.pyplot as plt

    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(12, 4))

    n = len(result.snapshots)
    x = np.arange(n)
    width = 0.35
    labels = [s.cutoff_date.strftime("%Y-%m") for s in result.snapshots]
    to_l = [s.turnover_long for s in result.snapshots]
    to_s = [s.turnover_short for s in result.snapshots]

    ax.bar(x - width / 2, to_l, width, label="Long turnover", color="#2ecc71",
           alpha=0.8)
    ax.bar(x + width / 2, to_s, width, label="Short turnover", color="#e74c3c",
           alpha=0.8)

    avg_l = float(np.nanmean(to_l))
    ax.axhline(avg_l, color="#2ecc71", linestyle="--", linewidth=1,
               label=f"Avg long = {avg_l:.0%}")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Turnover")
    ax.set_ylim(0, 1.1)
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)

    if created:
        plt.tight_layout()
    return ax


def plot_quantile_returns(
    result: RegressionBacktestResult,
    *,
    title: str = "Mean Actual Return by Predicted-Return Quantile",
    ax: Any | None = None,
) -> Any:
    """Bar chart of mean actual return per quantile (Q1 = top predicted).

    A monotonically decreasing pattern (Q1 > Q2 > … > QN) indicates that
    the model's ranking has predictive value.

    Parameters
    ----------
    result
        :class:`RegressionBacktestResult` from :func:`run_regression_backtest`.
    """
    import matplotlib.pyplot as plt

    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(8, 5))

    qdf = result.quantile_monthly
    if qdf.empty:
        ax.text(0.5, 0.5, "No quantile data", transform=ax.transAxes,
                ha="center", va="center", fontsize=12)
        return ax

    q_cols = [c for c in qdf.columns if c.startswith("Q")]
    means = qdf[q_cols].mean()

    n_q = len(q_cols)
    cmap = plt.cm.RdYlGn_r
    bar_colors = [cmap(i / max(n_q - 1, 1)) for i in range(n_q)]

    bars = ax.bar(q_cols, means.values, color=bar_colors, edgecolor="white")
    for bar, val in zip(bars, means.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(),
            f"{val:.2%}", ha="center",
            va="bottom" if val >= 0 else "top", fontsize=9,
        )

    spread = means.iloc[0] - means.iloc[-1] if len(means) >= 2 else float("nan")
    ax.set_title(f"{title}\n(Q1–Q{n_q} spread = {spread:+.2%})")
    ax.set_ylabel("Mean Actual 1M Return")
    ax.set_xlabel("Predicted-Return Quantile (Q1 = top)")
    ax.axhline(0, color="grey", linewidth=0.5)

    if created:
        plt.tight_layout()
    return ax
