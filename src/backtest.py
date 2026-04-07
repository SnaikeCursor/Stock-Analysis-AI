"""Out-of-sample forward-test: feature-shift, strategy simulation, benchmark comparison.

Phase 6 of the Swiss SPI Extra pipeline — computes OOS features on data up to
Q4 2024, predicts 2025 groups with the final model, simulates long/short
strategies, and benchmarks against a naive equal-weight portfolio.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

try:
    from config import OOS_FEATURE_CUTOFF_DATE, OOS_YEAR
except ImportError:
    OOS_FEATURE_CUTOFF_DATE = "2024-12-31"
    OOS_YEAR = 2025

logger = logging.getLogger(__name__)

__all__ = [
    "BootstrapPortfolioResult",
    "ForwardTestResult",
    "MultiYearForwardResult",
    "QuarterlyForwardResult",
    "QuarterlyRebalancePeriod",
    "RegimeForwardTestResult",
    "RegimeMultiYearForwardResult",
    "aggregate_forward_metrics_by_regime",
    "bootstrap_winner_portfolio_metrics",
    "build_oos_features",
    "build_quarterly_rebalance_schedule",
    "compare_insample_oos",
    "compare_regime_aware_vs_single",
    "compute_oos_returns",
    "compute_portfolio_weights",
    "evaluate_forward",
    "evaluate_forward_multi",
    "evaluate_forward_multi_regime_aware",
    "evaluate_forward_quarterly_regime_aware",
    "evaluate_forward_quarterly_regression",
    "evaluate_forward_regime_aware",
    "hit_rate_by_group",
    "plot_cumulative_returns",
    "plot_hit_rates",
    "plot_insample_vs_oos",
    "plot_oos_confusion_matrix",
    "strategy_daily_returns",
    "strategy_summary",
    "turnover_from_portfolio_change",
]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class ForwardTestResult:
    """Artefacts from a forward-test evaluation."""

    predictions: pd.Series
    actual_returns: pd.Series
    actual_groups: pd.Series
    long_only: dict[str, float]
    long_short: dict[str, float]
    benchmark: dict[str, float]
    classification: dict[str, Any]
    hit_rates: dict[str, dict[str, Any]]
    daily_returns: dict[str, pd.Series]
    portfolio_weights: pd.Series | None = None
    costs_bps: float = 0.0


@dataclass
class MultiYearForwardResult:
    """Aggregated results across multiple OOS years."""

    per_year: dict[int, ForwardTestResult]
    summary: pd.DataFrame


@dataclass
class RegimeForwardTestResult:
    """Out-of-sample forward test plus regime snapshot at the feature cutoff."""

    forward: ForwardTestResult
    regime_label: str
    regime_confidence: float
    cutoff_date: str


@dataclass
class RegimeMultiYearForwardResult:
    """Multi-year forward results with per-year regime labels (walk-forward)."""

    per_year: dict[int, RegimeForwardTestResult]
    summary: pd.DataFrame


@dataclass
class QuarterlyForwardResult:
    """Quarterly-rebalanced evaluation with hysteresis and proportional costs.

    Wraps per-year :class:`ForwardTestResult` entries plus granular quarterly
    metrics, a turnover log, and cumulative transaction costs.
    """

    per_year: dict[int, ForwardTestResult]
    quarterly_detail: pd.DataFrame
    turnover_log: pd.DataFrame
    total_costs_bps: float
    rebalance_freq: int


@dataclass(frozen=True)
class QuarterlyRebalancePeriod:
    """One walk-forward window: feature cutoff and the daily-return interval."""

    period_index: int
    cutoff: str
    period_start: pd.Timestamp
    period_end: pd.Timestamp
    label: str


def build_quarterly_rebalance_schedule(
    year: int,
    rebalance_freq: int,
    cutoff_shift_days: int = 0,
) -> list[QuarterlyRebalancePeriod]:
    """Cutoffs and trading windows for regime-aware quarterly evaluation.

    *rebalance_freq* ``1`` → quarter-ends Mar/Jun/Sep plus prior Dec-31 initial
    cutoff; ``2`` → mid-year Jun-30 only; any other value → annual (single
    period, initial cutoff only, full calendar year).

    When *cutoff_shift_days* > 0 every base cutoff is pushed forward by that
    many calendar days (publication-lag shift).  Period boundaries derive from
    the shifted cutoffs and may extend into the following calendar year.

    Returns
    -------
    One :class:`QuarterlyRebalancePeriod` per sub-period, ordered chronologically.
    """
    initial_cutoff = f"{year - 1}-12-31"
    if rebalance_freq == 1:
        mid_cutoffs = [f"{year}-03-31", f"{year}-06-30", f"{year}-09-30"]
    elif rebalance_freq == 2:
        mid_cutoffs = [f"{year}-06-30"]
    else:
        mid_cutoffs = []

    all_cutoffs = [initial_cutoff] + mid_cutoffs
    n_periods = len(all_cutoffs)

    shift_td = pd.Timedelta(days=cutoff_shift_days)
    shifted_cutoffs = [
        str((pd.Timestamp(c) + shift_td).date()) for c in all_cutoffs
    ]

    shifted_mid_ts = [pd.Timestamp(c) + shift_td for c in mid_cutoffs]

    period_starts: list[pd.Timestamp] = [
        pd.Timestamp(shifted_cutoffs[0]) + pd.Timedelta(days=1),
    ]
    for smc in shifted_mid_ts:
        period_starts.append(smc + pd.Timedelta(days=1))

    period_ends: list[pd.Timestamp] = list(shifted_mid_ts)
    period_ends.append(pd.Timestamp(f"{year}-12-31") + shift_td)

    periods: list[QuarterlyRebalancePeriod] = []
    for i in range(n_periods):
        if rebalance_freq == 1:
            label = f"Q{i + 1}"
        elif rebalance_freq == 2:
            label = f"H{i + 1}"
        else:
            label = "FY"
        periods.append(
            QuarterlyRebalancePeriod(
                period_index=i,
                cutoff=shifted_cutoffs[i],
                period_start=period_starts[i],
                period_end=period_ends[i],
                label=label,
            ),
        )
    return periods


def turnover_from_portfolio_change(
    prev: Sequence[str],
    new: Sequence[str],
    *,
    is_initial: bool = False,
) -> tuple[int, float, frozenset[str], frozenset[str]]:
    """Position churn for a rebalance: swap count, turnover fraction, in/out sets.

    *Turnover* is ``n_swapped / len(new)`` (or ``1.0`` on the initial formation).
    ``n_swapped`` is ``max(|swapped_in|, |swapped_out|)``, matching one-for-one
    replacement counting used for proportional transaction costs.
    """
    if is_initial:
        new_f = frozenset(new)
        return len(new), 1.0, new_f, frozenset()

    old_set = frozenset(prev)
    new_set = frozenset(new)
    swapped_in = new_set - old_set
    swapped_out = old_set - new_set
    n_swapped = max(len(swapped_in), len(swapped_out))
    denom = max(len(new), 1)
    turnover = n_swapped / denom
    return n_swapped, turnover, swapped_in, swapped_out


@dataclass
class BootstrapPortfolioResult:
    """Bootstrap percentiles for long-Winner portfolio (resample winners with replacement).

    Each iteration resamples *predicted_winners* with replacement, forms an
    implicit equal-weight portfolio (duplicate draws increase that ticker's
    weight), and recomputes daily returns for the OOS *year*.  Percentile dicts
    use keys ``p05``, ``p50``, ``p95``.
    """

    n_iterations: int
    n_winners: int
    cumulative_return: dict[str, float]
    sharpe_ratio: dict[str, float]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _daily_close_matrix(
    ohlcv_by_ticker: dict[str, pd.DataFrame],
    tickers: list[str],
    year: int,
    *,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Aligned daily close prices for *tickers* within *year*.

    Optional *start_date* / *end_date* override the default Jan-01 / Dec-31
    range (needed when shifted periods extend across year boundaries).
    """
    start = start_date if start_date is not None else pd.Timestamp(f"{year}-01-01")
    end = end_date if end_date is not None else pd.Timestamp(f"{year}-12-31")

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


def _portfolio_daily_returns(
    close_matrix: pd.DataFrame,
    tickers: list[str],
    direction: float = 1.0,
    weights: pd.Series | None = None,
    costs_bps: float = 0.0,
) -> pd.Series:
    """Weighted daily returns for a subset of tickers.

    *direction* is +1.0 for long, -1.0 for short.
    When *weights* is ``None``, uses equal weighting.
    *costs_bps* deducts one-way transaction costs (in basis points) on
    entry (first day) and exit (last day).
    """
    available = [t for t in tickers if t in close_matrix.columns]
    if not available:
        return pd.Series(dtype=float, name="portfolio")
    sub = close_matrix[available]
    daily_ret = sub.pct_change().dropna(how="all")

    if weights is not None:
        w = weights.reindex(available).fillna(0.0)
        w_sum = w.sum()
        if w_sum > 0:
            w = w / w_sum
        else:
            w = pd.Series(1.0 / len(available), index=available)
        portfolio = (daily_ret * w).sum(axis=1) * direction
    else:
        portfolio = daily_ret.mean(axis=1) * direction

    if costs_bps > 0 and not portfolio.empty:
        cost_frac = costs_bps / 10_000.0
        portfolio.iloc[0] -= cost_frac
        portfolio.iloc[-1] -= cost_frac

    portfolio.name = "portfolio"
    return portfolio


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


def bootstrap_winner_portfolio_metrics(
    ohlcv_by_ticker: dict[str, pd.DataFrame],
    predicted_winners: list[str],
    year: int,
    *,
    n_iterations: int = 1000,
    costs_bps: float = 0.0,
    random_state: int | None = None,
) -> BootstrapPortfolioResult:
    """Bootstrap the long-Winner leg: resample tickers with replacement, report CIs.

    For each of *n_iterations* draws, sample ``len(pool)`` names with
    replacement from *pool* (predicted Winners that have OOS daily prices).
    Portfolio weights are proportional to bootstrap counts.  Summarises
    cumulative return and Sharpe ratio with 5th / 50th / 95th percentiles.

    Parameters
    ----------
    ohlcv_by_ticker
        Ticker → OHLCV DataFrame.
    predicted_winners
        Tickers the model labelled ``Winners`` for this OOS period.
    year
        Calendar year for daily returns.
    n_iterations
        Number of bootstrap samples (default 1000).
    costs_bps
        Same one-way cost model as :func:`strategy_daily_returns`.
    random_state
        Seed for reproducibility.

    Returns
    -------
    BootstrapPortfolioResult
        Percentile tables; if *pool* is empty, percentiles are NaN.
    """
    empty_pct = {"p05": float("nan"), "p50": float("nan"), "p95": float("nan")}
    if not predicted_winners:
        logger.warning("bootstrap_winner_portfolio_metrics: no predicted winners")
        return BootstrapPortfolioResult(
            n_iterations=n_iterations,
            n_winners=0,
            cumulative_return=empty_pct.copy(),
            sharpe_ratio=empty_pct.copy(),
        )

    close = _daily_close_matrix(ohlcv_by_ticker, list(predicted_winners), year)
    pool = [t for t in predicted_winners if t in close.columns]
    if len(pool) < len(predicted_winners):
        logger.info(
            "Bootstrap: %d/%d winners have OOS prices in %d",
            len(pool),
            len(predicted_winners),
            year,
        )
    if not pool or close.empty:
        logger.warning(
            "bootstrap_winner_portfolio_metrics: no overlapping prices for year %s",
            year,
        )
        return BootstrapPortfolioResult(
            n_iterations=n_iterations,
            n_winners=len(pool),
            cumulative_return=empty_pct.copy(),
            sharpe_ratio=empty_pct.copy(),
        )

    rng = np.random.default_rng(random_state)
    n_draw = len(pool)
    cumrets: list[float] = []
    sharpes: list[float] = []

    for _ in range(n_iterations):
        draw = rng.choice(pool, size=n_draw, replace=True)
        counts = pd.Series(draw).value_counts()
        w = counts.astype(float)
        w = w / w.sum()
        dr = _portfolio_daily_returns(
            close,
            list(w.index),
            direction=1.0,
            weights=w,
            costs_bps=costs_bps,
        )
        m = _compute_metrics(dr)
        cumrets.append(m["cumulative_return"])
        sharpes.append(m["sharpe_ratio"])

    arr_c = np.asarray(cumrets, dtype=float)
    arr_s = np.asarray(sharpes, dtype=float)

    def _pct(a: np.ndarray) -> dict[str, float]:
        finite = a[np.isfinite(a)]
        if finite.size == 0:
            return empty_pct.copy()
        return {
            "p05": float(np.percentile(finite, 5)),
            "p50": float(np.percentile(finite, 50)),
            "p95": float(np.percentile(finite, 95)),
        }

    logger.info(
        "Bootstrap (%d iters, n_winners=%d): cum_ret p05/p50/p95=%s  Sharpe p05/p50/p95=%s",
        n_iterations,
        len(pool),
        _pct(arr_c),
        _pct(arr_s),
    )

    return BootstrapPortfolioResult(
        n_iterations=n_iterations,
        n_winners=len(pool),
        cumulative_return=_pct(arr_c),
        sharpe_ratio=_pct(arr_s),
    )


def _import_classifier():
    """Lazy import to avoid circular dependency."""
    try:
        from src.classifier import assign_groups, total_return_q1
    except ImportError:
        from classifier import assign_groups, total_return_q1
    return total_return_q1, assign_groups


def _import_model():
    """Lazy import to avoid circular dependency."""
    try:
        from src.model import evaluate_predictions
    except ImportError:
        from model import evaluate_predictions
    return evaluate_predictions


def _is_multi_quarter_ensemble(obj: Any) -> bool:
    """Check if *obj* is a MultiQuarterEnsembleResult without a hard import."""
    return type(obj).__name__ == "MultiQuarterEnsembleResult"


# ---------------------------------------------------------------------------
# Public API — data preparation
# ---------------------------------------------------------------------------


def compute_oos_returns(
    ohlcv_by_ticker: dict[str, pd.DataFrame],
    year: int | None = None,
) -> pd.Series:
    """Total return per ticker for the out-of-sample year.

    Uses first available close on or after Jan-02 and last close on or
    before Dec-31 of the target year.

    Parameters
    ----------
    ohlcv_by_ticker
        Ticker → OHLCV DataFrame.
    year
        Calendar year; defaults to :data:`config.OOS_YEAR` (2025).
    """
    yr = OOS_YEAR if year is None else year
    total_return_q1, _ = _import_classifier()

    out: dict[str, float] = {}
    for ticker, df in ohlcv_by_ticker.items():
        if df is None or df.empty or "Close" not in df.columns:
            out[ticker] = float("nan")
            continue
        out[ticker] = total_return_q1(df["Close"], f"{yr}-01-02", f"{yr}-12-31")

    result = pd.Series(out, name=f"return_{yr}")
    valid = result.dropna()
    logger.info(
        "OOS returns (%d): n=%d  mean=%.3f  median=%.3f  (NaN: %d)",
        yr,
        len(valid),
        valid.mean() if not valid.empty else float("nan"),
        valid.median() if not valid.empty else float("nan"),
        result.isna().sum(),
    )
    return result


def build_oos_features(
    ohlcv_by_ticker: dict[str, pd.DataFrame],
    fundamentals_by_ticker: dict[str, dict[str, Any]] | None = None,
    cutoff_date: str | None = None,
    *,
    eulerpool_quarterly: dict[str, list[dict]] | None = None,
    eulerpool_profiles: dict[str, dict] | None = None,
    publication_lag_days: int = 0,
) -> pd.DataFrame:
    """Build feature matrix for OOS prediction (shifted cutoff).

    Convenience wrapper around :func:`features.build_feature_matrix` that
    defaults the cutoff to :data:`config.OOS_FEATURE_CUTOFF_DATE` (Q4 2024).

    Parameters
    ----------
    ohlcv_by_ticker
        Ticker → OHLCV DataFrame.
    fundamentals_by_ticker
        Optional ticker → yfinance ``.info`` dict (fallback when Eulerpool
        is not provided).
    cutoff_date
        ISO date string; defaults to ``OOS_FEATURE_CUTOFF_DATE``.
    eulerpool_quarterly
        Optional ticker → list of Eulerpool quarterly records for PIT
        fundamental feature construction.
    eulerpool_profiles
        Optional ticker → Eulerpool company profile dict for sector info.
    publication_lag_days
        When Eulerpool quarterly data is used, shifts the fundamental cutoff
        backward by this many days (publication lag). See
        :func:`features.build_feature_matrix`.
    """
    cutoff = cutoff_date if cutoff_date is not None else OOS_FEATURE_CUTOFF_DATE
    try:
        from src.features import build_feature_matrix
    except ImportError:
        from features import build_feature_matrix

    logger.info("Building OOS features with cutoff=%s", cutoff)
    return build_feature_matrix(
        ohlcv_by_ticker,
        cutoff,
        fundamentals_by_ticker,
        eulerpool_quarterly=eulerpool_quarterly,
        eulerpool_profiles=eulerpool_profiles,
        publication_lag_days=publication_lag_days,
    )


# ---------------------------------------------------------------------------
# Public API — strategy simulation
# ---------------------------------------------------------------------------


def compute_portfolio_weights(
    winners: list[str],
    *,
    proba_weights: pd.Series | None = None,
    sector_map: dict[str, str] | None = None,
    max_sector_weight: float | None = None,
    top_n: int | None = None,
    min_winners: int | None = None,
    full_universe: list[str] | None = None,
) -> pd.Series:
    """Compute final portfolio weights for the long-Winners leg.

    Processing order: top-N filter → min-winners fill → confidence weighting
    → sector cap.

    Parameters
    ----------
    winners
        Predicted Winner tickers (full list before filtering).
    proba_weights
        ``P(Winner)`` per ticker from :func:`model.predict_proba`. When
        ``None``, equal weighting is used.
    sector_map
        Ticker → sector string (e.g. from ``yfinance .info["sector"]``).
        Required when *max_sector_weight* is set.
    max_sector_weight
        Maximum portfolio weight per sector (e.g. 0.30 = 30%). Excess is
        redistributed proportionally to uncapped sectors.
    top_n
        Keep only the top *N* tickers by ``proba_weights``. Ignored when
        ``proba_weights`` is ``None``.
    min_winners
        Minimum number of tickers in the long leg. When the pool is smaller
        and *proba_weights* / *full_universe* are available, the top tickers
        by ``P(Winner)`` from the full universe are added until the minimum
        is reached.
    full_universe
        All available tickers (prediction universe). Used only for
        *min_winners* back-fill.

    Returns
    -------
    pd.Series
        Normalised portfolio weights summing to 1.0, indexed by ticker.
    """
    pool = list(winners)

    if top_n is not None and proba_weights is not None and top_n < len(pool):
        pw = proba_weights.reindex(pool).fillna(0.0)
        pool = list(pw.nlargest(top_n).index)
        logger.info("Top-N filter: %d → %d Winners", len(winners), len(pool))

    if (
        min_winners is not None
        and len(pool) < min_winners
        and proba_weights is not None
        and full_universe is not None
    ):
        n_before = len(pool)
        pw_all = proba_weights.reindex(full_universe).fillna(0.0)
        top_candidates = list(pw_all.nlargest(min_winners).index)
        for t in top_candidates:
            if t not in pool:
                pool.append(t)
            if len(pool) >= min_winners:
                break
        logger.warning(
            "Min-winners fill: %d predicted → %d after fill (min_winners=%d)",
            n_before, len(pool), min_winners,
        )

    if not pool:
        return pd.Series(dtype=float, name="weight")

    if proba_weights is not None:
        w = proba_weights.reindex(pool).fillna(0.0)
        w_sum = w.sum()
        w = w / w_sum if w_sum > 0 else pd.Series(1.0 / len(pool), index=pool)
    else:
        w = pd.Series(1.0 / len(pool), index=pool)

    if max_sector_weight is not None and sector_map is not None:
        w = _apply_sector_cap(w, sector_map, max_sector_weight)

    w.name = "weight"
    return w


def _apply_sector_cap(
    weights: pd.Series,
    sector_map: dict[str, str],
    cap: float,
) -> pd.Series:
    """Iteratively clip sector weights at *cap* and redistribute excess.

    When the cap is too tight for the number of sectors (e.g. 3 sectors
    with a 30% cap cannot reach 100%), the algorithm converges to the
    closest feasible allocation.
    """
    w = weights.copy()
    sectors = pd.Series({t: sector_map.get(t, "Unknown") for t in w.index})
    n_sectors = sectors.nunique()
    capped_any = False

    if cap * n_sectors < 1.0:
        logger.warning(
            "Sector cap %.0f%% × %d sectors = %.0f%% < 100%%; "
            "constraints are infeasible — will approximate",
            cap * 100, n_sectors, cap * n_sectors * 100,
        )

    for _ in range(20):
        sector_w = w.groupby(sectors).sum()
        over = sector_w[sector_w > cap + 1e-9]
        if over.empty:
            break
        capped_any = True

        locked = set()
        for sec in over.index:
            locked.add(sec)
            mask = sectors == sec
            w[mask] = w[mask] * (cap / sector_w[sec])

        unlocked_mask = ~sectors.isin(locked)
        unlocked_sum = w[unlocked_mask].sum()
        total_excess = 1.0 - w.sum()

        if total_excess > 0 and unlocked_sum > 0:
            w[unlocked_mask] += w[unlocked_mask] / unlocked_sum * total_excess

        w_sum = w.sum()
        if w_sum > 0:
            w = w / w_sum

    if capped_any:
        final_sector_w = w.groupby(sectors).sum()
        capped_sectors = list(final_sector_w[final_sector_w >= cap - 1e-9].index)
        logger.info(
            "Sector cap (%.0f%%): capped %d sector(s): %s",
            cap * 100,
            len(capped_sectors),
            capped_sectors,
        )
    return w


def strategy_daily_returns(
    ohlcv_by_ticker: dict[str, pd.DataFrame],
    predictions: pd.Series,
    year: int | None = None,
    *,
    costs_bps: float = 0.0,
    proba_weights: pd.Series | None = None,
    sector_map: dict[str, str] | None = None,
    max_sector_weight: float | None = None,
    top_n: int | None = None,
    min_winners: int | None = None,
) -> dict[str, pd.Series]:
    """Daily portfolio returns for Long-Winners, Short-Losers, Long/Short, and Benchmark.

    Strategies
    ----------
    long_winners
        Portfolio of predicted Winners (equal-weight or confidence-weighted).
    short_losers
        Equal-weight short portfolio of predicted Losers (daily returns negated).
    long_short
        50 / 50 blend of long_winners + short_losers.
    benchmark
        Equal-weight long portfolio of *all* tickers in the prediction universe.

    Parameters
    ----------
    ohlcv_by_ticker
        Ticker → OHLCV DataFrame.
    predictions
        Predicted group labels indexed by ticker.
    year
        Calendar year; defaults to :data:`config.OOS_YEAR`.
    costs_bps
        One-way transaction costs in basis points (default 0). Entry costs
        are deducted on day 1, exit costs on the last trading day. For
        Swiss small caps, 30–50 bps is realistic.
    proba_weights
        ``P(Winner)`` per ticker from :func:`model.predict_proba`. Enables
        confidence-weighted long positions. When ``None``, equal weighting.
    sector_map
        Ticker → sector string. Required for sector diversification caps.
    max_sector_weight
        Maximum portfolio weight per sector (e.g. 0.30 for 30%).
    top_n
        Only go long on the top *N* predicted Winners by probability.
        Requires *proba_weights*.
    min_winners
        Minimum number of tickers in the long leg.  See
        :func:`compute_portfolio_weights`.

    Returns
    -------
    dict
        Strategy name → daily return :class:`pd.Series` (DatetimeIndex).
    """
    yr = OOS_YEAR if year is None else year
    all_tickers = list(predictions.index)
    winners = list(predictions[predictions == "Winners"].index)
    losers = list(predictions[predictions == "Losers"].index)

    winner_weights = compute_portfolio_weights(
        winners,
        proba_weights=proba_weights,
        sector_map=sector_map,
        max_sector_weight=max_sector_weight,
        top_n=top_n,
        min_winners=min_winners,
        full_universe=all_tickers,
    )
    effective_winners = list(winner_weights.index)

    logger.info(
        "Strategy universe: %d tickers (%d predicted Winners → %d after filters, %d Losers)%s",
        len(all_tickers),
        len(winners),
        len(effective_winners),
        len(losers),
        f"  costs={costs_bps:.0f}bps" if costs_bps > 0 else "",
    )

    close = _daily_close_matrix(ohlcv_by_ticker, all_tickers, yr)
    if close.empty:
        logger.warning("No daily close data for OOS year %d", yr)
        empty: pd.Series = pd.Series(dtype=float, name="portfolio")
        return {
            "long_winners": empty,
            "short_losers": empty,
            "long_short": empty,
            "benchmark": empty,
        }

    long_w = _portfolio_daily_returns(
        close,
        effective_winners,
        direction=1.0,
        weights=winner_weights if proba_weights is not None else None,
        costs_bps=costs_bps,
    )
    short_l = _portfolio_daily_returns(
        close, losers, direction=-1.0, costs_bps=costs_bps,
    )

    if not long_w.empty and not short_l.empty:
        aligned = pd.concat(
            [long_w.rename("lw"), short_l.rename("sl")], axis=1
        ).dropna()
        ls = (aligned["lw"] + aligned["sl"]) / 2.0
        ls.name = "portfolio"
    elif not long_w.empty:
        ls = long_w.copy()
    else:
        ls = short_l.copy()

    benchmark = _portfolio_daily_returns(close, all_tickers, direction=1.0)

    result = {
        "long_winners": long_w,
        "short_losers": short_l,
        "long_short": ls,
        "benchmark": benchmark,
    }
    for name, dr in result.items():
        m = _compute_metrics(dr)
        logger.info(
            "Strategy '%s': cum=%.3f  sharpe=%.2f  maxDD=%.3f  (%d days)",
            name,
            m["cumulative_return"],
            m["sharpe_ratio"],
            m["max_drawdown"],
            m["n_trading_days"],
        )
    return result


# ---------------------------------------------------------------------------
# Public API — evaluation
# ---------------------------------------------------------------------------


def evaluate_forward(
    predictions: pd.Series,
    returns_oos: pd.Series,
    ohlcv_by_ticker: dict[str, pd.DataFrame] | None = None,
    *,
    year: int | None = None,
    costs_bps: float = 0.0,
    proba_weights: pd.Series | None = None,
    sector_map: dict[str, str] | None = None,
    max_sector_weight: float | None = None,
    top_n: int | None = None,
    min_winners: int | None = None,
) -> ForwardTestResult:
    """Full out-of-sample evaluation: strategy metrics, classification, hit rates.

    Parameters
    ----------
    predictions
        Predicted group labels (``Winners`` / ``Steady`` / ``Losers``) indexed
        by ticker.
    returns_oos
        Actual total return per ticker for the OOS year.
    ohlcv_by_ticker
        OHLCV frames per ticker.  Required for daily-return-based metrics
        (Sharpe, max drawdown).  When ``None``, only total-return averages
        are computed.
    year
        OOS calendar year; defaults to :data:`config.OOS_YEAR`.
    costs_bps
        One-way transaction costs in basis points (deducted on entry and exit).
    proba_weights
        ``P(Winner)`` per ticker for confidence-weighted positions.
    sector_map
        Ticker → sector for diversification caps.
    max_sector_weight
        Maximum portfolio weight per sector.
    top_n
        Only long top-N Winners by probability.
    min_winners
        Minimum number of tickers in the long leg.  See
        :func:`compute_portfolio_weights`.

    Returns
    -------
    ForwardTestResult
        Complete forward-test artefacts including strategy metrics,
        classification metrics, hit rates, and daily return series.
    """
    _, assign_groups = _import_classifier()
    evaluate_predictions = _import_model()

    yr = OOS_YEAR if year is None else year

    common = predictions.dropna().index.intersection(returns_oos.dropna().index)
    preds = predictions.loc[common]
    rets = returns_oos.loc[common]

    logger.info(
        "Evaluating forward test: %d tickers with predictions and returns",
        len(common),
    )

    actual_groups = assign_groups(rets)

    class_names = sorted(
        set(preds.unique()) | set(actual_groups.dropna().unique()),
    )

    if len(common) == 0:
        logger.warning(
            "evaluate_forward: 0 tickers with both predictions and OOS returns; "
            "classification metrics undefined (refresh OHLCV if the OOS year predates cached history).",
        )
        classification = {
            "accuracy": float("nan"),
            "f1_macro": float("nan"),
            "f1_weighted": float("nan"),
            "confusion_matrix": np.zeros((0, 0), dtype=int),
            "report_dict": {},
            "report_str": "",
        }
    else:
        y_true = actual_groups.loc[common].values
        y_pred = preds.values
        classification = evaluate_predictions(y_true, y_pred, class_names)

    hr = hit_rate_by_group(preds, actual_groups)

    winners = list(preds[preds == "Winners"].index)
    portfolio_weights = compute_portfolio_weights(
        winners,
        proba_weights=proba_weights,
        sector_map=sector_map,
        max_sector_weight=max_sector_weight,
        top_n=top_n,
        min_winners=min_winners,
        full_universe=list(preds.index),
    )

    daily_rets: dict[str, pd.Series] = {}
    lo_metrics: dict[str, float]
    ls_metrics: dict[str, float]
    bm_metrics: dict[str, float]

    if ohlcv_by_ticker is not None:
        daily_rets = strategy_daily_returns(
            ohlcv_by_ticker,
            predictions,
            year=yr,
            costs_bps=costs_bps,
            proba_weights=proba_weights,
            sector_map=sector_map,
            max_sector_weight=max_sector_weight,
            top_n=top_n,
            min_winners=min_winners,
        )
        lo_metrics = _compute_metrics(
            daily_rets.get("long_winners", pd.Series(dtype=float)),
        )
        ls_metrics = _compute_metrics(
            daily_rets.get("long_short", pd.Series(dtype=float)),
        )
        bm_metrics = _compute_metrics(
            daily_rets.get("benchmark", pd.Series(dtype=float)),
        )
    else:
        winners_mask = preds == "Winners"
        losers_mask = preds == "Losers"
        lo_ret = (
            float(rets[winners_mask].mean()) if winners_mask.any() else float("nan")
        )
        ls_long = float(rets[winners_mask].mean()) if winners_mask.any() else 0.0
        ls_short = float(-rets[losers_mask].mean()) if losers_mask.any() else 0.0
        ls_ret = (
            (ls_long + ls_short) / 2.0
            if winners_mask.any() or losers_mask.any()
            else float("nan")
        )
        bm_ret = float(rets.mean())

        lo_metrics = {
            "cumulative_return": lo_ret,
            "sharpe_ratio": float("nan"),
            "max_drawdown": float("nan"),
        }
        ls_metrics = {
            "cumulative_return": ls_ret,
            "sharpe_ratio": float("nan"),
            "max_drawdown": float("nan"),
        }
        bm_metrics = {
            "cumulative_return": bm_ret,
            "sharpe_ratio": float("nan"),
            "max_drawdown": float("nan"),
        }

    return ForwardTestResult(
        predictions=preds,
        actual_returns=rets,
        actual_groups=actual_groups,
        long_only=lo_metrics,
        long_short=ls_metrics,
        benchmark=bm_metrics,
        classification=classification,
        hit_rates=hr,
        daily_returns=daily_rets,
        portfolio_weights=portfolio_weights,
        costs_bps=costs_bps,
    )


def _default_regime_cutoff_for_oos_year(year: int) -> str:
    """Feature cutoff aligned with walk-forward: prior calendar year-end."""
    return f"{year - 1}-12-31"


def evaluate_forward_regime_aware(
    predictions: pd.Series,
    returns_oos: pd.Series,
    ohlcv_by_ticker: dict[str, pd.DataFrame],
    *,
    year: int | None = None,
    cutoff_date: str | None = None,
    costs_bps: float = 0.0,
    proba_weights: pd.Series | None = None,
    sector_map: dict[str, str] | None = None,
    max_sector_weight: float | None = None,
    top_n: int | None = None,
    min_winners: int | None = None,
    benchmark_ticker: str | None = None,
) -> RegimeForwardTestResult:
    """Same as :func:`evaluate_forward`, plus regime at the feature cutoff on ``^SSMI``.

    Uses :func:`src.regime.detect_regime` on the benchmark OHLCV (no lookahead).
    The default *cutoff_date* matches walk-forward tests: ``{OOS year − 1}-12-31``.

    Parameters
    ----------
    predictions, returns_oos, ohlcv_by_ticker
        See :func:`evaluate_forward`.
    year, costs_bps, proba_weights, sector_map, max_sector_weight, top_n, min_winners
        See :func:`evaluate_forward`.
    cutoff_date
        ISO date for regime detection; defaults to :func:`_default_regime_cutoff_for_oos_year`.
    benchmark_ticker
        Key in *ohlcv_by_ticker* for the SMI series; defaults to
        :data:`src.features.MACRO_BENCHMARK_TICKER` (``^SSMI``).

    Returns
    -------
    RegimeForwardTestResult
        Wraps :class:`ForwardTestResult` with ``regime_label`` and
        ``regime_confidence`` from the detector.
    """
    try:
        from src.features import MACRO_BENCHMARK_TICKER
        from src.regime import detect_regime
    except ImportError:
        from features import MACRO_BENCHMARK_TICKER
        from regime import detect_regime

    yr = OOS_YEAR if year is None else year
    co = cutoff_date if cutoff_date is not None else _default_regime_cutoff_for_oos_year(yr)
    bench = benchmark_ticker if benchmark_ticker is not None else MACRO_BENCHMARK_TICKER
    smi = ohlcv_by_ticker.get(bench)
    if smi is None or smi.empty:
        raise ValueError(
            f"Regime-aware evaluation requires {bench!r} in ohlcv_by_ticker",
        )

    state = detect_regime(smi, co)
    regime_label = state.label.value
    regime_confidence = float(state.confidence)

    forward = evaluate_forward(
        predictions,
        returns_oos,
        ohlcv_by_ticker,
        year=yr,
        costs_bps=costs_bps,
        proba_weights=proba_weights,
        sector_map=sector_map,
        max_sector_weight=max_sector_weight,
        top_n=top_n,
        min_winners=min_winners,
    )

    logger.info(
        "Regime-aware forward eval: OOS year=%d cutoff=%s regime=%s conf=%.2f",
        yr,
        co,
        regime_label.upper(),
        regime_confidence,
    )

    return RegimeForwardTestResult(
        forward=forward,
        regime_label=regime_label,
        regime_confidence=regime_confidence,
        cutoff_date=co,
    )


def evaluate_forward_multi(
    ohlcv_by_ticker: dict[str, pd.DataFrame],
    fundamentals_by_ticker: dict[str, dict[str, Any]] | None = None,
    *,
    train_result: Any = None,
    oos_years: list[int] | None = None,
    oos_configs: list[dict[str, Any]] | None = None,
    costs_bps: float = 0.0,
    proba_weights_fn: Any | None = None,
    sector_map: dict[str, str] | None = None,
    max_sector_weight: float | None = None,
    top_n: int | None = None,
    min_winners: int | None = None,
) -> MultiYearForwardResult:
    """Run :func:`evaluate_forward` across multiple OOS years and aggregate.

    Two calling conventions are supported:

    **Simple** — pass ``oos_years`` (list of calendar years) plus a
    ``train_result`` (:class:`~model.TrainResult`). OOS features are
    computed with cutoff ``<year-1>-12-31`` and predictions come from
    :func:`model.predict`.

    **Advanced** — pass ``oos_configs``, a list of dicts each containing
    at minimum ``year``, ``predictions`` (pd.Series), and
    ``returns_oos`` (pd.Series). Optional keys: ``proba_weights``,
    ``cutoff_date``.

    Parameters
    ----------
    ohlcv_by_ticker
        Ticker → OHLCV DataFrame.
    fundamentals_by_ticker
        Ticker → yfinance ``.info`` dict (for feature building).
    train_result
        :class:`~model.TrainResult` used for predictions (simple mode).
    oos_years
        List of OOS calendar years (simple mode).
    oos_configs
        List of per-year config dicts (advanced mode).
    costs_bps, sector_map, max_sector_weight, top_n, min_winners
        Passed through to :func:`evaluate_forward`.
    proba_weights_fn
        Callable ``(train_result, X_oos) -> pd.Series`` returning
        ``P(Winner)`` per ticker. When ``None`` and ``train_result`` is
        available, uses :func:`model.predict_proba` with the Winner column.

    Returns
    -------
    MultiYearForwardResult
        Per-year results and an aggregated summary DataFrame.
    """
    per_year: dict[int, ForwardTestResult] = {}

    if oos_configs is not None:
        for cfg in oos_configs:
            yr = cfg["year"]
            preds = cfg["predictions"]
            rets = cfg["returns_oos"]
            pw = cfg.get("proba_weights")

            ftr = evaluate_forward(
                preds,
                rets,
                ohlcv_by_ticker,
                year=yr,
                costs_bps=costs_bps,
                proba_weights=pw,
                sector_map=sector_map,
                max_sector_weight=max_sector_weight,
                top_n=top_n,
                min_winners=min_winners,
            )
            per_year[yr] = ftr
            logger.info(
                "Multi-year OOS %d: accuracy=%.3f  Long-Winners cum=%.3f",
                yr,
                ftr.classification.get("accuracy", float("nan")),
                ftr.long_only.get("cumulative_return", float("nan")),
            )

    elif oos_years is not None and train_result is not None:
        try:
            from src.model import predict as model_predict
            from src.model import predict_proba as model_predict_proba
        except ImportError:
            from model import predict as model_predict
            from model import predict_proba as model_predict_proba

        for yr in oos_years:
            cutoff = f"{yr - 1}-12-31"
            X_oos = build_oos_features(
                ohlcv_by_ticker,
                fundamentals_by_ticker,
                cutoff_date=cutoff,
            )
            preds = model_predict(train_result, X_oos)
            rets = compute_oos_returns(ohlcv_by_ticker, year=yr)

            pw: pd.Series | None = None
            if proba_weights_fn is not None:
                pw = proba_weights_fn(train_result, X_oos)
            elif _is_multi_quarter_ensemble(train_result):
                proba_df = model_predict_proba(train_result, X_oos)
                if "Winners" in proba_df.columns:
                    pw = proba_df["Winners"]
            elif hasattr(train_result, "model") and hasattr(
                train_result.model, "predict_proba"
            ):
                proba_df = model_predict_proba(train_result, X_oos)
                if "Winners" in proba_df.columns:
                    pw = proba_df["Winners"]

            ftr = evaluate_forward(
                preds,
                rets,
                ohlcv_by_ticker,
                year=yr,
                costs_bps=costs_bps,
                proba_weights=pw,
                sector_map=sector_map,
                max_sector_weight=max_sector_weight,
                top_n=top_n,
                min_winners=min_winners,
            )
            per_year[yr] = ftr
            logger.info(
                "Multi-year OOS %d: accuracy=%.3f  Long-Winners cum=%.3f",
                yr,
                ftr.classification.get("accuracy", float("nan")),
                ftr.long_only.get("cumulative_return", float("nan")),
            )
    else:
        raise ValueError(
            "Provide either (oos_years + train_result) or oos_configs",
        )

    summary = _build_multi_year_summary(per_year)
    return MultiYearForwardResult(per_year=per_year, summary=summary)


def _build_regime_multi_year_summary(
    per_year: dict[int, RegimeForwardTestResult],
) -> pd.DataFrame:
    """Like :func:`_build_multi_year_summary` with regime columns per OOS year."""
    rows: list[dict[str, Any]] = []
    for yr in sorted(per_year):
        rf = per_year[yr]
        ftr = rf.forward
        row: dict[str, Any] = {"year": yr}
        row["regime"] = rf.regime_label
        row["regime_confidence"] = rf.regime_confidence
        row["cutoff_date"] = rf.cutoff_date
        row["accuracy"] = ftr.classification.get("accuracy", float("nan"))
        row["f1_macro"] = ftr.classification.get("f1_macro", float("nan"))
        row["long_winners_cum"] = ftr.long_only.get(
            "cumulative_return", float("nan"),
        )
        row["long_winners_sharpe"] = ftr.long_only.get(
            "sharpe_ratio", float("nan"),
        )
        row["long_short_cum"] = ftr.long_short.get(
            "cumulative_return", float("nan"),
        )
        row["benchmark_cum"] = ftr.benchmark.get(
            "cumulative_return", float("nan"),
        )
        row["long_winners_maxdd"] = ftr.long_only.get(
            "max_drawdown", float("nan"),
        )

        winner_hr = ftr.hit_rates.get("Winners", {})
        row["winner_hit_rate"] = winner_hr.get("hit_rate", float("nan"))
        row["n_winners_predicted"] = winner_hr.get("n_predicted", 0)
        rows.append(row)

    avg_row: dict[str, Any] = {"year": "Average"}
    df = pd.DataFrame(rows)
    for col in df.columns:
        if col == "year":
            continue
        if col in ("regime", "cutoff_date"):
            avg_row[col] = "—"
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        avg_row[col] = float(vals.mean()) if not vals.empty else float("nan")
    rows.append(avg_row)

    worst_row: dict[str, Any] = {"year": "Worst"}
    for col in df.columns:
        if col == "year":
            continue
        if col in ("regime", "cutoff_date"):
            worst_row[col] = "—"
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        if col in ("long_winners_maxdd",):
            worst_row[col] = float(vals.min()) if not vals.empty else float("nan")
        elif col.endswith("_cum") or col.endswith("_sharpe") or col in (
            "accuracy",
            "f1_macro",
            "winner_hit_rate",
            "regime_confidence",
        ):
            worst_row[col] = float(vals.min()) if not vals.empty else float("nan")
        else:
            worst_row[col] = float(vals.min()) if not vals.empty else float("nan")
    rows.append(worst_row)

    return pd.DataFrame(rows).set_index("year")


def evaluate_forward_multi_regime_aware(
    ohlcv_by_ticker: dict[str, pd.DataFrame],
    fundamentals_by_ticker: dict[str, dict[str, Any]] | None = None,
    *,
    regime_collection: Any,
    oos_years: list[int] | None = None,
    oos_configs: list[dict[str, Any]] | None = None,
    costs_bps: float = 0.0,
    proba_weights_fn: Any | None = None,
    sector_map: dict[str, str] | None = None,
    max_sector_weight: float | None = None,
    top_n: int | None = None,
    min_winners: int | None = None,
) -> RegimeMultiYearForwardResult:
    """Walk-forward OOS evaluation using :func:`model.predict_regime_aware`.

    For each OOS year, features are built with cutoff ``{year−1}-12-31``, the
    regime is detected on that date (inside :func:`predict_proba_regime_aware`),
    and probabilities are blended across regime models when confidence is low.

    Parameters
    ----------
    ohlcv_by_ticker, fundamentals_by_ticker
        Same as :func:`evaluate_forward_multi`.
    regime_collection
        Trained :class:`~model.RegimeModelCollection`.
    oos_years
        Calendar years to evaluate (simple mode). Mutually exclusive with
        ``oos_configs``.
    oos_configs
        Advanced mode: each dict must include ``year``, ``predictions``,
        ``returns_oos``. Optional: ``proba_weights``, ``cutoff_date``.
    costs_bps, sector_map, max_sector_weight, top_n, min_winners
        Passed to :func:`evaluate_forward_regime_aware`.
    proba_weights_fn
        Optional ``(regime_collection, X_oos, smi_ohlcv, cutoff_date) -> Series``.
        When ``None``, uses ``P(Winners)`` from :func:`predict_proba_regime_aware`.

    Returns
    -------
    RegimeMultiYearForwardResult
        Per-year :class:`RegimeForwardTestResult` and a summary table including
        ``regime`` / ``regime_confidence`` columns.
    """
    try:
        from src.features import MACRO_BENCHMARK_TICKER
        from src.model import predict_proba_regime_aware, predict_regime_aware
    except ImportError:
        from features import MACRO_BENCHMARK_TICKER
        from model import predict_proba_regime_aware, predict_regime_aware

    smi = ohlcv_by_ticker.get(MACRO_BENCHMARK_TICKER)
    if smi is None or smi.empty:
        raise ValueError(
            f"regime-aware multi-year eval requires {MACRO_BENCHMARK_TICKER!r} in "
            "ohlcv_by_ticker",
        )

    per_year: dict[int, RegimeForwardTestResult] = {}

    if oos_configs is not None:
        for cfg in oos_configs:
            yr = int(cfg["year"])
            preds = cfg["predictions"]
            rets = cfg["returns_oos"]
            pw = cfg.get("proba_weights")
            cutoff = cfg.get("cutoff_date") or _default_regime_cutoff_for_oos_year(yr)

            rf = evaluate_forward_regime_aware(
                preds,
                rets,
                ohlcv_by_ticker,
                year=yr,
                cutoff_date=cutoff,
                costs_bps=costs_bps,
                proba_weights=pw,
                sector_map=sector_map,
                max_sector_weight=max_sector_weight,
                top_n=top_n,
                min_winners=min_winners,
            )
            per_year[yr] = rf
            ftr = rf.forward
            logger.info(
                "Regime-aware multi-year OOS %d: regime=%s acc=%.3f  Long-Winners cum=%.3f",
                yr,
                rf.regime_label.upper(),
                ftr.classification.get("accuracy", float("nan")),
                ftr.long_only.get("cumulative_return", float("nan")),
            )

    elif oos_years is not None:
        for yr in oos_years:
            cutoff = _default_regime_cutoff_for_oos_year(yr)
            X_oos = build_oos_features(
                ohlcv_by_ticker,
                fundamentals_by_ticker,
                cutoff_date=cutoff,
            )
            preds = predict_regime_aware(regime_collection, X_oos, smi, cutoff)
            rets = compute_oos_returns(ohlcv_by_ticker, year=yr)

            pw: pd.Series | None = None
            if proba_weights_fn is not None:
                pw = proba_weights_fn(regime_collection, X_oos, smi, cutoff)
            else:
                proba_df = predict_proba_regime_aware(
                    regime_collection, X_oos, smi, cutoff,
                )
                if "Winners" in proba_df.columns:
                    pw = proba_df["Winners"]

            rf = evaluate_forward_regime_aware(
                preds,
                rets,
                ohlcv_by_ticker,
                year=yr,
                cutoff_date=cutoff,
                costs_bps=costs_bps,
                proba_weights=pw,
                sector_map=sector_map,
                max_sector_weight=max_sector_weight,
                top_n=top_n,
                min_winners=min_winners,
            )
            per_year[yr] = rf
            ftr = rf.forward
            logger.info(
                "Regime-aware multi-year OOS %d: regime=%s acc=%.3f  Long-Winners cum=%.3f",
                yr,
                rf.regime_label.upper(),
                ftr.classification.get("accuracy", float("nan")),
                ftr.long_only.get("cumulative_return", float("nan")),
            )
    else:
        raise ValueError(
            "Provide either (oos_years + regime_collection) or oos_configs",
        )

    summary = _build_regime_multi_year_summary(per_year)
    return RegimeMultiYearForwardResult(per_year=per_year, summary=summary)


def _capped_proba_weights(
    pw: pd.Series,
    tickers: list[str],
    max_weight: float | None = None,
) -> pd.Series:
    """Normalised P(Winner)-proportional weights with optional per-position cap.

    Iteratively clips weights at *max_weight* and redistributes the excess
    proportionally to uncapped positions until convergence.
    """
    w = pw.reindex(tickers).fillna(0.0)
    w_sum = w.sum()
    if w_sum <= 0:
        return pd.Series(1.0 / len(tickers), index=tickers, name="weight")
    w = w / w_sum

    if max_weight is not None and 0 < max_weight < 1.0:
        for _ in range(20):
            over = w > max_weight
            if not over.any():
                break
            excess = (w[over] - max_weight).sum()
            w[over] = max_weight
            under = ~over
            under_sum = w[under].sum()
            if under_sum > 0:
                w[under] += excess * (w[under] / under_sum)
            else:
                w = pd.Series(1.0 / len(tickers), index=tickers)
                break

    w.name = "weight"
    return w


def evaluate_forward_quarterly_regime_aware(
    ohlcv_by_ticker: dict[str, pd.DataFrame],
    fundamentals_by_ticker: dict[str, dict[str, Any]] | None = None,
    *,
    regime_collection: Any,
    oos_years: list[int],
    costs_bps: float = 40.0,
    min_winners: int = 5,
    rebalance_freq: int = 1,
    hysteresis_rule: str = "keep_non_losers",
    max_position_weight: float | None = 0.30,
) -> QuarterlyForwardResult:
    """Walk-forward evaluation with periodic rebalancing, hysteresis, and proportional costs.

    Applies an already-trained :class:`~model.RegimeModelCollection` at each
    rebalancing cutoff within a year.  Positions are held or swapped according
    to *hysteresis_rule*; realistic transaction costs are proportional to
    turnover.

    Parameters
    ----------
    ohlcv_by_ticker
        Ticker -> OHLCV DataFrame.  Must include the macro benchmark
        (``^SSMI``) for regime detection.
    fundamentals_by_ticker
        Ticker -> yfinance ``.info`` dict (for feature building).
    regime_collection
        Trained :class:`~model.RegimeModelCollection`.
    oos_years
        Calendar years to evaluate.
    costs_bps
        One-way transaction costs in basis points (default 40, realistic CH).
    min_winners
        Target number of portfolio positions.
    rebalance_freq
        ``1`` = quarterly (rebalance at Q1/Q2/Q3 ends),
        ``2`` = semi-annual (mid-year only),
        ``4`` (or anything else) = annual (no mid-year rebalancing).
    hysteresis_rule
        ``"keep_non_losers"`` retains positions not predicted as Losers and
        fills vacated slots with the next-best candidates by ``P(Winner)``.
        Any other value triggers a full top-N rebalance at each cutoff.
    max_position_weight
        Cap on any single position's weight (default 0.30 = 30%).  Weights
        are P(Winner)-proportional, then iteratively clipped at this cap with
        excess redistributed to uncapped positions.  ``None`` disables the
        cap (pure P(Winner)-proportional).

    Returns
    -------
    QuarterlyForwardResult
    """
    try:
        from src.features import MACRO_BENCHMARK_TICKER
        from src.model import predict_proba_regime_aware
    except ImportError:
        from features import MACRO_BENCHMARK_TICKER
        from model import predict_proba_regime_aware

    smi = ohlcv_by_ticker.get(MACRO_BENCHMARK_TICKER)
    if smi is None or smi.empty:
        raise ValueError(
            f"Quarterly regime-aware eval requires {MACRO_BENCHMARK_TICKER!r} "
            "in ohlcv_by_ticker",
        )

    per_year: dict[int, ForwardTestResult] = {}
    quarterly_rows: list[dict[str, Any]] = []
    turnover_rows: list[dict[str, Any]] = []
    total_costs_all = 0.0

    for yr in oos_years:
        schedule = build_quarterly_rebalance_schedule(yr, rebalance_freq)
        all_cutoffs = [p.cutoff for p in schedule]
        n_periods = len(schedule)

        # ---- predictions at each cutoff ----
        period_data: list[tuple[pd.Series, pd.Series | None]] = []
        universe_tickers: set[str] = set()

        for cutoff in all_cutoffs:
            X_oos = build_oos_features(
                ohlcv_by_ticker, fundamentals_by_ticker, cutoff_date=cutoff,
            )
            proba_df = predict_proba_regime_aware(
                regime_collection, X_oos, smi, cutoff,
            )
            cls_names = list(proba_df.columns)
            pred_idx = np.argmax(proba_df.values, axis=1)
            preds = pd.Series(
                [cls_names[k] for k in pred_idx],
                index=X_oos.index,
                name="predicted_group",
            )
            pw = proba_df["Winners"] if "Winners" in proba_df.columns else None
            period_data.append((preds, pw))
            universe_tickers.update(X_oos.index.tolist())

        # ---- full-year close & daily-return matrices ----
        all_tickers = sorted(universe_tickers)
        close = _daily_close_matrix(ohlcv_by_ticker, all_tickers, yr)
        if close.empty:
            logger.warning("No close data for year %d — skipping", yr)
            continue
        daily_ret_matrix = close.pct_change()

        # benchmark: equal-weight initial universe, no costs, full year hold
        initial_preds, _ = period_data[0]
        bm_tickers = [t for t in initial_preds.index if t in close.columns]
        bm_daily = daily_ret_matrix[bm_tickers].mean(axis=1).dropna()
        bm_daily.name = "benchmark"
        bm_metrics = _compute_metrics(bm_daily)

        # ---- period-by-period portfolio construction ----
        current_portfolio: list[str] = []
        portfolio_segments: list[pd.Series] = []
        year_cost_bps = 0.0

        for i in range(n_periods):
            preds, pw = period_data[i]
            seg = schedule[i]
            p_start = seg.period_start
            p_end = seg.period_end
            q_label = seg.label
            prev_portfolio = list(current_portfolio)

            if i == 0:
                # initial formation — all predicted Winners first, then
                # fill to min_winners by P(Winner) (matching Phase 5 logic)
                winners = [
                    t for t in preds[preds == "Winners"].index
                    if t in close.columns
                ]
                if len(winners) >= min_winners:
                    if pw is not None:
                        w_sorted = pw.reindex(winners).sort_values(ascending=False)
                        current_portfolio = list(w_sorted.head(min_winners).index)
                    else:
                        current_portfolio = winners[:min_winners]
                else:
                    current_portfolio = list(winners)
                    n_to_fill = min_winners - len(current_portfolio)
                    if n_to_fill > 0 and pw is not None:
                        excluded = set(current_portfolio)
                        cand = pw.drop(
                            labels=[t for t in excluded if t in pw.index],
                            errors="ignore",
                        )
                        cand = cand[cand.index.isin(close.columns)]
                        fill = list(cand.nlargest(n_to_fill).index)
                        current_portfolio = current_portfolio + fill
                    logger.info(
                        "Winners-first fill: %d predicted Winners, "
                        "filled to %d (min_winners=%d)",
                        len(winners), len(current_portfolio), min_winners,
                    )
                n_swapped, turnover, sin, sout = turnover_from_portfolio_change(
                    [], current_portfolio, is_initial=True,
                )
                swapped_in, swapped_out = set(sin), set(sout)
            else:
                if hysteresis_rule == "keep_non_losers":
                    kept = [
                        t for t in prev_portfolio
                        if t in preds.index
                        and preds[t] != "Losers"
                        and t in close.columns
                    ]
                    n_to_fill = max(0, min_winners - len(kept))
                    if n_to_fill > 0:
                        excluded = set(kept)
                        new_winners = [
                            t for t in preds[preds == "Winners"].index
                            if t not in excluded and t in close.columns
                        ]
                        if pw is not None:
                            pw_sub = pw.reindex(new_winners).sort_values(
                                ascending=False,
                            )
                            new_winners = list(pw_sub.index)
                        fill = new_winners[:n_to_fill]
                        still_needed = n_to_fill - len(fill)
                        if still_needed > 0 and pw is not None:
                            already = excluded | set(fill)
                            cand = pw.drop(
                                labels=[t for t in already if t in pw.index],
                                errors="ignore",
                            )
                            cand = cand[cand.index.isin(close.columns)]
                            fill += list(cand.nlargest(still_needed).index)
                        current_portfolio = kept + fill
                    else:
                        current_portfolio = kept[:min_winners]
                else:
                    winners = [
                        t for t in preds[preds == "Winners"].index
                        if t in close.columns
                    ]
                    if len(winners) >= min_winners:
                        if pw is not None:
                            w_sorted = pw.reindex(winners).sort_values(
                                ascending=False,
                            )
                            current_portfolio = list(
                                w_sorted.head(min_winners).index,
                            )
                        else:
                            current_portfolio = winners[:min_winners]
                    else:
                        current_portfolio = list(winners)
                        n_to_fill = min_winners - len(current_portfolio)
                        if n_to_fill > 0 and pw is not None:
                            excluded = set(current_portfolio)
                            cand = pw.drop(
                                labels=[
                                    t for t in excluded if t in pw.index
                                ],
                                errors="ignore",
                            )
                            cand = cand[cand.index.isin(close.columns)]
                            fill = list(cand.nlargest(n_to_fill).index)
                            current_portfolio = current_portfolio + fill

                n_swapped, turnover, sin, sout = turnover_from_portfolio_change(
                    prev_portfolio, current_portfolio, is_initial=False,
                )
                swapped_in, swapped_out = set(sin), set(sout)

            # ---- period daily returns (P(Winner)-weighted with position cap) ----
            period_dr = daily_ret_matrix.loc[p_start:p_end]
            available = [t for t in current_portfolio if t in period_dr.columns]
            if not available or period_dr.empty:
                logger.warning(
                    "Year %d period %d: no portfolio data — skipping", yr, i,
                )
                continue

            if pw is not None:
                w = _capped_proba_weights(pw, available, max_position_weight)
                segment = (period_dr[available] * w).sum(axis=1).dropna()
            else:
                segment = period_dr[available].mean(axis=1).dropna()
            segment.name = "portfolio"
            if segment.empty:
                continue

            # ---- transaction costs ----
            period_cost_bps = 0.0
            if i == 0:
                segment.iloc[0] -= costs_bps / 10_000.0
                period_cost_bps += costs_bps
            elif n_swapped > 0:
                rebal_cost_bps = turnover * costs_bps * 2
                segment.iloc[0] -= rebal_cost_bps / 10_000.0
                period_cost_bps += rebal_cost_bps

            if i == n_periods - 1:
                segment.iloc[-1] -= costs_bps / 10_000.0
                period_cost_bps += costs_bps

            year_cost_bps += period_cost_bps
            portfolio_segments.append(segment)

            # ---- logging & detail rows ----
            seg_m = _compute_metrics(segment)

            quarterly_rows.append({
                "year": yr,
                "quarter": q_label,
                "cutoff": all_cutoffs[i],
                "start": str(p_start.date()),
                "end": str(p_end.date()),
                "n_positions": len(available),
                "n_swapped": n_swapped,
                "turnover_pct": turnover * 100,
                "cost_bps": period_cost_bps,
                "cum_return": seg_m["cumulative_return"],
                "sharpe": seg_m["sharpe_ratio"],
                "max_dd": seg_m["max_drawdown"],
                "n_trading_days": seg_m["n_trading_days"],
            })

            turnover_rows.append({
                "year": yr,
                "quarter": q_label,
                "cutoff": all_cutoffs[i],
                "n_swapped": n_swapped,
                "turnover_pct": turnover * 100,
                "cost_bps": period_cost_bps,
                "held": list(available),
                "swapped_in": sorted(swapped_in),
                "swapped_out": sorted(swapped_out),
            })

            w_info = ""
            if pw is not None:
                w_cur = _capped_proba_weights(pw, available, max_position_weight)
                w_info = (
                    f" weights=[{', '.join(f'{t}:{v:.0%}' for t, v in w_cur.items())}]"
                )

            logger.info(
                "Hysteresis: year=%d %s kept %d/%d, replaced %d "
                "(turnover=%.0f%%, cost=%.0fbps)%s",
                yr,
                q_label,
                len(available) - n_swapped if i > 0 else 0,
                len(available),
                n_swapped,
                turnover * 100,
                period_cost_bps,
                w_info,
            )

        # ---- concatenate segments → full year ----
        if portfolio_segments:
            full_year_ret = pd.concat(portfolio_segments)
            full_year_ret.name = "portfolio"
        else:
            full_year_ret = pd.Series(dtype=float, name="portfolio")

        lo_metrics = _compute_metrics(full_year_ret)

        # classification metrics (initial-cutoff predictions vs actual groups)
        rets_oos = compute_oos_returns(ohlcv_by_ticker, year=yr)
        _, assign_groups = _import_classifier()
        evaluate_preds_fn = _import_model()

        common = initial_preds.dropna().index.intersection(rets_oos.dropna().index)
        if len(common) > 0:
            actual_groups = assign_groups(rets_oos.loc[common])
            preds_eval = initial_preds.loc[common]
            cls_names_eval = sorted(
                set(preds_eval.unique()) | set(actual_groups.dropna().unique()),
            )
            classification = evaluate_preds_fn(
                actual_groups.values, preds_eval.values, cls_names_eval,
            )
            hr = hit_rate_by_group(preds_eval, actual_groups)
        else:
            actual_groups = pd.Series(dtype=str)
            classification = {
                "accuracy": float("nan"),
                "f1_macro": float("nan"),
                "f1_weighted": float("nan"),
                "confusion_matrix": np.zeros((0, 0), dtype=int),
                "report_dict": {},
                "report_str": "",
            }
            hr = {}

        ftr = ForwardTestResult(
            predictions=initial_preds,
            actual_returns=rets_oos,
            actual_groups=actual_groups,
            long_only=lo_metrics,
            long_short={
                "cumulative_return": float("nan"),
                "sharpe_ratio": float("nan"),
                "max_drawdown": float("nan"),
            },
            benchmark=bm_metrics,
            classification=classification,
            hit_rates=hr,
            daily_returns={"long_winners": full_year_ret, "benchmark": bm_daily},
            costs_bps=year_cost_bps,
        )
        per_year[yr] = ftr
        total_costs_all += year_cost_bps

        beat_bm = lo_metrics.get(
            "cumulative_return", float("nan"),
        ) > bm_metrics.get("cumulative_return", float("nan"))
        logger.info(
            "Quarterly eval yr=%d freq=%d: cum=%.3f sharpe=%.2f maxDD=%.3f "
            "bm=%.3f costs=%.0fbps beat_bm=%s",
            yr,
            rebalance_freq,
            lo_metrics.get("cumulative_return", float("nan")),
            lo_metrics.get("sharpe_ratio", float("nan")),
            lo_metrics.get("max_drawdown", float("nan")),
            bm_metrics.get("cumulative_return", float("nan")),
            year_cost_bps,
            beat_bm,
        )

    return QuarterlyForwardResult(
        per_year=per_year,
        quarterly_detail=pd.DataFrame(quarterly_rows),
        turnover_log=pd.DataFrame(turnover_rows),
        total_costs_bps=total_costs_all,
        rebalance_freq=rebalance_freq,
    )


def evaluate_forward_quarterly_regression(
    ohlcv_by_ticker: dict[str, pd.DataFrame],
    fundamentals_by_ticker: dict[str, dict[str, Any]] | None = None,
    *,
    regression_result: Any,
    oos_years: list[int],
    costs_bps: float = 40.0,
    top_n: int = 5,
    rebalance_freq: int = 1,
    hysteresis_buffer: int = 2,
    max_position_weight: float | None = 0.30,
    publication_lag_days: int = 0,
    cutoff_shift_days: int = 0,
) -> QuarterlyForwardResult:
    """Walk-forward evaluation using a single regression model with rank-based hysteresis.

    At each rebalancing cutoff the trained regressor predicts 1-month forward
    returns for every ticker in the universe.  The portfolio is formed from
    the top-*top_n* predicted returns, subject to rank-based hysteresis
    (a position is retained as long as its rank stays within
    ``top_n + hysteresis_buffer``).  Weights are proportional to predicted
    returns (shifted non-negative, then capped at *max_position_weight*).

    Parameters
    ----------
    ohlcv_by_ticker
        Ticker -> OHLCV DataFrame.
    fundamentals_by_ticker
        Ticker -> yfinance ``.info`` dict (for feature building).
    regression_result
        Trained :class:`~regression_model.RegressionTrainResult`.
    oos_years
        Calendar years to evaluate.
    costs_bps
        One-way transaction costs in basis points (default 40).
    top_n
        Number of long positions to hold.
    rebalance_freq
        ``1`` = quarterly, ``2`` = semi-annual, else annual.
    hysteresis_buffer
        Retain a position while its rank stays within ``top_n + buffer``.
    max_position_weight
        Cap on any single position's weight (default 0.30 = 30 %).
        ``None`` disables the cap.
    publication_lag_days
        Passed to :func:`build_oos_features` for PIT fundamentals (Eulerpool).
    cutoff_shift_days
        Shift all rebalancing cutoffs forward by this many calendar days
        (publication-lag shift).  See :func:`build_quarterly_rebalance_schedule`.

    Returns
    -------
    QuarterlyForwardResult
    """
    try:
        from src.regression_backtest import _select_long_with_hysteresis
        from src.regression_model import predict_returns
    except ImportError:
        from regression_backtest import _select_long_with_hysteresis
        from regression_model import predict_returns

    from scipy.stats import spearmanr

    per_year: dict[int, ForwardTestResult] = {}
    quarterly_rows: list[dict[str, Any]] = []
    turnover_rows: list[dict[str, Any]] = []
    total_costs_all = 0.0

    for yr in oos_years:
        schedule = build_quarterly_rebalance_schedule(
            yr, rebalance_freq, cutoff_shift_days,
        )
        all_cutoffs = [p.cutoff for p in schedule]
        n_periods = len(schedule)

        # ---- predictions at each cutoff ----
        period_preds: list[pd.Series] = []
        universe_tickers: set[str] = set()

        for cutoff in all_cutoffs:
            X_oos = build_oos_features(
                ohlcv_by_ticker,
                fundamentals_by_ticker,
                cutoff_date=cutoff,
                publication_lag_days=publication_lag_days,
            )
            pred = predict_returns(regression_result, X_oos)
            period_preds.append(pred)
            universe_tickers.update(X_oos.index.tolist())

        # ---- close & daily-return matrices covering all periods ----
        all_tickers = sorted(universe_tickers)
        data_start = schedule[0].period_start
        data_end = schedule[-1].period_end
        close = _daily_close_matrix(
            ohlcv_by_ticker, all_tickers, yr,
            start_date=data_start, end_date=data_end,
        )
        if close.empty:
            logger.warning("No close data for year %d — skipping", yr)
            continue
        daily_ret_matrix = close.pct_change()

        initial_pred = period_preds[0]
        bm_tickers = [t for t in initial_pred.index if t in close.columns]
        bm_daily = daily_ret_matrix[bm_tickers].mean(axis=1).dropna()
        bm_daily.name = "benchmark"
        bm_metrics = _compute_metrics(bm_daily)

        # ---- period-by-period portfolio construction ----
        current_portfolio: list[str] = []
        portfolio_segments: list[pd.Series] = []
        year_cost_bps = 0.0
        ic_values: list[float] = []
        precision_strict: list[float] = []
        precision_top_quartile: list[float] = []
        avg_actual_ranks: list[float] = []

        for i in range(n_periods):
            pred = period_preds[i]
            seg = schedule[i]
            p_start = seg.period_start
            p_end = seg.period_end
            q_label = seg.label
            prev_portfolio = list(current_portfolio)

            pred_tradeable = pred.reindex(
                [t for t in pred.index if t in close.columns],
            ).dropna()

            current_portfolio = _select_long_with_hysteresis(
                pred_tradeable, prev_portfolio, top_n, hysteresis_buffer,
            )

            n_swapped, turnover, sin, sout = turnover_from_portfolio_change(
                prev_portfolio, current_portfolio, is_initial=(i == 0),
            )
            swapped_in, swapped_out = set(sin), set(sout)

            # ---- period daily returns (return-proportional + capped) ----
            period_dr = daily_ret_matrix.loc[p_start:p_end]
            available = [t for t in current_portfolio if t in period_dr.columns]
            if not available or period_dr.empty:
                logger.warning(
                    "Year %d period %d: no portfolio data — skipping", yr, i,
                )
                continue

            pred_score = pred.reindex(available) - pred.reindex(available).min() + 1e-12
            w = _capped_proba_weights(pred_score, available, max_position_weight)
            segment = (period_dr[available] * w).sum(axis=1).dropna()
            segment.name = "portfolio"
            if segment.empty:
                continue

            # ---- transaction costs ----
            period_cost_bps = 0.0
            if i == 0:
                segment.iloc[0] -= costs_bps / 10_000.0
                period_cost_bps += costs_bps
            elif n_swapped > 0:
                rebal_cost_bps = turnover * costs_bps * 2
                segment.iloc[0] -= rebal_cost_bps / 10_000.0
                period_cost_bps += rebal_cost_bps

            if i == n_periods - 1:
                segment.iloc[-1] -= costs_bps / 10_000.0
                period_cost_bps += costs_bps

            year_cost_bps += period_cost_bps
            portfolio_segments.append(segment)

            # ---- IC & Precision@N ----
            period_close = close.loc[p_start:p_end]
            if len(period_close) >= 2:
                actual_period_ret = period_close.iloc[-1] / period_close.iloc[0] - 1
                common_ic = pred_tradeable.index.intersection(
                    actual_period_ret.dropna().index,
                )
                if len(common_ic) >= 5:
                    corr, _ = spearmanr(
                        pred_tradeable.reindex(common_ic).values,
                        actual_period_ret.reindex(common_ic).values,
                    )
                    ic_values.append(float(corr))

                    actual_sorted = actual_period_ret.reindex(common_ic).sort_values(ascending=False)
                    actual_top_n = set(actual_sorted.head(top_n).index)
                    actual_top_quartile = set(
                        actual_sorted.head(max(1, len(common_ic) // 4)).index,
                    )

                    picks = set(available)
                    n_strict = len(picks & actual_top_n)
                    n_loose = len(picks & actual_top_quartile)
                    precision_strict.append(n_strict / top_n)
                    precision_top_quartile.append(n_loose / top_n)

                    actual_ranks = actual_period_ret.reindex(common_ic).rank(ascending=False)
                    pick_ranks = [float(actual_ranks.loc[t]) for t in available if t in actual_ranks.index]
                    if pick_ranks:
                        avg_actual_ranks.append(float(np.mean(pick_ranks)))

            # ---- logging & detail rows ----
            seg_m = _compute_metrics(segment)

            quarterly_rows.append({
                "year": yr,
                "quarter": q_label,
                "cutoff": all_cutoffs[i],
                "start": str(p_start.date()),
                "end": str(p_end.date()),
                "n_positions": len(available),
                "n_swapped": n_swapped,
                "turnover_pct": turnover * 100,
                "cost_bps": period_cost_bps,
                "cum_return": seg_m["cumulative_return"],
                "sharpe": seg_m["sharpe_ratio"],
                "max_dd": seg_m["max_drawdown"],
                "n_trading_days": seg_m["n_trading_days"],
            })

            turnover_rows.append({
                "year": yr,
                "quarter": q_label,
                "cutoff": all_cutoffs[i],
                "n_swapped": n_swapped,
                "turnover_pct": turnover * 100,
                "cost_bps": period_cost_bps,
                "held": list(available),
                "swapped_in": sorted(swapped_in),
                "swapped_out": sorted(swapped_out),
            })

            w_info = (
                f" weights=[{', '.join(f'{t}:{v:.0%}' for t, v in w.items())}]"
            )
            logger.info(
                "Regression hysteresis: year=%d %s kept %d/%d, replaced %d "
                "(turnover=%.0f%%, cost=%.0fbps)%s",
                yr,
                q_label,
                len(available) - n_swapped if i > 0 else 0,
                len(available),
                n_swapped,
                turnover * 100,
                period_cost_bps,
                w_info,
            )

        # ---- concatenate segments → full year ----
        if portfolio_segments:
            full_year_ret = pd.concat(portfolio_segments)
            full_year_ret.name = "portfolio"
        else:
            full_year_ret = pd.Series(dtype=float, name="portfolio")

        lo_metrics = _compute_metrics(full_year_ret)

        # ---- regression metrics (IC + Precision@N) ----
        mean_ic = float(np.nanmean(ic_values)) if ic_values else float("nan")
        ic_std = float(np.nanstd(ic_values)) if len(ic_values) > 1 else float("nan")
        mean_prec_strict = float(np.nanmean(precision_strict)) if precision_strict else float("nan")
        mean_prec_quartile = float(np.nanmean(precision_top_quartile)) if precision_top_quartile else float("nan")
        mean_avg_rank = float(np.nanmean(avg_actual_ranks)) if avg_actual_ranks else float("nan")

        rets_oos = compute_oos_returns(ohlcv_by_ticker, year=yr)

        classification: dict[str, Any] = {
            "accuracy": float("nan"),
            "f1_macro": float("nan"),
            "f1_weighted": float("nan"),
            "confusion_matrix": np.zeros((0, 0), dtype=int),
            "report_dict": {},
            "report_str": "",
            "ic": mean_ic,
            "ic_std": ic_std,
            "ic_per_cutoff": list(ic_values),
            "precision_at_n": mean_prec_strict,
            "precision_top_quartile": mean_prec_quartile,
            "avg_actual_rank": mean_avg_rank,
        }

        ftr = ForwardTestResult(
            predictions=initial_pred,
            actual_returns=rets_oos,
            actual_groups=pd.Series(dtype=str),
            long_only=lo_metrics,
            long_short={
                "cumulative_return": float("nan"),
                "sharpe_ratio": float("nan"),
                "max_drawdown": float("nan"),
            },
            benchmark=bm_metrics,
            classification=classification,
            hit_rates={},
            daily_returns={"long_winners": full_year_ret, "benchmark": bm_daily},
            costs_bps=year_cost_bps,
        )
        per_year[yr] = ftr
        total_costs_all += year_cost_bps

        beat_bm = lo_metrics.get(
            "cumulative_return", float("nan"),
        ) > bm_metrics.get("cumulative_return", float("nan"))
        logger.info(
            "Regression quarterly eval yr=%d freq=%d: cum=%.3f sharpe=%.2f "
            "maxDD=%.3f bm=%.3f costs=%.0fbps IC=%.3f beat_bm=%s",
            yr,
            rebalance_freq,
            lo_metrics.get("cumulative_return", float("nan")),
            lo_metrics.get("sharpe_ratio", float("nan")),
            lo_metrics.get("max_drawdown", float("nan")),
            bm_metrics.get("cumulative_return", float("nan")),
            year_cost_bps,
            mean_ic,
            beat_bm,
        )

    return QuarterlyForwardResult(
        per_year=per_year,
        quarterly_detail=pd.DataFrame(quarterly_rows),
        turnover_log=pd.DataFrame(turnover_rows),
        total_costs_bps=total_costs_all,
        rebalance_freq=rebalance_freq,
    )


def _build_multi_year_summary(
    per_year: dict[int, ForwardTestResult],
) -> pd.DataFrame:
    """Build aggregated summary table from per-year forward-test results."""
    rows: list[dict[str, Any]] = []
    for yr in sorted(per_year):
        ftr = per_year[yr]
        row: dict[str, Any] = {"year": yr}
        row["accuracy"] = ftr.classification.get("accuracy", float("nan"))
        row["f1_macro"] = ftr.classification.get("f1_macro", float("nan"))
        row["long_winners_cum"] = ftr.long_only.get(
            "cumulative_return", float("nan"),
        )
        row["long_winners_sharpe"] = ftr.long_only.get(
            "sharpe_ratio", float("nan"),
        )
        row["long_short_cum"] = ftr.long_short.get(
            "cumulative_return", float("nan"),
        )
        row["benchmark_cum"] = ftr.benchmark.get(
            "cumulative_return", float("nan"),
        )
        row["long_winners_maxdd"] = ftr.long_only.get(
            "max_drawdown", float("nan"),
        )

        winner_hr = ftr.hit_rates.get("Winners", {})
        row["winner_hit_rate"] = winner_hr.get("hit_rate", float("nan"))
        row["n_winners_predicted"] = winner_hr.get("n_predicted", 0)
        rows.append(row)

    avg_row: dict[str, Any] = {"year": "Average"}
    df = pd.DataFrame(rows)
    for col in df.columns:
        if col == "year":
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        avg_row[col] = float(vals.mean()) if not vals.empty else float("nan")
    rows.append(avg_row)

    worst_row: dict[str, Any] = {"year": "Worst"}
    for col in df.columns:
        if col == "year":
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        if col in ("long_winners_maxdd",):
            worst_row[col] = float(vals.min()) if not vals.empty else float("nan")
        elif col.endswith("_cum") or col.endswith("_sharpe") or col in (
            "accuracy",
            "f1_macro",
            "winner_hit_rate",
        ):
            worst_row[col] = float(vals.min()) if not vals.empty else float("nan")
        else:
            worst_row[col] = float(vals.min()) if not vals.empty else float("nan")
    rows.append(worst_row)

    return pd.DataFrame(rows).set_index("year")


def aggregate_forward_metrics_by_regime(
    per_year: dict[int, RegimeForwardTestResult],
) -> pd.DataFrame:
    """Mean OOS metrics grouped by detected regime (bull / bear / sideways).

    Parameters
    ----------
    per_year
        Mapping OOS calendar year → :class:`RegimeForwardTestResult` (e.g. from
        :attr:`RegimeMultiYearForwardResult.per_year`).

    Returns
    -------
    pd.DataFrame
        Index = regime label; columns include ``n_years``, mean long-Winner
        cumulative return, Sharpe, benchmark return, accuracy.
    """
    rows: list[dict[str, Any]] = []
    for yr in sorted(per_year):
        rf = per_year[yr]
        ftr = rf.forward
        rows.append(
            {
                "year": yr,
                "regime": rf.regime_label,
                "long_winners_cum": ftr.long_only.get(
                    "cumulative_return", float("nan"),
                ),
                "long_winners_sharpe": ftr.long_only.get(
                    "sharpe_ratio", float("nan"),
                ),
                "long_short_cum": ftr.long_short.get(
                    "cumulative_return", float("nan"),
                ),
                "benchmark_cum": ftr.benchmark.get(
                    "cumulative_return", float("nan"),
                ),
                "accuracy": ftr.classification.get("accuracy", float("nan")),
            },
        )
    detail = pd.DataFrame(rows)
    if detail.empty:
        return pd.DataFrame()

    agg = detail.groupby("regime", dropna=False).agg(
        n_years=("year", "count"),
        long_winners_cum_mean=("long_winners_cum", "mean"),
        long_winners_sharpe_mean=("long_winners_sharpe", "mean"),
        long_short_cum_mean=("long_short_cum", "mean"),
        benchmark_cum_mean=("benchmark_cum", "mean"),
        accuracy_mean=("accuracy", "mean"),
    )
    return agg


def _forward_summary_numeric_years(df: pd.DataFrame) -> pd.DataFrame:
    """Drop non-calendar-year index rows (e.g. Average / Worst)."""
    idx = pd.to_numeric(df.index, errors="coerce")
    return df.loc[idx.notna()].copy()


def compare_regime_aware_vs_single(
    regime_summary: pd.DataFrame,
    single_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Side-by-side walk-forward metrics: regime-aware vs single-model summaries.

    Parameters
    ----------
    regime_summary
        From :attr:`RegimeMultiYearForwardResult.summary` or any frame with the
        same metric columns as :func:`_build_multi_year_summary`, optionally
        plus ``regime`` / ``regime_confidence``.
    single_summary
        From :attr:`MultiYearForwardResult.summary`.

    Returns
    -------
    pd.DataFrame
        One row per OOS calendar year: regime columns (if present), paired
        metrics with ``_regime`` / ``_single`` suffixes, and ``*_delta`` =
        regime minus single for numeric columns.
    """
    r = _forward_summary_numeric_years(regime_summary)
    s = _forward_summary_numeric_years(single_summary)
    common_years = r.index.intersection(s.index)
    if len(common_years) == 0:
        logger.warning("compare_regime_aware_vs_single: no overlapping years")
        return pd.DataFrame()

    compare_cols = [
        c
        for c in r.columns
        if c in s.columns
        and c not in ("regime", "cutoff_date")
    ]
    out: dict[str, Any] = {}
    if "regime" in r.columns:
        out["regime"] = r.loc[common_years, "regime"]
    if "regime_confidence" in r.columns:
        out["regime_confidence"] = r.loc[common_years, "regime_confidence"]

    for c in compare_cols:
        vr = pd.to_numeric(r.loc[common_years, c], errors="coerce")
        vs = pd.to_numeric(s.loc[common_years, c], errors="coerce")
        out[f"{c}_regime"] = vr
        out[f"{c}_single"] = vs
        out[f"{c}_delta"] = vr - vs

    result = pd.DataFrame(out, index=common_years)
    result.index.name = "oos_year"
    result.sort_index(inplace=True)

    n_better_long = int(
        (result["long_winners_cum_delta"] > 0).sum()
    ) if "long_winners_cum_delta" in result.columns else 0
    if len(result) > 0:
        logger.info(
            "Regime vs single: regime wins long-Winner cum in %d/%d years",
            n_better_long,
            len(result),
        )
    return result


def hit_rate_by_group(
    predictions: pd.Series,
    actual_groups: pd.Series,
) -> dict[str, dict[str, Any]]:
    """Per-group hit rate and misclassification analysis.

    For each predicted group, reports how many were correctly classified,
    the hit rate, and the distribution of actual groups among the predicted
    members.

    Parameters
    ----------
    predictions
        Predicted group labels indexed by ticker.
    actual_groups
        Actual group labels indexed by ticker.

    Returns
    -------
    dict
        Group name → ``{n_predicted, n_correct, hit_rate, actual_distribution}``.
    """
    df = pd.DataFrame({"predicted": predictions, "actual": actual_groups}).dropna()
    groups = sorted(df["predicted"].unique())

    result: dict[str, dict[str, Any]] = {}
    for g in groups:
        mask = df["predicted"] == g
        n_predicted = int(mask.sum())
        if n_predicted == 0:
            result[g] = {
                "n_predicted": 0,
                "n_correct": 0,
                "hit_rate": float("nan"),
                "actual_distribution": {},
            }
            continue

        subset = df.loc[mask]
        n_correct = int((subset["actual"] == g).sum())
        hit_rate = n_correct / n_predicted
        actual_dist = subset["actual"].value_counts().to_dict()

        result[g] = {
            "n_predicted": n_predicted,
            "n_correct": n_correct,
            "hit_rate": float(hit_rate),
            "actual_distribution": actual_dist,
        }

    logger.info(
        "Hit rates: %s",
        {g: f"{v['hit_rate']:.1%}" for g, v in result.items()},
    )
    return result


def compare_insample_oos(
    insample_metrics: dict[str, Any],
    oos_metrics: dict[str, Any],
) -> pd.DataFrame:
    """Side-by-side comparison of in-sample hold-out vs OOS classification metrics.

    Parameters
    ----------
    insample_metrics
        Hold-out evaluation dict from in-sample training
        (``TrainResult.holdout_metrics`` or output of
        :func:`model.evaluate_predictions`).
    oos_metrics
        OOS classification dict from :attr:`ForwardTestResult.classification`.

    Returns
    -------
    pd.DataFrame
        Rows = metric names, columns = ``In-Sample`` / ``Out-of-Sample`` / ``Delta``.
    """
    keys = ["accuracy", "f1_macro", "f1_weighted"]
    rows: list[dict[str, Any]] = []
    for k in keys:
        ins = insample_metrics.get(k, float("nan"))
        oos = oos_metrics.get(k, float("nan"))
        delta = (
            oos - ins
            if isinstance(ins, (int, float))
            and isinstance(oos, (int, float))
            and np.isfinite(ins)
            and np.isfinite(oos)
            else float("nan")
        )
        rows.append(
            {"metric": k, "In-Sample": ins, "Out-of-Sample": oos, "Delta": delta},
        )
    return pd.DataFrame(rows).set_index("metric")


def strategy_summary(result: ForwardTestResult) -> pd.DataFrame:
    """Tabular summary of strategy performance vs benchmark.

    Parameters
    ----------
    result
        A :class:`ForwardTestResult` from :func:`evaluate_forward`.

    Returns
    -------
    pd.DataFrame
        One row per strategy with cumulative return, Sharpe, max drawdown, etc.
    """
    entries = [
        ("Long Winners", result.long_only),
        ("Long/Short", result.long_short),
        ("Benchmark (equal weight)", result.benchmark),
    ]
    rows: list[dict[str, Any]] = []
    for name, metrics in entries:
        row: dict[str, Any] = {"Strategy": name}
        row.update(metrics)
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


def plot_cumulative_returns(
    daily_returns: dict[str, pd.Series],
    *,
    title: str = "Cumulative returns — OOS 2025",
    ax: Any | None = None,
) -> Any:
    """Equity curves for each strategy."""
    import matplotlib.pyplot as plt

    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(12, 5))

    colours = {
        "long_winners": "#2ecc71",
        "short_losers": "#e74c3c",
        "long_short": "#3498db",
        "benchmark": "#7f8c8d",
    }
    labels = {
        "long_winners": "Long Winners",
        "short_losers": "Short Losers",
        "long_short": "Long/Short",
        "benchmark": "Benchmark (equal weight)",
    }

    for name, dr in daily_returns.items():
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


def plot_hit_rates(
    hit_rates: dict[str, dict[str, Any]],
    *,
    title: str = "Prediction hit rate by group",
    ax: Any | None = None,
) -> Any:
    """Bar chart of hit rates per predicted group."""
    import matplotlib.pyplot as plt

    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(7, 4))

    groups = list(hit_rates.keys())
    rates = [hit_rates[g]["hit_rate"] for g in groups]
    group_colours = {"Winners": "#2ecc71", "Steady": "#f39c12", "Losers": "#e74c3c"}
    bar_colours = [group_colours.get(g, "#3498db") for g in groups]

    bars = ax.bar(groups, rates, color=bar_colours, edgecolor="white")
    for bar, r, g in zip(bars, rates, groups):
        n = hit_rates[g]["n_predicted"]
        c = hit_rates[g]["n_correct"]
        label = f"{r:.0%}\n({c}/{n})"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_ylim(0, 1.15)
    ax.set_title(title)
    ax.set_ylabel("Hit rate")
    ax.axhline(
        1 / 3,
        color="gray",
        linewidth=0.7,
        linestyle="--",
        label="Random (1/3)",
    )
    ax.legend()

    if created:
        plt.tight_layout()
    return ax


def plot_oos_confusion_matrix(
    classification: dict[str, Any],
    class_names: list[str] | None = None,
    *,
    title: str = "OOS Confusion Matrix",
    normalize: bool = True,
    ax: Any | None = None,
) -> Any:
    """Heatmap of the OOS confusion matrix (delegates to :func:`model.plot_confusion_matrix`)."""
    try:
        from src.model import plot_confusion_matrix
    except ImportError:
        from model import plot_confusion_matrix

    if class_names is None:
        class_names = sorted(
            set(classification.get("report_dict", {}).keys())
            - {"accuracy", "macro avg", "weighted avg"},
        )

    return plot_confusion_matrix(
        classification,
        class_names,
        title=title,
        normalize=normalize,
        ax=ax,
    )


def plot_insample_vs_oos(
    comparison: pd.DataFrame,
    *,
    title: str = "In-Sample vs Out-of-Sample",
    ax: Any | None = None,
) -> Any:
    """Grouped bar chart of in-sample vs OOS metrics."""
    import matplotlib.pyplot as plt

    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(8, 4))

    x = np.arange(len(comparison))
    width = 0.35

    ins_vals = comparison["In-Sample"].values.astype(float)
    oos_vals = comparison["Out-of-Sample"].values.astype(float)

    ax.bar(x - width / 2, ins_vals, width, label="In-Sample", color="#3498db")
    ax.bar(x + width / 2, oos_vals, width, label="Out-of-Sample", color="#e74c3c")

    for i, (iv, ov) in enumerate(zip(ins_vals, oos_vals)):
        if np.isfinite(iv):
            ax.text(
                i - width / 2,
                iv + 0.01,
                f"{iv:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        if np.isfinite(ov):
            ax.text(
                i + width / 2,
                ov + 0.01,
                f"{ov:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ").title() for m in comparison.index])
    ax.set_ylim(0, 1.15)
    ax.set_title(title)
    ax.set_ylabel("Score")
    ax.legend()

    if created:
        plt.tight_layout()
    return ax
