"""Q1 performance labels: Winners / Losers / Steady.

Return-based classification with percentile cutoffs (primary) or K-Means (alternative),
optional threshold optimization (silhouette + group balance), and plotting helpers.
"""

from __future__ import annotations

import itertools
import logging
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

try:
    from config import (
        CLASS_Q_END,
        CLASS_Q_START,
        LOSER_PERCENTILE,
        RANDOM_SEED,
        WINNER_PERCENTILE,
    )
except ImportError:
    CLASS_Q_START = "2024-01-02"
    CLASS_Q_END = "2024-03-31"
    WINNER_PERCENTILE = 0.75
    LOSER_PERCENTILE = 0.25
    RANDOM_SEED = 42

logger = logging.getLogger(__name__)

CLASS_WINNERS = "Winners"
CLASS_LOSERS = "Losers"
CLASS_STEADY = "Steady"
CLASS_ORDER = (CLASS_WINNERS, CLASS_STEADY, CLASS_LOSERS)

ClassificationMethod = Literal["percentile", "kmeans", "auto"]

__all__ = [
    "CLASS_LOSERS",
    "CLASS_ORDER",
    "CLASS_STEADY",
    "CLASS_WINNERS",
    "assign_groups",
    "classification_quality",
    "compute_multi_period_returns",
    "compute_q1_returns",
    "optimize_percentile_cutoffs",
    "plot_group_characteristics",
    "plot_return_distribution",
    "total_return_q1",
]


def total_return_q1(close: pd.Series, q_start: str, q_end: str) -> float:
    """Total return ``(Close_end / Close_start) - 1`` for the classification window.

    Uses the first trading close on or after ``q_start`` and the last trading close
    on or before ``q_end`` (aligned with Q1 boundaries in :mod:`config`).

    Parameters
    ----------
    close
        Daily close prices (DatetimeIndex).
    q_start, q_end
        Inclusive window bounds (ISO date strings).

    Returns
    -------
    float
        Total return, or ``nan`` if prices are missing or invalid.
    """
    if close is None or len(close) == 0:
        return float("nan")
    s = close.copy()
    s.index = pd.to_datetime(s.index)
    s = s.sort_index()
    s = s.astype(float)
    s = s.dropna()
    if s.empty:
        return float("nan")

    start = pd.Timestamp(q_start)
    end = pd.Timestamp(q_end)
    at_or_after_start = s[s.index >= start]
    at_or_before_end = s[s.index <= end]
    if at_or_after_start.empty or at_or_before_end.empty:
        return float("nan")

    px_start = float(at_or_after_start.iloc[0])
    px_end = float(at_or_before_end.iloc[-1])
    if not np.isfinite(px_start) or not np.isfinite(px_end) or px_start == 0:
        return float("nan")
    return (px_end / px_start) - 1.0


def compute_q1_returns(
    ohlcv_by_ticker: dict[str, pd.DataFrame],
    q_start: str | None = None,
    q_end: str | None = None,
) -> pd.Series:
    """Compute Q1 total return per ticker from cached OHLCV frames (``Close`` column)."""
    qs = q_start if q_start is not None else CLASS_Q_START
    qe = q_end if q_end is not None else CLASS_Q_END
    out: dict[str, float] = {}
    for ticker, df in ohlcv_by_ticker.items():
        if df is None or df.empty or "Close" not in df.columns:
            out[ticker] = float("nan")
            continue
        out[ticker] = total_return_q1(df["Close"], qs, qe)
    return pd.Series(out, name="q1_return")


def compute_multi_period_returns(
    ohlcv_by_ticker: dict[str, pd.DataFrame],
    periods: list[tuple[str, str, str, str]],
    *,
    method: ClassificationMethod = "percentile",
    winner_percentile: float | None = None,
    loser_percentile: float | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Compute returns and assign labels across multiple classification windows.

    Parameters
    ----------
    ohlcv_by_ticker
        Mapping ticker -> OHLCV DataFrame.
    periods
        List of ``(feature_cutoff, q_start, q_end, period_label)`` tuples.
        Each defines one classification window; labels are assigned *within*
        each period independently (cross-sectional percentiles per quarter).
    method
        Classification method forwarded to :func:`assign_groups`.
    winner_percentile, loser_percentile
        Forwarded to :func:`assign_groups`.

    Returns
    -------
    returns_stacked : pd.DataFrame
        Columns: ``ticker``, ``period``, ``q1_return``.
        Index is a unique ``ticker__period`` composite key.
    labels_stacked : pd.Series
        Class labels aligned to ``returns_stacked.index``.
    """
    all_rows: list[dict[str, object]] = []

    for feature_cutoff, q_start, q_end, period_label in periods:
        returns = compute_q1_returns(ohlcv_by_ticker, q_start=q_start, q_end=q_end)
        labels = assign_groups(
            returns,
            method=method,
            winner_percentile=winner_percentile,
            loser_percentile=loser_percentile,
        )

        valid = returns.dropna().index.intersection(labels.dropna().index)
        for ticker in valid:
            all_rows.append({
                "ticker": ticker,
                "period": period_label,
                "feature_cutoff": feature_cutoff,
                "q_start": q_start,
                "q_end": q_end,
                "q1_return": float(returns.loc[ticker]),
                "label": labels.loc[ticker],
            })

    df = pd.DataFrame(all_rows)
    if df.empty:
        empty_idx = pd.Index([], name="obs_id")
        return (
            pd.DataFrame(columns=["ticker", "period", "feature_cutoff",
                                   "q_start", "q_end", "q1_return"],
                         index=empty_idx),
            pd.Series(dtype=object, name="label", index=empty_idx),
        )

    df["obs_id"] = df["ticker"] + "__" + df["period"]
    df = df.set_index("obs_id")

    labels_out = df["label"].copy()
    labels_out.name = "label"
    returns_out = df.drop(columns=["label"])

    n_periods = returns_out["period"].nunique()
    n_tickers = returns_out["ticker"].nunique()
    logger.info(
        "Multi-period stacking: %d periods × %d unique tickers → %d observations",
        n_periods, n_tickers, len(returns_out),
    )

    return returns_out, labels_out


def _silhouette_safe(X: np.ndarray, labels: np.ndarray) -> float | None:
    """Silhouette score if sklearn preconditions are met; else None."""
    labels = np.asarray(labels)
    if len(labels) < 3:
        return None
    uniq, counts = np.unique(labels, return_counts=True)
    if len(uniq) < 2 or (counts < 2).any():
        return None
    try:
        return float(silhouette_score(X, labels, metric="euclidean"))
    except ValueError:
        return None


def _group_balance_score(labels: pd.Series) -> float:
    """1 - normalized entropy of group proportions (higher = more even sizes)."""
    vc = labels.value_counts(normalize=True)
    if vc.empty:
        return 0.0
    p = vc.to_numpy(dtype=float)
    p = p[p > 0]
    if len(p) <= 1:
        return 0.0
    entropy = -float(np.sum(p * np.log(p)))
    max_ent = float(np.log(len(p)))
    if max_ent <= 0:
        return 0.0
    return 1.0 - (entropy / max_ent)


def assign_groups(
    returns: pd.Series,
    method: ClassificationMethod = "percentile",
    *,
    winner_percentile: float | None = None,
    loser_percentile: float | None = None,
    random_state: int | None = None,
) -> pd.Series:
    """Assign class labels per stock (percentile, K-Means, or auto).

    Parameters
    ----------
    returns
        Total returns indexed by ticker (NaNs kept as NaN in output).
    method
        ``percentile`` — top ``1 - winner_percentile`` fraction as Winners, bottom
        ``loser_percentile`` as Losers, rest Steady (via quantile thresholds).
        ``kmeans`` — 3 clusters on 1D returns; mapped to Losers / Steady / Winners by
        ascending cluster mean return.
        ``auto`` — pick percentile vs kmeans by higher silhouette on valid subsample;
        tie-break by group balance score.

    winner_percentile, loser_percentile
        Defaults from :mod:`config` (0.75 / 0.25). Must satisfy ``loser < winner``.

    random_state
        K-Means seed; defaults to :data:`config.RANDOM_SEED`.
    """
    wp = WINNER_PERCENTILE if winner_percentile is None else float(winner_percentile)
    lp = LOSER_PERCENTILE if loser_percentile is None else float(loser_percentile)
    rs = RANDOM_SEED if random_state is None else int(random_state)

    if lp >= wp:
        raise ValueError(f"loser_percentile ({lp}) must be < winner_percentile ({wp})")

    r_clean = returns.dropna()
    if r_clean.empty:
        return pd.Series(index=returns.index, dtype=object)

    if method == "percentile":
        labels = _assign_percentile_thresholds(returns, wp, lp)
    elif method == "kmeans":
        labels = _assign_kmeans(returns, random_state=rs)
    elif method == "auto":
        p_lab = _assign_percentile_thresholds(returns, wp, lp)
        k_lab = _assign_kmeans(returns, random_state=rs)
        p_score = _method_score(returns, p_lab)
        k_score = _method_score(returns, k_lab)
        if k_score > p_score:
            labels = k_lab
            logger.info("classification: auto selected kmeans (scores k=%.4f p=%.4f)", k_score, p_score)
        elif k_score < p_score:
            labels = p_lab
            logger.info("classification: auto selected percentile (scores p=%.4f k=%.4f)", p_score, k_score)
        else:
            bal_p = _group_balance_score(p_lab.dropna())
            bal_k = _group_balance_score(k_lab.dropna())
            labels = k_lab if bal_k > bal_p else p_lab
            logger.info("classification: auto tie-break by balance (p=%.3f k=%.3f)", bal_p, bal_k)
    else:
        raise ValueError(f"Unknown method: {method!r}; use 'percentile', 'kmeans', or 'auto'")

    return labels.reindex(returns.index)


def _assign_percentile_thresholds(
    returns: pd.Series,
    winner_q: float,
    loser_q: float,
) -> pd.Series:
    """Quantile thresholds: >= winner cutoff -> Winners, <= loser -> Losers, else Steady."""
    labels = pd.Series(index=returns.index, dtype=object)
    r = returns.dropna()
    if r.empty:
        return labels

    hi = r.quantile(winner_q)
    lo = r.quantile(loser_q)
    for idx in r.index:
        v = float(r.loc[idx])
        if v >= hi:
            labels.loc[idx] = CLASS_WINNERS
        elif v <= lo:
            labels.loc[idx] = CLASS_LOSERS
        else:
            labels.loc[idx] = CLASS_STEADY
    return labels


def _assign_kmeans(returns: pd.Series, *, random_state: int) -> pd.Series:
    labels = pd.Series(index=returns.index, dtype=object)
    r = returns.dropna()
    if r.empty:
        return labels
    if len(r) < 3:
        for idx in r.index:
            labels.loc[idx] = CLASS_STEADY
        return labels

    X = r.values.reshape(-1, 1)
    km = KMeans(n_clusters=3, random_state=random_state, n_init=10)
    clusters = km.fit_predict(X)
    centers = km.cluster_centers_.flatten()
    order = np.argsort(centers)  # low -> high mean return
    cluster_to_class = {
        int(order[0]): CLASS_LOSERS,
        int(order[1]): CLASS_STEADY,
        int(order[2]): CLASS_WINNERS,
    }
    for i, idx in enumerate(r.index):
        labels.loc[idx] = cluster_to_class[int(clusters[i])]
    return labels


def _method_score(returns: pd.Series, labels: pd.Series) -> float:
    """Combined score: silhouette (if defined) + small balance term."""
    aligned = pd.DataFrame({"r": returns, "lab": labels}).dropna()
    if len(aligned) < 3:
        return _group_balance_score(labels.dropna())
    X = aligned["r"].values.reshape(-1, 1)
    lab_map = {c: i for i, c in enumerate(sorted(aligned["lab"].unique(), key=str))}
    y = aligned["lab"].map(lab_map).to_numpy()
    sil = _silhouette_safe(X, y)
    bal = _group_balance_score(aligned["lab"])
    if sil is None:
        return bal
    return sil + 0.1 * bal


def classification_quality(
    returns: pd.Series,
    labels: pd.Series,
) -> dict[str, Any]:
    """Silhouette score (1D returns), per-class counts, and balance score."""
    df = pd.DataFrame({"r": returns, "lab": labels}).dropna()
    out: dict[str, Any] = {
        "n": int(len(df)),
        "counts": df["lab"].value_counts().to_dict(),
        "balance_score": _group_balance_score(df["lab"]),
        "silhouette": None,
    }
    if len(df) >= 3:
        X = df["r"].values.reshape(-1, 1)
        lab_map = {c: i for i, c in enumerate(sorted(df["lab"].unique(), key=str))}
        y = df["lab"].map(lab_map).to_numpy()
        sil = _silhouette_safe(X, y)
        if sil is not None:
            out["silhouette"] = sil
    return out


def optimize_percentile_cutoffs(
    returns: pd.Series,
    *,
    winner_quantiles: tuple[float, ...] = (0.65, 0.70, 0.75, 0.80, 0.85),
    loser_quantiles: tuple[float, ...] = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35),
    random_state: int | None = None,
) -> tuple[float, float, dict[str, Any]]:
    """Grid-search winner/loser quantile pairs; maximize silhouette + balance.

    Returns
    -------
    best_winner_q, best_loser_q, diagnostics
        Best thresholds and a dict with ``grid_results`` (list of scored tuples).
    """
    rs = RANDOM_SEED if random_state is None else int(random_state)
    r = returns.dropna()
    if len(r) < 3:
        wp, lp = WINNER_PERCENTILE, LOSER_PERCENTILE
        return wp, lp, {"grid_results": [], "note": "insufficient data"}

    best_score = -np.inf
    best_pair = (WINNER_PERCENTILE, LOSER_PERCENTILE)
    grid_results: list[tuple[float, float, float, dict[str, Any]]] = []

    for wq, lq in itertools.product(winner_quantiles, loser_quantiles):
        if lq >= wq:
            continue
        lab = _assign_percentile_thresholds(returns, wq, lq)
        q = classification_quality(returns, lab)
        sil = q.get("silhouette")
        bal = float(q.get("balance_score", 0.0))
        if sil is None:
            score = bal
        else:
            score = float(sil) + 0.15 * bal
        grid_results.append((wq, lq, score, q))
        if score > best_score:
            best_score = score
            best_pair = (wq, lq)

    diag = {
        "grid_results": grid_results,
        "best_score": best_score,
        "random_state_kmeans_unused": rs,
    }
    return best_pair[0], best_pair[1], diag


def plot_return_distribution(
    returns: pd.Series,
    *,
    title: str = "Q1 2024 return distribution",
    ax: Any | None = None,
) -> Any:
    """Histogram of total returns (dropna)."""
    import matplotlib.pyplot as plt

    r = returns.dropna()
    created_fig = ax is None
    if created_fig:
        _, ax = plt.subplots(figsize=(8, 4))
    ax.hist(r, bins=min(40, max(10, len(r) // 5)), color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(r.median(), color="black", linestyle="--", linewidth=1, label="median")
    ax.set_title(title)
    ax.set_xlabel("Total return")
    ax.set_ylabel("Count")
    ax.legend()
    if created_fig:
        plt.tight_layout()
    return ax


def plot_group_characteristics(
    returns: pd.Series,
    labels: pd.Series,
    *,
    title: str = "Returns by group",
    ax: Any | None = None,
) -> Any:
    """Boxplot of returns per class (Winners / Steady / Losers order)."""
    import matplotlib.pyplot as plt

    df = pd.DataFrame({"r": returns, "lab": labels}).dropna()
    created_fig = ax is None
    if created_fig:
        _, ax = plt.subplots(figsize=(8, 4))

    order = [c for c in CLASS_ORDER if c in set(df["lab"].unique())]
    data = [df.loc[df["lab"] == g, "r"].values for g in order]
    ax.boxplot(data, showmeans=True)
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels(order)
    ax.set_title(title)
    ax.set_ylabel("Total return")
    ax.axhline(0.0, color="gray", linewidth=0.8)
    if created_fig:
        plt.tight_layout()
    return ax
