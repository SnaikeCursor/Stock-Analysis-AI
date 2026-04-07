"""Leakage diagnostics for the classification pipeline.

Four systematic tests validate that model performance is not inflated by
data leakage (temporal, imputation, or correlation-filter):

1. **Embargo test** — walk-forward with a 1-quarter gap between training
   and evaluation.  If performance drops sharply vs. no gap, adjacent
   periods share leaked information.

2. **Shuffled-labels test** — permute Y-labels randomly (break the true
   feature→return link).  IC must be near zero; otherwise the model
   exploits structural artefacts rather than real signal.

3. **Retrodiction test** — train on 2020-2024, predict 2012-2019 (backward).
   Compare IC to the forward direction.  With clean data, backward IC is
   similar or worse; if it is *better*, future data leaked into training.

4. **Feature-future-correlation test** — rank correlation of each feature
   with the *next* quarter's return vs. the *previous* quarter's return.
   A systematic positive gap flags look-ahead contamination in features.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)

__all__ = [
    "LeakageTestResult",
    "embargo_test",
    "feature_future_correlation_test",
    "print_leakage_report",
    "retrodiction_test",
    "run_all_leakage_tests",
    "shuffled_labels_test",
]


@dataclass
class LeakageTestResult:
    """Outcome of a single leakage diagnostic."""

    test_name: str
    passed: bool
    metric_name: str
    metric_value: float
    baseline_value: float | None
    interpretation: str
    details: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _lazy_imports():
    """Avoid circular imports by loading src modules on demand."""
    from src.classifier import assign_groups, compute_q1_returns
    from src.features import build_feature_matrix, drop_correlated_features

    return build_feature_matrix, compute_q1_returns, assign_groups, drop_correlated_features


def _build_stacked_dataset(
    ohlcv: dict[str, pd.DataFrame],
    fundamentals: dict[str, dict],
    periods: list[tuple[str, str, str, str]],
    *,
    drop_correlated: bool = True,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, np.ndarray]:
    """Stack features, labels, and raw returns from multiple periods.

    Returns ``(X_stacked, y_stacked, returns_stacked, period_labels)``.
    Row indices are ``{ticker}__{period_label}`` to stay unique.
    """
    build_feature_matrix, compute_q1_returns, assign_groups, drop_corr = _lazy_imports()

    X_parts: list[pd.DataFrame] = []
    y_parts: list[pd.Series] = []
    r_parts: list[pd.Series] = []
    pl_parts: list[np.ndarray] = []

    for fc, q_start, q_end, plabel in periods:
        X = build_feature_matrix(ohlcv, fc, fundamentals)
        if drop_correlated:
            X, _ = drop_corr(X)
        returns = compute_q1_returns(ohlcv, q_start=q_start, q_end=q_end)
        labels = assign_groups(returns)

        common = X.index.intersection(labels.dropna().index).intersection(
            returns.dropna().index,
        )
        if len(common) < 10:
            logger.warning("Stacked dataset: skipping %s (%d samples)", plabel, len(common))
            continue

        X_al = X.loc[common].copy()
        y_al = labels.loc[common].copy()
        r_al = returns.loc[common].copy()

        uid = pd.Index([f"{t}__{plabel}" for t in X_al.index], name="obs_id")
        X_al.index = uid
        y_al.index = uid
        r_al.index = uid

        X_parts.append(X_al)
        y_parts.append(y_al)
        r_parts.append(r_al)
        pl_parts.append(np.full(len(X_al), plabel))

    if not X_parts:
        raise ValueError("No valid periods for stacked dataset")

    return (
        pd.concat(X_parts),
        pd.concat(y_parts),
        pd.concat(r_parts),
        np.concatenate(pl_parts),
    )


def _information_coefficient(
    predicted_proba_winners: np.ndarray,
    actual_returns: np.ndarray,
) -> float:
    """Spearman rank-correlation between P(Winner) and actual return (IC)."""
    mask = np.isfinite(predicted_proba_winners) & np.isfinite(actual_returns)
    if mask.sum() < 5:
        return float("nan")
    rho, _ = spearmanr(predicted_proba_winners[mask], actual_returns[mask])
    return float(rho)


def _impute_median_xy(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Median-impute training set; apply same medians to test set."""
    medians = X_train.median()
    return X_train.fillna(medians), X_test.fillna(medians)


def _unique_ordered(arr: np.ndarray) -> list[str]:
    """Deduplicated list preserving first-occurrence order."""
    seen: list[str] = []
    for v in arr:
        if v not in seen:
            seen.append(v)
    return seen


# ---------------------------------------------------------------------------
# Test 1: Embargo
# ---------------------------------------------------------------------------


def embargo_test(
    ohlcv: dict[str, pd.DataFrame],
    fundamentals: dict[str, dict],
    periods: list[tuple[str, str, str, str]] | None = None,
    *,
    embargo_quarters: int = 1,
    random_seed: int = 42,
) -> LeakageTestResult:
    """Walk-forward IC with vs. without an embargo gap.

    For each fold, training ends at period *k*; testing occurs at period
    *k + 1* (standard) or *k + 1 + embargo* (with gap).  The IC drop
    should be small if there is no adjacent-period leakage.
    """
    if periods is None:
        from config import CLASSIFICATION_PERIODS
        periods = list(CLASSIFICATION_PERIODS)

    X, y, rets, pl = _build_stacked_dataset(ohlcv, fundamentals, periods)
    unique_p = _unique_ordered(pl)
    n_p = len(unique_p)

    min_train = max(4, n_p // 4)

    def _wf_ic(embargo: int) -> list[float]:
        ics: list[float] = []
        for k in range(min_train, n_p - 1 - embargo):
            train_periods = set(unique_p[: k + 1])
            test_period = unique_p[k + 1 + embargo]

            train_mask = np.isin(pl, list(train_periods))
            test_mask = pl == test_period

            if train_mask.sum() < 20 or test_mask.sum() < 5:
                continue

            X_tr, X_te = _impute_median_xy(X[train_mask], X[test_mask])
            y_tr = y.values[train_mask]

            clf = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=random_seed,
                n_jobs=-1,
                class_weight="balanced",
            )
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y_enc = le.fit_transform(y_tr)
            clf.fit(X_tr.values, y_enc)

            proba = clf.predict_proba(X_te.values)
            cls = list(le.classes_)
            w_idx = cls.index("Winners") if "Winners" in cls else 0
            pw = proba[:, w_idx]
            ic = _information_coefficient(pw, rets.values[test_mask])
            ics.append(ic)
        return ics

    ics_std = _wf_ic(0)
    ics_emb = _wf_ic(embargo_quarters)

    mean_std = float(np.nanmean(ics_std)) if ics_std else float("nan")
    mean_emb = float(np.nanmean(ics_emb)) if ics_emb else float("nan")
    delta = mean_std - mean_emb if np.isfinite(mean_std) and np.isfinite(mean_emb) else float("nan")

    passed = np.isfinite(delta) and delta < 0.10

    return LeakageTestResult(
        test_name="embargo",
        passed=passed,
        metric_name="IC_drop (standard − embargo)",
        metric_value=delta,
        baseline_value=mean_std,
        interpretation=(
            f"Standard walk-forward IC={mean_std:.4f}, "
            f"embargo({embargo_quarters}Q) IC={mean_emb:.4f}, "
            f"delta={delta:.4f}.  "
            + ("PASS — small drop." if passed else "WARN — large drop suggests adjacent-period leakage.")
        ),
        details={
            "ic_standard_per_fold": ics_std,
            "ic_embargo_per_fold": ics_emb,
            "mean_ic_standard": mean_std,
            "mean_ic_embargo": mean_emb,
            "embargo_quarters": embargo_quarters,
            "n_folds_std": len(ics_std),
            "n_folds_emb": len(ics_emb),
        },
    )


# ---------------------------------------------------------------------------
# Test 2: Shuffled labels
# ---------------------------------------------------------------------------


def shuffled_labels_test(
    ohlcv: dict[str, pd.DataFrame],
    fundamentals: dict[str, dict],
    periods: list[tuple[str, str, str, str]] | None = None,
    *,
    n_iterations: int = 10,
    random_seed: int = 42,
) -> LeakageTestResult:
    """Train with permuted Y-labels; IC should collapse to ~0.

    The labels are shuffled *within each period* (preserving period
    structure and class balance) — only the ticker→label assignment is
    randomised.  A mean |IC| > 0.05 across iterations suggests the model
    captures structural artefacts rather than real signal.
    """
    if periods is None:
        from config import CLASSIFICATION_PERIODS
        periods = list(CLASSIFICATION_PERIODS)

    X, y, rets, pl = _build_stacked_dataset(ohlcv, fundamentals, periods)
    unique_p = _unique_ordered(pl)
    n_p = len(unique_p)

    split_idx = int(n_p * 0.8)
    train_periods = set(unique_p[:split_idx])
    test_periods = set(unique_p[split_idx:])

    train_mask = np.isin(pl, list(train_periods))
    test_mask = np.isin(pl, list(test_periods))

    if train_mask.sum() < 20 or test_mask.sum() < 5:
        return LeakageTestResult(
            test_name="shuffled_labels",
            passed=True,
            metric_name="mean_abs_IC_shuffled",
            metric_value=float("nan"),
            baseline_value=float("nan"),
            interpretation="Insufficient data for shuffled-labels test.",
        )

    X_tr, X_te = _impute_median_xy(X[train_mask], X[test_mask])
    y_tr_real = y.values[train_mask]

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.fit(y_tr_real)
    cls = list(le.classes_)
    w_idx = cls.index("Winners") if "Winners" in cls else 0

    clf_real = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=random_seed,
        n_jobs=-1,
        class_weight="balanced",
    )
    clf_real.fit(X_tr.values, le.transform(y_tr_real))
    proba_real = clf_real.predict_proba(X_te.values)
    ic_real = _information_coefficient(proba_real[:, w_idx], rets.values[test_mask])

    rng = np.random.default_rng(random_seed)
    shuffled_ics: list[float] = []

    pl_train = pl[train_mask]
    for _ in range(n_iterations):
        y_shuf = y_tr_real.copy()
        for p in set(pl_train):
            pmask = pl_train == p
            subset = y_shuf[pmask]
            rng.shuffle(subset)
            y_shuf[pmask] = subset

        clf_s = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=random_seed,
            n_jobs=-1,
            class_weight="balanced",
        )
        clf_s.fit(X_tr.values, le.transform(y_shuf))
        proba_s = clf_s.predict_proba(X_te.values)
        ic_s = _information_coefficient(proba_s[:, w_idx], rets.values[test_mask])
        shuffled_ics.append(ic_s)

    mean_abs_ic = float(np.nanmean(np.abs(shuffled_ics)))
    passed = mean_abs_ic < 0.05

    return LeakageTestResult(
        test_name="shuffled_labels",
        passed=passed,
        metric_name="mean_abs_IC_shuffled",
        metric_value=mean_abs_ic,
        baseline_value=ic_real,
        interpretation=(
            f"Real IC={ic_real:.4f}, shuffled mean|IC|={mean_abs_ic:.4f} "
            f"(n={n_iterations}).  "
            + ("PASS — shuffled IC near zero." if passed else "WARN — shuffled IC too high, possible leakage.")
        ),
        details={
            "ic_real": ic_real,
            "shuffled_ics": shuffled_ics,
            "mean_abs_ic_shuffled": mean_abs_ic,
            "n_iterations": n_iterations,
        },
    )


# ---------------------------------------------------------------------------
# Test 3: Retrodiction
# ---------------------------------------------------------------------------


def retrodiction_test(
    ohlcv: dict[str, pd.DataFrame],
    fundamentals: dict[str, dict],
    periods: list[tuple[str, str, str, str]] | None = None,
    *,
    forward_train_end_year: int = 2019,
    backward_train_start_year: int = 2020,
    random_seed: int = 42,
) -> LeakageTestResult:
    """Train on recent data and predict the past, then compare with forward prediction.

    * **Forward**: train on 2012-2019, predict 2020-2024.
    * **Backward**: train on 2020-2024, predict 2012-2019.

    Forward IC should be >= backward IC.  If backward is *better*, future
    information leaked into training features.
    """
    if periods is None:
        from config import CLASSIFICATION_PERIODS
        periods = list(CLASSIFICATION_PERIODS)

    X, y, rets, pl = _build_stacked_dataset(ohlcv, fundamentals, periods)

    def _year_from_label(label: str) -> int:
        return int(label.split("-")[1])

    years = np.array([_year_from_label(p) for p in pl])

    early_mask = years <= forward_train_end_year
    late_mask = years >= backward_train_start_year

    if early_mask.sum() < 20 or late_mask.sum() < 20:
        return LeakageTestResult(
            test_name="retrodiction",
            passed=True,
            metric_name="IC_forward − IC_backward",
            metric_value=float("nan"),
            baseline_value=float("nan"),
            interpretation="Insufficient data for retrodiction test.",
        )

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.fit(y.values)
    cls = list(le.classes_)
    w_idx = cls.index("Winners") if "Winners" in cls else 0

    def _train_predict_ic(
        train_mask: np.ndarray,
        test_mask: np.ndarray,
    ) -> float:
        X_tr, X_te = _impute_median_xy(X[train_mask], X[test_mask])
        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=random_seed,
            n_jobs=-1,
            class_weight="balanced",
        )
        clf.fit(X_tr.values, le.transform(y.values[train_mask]))
        proba = clf.predict_proba(X_te.values)
        return _information_coefficient(proba[:, w_idx], rets.values[test_mask])

    ic_forward = _train_predict_ic(early_mask, late_mask)
    ic_backward = _train_predict_ic(late_mask, early_mask)
    delta = ic_forward - ic_backward

    passed = np.isfinite(delta) and delta >= -0.05

    return LeakageTestResult(
        test_name="retrodiction",
        passed=passed,
        metric_name="IC_forward − IC_backward",
        metric_value=delta,
        baseline_value=ic_forward,
        interpretation=(
            f"Forward IC={ic_forward:.4f} (train ≤{forward_train_end_year}, "
            f"test ≥{backward_train_start_year}), "
            f"backward IC={ic_backward:.4f}, "
            f"delta={delta:.4f}.  "
            + (
                "PASS — forward >= backward."
                if passed
                else "WARN — backward IC higher than forward, potential future leakage."
            )
        ),
        details={
            "ic_forward": ic_forward,
            "ic_backward": ic_backward,
            "n_early": int(early_mask.sum()),
            "n_late": int(late_mask.sum()),
            "forward_train_end_year": forward_train_end_year,
            "backward_train_start_year": backward_train_start_year,
        },
    )


# ---------------------------------------------------------------------------
# Test 4: Feature–future correlation
# ---------------------------------------------------------------------------


def feature_future_correlation_test(
    ohlcv: dict[str, pd.DataFrame],
    fundamentals: dict[str, dict],
    periods: list[tuple[str, str, str, str]] | None = None,
) -> LeakageTestResult:
    """Per-feature rank-correlation with next-quarter return vs. previous-quarter return.

    For each feature, computes Spearman rho(feature, forward_return) and
    rho(feature, backward_return) across the stacked panel.  A systematic
    positive gap (features correlate *more* with the future than the past)
    flags look-ahead contamination.
    """
    if periods is None:
        from config import CLASSIFICATION_PERIODS
        periods = list(CLASSIFICATION_PERIODS)

    build_fm, compute_q1_returns, assign_groups, drop_corr = _lazy_imports()

    unique_labels = [p[3] for p in periods]
    label_to_period = {p[3]: p for p in periods}

    feature_rows: list[dict[str, Any]] = []

    for i, plabel in enumerate(unique_labels):
        fc, q_start, q_end, _ = label_to_period[plabel]

        X = build_fm(ohlcv, fc, fundamentals)
        fwd_ret = compute_q1_returns(ohlcv, q_start=q_start, q_end=q_end)

        if i > 0:
            prev_label = unique_labels[i - 1]
            prev_p = label_to_period[prev_label]
            bwd_ret = compute_q1_returns(ohlcv, q_start=prev_p[1], q_end=prev_p[2])
        else:
            bwd_ret = pd.Series(dtype=float)

        common_fwd = X.index.intersection(fwd_ret.dropna().index)
        common_bwd = X.index.intersection(bwd_ret.dropna().index) if not bwd_ret.empty else pd.Index([])

        for col in X.columns:
            row: dict[str, Any] = {"feature": col, "period": plabel}
            vals_fwd = X.loc[common_fwd, col].values
            ret_fwd = fwd_ret.loc[common_fwd].values
            mask_f = np.isfinite(vals_fwd) & np.isfinite(ret_fwd)
            if mask_f.sum() >= 10:
                rho_f, _ = spearmanr(vals_fwd[mask_f], ret_fwd[mask_f])
                row["rho_forward"] = float(rho_f)
            else:
                row["rho_forward"] = float("nan")

            if len(common_bwd) >= 10:
                vals_bwd = X.loc[common_bwd, col].values
                ret_bwd = bwd_ret.loc[common_bwd].values
                mask_b = np.isfinite(vals_bwd) & np.isfinite(ret_bwd)
                if mask_b.sum() >= 10:
                    rho_b, _ = spearmanr(vals_bwd[mask_b], ret_bwd[mask_b])
                    row["rho_backward"] = float(rho_b)
                else:
                    row["rho_backward"] = float("nan")
            else:
                row["rho_backward"] = float("nan")

            feature_rows.append(row)

    df = pd.DataFrame(feature_rows)
    if df.empty:
        return LeakageTestResult(
            test_name="feature_future_correlation",
            passed=True,
            metric_name="mean_rho_gap",
            metric_value=float("nan"),
            baseline_value=float("nan"),
            interpretation="No feature data available.",
        )

    agg = df.groupby("feature")[["rho_forward", "rho_backward"]].mean()
    agg["gap"] = agg["rho_forward"].abs() - agg["rho_backward"].abs()
    mean_gap = float(agg["gap"].mean())

    suspicious = agg[agg["gap"] > 0.05].sort_values("gap", ascending=False)
    passed = mean_gap < 0.03

    return LeakageTestResult(
        test_name="feature_future_correlation",
        passed=passed,
        metric_name="mean_rho_gap (|fwd| − |bwd|)",
        metric_value=mean_gap,
        baseline_value=None,
        interpretation=(
            f"Mean |rho_forward|−|rho_backward| = {mean_gap:.4f} across "
            f"{len(agg)} features.  "
            + (
                "PASS — no systematic forward bias."
                if passed
                else f"WARN — {len(suspicious)} features correlate more with future returns."
            )
        ),
        details={
            "per_feature": agg.to_dict(orient="index"),
            "suspicious_features": list(suspicious.index) if not suspicious.empty else [],
            "n_features": len(agg),
            "n_periods": len(unique_labels),
        },
    )


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------


def run_all_leakage_tests(
    ohlcv: dict[str, pd.DataFrame],
    fundamentals: dict[str, dict],
    periods: list[tuple[str, str, str, str]] | None = None,
    *,
    random_seed: int = 42,
) -> list[LeakageTestResult]:
    """Execute all four leakage diagnostics and return results."""
    results: list[LeakageTestResult] = []

    logger.info("Leakage diagnostics: running embargo test…")
    results.append(embargo_test(ohlcv, fundamentals, periods, random_seed=random_seed))

    logger.info("Leakage diagnostics: running shuffled-labels test…")
    results.append(shuffled_labels_test(ohlcv, fundamentals, periods, random_seed=random_seed))

    logger.info("Leakage diagnostics: running retrodiction test…")
    results.append(retrodiction_test(ohlcv, fundamentals, periods, random_seed=random_seed))

    logger.info("Leakage diagnostics: running feature-future correlation test…")
    results.append(feature_future_correlation_test(ohlcv, fundamentals, periods))

    n_pass = sum(1 for r in results if r.passed)
    logger.info("Leakage diagnostics: %d/%d passed.", n_pass, len(results))

    return results


def print_leakage_report(results: list[LeakageTestResult]) -> str:
    """Format leakage test results for console output."""
    lines: list[str] = []
    sep = "=" * 72
    lines.append(sep)
    lines.append("  LEAKAGE DIAGNOSTICS REPORT")
    lines.append(sep)

    for r in results:
        status = "✅ PASS" if r.passed else "⚠️  WARN"
        lines.append(f"\n--- {r.test_name.upper()} [{status}] ---")
        lines.append(f"  {r.metric_name} = {r.metric_value:.4f}"
                     if np.isfinite(r.metric_value) else f"  {r.metric_name} = N/A")
        if r.baseline_value is not None and np.isfinite(r.baseline_value):
            lines.append(f"  baseline = {r.baseline_value:.4f}")
        lines.append(f"  {r.interpretation}")

    n_pass = sum(1 for r in results if r.passed)
    lines.append(f"\n--- Overall: {n_pass}/{len(results)} passed ---")

    text = "\n".join(lines)
    print(text)
    return text
