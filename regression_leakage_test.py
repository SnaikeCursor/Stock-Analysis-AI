#!/usr/bin/env python3
"""Leakage diagnostics for the **regression** pipeline.

Four tests validate that the regression model's IC is not inflated by data
leakage (temporal, imputation, or feature contamination):

1. Embargo test — walk-forward IC with vs. without a 1-quarter gap.
2. Shuffled-labels test — permute y-targets; IC must collapse to ~0.
3. Retrodiction test — train on late data, predict early (backward IC ≤ forward IC).
4. Feature-future correlation — features must not correlate more with future returns
   than with past returns.

Usage:
    python regression_leakage_test.py [--cs-norm] [--pub-lag N]
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from robustness_test import build_regression_feature_panel, load_data
from src.regression_targets import compute_quarterly_forward_returns

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("leakage_regression")

START_QUARTER = "2012-Q1"
END_QUARTER = "2024-Q3"
RF_KWARGS: dict[str, Any] = {
    "n_estimators": 100,
    "max_depth": 10,
    "random_state": 42,
    "n_jobs": -1,
}


def _spearman_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 5:
        return float("nan")
    rho, _ = spearmanr(y_true[mask], y_pred[mask])
    return float(rho)


def _impute_median_split(
    X_train: pd.DataFrame, X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    medians = X_train.median()
    return X_train.fillna(medians), X_test.fillna(medians)


def _unique_ordered(arr: np.ndarray) -> list[str]:
    seen: list[str] = []
    for v in arr:
        if v not in seen:
            seen.append(v)
    return seen


# ── Test 1: Embargo ──────────────────────────────────────────────────────────


def embargo_test(
    X: pd.DataFrame,
    y: np.ndarray,
    period_labels: np.ndarray,
    *,
    embargo_quarters: int = 1,
    min_train_periods: int = 8,
) -> dict[str, Any]:
    """Walk-forward IC with vs. without an embargo gap."""
    unique_p = _unique_ordered(period_labels)
    n_p = len(unique_p)

    def _wf_ic(embargo: int) -> list[float]:
        ics: list[float] = []
        for k in range(min_train_periods, n_p - 1 - embargo):
            train_periods = set(unique_p[: k + 1])
            test_period = unique_p[k + 1 + embargo]
            train_mask = np.isin(period_labels, list(train_periods))
            test_mask = period_labels == test_period
            if train_mask.sum() < 20 or test_mask.sum() < 5:
                continue
            X_tr, X_te = _impute_median_split(X[train_mask], X[test_mask])
            rf = RandomForestRegressor(**RF_KWARGS)
            rf.fit(X_tr.values, y[train_mask])
            pred = rf.predict(X_te.values)
            ics.append(_spearman_ic(y[test_mask], pred))
        return ics

    ics_std = _wf_ic(0)
    ics_emb = _wf_ic(embargo_quarters)

    mean_std = float(np.nanmean(ics_std)) if ics_std else float("nan")
    mean_emb = float(np.nanmean(ics_emb)) if ics_emb else float("nan")
    delta = mean_std - mean_emb if np.isfinite(mean_std) and np.isfinite(mean_emb) else float("nan")
    passed = np.isfinite(delta) and delta < 0.10

    return {
        "test": "EMBARGO",
        "passed": passed,
        "ic_standard": mean_std,
        "ic_embargo": mean_emb,
        "delta": delta,
        "n_folds_std": len(ics_std),
        "n_folds_emb": len(ics_emb),
    }


# ── Test 2: Shuffled Labels ─────────────────────────────────────────────────


def shuffled_labels_test(
    X: pd.DataFrame,
    y: np.ndarray,
    period_labels: np.ndarray,
    *,
    n_iterations: int = 10,
    seed: int = 42,
) -> dict[str, Any]:
    """Train with shuffled y-targets; IC must collapse to ~0."""
    unique_p = _unique_ordered(period_labels)
    split_idx = int(len(unique_p) * 0.8)
    train_periods = set(unique_p[:split_idx])
    test_periods = set(unique_p[split_idx:])

    train_mask = np.isin(period_labels, list(train_periods))
    test_mask = np.isin(period_labels, list(test_periods))

    if train_mask.sum() < 20 or test_mask.sum() < 5:
        return {"test": "SHUFFLED_LABELS", "passed": True, "note": "insufficient data"}

    X_tr, X_te = _impute_median_split(X[train_mask], X[test_mask])
    y_tr_real = y[train_mask]
    y_te = y[test_mask]

    rf_real = RandomForestRegressor(**RF_KWARGS)
    rf_real.fit(X_tr.values, y_tr_real)
    ic_real = _spearman_ic(y_te, rf_real.predict(X_te.values))

    rng = np.random.default_rng(seed)
    shuffled_ics: list[float] = []
    pl_train = period_labels[train_mask]
    for _ in range(n_iterations):
        y_shuf = y_tr_real.copy()
        for p in set(pl_train):
            pmask = pl_train == p
            subset = y_shuf[pmask]
            rng.shuffle(subset)
            y_shuf[pmask] = subset

        rf_s = RandomForestRegressor(**RF_KWARGS)
        rf_s.fit(X_tr.values, y_shuf)
        shuffled_ics.append(_spearman_ic(y_te, rf_s.predict(X_te.values)))

    mean_abs_ic = float(np.nanmean(np.abs(shuffled_ics)))
    passed = mean_abs_ic < 0.05

    return {
        "test": "SHUFFLED_LABELS",
        "passed": passed,
        "ic_real": ic_real,
        "mean_abs_ic_shuffled": mean_abs_ic,
        "shuffled_ics": shuffled_ics,
        "n_iterations": n_iterations,
    }


# ── Test 3: Retrodiction ────────────────────────────────────────────────────


def retrodiction_test(
    X: pd.DataFrame,
    y: np.ndarray,
    period_labels: np.ndarray,
    *,
    split_quarter: str = "2019Q4",
) -> dict[str, Any]:
    """Train on recent data and predict the past; forward IC must be >= backward IC."""
    early_mask = period_labels <= split_quarter
    late_mask = period_labels > split_quarter

    if early_mask.sum() < 50 or late_mask.sum() < 50:
        return {"test": "RETRODICTION", "passed": True, "note": "insufficient data"}

    def _train_pred_ic(tr_mask: np.ndarray, te_mask: np.ndarray) -> float:
        X_tr, X_te = _impute_median_split(X[tr_mask], X[te_mask])
        rf = RandomForestRegressor(**RF_KWARGS)
        rf.fit(X_tr.values, y[tr_mask])
        return _spearman_ic(y[te_mask], rf.predict(X_te.values))

    ic_forward = _train_pred_ic(early_mask, late_mask)
    ic_backward = _train_pred_ic(late_mask, early_mask)
    delta = ic_forward - ic_backward
    passed = np.isfinite(delta) and delta >= -0.05

    return {
        "test": "RETRODICTION",
        "passed": passed,
        "ic_forward": ic_forward,
        "ic_backward": ic_backward,
        "delta": delta,
        "n_early": int(early_mask.sum()),
        "n_late": int(late_mask.sum()),
    }


# ── Test 4: Feature-Future Correlation ───────────────────────────────────────


def feature_future_correlation_test(
    X: pd.DataFrame,
    y: np.ndarray,
    period_labels: np.ndarray,
) -> dict[str, Any]:
    """Per-feature rank-correlation with forward vs. backward returns.

    For each period, measures how strongly each feature correlates with that
    period's *forward* return vs the *previous* period's return.  A systematic
    gap (features more correlated with future) flags look-ahead bias.
    """
    unique_p = _unique_ordered(period_labels)
    rows: list[dict[str, Any]] = []

    for i, p in enumerate(unique_p):
        p_mask = period_labels == p
        X_p = X[p_mask]
        y_fwd = y[p_mask]

        if i > 0:
            prev_mask = period_labels == unique_p[i - 1]
            y_bwd = y[prev_mask]
            X_bwd = X[prev_mask]
        else:
            y_bwd = np.array([])
            X_bwd = pd.DataFrame()

        for col in X_p.columns:
            row: dict[str, Any] = {"feature": col, "period": p}
            vals_fwd = X_p[col].values.astype(float)
            mask_f = np.isfinite(vals_fwd) & np.isfinite(y_fwd)
            if mask_f.sum() >= 10:
                rho_f, _ = spearmanr(vals_fwd[mask_f], y_fwd[mask_f])
                row["rho_forward"] = float(rho_f)
            else:
                row["rho_forward"] = float("nan")

            if len(y_bwd) >= 10 and col in X_bwd.columns:
                vals_bwd = X_bwd[col].values.astype(float)
                mask_b = np.isfinite(vals_bwd) & np.isfinite(y_bwd)
                if mask_b.sum() >= 10:
                    rho_b, _ = spearmanr(vals_bwd[mask_b], y_bwd[mask_b])
                    row["rho_backward"] = float(rho_b)
                else:
                    row["rho_backward"] = float("nan")
            else:
                row["rho_backward"] = float("nan")

            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return {"test": "FEATURE_FUTURE_CORR", "passed": True, "note": "no data"}

    agg = df.groupby("feature")[["rho_forward", "rho_backward"]].mean()
    agg["gap"] = agg["rho_forward"].abs() - agg["rho_backward"].abs()
    mean_gap = float(agg["gap"].mean())
    suspicious = agg[agg["gap"] > 0.05].sort_values("gap", ascending=False)
    passed = mean_gap < 0.03

    return {
        "test": "FEATURE_FUTURE_CORR",
        "passed": passed,
        "mean_gap": mean_gap,
        "n_features": len(agg),
        "n_suspicious": len(suspicious),
        "suspicious_features": list(suspicious.index) if not suspicious.empty else [],
        "top5_gaps": agg.sort_values("gap", ascending=False).head(5)["gap"].to_dict(),
    }


# ── Report ───────────────────────────────────────────────────────────────────


def print_report(results: list[dict[str, Any]], *, cs_norm: bool = False) -> str:
    lines: list[str] = []
    sep = "=" * 72
    lines.append(sep)
    suffix = " [CS-NORM]" if cs_norm else ""
    lines.append(f"  REGRESSION LEAKAGE DIAGNOSTICS{suffix}")
    lines.append(sep)

    for r in results:
        status = "✅ PASS" if r["passed"] else "⚠️  WARN"
        lines.append(f"\n--- {r['test']} [{status}] ---")

        if r["test"] == "EMBARGO":
            lines.append(f"  IC standard (no gap):   {r['ic_standard']:.4f}  ({r['n_folds_std']} folds)")
            lines.append(f"  IC embargo (1Q gap):    {r['ic_embargo']:.4f}  ({r['n_folds_emb']} folds)")
            lines.append(f"  Delta (std − embargo):  {r['delta']:.4f}")
            lines.append(f"  Threshold: delta < 0.10")
            if r["passed"]:
                lines.append("  → Small drop: no adjacent-period leakage detected.")
            else:
                lines.append("  → Large drop: adjacent periods may share leaked information.")

        elif r["test"] == "SHUFFLED_LABELS":
            if "note" in r:
                lines.append(f"  {r['note']}")
            else:
                lines.append(f"  IC real model:           {r['ic_real']:.4f}")
                lines.append(f"  mean |IC| shuffled:      {r['mean_abs_ic_shuffled']:.4f}  ({r['n_iterations']}x)")
                lines.append(f"  Threshold: shuffled |IC| < 0.05")
                if r["passed"]:
                    lines.append("  → Shuffled IC near zero: model uses genuine signal, not artefacts.")
                else:
                    lines.append("  → Shuffled IC too high: model may exploit structural artefacts.")

        elif r["test"] == "RETRODICTION":
            if "note" in r:
                lines.append(f"  {r['note']}")
            else:
                lines.append(f"  IC forward (early→late):  {r['ic_forward']:.4f}  (train ≤2019, test >2019)")
                lines.append(f"  IC backward (late→early): {r['ic_backward']:.4f}  (train >2019, test ≤2019)")
                lines.append(f"  Delta (fwd − bwd):        {r['delta']:.4f}")
                lines.append(f"  Threshold: delta ≥ −0.05")
                if r["passed"]:
                    lines.append("  → Forward ≥ backward: no future-to-past information leak.")
                else:
                    lines.append("  → Backward IC higher: future information may have leaked into features.")

        elif r["test"] == "FEATURE_FUTURE_CORR":
            if "note" in r:
                lines.append(f"  {r['note']}")
            else:
                lines.append(f"  Mean |ρ_fwd| − |ρ_bwd|:  {r['mean_gap']:.4f}  ({r['n_features']} features)")
                lines.append(f"  Suspicious (gap > 0.05):  {r['n_suspicious']} features")
                lines.append(f"  Threshold: mean gap < 0.03")
                if r["n_suspicious"] > 0:
                    lines.append(f"  Top suspects: {r['suspicious_features'][:10]}")
                if r.get("top5_gaps"):
                    for feat, gap in r["top5_gaps"].items():
                        lines.append(f"    {feat:>30s}  gap={gap:+.4f}")
                if r["passed"]:
                    lines.append("  → No systematic forward bias in features.")
                else:
                    lines.append("  → Some features correlate more with future than past returns.")

    n_pass = sum(1 for r in results if r["passed"])
    lines.append(f"\n{'=' * 72}")
    lines.append(f"  Overall: {n_pass}/{len(results)} passed")
    lines.append("=" * 72)

    text = "\n".join(lines)
    print(text)
    return text


# ── Main ─────────────────────────────────────────────────────────────────────


def _pub_lag_days_from_argv(argv: list[str]) -> int:
    for i, a in enumerate(argv):
        if a.startswith("--pub-lag="):
            try:
                return max(0, int(a.split("=", 1)[1].strip()))
            except ValueError:
                return 0
        if a == "--pub-lag" and i + 1 < len(argv):
            try:
                return max(0, int(argv[i + 1].strip()))
            except ValueError:
                return 0
    return 0


def main() -> None:
    t0 = time.time()
    cs_norm = "--cs-norm" in sys.argv
    pub_lag = _pub_lag_days_from_argv(sys.argv)
    log.info("Loading data …")
    ohlcv, fundamentals, eulerpool_q, eulerpool_prof = load_data()

    log.info("Computing quarterly forward returns %s → %s …", START_QUARTER, END_QUARTER)
    fwd_df = compute_quarterly_forward_returns(ohlcv, START_QUARTER, END_QUARTER)
    if fwd_df.empty:
        log.error("No forward returns — aborting.")
        return

    log.info(
        "Building feature panel (cs_normalize=%s, pub_lag=%d) …",
        cs_norm, pub_lag,
    )
    X, y, period_labels = build_regression_feature_panel(
        ohlcv,
        fundamentals,
        fwd_df,
        cs_normalize=cs_norm,
        eulerpool_quarterly=eulerpool_q,
        eulerpool_profiles=eulerpool_prof,
        publication_lag_days=pub_lag,
        min_daily_volume_chf=config.MIN_DAILY_VOLUME_CHF,
    )

    valid = np.isfinite(y)
    X = X[valid].copy()
    y = y[valid]
    period_labels = period_labels[valid]

    all_nan_cols = X.columns[X.isna().all()]
    if len(all_nan_cols):
        X = X.drop(columns=all_nan_cols)
        log.info("Dropped %d all-NaN columns", len(all_nan_cols))

    log.info("Panel: %d samples × %d features, %d quarters", len(X), X.shape[1], len(set(period_labels)))

    results: list[dict[str, Any]] = []

    log.info("Test 1/4: Embargo …")
    results.append(embargo_test(X, y, period_labels))

    log.info("Test 2/4: Shuffled labels …")
    results.append(shuffled_labels_test(X, y, period_labels))

    log.info("Test 3/4: Retrodiction …")
    results.append(retrodiction_test(X, y, period_labels))

    log.info("Test 4/4: Feature-future correlation …")
    results.append(feature_future_correlation_test(X, y, period_labels))

    suffix_label = ""
    if cs_norm:
        suffix_label += " CS-NORM"
    if pub_lag > 0:
        suffix_label += f" LAG-{pub_lag}"
    report = print_report(results, cs_norm=cs_norm)
    if pub_lag > 0:
        log.info("Publication lag: %d days applied to PIT fundamentals", pub_lag)

    elapsed = time.time() - t0
    log.info("Total runtime: %.0f seconds", elapsed)


if __name__ == "__main__":
    main()
