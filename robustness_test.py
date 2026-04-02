#!/usr/bin/env python3
"""Robustness validation of the classification strategy.

Three tests (extended dataset: full SPI universe, history from config.YF_START):
  1. Walk-Forward — one OOS calendar year per run (2015–2025), train on Q4 of
     the prior year (aligned with :data:`config.CLASSIFICATION_PERIODS`).
  2. Multi-quarter ensemble — consensus over Q1–Q4 2024, evaluated on OOS 2025.
  3. Permutation test — model long picks vs. random equal-size portfolios
     (one test per walk-forward OOS year).

Phase 5 (optional, ``--regime-validation`` or ``--regime-only``):
  Regime v2 labeling (Bull / Bear / Sideways via SMA(50)/SMA(200) alignment;
  early-warning dampening on confidence — no vol-based Crisis bucket). Report
  for all classification quarters, intra-quarter regime stability, walk-forward
  2015–2025 with :func:`train_regime_aware_models` vs a single multi-quarter
  ensemble, comparison table, and a ROBUST/INCONCLUSIVE verdict (≥8/11 OOS years
  beating both benchmark and single-model).

Phase 6 (optional, ``--quarterly``):
  Quarterly rebalancing with hysteresis and realistic transaction costs (40 bps
  one-way). Trains regime-aware models once, then evaluates at three frequencies
  (quarterly / semi-annual / annual) on OOS 2015–2025. Reports a comparison
  table, per-year detail, turnover logs, and a best-frequency verdict.

Optional ``--use-cache`` / ``--no-cache``: with ``--use-cache``, load or save
  trained regime models under ``data/cache/regime_models_robustness.joblib`` to
  skip ~65 min retraining on repeat runs (evaluation still runs). ``--no-cache``
  forces a fresh train and disables saving.

Uses existing modules from src/ — joblib (via scikit-learn) for regime cache.
Runtime: long (several GridSearchCV fits per run); Phase 5 trains two full stacks.
"""
from __future__ import annotations

import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.backtest import (
    _compute_metrics,
    _daily_close_matrix,
    _portfolio_daily_returns,
    aggregate_forward_metrics_by_regime,
    build_oos_features,
    compare_regime_aware_vs_single,
    compute_oos_returns,
    evaluate_forward,
    evaluate_forward_multi,
    evaluate_forward_multi_regime_aware,
    evaluate_forward_quarterly_regime_aware,
)
from src.classifier import assign_groups, compute_q1_returns
from src.data_loader import download_ohlcv, load_fundamentals
from src.features import build_feature_matrix, drop_correlated_features
from src.features import MACRO_BENCHMARK_TICKER
from src.model import (
    predict,
    refit_on_full_data,
    train_classifier,
    train_multi_quarter_ensemble,
    train_regime_aware_models,
)
from src.regime import get_regime_history, label_periods
from src.universe import SPI_TICKERS, filter_by_min_volume

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("robustness")

# ── Experiment definitions ───────────────────────────────────────────────────
# Each entry: (label, period_index into CLASSIFICATION_PERIODS, oos_year,
#              oos_feature_cutoff). Cutoffs match Q1 feature dates for the OOS year.

def _walk_forward_specs() -> list[tuple[str, int, int, str]]:
    """Walk-forward OOS years 2015–2025; train on Q4 of year Y−1; OOS cutoff from Q1-Y."""
    return [
        ("OOS 2015 (train Q4-2014)", config.get_period_index("Q4-2014"), 2015, "2014-12-31"),
        ("OOS 2016 (train Q4-2015)", config.get_period_index("Q4-2015"), 2016, "2015-12-31"),
        ("OOS 2017 (train Q4-2016)", config.get_period_index("Q4-2016"), 2017, "2016-12-30"),
        ("OOS 2018 (train Q4-2017)", config.get_period_index("Q4-2017"), 2018, "2017-12-29"),
        ("OOS 2019 (train Q4-2018)", config.get_period_index("Q4-2018"), 2019, "2018-12-28"),
        ("OOS 2020 (train Q4-2019)", config.get_period_index("Q4-2019"), 2020, "2019-12-31"),
        ("OOS 2021 (train Q4-2020)", config.get_period_index("Q4-2020"), 2021, "2020-12-31"),
        ("OOS 2022 (train Q4-2021)", config.get_period_index("Q4-2021"), 2022, "2021-12-31"),
        ("OOS 2023 (train Q4-2022)", config.get_period_index("Q4-2022"), 2023, "2022-12-30"),
        ("OOS 2024 (train Q4-2023)", config.get_period_index("Q4-2023"), 2024, "2023-12-29"),
        ("OOS 2025 (train Q4-2024)", config.get_period_index("Q4-2024"), 2025, "2024-12-31"),
    ]


WALK_FORWARD = _walk_forward_specs()

# Quarters for multi-quarter ensemble (Q1–Q4 2024) — OOS 2025
ENSEMBLE_2024_LABELS = ["Q1-2024", "Q2-2024", "Q3-2024", "Q4-2024"]

PERMUTATION_N_ITER = 1000

# Phase 5: same OOS years as walk-forward (2015–2025)
REGIME_VALIDATION_OOS_YEARS: list[int] = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]

# Cache for :func:`train_regime_aware_models` output (``data/cache/regime_models_robustness.joblib``).
_REGIME_CACHE_PATH = config.DATA_DIR / "cache" / "regime_models_robustness.joblib"
_REGIME_TRAIN_KWARGS: dict[str, Any] = {
    "model_type": "rf",
    "recency_decay": 0.85,
    "consensus_method": "proba_average",
    "drop_correlated": True,
    "tune": True,
    "refit_full": True,
    "feature_selection": False,
}


def _regime_robustness_cache_hash(ohlcv: dict[str, pd.DataFrame]) -> str:
    """Stable hash: config + universe tickers + fixed training kwargs (see plan)."""
    payload = {
        "seed": config.RANDOM_SEED,
        "periods": list(config.CLASSIFICATION_PERIODS),
        "tickers": sorted(ohlcv.keys()),
        "train_kwargs": _REGIME_TRAIN_KWARGS,
    }
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


def load_or_train_regime_collection(
    ohlcv: dict[str, pd.DataFrame],
    fundamentals: dict[str, dict],
    *,
    use_cache: bool,
) -> tuple[Any, bool]:
    """Load ``RegimeModelCollection`` from joblib cache if hash matches, else train.

    Returns ``(regime_collection, from_cache)``. When ``use_cache`` is True, saves
    after training. Cache invalidates if ``CLASSIFICATION_PERIODS``, seed, ticker
    set, or training kwargs change.
    """
    import joblib

    chash = _regime_robustness_cache_hash(ohlcv)
    if use_cache and _REGIME_CACHE_PATH.exists():
        try:
            bundle = joblib.load(_REGIME_CACHE_PATH)
            if (
                isinstance(bundle, dict)
                and bundle.get("config_hash") == chash
                and bundle.get("regime_collection") is not None
            ):
                log.info("Using cached regime models (%s)", _REGIME_CACHE_PATH)
                return bundle["regime_collection"], True
            log.info("Regime cache present but hash mismatch — retraining.")
        except Exception as exc:
            log.warning("Regime cache load failed (%s) — retraining.", exc)

    periods = list(config.CLASSIFICATION_PERIODS)
    log.info("Training regime-aware models (%d quarters)…", len(periods))
    regime_collection = train_regime_aware_models(
        ohlcv,
        fundamentals,
        periods,
        config.RANDOM_SEED,
        **_REGIME_TRAIN_KWARGS,
    )

    if use_cache:
        try:
            _REGIME_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(
                {"config_hash": chash, "regime_collection": regime_collection},
                _REGIME_CACHE_PATH,
            )
            log.info("Saved regime models to cache (%s)", _REGIME_CACHE_PATH)
        except Exception as exc:
            log.warning("Could not save regime cache: %s", exc)

    return regime_collection, False


# ── Single experiment pipeline ───────────────────────────────────────────────


def run_experiment(
    ohlcv: dict[str, pd.DataFrame],
    fundamentals: dict[str, dict],
    period: tuple[str, str, str, str],
    oos_year: int,
    oos_cutoff: str,
    label: str,
) -> dict:
    """Train on one quarter, refit, predict, and evaluate on OOS year."""
    fc, q_start, q_end, plabel = period
    log.info("━━━ %s ━━━", label)
    t0 = time.time()

    returns = compute_q1_returns(ohlcv, q_start=q_start, q_end=q_end)
    labels = assign_groups(returns)

    X = build_feature_matrix(ohlcv, fc, fundamentals)
    X, _ = drop_correlated_features(X)

    common = X.index.intersection(labels.dropna().index)
    X_al, y_al = X.loc[common], labels.loc[common]
    log.info(
        "Training: %d samples x %d features (period %s)",
        len(X_al), X_al.shape[1], plabel,
    )

    result = train_classifier(X_al, y_al, config.RANDOM_SEED)
    result = refit_on_full_data(X_al, y_al, result)

    X_oos = build_oos_features(ohlcv, fundamentals, cutoff_date=oos_cutoff)
    preds = predict(result, X_oos)
    rets_oos = compute_oos_returns(ohlcv, year=oos_year)
    fwd = evaluate_forward(preds, rets_oos, ohlcv, year=oos_year, min_winners=5)

    n_win = int((preds == "Winners").sum())
    elapsed = time.time() - t0

    long_ret = fwd.long_only.get("cumulative_return", float("nan"))
    bm_ret = fwd.benchmark.get("cumulative_return", float("nan"))
    sharpe = fwd.long_only.get("sharpe_ratio", float("nan"))

    log.info(
        "Done in %.0fs — Long %+.1f%%, BM %+.1f%%, Sharpe %.2f, Winners %d",
        elapsed, long_ret * 100, bm_ret * 100, sharpe, n_win,
    )

    return {
        "label": label,
        "preds": preds,
        "fwd": fwd,
        "n_winners": n_win,
        "long_ret": long_ret,
        "bm_ret": bm_ret,
        "sharpe": sharpe,
        "oos_year": oos_year,
    }


def run_ensemble_experiment(
    ohlcv: dict[str, pd.DataFrame],
    fundamentals: dict[str, dict],
    period_labels: list[str],
    oos_year: int,
    oos_cutoff: str,
    label: str,
) -> dict:
    """Train multi-quarter consensus ensemble, then evaluate on OOS year."""
    log.info("━━━ %s ━━━", label)
    t0 = time.time()

    periods = [config.get_period_by_label(pl) for pl in period_labels]
    ensemble = train_multi_quarter_ensemble(
        ohlcv,
        fundamentals,
        periods,
        config.RANDOM_SEED,
        model_type="rf",
        recency_decay=0.85,
        consensus_method="proba_average",
        drop_correlated=True,
        tune=True,
        refit_full=True,
    )

    X_oos = build_oos_features(ohlcv, fundamentals, cutoff_date=oos_cutoff)
    preds = predict(ensemble, X_oos)
    rets_oos = compute_oos_returns(ohlcv, year=oos_year)
    fwd = evaluate_forward(preds, rets_oos, ohlcv, year=oos_year, min_winners=5)

    n_win = int((preds == "Winners").sum())
    elapsed = time.time() - t0

    long_ret = fwd.long_only.get("cumulative_return", float("nan"))
    bm_ret = fwd.benchmark.get("cumulative_return", float("nan"))
    sharpe = fwd.long_only.get("sharpe_ratio", float("nan"))

    log.info(
        "Done in %.0fs — Long %+.1f%%, BM %+.1f%%, Sharpe %.2f, Winners %d",
        elapsed, long_ret * 100, bm_ret * 100, sharpe, n_win,
    )

    return {
        "label": label,
        "preds": preds,
        "fwd": fwd,
        "n_winners": n_win,
        "long_ret": long_ret,
        "bm_ret": bm_ret,
        "sharpe": sharpe,
        "oos_year": oos_year,
    }


# ── Permutation test ─────────────────────────────────────────────────────────


def permutation_test(
    ohlcv: dict[str, pd.DataFrame],
    predicted_winners: list[str],
    oos_year: int,
    n_iter: int = PERMUTATION_N_ITER,
) -> dict | None:
    """p-value: fraction of random equal-size portfolios beating the model."""
    all_tickers = list(ohlcv.keys())
    close = _daily_close_matrix(ohlcv, all_tickers, oos_year)
    pool = list(close.columns)
    winners_avail = [w for w in predicted_winners if w in pool]
    n = len(winners_avail)

    if n < 1 or n > len(pool):
        log.warning("Permutation test skipped for %d (n_picks=%d, pool=%d)", oos_year, n, len(pool))
        return None

    model_dr = _portfolio_daily_returns(close, winners_avail)
    model_ret = _compute_metrics(model_dr)["cumulative_return"]

    rng = np.random.default_rng(config.RANDOM_SEED)
    rand_rets = np.empty(n_iter, dtype=float)
    for i in range(n_iter):
        picks = list(rng.choice(pool, size=n, replace=False))
        dr = _portfolio_daily_returns(close, picks)
        rand_rets[i] = _compute_metrics(dr)["cumulative_return"]

    p_value = float(np.mean(rand_rets >= model_ret))
    log.info(
        "Permutation OOS %d: model %+.1f%%, random median %+.1f%%, p=%.3f",
        oos_year, model_ret * 100, float(np.median(rand_rets)) * 100, p_value,
    )

    return {
        "model_return": model_ret,
        "random_median": float(np.median(rand_rets)),
        "p_value": p_value,
        "n_picks": n,
        "n_iterations": n_iter,
    }


# ── Data loading ─────────────────────────────────────────────────────────────


def load_data(*, force_refresh_ohlcv: bool = False) -> tuple[dict[str, pd.DataFrame], dict[str, dict]]:
    cache = config.ensure_data_dir()
    log.info(
        "Loading OHLCV data (%d tickers, %s …)",
        len(SPI_TICKERS),
        "re-downloading" if force_refresh_ohlcv else "using cache where present",
    )
    tickers_with_macro = list(dict.fromkeys([*SPI_TICKERS, MACRO_BENCHMARK_TICKER]))
    ohlcv = download_ohlcv(
        tickers_with_macro,
        config.YF_START,
        config.YF_END,
        cache,
        force_refresh=force_refresh_ohlcv,
    )
    smi_ohlcv = ohlcv.get(MACRO_BENCHMARK_TICKER)
    liquid = filter_by_min_volume(ohlcv, config.MIN_DAILY_VOLUME_CHF)
    ohlcv = {t: ohlcv[t] for t in liquid}
    if smi_ohlcv is not None and not smi_ohlcv.empty:
        ohlcv[MACRO_BENCHMARK_TICKER] = smi_ohlcv
    log.info("Universe after liquidity filter: %d tickers", len(ohlcv))

    fundamentals: dict[str, dict] = {}
    for t in ohlcv:
        fundamentals[t] = load_fundamentals(t, cache_dir=cache)

    return ohlcv, fundamentals


# ── Summary output ───────────────────────────────────────────────────────────


def _fp(v: float) -> str:
    return f"{v:+.1%}" if np.isfinite(v) else "N/A"


def _ff(v: float, d: int = 2) -> str:
    return f"{v:.{d}f}" if np.isfinite(v) else "N/A"


def print_summary(
    wf: list[dict],
    mq: list[dict],
    perm: list[tuple[int, dict | None]],
) -> str:
    lines: list[str] = []

    def p(s: str = "") -> None:
        lines.append(s)

    sep = "=" * 60
    p(sep)
    p("  ROBUSTNESS VALIDATION SUMMARY")
    p(sep)

    p("\nWalk-Forward Results:")
    for r in wf:
        p(f"  {r['label']}: "
          f"Long {_fp(r['long_ret'])}, "
          f"Benchmark {_fp(r['bm_ret'])}, "
          f"Sharpe {_ff(r['sharpe'])}")

    p("\nMulti-Quarter Ensemble (OOS 2025):")
    for r in mq:
        p(f"  {r['label']}: "
          f"Long {_fp(r['long_ret'])}, "
          f"Sharpe {_ff(r['sharpe'])}, "
          f"Winners {r['n_winners']}")

    n_it = PERMUTATION_N_ITER
    p(f"\nPermutation Test ({n_it} iterations, walk-forward picks per year):")
    for year, pr in perm:
        if pr is None:
            p(f"  OOS {year}: SKIPPED")
            continue
        p(f"  OOS {year}: "
          f"Model {_fp(pr['model_return'])} vs "
          f"Random median {_fp(pr['random_median'])} "
          f"(p={pr['p_value']:.3f})")

    wf_beat = sum(
        1 for r in wf
        if np.isfinite(r["long_ret"]) and np.isfinite(r["bm_ret"])
        and r["long_ret"] > r["bm_ret"]
    )
    mq_beat = sum(
        1 for r in mq
        if np.isfinite(r["long_ret"]) and np.isfinite(r["bm_ret"])
        and r["long_ret"] > r["bm_ret"]
    )
    perm_valid = [(y, pr) for y, pr in perm if pr is not None]
    perm_sig_10 = sum(1 for _, pr in perm_valid if pr["p_value"] < 0.10)
    perm_sig_05 = sum(1 for _, pr in perm_valid if pr["p_value"] < 0.05)

    n_wf = len(wf)
    n_pv = len(perm_valid)
    plan_gate = (
        wf_beat >= 8
        and n_wf >= 11
        and perm_sig_05 >= 8
        and n_pv >= 11
    )

    if wf_beat == len(wf) and mq_beat == len(mq) and perm_sig_10 == len(perm_valid) and perm_valid:
        verdict = "ROBUST"
    elif wf_beat <= 1 and mq_beat <= 0:
        verdict = "FRAGILE"
    else:
        verdict = "INCONCLUSIVE"

    p(f"\nConclusion: {verdict}")
    p(f"  Walk-forward: {wf_beat}/{len(wf)} beat benchmark")
    p(f"  Multi-quarter ensemble: {mq_beat}/{len(mq)} beat benchmark")
    p(f"  Permutation: {perm_sig_10}/{len(perm_valid)} significant (p<0.10); "
      f"{perm_sig_05}/{len(perm_valid)} (p<0.05)")
    p(
        f"  Plan gate (≥8/11 WF beat BM & ≥8/11 years p<0.05): "
        f"{'PASS' if plan_gate else 'FAIL'}"
    )

    text = "\n".join(lines)
    print(text)
    return text


# ── Phase 5: Regime validation (walk-forward, labeling, stability, verdict) ─


def build_regime_labeling_table(smi_ohlcv: pd.DataFrame) -> pd.DataFrame:
    """One row per :data:`config.CLASSIFICATION_PERIODS` quarter with regime at feature cutoff.

    Regime labels are ``bull`` / ``bear`` / ``sideways`` (v2); there is no ``crisis`` label.
    """
    states = label_periods(smi_ohlcv, config.CLASSIFICATION_PERIODS)
    rows: list[dict[str, object]] = []
    for fc, _qs, _qe, plabel in config.CLASSIFICATION_PERIODS:
        st = states.get(plabel)
        if st is None:
            rows.append(
                {
                    "period": plabel,
                    "feature_cutoff": fc,
                    "regime": "N/A",
                    "confidence": float("nan"),
                },
            )
            continue
        rows.append(
            {
                "period": plabel,
                "feature_cutoff": fc,
                "regime": st.label.value,
                "confidence": st.confidence,
            },
        )
    return pd.DataFrame(rows)


def build_regime_stability_table(smi_ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Intra-quarter regime changes (daily history within each label window).

    Daily regimes are v2: bull, bear, or sideways (transitions may include
    sideways ↔ bull/bear).
    """
    rows: list[dict[str, object]] = []
    for _fc, q_start, q_end, plabel in config.CLASSIFICATION_PERIODS:
        hist = get_regime_history(smi_ohlcv, q_start, q_end)
        if hist.empty or len(hist) < 2:
            rows.append(
                {
                    "period": plabel,
                    "q_start": q_start,
                    "q_end": q_end,
                    "n_days": int(len(hist)),
                    "n_transitions": 0,
                    "dominant_regime": "N/A",
                    "dominant_share": float("nan"),
                },
            )
            continue
        regimes = hist["regime"].astype(str)
        transitions = int((regimes.values[1:] != regimes.values[:-1]).sum())
        vc = regimes.value_counts(normalize=True)
        dom = str(vc.index[0])
        dom_share = float(vc.iloc[0])
        rows.append(
            {
                "period": plabel,
                "q_start": q_start,
                "q_end": q_end,
                "n_days": int(len(hist)),
                "n_transitions": transitions,
                "dominant_regime": dom,
                "dominant_share": dom_share,
            },
        )
    return pd.DataFrame(rows)


def run_regime_phase5_validation(
    ohlcv: dict[str, pd.DataFrame],
    fundamentals: dict[str, dict],
    *,
    use_cache: bool = False,
) -> dict:
    """Train regime-aware vs single ensemble on all periods; walk-forward OOS 2015–2025.

    Uses regime v2 (Bull / Bear / Sideways) throughout; reporting never refers to
    a legacy Crisis label.

    Parameters
    ----------
    use_cache
        If True, load/save regime-only models via :func:`load_or_train_regime_collection`.
        The single ensemble is always trained (not cached).
    """
    smi = ohlcv.get(MACRO_BENCHMARK_TICKER)
    if smi is None or smi.empty:
        raise ValueError(
            f"Phase 5 requires {MACRO_BENCHMARK_TICKER!r} in OHLCV (macro benchmark).",
        )

    t0 = time.time()
    periods = list(config.CLASSIFICATION_PERIODS)

    log.info("Phase 5: regime-aware models (%d quarters)…", len(periods))
    regime_collection, _from_cache = load_or_train_regime_collection(
        ohlcv,
        fundamentals,
        use_cache=use_cache,
    )

    log.info("Phase 5: training single multi-quarter ensemble (same periods)…")
    single_ensemble = train_multi_quarter_ensemble(
        ohlcv,
        fundamentals,
        periods,
        config.RANDOM_SEED,
        model_type="rf",
        recency_decay=0.85,
        consensus_method="proba_average",
        drop_correlated=True,
        tune=True,
        refit_full=True,
    )

    log.info(
        "Phase 5: walk-forward OOS %s — regime-aware…",
        REGIME_VALIDATION_OOS_YEARS,
    )
    regime_walk = evaluate_forward_multi_regime_aware(
        ohlcv,
        fundamentals,
        regime_collection=regime_collection,
        oos_years=REGIME_VALIDATION_OOS_YEARS,
        min_winners=5,
    )

    log.info(
        "Phase 5: walk-forward OOS %s — single ensemble…",
        REGIME_VALIDATION_OOS_YEARS,
    )
    single_walk = evaluate_forward_multi(
        ohlcv,
        fundamentals,
        train_result=single_ensemble,
        oos_years=REGIME_VALIDATION_OOS_YEARS,
        min_winners=5,
    )

    compare_df = compare_regime_aware_vs_single(
        regime_walk.summary,
        single_walk.summary,
    )
    agg_regime = aggregate_forward_metrics_by_regime(regime_walk.per_year)

    elapsed = time.time() - t0
    log.info("Phase 5 finished in %.0f seconds", elapsed)

    return {
        "labeling_table": build_regime_labeling_table(smi),
        "stability_table": build_regime_stability_table(smi),
        "regime_walk": regime_walk,
        "single_walk": single_walk,
        "compare_df": compare_df,
        "agg_by_regime": agg_regime,
        "elapsed_s": elapsed,
    }


def _regime_phase5_verdict(compare_df: pd.DataFrame) -> tuple[str, dict[str, int | float]]:
    """Plan rule: ROBUST if ≥8/11 OOS years beat benchmark and beat single-model."""
    if compare_df.empty or "long_winners_cum_regime" not in compare_df.columns:
        return "INCONCLUSIVE", {"n_years": 0}

    sub = compare_df.copy()
    req = ["long_winners_cum_regime", "benchmark_cum_regime", "long_winners_cum_delta"]
    for c in req:
        if c not in sub.columns:
            return "INCONCLUSIVE", {"n_years": 0}

    m = sub[req[0]].notna() & sub[req[1]].notna() & sub[req[2]].notna()
    sub = sub.loc[m]
    n = int(len(sub))
    if n == 0:
        return "INCONCLUSIVE", {"n_years": 0}

    beat_bm = int((sub["long_winners_cum_regime"] > sub["benchmark_cum_regime"]).sum())
    beat_single = int((sub["long_winners_cum_delta"] > 0).sum())
    beat_both = int(
        (
            (sub["long_winners_cum_regime"] > sub["benchmark_cum_regime"])
            & (sub["long_winners_cum_delta"] > 0)
        ).sum(),
    )

    if n >= 11 and beat_both >= 8:
        verdict = "ROBUST"
    elif beat_both <= 2 and n >= 5:
        verdict = "FRAGILE"
    else:
        verdict = "INCONCLUSIVE"

    stats: dict[str, int | float] = {
        "n_years": n,
        "years_beat_benchmark": beat_bm,
        "years_beat_single": beat_single,
        "years_beat_both": beat_both,
    }
    return verdict, stats


def print_regime_phase5_report(result: dict) -> str:
    """Format Phase 5 tables, comparison, and verdict (regime v2: Bull / Bear / Sideways)."""
    lines: list[str] = []

    def p(s: str = "") -> None:
        lines.append(s)

    sep = "=" * 72
    p(sep)
    p("  PHASE 5 — REGIME VALIDATION v2 (Bull / Bear / Sideways, walk-forward 2015–2025)")
    p(sep)

    p("\n--- Regime labeling (feature cutoff per quarter) ---")
    p(
        "Regimes: bull = Close>SMA(50) and SMA(50)>SMA(200); bear = all three descending; "
        "sideways = otherwise. Confidence uses alignment strength plus early-warning "
        "dampening (sma_cross_gap, sma_50_slope). Legacy Crisis is not used.",
    )
    p(result["labeling_table"].to_string(index=False))

    p("\n--- Intra-quarter regime stability (^SSMI daily regime within Q window) ---")
    p(
        "n_transitions = days where regime differs from previous trading day; "
        "dominant_share = fraction of days in the modal regime (bull, bear, or sideways).",
    )
    stab = result["stability_table"]
    p(stab.to_string(index=False))
    if not stab.empty and "n_transitions" in stab.columns:
        nt = pd.to_numeric(stab["n_transitions"], errors="coerce").dropna()
        ds = pd.to_numeric(stab["dominant_share"], errors="coerce").dropna()
        n_switch_q = int((nt > 0).sum()) if len(nt) else 0
        p(
            f"  Summary: median transitions/quarter={float(nt.median()) if len(nt) else float('nan'):.1f}, "
            f"mean={float(nt.mean()) if len(nt) else float('nan'):.2f}, "
            f"quarters with ≥1 switch={n_switch_q}/{len(nt)}, "
            f"median dominant_share={float(ds.median()) if len(ds) else float('nan'):.2%}",
        )

    p("\n--- Walk-forward: regime-aware vs single ensemble (long Winners cumulative) ---")
    cdf = result["compare_df"]
    if cdf.empty:
        p("  (No overlapping OOS years between regime-aware and single summaries.)")
    else:
        p(cdf.to_string())

    agg = result["agg_by_regime"]
    if agg is not None and not agg.empty:
        p("\n--- OOS metrics aggregated by detected regime (bull / bear / sideways, regime-aware run) ---")
        p(agg.to_string())

    verdict, st = _regime_phase5_verdict(result["compare_df"])
    p(f"\n--- Verdict ({verdict}) ---")
    p(
        f"  Years evaluated: {st.get('n_years', 0)}  |  "
        f"beat benchmark: {st.get('years_beat_benchmark', '—')}  |  "
        f"beat single-model: {st.get('years_beat_single', '—')}  |  "
        f"beat both (same year): {st.get('years_beat_both', '—')}",
    )
    p(
        "  Rule: ROBUST if ≥8/11 OOS years have long-Winners return above benchmark "
        "and above single-model (long_winners_cum_delta > 0).",
    )
    p(f"  Phase 5 runtime: {result.get('elapsed_s', 0):.0f}s")

    text = "\n".join(lines)
    print(text)
    return text


# ── Phase 6: Quarterly rebalancing validation ────────────────────────────────


REBALANCE_FREQS: dict[str, int] = {
    "quarterly": 1,
    "semi_annual": 2,
    "annual": 4,
}


def run_quarterly_rebalance_validation(
    ohlcv: dict[str, pd.DataFrame],
    fundamentals: dict[str, dict],
    *,
    costs_bps: float = 40.0,
    min_winners: int = 5,
    hysteresis_rule: str = "keep_non_losers",
    max_position_weight: float | None = 0.30,
    use_cache: bool = False,
) -> dict:
    """Train regime-aware models, then evaluate at three rebalancing frequencies.

    Returns a dict keyed by frequency label (``quarterly``, ``semi_annual``,
    ``annual``) → :class:`~backtest.QuarterlyForwardResult`, plus the trained
    ``regime_collection`` and elapsed time.

    Parameters
    ----------
    max_position_weight
        Cap on any single position's weight (default 0.30 = 30%).
    use_cache
        If True, load/save regime models via :func:`load_or_train_regime_collection`.
    """
    smi = ohlcv.get(MACRO_BENCHMARK_TICKER)
    if smi is None or smi.empty:
        raise ValueError(
            f"Quarterly rebalancing validation requires {MACRO_BENCHMARK_TICKER!r} "
            "in OHLCV.",
        )

    t0 = time.time()

    log.info("Quarterly validation: regime-aware models…")
    regime_collection, _from_cache = load_or_train_regime_collection(
        ohlcv,
        fundamentals,
        use_cache=use_cache,
    )

    results: dict[str, object] = {}
    for freq_label, freq_val in REBALANCE_FREQS.items():
        log.info(
            "Quarterly validation: evaluating freq=%s (rebalance_freq=%d) …",
            freq_label,
            freq_val,
        )
        qfr = evaluate_forward_quarterly_regime_aware(
            ohlcv,
            fundamentals,
            regime_collection=regime_collection,
            oos_years=REGIME_VALIDATION_OOS_YEARS,
            costs_bps=costs_bps,
            min_winners=min_winners,
            rebalance_freq=freq_val,
            hysteresis_rule=hysteresis_rule,
            max_position_weight=max_position_weight,
        )
        results[freq_label] = qfr

    elapsed = time.time() - t0
    log.info("Quarterly validation finished in %.0f seconds", elapsed)

    results["regime_collection"] = regime_collection
    results["elapsed_s"] = elapsed
    return results


def print_quarterly_rebalance_report(result: dict, costs_bps: float = 40.0) -> str:
    """Format the 3-frequency comparison table and per-year detail."""
    lines: list[str] = []

    def p(s: str = "") -> None:
        lines.append(s)

    sep = "=" * 72
    p(sep)
    p("  QUARTERLY REBALANCING VALIDATION (Hysteresis + Transaction Costs)")
    p(sep)

    freq_labels = [k for k in REBALANCE_FREQS if k in result]
    if not freq_labels:
        p("  (No frequency results found.)")
        text = "\n".join(lines)
        print(text)
        return text

    # ── per-frequency summary rows ──
    summary_rows: list[dict[str, object]] = []
    for fl in freq_labels:
        qfr = result[fl]
        per_year = qfr.per_year
        n_years = len(per_year)
        if n_years == 0:
            summary_rows.append({"freq": fl})
            continue

        beat_bm = 0
        long_cums: list[float] = []
        bm_cums: list[float] = []
        sharpes: list[float] = []
        max_dds: list[float] = []

        for yr in sorted(per_year):
            ftr = per_year[yr]
            lc = ftr.long_only.get("cumulative_return", float("nan"))
            bc = ftr.benchmark.get("cumulative_return", float("nan"))
            sh = ftr.long_only.get("sharpe_ratio", float("nan"))
            md = ftr.long_only.get("max_drawdown", float("nan"))
            if np.isfinite(lc):
                long_cums.append(lc)
            if np.isfinite(bc):
                bm_cums.append(bc)
            if np.isfinite(sh):
                sharpes.append(sh)
            if np.isfinite(md):
                max_dds.append(md)
            if np.isfinite(lc) and np.isfinite(bc) and lc > bc:
                beat_bm += 1

        td = qfr.quarterly_detail
        avg_turnover = (
            float(td["turnover_pct"].mean()) if not td.empty and "turnover_pct" in td.columns else 0.0
        )

        avg_long = float(np.mean(long_cums)) if long_cums else float("nan")
        avg_bm = float(np.mean(bm_cums)) if bm_cums else float("nan")
        avg_sharpe = float(np.mean(sharpes)) if sharpes else float("nan")
        avg_maxdd = float(np.mean(max_dds)) if max_dds else float("nan")

        summary_rows.append({
            "freq": fl,
            "beat_bm": f"{beat_bm}/{n_years}",
            "avg_long_cum": avg_long,
            "avg_bm_cum": avg_bm,
            "avg_turnover": avg_turnover,
            "total_costs_bps": qfr.total_costs_bps,
            "avg_net_return": avg_long,
            "avg_sharpe": avg_sharpe,
            "avg_maxdd": avg_maxdd,
        })

    p(f"\n--- Rebalancing-Frequenz-Vergleich (costs={costs_bps:.0f}bps one-way) ---")

    header = f"{'':>16s}"
    for sr in summary_rows:
        header += f"  {sr['freq']:>14s}"
    p(header)

    metric_keys = [
        ("beat_bm", "beat_bm", None),
        ("avg_long_cum", "avg_long_cum", "pct"),
        ("avg_bm_cum", "avg_bm_cum", "pct"),
        ("avg_turnover", "avg_turnover", "pct1"),
        ("total_costs_bps", "total_costs", "bps"),
        ("avg_net_return", "avg_net_return", "pct"),
        ("avg_sharpe", "avg_sharpe", "f2"),
        ("avg_maxdd", "avg_maxdd", "pct"),
    ]

    for key, row_label, fmt in metric_keys:
        row = f"{row_label:>16s}"
        for sr in summary_rows:
            v = sr.get(key, "—")
            if isinstance(v, str):
                row += f"  {v:>14s}"
            elif fmt == "pct":
                row += f"  {v:>+13.1%}" if np.isfinite(v) else f"  {'N/A':>14s}"
            elif fmt == "pct1":
                row += f"  {v:>13.0f}%" if np.isfinite(v) else f"  {'N/A':>14s}"
            elif fmt == "bps":
                row += f"  {v:>13.0f}" if np.isfinite(v) else f"  {'N/A':>14s}"
            elif fmt == "f2":
                row += f"  {v:>14.2f}" if np.isfinite(v) else f"  {'N/A':>14s}"
            else:
                row += f"  {v!s:>14s}"
        p(row)

    # ── per-year detail per frequency ──
    for fl in freq_labels:
        qfr = result[fl]
        per_year = qfr.per_year
        p(f"\n--- Per-year detail: {fl} (rebalance_freq={REBALANCE_FREQS[fl]}) ---")
        detail_rows: list[str] = []
        detail_rows.append(
            f"{'year':>6s}  {'long_cum':>10s}  {'bm_cum':>10s}  "
            f"{'sharpe':>8s}  {'maxDD':>8s}  {'costs_bps':>10s}  {'beat_bm':>8s}"
        )
        for yr in sorted(per_year):
            ftr = per_year[yr]
            lc = ftr.long_only.get("cumulative_return", float("nan"))
            bc = ftr.benchmark.get("cumulative_return", float("nan"))
            sh = ftr.long_only.get("sharpe_ratio", float("nan"))
            md = ftr.long_only.get("max_drawdown", float("nan"))
            cost = ftr.costs_bps
            beat = "✓" if np.isfinite(lc) and np.isfinite(bc) and lc > bc else "✗"
            detail_rows.append(
                f"{yr:>6d}  {_fp(lc):>10s}  {_fp(bc):>10s}  "
                f"{_ff(sh):>8s}  {_ff(md):>8s}  {cost:>10.0f}  {beat:>8s}"
            )
        for dr in detail_rows:
            p(dr)

    # ── turnover log (quarterly detail) ──
    for fl in freq_labels:
        qfr = result[fl]
        if qfr.quarterly_detail.empty:
            continue
        p(f"\n--- Turnover detail: {fl} ---")
        cols = ["year", "quarter", "cutoff", "n_positions", "n_swapped",
                "turnover_pct", "cost_bps", "cum_return"]
        td = qfr.quarterly_detail
        avail_cols = [c for c in cols if c in td.columns]
        display = td[avail_cols].copy()
        if "cum_return" in display.columns:
            display["cum_return"] = display["cum_return"].map(
                lambda v: f"{v:+.1%}" if np.isfinite(v) else "N/A",
            )
        if "turnover_pct" in display.columns:
            display["turnover_pct"] = display["turnover_pct"].map(
                lambda v: f"{v:.0f}%" if np.isfinite(v) else "N/A",
            )
        p(display.to_string(index=False))

    # ── verdict ──
    p(f"\n--- Verdict ---")
    best_freq = None
    best_beat = -1
    best_avg = float("-inf")
    for fl in freq_labels:
        qfr = result[fl]
        per_year = qfr.per_year
        n_beat = sum(
            1 for ftr in per_year.values()
            if ftr.long_only.get("cumulative_return", float("nan"))
            > ftr.benchmark.get("cumulative_return", float("nan"))
        )
        cums = [
            ftr.long_only.get("cumulative_return", float("nan"))
            for ftr in per_year.values()
        ]
        avg = float(np.nanmean(cums)) if cums else float("-inf")
        if n_beat > best_beat or (n_beat == best_beat and avg > best_avg):
            best_freq = fl
            best_beat = n_beat
            best_avg = avg

    p(f"  Best frequency: {best_freq} ({best_beat} years beat benchmark, "
      f"avg long cum {best_avg:+.1%})")
    p(f"  Runtime: {result.get('elapsed_s', 0):.0f}s")

    text = "\n".join(lines)
    print(text)
    return text


# ── Main ─────────────────────────────────────────────────────────────────────


def _use_regime_cache_from_argv(argv: list[str]) -> bool:
    """``--no-cache`` overrides ``--use-cache``."""
    if "--no-cache" in argv:
        return False
    return "--use-cache" in argv


def main() -> None:
    t_global = time.time()
    argv = sys.argv
    refresh = "--refresh-ohlcv" in argv
    regime_only = "--regime-only" in argv
    quarterly_only = "--quarterly" in argv
    run_regime_phase5 = "--regime-validation" in argv or regime_only
    use_regime_cache = _use_regime_cache_from_argv(argv)

    if refresh:
        log.info("Flag --refresh-ohlcv: ignoring Parquet cache (full re-download).")
    if regime_only:
        log.info("Flag --regime-only: running Phase 5 regime validation only.")
    elif quarterly_only:
        log.info("Flag --quarterly: running quarterly rebalancing validation only.")
    elif run_regime_phase5:
        log.info("Flag --regime-validation: will run Phase 5 after standard tests.")
    if use_regime_cache:
        log.info(
            "Flag --use-cache: regime models may load from %s",
            _REGIME_CACHE_PATH,
        )
    elif "--no-cache" in argv:
        log.info("Flag --no-cache: regime model cache disabled.")

    if regime_only:
        ohlcv, fundamentals = load_data(force_refresh_ohlcv=refresh)
        log.info("╔══ Phase 5 only: Regime validation ══╗")
        r5 = run_regime_phase5_validation(ohlcv, fundamentals, use_cache=use_regime_cache)
        print_regime_phase5_report(r5)
        log.info("Total runtime: %.0f seconds", time.time() - t_global)
        return

    if quarterly_only:
        ohlcv, fundamentals = load_data(force_refresh_ohlcv=refresh)
        log.info("╔══ Quarterly rebalancing validation (3 frequencies) ══╗")
        rq = run_quarterly_rebalance_validation(ohlcv, fundamentals, use_cache=use_regime_cache)
        print_quarterly_rebalance_report(rq)
        log.info("Total runtime: %.0f seconds", time.time() - t_global)
        return

    ohlcv, fundamentals = load_data(force_refresh_ohlcv=refresh)

    cache: dict[tuple[int, int], dict] = {}

    def run_or_cache(label: str, pidx: int, oos_year: int, oos_cutoff: str) -> dict:
        key = (pidx, oos_year)
        if key in cache:
            log.info("Reusing cached result for period_idx=%d, oos=%d", pidx, oos_year)
            r = cache[key].copy()
            r["label"] = label
            return r
        period = config.CLASSIFICATION_PERIODS[pidx]
        r = run_experiment(ohlcv, fundamentals, period, oos_year, oos_cutoff, label)
        cache[key] = r
        return r

    log.info("╔══ TEST 1: Walk-Forward (2015–2025) ══╗")
    wf_results = [run_or_cache(*cfg) for cfg in WALK_FORWARD]

    log.info("╔══ TEST 2: Multi-Quarter Ensemble (Q1–Q4 2024 → OOS 2025) ══╗")
    mq_results = [
        run_ensemble_experiment(
            ohlcv,
            fundamentals,
            ENSEMBLE_2024_LABELS,
            2025,
            "2024-12-31",
            "Ensemble Q1–Q4-2024 (OOS 2025)",
        ),
    ]

    log.info("╔══ TEST 3: Permutation (per walk-forward year) ══╗")
    perm_results: list[tuple[int, dict | None]] = []
    for r in wf_results:
        oos_year = int(r["oos_year"])
        winners = list(r["preds"][r["preds"] == "Winners"].index)
        pr = permutation_test(ohlcv, winners, oos_year)
        perm_results.append((oos_year, pr))

    print_summary(wf_results, mq_results, perm_results)

    if run_regime_phase5:
        log.info("╔══ Phase 5: Regime validation (2015–2025) ══╗")
        r5 = run_regime_phase5_validation(ohlcv, fundamentals, use_cache=use_regime_cache)
        print_regime_phase5_report(r5)

    log.info("Total runtime: %.0f seconds", time.time() - t_global)


if __name__ == "__main__":
    main()
