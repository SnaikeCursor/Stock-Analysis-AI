#!/usr/bin/env python3
"""Robustness validation for the stock-selection pipeline.

**Primary optional path — Phase 6 (``--quarterly``):** walk-forward **regression**
quarterly rebalancing (1-month forward returns, single RF per OOS year, IC and
turnover reporting). This is the recommended entry point for OOS quarterly
studies.

**Legacy classification path (deprecated for new work; kept for regression
against prior baselines):** default runs Tests 1–3 below without flags.

Three classification tests (extended dataset: full SPI universe, history from
``config.YF_START``):
  1. Walk-Forward — one OOS calendar year per run (2015–2025), train on Q4 of
     the prior year (aligned with :data:`config.CLASSIFICATION_PERIODS`).
  2. Multi-quarter ensemble — consensus over Q1–Q4 2024, evaluated on OOS 2025.
  3. Permutation test — model long picks vs. random equal-size portfolios
     (one test per walk-forward OOS year).

Phase 5 (optional, ``--regime-validation`` or ``--regime-only``) — legacy
classification regime tooling:
  Regime v2 labeling (Bull / Bear / Sideways via SMA(50)/SMA(200) alignment;
  early-warning dampening on confidence — no vol-based Crisis bucket). Report
  for all classification quarters, intra-quarter regime stability, walk-forward
  2015–2025 with :func:`train_regime_aware_models` vs a single multi-quarter
  ensemble, comparison table, and a ROBUST/INCONCLUSIVE verdict (≥8/11 OOS years
  beating both benchmark and single-model).

Phase 6 (``--quarterly``):
  Regression-based quarterly rebalancing with rank-based hysteresis and
  realistic transaction costs (40 bps one-way).  Uses **walk-forward
  regression training** — a single RF regressor per OOS year trained on
  monthly forward returns up to Nov of year−1 (no future labels).  At each
  cutoff the regressor predicts 1-month returns; the top-N stocks are
  selected with hysteresis buffer and return-proportional weights.
  Evaluates at three frequencies (quarterly / semi-annual / annual) on OOS
  2015–2025.  Reports a comparison table with IC, per-year detail, turnover
  logs, and a best-frequency verdict.

Optional ``--use-cache`` / ``--no-cache``: with ``--use-cache``, load or save
  joblib caches to skip retraining (evaluation still runs).  Paths depend on
  mode: regime models under ``data/cache/regime_models_robustness.joblib`` and
  ``data/cache/regime_models_wf_<year>.joblib`` (Phase 5 / legacy quarterly
  regime path); walk-forward regression models under
  ``data/cache/regression_wf_<year>.joblib`` when using ``--quarterly``.
  ``--no-cache`` forces a fresh train and disables saving.

Uses existing modules from src/ — joblib (via scikit-learn) for model caches.
Runtime: long (several GridSearchCV fits per run); Phase 5 trains two full stacks.
"""
from __future__ import annotations

import hashlib
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.backtest import (
    QuarterlyForwardResult,
    _capped_proba_weights,
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
    evaluate_forward_quarterly_regression,
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
from src.regression_model import (
    RegressionTrainResult,
    refit_regressor_full,
    train_regressor,
)
from src.regression_targets import (
    _last_trading_day_on_or_before,
    compute_annual_forward_returns,
    compute_quarterly_forward_returns,
    normalize_forward_returns_cs,
)
from src.universe import SPI_TICKERS, filter_liquid_at_cutoff

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


# ── Walk-forward regime training (one model per OOS year) ─────────────────────


def _periods_before_year(oos_year: int) -> list[tuple[str, str, str, str]]:
    """Classification periods where ``q_end`` strictly precedes Jan-1 of *oos_year*."""
    cutoff = pd.Timestamp(f"{oos_year}-01-01")
    return [
        p for p in config.CLASSIFICATION_PERIODS
        if pd.Timestamp(p[2]) < cutoff
    ]


def _walk_forward_cache_path(oos_year: int) -> Path:
    return config.DATA_DIR / "cache" / f"regime_models_wf_{oos_year}.joblib"


def _walk_forward_cache_hash(
    ohlcv: dict[str, pd.DataFrame],
    oos_year: int,
) -> str:
    periods = _periods_before_year(oos_year)
    payload = {
        "seed": config.RANDOM_SEED,
        "periods": [p[3] for p in periods],
        "tickers": sorted(ohlcv.keys()),
        "train_kwargs": _REGIME_TRAIN_KWARGS,
        "oos_year": oos_year,
    }
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


def train_walk_forward_regime_collection(
    ohlcv: dict[str, pd.DataFrame],
    fundamentals: dict[str, dict],
    oos_year: int,
    *,
    use_cache: bool = False,
) -> tuple[Any, bool]:
    """Train regime models on quarters with ``q_end < {oos_year}-01-01`` only.

    Honest walk-forward: the model for OOS year *T* never sees labels from
    Q1-T or later, eliminating training-leakage identified in the audit.

    Returns ``(regime_collection, from_cache)``.
    """
    import joblib

    periods = _periods_before_year(oos_year)
    n_periods = len(periods)

    if n_periods < 4:
        raise ValueError(
            f"Walk-forward OOS {oos_year}: only {n_periods} training quarters "
            f"(need >= 4 for regime-aware models)"
        )

    cache_path = _walk_forward_cache_path(oos_year)
    chash = _walk_forward_cache_hash(ohlcv, oos_year)

    if use_cache and cache_path.exists():
        try:
            bundle = joblib.load(cache_path)
            if (
                isinstance(bundle, dict)
                and bundle.get("config_hash") == chash
                and bundle.get("regime_collection") is not None
            ):
                log.info(
                    "Walk-forward OOS %d: cached regime models loaded (%s)",
                    oos_year,
                    cache_path,
                )
                return bundle["regime_collection"], True
            log.info(
                "Walk-forward OOS %d: cache hash mismatch — retraining.",
                oos_year,
            )
        except Exception as exc:
            log.warning(
                "Walk-forward OOS %d: cache load failed (%s) — retraining.",
                oos_year,
                exc,
            )

    log.info(
        "Walk-forward OOS %d: training on %d quarters (%s … %s)",
        oos_year,
        n_periods,
        periods[0][3],
        periods[-1][3],
    )

    regime_collection = train_regime_aware_models(
        ohlcv,
        fundamentals,
        periods,
        config.RANDOM_SEED,
        **_REGIME_TRAIN_KWARGS,
    )

    if use_cache:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(
                {"config_hash": chash, "regime_collection": regime_collection},
                cache_path,
            )
            log.info(
                "Walk-forward OOS %d: regime cache saved to %s",
                oos_year,
                cache_path,
            )
        except Exception as exc:
            log.warning(
                "Walk-forward OOS %d: could not save cache: %s",
                oos_year,
                exc,
            )

    return regime_collection, False


# ── Walk-forward regression training (one regressor per OOS year) ─────────


_REGRESSION_START_QUARTER = "2012-Q1"
_REGRESSION_TRAIN_KWARGS: dict[str, Any] = {
    "model_type": "rf",
    "tune": True,
    "cv_folds": 10,
}


def _regression_wf_cache_path(
    oos_year: int,
    target_horizon: str = "quarterly",
    cs_normalize: bool = False,
    use_eulerpool: bool = False,
    publication_lag_days: int = 0,
    cutoff_shift_days: int = 0,
) -> Path:
    suffix = "" if target_horizon == "quarterly" else f"_{target_horizon}"
    if cs_normalize:
        suffix += "_cs"
    if use_eulerpool:
        suffix += "_pit"
    if publication_lag_days > 0:
        suffix += f"_lag{publication_lag_days}"
    if cutoff_shift_days > 0:
        suffix += f"_ps{cutoff_shift_days}"
    return config.DATA_DIR / "cache" / f"regression_wf_{oos_year}{suffix}.joblib"


def _regression_wf_cache_hash(
    ohlcv: dict[str, pd.DataFrame],
    oos_year: int,
    target_horizon: str = "quarterly",
    cs_normalize: bool = False,
    use_eulerpool: bool = False,
    publication_lag_days: int = 0,
    cutoff_shift_days: int = 0,
) -> str:
    payload = {
        "seed": config.RANDOM_SEED,
        "tickers": sorted(ohlcv.keys()),
        "train_kwargs": _REGRESSION_TRAIN_KWARGS,
        "oos_year": oos_year,
        "start_quarter": _REGRESSION_START_QUARTER,
        "target_horizon": target_horizon,
        "cs_normalize": cs_normalize,
        "use_eulerpool": use_eulerpool,
        "publication_lag_days": publication_lag_days,
        "cutoff_shift_days": cutoff_shift_days,
    }
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


def build_regression_feature_panel(
    ohlcv: dict[str, pd.DataFrame],
    fundamentals: dict[str, dict],
    forward_returns_df: pd.DataFrame,
    *,
    cs_normalize: bool = False,
    eulerpool_quarterly: dict[str, list[dict]] | None = None,
    eulerpool_profiles: dict[str, dict] | None = None,
    publication_lag_days: int = 0,
    cutoff_shift_days: int = 0,
    min_daily_volume_chf: float = 0.0,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Build stacked ``(X, y, period_labels)`` from quarterly cutoffs.

    For each unique ``cutoff_date`` in *forward_returns_df*, calls
    :func:`~features.build_feature_matrix` and aligns with forward returns.

    Parameters
    ----------
    forward_returns_df
        Output of :func:`~regression_targets.compute_quarterly_forward_returns`
        or :func:`~regression_targets.compute_annual_forward_returns`.
    cs_normalize
        If True, cross-sectionally z-score the target per ``cutoff_date`` (peer
        relative target) via :func:`~regression_targets.normalize_forward_returns_cs`.
    eulerpool_quarterly
        Ticker -> list of Eulerpool quarterly records.  When provided, features
        are constructed point-in-time via :func:`~features.build_fundamental_features_pit`.
    eulerpool_profiles
        Ticker -> Eulerpool company profile dict (sector info).
    publication_lag_days
        Passed to :func:`~features.build_feature_matrix` when Eulerpool
        quarterly data is used (PIT fundamentals).
    cutoff_shift_days
        Must match the value used when building *forward_returns_df* via
        :func:`~regression_targets.compute_quarterly_forward_returns` /
        :func:`~regression_targets.compute_annual_forward_returns` (metadata
        for callers; panel rows already reflect shifted cutoffs in the frame).
    min_daily_volume_chf
        When > 0, apply :func:`~universe.filter_liquid_at_cutoff` per cutoff
        (dynamic universe).  Tickers below the median-turnover threshold in
        the trailing 6 months are excluded for that cutoff.

    Returns
    -------
    (X, y, period_labels)
        *X*: stacked feature DataFrame (one row per ticker-quarter).
        *y*: forward returns (or CS z-scores when *cs_normalize*) aligned with *X*.
        *period_labels*: ``YYYY-QN`` strings aligned with *X* (for CV).
    """
    ret_col = next(
        (
            c
            for c in (
                "forward_1q_return",
                "forward_1y_return",
                "forward_1m_return",
            )
            if c in forward_returns_df.columns
        ),
        None,
    )
    if ret_col is None:
        raise ValueError(
            "forward_returns_df must contain one of: "
            "forward_1q_return, forward_1y_return, forward_1m_return",
        )

    if cs_normalize:
        forward_returns_df = normalize_forward_returns_cs(
            forward_returns_df,
            ret_col,
            method="zscore",
        )
        ret_col = f"{ret_col}_cs"

    use_dynamic_filter = min_daily_volume_chf > 0
    cutoff_dates = sorted(forward_returns_df["cutoff_date"].unique())
    x_parts: list[pd.DataFrame] = []
    y_parts: list[np.ndarray] = []
    pl_parts: list[np.ndarray] = []
    universe_sizes: list[int] = []

    for i, cd in enumerate(cutoff_dates):
        cd_str = pd.Timestamp(cd).strftime("%Y-%m-%d")
        period_str = str(pd.Timestamp(cd).to_period("Q"))

        # ── Dynamic universe: per-cutoff liquidity gate ──
        if use_dynamic_filter:
            eligible = filter_liquid_at_cutoff(
                ohlcv, cd_str, min_daily_volume_chf,
            )
            eligible.discard(MACRO_BENCHMARK_TICKER)
            ohlcv_cd = {
                t: v for t, v in ohlcv.items()
                if t in eligible or t == MACRO_BENCHMARK_TICKER
            }
        else:
            ohlcv_cd = ohlcv
            eligible = {t for t in ohlcv if t != MACRO_BENCHMARK_TICKER}

        universe_sizes.append(len(eligible))

        period_rets = (
            forward_returns_df[forward_returns_df["cutoff_date"] == cd]
            .set_index("ticker")
        )

        X_period = build_feature_matrix(
            ohlcv_cd,
            cd_str,
            fundamentals,
            eulerpool_quarterly=eulerpool_quarterly,
            eulerpool_profiles=eulerpool_profiles,
            publication_lag_days=publication_lag_days,
        )
        common = X_period.index.intersection(period_rets.index)
        if len(common) == 0:
            continue

        x_parts.append(X_period.loc[common])
        y_parts.append(period_rets.loc[common, ret_col].values)
        pl_parts.append(np.full(len(common), period_str))

        if (i + 1) % 4 == 0 or i + 1 == len(cutoff_dates):
            log.info(
                "Feature panel: %d/%d quarters processed (%d samples so far)",
                i + 1,
                len(cutoff_dates),
                sum(len(p) for p in y_parts),
            )

    X = pd.concat(x_parts, axis=0)
    y = np.concatenate(y_parts)
    period_labels = np.concatenate(pl_parts)

    if universe_sizes:
        log.info(
            "Dynamic universe: min=%d  median=%d  max=%d tickers per cutoff "
            "(pit=%s, filter=%.0f CHF)",
            min(universe_sizes),
            int(np.median(universe_sizes)),
            max(universe_sizes),
            eulerpool_quarterly is not None,
            min_daily_volume_chf,
        )
    log.info(
        "Feature panel complete: %d samples × %d features, %d quarters",
        len(X),
        X.shape[1],
        len(cutoff_dates),
    )
    return X, y, period_labels


def train_walk_forward_regressor(
    ohlcv: dict[str, pd.DataFrame],
    fundamentals: dict[str, dict],
    oos_year: int,
    *,
    use_cache: bool = False,
    feature_panel: tuple[pd.DataFrame, np.ndarray, np.ndarray] | None = None,
    target_horizon: str = "quarterly",
    eulerpool_quarterly: dict[str, list[dict]] | None = None,
    eulerpool_profiles: dict[str, dict] | None = None,
    cs_normalize: bool = False,
    publication_lag_days: int = 0,
    cutoff_shift_days: int = 0,
) -> tuple[RegressionTrainResult, bool]:
    """Train a single regression model for OOS year *oos_year*.

    Honest walk-forward: features and targets are restricted so that no
    forward return realises during or after the OOS year.

    * ``target_horizon="quarterly"`` — forward returns realise one quarter
      ahead; last safe training quarter is Q3 of ``oos_year − 1``.
    * ``target_horizon="annual"`` — forward returns realise four quarters
      ahead; last safe training quarter is Q4 of ``oos_year − 2``.

    Parameters
    ----------
    feature_panel
        Optional pre-computed ``(X, y, period_labels)`` covering all quarters
        (from :func:`build_regression_feature_panel`).  When provided, the
        relevant quarters are sliced out instead of rebuilding features from
        scratch — much faster when looping over multiple OOS years.
    target_horizon
        ``"quarterly"`` (default) for 1-quarter forward returns, or
        ``"annual"`` for 12-month forward returns.
    cs_normalize
        Whether cross-sectional z-score normalisation was applied to targets
        during training.  Included in the cache key so that cs-norm and
        non-cs-norm models are stored separately.

    Returns
    -------
    (RegressionTrainResult, from_cache)
    """
    if target_horizon not in ("quarterly", "annual"):
        raise ValueError(
            f"target_horizon must be 'quarterly' or 'annual', got {target_horizon!r}"
        )

    import joblib

    use_eulerpool = eulerpool_quarterly is not None
    cache_path = _regression_wf_cache_path(
        oos_year,
        target_horizon,
        cs_normalize=cs_normalize,
        use_eulerpool=use_eulerpool,
        publication_lag_days=publication_lag_days,
        cutoff_shift_days=cutoff_shift_days,
    )
    chash = _regression_wf_cache_hash(
        ohlcv,
        oos_year,
        target_horizon,
        cs_normalize=cs_normalize,
        use_eulerpool=use_eulerpool,
        publication_lag_days=publication_lag_days,
        cutoff_shift_days=cutoff_shift_days,
    )

    if use_cache and cache_path.exists():
        try:
            bundle = joblib.load(cache_path)
            if (
                isinstance(bundle, dict)
                and bundle.get("config_hash") == chash
                and bundle.get("regression_result") is not None
            ):
                log.info(
                    "Regression WF OOS %d: cached model loaded (%s)",
                    oos_year,
                    cache_path,
                )
                return bundle["regression_result"], True
            log.info(
                "Regression WF OOS %d: cache hash mismatch — retraining.",
                oos_year,
            )
        except Exception as exc:
            log.warning(
                "Regression WF OOS %d: cache load failed (%s) — retraining.",
                oos_year,
                exc,
            )

    # Quarterly: forward return realises 1 quarter ahead → last safe training
    #   quarter is Q3 of oos_year-1 (realises end of Q4, before OOS).
    # Annual: forward return realises 4 quarters ahead → last safe training
    #   quarter is Q4 of oos_year-2 (realises Q4 of oos_year-1, before OOS).
    if target_horizon == "annual":
        period_cutoff = f"{oos_year - 1}Q1"
    else:
        period_cutoff = f"{oos_year}Q1"

    if feature_panel is not None:
        X_all, y_all, pl_all = feature_panel
        mask = pl_all < period_cutoff
        X = X_all[mask].copy()
        y = y_all[mask].copy()
        period_labels = pl_all[mask].copy()
        log.info(
            "Regression WF OOS %d [%s]: sliced %d samples (%d quarters) from pre-built panel",
            oos_year,
            target_horizon,
            len(X),
            len(set(period_labels)),
        )
    else:
        if target_horizon == "annual":
            end_quarter = f"{oos_year - 2}-Q4"
            compute_fwd = compute_annual_forward_returns
        else:
            end_quarter = f"{oos_year - 1}-Q3"
            compute_fwd = compute_quarterly_forward_returns
        log.info(
            "Regression WF OOS %d [%s]: computing forward returns %s … %s",
            oos_year,
            target_horizon,
            _REGRESSION_START_QUARTER,
            end_quarter,
        )
        fwd_df = compute_fwd(
            ohlcv,
            _REGRESSION_START_QUARTER,
            end_quarter,
            cutoff_shift_days=cutoff_shift_days,
        )
        if fwd_df.empty:
            raise ValueError(
                f"No forward returns for quarters "
                f"{_REGRESSION_START_QUARTER}–{end_quarter}"
            )
        X, y, period_labels = build_regression_feature_panel(
            ohlcv, fundamentals, fwd_df,
            eulerpool_quarterly=eulerpool_quarterly,
            eulerpool_profiles=eulerpool_profiles,
            publication_lag_days=publication_lag_days,
            cutoff_shift_days=cutoff_shift_days,
            min_daily_volume_chf=config.MIN_DAILY_VOLUME_CHF,
        )

    n_quarters = len(set(period_labels))
    log.info(
        "Regression WF OOS %d [%s]: training on %d samples (%d quarters, %d features)",
        oos_year,
        target_horizon,
        len(X),
        n_quarters,
        X.shape[1],
    )

    result = train_regressor(
        X,
        y,
        period_labels=period_labels,
        random_state=config.RANDOM_SEED,
        **_REGRESSION_TRAIN_KWARGS,
    )

    result = refit_regressor_full(X, y, result, random_state=config.RANDOM_SEED)

    if use_cache:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(
                {"config_hash": chash, "regression_result": result},
                cache_path,
            )
            log.info(
                "Regression WF OOS %d: cache saved to %s",
                oos_year,
                cache_path,
            )
        except Exception as exc:
            log.warning(
                "Regression WF OOS %d: could not save cache: %s",
                oos_year,
                exc,
            )

    return result, False


def _merge_quarterly_forward_results(
    results: list[QuarterlyForwardResult],
) -> QuarterlyForwardResult:
    """Combine per-year :class:`QuarterlyForwardResult` objects into one."""
    per_year: dict = {}
    q_dfs: list[pd.DataFrame] = []
    t_dfs: list[pd.DataFrame] = []
    total_costs = 0.0

    for r in results:
        per_year.update(r.per_year)
        if not r.quarterly_detail.empty:
            q_dfs.append(r.quarterly_detail)
        if not r.turnover_log.empty:
            t_dfs.append(r.turnover_log)
        total_costs += r.total_costs_bps

    return QuarterlyForwardResult(
        per_year=per_year,
        quarterly_detail=(
            pd.concat(q_dfs, ignore_index=True) if q_dfs else pd.DataFrame()
        ),
        turnover_log=(
            pd.concat(t_dfs, ignore_index=True) if t_dfs else pd.DataFrame()
        ),
        total_costs_bps=total_costs,
        rebalance_freq=results[0].rebalance_freq if results else 1,
    )


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


def load_data(
    *, force_refresh_ohlcv: bool = False,
) -> tuple[
    dict[str, pd.DataFrame],
    dict[str, dict],
    dict[str, list[dict]],
    dict[str, dict],
]:
    """Load OHLCV, yfinance fundamentals, and Eulerpool PIT quarterly data.

    Returns ``(ohlcv, fundamentals, eulerpool_quarterly, eulerpool_profiles)``.

    The OHLCV universe is **not** statically filtered; dynamic per-cutoff
    liquidity gating is applied inside :func:`build_regression_feature_panel`
    via :func:`~universe.filter_liquid_at_cutoff`.
    """
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
    if smi_ohlcv is not None and not smi_ohlcv.empty:
        ohlcv[MACRO_BENCHMARK_TICKER] = smi_ohlcv
    log.info("Universe (no static filter; dynamic per cutoff): %d tickers", len(ohlcv))

    from src.eulerpool_fundamentals import fetch_all_quarterly

    stock_tickers = [t for t in ohlcv if t != MACRO_BENCHMARK_TICKER]
    eulerpool_q, eulerpool_profiles = fetch_all_quarterly(stock_tickers)
    n_with_data = sum(1 for v in eulerpool_q.values() if v)
    log.info(
        "Eulerpool PIT coverage: %d/%d tickers with quarterly data",
        n_with_data,
        len(stock_tickers),
    )

    fundamentals: dict[str, dict] = {}
    n_no_eulerpool = sum(1 for t in stock_tickers if not eulerpool_q.get(t))
    if n_no_eulerpool > 0:
        log.info(
            "Loading yfinance fundamentals as fallback for %d tickers without Eulerpool data",
            n_no_eulerpool,
        )
        for t in stock_tickers:
            if not eulerpool_q.get(t):
                fundamentals[t] = load_fundamentals(t, cache_dir=cache)
            else:
                fundamentals[t] = {}
    else:
        log.info("All tickers covered by Eulerpool — skipping yfinance fundamentals")
        fundamentals = {t: {} for t in ohlcv}

    return ohlcv, fundamentals, eulerpool_q, eulerpool_profiles


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
    walk_forward_training: bool = True,
) -> dict:
    """Train regime-aware models, then evaluate at three rebalancing frequencies.

    Returns a dict keyed by frequency label (``quarterly``, ``semi_annual``,
    ``annual``) → :class:`~backtest.QuarterlyForwardResult`, plus the trained
    ``regime_collection`` (or ``regime_collections``) and elapsed time.

    Parameters
    ----------
    max_position_weight
        Cap on any single position's weight (default 0.30 = 30%).
    use_cache
        If True, load/save regime models via cache.
    walk_forward_training
        If True (default), train a **separate** ``RegimeModelCollection`` per
        OOS year using only quarters with ``q_end < OOS-year-start``.  This
        eliminates training-leakage (future labels never enter training).
        If False, train ONE global model on all quarters (legacy behaviour).
    """
    smi = ohlcv.get(MACRO_BENCHMARK_TICKER)
    if smi is None or smi.empty:
        raise ValueError(
            f"Quarterly rebalancing validation requires {MACRO_BENCHMARK_TICKER!r} "
            "in OHLCV.",
        )

    t0 = time.time()
    results: dict[str, object] = {}

    if walk_forward_training:
        log.info(
            "Quarterly validation: walk-forward training (separate model per OOS year)…",
        )
        regime_collections: dict[int, Any] = {}
        for yr in REGIME_VALIDATION_OOS_YEARS:
            rc, _cached = train_walk_forward_regime_collection(
                ohlcv,
                fundamentals,
                yr,
                use_cache=use_cache,
            )
            regime_collections[yr] = rc

        for freq_label, freq_val in REBALANCE_FREQS.items():
            log.info(
                "Quarterly validation (walk-forward): freq=%s (rebalance_freq=%d) …",
                freq_label,
                freq_val,
            )
            per_year_results: list[QuarterlyForwardResult] = []
            for yr in REGIME_VALIDATION_OOS_YEARS:
                qfr = evaluate_forward_quarterly_regime_aware(
                    ohlcv,
                    fundamentals,
                    regime_collection=regime_collections[yr],
                    oos_years=[yr],
                    costs_bps=costs_bps,
                    min_winners=min_winners,
                    rebalance_freq=freq_val,
                    hysteresis_rule=hysteresis_rule,
                    max_position_weight=max_position_weight,
                )
                per_year_results.append(qfr)
            results[freq_label] = _merge_quarterly_forward_results(per_year_results)

        results["regime_collections"] = regime_collections
    else:
        log.info("Quarterly validation: global training (all quarters)…")
        regime_collection, _from_cache = load_or_train_regime_collection(
            ohlcv,
            fundamentals,
            use_cache=use_cache,
        )

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

        results["regime_collection"] = regime_collection

    elapsed = time.time() - t0
    log.info("Quarterly validation finished in %.0f seconds", elapsed)

    results["elapsed_s"] = elapsed
    return results


def run_quarterly_regression_validation(
    ohlcv: dict[str, pd.DataFrame],
    fundamentals: dict[str, dict],
    *,
    costs_bps: float = 40.0,
    top_n: int = 5,
    hysteresis_buffer: int = 2,
    max_position_weight: float | None = 0.30,
    use_cache: bool = False,
    cs_normalize: bool = False,
    target_horizon: str = "quarterly",
    eulerpool_quarterly: dict[str, list[dict]] | None = None,
    eulerpool_profiles: dict[str, dict] | None = None,
    publication_lag_days: int = 0,
    cutoff_shift_days: int = 0,
    freq_filter: str | None = None,
) -> dict:
    """Walk-forward regression evaluation at selected rebalancing frequencies.

    For each OOS year in :data:`REGIME_VALIDATION_OOS_YEARS`, trains a
    single regression model on forward returns, then evaluates via
    :func:`evaluate_forward_quarterly_regression`.

    Parameters
    ----------
    cs_normalize
        If True, cross-sectionally z-score the target per cutoff date before
        training (peer-relative targets).
    target_horizon
        ``"quarterly"`` (default) for 1-quarter forward returns, or
        ``"annual"`` for 12-month forward returns.  Annual targets shift
        the walk-forward embargo by 3 extra quarters to prevent leakage.
    publication_lag_days
        Shifts the PIT fundamental cutoff backward (Eulerpool); passed to
        feature building and :func:`~backtest.evaluate_forward_quarterly_regression`.
    cutoff_shift_days
        Forward shift (calendar days) for training labels and evaluation
        cutoffs; passed to :func:`~regression_targets.compute_quarterly_forward_returns`
        / :func:`~regression_targets.compute_annual_forward_returns`,
        :func:`build_regression_feature_panel`, and
        :func:`~backtest.evaluate_forward_quarterly_regression`.
    freq_filter
        If given (e.g. ``"semi_annual"``), only evaluate that single
        frequency instead of all three.

    Returns a dict keyed by frequency label -> :class:`~backtest.QuarterlyForwardResult`,
    plus ``regression_results`` and ``elapsed_s``.
    """
    t0 = time.time()

    max_oos = max(REGIME_VALIDATION_OOS_YEARS)
    if target_horizon == "annual":
        end_quarter = f"{max_oos - 2}-Q4"
        compute_fwd = compute_annual_forward_returns
    else:
        end_quarter = f"{max_oos - 1}-Q3"
        compute_fwd = compute_quarterly_forward_returns
    log.info(
        "Regression validation [%s]: computing forward returns %s … %s",
        target_horizon,
        _REGRESSION_START_QUARTER,
        end_quarter,
    )
    fwd_df = compute_fwd(
        ohlcv,
        _REGRESSION_START_QUARTER,
        end_quarter,
        cutoff_shift_days=cutoff_shift_days,
    )
    if fwd_df.empty:
        raise ValueError(
            f"No forward returns for {_REGRESSION_START_QUARTER}–{end_quarter}"
        )

    log.info("Regression validation: building feature panel …")
    feature_panel = build_regression_feature_panel(
        ohlcv,
        fundamentals,
        fwd_df,
        cs_normalize=cs_normalize,
        eulerpool_quarterly=eulerpool_quarterly,
        eulerpool_profiles=eulerpool_profiles,
        publication_lag_days=publication_lag_days,
        cutoff_shift_days=cutoff_shift_days,
        min_daily_volume_chf=config.MIN_DAILY_VOLUME_CHF,
    )

    regression_results: dict[int, RegressionTrainResult] = {}
    for yr in REGIME_VALIDATION_OOS_YEARS:
        reg_result, _cached = train_walk_forward_regressor(
            ohlcv,
            fundamentals,
            yr,
            use_cache=use_cache,
            feature_panel=feature_panel,
            target_horizon=target_horizon,
            eulerpool_quarterly=eulerpool_quarterly,
            eulerpool_profiles=eulerpool_profiles,
            cs_normalize=cs_normalize,
            publication_lag_days=publication_lag_days,
            cutoff_shift_days=cutoff_shift_days,
        )
        regression_results[yr] = reg_result

    freqs_to_eval = REBALANCE_FREQS
    if freq_filter is not None:
        if freq_filter not in REBALANCE_FREQS:
            raise ValueError(
                f"Unknown freq_filter={freq_filter!r}; "
                f"choose from {list(REBALANCE_FREQS)}"
            )
        freqs_to_eval = {freq_filter: REBALANCE_FREQS[freq_filter]}
        log.info("Frequency filter: evaluating only %s", freq_filter)

    results: dict[str, object] = {}
    for freq_label, freq_val in freqs_to_eval.items():
        log.info(
            "Regression validation: freq=%s (rebalance_freq=%d) …",
            freq_label,
            freq_val,
        )
        per_year_results: list[QuarterlyForwardResult] = []
        for yr in REGIME_VALIDATION_OOS_YEARS:
            qfr = evaluate_forward_quarterly_regression(
                ohlcv,
                fundamentals,
                regression_result=regression_results[yr],
                oos_years=[yr],
                costs_bps=costs_bps,
                top_n=top_n,
                rebalance_freq=freq_val,
                hysteresis_buffer=hysteresis_buffer,
                max_position_weight=max_position_weight,
                publication_lag_days=publication_lag_days,
                cutoff_shift_days=cutoff_shift_days,
            )
            per_year_results.append(qfr)
        results[freq_label] = _merge_quarterly_forward_results(per_year_results)

    results["regression_results"] = regression_results

    _save_wf_evaluation_results(
        results, publication_lag_days, costs_bps, top_n,
    )

    elapsed = time.time() - t0
    log.info("Regression validation finished in %.0f seconds", elapsed)
    results["elapsed_s"] = elapsed
    return results


def _quarterly_regression_ic(ftr: Any) -> tuple[float, float]:
    """Mean IC and IC std per OOS year (from regression eval; not classification accuracy)."""
    cls = ftr.classification
    ic = float(cls.get("ic", float("nan")))
    ic_std = float(cls.get("ic_std", float("nan")))
    return ic, ic_std


def _run_regression_leakage_checks(
    regression_results: dict[int, RegressionTrainResult],
    n_shuffles: int = 200,
) -> list[dict[str, Any]]:
    """Walk-forward leakage diagnostics for regression models.

    For each OOS year verifies:
    1. **Embargo**: last training quarter strictly before OOS year.
    2. **In-sample IC**: Spearman rho on training data (should be moderate, not 1.0).
    3. **Holdout IC**: Spearman rho on the time-series holdout split.
    4. **Shuffled-label IC**: Mean |IC| when holdout actuals are permuted
       (should collapse to ~0; a high value signals structural leakage).
    5. **IS/HO gap**: Large gap flags overfitting; negative HO IC flags
       broken signal.
    """
    from scipy.stats import spearmanr
    from src.regression_model import predict_returns

    rng = np.random.RandomState(42)
    report: list[dict[str, Any]] = []

    for yr in sorted(regression_results):
        rr = regression_results[yr]

        train_periods = (
            sorted(set(rr.period_labels_train))
            if rr.period_labels_train is not None
            else []
        )
        latest_train = train_periods[-1] if train_periods else "N/A"
        embargo_ok = latest_train < f"{yr}Q1" if latest_train != "N/A" else False

        is_preds = predict_returns(rr, rr.X_train)
        is_ic = float(spearmanr(is_preds, rr.y_train).statistic)

        ho_ic = float("nan")
        mean_shuf_ic = float("nan")
        if rr.X_test is not None and len(rr.X_test):
            ho_preds = predict_returns(rr, rr.X_test)
            ho_ic = float(spearmanr(ho_preds, rr.y_test).statistic)

            shuffled_ics = []
            for _ in range(n_shuffles):
                y_shuf = rng.permutation(rr.y_test)
                shuf_ic, _ = spearmanr(ho_preds, y_shuf)
                shuffled_ics.append(abs(shuf_ic))
            mean_shuf_ic = float(np.mean(shuffled_ics))

        entry: dict[str, Any] = {
            "oos_year": yr,
            "training_quarters": f"{train_periods[0]}–{latest_train}" if train_periods else "N/A",
            "n_train_quarters": len(train_periods),
            "n_train_samples": len(rr.X_train),
            "embargo_ok": embargo_ok,
            "in_sample_ic": round(is_ic, 4),
            "holdout_ic": round(ho_ic, 4),
            "shuffled_ic_mean": round(mean_shuf_ic, 4),
        }
        report.append(entry)

        status = "✅" if embargo_ok else "⚠️"
        log.info(
            "Leakage [OOS %d] embargo=%s  train=%s–%s (%d Q, %d samples)  "
            "IS-IC=%.3f  HO-IC=%.3f  shuffled-|IC|=%.4f",
            yr, status, train_periods[0] if train_periods else "?",
            latest_train, len(train_periods), len(rr.X_train),
            is_ic, ho_ic, mean_shuf_ic,
        )

    return report


def _print_leakage_report(report: list[dict[str, Any]]) -> str:
    """Pretty-print regression leakage diagnostics."""
    lines: list[str] = []
    sep = "=" * 72

    lines.append(sep)
    lines.append("  REGRESSION WALK-FORWARD LEAKAGE DIAGNOSTICS")
    lines.append(sep)

    all_embargo_ok = all(r["embargo_ok"] for r in report)
    lines.append(f"\n  Embargo check: {'✅ ALL OK' if all_embargo_ok else '⚠️  VIOLATIONS DETECTED'}")

    lines.append(
        f"\n  {'year':>6s}  {'embargo':>8s}  {'train window':>18s}  "
        f"{'#Q':>4s}  {'#samp':>6s}  {'IS-IC':>7s}  {'HO-IC':>7s}  "
        f"{'shuf|IC|':>9s}"
    )
    lines.append("  " + "-" * 78)
    for r in report:
        emb = "✅" if r["embargo_ok"] else "⚠️"
        lines.append(
            f"  {r['oos_year']:>6d}  {emb:>8s}  {r['training_quarters']:>18s}  "
            f"{r['n_train_quarters']:>4d}  {r['n_train_samples']:>6d}  "
            f"{r['in_sample_ic']:>7.4f}  {r['holdout_ic']:>7.4f}  "
            f"{r['shuffled_ic_mean']:>9.4f}"
        )

    mean_shuf = np.mean([r["shuffled_ic_mean"] for r in report if np.isfinite(r["shuffled_ic_mean"])])
    mean_ho = np.mean([r["holdout_ic"] for r in report if np.isfinite(r["holdout_ic"])])
    mean_is = np.mean([r["in_sample_ic"] for r in report if np.isfinite(r["in_sample_ic"])])

    lines.append(f"\n  Mean IS IC:         {mean_is:.4f}")
    lines.append(f"  Mean holdout IC:    {mean_ho:.4f}")
    lines.append(f"  Mean shuffled |IC|: {mean_shuf:.4f}")
    lines.append(f"  IS→HO gap:          {mean_is - mean_ho:.4f}")

    if mean_shuf < 0.05:
        lines.append("  → ✅ Shuffled IC near zero — no structural leakage detected.")
    else:
        lines.append("  → ⚠️  Shuffled IC elevated — possible structural artefact.")

    if mean_is - mean_ho > 0.3:
        lines.append("  → ⚠️  Large IS/HO gap — model may be overfitting.")
    else:
        lines.append("  → ✅ IS/HO gap moderate — no severe overfitting.")

    lines.append(sep)

    text = "\n".join(lines)
    print(text)
    return text


def _save_wf_evaluation_results(
    results: dict,
    publication_lag_days: int,
    costs_bps: float,
    top_n: int,
) -> None:
    """Persist walk-forward evaluation results so the Dashboard can read them directly."""
    out: dict[str, Any] = {}
    for freq_label, freq_val in REBALANCE_FREQS.items():
        qfr = results.get(freq_label)
        if qfr is None:
            continue
        per_year_ser: dict[str, Any] = {}
        for yr, ftr in qfr.per_year.items():
            per_year_ser[str(yr)] = {
                "long_only": ftr.long_only,
                "benchmark": ftr.benchmark,
                "costs_bps": ftr.costs_bps,
            }
        quarterly_detail = (
            qfr.quarterly_detail.to_dict(orient="records")
            if qfr.quarterly_detail is not None and not qfr.quarterly_detail.empty
            else []
        )
        out[freq_label] = {
            "per_year": per_year_ser,
            "quarterly_detail": quarterly_detail,
            "total_costs_bps": qfr.total_costs_bps,
            "rebalance_freq": qfr.rebalance_freq,
        }

    meta = {
        "publication_lag_days": publication_lag_days,
        "costs_bps": costs_bps,
        "top_n": top_n,
        "oos_years": list(REGIME_VALIDATION_OOS_YEARS),
        "generated_at": datetime.now().isoformat(),
    }
    payload = {"meta": meta, "frequencies": out}

    dest = config.DATA_DIR / "cache" / "wf_evaluation_results.json"
    dest.parent.mkdir(parents=True, exist_ok=True)

    def _json_safe(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_json_safe(x) for x in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating, float)) and np.isnan(obj):
            return None
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(dest, "w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, indent=2)
    log.info("Saved walk-forward evaluation results to %s", dest)


def print_quarterly_rebalance_report(result: dict, costs_bps: float = 40.0) -> str:
    """Format the 3-frequency comparison table, IC summary, and per-year detail.

    Shows **Spearman IC** (predicted vs. realized 1-month forward returns at each
    cutoff) and mean IC volatility across cutoffs. Does **not** print classifier
    metrics (accuracy, F1, confusion matrix).
    """
    lines: list[str] = []

    def p(s: str = "") -> None:
        lines.append(s)

    sep = "=" * 72
    p(sep)
    p("  REGRESSION QUARTERLY VALIDATION (Hysteresis + Transaction Costs)")
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
        ics: list[float] = []
        ic_stds: list[float] = []
        prec_stricts: list[float] = []
        prec_quartiles: list[float] = []
        avg_ranks: list[float] = []

        for yr in sorted(per_year):
            ftr = per_year[yr]
            lc = ftr.long_only.get("cumulative_return", float("nan"))
            bc = ftr.benchmark.get("cumulative_return", float("nan"))
            sh = ftr.long_only.get("sharpe_ratio", float("nan"))
            md = ftr.long_only.get("max_drawdown", float("nan"))
            ic, ic_sd = _quarterly_regression_ic(ftr)
            cls = ftr.classification
            ps = float(cls.get("precision_at_n", float("nan")))
            pq = float(cls.get("precision_top_quartile", float("nan")))
            ar = float(cls.get("avg_actual_rank", float("nan")))
            if np.isfinite(lc):
                long_cums.append(lc)
            if np.isfinite(bc):
                bm_cums.append(bc)
            if np.isfinite(sh):
                sharpes.append(sh)
            if np.isfinite(md):
                max_dds.append(md)
            if np.isfinite(ic):
                ics.append(ic)
            if np.isfinite(ic_sd):
                ic_stds.append(ic_sd)
            if np.isfinite(ps):
                prec_stricts.append(ps)
            if np.isfinite(pq):
                prec_quartiles.append(pq)
            if np.isfinite(ar):
                avg_ranks.append(ar)
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
        avg_ic = float(np.mean(ics)) if ics else float("nan")
        avg_ic_std = float(np.mean(ic_stds)) if ic_stds else float("nan")
        avg_prec_strict = float(np.mean(prec_stricts)) if prec_stricts else float("nan")
        avg_prec_quartile = float(np.mean(prec_quartiles)) if prec_quartiles else float("nan")
        avg_rank = float(np.mean(avg_ranks)) if avg_ranks else float("nan")

        summary_rows.append({
            "freq": fl,
            "beat_bm": f"{beat_bm}/{n_years}",
            "avg_long_cum": avg_long,
            "avg_bm_cum": avg_bm,
            "avg_turnover": avg_turnover,
            "total_costs_bps": qfr.total_costs_bps,
            "avg_sharpe": avg_sharpe,
            "avg_maxdd": avg_maxdd,
            "avg_ic": avg_ic,
            "avg_ic_std": avg_ic_std,
            "avg_prec_strict": avg_prec_strict,
            "avg_prec_quartile": avg_prec_quartile,
            "avg_rank": avg_rank,
        })

    p(f"\n--- Rebalancing-Frequenz-Vergleich (costs={costs_bps:.0f}bps one-way) ---")

    header = f"{'':>16s}"
    for sr in summary_rows:
        header += f"  {sr['freq']:>14s}"
    p(header)

    metric_keys = [
        ("beat_bm", "beat_bm", None),
        ("avg_long_cum", "long_cum", "pct"),
        ("avg_bm_cum", "bm_cum", "pct"),
        ("avg_turnover", "avg_turnover", "pct1"),
        ("total_costs_bps", "total_costs", "bps"),
        ("avg_sharpe", "sharpe", "f2"),
        ("avg_maxdd", "maxDD", "pct"),
        ("avg_ic", "IC (mean)", "f3"),
        ("avg_ic_std", "IC_std (mean)", "f3"),
        ("avg_prec_strict", "Prec@5 strict", "pct"),
        ("avg_prec_quartile", "Prec@5 Q1", "pct"),
        ("avg_rank", "avg real rank", "f1"),
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
            elif fmt == "f3":
                row += f"  {v:>14.3f}" if np.isfinite(v) else f"  {'N/A':>14s}"
            elif fmt == "f1":
                row += f"  {v:>14.1f}" if np.isfinite(v) else f"  {'N/A':>14s}"
            else:
                row += f"  {v!s:>14s}"
        p(row)

    p("")
    p("  IC: Spearman rank correlation (predicted vs. actual quarterly return).")
    p("  Prec@5 strict: fraction of our 5 picks that were actually in the real top 5.")
    p("  Prec@5 Q1: fraction of our 5 picks in the actual top quartile.")
    p("  avg real rank: mean actual rank of our 5 picks (1 = best).")

    # ── per-year detail per frequency ──
    for fl in freq_labels:
        qfr = result[fl]
        per_year = qfr.per_year
        p(f"\n--- Per-year detail: {fl} (rebalance_freq={REBALANCE_FREQS[fl]}) ---")
        detail_rows: list[str] = []
        detail_rows.append(
            f"{'year':>6s}  {'long_cum':>10s}  {'bm_cum':>10s}  "
            f"{'sharpe':>8s}  {'maxDD':>8s}  {'IC':>8s}  "
            f"{'P@5':>6s}  {'P@Q1':>6s}  {'rank':>6s}  "
            f"{'costs':>8s}  {'beat':>5s}"
        )
        for yr in sorted(per_year):
            ftr = per_year[yr]
            lc = ftr.long_only.get("cumulative_return", float("nan"))
            bc = ftr.benchmark.get("cumulative_return", float("nan"))
            sh = ftr.long_only.get("sharpe_ratio", float("nan"))
            md = ftr.long_only.get("max_drawdown", float("nan"))
            ic, ic_sd = _quarterly_regression_ic(ftr)
            cls = ftr.classification
            ps = float(cls.get("precision_at_n", float("nan")))
            pq = float(cls.get("precision_top_quartile", float("nan")))
            ar = float(cls.get("avg_actual_rank", float("nan")))
            cost = ftr.costs_bps
            beat = "✓" if np.isfinite(lc) and np.isfinite(bc) and lc > bc else "✗"
            ps_s = f"{ps:.0%}" if np.isfinite(ps) else "N/A"
            pq_s = f"{pq:.0%}" if np.isfinite(pq) else "N/A"
            ar_s = f"{ar:.1f}" if np.isfinite(ar) else "N/A"
            detail_rows.append(
                f"{yr:>6d}  {_fp(lc):>10s}  {_fp(bc):>10s}  "
                f"{_ff(sh):>8s}  {_ff(md):>8s}  {_ff(ic, 3):>8s}  "
                f"{ps_s:>6s}  {pq_s:>6s}  {ar_s:>6s}  "
                f"{cost:>8.0f}  {beat:>5s}"
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


# ── Rolling-Annual Evaluation ────────────────────────────────────────────────


_QUARTER_CUTOFF_MONTHS = {1: 12, 2: 3, 3: 6, 4: 9}
_QUARTER_LABELS = {1: "Q1 (Jan)", 2: "Q2 (Apr)", 3: "Q3 (Jul)", 4: "Q4 (Okt)"}


def _merged_trading_index(ohlcv: dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    """Union of all OHLCV calendar dates (for snapping shifted cutoffs)."""
    dates: list[pd.Timestamp] = []
    for df in ohlcv.values():
        if df is None or df.empty or "Close" not in df.columns:
            continue
        dates.extend(pd.to_datetime(df.index).tolist())
    if not dates:
        return pd.DatetimeIndex([])
    return pd.DatetimeIndex(sorted(set(dates)))


def _daily_close_matrix_range(
    ohlcv_by_ticker: dict[str, pd.DataFrame],
    tickers: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Aligned daily close prices for *tickers* between arbitrary dates."""
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


def evaluate_rolling_annual(
    ohlcv: dict[str, pd.DataFrame],
    fundamentals: dict[str, dict],
    regression_results: dict[int, RegressionTrainResult],
    *,
    oos_years: list[int],
    top_n: int = 5,
    costs_bps: float = 40.0,
    max_position_weight: float | None = 0.30,
    cutoff_shift_days: int = 0,
) -> list[dict[str, Any]]:
    """Evaluate 12-month buy-and-hold from each quarterly cutoff.

    For every OOS year and each of the 4 quarterly cutoffs, the model
    predicts returns and selects the top-*top_n* stocks (no hysteresis —
    fresh picks each time).  The picks are held for 12 months from the
    first trading day after the cutoff.  Returns a list of per-window
    result dicts.

    When *cutoff_shift_days* > 0, each calendar cutoff is shifted forward by
    that many days and snapped to the last trading day on or before that date
    (same convention as :func:`~regression_targets.compute_quarterly_forward_returns`).
    For a non-zero shift, Q1 windows use the same day-after-cutoff / 12-month
    hold logic as other quarters (instead of the calendar Jan–Dec OOS year).
    """
    from scipy.stats import spearmanr
    try:
        from src.regression_model import predict_returns
    except ImportError:
        from regression_model import predict_returns

    results: list[dict[str, Any]] = []
    trade_cal = _merged_trading_index(ohlcv)

    for yr in oos_years:
        reg_result = regression_results[yr]

        for entry_q in (1, 2, 3, 4):
            cm = _QUARTER_CUTOFF_MONTHS[entry_q]
            if entry_q == 1:
                cutoff_date = f"{yr - 1}-12-31"
                if cutoff_shift_days > 0:
                    shifted = pd.Timestamp(cutoff_date) + pd.Timedelta(
                        days=cutoff_shift_days,
                    )
                    snapped = _last_trading_day_on_or_before(trade_cal, shifted)
                    if snapped is None:
                        continue
                    cutoff_date = snapped.strftime("%Y-%m-%d")
                    hold_start = snapped + pd.Timedelta(days=1)
                    hold_end = (
                        hold_start + pd.DateOffset(months=12) - pd.Timedelta(days=1)
                    )
                else:
                    hold_start = pd.Timestamp(f"{yr}-01-01")
                    hold_end = pd.Timestamp(f"{yr}-12-31")
            else:
                cutoff_year = yr
                cutoff_date = (
                    pd.Timestamp(year=cutoff_year, month=cm, day=1)
                    + pd.offsets.MonthEnd(0)
                ).strftime("%Y-%m-%d")
                if cutoff_shift_days > 0:
                    shifted = pd.Timestamp(cutoff_date) + pd.Timedelta(
                        days=cutoff_shift_days,
                    )
                    snapped = _last_trading_day_on_or_before(trade_cal, shifted)
                    if snapped is None:
                        continue
                    cutoff_date = snapped.strftime("%Y-%m-%d")
                    hold_start = snapped + pd.Timedelta(days=1)
                    hold_end = (
                        hold_start + pd.DateOffset(months=12) - pd.Timedelta(days=1)
                    )
                else:
                    hold_start = pd.Timestamp(cutoff_date) + pd.Timedelta(days=1)
                    hold_end = (
                        hold_start + pd.DateOffset(months=12) - pd.Timedelta(days=1)
                    )

            X_oos = build_oos_features(ohlcv, fundamentals, cutoff_date=cutoff_date)
            pred = predict_returns(reg_result, X_oos)

            top = pred.dropna().sort_values(ascending=False).head(top_n)
            picks = list(top.index)
            if not picks:
                continue

            close = _daily_close_matrix_range(ohlcv, picks, hold_start, hold_end)
            if close.empty or len(close) < 2:
                continue

            pred_score = pred.reindex(picks) - pred.reindex(picks).min() + 1e-12
            w = _capped_proba_weights(pred_score, picks, max_position_weight)

            daily_ret = close.pct_change().dropna(how="all")
            portfolio_ret = (daily_ret[w.index.intersection(daily_ret.columns)]
                             * w.reindex(daily_ret.columns).fillna(0)).sum(axis=1)

            cost_entry = costs_bps / 10_000.0
            cost_exit = costs_bps / 10_000.0
            if len(portfolio_ret) >= 2:
                portfolio_ret.iloc[0] -= cost_entry
                portfolio_ret.iloc[-1] -= cost_exit

            cum_ret = float((1 + portfolio_ret).prod() - 1)
            n_days = len(portfolio_ret)
            ann_factor = np.sqrt(252)
            sharpe = (
                float(portfolio_ret.mean() / portfolio_ret.std() * ann_factor)
                if portfolio_ret.std() > 0 else float("nan")
            )
            running_max = (1 + portfolio_ret).cumprod().cummax()
            drawdown = (1 + portfolio_ret).cumprod() / running_max - 1
            max_dd = float(drawdown.min())

            bm_tickers = [t for t in pred.dropna().index if t in ohlcv]
            bm_close = _daily_close_matrix_range(
                ohlcv, bm_tickers, hold_start, hold_end,
            )
            if not bm_close.empty and len(bm_close) >= 2:
                bm_ret = bm_close.pct_change().dropna(how="all").mean(axis=1)
                bm_cum = float((1 + bm_ret).prod() - 1)
            else:
                bm_cum = float("nan")

            actual_12m = close.iloc[-1] / close.iloc[0] - 1
            common_ic = pred.dropna().index.intersection(actual_12m.dropna().index)
            if len(common_ic) >= 5:
                ic_val, _ = spearmanr(
                    pred.reindex(common_ic).values,
                    actual_12m.reindex(common_ic).values,
                )
                ic_val = float(ic_val)
            else:
                ic_val = float("nan")

            results.append({
                "oos_year": yr,
                "entry_q": entry_q,
                "entry_label": _QUARTER_LABELS[entry_q],
                "cutoff": cutoff_date,
                "hold_start": hold_start.strftime("%Y-%m-%d"),
                "hold_end": hold_end.strftime("%Y-%m-%d"),
                "actual_end": close.index[-1].strftime("%Y-%m-%d"),
                "n_days": n_days,
                "picks": picks,
                "long_cum": cum_ret,
                "bm_cum": bm_cum,
                "sharpe": sharpe,
                "max_dd": max_dd,
                "ic": ic_val,
                "beat_bm": cum_ret > bm_cum if np.isfinite(bm_cum) else False,
                "costs_bps": 2 * costs_bps,
            })

    log.info(
        "Rolling-annual: %d windows evaluated (%d entry points × %d OOS years)",
        len(results),
        4,
        len(oos_years),
    )
    return results


def print_rolling_annual_report(windows: list[dict[str, Any]]) -> str:
    """Format a rolling-annual summary: per-entry-quarter aggregates and per-window detail."""
    lines: list[str] = []

    def p(s: str = "") -> None:
        lines.append(s)

    sep = "=" * 72
    p(sep)
    p("  ROLLING-ANNUAL EVALUATION (12-Monats-Hold ab jedem Quartals-Cutoff)")
    p(sep)

    if not windows:
        p("  (Keine Ergebnisse.)")
        text = "\n".join(lines)
        print(text)
        return text

    df = pd.DataFrame(windows)

    # ── aggregate by entry quarter ──
    p("\n--- Aggregat nach Einstiegs-Quartal ---")
    agg_rows: list[str] = []
    agg_rows.append(
        f"{'entry':>12s}  {'N':>4s}  {'beat_bm':>8s}  {'Ø long':>10s}  "
        f"{'Ø bm':>10s}  {'Ø sharpe':>8s}  {'Ø maxDD':>8s}  {'Ø IC':>8s}"
    )
    for eq in sorted(df["entry_q"].unique()):
        sub = df[df["entry_q"] == eq]
        n = len(sub)
        n_beat = int(sub["beat_bm"].sum())
        avg_long = float(sub["long_cum"].mean())
        avg_bm = float(sub["bm_cum"].mean())
        avg_sh = float(sub["sharpe"].mean())
        avg_dd = float(sub["max_dd"].mean())
        avg_ic = float(sub["ic"].dropna().mean()) if sub["ic"].notna().any() else float("nan")
        label = _QUARTER_LABELS[eq]
        agg_rows.append(
            f"{label:>12s}  {n:>4d}  {n_beat:>3d}/{n:<3d}  {avg_long:>+9.1%}  "
            f"{avg_bm:>+9.1%}  {avg_sh:>8.2f}  {avg_dd:>8.1%}  {_ff(avg_ic, 3):>8s}"
        )

    all_beat = int(df["beat_bm"].sum())
    all_n = len(df)
    agg_rows.append(
        f"{'GESAMT':>12s}  {all_n:>4d}  {all_beat:>3d}/{all_n:<3d}  "
        f"{df['long_cum'].mean():>+9.1%}  {df['bm_cum'].mean():>+9.1%}  "
        f"{df['sharpe'].mean():>8.2f}  {df['max_dd'].mean():>8.1%}  "
        f"{_ff(float(df['ic'].dropna().mean()), 3):>8s}"
    )
    for r in agg_rows:
        p(r)

    # ── per-window detail ──
    p("\n--- Per-Window Detail ---")
    detail_header = (
        f"{'year':>6s}  {'entry':>12s}  {'cutoff':>12s}  {'hold_end':>12s}  "
        f"{'days':>5s}  {'long':>10s}  {'bm':>10s}  {'sharpe':>7s}  "
        f"{'maxDD':>7s}  {'IC':>7s}  {'>BM':>4s}"
    )
    p(detail_header)
    for _, row in df.sort_values(["oos_year", "entry_q"]).iterrows():
        beat = "✓" if row["beat_bm"] else "✗"
        p(
            f"{row['oos_year']:>6d}  {_QUARTER_LABELS[row['entry_q']]:>12s}  "
            f"{row['cutoff']:>12s}  {row['actual_end']:>12s}  "
            f"{row['n_days']:>5d}  {row['long_cum']:>+9.1%}  "
            f"{row['bm_cum']:>+9.1%}  {row['sharpe']:>7.2f}  "
            f"{row['max_dd']:>7.1%}  {_ff(row['ic'], 3):>7s}  {beat:>4s}"
        )

    # ── verdict ──
    p(f"\n--- Verdict ---")
    best_q = None
    best_beat = -1
    best_avg = float("-inf")
    for eq in sorted(df["entry_q"].unique()):
        sub = df[df["entry_q"] == eq]
        nb = int(sub["beat_bm"].sum())
        avg = float(sub["long_cum"].mean())
        if nb > best_beat or (nb == best_beat and avg > best_avg):
            best_q = eq
            best_beat = nb
            best_avg = avg
    p(f"  Bester Einstiegs-Zeitpunkt: {_QUARTER_LABELS.get(best_q, '?')} "
      f"({best_beat}/{len(df[df['entry_q'] == best_q])} beat BM, "
      f"Ø long cum {best_avg:+.1%})")

    text = "\n".join(lines)
    print(text)
    return text


# ── Main ─────────────────────────────────────────────────────────────────────


def _use_joblib_model_cache_from_argv(argv: list[str]) -> bool:
    """Whether to load/save joblib-trained models (regime or regression WF).

    ``--no-cache`` overrides ``--use-cache``.
    """
    if "--no-cache" in argv:
        return False
    return "--use-cache" in argv


def _pub_lag_days_from_argv(argv: list[str]) -> int:
    """Parse ``--pub-lag N`` or ``--pub-lag=N`` (publication lag for PIT fundamentals)."""
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


def _pub_shift_days_from_argv(argv: list[str]) -> int:
    """Parse ``--pub-shift N`` or ``--pub-shift=N`` (publication-lag cutoff shift)."""
    for i, a in enumerate(argv):
        if a.startswith("--pub-shift="):
            try:
                return max(0, int(a.split("=", 1)[1].strip()))
            except ValueError:
                return 0
        if a == "--pub-shift" and i + 1 < len(argv):
            try:
                return max(0, int(argv[i + 1].strip()))
            except ValueError:
                return 0
    return 0


def _freq_from_argv(argv: list[str]) -> str | None:
    """Parse ``--freq semi_annual`` or ``--freq=semi_annual``."""
    for i, a in enumerate(argv):
        if a.startswith("--freq="):
            return a.split("=", 1)[1].strip() or None
        if a == "--freq" and i + 1 < len(argv):
            return argv[i + 1].strip() or None
    return None


def main() -> None:
    t_global = time.time()
    argv = sys.argv
    refresh = "--refresh-ohlcv" in argv
    regime_only = "--regime-only" in argv
    quarterly_only = "--quarterly" in argv
    annual_target = "--annual-target" in argv
    cs_norm = "--cs-norm" in argv
    pub_lag_days = _pub_lag_days_from_argv(argv)
    pub_shift_days = _pub_shift_days_from_argv(argv)
    freq_filter = _freq_from_argv(argv)
    leakage_check = "--leakage-check" in argv
    run_regime_phase5 = "--regime-validation" in argv or regime_only
    use_model_cache = _use_joblib_model_cache_from_argv(argv)
    target_horizon = "annual" if annual_target else "quarterly"

    if refresh:
        log.info("Flag --refresh-ohlcv: ignoring Parquet cache (full re-download).")
    if regime_only:
        log.info("Flag --regime-only: running Phase 5 regime validation only.")
    elif quarterly_only:
        horizon_info = (
            f" [target_horizon={target_horizon}, cs_normalize={cs_norm}]"
        )
        log.info("Flag --quarterly: running Phase 6 regression quarterly validation only.%s", horizon_info)
    elif run_regime_phase5:
        log.info("Flag --regime-validation: will run Phase 5 after standard tests.")
    if use_model_cache:
        if quarterly_only:
            log.info(
                "Flag --use-cache: walk-forward regression models may load from %s",
                config.DATA_DIR / "cache" / "regression_wf_*.joblib",
            )
        else:
            log.info(
                "Flag --use-cache: regime models may load from %s",
                _REGIME_CACHE_PATH,
            )
    elif "--no-cache" in argv:
        log.info("Flag --no-cache: joblib model cache disabled.")
    if annual_target and quarterly_only:
        log.info("Flag --annual-target: 12-month forward-return targets (annual walk-forward embargo).")
    if cs_norm and quarterly_only:
        log.info("Flag --cs-norm: cross-sectional z-score targets per cutoff.")
    if pub_lag_days > 0 and quarterly_only:
        log.info(
            "Flag --pub-lag: PIT fundamentals use cutoff minus %d days",
            pub_lag_days,
        )
    if pub_shift_days > 0 and quarterly_only:
        log.info(
            "Flag --pub-shift: training and eval cutoffs shifted forward %d calendar days",
            pub_shift_days,
        )
    if freq_filter and quarterly_only:
        log.info("Flag --freq: evaluating only frequency '%s'", freq_filter)
    if leakage_check and quarterly_only:
        log.info("Flag --leakage-check: running walk-forward leakage diagnostics after training")

    if regime_only:
        ohlcv, fundamentals, _, _ = load_data(force_refresh_ohlcv=refresh)
        log.info("╔══ Phase 5 only: Regime validation ══╗")
        r5 = run_regime_phase5_validation(ohlcv, fundamentals, use_cache=use_model_cache)
        print_regime_phase5_report(r5)
        log.info("Total runtime: %.0f seconds", time.time() - t_global)
        return

    if quarterly_only:
        ohlcv, fundamentals, eulerpool_q, eulerpool_prof = load_data(
            force_refresh_ohlcv=refresh,
        )
        freq_desc = freq_filter or "3 frequencies"
        log.info("╔══ Regression quarterly validation (walk-forward, %s) ══╗", freq_desc)
        rq = run_quarterly_regression_validation(
            ohlcv,
            fundamentals,
            use_cache=use_model_cache,
            cs_normalize=cs_norm,
            target_horizon=target_horizon,
            eulerpool_quarterly=eulerpool_q,
            eulerpool_profiles=eulerpool_prof,
            publication_lag_days=pub_lag_days,
            cutoff_shift_days=pub_shift_days,
            freq_filter=freq_filter,
        )
        print_quarterly_rebalance_report(rq)

        if leakage_check and "regression_results" in rq:
            log.info("╔══ Leakage Diagnostics ══╗")
            lk_report = _run_regression_leakage_checks(rq["regression_results"])
            _print_leakage_report(lk_report)

        log.info("╔══ Rolling-Annual Evaluation (12-Monats-Hold ab jedem Cutoff) ══╗")
        rolling = evaluate_rolling_annual(
            ohlcv,
            fundamentals,
            rq["regression_results"],
            oos_years=REGIME_VALIDATION_OOS_YEARS,
            cutoff_shift_days=pub_shift_days,
        )
        print_rolling_annual_report(rolling)

        log.info("Total runtime: %.0f seconds", time.time() - t_global)
        return

    ohlcv, fundamentals, _, _ = load_data(force_refresh_ohlcv=refresh)

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
        r5 = run_regime_phase5_validation(ohlcv, fundamentals, use_cache=use_model_cache)
        print_regime_phase5_report(r5)

    log.info("Total runtime: %.0f seconds", time.time() - t_global)


if __name__ == "__main__":
    main()
