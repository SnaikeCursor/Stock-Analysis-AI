"""Model Service — wraps src/regression_model.py, src/features.py, src/regime.py, src/backtest.py.

Loads the cached ``RegressionTrainResult`` (Lag60-SA model), generates
trading signals with regression-based ranking, exposes regime status for
informational purposes, and runs historical backtests — all without
exposing the underlying ML plumbing to the API layer.

Model variant: Lag60-SA (PIT Fundamentals + CS z-score quarterly target
+ 60-day publication lag + semi-annual rebalancing).
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.backtest import (
    build_oos_features,
    compute_portfolio_weights,
    evaluate_forward_quarterly_regression,
)
from src.features import MACRO_BENCHMARK_TICKER, build_feature_matrix
from src.regression_model import (
    RegressionTrainResult,
    predict_returns,
)
from src.regime import Regime, RegimeState, detect_regime

from backend.services.data_service import DataService

logger = logging.getLogger(__name__)

_DEFAULT_TOP_N = 5
_DEFAULT_MAX_WEIGHT = 0.30
_DEFAULT_HYSTERESIS_BUFFER = 2
_SEMI_ANNUAL_REBALANCE_FREQ = 2
_PUBLICATION_LAG_DAYS = 60
_MODEL_SUFFIX = "_cs_lag60"


def _regression_cache_path(oos_year: int) -> Path:
    """Cache path for walk-forward regression model (Lag60-SA)."""
    return config.DATA_DIR / "cache" / f"regression_wf_{oos_year}{_MODEL_SUFFIX}.joblib"


def _find_latest_regression_model() -> tuple[Path, int]:
    """Find the most recent ``regression_wf_{year}_cs_lag60.joblib`` cache file.

    Returns ``(path, oos_year)`` for the newest available model.
    """
    cache_dir = config.DATA_DIR / "cache"
    if not cache_dir.exists():
        raise FileNotFoundError(
            f"Cache directory not found at {cache_dir}. "
            "Run `python robustness_test.py --quarterly --cs-norm --pub-lag 60 --use-cache` to train."
        )

    candidates: list[tuple[Path, int]] = []
    for f in cache_dir.glob(f"regression_wf_*{_MODEL_SUFFIX}.joblib"):
        stem = f.stem
        prefix = "regression_wf_"
        if not stem.startswith(prefix) or not stem.endswith(_MODEL_SUFFIX):
            continue
        year_str = stem[len(prefix):-len(_MODEL_SUFFIX)]
        try:
            year = int(year_str)
            candidates.append((f, year))
        except ValueError:
            continue

    if not candidates:
        raise FileNotFoundError(
            f"No Lag60-SA regression model cache found in {cache_dir}. "
            "Run `python robustness_test.py --quarterly --cs-norm --pub-lag 60 --use-cache` to train."
        )

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0]


# ------------------------------------------------------------------
# DTOs returned to callers
# ------------------------------------------------------------------


@dataclass
class SignalResult:
    """Generated trading signal with portfolio composition and regime context."""

    cutoff_date: str
    regime_label: str
    regime_confidence: float
    portfolio: list[dict[str, Any]]
    portfolio_json: str
    model_phase: str = "Lag60-SA"
    rebalance_freq: str = "semi-annual"


@dataclass
class RegimeStatus:
    """Current regime snapshot (informational — not used for model selection)."""

    label: str
    confidence: float
    close: float
    sma_50: float
    sma_200: float
    sma_cross_gap: float
    sma_50_slope: float
    trend_strength: float


# ------------------------------------------------------------------
# Service
# ------------------------------------------------------------------


class ModelService:
    """Stateful service managing the Lag60-SA regression model.

    Depends on a :class:`DataService` for OHLCV, yfinance fundamentals,
    and Eulerpool PIT quarterly data.  The model is loaded lazily on
    first use from the joblib cache produced by
    ``robustness_test.py --quarterly --cs-norm --pub-lag 60 --use-cache``.
    """

    def __init__(self, data_service: DataService) -> None:
        self._data = data_service
        self._regression_result: RegressionTrainResult | None = None
        self._model_year: int | None = None
        self._backtest_cache: dict[tuple, dict[str, Any]] = {}

    @property
    def is_loaded(self) -> bool:
        return self._regression_result is not None

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def load_model(self) -> RegressionTrainResult:
        """Load the ``RegressionTrainResult`` from the joblib cache.

        Finds the most recent ``regression_wf_{year}_cs_lag60.joblib`` file.
        Raises ``FileNotFoundError`` if no cache exists — run
        ``robustness_test.py --quarterly --cs-norm --pub-lag 60 --use-cache`` first.
        """
        if self._regression_result is not None:
            return self._regression_result

        import joblib

        path, year = _find_latest_regression_model()
        bundle = joblib.load(path)

        if isinstance(bundle, RegressionTrainResult):
            result = bundle
        elif isinstance(bundle, dict) and "model" in bundle:
            result = RegressionTrainResult(**bundle)
        else:
            raise TypeError(
                f"Expected RegressionTrainResult, got {type(bundle).__name__}"
            )

        self._regression_result = result
        self._model_year = year
        logger.info(
            "Loaded RegressionTrainResult (OOS %d) from %s — %d features",
            year, path, len(result.feature_names),
        )
        return result

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def generate_signal(
        self,
        cutoff_date: str,
        *,
        top_n: int = _DEFAULT_TOP_N,
        max_weight: float = _DEFAULT_MAX_WEIGHT,
    ) -> SignalResult:
        """Build features, predict returns, and select top-N portfolio.

        The pipeline mirrors ``robustness_test.py``'s Lag60-SA evaluation:
        1. Build feature matrix at *cutoff_date* with Eulerpool PIT fundamentals
           (60-day publication lag applied)
        2. Predict cross-sectional z-scored forward returns via regression
        3. Rank tickers by predicted return (descending)
        4. Select top-N and weight proportionally to predicted returns
        5. Detect regime at *cutoff_date* (informational only)

        Parameters
        ----------
        cutoff_date
            ISO date for feature computation (no lookahead).
        top_n
            Number of tickers for the long portfolio.
        max_weight
            Maximum position weight (e.g. 0.30 = 30%).
        """
        result = self.load_model()
        self._data.ensure_data_covers(cutoff_date)
        ohlcv = self._data.ohlcv
        fundamentals = self._data.fundamentals
        eulerpool_q = self._data.eulerpool_quarterly
        eulerpool_p = self._data.eulerpool_profiles
        smi_ohlcv = self._data.get_smi_ohlcv()

        X_oos = build_oos_features(
            ohlcv,
            fundamentals,
            cutoff_date=cutoff_date,
            eulerpool_quarterly=eulerpool_q if eulerpool_q else None,
            eulerpool_profiles=eulerpool_p if eulerpool_p else None,
            publication_lag_days=_PUBLICATION_LAG_DAYS,
        )
        logger.info("OOS features: %d tickers × %d features", *X_oos.shape)

        pred = predict_returns(result, X_oos)
        pred_sorted = pred.sort_values(ascending=False)
        logger.info(
            "Top-5 predicted returns: %s",
            {t: f"{v:.4f}" for t, v in pred_sorted.head(5).items()},
        )

        selected = list(pred_sorted.head(top_n).index)

        pred_score = pred.reindex(selected) - pred.reindex(selected).min() + 1e-12
        weights = pred_score / pred_score.sum()

        if max_weight < 1.0:
            weights = _apply_max_weight_cap(weights, max_weight)

        regime_state = detect_regime(smi_ohlcv, cutoff_date)
        logger.info(
            "Regime at %s: %s (confidence=%.2f) — informational only",
            cutoff_date,
            regime_state.label.value.upper(),
            regime_state.confidence,
        )

        portfolio: list[dict[str, Any]] = []
        for ticker in weights.index:
            portfolio.append(
                {
                    "ticker": ticker,
                    "weight": round(float(weights[ticker]), 4),
                    "predicted_return": round(float(pred.get(ticker, 0.0)), 4),
                }
            )

        portfolio_json = json.dumps(portfolio, indent=2)

        logger.info(
            "Signal generated: %d positions, model=Lag60-SA (semi-annual), regime=%s",
            len(portfolio),
            regime_state.label.value,
        )

        return SignalResult(
            cutoff_date=cutoff_date,
            regime_label=regime_state.label.value,
            regime_confidence=round(regime_state.confidence, 4),
            portfolio=portfolio,
            portfolio_json=portfolio_json,
        )

    # ------------------------------------------------------------------
    # Regime status (informational — no longer drives model selection)
    # ------------------------------------------------------------------

    def get_regime_status(self, date: str) -> RegimeStatus:
        """Detect regime at *date* and return a structured snapshot.

        Regime detection is purely informational in Lag60-SA — the
        regression model does not use regime-specific sub-models.
        """
        self._data.ensure_data_covers(date)
        smi_ohlcv = self._data.get_smi_ohlcv()
        state = detect_regime(smi_ohlcv, date)

        return RegimeStatus(
            label=state.label.value,
            confidence=round(state.confidence, 4),
            close=round(state.indicators.close, 2),
            sma_50=round(state.indicators.sma_50, 2),
            sma_200=round(state.indicators.sma_200, 2),
            sma_cross_gap=round(state.indicators.sma_cross_gap, 6),
            sma_50_slope=round(state.indicators.sma_50_slope, 6),
            trend_strength=round(state.indicators.trend_strength, 6),
        )

    # ------------------------------------------------------------------
    # Historical backtest
    # ------------------------------------------------------------------

    def run_historical_backtest(
        self,
        oos_years: list[int] | None = None,
        *,
        costs_bps: float = 40.0,
        top_n: int = _DEFAULT_TOP_N,
        rebalance_freq: int = _SEMI_ANNUAL_REBALANCE_FREQ,
    ) -> dict[str, Any]:
        """Run walk-forward regression backtest and return serialisable metrics.

        Wraps :func:`evaluate_forward_quarterly_regression` from
        ``src/backtest.py`` with semi-annual rebalancing and 60-day
        publication lag (default).

        Parameters
        ----------
        oos_years
            Calendar years to evaluate; defaults to 2015–2025.
        costs_bps
            One-way transaction cost in basis points (default 40).
        top_n
            Number of long positions per rebalance.
        rebalance_freq
            Quarters between rebalances (2 = semi-annual, 1 = quarterly).
        """
        result = self.load_model()
        ohlcv = self._data.ohlcv
        fundamentals = self._data.fundamentals

        if oos_years is None:
            oos_years = list(range(2015, 2026))

        cache_key = (tuple(oos_years), costs_bps, top_n, rebalance_freq)
        if cache_key in self._backtest_cache:
            logger.info("Returning cached backtest for %s", cache_key)
            return self._backtest_cache[cache_key]

        qfr = evaluate_forward_quarterly_regression(
            ohlcv,
            fundamentals,
            regression_result=result,
            oos_years=oos_years,
            costs_bps=costs_bps,
            top_n=top_n,
            rebalance_freq=rebalance_freq,
            publication_lag_days=_PUBLICATION_LAG_DAYS,
        )

        per_year_summary: dict[int, dict[str, Any]] = {}
        for year, ftr in qfr.per_year.items():
            per_year_summary[year] = {
                "long_only": ftr.long_only,
                "benchmark": ftr.benchmark,
                "costs_bps": ftr.costs_bps,
            }

        quarterly_detail = (
            qfr.quarterly_detail.to_dict(orient="records")
            if qfr.quarterly_detail is not None and not qfr.quarterly_detail.empty
            else []
        )

        serialised = {
            "per_year": per_year_summary,
            "quarterly_detail": quarterly_detail,
            "total_costs_bps": qfr.total_costs_bps,
            "rebalance_freq": qfr.rebalance_freq,
        }
        self._backtest_cache[cache_key] = serialised
        logger.info("Cached backtest result for %s", cache_key)
        return serialised


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _apply_max_weight_cap(weights: pd.Series, cap: float) -> pd.Series:
    """Redistribute excess weight proportionally when any position exceeds *cap*."""
    w = weights.copy()
    for _ in range(20):
        above = w > cap
        if not above.any():
            break
        excess = (w[above] - cap).sum()
        w[above] = cap
        below = w[~above]
        if below.sum() > 0:
            w[~above] += below / below.sum() * excess
    w = w / w.sum()
    return w
