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
from datetime import date, datetime
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
    build_quarterly_rebalance_schedule,
    compute_portfolio_weights,
    evaluate_forward_quarterly_regression,
)
from src.features import MACRO_BENCHMARK_TICKER, build_feature_matrix
from src.regression_model import (
    RegressionTrainResult,
    predict_returns,
)
from src.regime import Regime, RegimeState, detect_regime

from backend.portfolio_json import serialize_portfolio_bundle
from backend.services.data_service import DataService

logger = logging.getLogger(__name__)

_DEFAULT_TOP_N = 5
_DEFAULT_MAX_WEIGHT = 0.30
_DEFAULT_HYSTERESIS_BUFFER = 2
_SEMI_ANNUAL_REBALANCE_FREQ = 2
_PUBLICATION_LAG_DAYS = 60
_MODEL_SUFFIX = "_cs_lag60"

BACKTEST_CACHE_PATH = config.DATA_DIR / "cache" / "backtest_results.json"


def _backtest_cache_key_to_json(key: tuple[Any, ...]) -> str:
    """Serialise ``(oos_years_tuple, costs_bps, top_n, rebalance_freq)`` for JSON object keys."""
    years, costs_bps, top_n, rebalance_freq = key
    return json.dumps([list(years), costs_bps, top_n, rebalance_freq])


def _backtest_cache_key_from_json(key_str: str) -> tuple[Any, ...]:
    """Restore cache key from :func:`_backtest_cache_key_to_json`."""
    raw = json.loads(key_str)
    return (tuple(raw[0]), float(raw[1]), int(raw[2]), int(raw[3]))


def _normalize_backtest_dict_from_disk(val: dict[str, Any]) -> dict[str, Any]:
    """Restore integer year keys in ``per_year`` after :func:`json.load` stringifies them."""
    out = dict(val)
    py = out.get("per_year")
    if isinstance(py, dict):
        fixed: dict[int, Any] = {}
        for k, v in py.items():
            try:
                fixed[int(k)] = v
            except (TypeError, ValueError):
                continue
        out["per_year"] = fixed
    return out


def _json_safe_for_disk(obj: Any) -> Any:
    """Recursively convert numpy/pandas scalars so :func:`json.dump` succeeds."""
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, dict):
        return {k: _json_safe_for_disk(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe_for_disk(x) for x in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (int, float)) and not isinstance(obj, bool):
        return obj
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, date):
        return obj.isoformat()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return obj


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
    requested_top_n: int
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
        self._load_disk_backtest_cache()

    def _load_disk_backtest_cache(self) -> None:
        """Hydrate RAM cache from ``data/cache/backtest_results.json`` if present."""
        if not BACKTEST_CACHE_PATH.exists():
            return
        try:
            with open(BACKTEST_CACHE_PATH, encoding="utf-8") as f:
                raw = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not load backtest disk cache from %s: %s", BACKTEST_CACHE_PATH, exc)
            return
        for key_str, val in raw.items():
            try:
                k = _backtest_cache_key_from_json(key_str)
                if isinstance(val, dict):
                    self._backtest_cache[k] = _normalize_backtest_dict_from_disk(val)
            except (json.JSONDecodeError, TypeError, ValueError, KeyError) as exc:
                logger.debug("Skipping bad backtest cache entry %r: %s", key_str, exc)

    def _save_disk_backtest_cache(self) -> None:
        """Persist the full in-memory backtest cache to disk (JSON, inspectable)."""
        BACKTEST_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        serializable = {
            _backtest_cache_key_to_json(k): _json_safe_for_disk(v)
            for k, v in self._backtest_cache.items()
        }
        try:
            with open(BACKTEST_CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(serializable, f, indent=2)
        except (OSError, TypeError) as exc:
            logger.warning("Could not write backtest disk cache to %s: %s", BACKTEST_CACHE_PATH, exc)

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
        n_actual = len(selected)
        if n_actual < top_n:
            logger.warning(
                "Signal top_n shortfall: requested %d long positions but only %d tickers "
                "have valid predictions at cutoff %s (check OHLCV / feature coverage).",
                top_n,
                n_actual,
                cutoff_date,
            )

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

        portfolio_json = serialize_portfolio_bundle(portfolio, top_n)

        logger.info(
            "Signal generated: %d positions (requested top_n=%d), model=Lag60-SA (semi-annual), regime=%s",
            len(portfolio),
            top_n,
            regime_state.label.value,
        )

        return SignalResult(
            cutoff_date=cutoff_date,
            regime_label=regime_state.label.value,
            regime_confidence=round(regime_state.confidence, 4),
            portfolio=portfolio,
            portfolio_json=portfolio_json,
            requested_top_n=top_n,
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
        self._save_disk_backtest_cache()
        logger.info("Cached backtest result for %s", cache_key)
        return serialised

    # ------------------------------------------------------------------
    # Backtest simulation (capital-space with transactions)
    # ------------------------------------------------------------------

    def run_backtest_simulation(
        self,
        start_date: str,
        initial_capital: float,
        *,
        costs_bps: float = 40.0,
        top_n: int = _DEFAULT_TOP_N,
        rebalance_freq: int = _SEMI_ANNUAL_REBALANCE_FREQ,
    ) -> dict[str, Any]:
        """Simulate a backtest with explicit capital tracking and trade records.

        Uses the same walk-forward methodology as
        :func:`evaluate_forward_quarterly_regression` — regression predictions,
        rank-based hysteresis, return-proportional weights capped per position —
        but tracks absolute CHF values, share counts, and individual buy/sell
        transactions.

        Parameters
        ----------
        start_date
            ISO date; the simulation covers full calendar years starting from
            this date's year (earliest 2015).
        initial_capital
            Starting capital in CHF.
        costs_bps
            One-way transaction costs in basis points (default 40).
        top_n
            Number of long positions per rebalance.
        rebalance_freq
            ``2`` = semi-annual (default), ``1`` = quarterly.

        Returns
        -------
        dict with keys ``timeline``, ``transactions``, ``summary``.
        """
        from src.backtest import (
            _capped_proba_weights,
            _daily_close_matrix,
        )
        from src.regression_backtest import _select_long_with_hysteresis

        result = self.load_model()
        ohlcv = self._data.ohlcv
        fundamentals = self._data.fundamentals

        start_ts = pd.Timestamp(start_date)
        first_year = max(start_ts.year, 2015)
        end_year = 2025
        oos_years = list(range(first_year, end_year + 1))
        if not oos_years:
            raise ValueError(f"No simulation years for start_date={start_date}")

        all_periods = []
        for yr in oos_years:
            all_periods.extend(
                build_quarterly_rebalance_schedule(yr, rebalance_freq, 0),
            )

        cutoff_preds: dict[str, pd.Series] = {}
        universe_tickers: set[str] = set()
        for p in all_periods:
            if p.cutoff in cutoff_preds:
                continue
            X_oos = build_oos_features(
                ohlcv,
                fundamentals,
                cutoff_date=p.cutoff,
                publication_lag_days=_PUBLICATION_LAG_DAYS,
            )
            cutoff_preds[p.cutoff] = predict_returns(result, X_oos)
            universe_tickers.update(X_oos.index.tolist())

        all_tickers = sorted(universe_tickers)
        data_start = all_periods[0].period_start
        data_end = all_periods[-1].period_end
        close = _daily_close_matrix(
            ohlcv,
            all_tickers,
            first_year,
            start_date=data_start,
            end_date=data_end,
        )
        if close.empty:
            raise ValueError("No close data for the simulation period")
        close = close.ffill()

        first_pred = cutoff_preds[all_periods[0].cutoff]
        bm_tickers = [t for t in first_pred.index if t in close.columns]
        bm_close = close[bm_tickers].dropna(how="all", axis=1)
        bm_daily_ret = bm_close.pct_change().mean(axis=1).fillna(0)
        bm_values = initial_capital * (1 + bm_daily_ret).cumprod()

        holdings: dict[str, int] = {}
        cash = float(initial_capital)
        prev_portfolio: list[str] = []
        transactions: list[dict[str, Any]] = []
        timeline_dates: list[str] = []
        timeline_pf: list[float] = []
        timeline_bm: list[float] = []
        total_costs_chf = 0.0

        for period in all_periods:
            pred = cutoff_preds[period.cutoff]
            period_close = close.loc[period.period_start : period.period_end]
            if period_close.empty:
                continue

            pred_tradeable = pred.reindex(
                [t for t in pred.index if t in close.columns],
            ).dropna()
            new_portfolio = _select_long_with_hysteresis(
                pred_tradeable,
                prev_portfolio,
                top_n,
                _DEFAULT_HYSTERESIS_BUFFER,
            )

            pred_sel = pred.reindex(new_portfolio)
            pred_score = pred_sel - pred_sel.min() + 1e-12
            weights = _capped_proba_weights(
                pred_score, new_portfolio, _DEFAULT_MAX_WEIGHT,
            )

            first_day = period_close.index[0]
            trade_date = str(first_day.date())
            prices = period_close.iloc[0]

            for t in list(holdings):
                if t not in new_portfolio and holdings[t] > 0:
                    price = float(prices.get(t, 0))
                    if price > 0:
                        shares = holdings[t]
                        value = shares * price
                        cost = value * costs_bps / 10_000
                        cash += value - cost
                        total_costs_chf += cost
                        transactions.append({
                            "date": trade_date,
                            "ticker": t,
                            "action": "sell",
                            "shares": shares,
                            "price": round(price, 2),
                            "value": round(value, 2),
                        })
                    del holdings[t]

            portfolio_value = cash
            for t, s in holdings.items():
                p = float(prices.get(t, 0))
                if p > 0:
                    portfolio_value += s * p

            for t in new_portfolio:
                price = float(prices.get(t, 0))
                if price <= 0:
                    continue
                target = portfolio_value * float(weights.get(t, 0))
                cur_shares = holdings.get(t, 0)
                cur_val = cur_shares * price
                delta = target - cur_val

                if delta >= price:
                    buy_shares = int(delta / price)
                    buy_val = buy_shares * price
                    cost = buy_val * costs_bps / 10_000
                    if buy_val + cost > cash:
                        buy_shares = int(
                            cash / (price * (1 + costs_bps / 10_000)),
                        )
                        buy_val = buy_shares * price
                        cost = buy_val * costs_bps / 10_000
                    if buy_shares > 0:
                        cash -= buy_val + cost
                        total_costs_chf += cost
                        holdings[t] = cur_shares + buy_shares
                        transactions.append({
                            "date": trade_date,
                            "ticker": t,
                            "action": "buy",
                            "shares": buy_shares,
                            "price": round(price, 2),
                            "value": round(buy_val, 2),
                        })
                elif delta <= -price and cur_shares > 0:
                    sell_shares = min(int(-delta / price), cur_shares)
                    if sell_shares > 0:
                        sell_val = sell_shares * price
                        cost = sell_val * costs_bps / 10_000
                        cash += sell_val - cost
                        total_costs_chf += cost
                        holdings[t] = cur_shares - sell_shares
                        if holdings[t] == 0:
                            del holdings[t]
                        transactions.append({
                            "date": trade_date,
                            "ticker": t,
                            "action": "sell",
                            "shares": sell_shares,
                            "price": round(price, 2),
                            "value": round(sell_val, 2),
                        })

            held = [t for t in holdings if t in period_close.columns]
            if held:
                h_vec = pd.Series(
                    {t: holdings[t] for t in held}, dtype=float,
                )
                daily_pf = (period_close[held] * h_vec).sum(axis=1) + cash
            else:
                daily_pf = pd.Series(cash, index=period_close.index)

            daily_bm = bm_values.reindex(period_close.index)
            daily_bm = daily_bm.ffill().fillna(initial_capital)

            for ts in period_close.index:
                timeline_dates.append(str(ts.date()))
                timeline_pf.append(round(float(daily_pf.loc[ts]), 2))
                timeline_bm.append(round(float(daily_bm.loc[ts]), 2))

            prev_portfolio = list(new_portfolio)

        timeline = [
            {"date": d, "portfolio_value": p, "benchmark_value": b}
            for d, p, b in zip(timeline_dates, timeline_pf, timeline_bm)
        ]

        if len(timeline) > 1:
            pf_arr = np.array(timeline_pf, dtype=float)
            daily_rets = np.diff(pf_arr) / pf_arr[:-1]
            total_return = float(pf_arr[-1] / pf_arr[0] - 1)
            n_days = len(daily_rets)
            n_years = n_days / 252 if n_days > 0 else 1.0
            cagr = (
                float((1 + total_return) ** (1 / n_years) - 1)
                if n_years > 0 and total_return > -1
                else 0.0
            )
            vol = (
                float(np.std(daily_rets, ddof=1) * np.sqrt(252))
                if n_days > 1
                else 0.0
            )
            sharpe = cagr / vol if vol > 0 else 0.0

            wealth = np.concatenate([[1.0], np.cumprod(1 + daily_rets)])
            running_max = np.maximum.accumulate(wealth)
            drawdowns = wealth / running_max - 1
            max_dd = float(np.min(drawdowns))

            bm_start = timeline_bm[0]
            bm_end = timeline_bm[-1]
            bm_return = (
                float(bm_end / bm_start - 1) if bm_start > 0 else 0.0
            )

            summary = {
                "initial_capital": initial_capital,
                "final_value": round(float(pf_arr[-1]), 2),
                "total_return": round(total_return, 4),
                "annualized_return": round(cagr, 4),
                "sharpe_ratio": round(sharpe, 4),
                "max_drawdown": round(max_dd, 4),
                "total_costs": round(total_costs_chf, 2),
                "n_trades": len(transactions),
                "benchmark_final_value": round(float(bm_end), 2),
                "benchmark_total_return": round(bm_return, 4),
            }
        else:
            summary = {
                "initial_capital": initial_capital,
                "final_value": initial_capital,
                "total_return": 0.0,
                "annualized_return": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "total_costs": 0.0,
                "n_trades": 0,
                "benchmark_final_value": initial_capital,
                "benchmark_total_return": 0.0,
            }

        logger.info(
            "Simulation %d→%d: %.0f CHF → %.0f CHF (%.1f%%), "
            "%d trades, costs %.0f CHF",
            first_year,
            end_year,
            initial_capital,
            summary["final_value"],
            summary["total_return"] * 100,
            len(transactions),
            total_costs_chf,
        )

        return {
            "timeline": timeline,
            "transactions": transactions,
            "summary": summary,
        }


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
