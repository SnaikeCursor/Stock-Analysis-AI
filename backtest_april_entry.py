"""Backtest: Mid-April entry vs. January entry using the same year-end signal.

For each OOS year (2015–2025), compares three strategies — all using the same
walk-forward Lag60-SA model and the December-31 cutoff signal:

  A) FULL H1      — Jan 2 → Jun 30  (normal semi-annual holding period)
  B) APRIL ENTRY  — Apr 15 → Jun 30 (late entry, same picks as A)
  C) FULL YEAR    — Jan 2 → Dec 31  (hold all year without July rebalance)

This answers: "How much return do I miss (or gain) by entering mid-April
instead of early January, with the same signal?"

Uses existing trained models — no retraining required.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.backtest import (
    _capped_proba_weights,
    _compute_metrics,
    build_oos_features,
)
from src.regression_model import RegressionTrainResult, predict_returns

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_DEFAULT_TOP_N = 5
_DEFAULT_MAX_WEIGHT = 0.30
_MIN_POSITION_WEIGHT = 0.05
_PUBLICATION_LAG_DAYS = 60
_MODEL_SUFFIXES = ("_cs_pit_lag60", "_cs_lag60")
COSTS_BPS = 40.0

OOS_YEARS = list(range(2015, 2026))

APRIL_ENTRY_DAY = 15


def _load_walk_forward_models(
    oos_years: list[int],
) -> dict[int, RegressionTrainResult]:
    import joblib

    cache_dir = config.DATA_DIR / "cache"
    available: dict[int, tuple[Path, RegressionTrainResult]] = {}
    for suffix in _MODEL_SUFFIXES:
        for f in cache_dir.glob(f"regression_wf_*{suffix}.joblib"):
            stem = f.stem
            prefix = "regression_wf_"
            if not stem.startswith(prefix) or not stem.endswith(suffix):
                continue
            year_str = stem[len(prefix) : -len(suffix)]
            try:
                year = int(year_str)
            except ValueError:
                continue
            if year not in available:
                bundle = joblib.load(f)
                if isinstance(bundle, RegressionTrainResult):
                    result = bundle
                elif isinstance(bundle, dict) and "regression_result" in bundle:
                    inner = bundle["regression_result"]
                    result = inner if isinstance(inner, RegressionTrainResult) else RegressionTrainResult(**inner)
                elif isinstance(bundle, dict) and "model" in bundle:
                    result = RegressionTrainResult(**bundle)
                else:
                    continue
                available[year] = (f, result)

    models: dict[int, RegressionTrainResult] = {}
    for yr in oos_years:
        if yr in available:
            models[yr] = available[yr][1]
        else:
            prior = sorted(y for y in available if y < yr)
            if not prior:
                raise FileNotFoundError(f"No model for OOS year {yr}")
            models[yr] = available[prior[-1]][1]

    logger.info("Loaded %d walk-forward models", len(models))
    return models


def _load_data() -> tuple[dict, dict, dict, dict]:
    from backend.services.data_service import DataService

    ds = DataService()
    ds.refresh_ohlcv()
    return ds.ohlcv, ds.fundamentals, ds.eulerpool_quarterly, ds.eulerpool_profiles


def _daily_close(
    ohlcv: dict[str, pd.DataFrame],
    tickers: list[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    series = {}
    for t in tickers:
        df = ohlcv.get(t)
        if df is None or df.empty or "Close" not in df.columns:
            continue
        s = df["Close"].copy()
        s.index = pd.to_datetime(s.index)
        s = s.sort_index().loc[start:end].dropna()
        if not s.empty:
            series[t] = s
    if not series:
        return pd.DataFrame()
    return pd.DataFrame(series).sort_index()


def _portfolio_return(
    close: pd.DataFrame,
    weights: pd.Series,
    costs_bps: float,
) -> pd.Series:
    """Weighted daily returns with entry/exit costs."""
    available = [t for t in weights.index if t in close.columns]
    if not available:
        return pd.Series(dtype=float)
    w = weights.reindex(available).dropna()
    w = w / w.sum()
    daily_ret = close[available].pct_change().dropna(how="all")
    port_ret = (daily_ret * w).sum(axis=1).dropna()
    if port_ret.empty:
        return port_ret
    port_ret.iloc[0] -= costs_bps / 10_000.0
    port_ret.iloc[-1] -= costs_bps / 10_000.0
    return port_ret


def _select_and_weight(pred: pd.Series, close_cols: list[str]) -> pd.Series:
    """Top-N selection with max-weight cap and min-weight filter."""
    tradeable = pred.reindex([t for t in pred.index if t in close_cols]).dropna()
    if tradeable.empty:
        return pd.Series(dtype=float)
    top = tradeable.sort_values(ascending=False).head(_DEFAULT_TOP_N)
    picks = list(top.index)
    pred_score = pred.reindex(picks) - pred.reindex(picks).min() + 1e-12
    w = _capped_proba_weights(pred_score, picks, _DEFAULT_MAX_WEIGHT)
    w = w[w >= _MIN_POSITION_WEIGHT]
    if w.empty:
        w = pred_score.nlargest(1) / pred_score.nlargest(1).sum()
    else:
        w = w / w.sum()
    return w


def run_backtest(
    ohlcv: dict,
    fundamentals: dict,
    eulerpool_q: dict,
    eulerpool_p: dict,
    wf_models: dict[int, RegressionTrainResult],
) -> dict[int, dict]:
    results: dict[int, dict] = {}

    for yr in OOS_YEARS:
        if yr not in wf_models:
            logger.warning("No model for year %d — skipping", yr)
            continue

        cutoff = f"{yr - 1}-12-31"

        X_oos = build_oos_features(
            ohlcv,
            fundamentals,
            cutoff_date=cutoff,
            eulerpool_quarterly=eulerpool_q if eulerpool_q else None,
            eulerpool_profiles=eulerpool_p if eulerpool_p else None,
            publication_lag_days=_PUBLICATION_LAG_DAYS,
        )
        pred = predict_returns(wf_models[yr], X_oos)

        jan_start = f"{yr}-01-02"
        apr_start = f"{yr}-04-{APRIL_ENTRY_DAY}"
        h1_end = f"{yr}-06-30"
        year_end = f"{yr}-12-31"

        close_full = _daily_close(ohlcv, list(pred.index), jan_start, year_end)
        if close_full.empty:
            logger.warning("No close data for year %d — skipping", yr)
            continue

        w = _select_and_weight(pred, list(close_full.columns))
        if w.empty:
            logger.warning("No valid picks for year %d — skipping", yr)
            continue

        picks = list(w.index)

        # A) Full H1: Jan 2 → Jun 30
        close_h1 = _daily_close(ohlcv, picks, jan_start, h1_end)
        ret_full_h1 = _portfolio_return(close_h1, w, COSTS_BPS)
        m_full_h1 = _compute_metrics(ret_full_h1)

        # B) April entry: Apr 15 → Jun 30
        close_apr = _daily_close(ohlcv, picks, apr_start, h1_end)
        ret_apr = _portfolio_return(close_apr, w, COSTS_BPS)
        m_apr = _compute_metrics(ret_apr)

        # C) Full year: Jan 2 → Dec 31 (no July rebalance)
        close_yr = _daily_close(ohlcv, picks, jan_start, year_end)
        ret_full_yr = _portfolio_return(close_yr, w, COSTS_BPS)
        m_full_yr = _compute_metrics(ret_full_yr)

        # Jan 2 → Apr 14 ("missed" period)
        close_missed = _daily_close(ohlcv, picks, jan_start, f"{yr}-04-14")
        ret_missed = _portfolio_return(close_missed, w, 0.0)
        m_missed = _compute_metrics(ret_missed)

        results[yr] = {
            "picks": picks,
            "weights": {t: round(float(w[t]), 4) for t in picks},
            "full_h1_cum": m_full_h1.get("cumulative_return", float("nan")),
            "full_h1_sharpe": m_full_h1.get("sharpe_ratio", float("nan")),
            "apr_cum": m_apr.get("cumulative_return", float("nan")),
            "apr_sharpe": m_apr.get("sharpe_ratio", float("nan")),
            "missed_cum": m_missed.get("cumulative_return", float("nan")),
            "full_yr_cum": m_full_yr.get("cumulative_return", float("nan")),
            "full_yr_sharpe": m_full_yr.get("sharpe_ratio", float("nan")),
        }

        logger.info(
            "  %d: H1=%.1f%% | Apr-Jun=%.1f%% | Missed Jan-Apr=%.1f%% | FullYr=%.1f%% | picks=%s",
            yr,
            results[yr]["full_h1_cum"] * 100,
            results[yr]["apr_cum"] * 100,
            results[yr]["missed_cum"] * 100,
            results[yr]["full_yr_cum"] * 100,
            picks,
        )

    return results


def print_results(results: dict[int, dict]) -> None:
    years = sorted(results)
    if not years:
        print("No results.")
        return

    w = 130
    print("\n" + "=" * w)
    print("BACKTEST: Mid-April Entry vs. Full H1 (same year-end signal)")
    print(f"Model: Lag60-SA | Costs: {COSTS_BPS:.0f}bps entry+exit | Top-N: {_DEFAULT_TOP_N} | Max Weight: {_DEFAULT_MAX_WEIGHT*100:.0f}%")
    print("=" * w)

    header = (
        f"{'Year':>6}  |"
        f"{'Full H1 (Jan→Jun)':^22}|"
        f"{'Apr→Jun':^22}|"
        f"{'Missed (Jan→Apr)':^18}|"
        f"{'FullYr (Jan→Dec)':^22}|"
        f" {'Picks'}"
    )
    sub = (
        f"{'':>6}  |"
        f"{'Cum%':>9} {'Sharpe':>9}  |"
        f"{'Cum%':>9} {'Sharpe':>9}  |"
        f"{'Cum%':>9}        |"
        f"{'Cum%':>9} {'Sharpe':>9}  |"
    )
    print(header)
    print(sub)
    print("-" * w)

    full_h1_rets = []
    apr_rets = []
    missed_rets = []
    full_yr_rets = []

    for yr in years:
        r = results[yr]
        full_h1 = r["full_h1_cum"]
        apr = r["apr_cum"]
        missed = r["missed_cum"]
        full_yr = r["full_yr_cum"]

        if np.isfinite(full_h1):
            full_h1_rets.append(full_h1)
        if np.isfinite(apr):
            apr_rets.append(apr)
        if np.isfinite(missed):
            missed_rets.append(missed)
        if np.isfinite(full_yr):
            full_yr_rets.append(full_yr)

        picks_str = ", ".join(r["picks"][:5])
        missed_sign = "+" if missed >= 0 else ""

        print(
            f"{yr:>6}  |"
            f"{full_h1*100:>+9.1f} {r['full_h1_sharpe']:>9.2f}  |"
            f"{apr*100:>+9.1f} {r['apr_sharpe']:>9.2f}  |"
            f"{missed_sign}{missed*100:>8.1f}        |"
            f"{full_yr*100:>+9.1f} {r['full_yr_sharpe']:>9.2f}  |"
            f" {picks_str}"
        )

    print("-" * w)

    avg_h1 = np.mean(full_h1_rets) if full_h1_rets else float("nan")
    avg_apr = np.mean(apr_rets) if apr_rets else float("nan")
    avg_missed = np.mean(missed_rets) if missed_rets else float("nan")
    avg_full_yr = np.mean(full_yr_rets) if full_yr_rets else float("nan")

    med_h1 = np.median(full_h1_rets) if full_h1_rets else float("nan")
    med_apr = np.median(apr_rets) if apr_rets else float("nan")
    med_missed = np.median(missed_rets) if missed_rets else float("nan")

    geo_h1 = float(np.prod([1 + r for r in full_h1_rets]) ** (1 / len(full_h1_rets)) - 1) if full_h1_rets else float("nan")
    geo_apr = float(np.prod([1 + r for r in apr_rets]) ** (1 / len(apr_rets)) - 1) if apr_rets else float("nan")
    geo_full = float(np.prod([1 + r for r in full_yr_rets]) ** (1 / len(full_yr_rets)) - 1) if full_yr_rets else float("nan")

    n_apr_positive = sum(1 for r in apr_rets if r > 0)
    n_missed_positive = sum(1 for r in missed_rets if r > 0)
    n_h1_positive = sum(1 for r in full_h1_rets if r > 0)

    print(f"\n{'SUMMARY':^{w}}")
    print("=" * w)

    n = len(years)
    print(f"\n  {'Metric':<40} {'Full H1':>14} {'Apr→Jun':>14} {'Missed Jan→Apr':>16} {'Full Year':>14}")
    print(f"  {'─'*40} {'─'*14} {'─'*14} {'─'*16} {'─'*14}")
    print(f"  {'Avg Return':.<40} {avg_h1*100:>+13.2f}% {avg_apr*100:>+13.2f}% {avg_missed*100:>+15.2f}% {avg_full_yr*100:>+13.2f}%")
    print(f"  {'Median Return':.<40} {med_h1*100:>+13.2f}% {med_apr*100:>+13.2f}% {med_missed*100:>+15.2f}%")
    print(f"  {'Geometric Avg Return':.<40} {geo_h1*100:>+13.2f}% {geo_apr*100:>+13.2f}% {'':>16} {geo_full*100:>+13.2f}%")
    print(f"  {'Positive Years':.<40} {n_h1_positive:>11}/{n} {n_apr_positive:>11}/{n} {n_missed_positive:>13}/{n}")
    print()

    apr_better = sum(1 for yr in years if results[yr]["apr_cum"] > results[yr]["full_h1_cum"])
    print(f"  Apr-Entry BETTER than Full H1 in {apr_better}/{n} years")
    print(f"  (= years where Jan→Apr was negative, so waiting was beneficial)")
    print()

    print(f"  INTERPRETATION:")
    if avg_missed > 0.01:
        print(f"  → On average you MISS +{avg_missed*100:.1f}% by waiting until April.")
        print(f"  → Recommendation: Invest at the regular January signal.")
    elif avg_missed < -0.01:
        print(f"  → On average you GAIN {abs(avg_missed)*100:.1f}% by waiting until April (Jan→Apr was negative).")
        print(f"  → Waiting until April would have been beneficial historically.")
    else:
        print(f"  → Jan→Apr return is close to zero on average ({avg_missed*100:.1f}%).")
        print(f"  → Timing difference is minimal — stick with the regular cycle.")
    print("=" * w)


def main() -> None:
    logger.info("Loading data...")
    ohlcv, fundamentals, eulerpool_q, eulerpool_p = _load_data()

    logger.info("Loading walk-forward models...")
    wf_models = _load_walk_forward_models(OOS_YEARS)

    logger.info("Running April-entry backtest...")
    results = run_backtest(ohlcv, fundamentals, eulerpool_q, eulerpool_p, wf_models)

    print_results(results)


if __name__ == "__main__":
    main()
