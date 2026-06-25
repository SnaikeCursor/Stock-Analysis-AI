"""Backtest comparison: with vs without 5% minimum position weight filter.

Runs the walk-forward regression backtest (Lag60-SA, semi-annual rebalancing)
in two modes:
  A) BASELINE  — current backtest logic (no minimum weight filter)
  B) WITH 5%   — applies the same _MIN_POSITION_WEIGHT=0.05 filter used in
                  production signal generation

Uses existing trained models (no retraining required).
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
    _daily_close_matrix,
    build_oos_features,
    build_quarterly_rebalance_schedule,
    turnover_from_portfolio_change,
)
from src.regression_backtest import _select_long_with_hysteresis
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
_DEFAULT_HYSTERESIS_BUFFER = 2
_SEMI_ANNUAL_REBALANCE_FREQ = 2
_PUBLICATION_LAG_DAYS = 60
_MODEL_SUFFIXES = ("_cs_pit_lag60", "_cs_lag60")

OOS_YEARS = list(range(2015, 2026))
COSTS_BPS = 40.0


def _apply_min_weight_filter(
    weights: pd.Series,
    pred_score: pd.Series,
    min_weight: float,
) -> pd.Series:
    """Drop positions below *min_weight*, renormalize. Fallback: single best."""
    w = weights[weights >= min_weight]
    if len(w) == 0:
        w = pred_score.nlargest(1) / pred_score.nlargest(1).sum()
    else:
        w = w / w.sum()
    return w


def _load_walk_forward_models(
    oos_years: list[int],
) -> dict[int, RegressionTrainResult]:
    """Load per-year walk-forward regression models."""
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

    logger.info("Loaded %d walk-forward models for years %s", len(models), sorted(models))
    return models


def _load_data() -> tuple[dict, dict]:
    """Load OHLCV and fundamentals via DataService."""
    from backend.services.data_service import DataService

    ds = DataService()
    ds.refresh_ohlcv()
    return ds.ohlcv, ds.fundamentals


def run_comparison(
    ohlcv: dict,
    fundamentals: dict,
    wf_models: dict[int, RegressionTrainResult],
    *,
    apply_min_weight: bool,
) -> dict[int, dict]:
    """Run walk-forward backtest, optionally applying the 5% min-weight filter.

    Returns per-year metrics dict.
    """
    label = "WITH 5% min" if apply_min_weight else "BASELINE (no min)"
    logger.info("=" * 70)
    logger.info("Running backtest: %s", label)
    logger.info("=" * 70)

    per_year: dict[int, dict] = {}

    for yr in OOS_YEARS:
        if yr not in wf_models:
            logger.warning("No model for year %d — skipping", yr)
            continue

        schedule = build_quarterly_rebalance_schedule(yr, _SEMI_ANNUAL_REBALANCE_FREQ, 0)
        n_periods = len(schedule)

        period_preds: list[pd.Series] = []
        universe_tickers: set[str] = set()
        for p in schedule:
            X_oos = build_oos_features(
                ohlcv, fundamentals, cutoff_date=p.cutoff,
                publication_lag_days=_PUBLICATION_LAG_DAYS,
            )
            pred = predict_returns(wf_models[yr], X_oos)
            period_preds.append(pred)
            universe_tickers.update(X_oos.index.tolist())

        all_tickers = sorted(universe_tickers)
        data_start = schedule[0].period_start
        data_end = schedule[-1].period_end
        close = _daily_close_matrix(ohlcv, all_tickers, yr, start_date=data_start, end_date=data_end)
        if close.empty:
            logger.warning("No close data for year %d — skipping", yr)
            continue
        daily_ret_matrix = close.pct_change()

        bm_tickers = [t for t in period_preds[0].index if t in close.columns]
        bm_daily = daily_ret_matrix[bm_tickers].mean(axis=1).dropna()

        current_portfolio: list[str] = []
        portfolio_segments: list[pd.Series] = []
        year_cost_bps = 0.0
        all_n_positions: list[int] = []
        total_turnover_events = 0

        for i in range(n_periods):
            pred = period_preds[i]
            seg = schedule[i]
            prev_portfolio = list(current_portfolio)

            pred_tradeable = pred.reindex(
                [t for t in pred.index if t in close.columns],
            ).dropna()

            current_portfolio = _select_long_with_hysteresis(
                pred_tradeable, prev_portfolio, _DEFAULT_TOP_N, _DEFAULT_HYSTERESIS_BUFFER,
            )
            if not current_portfolio:
                continue

            n_swapped, turnover, _, _ = turnover_from_portfolio_change(
                prev_portfolio, current_portfolio, is_initial=(i == 0),
            )
            total_turnover_events += n_swapped

            period_dr = daily_ret_matrix.loc[seg.period_start : seg.period_end]
            available = [t for t in current_portfolio if t in period_dr.columns]
            if not available or period_dr.empty:
                continue

            pred_score = pred.reindex(available) - pred.reindex(available).min() + 1e-12
            w = _capped_proba_weights(pred_score, available, _DEFAULT_MAX_WEIGHT)

            if apply_min_weight:
                w = _apply_min_weight_filter(w, pred_score, _MIN_POSITION_WEIGHT)
                available = list(w.index)

            all_n_positions.append(len(available))

            segment = (period_dr[available] * w).sum(axis=1).dropna()
            segment.name = "portfolio"
            if segment.empty:
                continue

            period_cost_bps = 0.0
            if i == 0:
                segment.iloc[0] -= COSTS_BPS / 10_000.0
                period_cost_bps += COSTS_BPS
            elif n_swapped > 0:
                rebal_cost_bps = turnover * COSTS_BPS * 2
                segment.iloc[0] -= rebal_cost_bps / 10_000.0
                period_cost_bps += rebal_cost_bps
            if i == n_periods - 1:
                segment.iloc[-1] -= COSTS_BPS / 10_000.0
                period_cost_bps += COSTS_BPS

            year_cost_bps += period_cost_bps
            portfolio_segments.append(segment)

        if portfolio_segments:
            full_year_ret = pd.concat(portfolio_segments)
        else:
            full_year_ret = pd.Series(dtype=float)

        lo_metrics = _compute_metrics(full_year_ret)
        bm_metrics = _compute_metrics(bm_daily)

        per_year[yr] = {
            "cum_return": lo_metrics.get("cumulative_return", float("nan")),
            "ann_return": lo_metrics.get("annualized_return", float("nan")),
            "sharpe": lo_metrics.get("sharpe_ratio", float("nan")),
            "max_dd": lo_metrics.get("max_drawdown", float("nan")),
            "volatility": lo_metrics.get("volatility", float("nan")),
            "bm_cum": bm_metrics.get("cumulative_return", float("nan")),
            "bm_sharpe": bm_metrics.get("sharpe_ratio", float("nan")),
            "costs_bps": year_cost_bps,
            "avg_positions": float(np.mean(all_n_positions)) if all_n_positions else 0,
            "n_trades": total_turnover_events,
        }
        beat = lo_metrics.get("cumulative_return", 0) > bm_metrics.get("cumulative_return", 0)
        logger.info(
            "  %d: cum=%.1f%%  sharpe=%.2f  maxDD=%.1f%%  bm=%.1f%%  pos=%.1f  beat=%s",
            yr,
            per_year[yr]["cum_return"] * 100,
            per_year[yr]["sharpe"],
            per_year[yr]["max_dd"] * 100,
            per_year[yr]["bm_cum"] * 100,
            per_year[yr]["avg_positions"],
            beat,
        )

    return per_year


def print_comparison(baseline: dict[int, dict], with_filter: dict[int, dict]) -> None:
    """Print a formatted comparison table."""
    years = sorted(set(baseline) & set(with_filter))
    if not years:
        print("No overlapping years to compare.")
        return

    sep = "-" * 120
    print("\n" + "=" * 120)
    print("BACKTEST COMPARISON: 5% Minimum Position Weight Filter")
    print("Model: Lag60-SA | Rebalancing: Semi-annual | Costs: 40bps | Top-N: 5 | Max Weight: 30%")
    print("=" * 120)

    print(f"\n{'Year':>6}  |{'--- BASELINE (no filter) ---':^42}|{'--- WITH 5% MIN WEIGHT ---':^42}|{'-- DELTA --':^20}")
    print(f"{'':>6}  |{'Cum%':>8} {'Sharpe':>7} {'MaxDD%':>8} {'Pos':>5} {'BM%':>7} |{'Cum%':>8} {'Sharpe':>7} {'MaxDD%':>8} {'Pos':>5} {'BM%':>7} |{'dCum%':>8} {'dSharpe':>8}")
    print(sep)

    cum_base_rets = []
    cum_filt_rets = []
    sharpe_base = []
    sharpe_filt = []
    dd_base = []
    dd_filt = []
    n_beat_bm_base = 0
    n_beat_bm_filt = 0

    for yr in years:
        b = baseline[yr]
        f = with_filter[yr]
        d_cum = (f["cum_return"] - b["cum_return"]) * 100
        d_sharpe = f["sharpe"] - b["sharpe"]

        if np.isfinite(b["cum_return"]):
            cum_base_rets.append(b["cum_return"])
            if b["cum_return"] > b["bm_cum"]:
                n_beat_bm_base += 1
        if np.isfinite(f["cum_return"]):
            cum_filt_rets.append(f["cum_return"])
            if f["cum_return"] > f["bm_cum"]:
                n_beat_bm_filt += 1
        if np.isfinite(b["sharpe"]):
            sharpe_base.append(b["sharpe"])
        if np.isfinite(f["sharpe"]):
            sharpe_filt.append(f["sharpe"])
        if np.isfinite(b["max_dd"]):
            dd_base.append(b["max_dd"])
        if np.isfinite(f["max_dd"]):
            dd_filt.append(f["max_dd"])

        sign = "+" if d_cum >= 0 else ""
        print(
            f"{yr:>6}  |{b['cum_return']*100:>8.1f} {b['sharpe']:>7.2f} {b['max_dd']*100:>8.1f} {b['avg_positions']:>5.1f} {b['bm_cum']*100:>7.1f}"
            f" |{f['cum_return']*100:>8.1f} {f['sharpe']:>7.2f} {f['max_dd']*100:>8.1f} {f['avg_positions']:>5.1f} {f['bm_cum']*100:>7.1f}"
            f" |{sign}{d_cum:>7.1f} {d_sharpe:>+8.2f}"
        )

    print(sep)

    avg_cum_b = np.mean(cum_base_rets) if cum_base_rets else float("nan")
    avg_cum_f = np.mean(cum_filt_rets) if cum_filt_rets else float("nan")
    avg_sh_b = np.mean(sharpe_base) if sharpe_base else float("nan")
    avg_sh_f = np.mean(sharpe_filt) if sharpe_filt else float("nan")
    avg_dd_b = np.mean(dd_base) if dd_base else float("nan")
    avg_dd_f = np.mean(dd_filt) if dd_filt else float("nan")

    d_avg_cum = (avg_cum_f - avg_cum_b) * 100
    d_avg_sh = avg_sh_f - avg_sh_b
    sign = "+" if d_avg_cum >= 0 else ""
    print(
        f"{'AVG':>6}  |{avg_cum_b*100:>8.1f} {avg_sh_b:>7.2f} {avg_dd_b*100:>8.1f} {'':>5} {'':>7}"
        f" |{avg_cum_f*100:>8.1f} {avg_sh_f:>7.2f} {avg_dd_f*100:>8.1f} {'':>5} {'':>7}"
        f" |{sign}{d_avg_cum:>7.1f} {d_avg_sh:>+8.2f}"
    )

    # Geometric compounded return
    geo_base = float(np.prod([1 + r for r in cum_base_rets]) - 1) if cum_base_rets else float("nan")
    geo_filt = float(np.prod([1 + r for r in cum_filt_rets]) - 1) if cum_filt_rets else float("nan")
    n_years = len(cum_base_rets)
    cagr_base = float((1 + geo_base) ** (1 / n_years) - 1) if n_years > 0 and geo_base > -1 else float("nan")
    cagr_filt = float((1 + geo_filt) ** (1 / n_years) - 1) if n_years > 0 and geo_filt > -1 else float("nan")

    print(f"\n{'SUMMARY':^120}")
    print(sep)
    print(f"  {'Metric':<35} {'BASELINE':>15} {'WITH 5% MIN':>15} {'DELTA':>15}")
    print(f"  {'─'*35} {'─'*15} {'─'*15} {'─'*15}")
    print(f"  {'Avg Annual Return':.<35} {avg_cum_b*100:>14.2f}% {avg_cum_f*100:>14.2f}% {d_avg_cum:>+14.2f}%")
    print(f"  {'Compounded Total Return':.<35} {geo_base*100:>14.1f}% {geo_filt*100:>14.1f}% {(geo_filt-geo_base)*100:>+14.1f}%")
    print(f"  {'CAGR':.<35} {cagr_base*100:>14.2f}% {cagr_filt*100:>14.2f}% {(cagr_filt-cagr_base)*100:>+14.2f}%")
    print(f"  {'Avg Sharpe Ratio':.<35} {avg_sh_b:>15.3f} {avg_sh_f:>15.3f} {d_avg_sh:>+15.3f}")
    print(f"  {'Avg Max Drawdown':.<35} {avg_dd_b*100:>14.2f}% {avg_dd_f*100:>14.2f}% {(avg_dd_f-avg_dd_b)*100:>+14.2f}%")
    print(f"  {'Worst Max Drawdown':.<35} {min(dd_base)*100:>14.2f}% {min(dd_filt)*100:>14.2f}% {(min(dd_filt)-min(dd_base))*100:>+14.2f}%")
    print(f"  {'Years Beating Benchmark':.<35} {n_beat_bm_base:>12}/{n_years} {n_beat_bm_filt:>12}/{n_years} {n_beat_bm_filt-n_beat_bm_base:>+15d}")

    n_filter_better = sum(
        1
        for yr in years
        if np.isfinite(with_filter[yr]["cum_return"])
        and np.isfinite(baseline[yr]["cum_return"])
        and with_filter[yr]["cum_return"] > baseline[yr]["cum_return"]
    )
    n_filter_worse = sum(
        1
        for yr in years
        if np.isfinite(with_filter[yr]["cum_return"])
        and np.isfinite(baseline[yr]["cum_return"])
        and with_filter[yr]["cum_return"] < baseline[yr]["cum_return"]
    )
    n_equal = len(years) - n_filter_better - n_filter_worse

    print(f"\n  5% filter BETTER in {n_filter_better}/{len(years)} years, "
          f"WORSE in {n_filter_worse}/{len(years)}, "
          f"EQUAL in {n_equal}/{len(years)}")
    print("=" * 120)


def main() -> None:
    logger.info("Loading OHLCV data and fundamentals...")
    ohlcv, fundamentals = _load_data()

    logger.info("Loading walk-forward models...")
    wf_models = _load_walk_forward_models(OOS_YEARS)

    baseline = run_comparison(ohlcv, fundamentals, wf_models, apply_min_weight=False)
    with_filter = run_comparison(ohlcv, fundamentals, wf_models, apply_min_weight=True)

    print_comparison(baseline, with_filter)


if __name__ == "__main__":
    main()
