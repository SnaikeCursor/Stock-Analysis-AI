"""Rule-based market regime detection on the Swiss Market Index (^SSMI).

Classifies each trading day into one of three regimes — **Bull**, **Bear**,
or **Sideways** — using SMA(50)/SMA(200) alignment:

* **Bull**: Close > SMA(50) **and** SMA(50) > SMA(200) — all three ascending.
* **Bear**: Close < SMA(50) **and** SMA(50) < SMA(200) — all three descending.
* **Sideways**: everything else — SMAs contradictory, no clear trend.

Early-warning indicators flag regime transitions before they happen:

* **sma_cross_gap**: ``(SMA50 − SMA200) / SMA200`` — gap shrinking signals
  an imminent crossover.
* **sma_50_slope**: 20-day rate of change of SMA(50) — flattening or
  counter-regime slope dampens confidence.

Confidence score combines alignment strength with cross-gap and slope
dampening, mapped to [0, 1].  Designed to feed regime-specific model
selection without introducing any look-ahead bias (all indicators are causal).
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "Regime",
    "RegimeState",
    "detect_regime",
    "get_regime_history",
    "label_periods",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SMA_WINDOW: int = 200
_VOL_WINDOW: int = 60
_VOL_PERCENTILE: float = 0.75
_ANNUALISATION_FACTOR: float = np.sqrt(252)

_SMA_SHORT_WINDOW: int = 50
_SMA_SLOPE_LOOKBACK: int = 20  # days for SMA(50) rate-of-change

# Confidence mapping — maximum gap (%) that maps to 1.0
_TREND_SATURATION_PCT: float = 0.05  # ≥ 5 % alignment gap → full confidence

# Cross-gap dampening (early-warning near SMA crossover)
_CROSS_GAP_HIGH: float = 0.03  # |gap| ≥ 3 % → no dampening
_CROSS_GAP_LOW: float = 0.01   # |gap| ≤ 1 % → maximum dampening
_CROSS_GAP_DAMP_MIN: float = 0.5  # factor at |gap| ≤ 1 %

# Slope dampening (counter-regime SMA(50) direction)
_SLOPE_COUNTER_DAMP: float = 0.6  # factor when slope contradicts regime

# Legacy vol constants — kept for backward-compatible indicator computation
_VOL_SATURATION_FRAC: float = 0.50


# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------


class Regime(str, Enum):
    """Market regime labels."""

    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"


@dataclass(frozen=True)
class RegimeIndicators:
    """Raw indicator values underlying a regime decision."""

    close: float
    sma_200: float
    sma_50: float
    trend_strength: float  # (close - sma200) / sma200
    realised_vol: float  # annualised 60d vol
    vol_threshold: float  # expanding 75th-pctl of realised_vol
    sma_cross_gap: float  # (sma50 - sma200) / sma200
    sma_50_slope: float  # 20d rate-of-change of sma50


@dataclass(frozen=True)
class RegimeState:
    """Regime classification for a single point in time."""

    label: Regime
    confidence: float  # 0.0 … 1.0
    indicators: RegimeIndicators


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _prepare_ohlcv(smi_ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Normalise index to ``DatetimeIndex`` and sort chronologically."""
    df = smi_ohlcv.copy()
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def _compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Attach regime indicator columns to a prepared OHLCV frame.

    Columns added: ``sma_200``, ``sma_50``, ``trend_strength``,
    ``sma_cross_gap``, ``sma_50_slope``, ``realised_vol``, ``vol_threshold``.
    """
    close = df["Close"].astype(float)

    df = df.copy()
    df["sma_200"] = close.rolling(_SMA_WINDOW, min_periods=_SMA_WINDOW).mean()
    df["sma_50"] = close.rolling(_SMA_SHORT_WINDOW, min_periods=_SMA_SHORT_WINDOW).mean()

    df["trend_strength"] = (close - df["sma_200"]) / df["sma_200"]

    # Early-warning indicators
    df["sma_cross_gap"] = (df["sma_50"] - df["sma_200"]) / df["sma_200"]
    sma_50_shifted = df["sma_50"].shift(_SMA_SLOPE_LOOKBACK)
    df["sma_50_slope"] = (df["sma_50"] - sma_50_shifted) / sma_50_shifted

    # Legacy vol indicators (kept for downstream compatibility)
    log_ret = np.log(close / close.shift(1))
    df["realised_vol"] = (
        log_ret.rolling(_VOL_WINDOW, min_periods=_VOL_WINDOW).std()
        * _ANNUALISATION_FACTOR
    )

    df["vol_threshold"] = df["realised_vol"].expanding(min_periods=_VOL_WINDOW).quantile(
        _VOL_PERCENTILE
    )
    return df


def _classify_row(row: pd.Series) -> tuple[Regime, float]:
    """Determine regime label and confidence from SMA(50)/SMA(200) alignment.

    Classification:
      Bull    — Close > SMA(50) and SMA(50) > SMA(200)
      Bear    — Close < SMA(50) and SMA(50) < SMA(200)
      Sideways — everything else (conflicting signals)

    Confidence = base_confidence × cross_gap_factor × slope_factor.
    """
    close = float(row["Close"])
    sma_50 = float(row["sma_50"])
    sma_200 = float(row["sma_200"])
    cross_gap = float(row["sma_cross_gap"])
    slope = float(row["sma_50_slope"])

    close_above_50 = close > sma_50
    sma50_above_200 = sma_50 > sma_200

    # --- label ---
    if close_above_50 and sma50_above_200:
        label = Regime.BULL
    elif (not close_above_50) and (not sma50_above_200):
        label = Regime.BEAR
    else:
        label = Regime.SIDEWAYS

    # --- base confidence (alignment strength) ---
    gap_c_50 = (close - sma_50) / sma_50 if sma_50 != 0 else 0.0
    gap_50_200 = (sma_50 - sma_200) / sma_200 if sma_200 != 0 else 0.0

    if label == Regime.BULL:
        min_gap = min(gap_c_50, gap_50_200)
        base_confidence = min(1.0, min_gap / _TREND_SATURATION_PCT)
    elif label == Regime.BEAR:
        min_gap = min(abs(gap_c_50), abs(gap_50_200))
        base_confidence = min(1.0, min_gap / _TREND_SATURATION_PCT)
    else:
        disagreement = min(abs(gap_c_50), abs(gap_50_200))
        base_confidence = min(1.0, disagreement / _TREND_SATURATION_PCT)

    # --- cross-gap dampening (early warning near SMA crossover) ---
    abs_gap = abs(cross_gap)
    if abs_gap >= _CROSS_GAP_HIGH:
        cross_gap_factor = 1.0
    elif abs_gap <= _CROSS_GAP_LOW:
        cross_gap_factor = _CROSS_GAP_DAMP_MIN
    else:
        t = (abs_gap - _CROSS_GAP_LOW) / (_CROSS_GAP_HIGH - _CROSS_GAP_LOW)
        cross_gap_factor = _CROSS_GAP_DAMP_MIN + t * (1.0 - _CROSS_GAP_DAMP_MIN)

    # --- slope dampening (counter-regime SMA(50) direction) ---
    if label == Regime.BULL and slope < 0:
        slope_factor = _SLOPE_COUNTER_DAMP
    elif label == Regime.BEAR and slope > 0:
        slope_factor = _SLOPE_COUNTER_DAMP
    else:
        slope_factor = 1.0

    confidence = base_confidence * cross_gap_factor * slope_factor
    return label, float(np.clip(confidence, 0.0, 1.0))


def _row_to_state(row: pd.Series) -> RegimeState:
    """Build a ``RegimeState`` from an indicator-enriched row."""
    label, confidence = _classify_row(row)
    indicators = RegimeIndicators(
        close=float(row["Close"]),
        sma_200=float(row["sma_200"]),
        sma_50=float(row["sma_50"]),
        trend_strength=float(row["trend_strength"]),
        realised_vol=float(row["realised_vol"]),
        vol_threshold=float(row["vol_threshold"]),
        sma_cross_gap=float(row["sma_cross_gap"]),
        sma_50_slope=float(row["sma_50_slope"]),
    )
    return RegimeState(label=label, confidence=confidence, indicators=indicators)


_DEFAULT_SMOOTHING_WINDOW: int = 10


def _as_regime(label: Regime | str | np.generic) -> Regime:
    """Coerce label to :class:`Regime` (``np.unique`` may stringify enums)."""
    if isinstance(label, Regime):
        return label
    return Regime(str(label))


def _smooth_regime(
    labels: np.ndarray,
    confidences: np.ndarray,
    window: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply rolling majority vote to daily regime labels.

    Parameters
    ----------
    labels
        1-D array of :class:`Regime` values (one per trading day).
    confidences
        1-D array of float confidences aligned with *labels*.
    window
        Number of trailing days for the majority vote.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(smoothed_labels, smoothed_confidences)`` — same length as input.
        For the first ``window - 1`` rows where a full window is unavailable,
        the available history is used (expanding window).
    """
    n = len(labels)
    if n == 0:
        return labels.copy(), confidences.copy()

    smoothed_labels = np.empty(n, dtype=object)
    smoothed_confs = np.empty(n, dtype=float)

    for i in range(n):
        start = max(0, i - window + 1)
        win_labels = labels[start : i + 1]
        win_confs = confidences[start : i + 1]

        # Avoid np.unique on enums — it stringifies members unreliably.
        coerced = [_as_regime(x) for x in win_labels]
        majority_label = Counter(coerced).most_common(1)[0][0]

        mask = np.array([c == majority_label for c in coerced])
        smoothed_labels[i] = majority_label
        smoothed_confs[i] = float(np.mean(win_confs[mask], dtype=float))

    return smoothed_labels, smoothed_confs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_regime(
    smi_ohlcv: pd.DataFrame,
    date: str,
    *,
    smoothing_window: int = _DEFAULT_SMOOTHING_WINDOW,
) -> RegimeState:
    """Classify the market regime on a specific *date*.

    Parameters
    ----------
    smi_ohlcv
        OHLCV DataFrame for ``^SSMI`` with a ``DatetimeIndex`` and ``Close``
        column (as returned by :func:`src.data_loader.download_spi_universe`
        for the ``^SSMI`` key).
    date
        ISO date string (e.g. ``"2022-06-30"``).  The regime is determined
        using only data up to and including this date (no lookahead).
    smoothing_window
        Number of trailing trading days for the rolling majority vote
        (default 10 ≈ 2 weeks).  Set to 1 to disable smoothing.

    Returns
    -------
    RegimeState
        Named tuple with ``label`` (:class:`Regime`), ``confidence`` (float),
        and ``indicators`` (:class:`RegimeIndicators`).

    Raises
    ------
    ValueError
        If *date* is not found in the data or if insufficient history exists
        for the SMA(200) warm-up.
    """
    df = _prepare_ohlcv(smi_ohlcv)
    ts = pd.Timestamp(date)

    df = df.loc[:ts]
    if df.empty:
        raise ValueError(f"No data on or before {date}")

    df = _compute_indicators(df)

    valid = df.dropna(
        subset=["sma_200", "sma_50", "sma_cross_gap", "sma_50_slope",
                "realised_vol", "vol_threshold"]
    )
    if valid.empty:
        raise ValueError(
            f"Insufficient history for SMA({_SMA_WINDOW}) warm-up by {date}"
        )

    tail = valid.iloc[-smoothing_window:]
    raw_labels = np.array(
        [_classify_row(row)[0] for _, row in tail.iterrows()], dtype=object,
    )
    raw_confs = np.array([_classify_row(row)[1] for _, row in tail.iterrows()])

    sm_labels, sm_confs = _smooth_regime(raw_labels, raw_confs, smoothing_window)

    last_row = valid.iloc[-1]
    indicators = RegimeIndicators(
        close=float(last_row["Close"]),
        sma_200=float(last_row["sma_200"]),
        sma_50=float(last_row["sma_50"]),
        trend_strength=float(last_row["trend_strength"]),
        realised_vol=float(last_row["realised_vol"]),
        vol_threshold=float(last_row["vol_threshold"]),
        sma_cross_gap=float(last_row["sma_cross_gap"]),
        sma_50_slope=float(last_row["sma_50_slope"]),
    )
    return RegimeState(
        label=_as_regime(sm_labels[-1]),
        confidence=float(sm_confs[-1]),
        indicators=indicators,
    )


def label_periods(
    smi_ohlcv: pd.DataFrame,
    periods: Sequence[tuple[str, str, str, str]],
    *,
    smoothing_window: int = _DEFAULT_SMOOTHING_WINDOW,
) -> dict[str, RegimeState]:
    """Label each classification period with its regime at the feature cutoff.

    Parameters
    ----------
    smi_ohlcv
        ``^SSMI`` OHLCV (see :func:`detect_regime`).
    periods
        List of ``(feature_cutoff, q_start, q_end, period_label)`` tuples,
        matching the shape of :data:`config.CLASSIFICATION_PERIODS`.
    smoothing_window
        Number of trailing trading days for the rolling majority vote
        (default 10 ≈ 2 weeks).  Set to 1 to disable smoothing.

    Returns
    -------
    dict
        Mapping ``{period_label: RegimeState}``.  Periods that cannot be
        classified (e.g. insufficient warm-up) are logged and omitted.
    """
    df = _prepare_ohlcv(smi_ohlcv)
    df = _compute_indicators(df)
    valid = df.dropna(
        subset=["sma_200", "sma_50", "sma_cross_gap", "sma_50_slope",
                "realised_vol", "vol_threshold"]
    )

    result: dict[str, RegimeState] = {}
    for feature_cutoff, _q_start, _q_end, plabel in periods:
        ts = pd.Timestamp(feature_cutoff)
        subset = valid.loc[:ts]
        if subset.empty:
            logger.warning("Cannot classify %s — no valid data by %s", plabel, feature_cutoff)
            continue

        tail = subset.iloc[-smoothing_window:]
        raw_labels = np.array(
            [_classify_row(row)[0] for _, row in tail.iterrows()], dtype=object,
        )
        raw_confs = np.array([_classify_row(row)[1] for _, row in tail.iterrows()])

        sm_labels, sm_confs = _smooth_regime(raw_labels, raw_confs, smoothing_window)

        last_row = subset.iloc[-1]
        indicators = RegimeIndicators(
            close=float(last_row["Close"]),
            sma_200=float(last_row["sma_200"]),
            sma_50=float(last_row["sma_50"]),
            trend_strength=float(last_row["trend_strength"]),
            realised_vol=float(last_row["realised_vol"]),
            vol_threshold=float(last_row["vol_threshold"]),
            sma_cross_gap=float(last_row["sma_cross_gap"]),
            sma_50_slope=float(last_row["sma_50_slope"]),
        )
        result[plabel] = RegimeState(
            label=_as_regime(sm_labels[-1]),
            confidence=float(sm_confs[-1]),
            indicators=indicators,
        )
    return result


def get_regime_history(
    smi_ohlcv: pd.DataFrame,
    start: str,
    end: str,
    *,
    smoothing_window: int = _DEFAULT_SMOOTHING_WINDOW,
) -> pd.DataFrame:
    """Produce a daily regime time-series between *start* and *end*.

    Parameters
    ----------
    smi_ohlcv
        ``^SSMI`` OHLCV (see :func:`detect_regime`).
    start, end
        ISO date strings bounding the output window (inclusive).
    smoothing_window
        Number of trailing trading days for the rolling majority vote
        (default 10 ≈ 2 weeks).  Set to 1 to disable smoothing.

    Returns
    -------
    pd.DataFrame
        Columns: ``regime``, ``confidence``, ``close``, ``sma_200``,
        ``sma_50``, ``trend_strength``, ``sma_cross_gap``, ``sma_50_slope``,
        ``realised_vol``, ``vol_threshold``.  Indexed by date.  Rows prior
        to SMA warm-up are excluded.
    """
    df = _prepare_ohlcv(smi_ohlcv)
    df = _compute_indicators(df)
    valid = df.dropna(
        subset=["sma_200", "sma_50", "sma_cross_gap", "sma_50_slope",
                "realised_vol", "vol_threshold"]
    )

    if valid.empty:
        return pd.DataFrame(
            columns=[
                "regime", "confidence", "close", "sma_200", "sma_50",
                "trend_strength", "sma_cross_gap", "sma_50_slope",
                "realised_vol", "vol_threshold",
            ]
        )

    all_labels = np.array(
        [_classify_row(row)[0] for _, row in valid.iterrows()], dtype=object,
    )
    all_confs = np.array([_classify_row(row)[1] for _, row in valid.iterrows()])

    sm_labels, sm_confs = _smooth_regime(all_labels, all_confs, smoothing_window)

    ts_start, ts_end = pd.Timestamp(start), pd.Timestamp(end)

    valid_idx = valid.index
    mask = np.asarray(
        (valid_idx >= ts_start) & (valid_idx <= ts_end), dtype=bool,
    )

    records: list[dict] = []
    for i, (idx, row) in enumerate(valid.iterrows()):
        if not mask[i]:
            continue
        records.append(
            {
                "date": idx,
                "regime": _as_regime(sm_labels[i]).value,
                "confidence": float(sm_confs[i]),
                "close": float(row["Close"]),
                "sma_200": float(row["sma_200"]),
                "sma_50": float(row["sma_50"]),
                "trend_strength": float(row["trend_strength"]),
                "sma_cross_gap": float(row["sma_cross_gap"]),
                "sma_50_slope": float(row["sma_50_slope"]),
                "realised_vol": float(row["realised_vol"]),
                "vol_threshold": float(row["vol_threshold"]),
            }
        )

    if not records:
        return pd.DataFrame(
            columns=[
                "regime", "confidence", "close", "sma_200", "sma_50",
                "trend_strength", "sma_cross_gap", "sma_50_slope",
                "realised_vol", "vol_threshold",
            ]
        )

    out = pd.DataFrame(records).set_index("date")
    out.index.name = None
    return out
