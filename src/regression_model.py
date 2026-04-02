"""Regression training, CV, and prediction (RF / XGBoost / LightGBM regressors).

Cross-sectional return prediction for the Swiss SPI Extra pipeline —
expanding-window walk-forward cross-validation (with optional purged embargo),
hyperparameter tuning via GridSearchCV with IC (Spearman rank correlation) as
primary metric, LightGBM quantile regression for outlier-robust predictions,
SHAP explanations, and model persistence.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import reduce
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

try:
    from config import CV_FOLDS, RANDOM_SEED
except ImportError:
    RANDOM_SEED = 42
    CV_FOLDS = 5

logger = logging.getLogger(__name__)

__all__ = [
    "AveragedRegressor",
    "ExpandingWindowSplit",
    "RegressionTrainResult",
    "compute_ic_by_period",
    "evaluate_regression",
    "load_regression_model",
    "plot_feature_importance_regression",
    "plot_ic_by_period",
    "plot_predicted_vs_actual",
    "predict_returns",
    "refit_regressor_full",
    "save_regression_model",
    "shap_explain_regression",
    "train_regression_ensemble",
    "train_regressor",
]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class RegressionTrainResult:
    """Container for regression training artefacts and evaluation metrics."""

    model: Any
    imputer_medians: pd.Series
    feature_names: list[str]
    best_params: dict[str, Any]
    cv_results: dict[str, Any]
    holdout_metrics: dict[str, Any]
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    period_labels_train: np.ndarray | None = None
    period_labels_test: np.ndarray | None = None
    model_type: str = "rf"


# ---------------------------------------------------------------------------
# Expanding-window cross-validation (time-series aware)
# ---------------------------------------------------------------------------


class ExpandingWindowSplit:
    """Expanding-window walk-forward splitter for month-labelled data.

    Train on periods ``0..k``, validate on period ``k+1+embargo``.
    Compatible with scikit-learn's CV interface (``split(X, y, groups)``).

    Parameters
    ----------
    period_labels
        Ordered array of period identifiers aligned row-by-row with the
        training data (e.g. ``"2023-01"``, ``"2023-02"``, …).  Unique
        values are deduced in encounter order and assumed chronological.
    min_train_periods
        Minimum number of training periods before the first split.
    embargo_periods
        Number of periods to skip between the end of the training window
        and the validation period (purged walk-forward).  ``0`` means no
        gap; ``1`` leaves a 1-period buffer to prevent information leakage
        from overlapping feature windows.
    """

    def __init__(
        self,
        period_labels: np.ndarray | pd.Series | list[str],
        *,
        min_train_periods: int = 3,
        embargo_periods: int = 0,
    ) -> None:
        self.period_labels = np.asarray(period_labels)
        seen: list[str] = []
        for p in self.period_labels:
            if p not in seen:
                seen.append(p)
        self.unique_periods: list[str] = seen
        self.min_train_periods = max(1, min_train_periods)
        self.embargo_periods = max(0, embargo_periods)

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        return max(
            0,
            len(self.unique_periods) - self.min_train_periods - self.embargo_periods,
        )

    def split(self, X: Any = None, y: Any = None, groups: Any = None):
        """Yield ``(train_indices, test_indices)`` tuples.

        When ``embargo_periods > 0``, periods ``k+1 .. k+embargo`` are
        excluded from both train and validation sets.
        """
        n = len(self.unique_periods)
        for k in range(self.min_train_periods, n - self.embargo_periods):
            train_periods = set(self.unique_periods[:k])
            val_idx = k + self.embargo_periods
            if val_idx >= n:
                continue
            test_period = self.unique_periods[val_idx]
            train_idx = np.where(
                np.isin(self.period_labels, list(train_periods))
            )[0]
            test_idx = np.where(self.period_labels == test_period)[0]
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
            yield train_idx, test_idx


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _impute_median(X: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Fill NaN with per-column median; drop all-NaN columns."""
    out = X.copy()
    all_nan = out.columns[out.isna().all()]
    if len(all_nan):
        logger.warning("Dropping %d all-NaN columns: %s", len(all_nan), list(all_nan))
        out = out.drop(columns=all_nan)
    medians = out.median()
    return out.fillna(medians), medians


def _apply_imputation(
    X: pd.DataFrame,
    medians: pd.Series,
    feature_names: list[str],
) -> pd.DataFrame:
    """Apply pre-computed medians and align to training feature order."""
    common = [c for c in feature_names if c in X.columns]
    out = X[common].copy()
    out = out.fillna(medians.reindex(common))
    for c in feature_names:
        if c not in out.columns:
            out[c] = 0.0
    return out[feature_names]


def _rf_reg_param_grid() -> dict[str, list]:
    return {
        "n_estimators": [200, 500],
        "max_depth": [None, 10, 20],
        "min_samples_split": [5, 10],
        "min_samples_leaf": [3, 5],
        "max_features": ["sqrt", "log2"],
    }


def _xgb_reg_param_grid() -> dict[str, list]:
    return {
        "n_estimators": [200, 500],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }


def _lgb_reg_param_grid() -> dict[str, list]:
    return {
        "n_estimators": [200, 500],
        "max_depth": [3, 7, -1],
        "learning_rate": [0.01, 0.05, 0.1],
        "num_leaves": [31, 63, 127],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }


def _grid_size(grid: dict[str, list]) -> int:
    return reduce(lambda a, b: a * b, (len(v) for v in grid.values()), 1)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _spearman_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman rank correlation (Information Coefficient).

    Returns NaN when fewer than 5 finite paired observations exist.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(yt) & np.isfinite(yp)
    if mask.sum() < 5:
        return float("nan")
    corr, _ = spearmanr(yt[mask], yp[mask])
    return float(corr)


def _ic_score_func(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Score function compatible with :func:`sklearn.metrics.make_scorer`."""
    return _spearman_ic(y_true, y_pred)


_ic_scorer = make_scorer(_ic_score_func, greater_is_better=True)


def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    period_labels: np.ndarray | None = None,
) -> dict[str, Any]:
    """IC (Spearman), RMSE, MAE, R², and optional per-period IC breakdown.

    Parameters
    ----------
    y_true, y_pred
        Actual and predicted returns (same length).
    period_labels
        Optional period identifiers (same length as *y_true*).  When
        provided, the result includes ``ic_by_period``, ``ic_mean``,
        ``ic_std``, and ``ic_ir`` (IC Information Ratio = mean/std).
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(yt) & np.isfinite(yp)
    yt_c = yt[mask]
    yp_c = yp[mask]

    ic = _spearman_ic(yt_c, yp_c)
    rmse = float(np.sqrt(mean_squared_error(yt_c, yp_c)))
    mae = float(mean_absolute_error(yt_c, yp_c))
    r2 = float(r2_score(yt_c, yp_c)) if len(yt_c) > 1 else float("nan")

    result: dict[str, Any] = {
        "ic": ic,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "n_samples": int(mask.sum()),
    }

    if period_labels is not None:
        pl = np.asarray(period_labels)[mask]
        ic_df = compute_ic_by_period(yt_c, yp_c, pl)
        result["ic_by_period"] = ic_df
        valid_ics = ic_df["ic"].dropna()
        result["ic_mean"] = float(valid_ics.mean()) if len(valid_ics) else float("nan")
        result["ic_std"] = float(valid_ics.std()) if len(valid_ics) > 1 else float("nan")
        result["ic_ir"] = (
            result["ic_mean"] / result["ic_std"]
            if np.isfinite(result.get("ic_std", 0)) and result["ic_std"] > 0
            else float("nan")
        )

    return result


def compute_ic_by_period(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    period_labels: np.ndarray,
) -> pd.DataFrame:
    """Per-period IC (Spearman) for stability analysis.

    Parameters
    ----------
    y_true, y_pred
        Actual and predicted returns (same length).
    period_labels
        Period identifiers (same length), e.g. ``"2023-09"``.

    Returns
    -------
    pd.DataFrame
        Columns: ``period``, ``ic``, ``n`` (sample count per period).
    """
    unique_periods: list = []
    for p in period_labels:
        if p not in unique_periods:
            unique_periods.append(p)

    rows = []
    for p in unique_periods:
        mask = period_labels == p
        ic = _spearman_ic(y_true[mask], y_pred[mask])
        rows.append({"period": p, "ic": ic, "n": int(mask.sum())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Ensemble regressor
# ---------------------------------------------------------------------------


class AveragedRegressor(BaseEstimator, RegressorMixin):
    """Blend two regressors by weighted-average predictions.

    Used by :func:`train_regression_ensemble` to combine Random Forest and
    XGBoost regressors.

    Parameters
    ----------
    rf_estimator, xgb_estimator
        Un-fitted (template) estimators.  Cloned and fitted in :meth:`fit`.
    weights
        ``(rf_weight, xgb_weight)`` for the prediction average.
    """

    def __init__(
        self,
        rf_estimator: Any = None,
        xgb_estimator: Any = None,
        *,
        weights: tuple[float, float] = (0.5, 0.5),
    ) -> None:
        self.rf_estimator = rf_estimator
        self.xgb_estimator = xgb_estimator
        self.weights = weights

    def fit(self, X: np.ndarray | pd.DataFrame, y: np.ndarray) -> AveragedRegressor:
        self.rf_ = clone(self.rf_estimator).fit(X, y)
        self.xgb_ = clone(self.xgb_estimator).fit(X, y)
        return self

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        p1 = self.rf_.predict(X)
        p2 = self.xgb_.predict(X)
        w1, w2 = self.weights
        return w1 * p1 + w2 * p2


# ---------------------------------------------------------------------------
# Data preparation (shared by train_regressor / train_regression_ensemble)
# ---------------------------------------------------------------------------


@dataclass
class _PreparedRegressionData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    cv: Any
    feature_names: list[str]
    medians: pd.Series
    pl_train: np.ndarray | None
    pl_test: np.ndarray | None


def _prepare_regression_data(
    X: pd.DataFrame,
    y: np.ndarray | pd.Series,
    *,
    period_labels: np.ndarray | pd.Series | None,
    min_train_periods: int,
    cv_folds: int | None,
    holdout_periods: int = 1,
    embargo_periods: int = 0,
) -> _PreparedRegressionData:
    """Clean NaN targets, impute features, build train/test split and CV splitter."""
    n_cv = CV_FOLDS if cv_folds is None else cv_folds
    y_arr = np.asarray(y, dtype=float)

    valid = np.isfinite(y_arr)
    if hasattr(X, "notna"):
        valid &= X.notna().any(axis=1).values
    X_clean = X.loc[valid].copy()
    y_clean = y_arr[valid]

    pl_clean: np.ndarray | None = None
    if period_labels is not None:
        pl_clean = np.asarray(period_labels)[valid]

    X_imp, medians = _impute_median(X_clean)
    feature_names = list(X_imp.columns)

    pl_train: np.ndarray | None = None
    pl_test: np.ndarray | None = None

    if pl_clean is not None:
        unique_periods: list = []
        for p in pl_clean:
            if p not in unique_periods:
                unique_periods.append(p)

        n_holdout = max(1, min(holdout_periods, len(unique_periods) - min_train_periods))
        holdout_periods_list = unique_periods[-n_holdout:]
        holdout_set = set(holdout_periods_list)
        train_mask = ~np.isin(pl_clean, list(holdout_set))
        test_mask = np.isin(pl_clean, list(holdout_set))

        X_train = X_imp[train_mask].copy()
        X_test = X_imp[test_mask].copy()
        y_train = y_clean[train_mask]
        y_test = y_clean[test_mask]
        pl_train = pl_clean[train_mask]
        pl_test = pl_clean[test_mask]

        logger.info(
            "Walk-forward split: %d train (%d periods) / %d test (%d holdout periods: %s), %d features",
            len(X_train),
            len(unique_periods) - n_holdout,
            len(X_test),
            n_holdout,
            ", ".join(str(p) for p in holdout_periods_list),
            len(feature_names),
        )

        cv: Any = ExpandingWindowSplit(
            pl_train,
            min_train_periods=min_train_periods,
            embargo_periods=embargo_periods,
        )
        n_splits = cv.get_n_splits()
        if n_splits < 2:
            logger.warning(
                "Expanding-window CV has %d splits (need >= 2). "
                "Falling back to TimeSeriesSplit(n_splits=%d).",
                n_splits,
                n_cv,
            )
            cv = TimeSeriesSplit(n_splits=n_cv)
    else:
        split_idx = int(len(X_imp) * 0.75)
        X_train = X_imp.iloc[:split_idx].copy()
        X_test = X_imp.iloc[split_idx:].copy()
        y_train = y_clean[:split_idx]
        y_test = y_clean[split_idx:]
        cv = TimeSeriesSplit(n_splits=n_cv)
        logger.info(
            "Chronological split: %d train / %d test, %d features",
            len(X_train),
            len(X_test),
            len(feature_names),
        )

    return _PreparedRegressionData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        cv=cv,
        feature_names=feature_names,
        medians=medians,
        pl_train=pl_train,
        pl_test=pl_test,
    )


# ---------------------------------------------------------------------------
# Core training API
# ---------------------------------------------------------------------------


def train_regressor(
    X: pd.DataFrame,
    y: np.ndarray | pd.Series,
    *,
    model_type: str = "rf",
    period_labels: np.ndarray | pd.Series | None = None,
    random_state: int | None = None,
    param_grid: dict[str, list] | None = None,
    tune: bool = True,
    min_train_periods: int = 3,
    cv_folds: int | None = None,
    holdout_periods: int = 1,
    quantile_alpha: float | None = None,
    embargo_periods: int = 0,
) -> RegressionTrainResult:
    """Fit a regression model with expanding-window or time-series CV.

    Parameters
    ----------
    X
        Feature matrix (observation index, feature columns).  NaN is
        median-imputed.
    y
        1-month forward returns (target vector, aligned with *X*).
    model_type
        ``"rf"`` for Random Forest, ``"xgb"`` for XGBoost, ``"lgb"``
        for LightGBM.
    period_labels
        Month labels for walk-forward split (aligned with *X*/*y*).  When
        provided, the last *holdout_periods* periods become the hold-out
        set and :class:`ExpandingWindowSplit` is used for CV on the
        remaining periods.  When ``None``, a 75/25 chronological split
        with :class:`~sklearn.model_selection.TimeSeriesSplit` is used.
    random_state
        Reproducibility seed; defaults to ``config.RANDOM_SEED``.
    param_grid
        Custom hyperparameter grid; when ``None``, a default grid is used.
    tune
        If ``False``, skip GridSearch and use default hyperparameters.
    min_train_periods
        Minimum training months before first expanding-window CV split.
    cv_folds
        Fallback CV folds when expanding-window yields too few splits.
    holdout_periods
        Number of trailing periods to reserve for hold-out evaluation.
        Defaults to ``1`` (last period only).
    quantile_alpha
        When set (e.g. ``0.5`` for the median), switches the LightGBM
        objective to ``"quantile"`` regression.  More robust to outlier
        returns than MSE.  Only valid for ``model_type="lgb"``.
    embargo_periods
        Number of periods to skip between training and validation in
        expanding-window CV (purged walk-forward).  ``1`` leaves a
        1-month buffer to prevent information leakage.

    Returns
    -------
    RegressionTrainResult
        Fitted model, imputation medians, feature names, CV and hold-out
        metrics (IC, RMSE, MAE, R²).
    """
    rs = RANDOM_SEED if random_state is None else random_state

    if model_type not in ("rf", "xgb", "lgb"):
        raise ValueError(
            f"model_type must be 'rf', 'xgb', or 'lgb', got {model_type!r}",
        )

    if quantile_alpha is not None and model_type != "lgb":
        raise ValueError(
            "quantile_alpha is only supported for model_type='lgb'",
        )

    prep = _prepare_regression_data(
        X,
        y,
        period_labels=period_labels,
        min_train_periods=min_train_periods,
        cv_folds=cv_folds,
        holdout_periods=holdout_periods,
        embargo_periods=embargo_periods,
    )

    if model_type == "lgb":
        from lightgbm import LGBMRegressor

        lgb_kwargs: dict[str, Any] = {
            "random_state": rs,
            "n_jobs": -1,
            "verbosity": -1,
            "force_col_wise": True,
        }
        if quantile_alpha is not None:
            lgb_kwargs["objective"] = "quantile"
            lgb_kwargs["alpha"] = quantile_alpha
            logger.info(
                "LightGBM quantile regression (alpha=%.2f)", quantile_alpha,
            )
        base = LGBMRegressor(**lgb_kwargs)
        grid = param_grid or _lgb_reg_param_grid()
    elif model_type == "xgb":
        from xgboost import XGBRegressor

        base = XGBRegressor(
            random_state=rs,
            n_jobs=-1,
            verbosity=0,
            objective="reg:squarederror",
        )
        grid = param_grid or _xgb_reg_param_grid()
    else:
        base = RandomForestRegressor(random_state=rs, n_jobs=-1)
        grid = param_grid or _rf_reg_param_grid()

    if tune:
        n_combos = _grid_size(grid)
        n_cv_label = (
            prep.cv.get_n_splits()
            if hasattr(prep.cv, "get_n_splits")
            else "?"
        )
        logger.info(
            "GridSearchCV: %d param combos × %s folds (scoring=IC)",
            n_combos,
            n_cv_label,
        )
        gs = GridSearchCV(
            base,
            grid,
            cv=prep.cv,
            scoring=_ic_scorer,
            n_jobs=-1,
            refit=True,
            return_train_score=False,
        )
        gs.fit(prep.X_train.values, prep.y_train)
        best_model = gs.best_estimator_
        best_params = gs.best_params_
        logger.info("Best params: %s  |  CV IC=%.4f", best_params, gs.best_score_)
    else:
        base.fit(prep.X_train.values, prep.y_train)
        best_model = base
        best_params = {}

    return _finalize_result(best_model, prep, best_params, model_type)


def train_regression_ensemble(
    X: pd.DataFrame,
    y: np.ndarray | pd.Series,
    *,
    period_labels: np.ndarray | pd.Series | None = None,
    random_state: int | None = None,
    rf_param_grid: dict[str, list] | None = None,
    xgb_param_grid: dict[str, list] | None = None,
    tune: bool = True,
    min_train_periods: int = 3,
    cv_folds: int | None = None,
    weights: tuple[float, float] = (0.5, 0.5),
    holdout_periods: int = 1,
    embargo_periods: int = 0,
) -> RegressionTrainResult:
    """Train RF + XGBoost separately, then blend with weighted-average predictions.

    Hyperparameters are tuned independently with the same CV splitter.  The
    final estimator is an :class:`AveragedRegressor`.

    Parameters
    ----------
    rf_param_grid, xgb_param_grid
        Optional grids; defaults match :func:`train_regressor` for each model.
    weights
        ``(rf_weight, xgb_weight)`` for the ensemble average (default equal).
    holdout_periods
        Number of trailing periods to reserve for hold-out evaluation.
        Defaults to ``1`` (last period only).
    embargo_periods
        Purged walk-forward embargo (see :func:`train_regressor`).
    Other parameters
        Same semantics as :func:`train_regressor`.
    """
    rs = RANDOM_SEED if random_state is None else random_state

    prep = _prepare_regression_data(
        X,
        y,
        period_labels=period_labels,
        min_train_periods=min_train_periods,
        cv_folds=cv_folds,
        holdout_periods=holdout_periods,
        embargo_periods=embargo_periods,
    )

    from xgboost import XGBRegressor

    rf_base = RandomForestRegressor(random_state=rs, n_jobs=-1)
    xgb_base = XGBRegressor(
        random_state=rs,
        n_jobs=-1,
        verbosity=0,
        objective="reg:squarederror",
    )
    grid_rf = rf_param_grid or _rf_reg_param_grid()
    grid_xgb = xgb_param_grid or _xgb_reg_param_grid()

    if tune:
        n_cv_label = (
            prep.cv.get_n_splits()
            if hasattr(prep.cv, "get_n_splits")
            else "?"
        )
        logger.info(
            "Ensemble GridSearchCV: RF %d × %s folds; XGB %d × %s folds",
            _grid_size(grid_rf),
            n_cv_label,
            _grid_size(grid_xgb),
            n_cv_label,
        )

        gs_rf = GridSearchCV(
            rf_base,
            grid_rf,
            cv=prep.cv,
            scoring=_ic_scorer,
            n_jobs=-1,
            refit=True,
            return_train_score=False,
        )
        gs_rf.fit(prep.X_train.values, prep.y_train)

        gs_xgb = GridSearchCV(
            xgb_base,
            grid_xgb,
            cv=prep.cv,
            scoring=_ic_scorer,
            n_jobs=-1,
            refit=True,
            return_train_score=False,
        )
        gs_xgb.fit(prep.X_train.values, prep.y_train)

        rf_template = clone(gs_rf.best_estimator_)
        xgb_template = clone(gs_xgb.best_estimator_)
        best_params: dict[str, Any] = {
            "rf": gs_rf.best_params_,
            "xgb": gs_xgb.best_params_,
        }
        logger.info(
            "Ensemble best RF params (CV IC=%.4f): %s",
            gs_rf.best_score_,
            gs_rf.best_params_,
        )
        logger.info(
            "Ensemble best XGB params (CV IC=%.4f): %s",
            gs_xgb.best_score_,
            gs_xgb.best_params_,
        )
    else:
        rf_template = rf_base
        xgb_template = xgb_base
        best_params = {"rf": {}, "xgb": {}}

    best_model = AveragedRegressor(
        rf_template,
        xgb_template,
        weights=weights,
    )
    best_model.fit(prep.X_train.values, prep.y_train)

    return _finalize_result(best_model, prep, best_params, "ensemble")


def _finalize_result(
    model: Any,
    prep: _PreparedRegressionData,
    best_params: dict[str, Any],
    model_type: str,
) -> RegressionTrainResult:
    """Compute holdout + CV metrics and build the result container."""
    y_pred_test = model.predict(prep.X_test.values)
    holdout = evaluate_regression(
        prep.y_test,
        y_pred_test,
        period_labels=prep.pl_test,
    )
    logger.info(
        "Hold-out: IC=%.4f  RMSE=%.4f  MAE=%.4f  R²=%.4f",
        holdout["ic"],
        holdout["rmse"],
        holdout["mae"],
        holdout["r2"],
    )

    cv_ic_scores: list[float] = []
    cv_rmse_scores: list[float] = []
    for train_idx, val_idx in prep.cv.split(prep.X_train.values, prep.y_train):
        fold_model = clone(model)
        fold_model.fit(prep.X_train.values[train_idx], prep.y_train[train_idx])
        fold_pred = fold_model.predict(prep.X_train.values[val_idx])
        cv_ic_scores.append(_spearman_ic(prep.y_train[val_idx], fold_pred))
        cv_rmse_scores.append(
            float(np.sqrt(mean_squared_error(prep.y_train[val_idx], fold_pred)))
        )

    cv_results = {
        "ic_mean": float(np.nanmean(cv_ic_scores)),
        "ic_std": float(np.nanstd(cv_ic_scores)),
        "ic_per_fold": cv_ic_scores,
        "rmse_mean": float(np.nanmean(cv_rmse_scores)),
        "rmse_std": float(np.nanstd(cv_rmse_scores)),
        "rmse_per_fold": cv_rmse_scores,
        "n_folds": len(cv_ic_scores),
    }
    logger.info(
        "CV: IC=%.4f ± %.4f  RMSE=%.4f ± %.4f  (%d folds)",
        cv_results["ic_mean"],
        cv_results["ic_std"],
        cv_results["rmse_mean"],
        cv_results["rmse_std"],
        cv_results["n_folds"],
    )

    return RegressionTrainResult(
        model=model,
        imputer_medians=prep.medians,
        feature_names=prep.feature_names,
        best_params=best_params,
        cv_results=cv_results,
        holdout_metrics=holdout,
        X_train=prep.X_train,
        X_test=prep.X_test,
        y_train=prep.y_train,
        y_test=prep.y_test,
        period_labels_train=prep.pl_train,
        period_labels_test=prep.pl_test,
        model_type=model_type,
    )


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def predict_returns(
    result: RegressionTrainResult,
    X: pd.DataFrame,
) -> pd.Series:
    """Predicted 1-month forward returns for new data.

    Parameters
    ----------
    result
        A :class:`RegressionTrainResult` from :func:`train_regressor` or
        :func:`train_regression_ensemble`.
    X
        Feature DataFrame with the same columns used during training.

    Returns
    -------
    pd.Series
        Predicted returns, indexed like *X*.
    """
    X_prep = _apply_imputation(X, result.imputer_medians, result.feature_names)
    preds = result.model.predict(X_prep.values)
    return pd.Series(preds, index=X_prep.index, name="predicted_return")


# ---------------------------------------------------------------------------
# Refit on full data
# ---------------------------------------------------------------------------


def refit_regressor_full(
    X: pd.DataFrame,
    y: np.ndarray | pd.Series,
    result: RegressionTrainResult,
    random_state: int | None = None,
) -> RegressionTrainResult:
    """Retrain the best model on *all* data (no hold-out).

    Uses the same hyperparameters discovered during :func:`train_regressor`.
    The returned :class:`RegressionTrainResult` is ready for forward-test
    predictions.
    """
    rs = RANDOM_SEED if random_state is None else random_state

    y_arr = np.asarray(y, dtype=float)
    valid = np.isfinite(y_arr)
    if hasattr(X, "notna"):
        valid &= X.notna().any(axis=1).values
    X_clean = X.loc[valid].copy()
    y_clean = y_arr[valid]

    X_imp, medians = _impute_median(X_clean)
    shared_cols = [c for c in result.feature_names if c in X_imp.columns]
    X_imp = X_imp[shared_cols]

    model = clone(result.model)
    if hasattr(model, "random_state"):
        model.set_params(random_state=rs)
    model.fit(X_imp.values, y_clean)

    logger.info(
        "Refit on full data: %d samples × %d features",
        len(X_imp),
        len(shared_cols),
    )

    return RegressionTrainResult(
        model=model,
        imputer_medians=medians,
        feature_names=shared_cols,
        best_params=result.best_params,
        cv_results=result.cv_results,
        holdout_metrics=result.holdout_metrics,
        X_train=X_imp,
        X_test=pd.DataFrame(),
        y_train=y_clean,
        y_test=np.array([]),
        model_type=result.model_type,
    )


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------


def shap_explain_regression(result: RegressionTrainResult) -> dict[str, Any]:
    """SHAP TreeExplainer analysis on the trained regression model.

    For :class:`AveragedRegressor` ensembles the Random Forest component is
    used (both sub-models share the same feature space).

    Returns
    -------
    dict
        ``shap_values`` (2-D array *n_samples × n_features*),
        ``X_display`` (DataFrame), ``feature_names``,
        ``mean_abs_shap`` (Series sorted descending), ``explainer``.
    """
    import shap

    if not result.X_test.empty:
        X_all = pd.concat([result.X_train, result.X_test])
    else:
        X_all = result.X_train

    tree_model = result.model
    if isinstance(tree_model, AveragedRegressor):
        tree_model = tree_model.rf_

    explainer = shap.TreeExplainer(tree_model)
    sv = explainer.shap_values(X_all.values)
    sv_arr = np.asarray(sv)

    mean_abs = pd.Series(
        np.abs(sv_arr).mean(axis=0),
        index=result.feature_names,
        name="mean_abs_shap",
    )
    mean_abs = mean_abs.sort_values(ascending=False)

    return {
        "shap_values": sv_arr,
        "X_display": X_all,
        "feature_names": result.feature_names,
        "mean_abs_shap": mean_abs,
        "explainer": explainer,
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def plot_ic_by_period(
    ic_df: pd.DataFrame,
    *,
    title: str = "IC (Spearman) by Period",
    ax: Any | None = None,
) -> Any:
    """Bar chart of per-period IC with mean reference line.

    Parameters
    ----------
    ic_df
        Output of :func:`compute_ic_by_period` (columns ``period``, ``ic``, ``n``).
    """
    import matplotlib.pyplot as plt

    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(12, 4))

    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in ic_df["ic"]]
    ax.bar(
        range(len(ic_df)),
        ic_df["ic"],
        color=colors,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_xticks(range(len(ic_df)))
    ax.set_xticklabels(
        ic_df["period"].astype(str), rotation=45, ha="right", fontsize=8,
    )

    ic_mean = float(ic_df["ic"].mean())
    ax.axhline(
        ic_mean,
        color="#3498db",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean IC = {ic_mean:.3f}",
    )
    ax.axhline(0, color="grey", linestyle="-", linewidth=0.5)
    ax.set_ylabel("IC (Spearman)")
    ax.set_title(title)
    ax.legend(loc="upper right")

    if created:
        plt.tight_layout()
    return ax


def plot_predicted_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    title: str = "Predicted vs Actual Returns",
    ax: Any | None = None,
) -> Any:
    """Scatter plot with Spearman correlation annotation."""
    import matplotlib.pyplot as plt

    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(6, 6))

    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[mask], yp[mask]
    ic = _spearman_ic(yt, yp)

    ax.scatter(yp, yt, alpha=0.3, s=10, color="#3498db")
    lo = min(float(yt.min()), float(yp.min()))
    hi = max(float(yt.max()), float(yp.max()))
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Predicted Return")
    ax.set_ylabel("Actual Return")
    ax.set_title(f"{title}  (IC = {ic:.3f})")

    if created:
        plt.tight_layout()
    return ax


def plot_feature_importance_regression(
    mean_abs_shap: pd.Series,
    *,
    top_n: int = 20,
    title: str = "SHAP Feature Importance (Regression)",
    ax: Any | None = None,
) -> Any:
    """Horizontal bar chart of mean |SHAP| values.

    Parameters
    ----------
    mean_abs_shap
        ``"mean_abs_shap"`` key from :func:`shap_explain_regression`.
    top_n
        Number of top features to display.
    """
    import matplotlib.pyplot as plt

    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(8, max(4, top_n * 0.3)))

    top = mean_abs_shap.head(top_n).sort_values()
    ax.barh(range(len(top)), top.values, color="#3498db")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index, fontsize=9)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(title)

    if created:
        plt.tight_layout()
    return ax


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_regression_model(result: RegressionTrainResult, path: str | Path) -> Path:
    """Serialise a :class:`RegressionTrainResult` to disk via joblib."""
    import joblib

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result, p)
    logger.info("Regression model saved to %s", p)
    return p


def load_regression_model(path: str | Path) -> RegressionTrainResult:
    """Load a :class:`RegressionTrainResult` from disk."""
    import joblib

    result = joblib.load(Path(path))
    if not isinstance(result, RegressionTrainResult):
        raise TypeError(
            f"Expected RegressionTrainResult, got {type(result).__name__}",
        )
    return result
