"""ML training, CV, and prediction (Random Forest / XGBoost / LightGBM / ensemble).

Phase 5 of the Swiss SPI Extra pipeline — stratified hold-out split,
stratified k-fold cross-validation, hyperparameter tuning via GridSearch,
optional probability calibration (``CalibratedClassifierCV``), thresholded
Winner predictions, SHAP explanations, and model persistence.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import reduce
from pathlib import Path
from typing import Any, Literal

ProblemType = Literal["multiclass", "binary"]

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.feature_selection import RFECV
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.preprocessing import LabelEncoder

try:
    from config import CV_FOLDS, RANDOM_SEED, TRAIN_TEST_SPLIT
except ImportError:
    RANDOM_SEED = 42
    TRAIN_TEST_SPLIT = 0.75
    CV_FOLDS = 5

logger = logging.getLogger(__name__)

__all__ = [
    "AveragedProbaEnsemble",
    "MultiQuarterEnsembleResult",
    "ProblemType",
    "RegimeModelCollection",
    "TrainResult",
    "WalkForwardSplit",
    "evaluate_predictions",
    "load_model",
    "permutation_importance_walk_forward",
    "plot_confusion_matrix",
    "plot_cv_folds",
    "plot_model_comparison",
    "predict",
    "predict_proba",
    "predict_proba_regime_aware",
    "predict_regime_aware",
    "predict_with_threshold",
    "refit_on_full_data",
    "save_model",
    "shap_explain",
    "shap_explain_per_regime",
    "shap_regime_summary",
    "train_classifier",
    "train_ensemble",
    "train_multi_quarter_ensemble",
    "train_regime_aware_models",
]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class TrainResult:
    """Container for training artefacts and evaluation metrics."""

    model: Any
    label_encoder: LabelEncoder
    imputer_medians: pd.Series
    feature_names: list[str]
    class_names: list[str]
    best_params: dict[str, Any]
    cv_results: dict[str, Any]
    holdout_metrics: dict[str, Any]
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train_enc: np.ndarray
    y_test_enc: np.ndarray
    model_type: str = "rf"  # "rf" | "xgb" | "lgb" | "ensemble"
    selected_features: list[str] | None = None
    calibrated: bool = False
    classification_mode: ProblemType = "multiclass"


@dataclass
class MultiQuarterEnsembleResult:
    """Consensus ensemble of classifiers, each trained on a single quarter.

    Predictions are aggregated via weighted probability averaging (default)
    or majority vote across sub-models, reducing variance from single-quarter
    training instability.
    """

    sub_results: list[TrainResult]
    period_labels: list[str]
    weights: np.ndarray
    consensus_method: str  # "proba_average" | "majority"
    class_names: list[str]
    model_type: str = "multi_quarter_ensemble"


@dataclass
class RegimeModelCollection:
    """Regime-specific classifiers with optional fallback for underrepresented regimes.

    Regimes with >= ``min_regime_quarters`` quarters receive a dedicated
    :class:`MultiQuarterEnsembleResult`.  Underrepresented regimes share a
    single ``fallback_model`` trained on all quarters with one-hot regime
    indicators as additional features.
    """

    regime_models: dict[str, MultiQuarterEnsembleResult]
    fallback_model: TrainResult | None
    fallback_regimes: list[str]
    regime_period_map: dict[str, list[str]]  # regime_label → [period_labels]
    period_regime_map: dict[str, str]  # period_label → regime_label
    class_names: list[str]
    min_regime_quarters: int
    model_type: str = "regime_aware"


# ---------------------------------------------------------------------------
# Walk-forward cross-validation
# ---------------------------------------------------------------------------


class WalkForwardSplit:
    """Expanding-window walk-forward splitter for temporally-ordered periods.

    Train on periods ``0..k``, validate on period ``k+1``. Compatible with
    scikit-learn's CV interface (``split(X, y, groups)``).

    Parameters
    ----------
    period_labels
        Ordered list/array of period identifiers aligned row-by-row with
        the training data (e.g. ``["Q2-2023", "Q2-2023", ..., "Q3-2023", ...]``).
        The unique values determine temporal ordering, so they must be
        chronologically sorted when deduplicated.
    min_train_periods
        Minimum number of training periods before the first split.
        Defaults to 2 (train on >= 2 periods, test on the next).
    """

    def __init__(
        self,
        period_labels: np.ndarray | pd.Series | list[str],
        *,
        min_train_periods: int = 2,
    ) -> None:
        self.period_labels = np.asarray(period_labels)
        seen: list[str] = []
        for p in self.period_labels:
            if p not in seen:
                seen.append(p)
        self.unique_periods: list[str] = seen
        self.min_train_periods = max(1, min_train_periods)

    def get_n_splits(
        self,
        X: Any = None,
        y: Any = None,
        groups: Any = None,
    ) -> int:
        return max(0, len(self.unique_periods) - self.min_train_periods)

    def split(
        self,
        X: Any = None,
        y: Any = None,
        groups: Any = None,
    ):
        """Yield (train_indices, test_indices) tuples."""
        for k in range(self.min_train_periods, len(self.unique_periods)):
            train_periods = set(self.unique_periods[:k])
            test_period = self.unique_periods[k]

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


def _rf_param_grid() -> dict[str, list]:
    return {
        "n_estimators": [100, 200, 500],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 3],
        "max_features": ["sqrt", "log2"],
    }


def _xgb_param_grid() -> dict[str, list]:
    return {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.2],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }


def _lgb_param_grid() -> dict[str, list]:
    return {
        "n_estimators": [100, 300],
        "max_depth": [3, 7],
        "learning_rate": [0.05, 0.1],
        "num_leaves": [31, 63],
    }


def _grid_size(grid: dict[str, list]) -> int:
    return reduce(lambda a, b: a * b, (len(v) for v in grid.values()), 1)


@dataclass
class _PreparedTrainData:
    """Split + imputation + optional RFECV artefacts shared by training entry points."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    cv: Any
    feature_names: list[str]
    le: LabelEncoder
    medians: pd.Series
    selected_features: list[str] | None
    cv_scoring: str
    class_names: list[str]


def _prepare_train_classifier_data(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    problem_type: ProblemType,
    test_size: float | None,
    cv_folds: int | None,
    random_state: int,
    period_labels: pd.Series | np.ndarray | None,
    walk_forward: bool,
    min_train_periods: int,
    feature_selection: bool,
    rfecv_min_features: int,
) -> _PreparedTrainData:
    """Encode labels, split, median-impute from train only, optional RFECV — shared by :func:`train_classifier` / :func:`train_ensemble`."""
    ts = (1.0 - TRAIN_TEST_SPLIT) if test_size is None else test_size
    n_cv = CV_FOLDS if cv_folds is None else cv_folds

    common = X.index.intersection(y.dropna().index)
    X_clean = X.loc[common].copy()
    y_clean = y.loc[common].copy()

    if problem_type == "binary":
        y_clean = _to_binary_labels(y_clean)

    le = LabelEncoder()
    y_enc = le.fit_transform(y_clean.values)
    class_names = list(le.classes_)

    cv_scoring = "f1" if problem_type == "binary" else "f1_macro"
    if problem_type == "binary":
        logger.info(
            "Binary classification (Winner vs Rest): %d Winners / %d Rest",
            int((y_clean == "Winners").sum()),
            int((y_clean == "Rest").sum()),
        )

    if walk_forward:
        if period_labels is None:
            raise ValueError("period_labels is required when walk_forward=True")

        pl = np.asarray(period_labels)
        pl_aligned = pl[np.isin(np.arange(len(pl)),
                                np.where(np.isin(X.index, common))[0])] \
            if len(pl) == len(X) else pl

        if len(pl_aligned) != len(X_clean):
            pl_series = pd.Series(pl, index=X.index)
            pl_aligned = pl_series.loc[common].values

        unique_periods: list[str] = []
        for p in pl_aligned:
            if p not in unique_periods:
                unique_periods.append(p)

        last_period = unique_periods[-1]
        train_mask = pl_aligned != last_period
        test_mask = pl_aligned == last_period

        X_train = X_clean[train_mask].copy()
        X_test = X_clean[test_mask].copy()
        y_train = y_enc[train_mask]
        y_test = y_enc[test_mask]
        pl_train = pl_aligned[train_mask]

        logger.info(
            "Walk-forward split: %d train (%d periods) / %d test (period %s), %d raw features",
            len(X_train), len(unique_periods) - 1, len(X_test),
            last_period, X_train.shape[1],
        )

        wf_cv = WalkForwardSplit(pl_train, min_train_periods=min_train_periods)
        n_splits = wf_cv.get_n_splits()
        if n_splits < 1:
            logger.warning(
                "Walk-forward CV has 0 splits (only %d training periods, "
                "need >%d). Falling back to stratified k-fold on training set.",
                len(set(pl_train)), min_train_periods,
            )
            cv: Any = StratifiedKFold(
                n_splits=n_cv, shuffle=True, random_state=random_state,
            )
        else:
            cv = wf_cv
            logger.info("Walk-forward CV: %d splits", n_splits)
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_clean,
            y_enc,
            test_size=ts,
            random_state=random_state,
            stratify=y_enc,
        )

        logger.info(
            "Split: %d train / %d test (%.0f%% hold-out), %d raw features",
            len(X_train),
            len(X_test),
            ts * 100,
            X_train.shape[1],
        )

        cv = StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=random_state)

    X_train_imp, medians = _impute_median(X_train)
    feature_names = list(X_train_imp.columns)
    X_test_imp = _apply_imputation(X_test, medians, feature_names)
    X_train = X_train_imp
    X_test = X_test_imp

    selected_features: list[str] | None = None
    if feature_selection:
        rfe_estimator = RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced",
        )
        rfe_cv = StratifiedKFold(
            n_splits=min(n_cv, 5), shuffle=True, random_state=random_state,
        )
        rfecv = RFECV(
            estimator=rfe_estimator,
            step=1,
            cv=rfe_cv,
            scoring=cv_scoring,
            min_features_to_select=rfecv_min_features,
            n_jobs=-1,
        )
        rfecv.fit(X_train.values, y_train)

        mask = rfecv.support_
        selected_features = [f for f, sel in zip(feature_names, mask) if sel]

        n_before = len(feature_names)
        n_after = len(selected_features)
        logger.info(
            "RFECV: %d → %d features (dropped %d): %s",
            n_before,
            n_after,
            n_before - n_after,
            [f for f, sel in zip(feature_names, mask) if not sel],
        )

        X_train = X_train[selected_features]
        X_test = X_test[selected_features]
        feature_names = selected_features

    return _PreparedTrainData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        cv=cv,
        feature_names=feature_names,
        le=le,
        medians=medians,
        selected_features=selected_features,
        cv_scoring=cv_scoring,
        class_names=class_names,
    )


class AveragedProbaEnsemble(BaseEstimator, ClassifierMixin):
    """Blend two classifiers by averaging ``predict_proba`` (then argmax for ``predict``).

    Used by :func:`train_ensemble` with Random Forest + XGBoost. Optional
    per-model probability calibration matches :func:`train_classifier`.
    """

    def __init__(
        self,
        rf_estimator: Any = None,
        xgb_estimator: Any = None,
        *,
        calibrate: bool = False,
        calibration_method: Literal["sigmoid", "isotonic"] = "sigmoid",
        random_state: int | None = None,
    ) -> None:
        self.rf_estimator = rf_estimator
        self.xgb_estimator = xgb_estimator
        self.calibrate = calibrate
        self.calibration_method = calibration_method
        self.random_state = random_state

    def fit(self, X: np.ndarray | pd.DataFrame, y: np.ndarray) -> AveragedProbaEnsemble:
        rs = RANDOM_SEED if self.random_state is None else self.random_state
        rf = clone(self.rf_estimator).fit(X, y)
        xgb = clone(self.xgb_estimator).fit(X, y)
        if self.calibrate:
            _, counts = np.unique(y, return_counts=True)
            min_class = int(counts.min())
            n_splits_cal = min(5, max(2, min_class))
            if n_splits_cal < 2 or min_class < 2:
                logger.warning(
                    "Ensemble: skipping per-model calibration (min class count=%d)",
                    min_class,
                )
            else:
                cal_cv = StratifiedKFold(
                    n_splits=n_splits_cal,
                    shuffle=True,
                    random_state=rs,
                )
                rf = CalibratedClassifierCV(
                    clone(rf),
                    method=self.calibration_method,
                    cv=cal_cv,
                    n_jobs=-1,
                )
                rf.fit(X, y)
                xgb = CalibratedClassifierCV(
                    clone(xgb),
                    method=self.calibration_method,
                    cv=cal_cv,
                    n_jobs=-1,
                )
                xgb.fit(X, y)
        self.rf_ = rf
        self.xgb_ = xgb
        self.classes_ = np.asarray(self.rf_.classes_)
        return self

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        p1 = self.rf_.predict_proba(X)
        p2 = self.xgb_.predict_proba(X)
        return (p1 + p2) / 2.0

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


# ---------------------------------------------------------------------------
# Core API
# ---------------------------------------------------------------------------


def _to_binary_labels(y: pd.Series) -> pd.Series:
    """Map 3-class pipeline labels to binary *Winner vs Rest*."""
    return y.map(lambda g: "Winners" if g == "Winners" else "Rest")


def train_classifier(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int,
    *,
    model_type: str = "rf",
    problem_type: ProblemType = "multiclass",
    test_size: float | None = None,
    cv_folds: int | None = None,
    param_grid: dict[str, list] | None = None,
    tune: bool = True,
    period_labels: pd.Series | np.ndarray | None = None,
    walk_forward: bool = False,
    min_train_periods: int = 2,
    feature_selection: bool = False,
    rfecv_min_features: int = 5,
    calibrate: bool = False,
    calibration_method: Literal["sigmoid", "isotonic"] = "sigmoid",
) -> TrainResult:
    """Fit classifier with stratified hold-out split, CV, and optional grid search.

    Parameters
    ----------
    X
        Feature matrix (ticker index, feature columns). NaNs are median-imputed
        using training-split statistics only.
    y
        Class labels (``Winners`` / ``Steady`` / ``Losers``). When
        ``problem_type="binary"``, Steady and Losers are merged into *Rest*;
        the model learns **Winner vs Rest** (Winners positive).
    random_state
        Reproducibility seed.
    model_type
        ``"rf"`` for Random Forest, ``"xgb"`` for XGBoost, ``"lgb"`` for LightGBM.
    problem_type
        ``"multiclass"`` (3-way) or ``"binary"`` (Winners vs Rest). Grid search
        and CV use ``f1`` for binary (positive class = Winners) and
        ``f1_macro`` for multiclass.
    test_size
        Hold-out fraction; defaults to ``1 - config.TRAIN_TEST_SPLIT``.
        Ignored when ``walk_forward=True`` (the last period is used as hold-out).
    cv_folds
        Stratified CV folds; defaults to ``config.CV_FOLDS``.
        Ignored when ``walk_forward=True``.
    param_grid
        Custom hyperparameter grid; when ``None``, a default grid is used.
    tune
        If ``False``, skip GridSearch and use default hyperparameters.
    period_labels
        Period identifiers aligned with ``X`` / ``y`` (e.g. ``"Q2-2023"``).
        Required when ``walk_forward=True``.
    walk_forward
        If ``True``, use expanding-window walk-forward CV instead of
        stratified k-fold. The last period becomes the hold-out set and
        :class:`WalkForwardSplit` is used for CV on the remaining periods.
    min_train_periods
        Minimum training periods for walk-forward splits (default 2).
    feature_selection
        If ``True``, run RFECV on the training set before hyperparameter
        tuning to eliminate weak features. The surviving feature names are
        stored in ``TrainResult.selected_features``.
    rfecv_min_features
        Minimum number of features to retain during RFECV (default 5).
    calibrate
        If ``True``, wrap the tuned base estimator in
        :class:`~sklearn.calibration.CalibratedClassifierCV` (sigmoid or
        isotonic) fit on the training fold; improves probability quality for
        :func:`predict_proba` / :func:`predict_with_threshold`.
    calibration_method
        ``"sigmoid"`` (Platt) or ``"isotonic"``; passed to
        ``CalibratedClassifierCV(method=...)``.

    Returns
    -------
    TrainResult
        Fitted model, encoder, imputation medians, CV and hold-out metrics.
    """
    if model_type not in ("rf", "xgb", "lgb"):
        raise ValueError(
            f"model_type must be 'rf', 'xgb', or 'lgb', got {model_type!r}",
        )

    prep = _prepare_train_classifier_data(
        X,
        y,
        problem_type=problem_type,
        test_size=test_size,
        cv_folds=cv_folds,
        random_state=random_state,
        period_labels=period_labels,
        walk_forward=walk_forward,
        min_train_periods=min_train_periods,
        feature_selection=feature_selection,
        rfecv_min_features=rfecv_min_features,
    )
    X_train = prep.X_train
    X_test = prep.X_test
    y_train = prep.y_train
    y_test = prep.y_test
    cv = prep.cv
    feature_names = prep.feature_names
    le = prep.le
    medians = prep.medians
    selected_features = prep.selected_features
    cv_scoring = prep.cv_scoring
    class_names = prep.class_names

    if model_type == "xgb":
        from xgboost import XGBClassifier

        base = XGBClassifier(
            random_state=random_state,
            n_jobs=-1,
            verbosity=0,
        )
        grid = param_grid or _xgb_param_grid()
    elif model_type == "lgb":
        from lightgbm import LGBMClassifier

        base = LGBMClassifier(
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1,
            force_col_wise=True,
        )
        grid = param_grid or _lgb_param_grid()
    else:
        base = RandomForestClassifier(
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced",
        )
        grid = param_grid or _rf_param_grid()

    if tune:
        n_combos = _grid_size(grid)
        n_cv_label = cv.get_n_splits() if hasattr(cv, "get_n_splits") else "?"
        logger.info("GridSearchCV: %d param combos × %s folds", n_combos, n_cv_label)
        gs = GridSearchCV(
            base,
            grid,
            cv=cv,
            scoring=cv_scoring,
            n_jobs=-1,
            refit=True,
            return_train_score=True,
        )
        gs.fit(X_train.values, y_train)
        best_model = gs.best_estimator_
        best_params = gs.best_params_
        logger.info(
            "Best params: %s  |  CV %s=%.4f",
            best_params,
            "F1(Winners)" if problem_type == "binary" else "F1-macro",
            gs.best_score_,
        )
    else:
        base.fit(X_train.values, y_train)
        best_model = base
        best_params = {
            k: v
            for k, v in base.get_params().items()
            if k in (grid or {})
        }

    calibrated = False
    if calibrate:
        _, counts = np.unique(y_train, return_counts=True)
        min_class = int(counts.min())
        n_splits_cal = min(5, max(2, min_class))
        if n_splits_cal < 2 or min_class < 2:
            logger.warning(
                "Skipping probability calibration: need at least 2 samples per "
                "class for stratified CV (min class count=%d)",
                min_class,
            )
        else:
            cal_cv = StratifiedKFold(
                n_splits=n_splits_cal,
                shuffle=True,
                random_state=random_state,
            )
            best_model = CalibratedClassifierCV(
                clone(best_model),
                method=calibration_method,
                cv=cal_cv,
                n_jobs=-1,
            )
            best_model.fit(X_train.values, y_train)
            calibrated = True
            logger.info(
                "Probability calibration (%s) fitted on training set (%d folds)",
                calibration_method,
                n_splits_cal,
            )

    if problem_type == "binary":
        cv_detail = cross_validate(
            clone(best_model),
            X_train.values,
            y_train,
            cv=cv,
            scoring=["accuracy", "f1", "f1_weighted"],
        )
        cv_results = {
            "accuracy_mean": float(cv_detail["test_accuracy"].mean()),
            "accuracy_std": float(cv_detail["test_accuracy"].std()),
            "f1_macro_mean": float(cv_detail["test_f1"].mean()),
            "f1_macro_std": float(cv_detail["test_f1"].std()),
            "f1_weighted_mean": float(cv_detail["test_f1_weighted"].mean()),
            "f1_weighted_std": float(cv_detail["test_f1_weighted"].std()),
            "fold_scores": {
                "accuracy": cv_detail["test_accuracy"].tolist(),
                "f1_macro": cv_detail["test_f1"].tolist(),
                "f1_weighted": cv_detail["test_f1_weighted"].tolist(),
            },
        }
    else:
        cv_detail = cross_validate(
            clone(best_model),
            X_train.values,
            y_train,
            cv=cv,
            scoring=["accuracy", "f1_macro", "f1_weighted"],
        )
        cv_results = {
            "accuracy_mean": float(cv_detail["test_accuracy"].mean()),
            "accuracy_std": float(cv_detail["test_accuracy"].std()),
            "f1_macro_mean": float(cv_detail["test_f1_macro"].mean()),
            "f1_macro_std": float(cv_detail["test_f1_macro"].std()),
            "f1_weighted_mean": float(cv_detail["test_f1_weighted"].mean()),
            "f1_weighted_std": float(cv_detail["test_f1_weighted"].std()),
            "fold_scores": {
                "accuracy": cv_detail["test_accuracy"].tolist(),
                "f1_macro": cv_detail["test_f1_macro"].tolist(),
                "f1_weighted": cv_detail["test_f1_weighted"].tolist(),
            },
        }

    y_pred = best_model.predict(X_test.values)
    holdout = evaluate_predictions(y_test, y_pred, class_names)

    logger.info(
        "Hold-out: Accuracy=%.3f  F1-macro=%.3f  F1-weighted=%.3f",
        holdout["accuracy"],
        holdout["f1_macro"],
        holdout["f1_weighted"],
    )

    return TrainResult(
        model=best_model,
        label_encoder=le,
        imputer_medians=medians,
        feature_names=feature_names,
        class_names=class_names,
        best_params=best_params,
        cv_results=cv_results,
        holdout_metrics=holdout,
        X_train=X_train,
        X_test=X_test,
        y_train_enc=y_train,
        y_test_enc=y_test,
        model_type=model_type,
        selected_features=selected_features,
        calibrated=calibrated,
        classification_mode=problem_type,
    )


def train_ensemble(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int,
    *,
    problem_type: ProblemType = "multiclass",
    test_size: float | None = None,
    cv_folds: int | None = None,
    rf_param_grid: dict[str, list] | None = None,
    xgb_param_grid: dict[str, list] | None = None,
    tune: bool = True,
    period_labels: pd.Series | np.ndarray | None = None,
    walk_forward: bool = False,
    min_train_periods: int = 2,
    feature_selection: bool = False,
    rfecv_min_features: int = 5,
    calibrate: bool = False,
    calibration_method: Literal["sigmoid", "isotonic"] = "sigmoid",
) -> TrainResult:
    """Train Random Forest and XGBoost separately, then blend with averaged ``predict_proba``.

    Hyperparameters are tuned independently with the same CV splitter as
    :func:`train_classifier`. The final estimator is an :class:`AveragedProbaEnsemble`
    (optional per-model calibration, then mean of probabilities). Works with
    :func:`predict_proba`, :func:`predict_with_threshold`, and persistence.

    Parameters
    ----------
    rf_param_grid, xgb_param_grid
        Optional grids; defaults match :func:`train_classifier` for each model.
    Other parameters
        Same semantics as :func:`train_classifier` (except ``model_type``).
    """
    prep = _prepare_train_classifier_data(
        X,
        y,
        problem_type=problem_type,
        test_size=test_size,
        cv_folds=cv_folds,
        random_state=random_state,
        period_labels=period_labels,
        walk_forward=walk_forward,
        min_train_periods=min_train_periods,
        feature_selection=feature_selection,
        rfecv_min_features=rfecv_min_features,
    )
    X_train = prep.X_train
    X_test = prep.X_test
    y_train = prep.y_train
    y_test = prep.y_test
    cv = prep.cv
    feature_names = prep.feature_names
    le = prep.le
    medians = prep.medians
    selected_features = prep.selected_features
    cv_scoring = prep.cv_scoring
    class_names = prep.class_names

    from xgboost import XGBClassifier

    rf_base = RandomForestClassifier(
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced",
    )
    xgb_base = XGBClassifier(
        random_state=random_state,
        n_jobs=-1,
        verbosity=0,
    )
    grid_rf = rf_param_grid or _rf_param_grid()
    grid_xgb = xgb_param_grid or _xgb_param_grid()

    if tune:
        n_cv_label = cv.get_n_splits() if hasattr(cv, "get_n_splits") else "?"
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
            cv=cv,
            scoring=cv_scoring,
            n_jobs=-1,
            refit=True,
            return_train_score=True,
        )
        gs_rf.fit(X_train.values, y_train)
        gs_xgb = GridSearchCV(
            xgb_base,
            grid_xgb,
            cv=cv,
            scoring=cv_scoring,
            n_jobs=-1,
            refit=True,
            return_train_score=True,
        )
        gs_xgb.fit(X_train.values, y_train)
        rf_template = clone(gs_rf.best_estimator_)
        xgb_template = clone(gs_xgb.best_estimator_)
        best_params: dict[str, Any] = {
            "rf": gs_rf.best_params_,
            "xgb": gs_xgb.best_params_,
        }
        logger.info(
            "Ensemble best RF params (CV %s=%.4f): %s",
            "F1(Winners)" if problem_type == "binary" else "F1-macro",
            gs_rf.best_score_,
            gs_rf.best_params_,
        )
        logger.info(
            "Ensemble best XGB params (CV %s=%.4f): %s",
            "F1(Winners)" if problem_type == "binary" else "F1-macro",
            gs_xgb.best_score_,
            gs_xgb.best_params_,
        )
    else:
        rf_template = rf_base
        xgb_template = xgb_base
        best_params = {
            "rf": {
                k: v
                for k, v in rf_base.get_params().items()
                if k in grid_rf
            },
            "xgb": {
                k: v
                for k, v in xgb_base.get_params().items()
                if k in grid_xgb
            },
        }

    best_model = AveragedProbaEnsemble(
        rf_template,
        xgb_template,
        calibrate=calibrate,
        calibration_method=calibration_method,
        random_state=random_state,
    )
    best_model.fit(X_train.values, y_train)
    calibrated = calibrate
    if calibrate:
        logger.info(
            "Ensemble: probability calibration (%s) on RF + XGB",
            calibration_method,
        )

    if problem_type == "binary":
        cv_detail = cross_validate(
            clone(best_model),
            X_train.values,
            y_train,
            cv=cv,
            scoring=["accuracy", "f1", "f1_weighted"],
        )
        cv_results = {
            "accuracy_mean": float(cv_detail["test_accuracy"].mean()),
            "accuracy_std": float(cv_detail["test_accuracy"].std()),
            "f1_macro_mean": float(cv_detail["test_f1"].mean()),
            "f1_macro_std": float(cv_detail["test_f1"].std()),
            "f1_weighted_mean": float(cv_detail["test_f1_weighted"].mean()),
            "f1_weighted_std": float(cv_detail["test_f1_weighted"].std()),
            "fold_scores": {
                "accuracy": cv_detail["test_accuracy"].tolist(),
                "f1_macro": cv_detail["test_f1"].tolist(),
                "f1_weighted": cv_detail["test_f1_weighted"].tolist(),
            },
        }
    else:
        cv_detail = cross_validate(
            clone(best_model),
            X_train.values,
            y_train,
            cv=cv,
            scoring=["accuracy", "f1_macro", "f1_weighted"],
        )
        cv_results = {
            "accuracy_mean": float(cv_detail["test_accuracy"].mean()),
            "accuracy_std": float(cv_detail["test_accuracy"].std()),
            "f1_macro_mean": float(cv_detail["test_f1_macro"].mean()),
            "f1_macro_std": float(cv_detail["test_f1_macro"].std()),
            "f1_weighted_mean": float(cv_detail["test_f1_weighted"].mean()),
            "f1_weighted_std": float(cv_detail["test_f1_weighted"].std()),
            "fold_scores": {
                "accuracy": cv_detail["test_accuracy"].tolist(),
                "f1_macro": cv_detail["test_f1_macro"].tolist(),
                "f1_weighted": cv_detail["test_f1_weighted"].tolist(),
            },
        }

    y_pred = best_model.predict(X_test.values)
    holdout = evaluate_predictions(y_test, y_pred, class_names)

    logger.info(
        "Ensemble hold-out: Accuracy=%.3f  F1-macro=%.3f  F1-weighted=%.3f",
        holdout["accuracy"],
        holdout["f1_macro"],
        holdout["f1_weighted"],
    )

    return TrainResult(
        model=best_model,
        label_encoder=le,
        imputer_medians=medians,
        feature_names=feature_names,
        class_names=class_names,
        best_params=best_params,
        cv_results=cv_results,
        holdout_metrics=holdout,
        X_train=X_train,
        X_test=X_test,
        y_train_enc=y_train,
        y_test_enc=y_test,
        model_type="ensemble",
        selected_features=selected_features,
        calibrated=calibrated,
        classification_mode=problem_type,
    )


def train_multi_quarter_ensemble(
    ohlcv_by_ticker: dict[str, pd.DataFrame],
    fundamentals_by_ticker: dict[str, dict[str, Any]] | None,
    periods: list[tuple[str, str, str, str]],
    random_state: int,
    *,
    model_type: str = "rf",
    problem_type: ProblemType = "multiclass",
    recency_decay: float = 0.85,
    consensus_method: str = "proba_average",
    drop_correlated: bool = True,
    tune: bool = True,
    calibrate: bool = False,
    calibration_method: Literal["sigmoid", "isotonic"] = "sigmoid",
    refit_full: bool = True,
    feature_selection: bool = False,
    rfecv_min_features: int = 5,
) -> MultiQuarterEnsembleResult:
    """Train one classifier per quarter and combine into a consensus ensemble.

    For each quarter in *periods*, builds features at the quarter's
    ``feature_cutoff``, computes return-based labels, trains a classifier,
    and optionally refits on all labelled data (no hold-out). The resulting
    sub-models predict independently on OOS data; their probabilities are
    aggregated via weighted averaging or majority vote.

    Parameters
    ----------
    ohlcv_by_ticker
        Ticker -> OHLCV DataFrame.
    fundamentals_by_ticker
        Ticker -> yfinance ``.info`` dict. Can be ``None``.
    periods
        List of ``(feature_cutoff, q_start, q_end, period_label)`` tuples
        from :data:`config.CLASSIFICATION_PERIODS`.
    random_state
        Reproducibility seed.
    model_type
        ``"rf"``, ``"xgb"``, ``"lgb"``, or ``"ensemble"`` (RF+XGB blend
        per quarter).
    problem_type
        ``"multiclass"`` or ``"binary"``.
    recency_decay
        Exponential decay for recency weighting.  Weight for quarter *i*
        (0 = oldest) is ``decay ** (n - 1 - i)``.  Set to ``1.0`` for
        equal weights across all quarters.
    consensus_method
        ``"proba_average"`` — weighted average of ``predict_proba``, then
        argmax.  ``"majority"`` — weighted majority vote of individual
        class predictions.
    drop_correlated
        Drop highly correlated features per quarter before training.
    tune
        Use ``GridSearchCV`` for each sub-model.
    calibrate
        Calibrate probabilities for each sub-model.
    calibration_method
        ``"sigmoid"`` (Platt) or ``"isotonic"``.
    refit_full
        Refit each model on all available data for that quarter (no
        hold-out) before storing, maximising training signal for OOS use.
    feature_selection
        Run RFECV per sub-model to eliminate weak features.
    rfecv_min_features
        Minimum features to retain during RFECV.

    Returns
    -------
    MultiQuarterEnsembleResult
    """
    try:
        from src.classifier import assign_groups, compute_q1_returns
        from src.features import (
            build_feature_matrix,
            drop_correlated_features as _drop_corr,
        )
    except ImportError:
        from classifier import assign_groups, compute_q1_returns
        from features import (
            build_feature_matrix,
            drop_correlated_features as _drop_corr,
        )

    sub_results: list[TrainResult] = []
    trained_labels: list[str] = []

    for i, (fc, q_start, q_end, plabel) in enumerate(periods):
        logger.info(
            "Multi-quarter ensemble: training sub-model %d/%d on period %s",
            i + 1, len(periods), plabel,
        )

        returns = compute_q1_returns(ohlcv_by_ticker, q_start=q_start, q_end=q_end)
        labels = assign_groups(returns)
        X = build_feature_matrix(ohlcv_by_ticker, fc, fundamentals_by_ticker)

        if drop_correlated:
            X, _ = _drop_corr(X)

        common = X.index.intersection(labels.dropna().index)
        if len(common) < 10:
            logger.warning(
                "Skipping period %s: only %d valid samples", plabel, len(common),
            )
            continue

        X_al = X.loc[common]
        y_al = labels.loc[common]

        logger.info(
            "Period %s: %d samples x %d features",
            plabel, len(X_al), X_al.shape[1],
        )

        if model_type == "ensemble":
            result = train_ensemble(
                X_al, y_al, random_state,
                problem_type=problem_type,
                tune=tune,
                calibrate=calibrate,
                calibration_method=calibration_method,
                feature_selection=feature_selection,
                rfecv_min_features=rfecv_min_features,
            )
        else:
            result = train_classifier(
                X_al, y_al, random_state,
                model_type=model_type,
                problem_type=problem_type,
                tune=tune,
                calibrate=calibrate,
                calibration_method=calibration_method,
                feature_selection=feature_selection,
                rfecv_min_features=rfecv_min_features,
            )

        if refit_full:
            result = refit_on_full_data(X_al, y_al, result, random_state)

        sub_results.append(result)
        trained_labels.append(plabel)

    if not sub_results:
        raise ValueError("No sub-models were successfully trained")

    n = len(sub_results)
    if recency_decay < 1.0 and n > 1:
        raw_weights = np.array(
            [recency_decay ** (n - 1 - i) for i in range(n)],
        )
    else:
        raw_weights = np.ones(n)
    weights = raw_weights / raw_weights.sum()

    class_names = sub_results[0].class_names

    logger.info(
        "Multi-quarter ensemble: %d sub-models on %s, weights=[%s]",
        n, trained_labels,
        ", ".join(f"{w:.3f}" for w in weights),
    )

    return MultiQuarterEnsembleResult(
        sub_results=sub_results,
        period_labels=trained_labels,
        weights=weights,
        consensus_method=consensus_method,
        class_names=class_names,
    )


def _train_fallback_with_regime_feature(
    ohlcv_by_ticker: dict[str, pd.DataFrame],
    fundamentals_by_ticker: dict[str, dict[str, Any]] | None,
    periods: list[tuple[str, str, str, str]],
    period_regime_map: dict[str, str],
    random_state: int,
    *,
    model_type: str = "rf",
    problem_type: ProblemType = "multiclass",
    drop_correlated: bool = True,
    tune: bool = True,
    calibrate: bool = False,
    calibration_method: Literal["sigmoid", "isotonic"] = "sigmoid",
    feature_selection: bool = False,
    rfecv_min_features: int = 5,
) -> TrainResult:
    """Train a single classifier on stacked data from ALL quarters with regime dummies.

    Used as fallback for regimes with too few quarters for a dedicated model.
    One-hot columns ``regime_bull``, ``regime_bear``, ``regime_sideways`` are
    appended so the model can condition on market context.
    """
    try:
        from src.classifier import assign_groups, compute_q1_returns
        from src.features import (
            build_feature_matrix,
            drop_correlated_features as _drop_corr,
        )
    except ImportError:
        from classifier import assign_groups, compute_q1_returns
        from features import (
            build_feature_matrix,
            drop_correlated_features as _drop_corr,
        )

    _REGIME_DUMMY_COLS = ["regime_bull", "regime_bear", "regime_sideways"]

    X_parts: list[pd.DataFrame] = []
    y_parts: list[pd.Series] = []
    pl_parts: list[np.ndarray] = []

    for fc, q_start, q_end, plabel in periods:
        regime_label = period_regime_map.get(plabel)
        if regime_label is None:
            continue

        returns = compute_q1_returns(ohlcv_by_ticker, q_start=q_start, q_end=q_end)
        labels = assign_groups(returns)
        X = build_feature_matrix(ohlcv_by_ticker, fc, fundamentals_by_ticker)

        common = X.index.intersection(labels.dropna().index)
        if len(common) < 10:
            logger.warning(
                "Fallback: skipping period %s (%d valid samples)", plabel, len(common),
            )
            continue

        X_al = X.loc[common].copy()
        y_al = labels.loc[common].copy()

        for rv in ("bull", "bear", "sideways"):
            X_al[f"regime_{rv}"] = float(regime_label == rv)

        X_al.index = pd.Index(
            [f"{t}__{plabel}" for t in X_al.index], name="obs_id",
        )
        y_al.index = X_al.index

        X_parts.append(X_al)
        y_parts.append(y_al)
        pl_parts.append(np.full(len(X_al), plabel))

        logger.info(
            "Fallback: period %s [%s] — %d samples", plabel, regime_label, len(X_al),
        )

    if not X_parts:
        raise ValueError("No valid periods for fallback model")

    X_stacked = pd.concat(X_parts)
    y_stacked = pd.concat(y_parts)
    pl_stacked = np.concatenate(pl_parts)

    if drop_correlated:
        non_regime = X_stacked.drop(columns=_REGIME_DUMMY_COLS, errors="ignore")
        non_regime, _ = _drop_corr(non_regime)
        regime_cols = X_stacked[[c for c in _REGIME_DUMMY_COLS if c in X_stacked.columns]]
        X_stacked = pd.concat([non_regime, regime_cols], axis=1)

    logger.info(
        "Fallback model: %d total samples × %d features (incl. regime dummies)",
        len(X_stacked), X_stacked.shape[1],
    )

    if model_type == "ensemble":
        result = train_ensemble(
            X_stacked, y_stacked, random_state,
            problem_type=problem_type,
            tune=tune,
            calibrate=calibrate,
            calibration_method=calibration_method,
            period_labels=pl_stacked,
            walk_forward=True,
            min_train_periods=2,
            feature_selection=feature_selection,
            rfecv_min_features=rfecv_min_features,
        )
    else:
        result = train_classifier(
            X_stacked, y_stacked, random_state,
            model_type=model_type,
            problem_type=problem_type,
            tune=tune,
            calibrate=calibrate,
            calibration_method=calibration_method,
            period_labels=pl_stacked,
            walk_forward=True,
            min_train_periods=2,
            feature_selection=feature_selection,
            rfecv_min_features=rfecv_min_features,
        )

    return result


def train_regime_aware_models(
    ohlcv_by_ticker: dict[str, pd.DataFrame],
    fundamentals_by_ticker: dict[str, dict[str, Any]] | None,
    periods: list[tuple[str, str, str, str]],
    random_state: int,
    *,
    model_type: str = "rf",
    problem_type: ProblemType = "multiclass",
    min_regime_quarters: int = 4,
    recency_decay: float = 0.85,
    consensus_method: str = "proba_average",
    drop_correlated: bool = True,
    tune: bool = True,
    calibrate: bool = False,
    calibration_method: Literal["sigmoid", "isotonic"] = "sigmoid",
    refit_full: bool = True,
    feature_selection: bool = False,
    rfecv_min_features: int = 5,
) -> RegimeModelCollection:
    """Train regime-specific classifiers with separate models per market regime.

    Groups *periods* by detected regime (bull / bear / sideways) using the
    ``^SSMI`` regime detector.  Regimes with >= *min_regime_quarters*
    quarters get their own :class:`MultiQuarterEnsembleResult` (via
    :func:`train_multi_quarter_ensemble`).  Underrepresented regimes share a
    single fallback :class:`TrainResult` trained on ALL quarters with one-hot
    regime indicators as additional features.

    Parameters
    ----------
    ohlcv_by_ticker
        Ticker -> OHLCV DataFrame.  **Must** include ``^SSMI`` for regime
        detection.
    fundamentals_by_ticker
        Ticker -> yfinance ``.info`` dict.  Can be ``None``.
    periods
        ``(feature_cutoff, q_start, q_end, period_label)`` tuples from
        :data:`config.CLASSIFICATION_PERIODS`.
    random_state
        Reproducibility seed.
    model_type
        ``"rf"``, ``"xgb"``, ``"lgb"``, or ``"ensemble"``.
    problem_type
        ``"multiclass"`` or ``"binary"``.
    min_regime_quarters
        Minimum quarters for a regime to get its own dedicated model.
        Regimes below this threshold use the fallback.
    recency_decay
        Exponential decay for within-regime recency weighting.
    consensus_method
        ``"proba_average"`` or ``"majority"`` for per-regime ensembles.
    drop_correlated
        Drop highly correlated features before training.
    tune
        Use ``GridSearchCV`` for each sub-model.
    calibrate
        Calibrate probabilities for each sub-model.
    calibration_method
        ``"sigmoid"`` or ``"isotonic"``.
    refit_full
        Refit each model on all available data for that quarter.
    feature_selection
        Run RFECV per regime to eliminate weak features.
    rfecv_min_features
        Minimum features to retain during RFECV.

    Returns
    -------
    RegimeModelCollection
    """
    try:
        from src.regime import label_periods as _label_periods
        from src.features import MACRO_BENCHMARK_TICKER
    except ImportError:
        from regime import label_periods as _label_periods
        from features import MACRO_BENCHMARK_TICKER

    # --- 1. Regime detection ---
    smi_ohlcv = ohlcv_by_ticker.get(MACRO_BENCHMARK_TICKER)
    if smi_ohlcv is None or smi_ohlcv.empty:
        raise ValueError(
            f"ohlcv_by_ticker must contain {MACRO_BENCHMARK_TICKER} for regime detection"
        )

    regime_states = _label_periods(smi_ohlcv, periods)

    # --- 2. Group periods by regime ---
    regime_period_tuples: dict[str, list[tuple[str, str, str, str]]] = {}
    period_regime_map: dict[str, str] = {}

    for fc, q_start, q_end, plabel in periods:
        if plabel not in regime_states:
            logger.warning(
                "Period %s could not be regime-classified — skipping", plabel,
            )
            continue
        regime_label = regime_states[plabel].label.value
        period_regime_map[plabel] = regime_label
        regime_period_tuples.setdefault(regime_label, []).append(
            (fc, q_start, q_end, plabel),
        )

    logger.info("Regime grouping of %d periods:", len(period_regime_map))
    for regime, rp in regime_period_tuples.items():
        plabels = [p[3] for p in rp]
        confs = [
            f"{regime_states[pl].confidence:.2f}" for pl in plabels
        ]
        logger.info(
            "  %s: %d quarters — %s  (confidence: %s)",
            regime.upper(), len(plabels), plabels, confs,
        )

    # --- 3. Dedicated vs. fallback ---
    dedicated_regimes: dict[str, list[tuple[str, str, str, str]]] = {}
    fallback_regimes: list[str] = []

    for regime, regime_periods in regime_period_tuples.items():
        if len(regime_periods) >= min_regime_quarters:
            dedicated_regimes[regime] = regime_periods
        else:
            fallback_regimes.append(regime)
            logger.info(
                "Regime %s has %d quarters (< %d) → fallback model",
                regime.upper(), len(regime_periods), min_regime_quarters,
            )

    # --- 4. Train dedicated regime models ---
    regime_models: dict[str, MultiQuarterEnsembleResult] = {}

    for regime, regime_periods in dedicated_regimes.items():
        logger.info(
            "Training dedicated model for regime %s (%d quarters)",
            regime.upper(), len(regime_periods),
        )
        mqe = train_multi_quarter_ensemble(
            ohlcv_by_ticker,
            fundamentals_by_ticker,
            regime_periods,
            random_state,
            model_type=model_type,
            problem_type=problem_type,
            recency_decay=recency_decay,
            consensus_method=consensus_method,
            drop_correlated=drop_correlated,
            tune=tune,
            calibrate=calibrate,
            calibration_method=calibration_method,
            refit_full=refit_full,
            feature_selection=feature_selection,
            rfecv_min_features=rfecv_min_features,
        )
        regime_models[regime] = mqe

    # --- 5. Train fallback model (if any regime is underrepresented) ---
    fallback_model: TrainResult | None = None

    if fallback_regimes:
        logger.info(
            "Training fallback model on ALL %d quarters with regime feature "
            "(for regimes: %s)",
            len(periods), [r.upper() for r in fallback_regimes],
        )
        fallback_model = _train_fallback_with_regime_feature(
            ohlcv_by_ticker=ohlcv_by_ticker,
            fundamentals_by_ticker=fundamentals_by_ticker,
            periods=[
                p for p in periods if p[3] in period_regime_map
            ],
            period_regime_map=period_regime_map,
            random_state=random_state,
            model_type=model_type,
            problem_type=problem_type,
            drop_correlated=drop_correlated,
            tune=tune,
            calibrate=calibrate,
            calibration_method=calibration_method,
            feature_selection=feature_selection,
            rfecv_min_features=rfecv_min_features,
        )

    # --- Determine canonical class names ---
    class_names: list[str] = []
    if regime_models:
        class_names = next(iter(regime_models.values())).class_names
    elif fallback_model is not None:
        class_names = fallback_model.class_names

    regime_period_label_map = {
        regime: [p[3] for p in plist]
        for regime, plist in regime_period_tuples.items()
    }

    logger.info(
        "RegimeModelCollection ready: %d dedicated regime(s) %s, "
        "%d fallback regime(s) %s",
        len(regime_models), list(regime_models.keys()),
        len(fallback_regimes), fallback_regimes,
    )

    return RegimeModelCollection(
        regime_models=regime_models,
        fallback_model=fallback_model,
        fallback_regimes=fallback_regimes,
        regime_period_map=regime_period_label_map,
        period_regime_map=period_regime_map,
        class_names=class_names,
        min_regime_quarters=min_regime_quarters,
    )


def _predict_proba_mqe(
    ensemble: MultiQuarterEnsembleResult,
    X: pd.DataFrame,
) -> pd.DataFrame:
    """Weighted average of per-sub-model probabilities."""
    ens_classes = ensemble.class_names
    proba_sum = np.zeros((len(X), len(ens_classes)))

    for i, sub_result in enumerate(ensemble.sub_results):
        w = float(ensemble.weights[i])
        X_prep = _apply_imputation(X, sub_result.imputer_medians, sub_result.feature_names)
        proba = sub_result.model.predict_proba(X_prep.values)
        sub_classes = list(sub_result.class_names)

        for j, cn in enumerate(ens_classes):
            if cn in sub_classes:
                proba_sum[:, j] += proba[:, sub_classes.index(cn)] * w

    return pd.DataFrame(proba_sum, index=X.index, columns=ens_classes)


def _predict_mqe(
    ensemble: MultiQuarterEnsembleResult,
    X: pd.DataFrame,
) -> pd.Series:
    """Consensus predictions from a multi-quarter ensemble."""
    if ensemble.consensus_method == "majority":
        ens_classes = ensemble.class_names
        tallies = np.zeros((len(X), len(ens_classes)))

        for i, sub_result in enumerate(ensemble.sub_results):
            w = float(ensemble.weights[i])
            preds = predict(sub_result, X)
            for j, cn in enumerate(ens_classes):
                tallies[:, j] += (preds.values == cn).astype(float) * w

        pred_idx = np.argmax(tallies, axis=1)
        labels = [ens_classes[k] for k in pred_idx]
        return pd.Series(labels, index=X.index, name="predicted_group")

    proba = _predict_proba_mqe(ensemble, X)
    pred_idx = np.argmax(proba.values, axis=1)
    labels = [ensemble.class_names[k] for k in pred_idx]
    return pd.Series(labels, index=X.index, name="predicted_group")


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    """Accuracy, F1 (macro/weighted), confusion matrix, classification report."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0),
        ),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0),
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "report_dict": classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        ),
        "report_str": classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            zero_division=0,
        ),
    }


def predict(model: Any, X: pd.DataFrame) -> pd.Series:
    """Class predictions for new data.

    Parameters
    ----------
    model
        A :class:`TrainResult`, :class:`MultiQuarterEnsembleResult`, or a
        raw fitted estimator.
    X
        Feature DataFrame with the same columns used during training.
    """
    if isinstance(model, MultiQuarterEnsembleResult):
        return _predict_mqe(model, X)

    if isinstance(model, TrainResult):
        result = model
        X_prep = _apply_imputation(X, result.imputer_medians, result.feature_names)
        y_enc = result.model.predict(X_prep.values)
        y_labels = result.label_encoder.inverse_transform(y_enc)
        return pd.Series(y_labels, index=X.index, name="predicted_group")

    y_enc = model.predict(X.values)
    return pd.Series(y_enc, index=X.index, name="predicted_group")


def predict_proba(
    result: TrainResult | MultiQuarterEnsembleResult,
    X: pd.DataFrame,
) -> pd.DataFrame:
    """Per-class probabilities for new data.

    Accepts both :class:`TrainResult` (single model) and
    :class:`MultiQuarterEnsembleResult` (weighted average across sub-models).

    Parameters
    ----------
    result
        Training result whose ``model`` exposes ``predict_proba``, or a
        multi-quarter ensemble result.
    X
        Feature DataFrame aligned with training columns.
    """
    if isinstance(result, MultiQuarterEnsembleResult):
        return _predict_proba_mqe(result, X)

    X_prep = _apply_imputation(X, result.imputer_medians, result.feature_names)
    proba = result.model.predict_proba(X_prep.values)
    return pd.DataFrame(
        proba,
        index=X_prep.index,
        columns=result.class_names,
    )


_REGIME_CONFIDENCE_THRESHOLD: float = 0.7


def _compute_regime_blend_weights(
    regime_state: Any,
) -> dict[str, float]:
    """Blend weights for each regime based on confidence and SMA alignment.

    Clear regimes (confidence >= 0.7) put 100 % weight on the detected
    regime.  Transition zones distribute the residual weight:

    * **Bull / Bear** low confidence → blend towards Sideways (the
      adjacent neutral regime in the SMA-alignment scheme).
    * **Sideways** low confidence → blend towards Bull or Bear based on
      the sign of ``sma_cross_gap`` (positive = SMA(50) above SMA(200),
      leaning Bull; negative = leaning Bear).
    """
    label_str: str = regime_state.label.value
    confidence: float = regime_state.confidence
    indicators = regime_state.indicators

    weights: dict[str, float] = {"bull": 0.0, "bear": 0.0, "sideways": 0.0}

    if confidence >= _REGIME_CONFIDENCE_THRESHOLD:
        weights[label_str] = 1.0
        return weights

    weights[label_str] = confidence
    remaining = 1.0 - confidence

    if label_str == "sideways":
        if indicators.sma_cross_gap >= 0:
            weights["bull"] += remaining
        else:
            weights["bear"] += remaining
    else:
        weights["sideways"] += remaining

    return weights


def _predict_proba_for_regime(
    regime_collection: RegimeModelCollection,
    X: pd.DataFrame,
    regime_label: str,
) -> pd.DataFrame | None:
    """Probability matrix from a single regime's model.

    For dedicated regimes the per-quarter ensemble is used directly.  For
    fallback regimes the shared model is called after injecting one-hot
    regime columns (``regime_bull``, ``regime_bear``, ``regime_sideways``).
    Returns ``None`` when no model is available.
    """
    if regime_label in regime_collection.regime_models:
        mqe = regime_collection.regime_models[regime_label]
        return _predict_proba_mqe(mqe, X)

    if (
        regime_label in regime_collection.fallback_regimes
        and regime_collection.fallback_model is not None
    ):
        X_fb = X.copy()
        for rv in ("bull", "bear", "sideways"):
            X_fb[f"regime_{rv}"] = float(regime_label == rv)
        return predict_proba(regime_collection.fallback_model, X_fb)

    return None


def predict_proba_regime_aware(
    regime_models: RegimeModelCollection,
    X: pd.DataFrame,
    smi_ohlcv: pd.DataFrame,
    date: str,
) -> pd.DataFrame:
    """Regime-aware class probabilities with confidence-based blending.

    Detects the current market regime on *date*, selects the matching
    regime-specific model (or fallback), and blends probabilities when
    the regime confidence falls below the transition threshold (0.7).

    Parameters
    ----------
    regime_models
        Trained :class:`RegimeModelCollection` from
        :func:`train_regime_aware_models`.
    X
        Feature matrix for prediction (ticker index, feature columns).
    smi_ohlcv
        ``^SSMI`` OHLCV DataFrame for regime detection (data up to *date*).
    date
        ISO date string for regime detection (no lookahead).

    Returns
    -------
    pd.DataFrame
        Per-class probabilities (ticker index, class-name columns).
    """
    try:
        from src.regime import detect_regime as _detect
    except ImportError:
        from regime import detect_regime as _detect

    regime_state = _detect(smi_ohlcv, date)
    blend_weights = _compute_regime_blend_weights(regime_state)

    class_names = regime_models.class_names
    blended = np.zeros((len(X), len(class_names)))
    applied_weight = 0.0

    for regime_label, weight in blend_weights.items():
        if weight <= 0:
            continue

        proba = _predict_proba_for_regime(regime_models, X, regime_label)
        if proba is None:
            logger.warning(
                "No model for regime %s (weight=%.2f) — weight redistributed",
                regime_label.upper(), weight,
            )
            continue

        for j, cn in enumerate(class_names):
            if cn in proba.columns:
                blended[:, j] += proba[cn].values * weight
        applied_weight += weight

    if applied_weight > 0 and abs(applied_weight - 1.0) > 1e-9:
        blended /= applied_weight

    logger.info(
        "Regime-aware predict_proba: regime=%s conf=%.2f, blend=%s",
        regime_state.label.value.upper(),
        regime_state.confidence,
        {k: f"{v:.2f}" for k, v in blend_weights.items() if v > 0},
    )

    return pd.DataFrame(blended, index=X.index, columns=class_names)


def predict_regime_aware(
    regime_models: RegimeModelCollection,
    X: pd.DataFrame,
    smi_ohlcv: pd.DataFrame,
    date: str,
) -> pd.Series:
    """Regime-aware class predictions with confidence-based blending.

    Calls :func:`predict_proba_regime_aware` and returns the argmax class
    per ticker.

    Parameters
    ----------
    regime_models
        Trained :class:`RegimeModelCollection`.
    X
        Feature matrix for prediction.
    smi_ohlcv
        ``^SSMI`` OHLCV for regime detection.
    date
        ISO date string for regime detection.

    Returns
    -------
    pd.Series
        Predicted class labels (index = ticker).
    """
    proba = predict_proba_regime_aware(regime_models, X, smi_ohlcv, date)
    class_names = list(proba.columns)
    pred_idx = np.argmax(proba.values, axis=1)
    labels = [class_names[k] for k in pred_idx]
    return pd.Series(labels, index=X.index, name="predicted_group")


def predict_with_threshold(
    result: TrainResult,
    X: pd.DataFrame,
    *,
    winner_class: str = "Winners",
    winner_threshold: float = 0.5,
    class_thresholds: dict[str, float] | None = None,
) -> pd.Series:
    """Assign labels using ``predict_proba`` with optional per-class floors.

    By default, a ticker is predicted as *winner_class* only if
    ``P(winner_class) >= winner_threshold``; otherwise the label is the
    argmax among the remaining classes (winner column excluded). This
    improves precision on Winners at the expense of recall.

    If *class_thresholds* is set, it overrides the winner-only rule: each
    class ``c`` is eligible only if ``P(c) >= class_thresholds[c]`` (classes
    omitted from the dict are unconstrained). Among eligible classes, the
    highest probability wins; if none are eligible, falls back to global argmax.

    Parameters
    ----------
    result
        :class:`TrainResult` from :func:`train_classifier` (optionally with
        ``calibrate=True`` for better-calibrated probabilities).
    X
        Feature matrix for prediction.
    winner_class
        Name of the positive / long class (default ``Winners``).
    winner_threshold
        Minimum ``P(winner_class)`` to predict that class when
        *class_thresholds* is ``None``.
    class_thresholds
        Optional mapping ``class_name -> minimum probability`` for eligibility.
    """
    X_prep = _apply_imputation(X, result.imputer_medians, result.feature_names)
    proba = result.model.predict_proba(X_prep.values)
    class_names = list(result.class_names)
    if winner_class not in class_names:
        raise ValueError(
            f"winner_class {winner_class!r} not in model classes {class_names}",
        )

    n = proba.shape[0]
    n_classes = len(class_names)
    pred_idx = np.empty(n, dtype=int)

    if class_thresholds is None:
        wi = class_names.index(winner_class)
        for i in range(n):
            if proba[i, wi] >= winner_threshold:
                pred_idx[i] = wi
            else:
                p_row = proba[i].copy()
                p_row[wi] = -1.0
                pred_idx[i] = int(np.argmax(p_row))
    else:
        for i in range(n):
            eligible = np.zeros(n_classes, dtype=bool)
            for j, cname in enumerate(class_names):
                min_p = class_thresholds.get(cname)
                if min_p is None:
                    eligible[j] = True
                elif proba[i, j] >= min_p:
                    eligible[j] = True
            if eligible.any():
                masked = np.where(eligible, proba[i], -1.0)
                pred_idx[i] = int(np.argmax(masked))
            else:
                pred_idx[i] = int(np.argmax(proba[i]))

    labels = result.label_encoder.inverse_transform(pred_idx)
    return pd.Series(labels, index=X_prep.index, name="predicted_group")


def refit_on_full_data(
    X: pd.DataFrame,
    y: pd.Series,
    result: TrainResult,
    random_state: int | None = None,
) -> TrainResult:
    """Retrain the best model on *all* labelled data (no hold-out).

    Uses the same hyperparameters discovered during :func:`train_classifier`.
    The returned :class:`TrainResult` is ready for forward-test predictions.
    """
    rs = RANDOM_SEED if random_state is None else random_state

    common = X.index.intersection(y.dropna().index)
    X_clean = X.loc[common].copy()
    y_clean = y.loc[common].copy()

    if getattr(result, "classification_mode", "multiclass") == "binary":
        y_clean = _to_binary_labels(y_clean)

    X_imp, medians = _impute_median(X_clean)
    shared_cols = [c for c in result.feature_names if c in X_imp.columns]
    X_imp = X_imp[shared_cols]

    le = result.label_encoder
    y_enc = le.transform(y_clean.values)

    model = clone(result.model)
    if hasattr(model, "random_state"):
        model.set_params(random_state=rs)
    model.fit(X_imp.values, y_enc)

    logger.info(
        "Refit on full data: %d samples × %d features",
        len(X_imp),
        len(shared_cols),
    )

    return TrainResult(
        model=model,
        label_encoder=le,
        imputer_medians=medians,
        feature_names=shared_cols,
        class_names=list(le.classes_),
        best_params=result.best_params,
        cv_results=result.cv_results,
        holdout_metrics=result.holdout_metrics,
        X_train=X_imp,
        X_test=pd.DataFrame(),
        y_train_enc=y_enc,
        y_test_enc=np.array([]),
        model_type=result.model_type,
        calibrated=getattr(result, "calibrated", False),
        classification_mode=getattr(result, "classification_mode", "multiclass"),
    )


def shap_explain(result: TrainResult) -> dict[str, Any]:
    """SHAP TreeExplainer analysis on the trained model.

    Returns
    -------
    dict
        ``shap_values`` (3-D array), ``X_display`` (DataFrame),
        ``feature_names``, ``class_names``, ``mean_abs_shap`` (DataFrame),
        ``explainer``.
    """
    import shap

    if not result.X_test.empty:
        X_all = pd.concat([result.X_train, result.X_test])
    else:
        X_all = result.X_train

    tree_model = getattr(result.model, "rf_", result.model)
    if hasattr(tree_model, "calibrated_classifiers_"):
        tree_model = tree_model.calibrated_classifiers_[0].estimator
    explainer = shap.TreeExplainer(tree_model)
    sv = explainer.shap_values(X_all.values)

    if isinstance(sv, list):
        sv_arr = np.stack(sv, axis=-1)
    else:
        sv_arr = np.asarray(sv)

    mean_abs: dict[str, np.ndarray] = {}
    for ci, cn in enumerate(result.class_names):
        if sv_arr.ndim == 3:
            mean_abs[cn] = np.abs(sv_arr[:, :, ci]).mean(axis=0)
        else:
            mean_abs[cn] = np.abs(sv_arr).mean(axis=0)

    mean_abs_df = pd.DataFrame(mean_abs, index=result.feature_names)
    mean_abs_df.index.name = "feature"

    return {
        "shap_values": sv_arr,
        "X_display": X_all,
        "feature_names": result.feature_names,
        "class_names": result.class_names,
        "mean_abs_shap": mean_abs_df,
        "explainer": explainer,
    }


def shap_explain_per_regime(
    regime_collection: RegimeModelCollection,
) -> dict[str, dict[str, Any]]:
    """SHAP analysis for each dedicated regime model (Bull/Bear/Sideways).

    For each regime that has a dedicated :class:`MultiQuarterEnsembleResult`,
    the first sub-model is used for SHAP explanation (TreeExplainer requires
    a single estimator, not an ensemble-of-ensembles).

    Returns
    -------
    dict
        ``regime_label → shap_explain()``-style result dict.  Regimes using
        the fallback model are included if a fallback exists.
    """
    results: dict[str, dict[str, Any]] = {}

    for regime_label, mqe in regime_collection.regime_models.items():
        if not mqe.sub_results:
            logger.warning(
                "SHAP per regime: %s has no sub-results — skipping",
                regime_label.upper(),
            )
            continue

        sub = mqe.sub_results[-1]
        logger.info(
            "SHAP per regime: explaining %s (%d features, period %s)",
            regime_label.upper(),
            len(sub.feature_names),
            mqe.period_labels[-1] if mqe.period_labels else "?",
        )
        try:
            results[regime_label] = shap_explain(sub)
        except Exception as exc:
            logger.warning(
                "SHAP per regime: %s failed — %s", regime_label.upper(), exc,
            )

    if (
        regime_collection.fallback_model is not None
        and regime_collection.fallback_regimes
    ):
        fb_label = "fallback_" + "+".join(regime_collection.fallback_regimes)
        logger.info("SHAP per regime: explaining fallback model")
        try:
            results[fb_label] = shap_explain(regime_collection.fallback_model)
        except Exception as exc:
            logger.warning("SHAP fallback failed — %s", exc)

    return results


def shap_regime_summary(
    regime_shap: dict[str, dict[str, Any]],
    top_n: int = 15,
) -> pd.DataFrame:
    """Aggregate mean|SHAP| across regimes into one ranked table.

    Returns a DataFrame with columns ``feature``, ``regime``,
    ``mean_abs_shap_Winners``, sorted descending.
    """
    rows: list[dict[str, Any]] = []
    for regime, shap_result in regime_shap.items():
        mas = shap_result.get("mean_abs_shap")
        if mas is None or mas.empty:
            continue
        for feature in mas.index:
            row: dict[str, Any] = {"feature": feature, "regime": regime}
            for cn in mas.columns:
                row[f"mean_abs_shap_{cn}"] = float(mas.loc[feature, cn])
            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    sort_col = (
        "mean_abs_shap_Winners"
        if "mean_abs_shap_Winners" in df.columns
        else df.columns[-1]
    )
    return df.sort_values(sort_col, ascending=False).head(top_n * len(regime_shap))


def permutation_importance_walk_forward(
    ohlcv: dict[str, pd.DataFrame],
    fundamentals: dict[str, dict],
    oos_years: list[int],
    *,
    periods_filter_fn: Any = None,
    random_seed: int = 42,
    n_repeats: int = 5,
) -> pd.DataFrame:
    """Permutation importance on walk-forward holdout sets.

    For each OOS year, trains a simple RF on the pre-OOS periods, then
    computes sklearn :func:`~sklearn.inspection.permutation_importance`
    on the OOS data.  Results are averaged across years.

    Parameters
    ----------
    ohlcv
        Ticker → OHLCV DataFrame.
    fundamentals
        Ticker → yfinance ``.info`` dict.
    oos_years
        Calendar years to evaluate (e.g. ``[2020, 2021, 2022, 2023, 2024]``).
    periods_filter_fn
        ``(oos_year) → list[tuple]`` — function returning training periods
        for a given OOS year.  Defaults to all periods with q_end < OOS start.
    random_seed
        Reproducibility seed.
    n_repeats
        Number of permutation repeats per feature.

    Returns
    -------
    pd.DataFrame
        Index = feature name, columns = ``importance_mean``,
        ``importance_std``, ``n_years``.
    """
    from sklearn.inspection import permutation_importance as sklearn_pi

    try:
        from src.classifier import assign_groups, compute_q1_returns
        from src.features import build_feature_matrix, drop_correlated_features
        from config import CLASSIFICATION_PERIODS
    except ImportError:
        from classifier import assign_groups, compute_q1_returns
        from features import build_feature_matrix, drop_correlated_features
        from config import CLASSIFICATION_PERIODS

    def _default_periods_before(yr: int) -> list[tuple[str, str, str, str]]:
        cutoff = pd.Timestamp(f"{yr}-01-01")
        return [
            p for p in CLASSIFICATION_PERIODS
            if pd.Timestamp(p[2]) < cutoff
        ]

    filter_fn = periods_filter_fn or _default_periods_before

    all_importances: dict[str, list[float]] = {}
    n_evaluated = 0

    for yr in oos_years:
        train_periods = filter_fn(yr)
        if len(train_periods) < 4:
            logger.warning(
                "Permutation importance: skipping OOS %d (only %d training quarters)",
                yr, len(train_periods),
            )
            continue

        X_parts, y_parts = [], []
        for fc, q_start, q_end, plabel in train_periods:
            X = build_feature_matrix(ohlcv, fc, fundamentals)
            X, _ = drop_correlated_features(X)
            rets = compute_q1_returns(ohlcv, q_start=q_start, q_end=q_end)
            labels = assign_groups(rets)
            common = X.index.intersection(labels.dropna().index)
            if len(common) < 10:
                continue
            xp = X.loc[common].copy()
            yp = labels.loc[common].copy()
            uid = pd.Index([f"{t}__{plabel}" for t in xp.index])
            xp.index = uid
            yp.index = uid
            X_parts.append(xp)
            y_parts.append(yp)

        if not X_parts:
            continue

        X_train = pd.concat(X_parts)
        y_train = pd.concat(y_parts)

        oos_cutoff = f"{yr - 1}-12-31"
        try:
            from src.backtest import build_oos_features, compute_oos_returns
        except ImportError:
            from backtest import build_oos_features, compute_oos_returns

        X_oos = build_oos_features(ohlcv, fundamentals, cutoff_date=oos_cutoff)
        rets_oos = compute_oos_returns(ohlcv, year=yr)
        labels_oos = assign_groups(rets_oos)
        common_oos = X_oos.index.intersection(labels_oos.dropna().index)
        if len(common_oos) < 10:
            continue

        X_oos = X_oos.loc[common_oos]
        y_oos = labels_oos.loc[common_oos]

        shared_cols = [c for c in X_train.columns if c in X_oos.columns]
        X_train = X_train[shared_cols]
        X_oos = X_oos[shared_cols]

        medians = X_train.median()
        X_train = X_train.fillna(medians)
        X_oos = X_oos.fillna(medians)

        le = LabelEncoder()
        y_tr_enc = le.fit_transform(y_train.values)
        y_oos_enc = le.transform(y_oos.values)

        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=random_seed,
            n_jobs=-1,
            class_weight="balanced",
        )
        clf.fit(X_train.values, y_tr_enc)

        pi = sklearn_pi(
            clf,
            X_oos.values,
            y_oos_enc,
            n_repeats=n_repeats,
            random_state=random_seed,
            scoring="f1_macro",
            n_jobs=-1,
        )

        for j, feat in enumerate(shared_cols):
            all_importances.setdefault(feat, []).append(
                float(pi.importances_mean[j]),
            )
        n_evaluated += 1

        logger.info(
            "Permutation importance OOS %d: top-3 = %s",
            yr,
            sorted(
                zip(shared_cols, pi.importances_mean),
                key=lambda x: -x[1],
            )[:3],
        )

    if not all_importances:
        return pd.DataFrame(columns=["importance_mean", "importance_std", "n_years"])

    rows = []
    for feat, vals in all_importances.items():
        rows.append({
            "feature": feat,
            "importance_mean": float(np.mean(vals)),
            "importance_std": float(np.std(vals)),
            "n_years": len(vals),
        })

    df = pd.DataFrame(rows).set_index("feature").sort_values(
        "importance_mean", ascending=False,
    )
    return df


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def plot_confusion_matrix(
    metrics: dict[str, Any],
    class_names: list[str],
    *,
    title: str = "Confusion Matrix",
    normalize: bool = False,
    ax: Any | None = None,
) -> Any:
    """Heatmap of the confusion matrix from :func:`evaluate_predictions`."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = metrics["confusion_matrix"]
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.where(row_sums > 0, cm.astype(float) / row_sums, 0.0)

    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)

    if created:
        plt.tight_layout()
    return ax


def plot_cv_folds(
    cv_results: dict[str, Any],
    *,
    title: str = "Cross-validation fold scores",
    ax: Any | None = None,
) -> Any:
    """Grouped bar chart of per-fold CV scores."""
    import matplotlib.pyplot as plt

    fold_scores = cv_results["fold_scores"]
    n_folds = len(fold_scores["accuracy"])
    x = np.arange(n_folds)
    width = 0.25

    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(10, 4))

    ax.bar(x - width, fold_scores["accuracy"], width, label="Accuracy", color="#3498db")
    ax.bar(x, fold_scores["f1_macro"], width, label="F1 macro", color="#e74c3c")
    ax.bar(x + width, fold_scores["f1_weighted"], width, label="F1 weighted", color="#2ecc71")

    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {i + 1}" for i in range(n_folds)])
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.set_ylabel("Score")
    ax.legend()

    if created:
        plt.tight_layout()
    return ax


def plot_model_comparison(
    results: dict[str, TrainResult],
    *,
    title: str = "Model comparison (hold-out)",
    ax: Any | None = None,
) -> Any:
    """Side-by-side bar chart comparing multiple trained models on hold-out metrics."""
    import matplotlib.pyplot as plt

    metric_keys = ["accuracy", "f1_macro", "f1_weighted"]
    model_names = list(results.keys())
    n_models = len(model_names)
    n_metrics = len(metric_keys)

    x = np.arange(n_metrics)
    width = 0.8 / max(n_models, 1)
    colours = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"]

    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(8, 4))

    for i, name in enumerate(model_names):
        holdout = results[name].holdout_metrics
        vals = [holdout[m] for m in metric_keys]
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(
            x + offset,
            vals,
            width,
            label=name,
            color=colours[i % len(colours)],
        )
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ").title() for m in metric_keys])
    ax.set_ylim(0, 1.15)
    ax.set_title(title)
    ax.set_ylabel("Score")
    ax.legend()

    if created:
        plt.tight_layout()
    return ax


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_model(
    result: TrainResult | MultiQuarterEnsembleResult,
    path: str | Path,
) -> Path:
    """Serialise a :class:`TrainResult` or :class:`MultiQuarterEnsembleResult` to disk via joblib."""
    import joblib

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result, p)
    logger.info("Model saved to %s", p)
    return p


def load_model(
    path: str | Path,
) -> TrainResult | MultiQuarterEnsembleResult:
    """Load a :class:`TrainResult` or :class:`MultiQuarterEnsembleResult` from disk."""
    import joblib

    result = joblib.load(Path(path))
    if not isinstance(result, (TrainResult, MultiQuarterEnsembleResult)):
        raise TypeError(
            f"Expected TrainResult or MultiQuarterEnsembleResult, "
            f"got {type(result).__name__}",
        )
    return result
