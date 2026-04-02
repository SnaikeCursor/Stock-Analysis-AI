"""Statistical feature analysis: ANOVA, post-hoc, VIF, plots.

Phase 4 of the Swiss SPI Extra pipeline — discriminant testing,
multicollinearity checks, ML-based feature importance, and
group-specific feature selection.
"""

from __future__ import annotations

import logging
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import f_oneway, kruskal, rankdata
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools import add_constant

try:
    from config import RANDOM_SEED
except ImportError:
    RANDOM_SEED = 42

logger = logging.getLogger(__name__)

GROUP_ORDER: tuple[str, ...] = ("Winners", "Steady", "Losers")

__all__ = [
    "anova_by_group",
    "compute_vif",
    "correlation_matrix",
    "effect_sizes",
    "full_analysis_report",
    "plot_feature_boxplots",
    "plot_feature_violins",
    "plot_importance_bar",
    "plot_radar_chart",
    "posthoc_pairwise",
    "recursive_feature_elimination",
    "rf_feature_importance",
    "select_discriminative_features",
    "shap_feature_importance",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _prepare(
    features: pd.DataFrame,
    labels: pd.Series,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Align indices, drop NaN labels, determine group ordering."""
    common = features.index.intersection(labels.dropna().index)
    X = features.loc[common].copy()
    y = labels.loc[common].copy()
    groups = [g for g in GROUP_ORDER if g in y.unique()]
    if not groups:
        groups = sorted(y.unique())
    return X, y, groups


def _impute_median(X: pd.DataFrame) -> pd.DataFrame:
    """Fill NaN with per-column median; drop all-NaN columns."""
    out = X.copy()
    all_nan = out.columns[out.isna().all()]
    if len(all_nan):
        logger.warning("Dropping %d all-NaN columns: %s", len(all_nan), list(all_nan))
        out = out.drop(columns=all_nan)
    return out.fillna(out.median())


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d with pooled standard deviation."""
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    na, nb = len(a), len(b)
    var_a, var_b = float(a.var(ddof=1)), float(b.var(ddof=1))
    pooled_std = np.sqrt(((na - 1) * var_a + (nb - 1) * var_b) / (na + nb - 2))
    if pooled_std == 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled_std)


def _dunns_test(
    group_data: list[np.ndarray],
    group_names: list[str],
) -> pd.DataFrame:
    """Dunn's post-hoc test after Kruskal-Wallis (Bonferroni-corrected).

    Parameters
    ----------
    group_data
        List of 1-D arrays, one per group (NaN-free).
    group_names
        Corresponding group labels.

    Returns
    -------
    pd.DataFrame
        Columns: group_a, group_b, z_stat, p_raw, p_corrected, significant.
    """
    k = len(group_data)
    empty = pd.DataFrame(
        columns=["group_a", "group_b", "z_stat", "p_raw", "p_corrected", "significant"],
    )
    if k < 2:
        return empty

    all_data = np.concatenate(group_data)
    N = len(all_data)
    if N < 3:
        return empty

    ranks = rankdata(all_data, method="average")

    group_sizes = [len(g) for g in group_data]
    group_ranks: list[np.ndarray] = []
    idx = 0
    for n_i in group_sizes:
        group_ranks.append(ranks[idx : idx + n_i])
        idx += n_i

    mean_ranks = [float(r.mean()) for r in group_ranks]

    # Tie correction: C = 1 - Σ(t³ - t) / (N³ - N)
    _, tie_counts = np.unique(ranks, return_counts=True)
    denom = N**3 - N
    C = 1.0 - float(np.sum(tie_counts**3 - tie_counts)) / denom if denom > 0 else 1.0

    sigma_sq = (N * (N + 1) / 12.0) * C
    n_comp = k * (k - 1) // 2

    rows: list[dict[str, Any]] = []
    for i, j in combinations(range(k), 2):
        se = np.sqrt(sigma_sq * (1.0 / group_sizes[i] + 1.0 / group_sizes[j]))
        if se > 0:
            z = (mean_ranks[i] - mean_ranks[j]) / se
            p_raw = 2.0 * stats.norm.sf(abs(z))
        else:
            z = 0.0
            p_raw = 1.0
        p_corr = min(p_raw * n_comp, 1.0)
        rows.append({
            "group_a": group_names[i],
            "group_b": group_names[j],
            "z_stat": float(z),
            "p_raw": float(p_raw),
            "p_corrected": float(p_corr),
            "significant": p_corr < 0.05,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. Statistical omnibus tests
# ---------------------------------------------------------------------------

def anova_by_group(features: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """ANOVA and Kruskal-Wallis omnibus tests per feature.

    Returns a DataFrame (index = ``feature``) sorted by ascending
    ``kruskal_p`` (most discriminative features first) with columns:

    * ``anova_f``, ``anova_p`` — parametric F-test
    * ``kruskal_h``, ``kruskal_p`` — non-parametric rank test
    * ``kruskal_p_fdr`` — Benjamini-Hochberg adjusted p across features
    * ``eta_squared`` — effect size (SS_between / SS_total)
    * ``significant`` — ``kruskal_p_fdr < 0.05``
    * ``mean_<Group>``, ``median_<Group>`` — per-group descriptives
    """
    X, y, groups = _prepare(features, labels)

    records: list[dict[str, Any]] = []
    for feat in X.columns:
        vals = X[feat]
        group_vals = [vals[y == g].dropna().values for g in groups]
        valid = [gv for gv in group_vals if len(gv) >= 2]

        row: dict[str, Any] = {"feature": feat}

        if len(valid) >= 2:
            try:
                f_stat, f_p = f_oneway(*valid)
                row["anova_f"] = float(f_stat)
                row["anova_p"] = float(f_p)
            except Exception:
                row["anova_f"] = np.nan
                row["anova_p"] = np.nan

            try:
                h_stat, h_p = kruskal(*valid)
                row["kruskal_h"] = float(h_stat)
                row["kruskal_p"] = float(h_p)
            except Exception:
                row["kruskal_h"] = np.nan
                row["kruskal_p"] = np.nan

            pooled = np.concatenate(valid)
            grand_mean = pooled.mean()
            ss_b = sum(len(gv) * (gv.mean() - grand_mean) ** 2 for gv in valid)
            ss_t = float(np.sum((pooled - grand_mean) ** 2))
            row["eta_squared"] = float(ss_b / ss_t) if ss_t > 0 else np.nan
        else:
            row.update({
                "anova_f": np.nan, "anova_p": np.nan,
                "kruskal_h": np.nan, "kruskal_p": np.nan,
                "eta_squared": np.nan,
            })

        for g in groups:
            gv = vals[y == g].dropna()
            row[f"mean_{g}"] = float(gv.mean()) if len(gv) else np.nan
            row[f"median_{g}"] = float(gv.median()) if len(gv) else np.nan

        records.append(row)

    df = pd.DataFrame(records).set_index("feature")

    # FDR correction across features
    mask = df["kruskal_p"].notna()
    df["kruskal_p_fdr"] = np.nan
    if mask.sum() > 0:
        _, p_adj, _, _ = multipletests(df.loc[mask, "kruskal_p"], method="fdr_bh")
        df.loc[mask, "kruskal_p_fdr"] = p_adj

    df["significant"] = df["kruskal_p_fdr"] < 0.05
    return df.sort_values("kruskal_p", na_position="last")


# ---------------------------------------------------------------------------
# 2. Post-hoc pairwise tests (Dunn's)
# ---------------------------------------------------------------------------

def posthoc_pairwise(
    features: pd.DataFrame,
    labels: pd.Series,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Dunn's pairwise post-hoc tests for every feature.

    Returns a flat DataFrame with columns: ``feature``, ``group_a``,
    ``group_b``, ``z_stat``, ``p_raw``, ``p_corrected`` (Bonferroni within
    each feature), ``significant``, ``cohens_d``.
    """
    X, y, groups = _prepare(features, labels)

    parts: list[pd.DataFrame] = []
    for feat in X.columns:
        vals = X[feat]
        gdata, gnames = [], []
        for g in groups:
            gv = vals[y == g].dropna().values
            if len(gv) >= 2:
                gdata.append(gv)
                gnames.append(g)

        if len(gdata) < 2:
            continue

        dunn_df = _dunns_test(gdata, gnames)
        dunn_df["feature"] = feat

        name_to_arr = dict(zip(gnames, gdata))
        dunn_df["cohens_d"] = [
            _cohens_d(name_to_arr[r["group_a"]], name_to_arr[r["group_b"]])
            for _, r in dunn_df.iterrows()
        ]
        dunn_df["significant"] = dunn_df["p_corrected"] < alpha
        parts.append(dunn_df)

    cols = ["feature", "group_a", "group_b", "z_stat",
            "p_raw", "p_corrected", "significant", "cohens_d"]
    if not parts:
        return pd.DataFrame(columns=cols)
    return pd.concat(parts, ignore_index=True)[cols]


# ---------------------------------------------------------------------------
# 3. Effect sizes
# ---------------------------------------------------------------------------

def effect_sizes(features: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """Eta-squared (omnibus) and pairwise Cohen's d per feature.

    Returns DataFrame (index = ``feature``) with ``eta_squared`` and
    ``d_<A>_vs_<B>`` columns for every group pair, sorted by descending
    eta-squared.
    """
    X, y, groups = _prepare(features, labels)
    pairs = list(combinations(groups, 2))

    records: list[dict[str, Any]] = []
    for feat in X.columns:
        vals = X[feat]
        row: dict[str, Any] = {"feature": feat}

        group_arrs = {g: vals[y == g].dropna().values for g in groups}
        valid = [a for a in group_arrs.values() if len(a) >= 2]

        if len(valid) >= 2:
            pooled = np.concatenate(valid)
            gm = pooled.mean()
            ss_b = sum(len(a) * (a.mean() - gm) ** 2 for a in valid)
            ss_t = float(np.sum((pooled - gm) ** 2))
            row["eta_squared"] = float(ss_b / ss_t) if ss_t > 0 else np.nan
        else:
            row["eta_squared"] = np.nan

        for ga, gb in pairs:
            row[f"d_{ga}_vs_{gb}"] = _cohens_d(group_arrs[ga], group_arrs[gb])

        records.append(row)

    df = pd.DataFrame(records).set_index("feature")
    return df.sort_values("eta_squared", ascending=False, na_position="last")


# ---------------------------------------------------------------------------
# 4. Multicollinearity
# ---------------------------------------------------------------------------

def correlation_matrix(features: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation matrix of feature columns."""
    return features.corr()


def compute_vif(
    features: pd.DataFrame,
    max_features: int | None = None,
) -> pd.DataFrame:
    """Variance Inflation Factor per feature (median-imputed).

    Parameters
    ----------
    max_features
        Cap the number of columns analysed (useful for very wide matrices).

    Returns
    -------
    pd.DataFrame
        Columns: ``feature``, ``vif``, ``collinear`` (VIF > 10).
    """
    Xi = _impute_median(features)
    if max_features is not None:
        Xi = Xi.iloc[:, :max_features]

    cols = list(Xi.columns)
    if len(cols) < 2:
        return pd.DataFrame({
            "feature": cols,
            "vif": [np.nan] * len(cols),
            "collinear": [False] * len(cols),
        })

    X_const = add_constant(Xi.values.astype(float))

    vif_vals: list[float] = []
    for i in range(len(cols)):
        try:
            v = variance_inflation_factor(X_const, i + 1)
            vif_vals.append(float(v))
        except Exception:
            vif_vals.append(np.nan)

    df = pd.DataFrame({"feature": cols, "vif": vif_vals})
    df["collinear"] = df["vif"] > 10.0
    return df.sort_values("vif", ascending=False, na_position="last").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 5. ML-based feature importance
# ---------------------------------------------------------------------------

def _train_rf(
    features: pd.DataFrame,
    labels: pd.Series,
    n_estimators: int = 200,
    random_state: int | None = None,
) -> tuple[RandomForestClassifier, pd.DataFrame, np.ndarray, LabelEncoder]:
    """Train a balanced Random Forest on median-imputed features."""
    rs = RANDOM_SEED if random_state is None else random_state
    X, y, _ = _prepare(features, labels)
    X_imp = _impute_median(X)

    le = LabelEncoder()
    y_enc = le.fit_transform(y.values)

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=rs,
        n_jobs=-1,
        class_weight="balanced",
    )
    clf.fit(X_imp.values, y_enc)
    return clf, X_imp, y_enc, le


def rf_feature_importance(
    features: pd.DataFrame,
    labels: pd.Series,
    n_estimators: int = 200,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Random Forest Gini feature importance.

    Returns DataFrame (index = ``feature``) with columns ``importance``
    and ``rank``, sorted by descending importance.
    """
    clf, X_imp, _, _ = _train_rf(features, labels, n_estimators, random_state)

    imp = pd.DataFrame(
        {"importance": clf.feature_importances_},
        index=X_imp.columns,
    )
    imp = imp.sort_values("importance", ascending=False)
    imp["rank"] = range(1, len(imp) + 1)
    imp.index.name = "feature"
    return imp


def shap_feature_importance(
    features: pd.DataFrame,
    labels: pd.Series,
    n_estimators: int = 200,
    random_state: int | None = None,
) -> dict[str, Any]:
    """SHAP TreeExplainer — group-specific feature contributions.

    Returns
    -------
    dict
        * ``shap_values`` — array ``(n_samples, n_features, n_classes)``
        * ``feature_names`` — list of column names
        * ``class_names`` — decoded label names
        * ``mean_abs_shap`` — DataFrame ``(feature × class)`` of mean |SHAP|
    """
    import shap

    clf, X_imp, _, le = _train_rf(features, labels, n_estimators, random_state)
    explainer = shap.TreeExplainer(clf)
    sv = explainer.shap_values(X_imp.values)

    class_names = list(le.classes_)
    feat_names = list(X_imp.columns)

    # RF TreeExplainer returns list[ndarray] (one per class)
    if isinstance(sv, list):
        sv_arr = np.stack(sv, axis=-1)
    else:
        sv_arr = np.asarray(sv)

    mean_abs: dict[str, np.ndarray] = {}
    for ci, cn in enumerate(class_names):
        if sv_arr.ndim == 3:
            mean_abs[cn] = np.abs(sv_arr[:, :, ci]).mean(axis=0)
        else:
            mean_abs[cn] = np.abs(sv_arr).mean(axis=0)

    mean_abs_df = pd.DataFrame(mean_abs, index=feat_names)
    mean_abs_df.index.name = "feature"

    return {
        "shap_values": sv_arr,
        "feature_names": feat_names,
        "class_names": class_names,
        "mean_abs_shap": mean_abs_df,
    }


def recursive_feature_elimination(
    features: pd.DataFrame,
    labels: pd.Series,
    min_features: int = 5,
    cv_folds: int = 5,
    random_state: int | None = None,
) -> dict[str, Any]:
    """RFECV with Random Forest: find the minimal effective feature set.

    Returns
    -------
    dict
        * ``selected_features`` — list of surviving column names
        * ``ranking`` — DataFrame with ``feature``, ``rank``, ``selected``
        * ``n_features_optimal`` — int
        * ``cv_scores`` — mean cross-val score per *n_features* tried
    """
    rs = RANDOM_SEED if random_state is None else random_state
    X, y, _ = _prepare(features, labels)
    X_imp = _impute_median(X)

    le = LabelEncoder()
    y_enc = le.fit_transform(y.values)

    estimator = RandomForestClassifier(
        n_estimators=100, random_state=rs,
        n_jobs=-1, class_weight="balanced",
    )
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=rs)
    rfecv = RFECV(
        estimator=estimator,
        step=1,
        cv=cv,
        scoring="f1_macro",
        min_features_to_select=min_features,
        n_jobs=-1,
    )
    rfecv.fit(X_imp.values, y_enc)

    ranking_df = pd.DataFrame({
        "feature": X_imp.columns,
        "rank": rfecv.ranking_,
        "selected": rfecv.support_,
    }).sort_values("rank")

    selected = list(ranking_df.loc[ranking_df["selected"], "feature"])

    cv_scores = None
    if hasattr(rfecv, "cv_results_"):
        cv_scores = rfecv.cv_results_.get("mean_test_score")

    return {
        "selected_features": selected,
        "ranking": ranking_df,
        "n_features_optimal": int(rfecv.n_features_),
        "cv_scores": cv_scores,
    }


# ---------------------------------------------------------------------------
# 6. Group-specific discriminative feature selection
# ---------------------------------------------------------------------------

def select_discriminative_features(
    features: pd.DataFrame,
    labels: pd.Series,
    n_per_group: int = 5,
    alpha: float = 0.05,
) -> dict[str, list[str]]:
    """Top *n_per_group* discriminative features per class.

    Scoring combines:

    * Mean |Cohen's d| of the target group vs every other group
      (weighted by statistical significance from Dunn's test).
    * Random Forest Gini importance.

    Features already selected for another group receive a diversity penalty
    so each group surfaces different characteristics where possible.

    Returns
    -------
    dict
        Mapping ``group_name -> [feature, …]``.
    """
    X, y, groups = _prepare(features, labels)

    ph = posthoc_pairwise(X, y, alpha=alpha)
    rf_imp = rf_feature_importance(X, y)

    feat_names = list(X.columns)
    scores: dict[str, dict[str, float]] = {g: {} for g in groups}

    for feat in feat_names:
        feat_ph = ph[ph["feature"] == feat]
        rf_sc = float(rf_imp.loc[feat, "importance"]) if feat in rf_imp.index else 0.0

        for g in groups:
            others = [og for og in groups if og != g]
            d_sum, sig_count, n_pairs = 0.0, 0, 0

            for og in others:
                mask = (
                    ((feat_ph["group_a"] == g) & (feat_ph["group_b"] == og))
                    | ((feat_ph["group_a"] == og) & (feat_ph["group_b"] == g))
                )
                pair = feat_ph[mask]
                if pair.empty:
                    continue
                r = pair.iloc[0]
                cd = abs(r["cohens_d"]) if np.isfinite(r["cohens_d"]) else 0.0
                d_sum += cd
                sig_count += int(r["significant"])
                n_pairs += 1

            if n_pairs > 0:
                mean_d = d_sum / n_pairs
                sig_frac = sig_count / n_pairs
                score = mean_d * (0.5 + 0.5 * sig_frac) + 0.3 * rf_sc
            else:
                score = 0.1 * rf_sc

            scores[g][feat] = score

    selected: dict[str, list[str]] = {}
    used: set[str] = set()

    for g in groups:
        ranking = sorted(scores[g].items(), key=lambda x: x[1], reverse=True)
        chosen: list[str] = []
        for feat, sc in ranking:
            if len(chosen) >= n_per_group:
                break
            adjusted = sc * (0.5 if feat in used else 1.0)
            if adjusted > 0:
                chosen.append(feat)
        selected[g] = chosen
        used.update(chosen)

    return selected


# ---------------------------------------------------------------------------
# 7. Visualization
# ---------------------------------------------------------------------------

def plot_feature_boxplots(
    features: pd.DataFrame,
    labels: pd.Series,
    feature_names: list[str] | None = None,
    ncols: int = 4,
    figsize: tuple[float, float] | None = None,
) -> Any:
    """Grid of boxplots — one panel per feature, coloured by group."""
    import matplotlib.pyplot as plt

    X, y, groups = _prepare(features, labels)
    cols = feature_names or list(X.columns)
    nrows = max(1, int(np.ceil(len(cols) / ncols)))
    if figsize is None:
        figsize = (4 * ncols, 3.5 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_2d(axes)
    palette = ["#2ecc71", "#3498db", "#e74c3c"]

    for i, feat in enumerate(cols):
        ax = axes[i // ncols, i % ncols]
        data = [X.loc[y == g, feat].dropna().values for g in groups]
        bp = ax.boxplot(data, showmeans=True, patch_artist=True)
        for patch, colour in zip(bp["boxes"], palette[: len(groups)]):
            patch.set_facecolor(colour)
            patch.set_alpha(0.6)
        ax.set_xticklabels(groups, fontsize=8)
        ax.set_title(feat, fontsize=9)

    for i in range(len(cols), nrows * ncols):
        axes[i // ncols, i % ncols].set_visible(False)

    fig.suptitle("Feature distributions by group", fontsize=13)
    fig.tight_layout()
    return fig


def plot_feature_violins(
    features: pd.DataFrame,
    labels: pd.Series,
    feature_names: list[str] | None = None,
    ncols: int = 4,
    figsize: tuple[float, float] | None = None,
) -> Any:
    """Grid of violin plots — one panel per feature, coloured by group."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    X, y, groups = _prepare(features, labels)
    cols = feature_names or list(X.columns)
    nrows = max(1, int(np.ceil(len(cols) / ncols)))
    if figsize is None:
        figsize = (4 * ncols, 3.5 * nrows)

    df = X[cols].copy()
    df["group"] = y
    df_melt = df.melt(id_vars="group", var_name="feature", value_name="value")

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_2d(axes)
    palette = {"Winners": "#2ecc71", "Steady": "#3498db", "Losers": "#e74c3c"}

    for i, feat in enumerate(cols):
        ax = axes[i // ncols, i % ncols]
        subset = df_melt[df_melt["feature"] == feat]
        sns.violinplot(
            data=subset, x="group", y="value", hue="group",
            order=groups, hue_order=groups, palette=palette,
            ax=ax, inner="quartile", density_norm="width", legend=False,
        )
        ax.set_title(feat, fontsize=9)
        ax.set_xlabel("")
        ax.set_ylabel("")

    for i in range(len(cols), nrows * ncols):
        axes[i // ncols, i % ncols].set_visible(False)

    fig.suptitle("Feature distributions by group (violin)", fontsize=13)
    fig.tight_layout()
    return fig


def plot_radar_chart(
    features: pd.DataFrame,
    labels: pd.Series,
    feature_names: list[str] | None = None,
    figsize: tuple[float, float] = (8, 8),
) -> Any:
    """Radar (spider) chart of z-scored group medians."""
    import matplotlib.pyplot as plt

    X, y, groups = _prepare(features, labels)
    cols = feature_names or list(X.columns)

    X_std = (X[cols] - X[cols].mean()) / X[cols].std().replace(0, np.nan)
    medians = X_std.copy()
    medians["group"] = y
    group_med = medians.groupby("group")[cols].median()
    group_med = group_med.reindex([g for g in groups if g in group_med.index])

    n_feat = len(cols)
    angles = np.linspace(0, 2 * np.pi, n_feat, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"polar": True})
    colours = {"Winners": "#2ecc71", "Steady": "#3498db", "Losers": "#e74c3c"}

    for g in group_med.index:
        vals = group_med.loc[g].fillna(0).tolist()
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=2, label=g, color=colours.get(g))
        ax.fill(angles, vals, alpha=0.15, color=colours.get(g))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cols, fontsize=7)
    ax.set_title("Radar chart: standardised group medians", fontsize=12, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    fig.tight_layout()
    return fig


def plot_importance_bar(
    importance: pd.DataFrame,
    top_n: int = 15,
    ax: Any | None = None,
) -> Any:
    """Horizontal bar chart of feature importances.

    Parameters
    ----------
    importance
        DataFrame with an ``importance`` column (index = feature name),
        e.g. from :func:`rf_feature_importance`.
    """
    import matplotlib.pyplot as plt

    top = importance.nlargest(top_n, "importance").sort_values("importance")
    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(7, max(3, 0.4 * top_n)))

    ax.barh(top.index, top["importance"], color="steelblue", edgecolor="white")
    ax.set_xlabel("Importance (Gini)")
    ax.set_title(f"Top {top_n} features — Random Forest importance")

    if created:
        plt.tight_layout()
    return ax


# ---------------------------------------------------------------------------
# 8. Convenience: full analysis report
# ---------------------------------------------------------------------------

def full_analysis_report(
    features: pd.DataFrame,
    labels: pd.Series,
    *,
    run_shap: bool = True,
    run_rfe: bool = True,
    n_per_group: int = 5,
) -> dict[str, Any]:
    """Run the complete Phase-4 analysis battery.

    Returns a dict whose keys mirror the individual function names:
    ``anova``, ``posthoc``, ``effects``, ``correlation``, ``vif``,
    ``rf_importance``, ``selected``, and optionally ``shap``, ``rfe``.
    """
    logger.info("Running full analysis report …")
    report: dict[str, Any] = {}

    report["anova"] = anova_by_group(features, labels)
    report["posthoc"] = posthoc_pairwise(features, labels)
    report["effects"] = effect_sizes(features, labels)
    report["correlation"] = correlation_matrix(features)
    report["vif"] = compute_vif(features)
    report["rf_importance"] = rf_feature_importance(features, labels)

    if run_shap:
        try:
            report["shap"] = shap_feature_importance(features, labels)
        except Exception:
            logger.warning("SHAP analysis failed", exc_info=True)
            report["shap"] = None

    if run_rfe:
        try:
            report["rfe"] = recursive_feature_elimination(features, labels)
        except Exception:
            logger.warning("RFE failed", exc_info=True)
            report["rfe"] = None

    report["selected"] = select_discriminative_features(
        features, labels, n_per_group=n_per_group,
    )

    logger.info("Full analysis report complete.")
    return report
