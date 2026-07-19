"""Correlation-based clustering of observed confounders."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Optional

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

if TYPE_CHECKING:
    from causalis.dgp.causaldata import CausalData
    from causalis.dgp.multicausaldata import MultiCausalData


CorrelationMethod = Literal["pearson", "spearman"]
LinkageMethod = Literal["average", "complete", "single"]
OutcomeImportanceMethod = Literal["mean", "max"]


def _resolve_feature_importance(effect_estimation: Any) -> dict[str, Any]:
    """Resolve an IRM native feature-importance payload."""
    candidates: list[Any] = [effect_estimation]
    if isinstance(effect_estimation, dict):
        candidates.extend(
            effect_estimation.get(key)
            for key in ("feature_importance", "estimate", "model", "diagnostic_data")
        )

    for candidate in candidates:
        if candidate is None:
            continue
        if isinstance(candidate, dict) and {
            "feature_names",
            "nuisances",
        } <= set(candidate):
            return candidate

        diagnostic_data = getattr(candidate, "diagnostic_data", None)
        diagnostic_importance = getattr(
            diagnostic_data,
            "feature_importance",
            None,
        )
        if isinstance(diagnostic_importance, dict):
            return diagnostic_importance

        model_importance = getattr(candidate, "feature_importance_", None)
        if isinstance(model_importance, dict):
            return model_importance

        direct_importance = getattr(candidate, "feature_importance", None)
        if isinstance(direct_importance, dict):
            return direct_importance

    raise ValueError(
        "Feature importance is unavailable. Fit IRM with "
        "store_diagnostics=True and pass its estimate or fitted model."
    )


def _normalized_nuisance_importance(
    feature_importance: dict[str, Any],
    nuisance: str,
    *,
    n_features: int,
) -> np.ndarray:
    """Validate and normalize one nuisance-importance vector."""
    nuisances = feature_importance.get("nuisances")
    if not isinstance(nuisances, dict):
        raise ValueError("Feature importance must contain a 'nuisances' mapping.")

    payload = nuisances.get(nuisance)
    if not isinstance(payload, dict) or not payload.get("available", False):
        raise ValueError(f"Feature importance for nuisance {nuisance!r} is unavailable.")

    try:
        values = np.asarray(payload.get("mean"), dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Feature importance for nuisance {nuisance!r} must be numeric."
        ) from exc

    if values.size != n_features:
        raise ValueError(
            f"Feature importance for nuisance {nuisance!r} has {values.size} "
            f"values; expected {n_features}."
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(
            f"Feature importance for nuisance {nuisance!r} must be finite."
        )

    values = np.maximum(values, 0.0)
    total = float(np.sum(values))
    if total <= 0.0:
        raise ValueError(
            f"Feature importance for nuisance {nuisance!r} must have a positive sum."
        )
    return values / total


def cluster_confounders(
    data: CausalData | MultiCausalData,
    *,
    min_abs_correlation: float = 0.7,
    correlation_method: CorrelationMethod = "pearson",
    linkage_method: LinkageMethod = "complete",
    max_samples: Optional[int] = 200_000,
    random_state: Optional[int] = 42,
) -> list[list[str]]:
    r"""
    Cluster confounders using their absolute pairwise correlations.

    The clustering distance is

    .. math::

        d(X_i, X_j) = 1 - |\operatorname{corr}(X_i, X_j)|.

    Hierarchical clustering is cut at ``1 - min_abs_correlation``. Negative
    and positive correlations of the same magnitude are therefore treated as
    equally similar. By default, complete linkage is used, so every pair of
    features in a non-singleton cluster has an absolute correlation at least
    as large as ``min_abs_correlation`` on the sampled data.

    Parameters
    ----------
    data : CausalData or MultiCausalData
        Causal data contract exposing numeric ``df`` and ``confounders``
        attributes.
    min_abs_correlation : float, default 0.7
        Correlation cut-off in the closed interval [0, 1]. Larger values form
        smaller, more strongly related clusters.
    correlation_method : {"pearson", "spearman"}, default "pearson"
        Pairwise correlation measure. Spearman correlation is useful for
        monotonic non-linear relationships but requires more work and memory.
    linkage_method : {"average", "complete", "single"}, default "complete"
        Hierarchical-linkage rule. Only complete linkage guarantees the
        pairwise threshold interpretation described above.
    max_samples : int or None, default 200000
        Maximum number of rows used to estimate correlations. If the dataset
        is larger, rows are sampled without replacement. Pass ``None`` to use
        all rows.
    random_state : int or None, default 42
        Seed used for row sampling.

    Returns
    -------
    list[list[str]]
        Clusters in the original confounder order. Features within each cluster
        also preserve their original order. Singleton clusters are retained.

    Examples
    --------
    >>> clusters = cluster_confounders(data, min_abs_correlation=0.8)
    >>> strongest_group = clusters[0]
    >>> strongest_group  # doctest: +SKIP
    ['income', 'salary', 'credit_limit']

    Notes
    -----
    Missing correlations can occur when a rare feature is constant in the
    sampled rows. They are treated as zero correlation, leaving that feature
    separate unless it is linked to another feature by valid correlations.
    """
    if not hasattr(data, "df") or not hasattr(data, "confounders"):
        raise TypeError(
            "data must be a CausalData or MultiCausalData object exposing "
            "'df' and 'confounders'."
        )

    df = getattr(data, "df")
    if not isinstance(df, pd.DataFrame):
        raise TypeError("data.df must be a pandas DataFrame.")

    confounders = list(getattr(data, "confounders"))
    if not confounders:
        raise ValueError("data.confounders must contain at least one feature.")
    if not all(isinstance(feature, str) for feature in confounders):
        raise TypeError("data.confounders must contain only strings.")
    if len(set(confounders)) != len(confounders):
        raise ValueError("data.confounders must not contain duplicate names.")

    missing = [feature for feature in confounders if feature not in df.columns]
    if missing:
        raise ValueError(f"Confounders are missing from data.df: {missing}.")

    threshold = float(min_abs_correlation)
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("min_abs_correlation must be a finite value in [0, 1].")
    if correlation_method not in {"pearson", "spearman"}:
        raise ValueError("correlation_method must be 'pearson' or 'spearman'.")
    if linkage_method not in {"average", "complete", "single"}:
        raise ValueError("linkage_method must be 'average', 'complete', or 'single'.")

    if max_samples is not None:
        if isinstance(max_samples, bool) or not isinstance(max_samples, (int, np.integer)):
            raise TypeError("max_samples must be an integer or None.")
        if int(max_samples) < 2:
            raise ValueError("max_samples must be at least 2 when provided.")

    if len(df) < 2:
        raise ValueError("At least two observations are required for clustering.")
    if len(confounders) == 1:
        return [confounders]

    if max_samples is not None and len(df) > int(max_samples):
        rng = np.random.default_rng(random_state)
        row_positions = np.sort(
            rng.choice(len(df), size=int(max_samples), replace=False)
        )
        features_df = df.iloc[row_positions][confounders]
    else:
        features_df = df[confounders]

    correlation = features_df.corr(method=correlation_method).to_numpy(dtype=float)
    abs_correlation = np.abs(correlation)
    abs_correlation = np.nan_to_num(
        abs_correlation,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    abs_correlation = np.clip(abs_correlation, 0.0, 1.0)
    np.fill_diagonal(abs_correlation, 1.0)

    distance = 1.0 - abs_correlation
    distance = (distance + distance.T) / 2.0
    np.fill_diagonal(distance, 0.0)

    condensed_distance = squareform(distance, checks=False)
    hierarchy = linkage(condensed_distance, method=linkage_method)
    raw_labels = fcluster(
        hierarchy,
        t=1.0 - threshold,
        criterion="distance",
    )

    clusters_by_label: dict[int, list[str]] = {}
    for feature, raw_label in zip(confounders, raw_labels):
        clusters_by_label.setdefault(int(raw_label), []).append(feature)

    clusters = list(clusters_by_label.values())
    feature_position = {feature: position for position, feature in enumerate(confounders)}
    clusters.sort(key=lambda group: min(feature_position[feature] for feature in group))
    return clusters


def rank_confounder_clusters(
    data: CausalData | MultiCausalData,
    effect_estimation: Any,
    *,
    min_abs_correlation: float = 0.7,
    correlation_method: CorrelationMethod = "pearson",
    linkage_method: LinkageMethod = "complete",
    max_samples: Optional[int] = 200_000,
    random_state: Optional[int] = 42,
    outcome_importance: OutcomeImportanceMethod = "mean",
) -> pd.DataFrame:
    r"""
    Cluster confounders and rank the clusters by joint nuisance importance.

    For each correlation cluster :math:`C_g`, the treatment and outcome
    importances are aggregated as

    .. math::

        I_{D,g} = \sum_{j \in C_g} I_{m,j},

    and, by default,

    .. math::

        I_{Y,g} = \frac{1}{2}\left(
            \sum_{j \in C_g} I_{g0,j}
            + \sum_{j \in C_g} I_{g1,j}
        \right).

    Clusters are sorted by ``score = importance_d * importance_y`` in
    descending order. Native importance vectors are normalized separately for
    ``m``, ``g0``, and ``g1`` before aggregation.

    Parameters
    ----------
    data : CausalData or MultiCausalData
        Data containing the confounders to cluster.
    effect_estimation : Any
        Fitted IRM, causal estimate with diagnostic data, or dictionary
        containing one of them. Feature importance must have been collected.
    min_abs_correlation, correlation_method, linkage_method, max_samples, random_state
        Forwarded to :func:`cluster_confounders`.
    outcome_importance : {"mean", "max"}, default "mean"
        Combine the cluster-level ``g0`` and ``g1`` importance by their mean or
        maximum. ``"max"`` is more conservative for features important in only
        one treatment arm.

    Returns
    -------
    pandas.DataFrame
        Cluster ranking with columns ``cluster_id``, ``features``,
        ``n_features``, ``importance_d``, ``importance_g0``,
        ``importance_g1``, ``importance_y``, and ``score``. Row zero is the
        highest-ranked cluster; ``features`` can be passed directly to
        ``sensitivity_benchmark_group``.

    Examples
    --------
    >>> ranking = rank_confounder_clusters(data, estimate)
    >>> top_group = ranking.loc[0, "features"]
    """
    if outcome_importance not in {"mean", "max"}:
        raise ValueError("outcome_importance must be 'mean' or 'max'.")

    clusters = cluster_confounders(
        data,
        min_abs_correlation=min_abs_correlation,
        correlation_method=correlation_method,
        linkage_method=linkage_method,
        max_samples=max_samples,
        random_state=random_state,
    )

    confounders = list(getattr(data, "confounders"))
    feature_importance = _resolve_feature_importance(effect_estimation)
    feature_names_raw = feature_importance.get("feature_names")
    if not isinstance(feature_names_raw, (list, tuple)):
        raise ValueError("Feature importance must contain 'feature_names'.")
    feature_names = [str(feature) for feature in feature_names_raw]
    if len(set(feature_names)) != len(feature_names):
        raise ValueError("Feature importance contains duplicate feature names.")
    if set(feature_names) != set(confounders):
        missing = [feature for feature in confounders if feature not in feature_names]
        extra = [feature for feature in feature_names if feature not in confounders]
        raise ValueError(
            "Feature importance must match data.confounders. "
            f"Missing: {missing}; extra: {extra}."
        )

    n_features = len(feature_names)
    importance_m = _normalized_nuisance_importance(
        feature_importance,
        "m",
        n_features=n_features,
    )
    importance_g0 = _normalized_nuisance_importance(
        feature_importance,
        "g0",
        n_features=n_features,
    )
    importance_g1 = _normalized_nuisance_importance(
        feature_importance,
        "g1",
        n_features=n_features,
    )
    feature_index = {
        feature: position for position, feature in enumerate(feature_names)
    }

    rows: list[dict[str, Any]] = []
    for cluster_id, features in enumerate(clusters):
        indices = [feature_index[feature] for feature in features]
        cluster_importance_d = float(np.sum(importance_m[indices]))
        cluster_importance_g0 = float(np.sum(importance_g0[indices]))
        cluster_importance_g1 = float(np.sum(importance_g1[indices]))
        if outcome_importance == "mean":
            cluster_importance_y = 0.5 * (
                cluster_importance_g0 + cluster_importance_g1
            )
        else:
            cluster_importance_y = max(
                cluster_importance_g0,
                cluster_importance_g1,
            )

        rows.append(
            {
                "cluster_id": cluster_id,
                "features": list(features),
                "n_features": len(features),
                "importance_d": cluster_importance_d,
                "importance_g0": cluster_importance_g0,
                "importance_g1": cluster_importance_g1,
                "importance_y": cluster_importance_y,
                "score": cluster_importance_d * cluster_importance_y,
            }
        )

    columns = [
        "cluster_id",
        "features",
        "n_features",
        "importance_d",
        "importance_g0",
        "importance_g1",
        "importance_y",
        "score",
    ]
    ranking = pd.DataFrame(rows, columns=columns)
    return ranking.sort_values(
        "score",
        ascending=False,
        kind="mergesort",
    ).reset_index(drop=True)


__all__ = ["cluster_confounders", "rank_confounder_clusters"]
