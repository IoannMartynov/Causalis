from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from causalis.data_contracts import PanelDataDID
from causalis.data_contracts.panel_did_estimate import CallawaySantAnnaDIDEstimate


_CHECK_COLUMNS = ["test", "flag", "value", "threshold", "message"]


def _as_finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def _validate_positive_float(value: float, name: str) -> float:
    out = float(value)
    if not np.isfinite(out) or out <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return out


def _validate_probability(value: float, name: str) -> float:
    out = float(value)
    if not np.isfinite(out) or not (0.0 <= out <= 1.0):
        raise ValueError(f"{name} must be finite and in [0, 1].")
    return out


def _validate_positive_int(value: int, name: str) -> int:
    out = int(value)
    if out <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return out


def _append_check(
    checks: list[dict[str, Any]],
    *,
    test: str,
    flag: str,
    value: Any,
    threshold: str,
    message: str,
) -> None:
    checks.append(
        {
            "test": test,
            "flag": flag,
            "value": value,
            "threshold": threshold,
            "message": message,
        }
    )


def _ensure_estimate(estimate: Any) -> CallawaySantAnnaDIDEstimate:
    if not isinstance(estimate, CallawaySantAnnaDIDEstimate):
        raise TypeError("estimate must be a CallawaySantAnnaDIDEstimate instance.")
    return estimate


def _resolve_inputs(
    data_or_estimate: PanelDataDID | CallawaySantAnnaDIDEstimate,
    estimate: Optional[CallawaySantAnnaDIDEstimate],
) -> tuple[Optional[PanelDataDID], CallawaySantAnnaDIDEstimate]:
    if estimate is None:
        return None, _ensure_estimate(data_or_estimate)
    if not isinstance(data_or_estimate, PanelDataDID):
        raise TypeError("data must be a PanelDataDID instance when estimate is passed.")
    return data_or_estimate, _ensure_estimate(estimate)


def _get_diagnostic_frame(
    estimate: CallawaySantAnnaDIDEstimate,
    name: str,
) -> pd.DataFrame:
    value = dict(estimate.diagnostics or {}).get(name)
    if isinstance(value, pd.DataFrame):
        return value.copy()
    return pd.DataFrame()


def _finite_numeric(series: pd.Series) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    return out[np.isfinite(out)]


def _safe_t_stat(estimate: pd.Series, se: pd.Series) -> pd.Series:
    estimate_values = pd.to_numeric(estimate, errors="coerce").to_numpy(dtype=float)
    se_values = pd.to_numeric(se, errors="coerce").to_numpy(dtype=float)
    out = np.full(estimate_values.shape[0], np.nan, dtype=float)
    positive = np.isfinite(se_values) & (se_values > 0.0)
    out[positive] = estimate_values[positive] / se_values[positive]
    zero = np.isfinite(se_values) & (se_values == 0.0) & np.isfinite(estimate_values)
    out[zero & (np.abs(estimate_values) <= 1e-16)] = 0.0
    out[zero & (np.abs(estimate_values) > 1e-16)] = np.inf
    return pd.Series(out, index=estimate.index)


def _simple_weight_rows(estimate: CallawaySantAnnaDIDEstimate) -> pd.DataFrame:
    weights = _get_diagnostic_frame(estimate, "weights")
    if not weights.empty and {"aggregate", "cell_id", "weight"}.issubset(weights.columns):
        simple = weights.loc[weights["aggregate"] == "simple", ["cell_id", "weight"]].copy()
        if not simple.empty:
            simple["cell_id"] = pd.to_numeric(simple["cell_id"], errors="raise").astype(int)
            simple["weight"] = pd.to_numeric(simple["weight"], errors="raise").astype(float)
            return simple

    post = estimate.att_gt.loc[
        estimate.att_gt["is_post_treatment"].fillna(False).astype(bool)
    ].copy()
    if post.empty:
        return pd.DataFrame(columns=["cell_id", "weight"])
    if "n_treated" in post.columns:
        weights_raw = pd.to_numeric(post["n_treated"], errors="coerce").to_numpy(dtype=float)
    else:
        weights_raw = np.ones(len(post), dtype=float)
    if not np.isfinite(weights_raw).all() or float(weights_raw.sum()) <= 0.0:
        weights_raw = np.ones(len(post), dtype=float)
    weights_raw = weights_raw / float(weights_raw.sum())
    return pd.DataFrame(
        {
            "cell_id": pd.to_numeric(post["cell_id"], errors="raise").astype(int).to_numpy(),
            "weight": weights_raw,
        }
    )


def _aggregate_unit_scores(estimate: CallawaySantAnnaDIDEstimate) -> pd.Series:
    influence = _get_diagnostic_frame(estimate, "influence_scores")
    if influence.empty:
        raise ValueError(
            "estimate.diagnostics must include influence_scores. "
            "Call estimate(diagnostic_data=True) for post-inference influence diagnostics."
        )
    weights = _simple_weight_rows(estimate)
    if weights.empty:
        raise ValueError("No simple aggregate ATT weights are available for influence diagnostics.")

    score_parts: list[np.ndarray] = []
    weight_values: list[float] = []
    for row in weights.to_dict("records"):
        cell_id = int(row["cell_id"])
        column: Any
        if cell_id in influence.columns:
            column = cell_id
        elif str(cell_id) in influence.columns:
            column = str(cell_id)
        else:
            continue
        score_parts.append(pd.to_numeric(influence[column], errors="coerce").to_numpy(dtype=float))
        weight_values.append(float(row["weight"]))

    if not score_parts:
        raise ValueError("No influence-score columns match the simple aggregate ATT cells.")
    scores = np.column_stack(score_parts)
    weights_array = np.asarray(weight_values, dtype=float)
    aggregate_scores = scores @ weights_array
    return pd.Series(aggregate_scores, index=influence.index, name="influence_score")


def did_post_inference_cell_table(
    estimate: CallawaySantAnnaDIDEstimate,
) -> pd.DataFrame:
    """
    Return fitted cell-level inference diagnostics for a Callaway & Sant'Anna estimate.

    This function merges the point estimates and standard errors with auxiliary
    cell-level diagnostics produced during the estimation process (e.g., propensity
    clipping, covariate balance, and effective sample sizes).

    Parameters
    ----------
    estimate : CallawaySantAnnaDIDEstimate
        The fitted estimate object from a ``CallawaySantAnnaDID`` model.

    Returns
    -------
    pd.DataFrame
        A table where each row corresponds to an ATT(g,t) cell, containing:
            - ``group``: The treatment group (first treated period).
            - ``time``: The calendar time period.
            - ``att``: The point estimate for this cell.
            - ``se``: The standard error.
            - ``t_stat``: The t-statistic (att / se).
            - ``is_post_treatment``: Whether the cell is in the post-treatment period.
            - Additional columns from the model's cell diagnostics (e.g., ``control_ess``,
              ``max_propensity_clip``).

    Examples
    --------
    >>> from causalis.scenarios.did import CallawaySantAnnaDID, generate_did_gamma_26
    >>> from causalis.scenarios.did.refutation import did_post_inference_cell_table
    >>> data = generate_did_gamma_26(n_units=200, n_periods=5, seed=42)
    >>> model = CallawaySantAnnaDID().fit(data)
    >>> est = model.estimate(diagnostic_data=True)
    >>> cells = did_post_inference_cell_table(est)
    >>> cells[["group", "time", "att", "t_stat"]].head()
    """

    estimate = _ensure_estimate(estimate)
    out = estimate.att_gt.copy()
    if out.empty:
        return out
    if {"att", "se"}.issubset(out.columns):
        out["t_stat"] = _safe_t_stat(out["att"], out["se"])
        out["abs_t_stat"] = out["t_stat"].abs()

    cell_diagnostics = _get_diagnostic_frame(estimate, "cell_diagnostics")
    if not cell_diagnostics.empty and "cell_id" in cell_diagnostics.columns:
        extra_cols = [
            col
            for col in cell_diagnostics.columns
            if col != "cell_id" and col not in out.columns
        ]
        if extra_cols:
            out = out.merge(
                cell_diagnostics[["cell_id", *extra_cols]],
                on="cell_id",
                how="left",
                validate="one_to_one",
            )
    return out


def did_influence_table(
    data_or_estimate: PanelDataDID | CallawaySantAnnaDIDEstimate,
    estimate: Optional[CallawaySantAnnaDIDEstimate] = None,
    *,
    top_n: Optional[int] = None,
) -> pd.DataFrame:
    """
    Return unit-level influence shares for the simple overall ATT.

    Influence scores represent the first-order contribution of each unit to the
    final estimate. This function aggregates influence scores across all ATT(g,t)
    cells used in the simple average ATT and calculates their relative shares.

    Parameters
    ----------
    data_or_estimate : PanelDataDID or CallawaySantAnnaDIDEstimate
        If a ``PanelDataDID`` object is provided, the second argument ``estimate``
        must be supplied. If a ``CallawaySantAnnaDIDEstimate`` is provided, the
        data will be resolved from the estimate if possible.
    estimate : CallawaySantAnnaDIDEstimate, optional
        The fitted estimate object. Required if the first argument is data.
    top_n : int, optional
        If provided, only return the top N units by absolute influence share.

    Returns
    -------
    pd.DataFrame
        A table of unit-level influence metrics:
            - ``unit_id``: The unit identifier.
            - ``rank``: Ranking by absolute influence.
            - ``influence_score``: The raw aggregated influence score.
            - ``abs_influence_score``: The absolute value of the score.
            - ``abs_influence_share``: The share of total absolute influence.

    Notes
    -----
    The influence share for unit :math:`i` is defined as:

    .. math::
        S_i = \\frac{|\\psi_i|}{\\sum_{j=1}^n |\\psi_j|}

    where :math:`\\psi_i` is the influence score of unit :math:`i` on the
    target parameter (e.g., the simple aggregate ATT). High influence shares
    indicate units that have a disproportionate impact on the final result.

    Examples
    --------
    >>> from causalis.scenarios.did import CallawaySantAnnaDID, generate_did_gamma_26
    >>> from causalis.scenarios.did.refutation import did_influence_table
    >>> data = generate_did_gamma_26(n_units=200, n_periods=5, seed=42)
    >>> model = CallawaySantAnnaDID().fit(data)
    >>> est = model.estimate(diagnostic_data=True)
    >>> influence = did_influence_table(data, est, top_n=5)
    >>> influence[["unit_id", "abs_influence_share"]]
    """

    data, estimate = _resolve_inputs(data_or_estimate, estimate)
    scores = _aggregate_unit_scores(estimate)
    abs_scores = scores.abs()
    denominator = float(abs_scores.sum())
    if denominator > 0.0 and np.isfinite(denominator):
        shares = abs_scores / denominator
    else:
        shares = pd.Series(np.nan, index=scores.index, dtype=float)

    out = pd.DataFrame(
        {
            estimate.unit_col: scores.index,
            "influence_score": scores.to_numpy(dtype=float),
            "abs_influence_score": abs_scores.to_numpy(dtype=float),
            "abs_influence_share": shares.to_numpy(dtype=float),
        }
    )
    if data is not None and estimate.cluster_col is not None:
        unit_cluster = (
            data.df_analysis()[[data.unit_col, estimate.cluster_col]]
            .drop_duplicates(subset=[data.unit_col])
            .set_index(data.unit_col)[estimate.cluster_col]
        )
        out[estimate.cluster_col] = out[estimate.unit_col].map(unit_cluster)

    out = out.sort_values("abs_influence_score", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1, dtype=int)
    columns = [estimate.unit_col]
    if data is not None and estimate.cluster_col is not None:
        columns.append(estimate.cluster_col)
    columns.extend(["rank", "influence_score", "abs_influence_score", "abs_influence_share"])
    out = out[columns]
    if top_n is not None:
        top_n = _validate_positive_int(top_n, "top_n")
        out = out.head(top_n).reset_index(drop=True)
    return out


def did_cluster_influence_table(
    data: PanelDataDID,
    estimate: CallawaySantAnnaDIDEstimate,
) -> pd.DataFrame:
    """
    Return cluster-level influence shares for clustered Callaway & Sant'Anna estimates.

    Aggregates unit-level influence scores up to the cluster level. This is
    essential for diagnostics when the model was fitted using clustered
    standard errors, as the independence assumption holds at the cluster level.

    Parameters
    ----------
    data : PanelDataDID
        The original panel data used for estimation.
    estimate : CallawaySantAnnaDIDEstimate
        The fitted estimate object, which must have ``cluster_col`` defined.

    Returns
    -------
    pd.DataFrame
        A table of cluster-level influence metrics:
            - ``cluster_id``: The cluster identifier.
            - ``rank``: Ranking by absolute influence.
            - ``influence_score``: Sum of unit influence scores within the cluster.
            - ``abs_influence_share``: Share of total absolute influence.

    Examples
    --------
    >>> from causalis.scenarios.did import CallawaySantAnnaDID, generate_did_gamma_26
    >>> from causalis.scenarios.did.refutation import did_cluster_influence_table
    >>> # Assuming data has a 'cluster' column
    >>> # model = CallawaySantAnnaDID(cluster_col='cluster').fit(data)
    >>> # est = model.estimate(diagnostic_data=True)
    >>> # clusters = did_cluster_influence_table(data, est)
    """

    if not isinstance(data, PanelDataDID):
        raise TypeError("data must be a PanelDataDID instance.")
    estimate = _ensure_estimate(estimate)
    if estimate.cluster_col is None:
        raise ValueError("estimate.cluster_col is not set.")

    scores = _aggregate_unit_scores(estimate)
    df = data.df_analysis()
    cluster_counts = df.groupby(data.unit_col, sort=False)[estimate.cluster_col].nunique(
        dropna=False
    )
    if bool((cluster_counts > 1).any()):
        raise ValueError("cluster_col is not stable within unit.")
    unit_cluster = (
        df[[data.unit_col, estimate.cluster_col]]
        .drop_duplicates(subset=[data.unit_col])
        .set_index(data.unit_col)[estimate.cluster_col]
    )
    cluster_index = pd.Series(scores.index, index=scores.index).map(unit_cluster)
    cluster_scores = scores.groupby(cluster_index, sort=False).sum()
    abs_scores = cluster_scores.abs()
    denominator = float(abs_scores.sum())
    shares = (
        abs_scores / denominator
        if denominator > 0.0 and np.isfinite(denominator)
        else pd.Series(np.nan, index=cluster_scores.index, dtype=float)
    )
    out = pd.DataFrame(
        {
            estimate.cluster_col: cluster_scores.index,
            "influence_score": cluster_scores.to_numpy(dtype=float),
            "abs_influence_score": abs_scores.to_numpy(dtype=float),
            "abs_influence_share": shares.to_numpy(dtype=float),
        }
    )
    out = out.sort_values("abs_influence_score", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1, dtype=int)
    return out[
        [
            estimate.cluster_col,
            "rank",
            "influence_score",
            "abs_influence_score",
            "abs_influence_share",
        ]
    ]


def _influence_ess(abs_scores: pd.Series) -> float | None:
    values = pd.to_numeric(abs_scores, errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    denom = float(np.sum(values**2))
    if denom <= 0.0:
        return None
    return float(np.sum(values) ** 2 / denom)


def _metadata_alignment_check(
    data: Optional[PanelDataDID],
    estimate: CallawaySantAnnaDIDEstimate,
    checks: list[dict[str, Any]],
) -> None:
    if data is None:
        _append_check(
            checks,
            test="data_estimate_alignment",
            flag="YELLOW",
            value="not_checked",
            threshold="PanelDataDID supplied",
            message="PanelDataDID was not supplied; metadata alignment with the fitted data was not checked.",
        )
        return

    mismatches: list[str] = []
    if data.y != estimate.outcome:
        mismatches.append("outcome")
    if data.treated_time != estimate.treatment:
        mismatches.append("treatment")
    if data.unit_col != estimate.unit_col:
        mismatches.append("unit_col")
    if data.time_col != estimate.time_col:
        mismatches.append("time_col")
    if tuple(data.covariates) != tuple(estimate.covariates):
        mismatches.append("covariates")
    if data.cluster_col != estimate.cluster_col:
        mismatches.append("cluster_col")

    n_units_diag = _as_finite_float(dict(estimate.diagnostics or {}).get("n_units"))
    n_units_data = int(data.df_analysis()[data.unit_col].nunique(dropna=False))
    if n_units_diag is not None and int(n_units_diag) != n_units_data:
        mismatches.append("n_units")

    _append_check(
        checks,
        test="data_estimate_alignment",
        flag="RED" if mismatches else "GREEN",
        value=mismatches if mismatches else "aligned",
        threshold="no mismatches",
        message=(
            f"PanelDataDID and estimate metadata differ: {', '.join(mismatches)}."
            if mismatches
            else "PanelDataDID metadata matches the fitted estimate."
        ),
    )


def run_did_post_inference_diagnostics(
    data_or_estimate: PanelDataDID | CallawaySantAnnaDIDEstimate,
    estimate: Optional[CallawaySantAnnaDIDEstimate] = None,
    *,
    max_skipped_post_cell_share: float = 0.20,
    min_control_ess: float = 20.0,
    max_propensity_clip_share: float = 0.05,
    max_abs_weighted_smd: float = 0.25,
    max_top_unit_influence_share: float = 0.20,
    max_top_cluster_influence_share: float = 0.50,
    min_influence_ess: float = 10.0,
    max_abs_pretrend_t_stat: float = 2.0,
    max_simple_cell_weight_share: float = 0.50,
    min_clusters: int = 2,
    require_multiplier_bootstrap: bool = False,
) -> pd.DataFrame:
    """
    Run post-fit Callaway & Sant'Anna inference diagnostics and return a reliability report.

    This function performs a comprehensive suite of checks on the fitted model
    to detect potential issues with identification, estimation stability, and
    excessive influence of individual observations.

    Parameters
    ----------
    data_or_estimate : PanelDataDID or CallawaySantAnnaDIDEstimate
        The first argument can be either the data or the estimate. If data is
        passed, the ``estimate`` argument must also be provided.
    estimate : CallawaySantAnnaDIDEstimate, optional
        The fitted estimate object. Required if the first argument is data.
    max_skipped_post_cell_share : float, default 0.20
        Maximum allowable share of requested post-treatment cells that were
        skipped (e.g., due to lack of treated/control units).
    min_control_ess : float, default 20.0
        Minimum effective sample size for the control group in each ATT(g,t) cell.
    max_propensity_clip_share : float, default 0.05
        Maximum share of units in a cell that can have their propensity scores
        clipped to the boundaries.
    max_abs_weighted_smd : float, default 0.25
        Maximum allowable absolute weighted Standardized Mean Difference (SMD)
        for covariates after IPW reweighting.
    max_top_unit_influence_share : float, default 0.20
        Maximum absolute influence share of the single most influential unit.
    max_top_cluster_influence_share : float, default 0.50
        Maximum absolute influence share of the single most influential cluster.
    min_influence_ess : float, default 10.0
        Minimum Effective Sample Size (ESS) based on influence scores, calculated
        as :math:`1 / \\sum w_i^2`.
    max_abs_pretrend_t_stat : float, default 2.0
        Maximum absolute t-statistic allowed for pre-treatment testing cells.
    max_simple_cell_weight_share : float, default 0.50
        Maximum weight share of any single ATT(g,t) cell in the simple aggregate ATT.
    min_clusters : int, default 2
        Minimum number of clusters required for valid clustered inference.
    require_multiplier_bootstrap : bool, default False
        If True, flags a warning if the multiplier bootstrap was not used for
        simultaneous inference.

    Returns
    -------
    pd.DataFrame
        A diagnostic report with columns:
            - ``test``: Name of the diagnostic check.
            - ``flag``: Status (GREEN, YELLOW, RED).
            - ``value``: Observed value of the metric.
            - ``threshold``: The threshold used for the check.
            - ``message``: Descriptive result message.

    Notes
    -----
    The influence-based ESS is a measure of how concentrated the estimate's
    dependence is on a small subset of units. It is defined as:

    .. math::
        ESS_{\\psi} = \\frac{(\\sum |\\psi_i|)^2}{\\sum \\psi_i^2}

    Low values suggest the result may be driven by outliers.

    Examples
    --------
    >>> from causalis.scenarios.did import CallawaySantAnnaDID, generate_did_gamma_26
    >>> from causalis.scenarios.did.refutation import run_did_post_inference_diagnostics
    >>> data = generate_did_gamma_26(n_units=300, n_periods=5, seed=42)
    >>> model = CallawaySantAnnaDID().fit(data)
    >>> est = model.estimate(diagnostic_data=True)
    >>> report = run_did_post_inference_diagnostics(data, est)
    >>> report[["test", "flag", "value"]]
    """

    data, estimate = _resolve_inputs(data_or_estimate, estimate)
    max_skipped_post_cell_share = _validate_probability(
        max_skipped_post_cell_share,
        "max_skipped_post_cell_share",
    )
    min_control_ess = _validate_positive_float(min_control_ess, "min_control_ess")
    max_propensity_clip_share = _validate_probability(
        max_propensity_clip_share,
        "max_propensity_clip_share",
    )
    max_abs_weighted_smd = _validate_positive_float(
        max_abs_weighted_smd,
        "max_abs_weighted_smd",
    )
    max_top_unit_influence_share = _validate_probability(
        max_top_unit_influence_share,
        "max_top_unit_influence_share",
    )
    max_top_cluster_influence_share = _validate_probability(
        max_top_cluster_influence_share,
        "max_top_cluster_influence_share",
    )
    min_influence_ess = _validate_positive_float(
        min_influence_ess,
        "min_influence_ess",
    )
    max_abs_pretrend_t_stat = _validate_positive_float(
        max_abs_pretrend_t_stat,
        "max_abs_pretrend_t_stat",
    )
    max_simple_cell_weight_share = _validate_probability(
        max_simple_cell_weight_share,
        "max_simple_cell_weight_share",
    )
    min_clusters = _validate_positive_int(min_clusters, "min_clusters")

    diagnostics = dict(estimate.diagnostics or {})
    cell_table = did_post_inference_cell_table(estimate)
    checks: list[dict[str, Any]] = []

    _metadata_alignment_check(data, estimate, checks)

    requested_diag = diagnostics.get("diagnostic_data_requested")
    required_frames = ["unit_level", "overlap", "influence_scores"]
    missing_frames = [
        name
        for name in required_frames
        if _get_diagnostic_frame(estimate, name).empty
    ]
    _append_check(
        checks,
        test="diagnostic_payload_available",
        flag="YELLOW" if requested_diag is False or missing_frames else "GREEN",
        value=missing_frames if missing_frames else "available",
        threshold="unit_level, overlap, influence_scores available",
        message=(
            "Some post-inference diagnostics are unavailable; call estimate(diagnostic_data=True)."
            if requested_diag is False or missing_frames
            else "Diagnostic payload required for post-inference checks is available."
        ),
    )

    post = cell_table.loc[
        cell_table["is_post_treatment"].fillna(False).astype(bool)
    ].copy()
    support = estimate.support.copy()
    requested_post = support.loc[
        support["is_post_treatment"].fillna(False).astype(bool)
    ]
    skipped = estimate.skipped_cells.copy()
    skipped_post = (
        skipped.loc[skipped["is_post_treatment"].fillna(False).astype(bool)]
        if "is_post_treatment" in skipped.columns
        else pd.DataFrame()
    )
    if requested_post.empty:
        skipped_share = None
    else:
        skipped_share = float(len(skipped_post) / len(requested_post))
    if post.empty:
        support_flag = "RED"
        support_message = "No post-treatment ATT(g,t) cells were fitted."
    elif skipped_share is not None and skipped_share > max_skipped_post_cell_share:
        support_flag = "YELLOW"
        support_message = "A material share of requested post-treatment cells was skipped."
    else:
        support_flag = "GREEN"
        support_message = "Fitted post-treatment cell support is within tolerance."
    _append_check(
        checks,
        test="fitted_post_cell_support",
        flag=support_flag,
        value={
            "fitted_post_cells": int(len(post)),
            "skipped_post_cell_share": skipped_share,
        },
        threshold=f"skipped share <= {max_skipped_post_cell_share:.3g}",
        message=support_message,
    )

    statuses = (
        cell_table["diagnostic_status"].astype(str).str.lower()
        if "diagnostic_status" in cell_table.columns
        else pd.Series(dtype=str)
    )
    status_counts = statuses.value_counts().to_dict()
    n_red = int(status_counts.get("red", 0))
    n_yellow = int(status_counts.get("yellow", 0))
    _append_check(
        checks,
        test="cell_warning_flags",
        flag="RED" if n_red else "YELLOW" if n_yellow else "GREEN",
        value=status_counts if status_counts else "not_available",
        threshold="0 red cells, 0 yellow cells",
        message=(
            "At least one fitted cell has red diagnostics."
            if n_red
            else "At least one fitted cell has yellow diagnostics."
            if n_yellow
            else "No fitted cells have model diagnostic warnings."
        ),
    )

    control_ess = (
        _finite_numeric(post["control_weight_ess"])
        if "control_weight_ess" in post.columns
        else pd.Series(dtype=float)
    )
    min_ess = _as_finite_float(control_ess.min()) if not control_ess.empty else None
    _append_check(
        checks,
        test="min_control_weight_ess",
        flag="YELLOW" if min_ess is None or min_ess < min_control_ess else "GREEN",
        value=min_ess,
        threshold=f">= {min_control_ess:.3g}",
        message=(
            "At least one post cell has low effective comparison weight support."
            if min_ess is None or min_ess < min_control_ess
            else "Comparison weight effective sample sizes are within tolerance."
        ),
    )

    clip_share = (
        _finite_numeric(post["propensity_clip_share"])
        if "propensity_clip_share" in post.columns
        else pd.Series(dtype=float)
    )
    max_clip = _as_finite_float(clip_share.max()) if not clip_share.empty else None
    _append_check(
        checks,
        test="max_propensity_clip_share",
        flag=(
            "YELLOW"
            if max_clip is None or max_clip > max_propensity_clip_share
            else "GREEN"
        ),
        value=max_clip,
        threshold=f"<= {max_propensity_clip_share:.3g}",
        message=(
            "At least one post cell has many clipped propensities."
            if max_clip is None or max_clip > max_propensity_clip_share
            else "Propensity clipping is within tolerance."
        ),
    )

    balance = _get_diagnostic_frame(estimate, "balance")
    if not estimate.covariates:
        _append_check(
            checks,
            test="weighted_covariate_balance",
            flag="GREEN",
            value="no_covariates",
            threshold="n/a",
            message="No covariates were supplied; weighted covariate balance is not applicable.",
        )
    elif balance.empty or "smd_weighted" not in balance.columns:
        _append_check(
            checks,
            test="weighted_covariate_balance",
            flag="YELLOW",
            value=None,
            threshold=f"max |weighted SMD| <= {max_abs_weighted_smd:.3g}",
            message="Weighted covariate balance diagnostics are unavailable.",
        )
    else:
        weighted_smd = _finite_numeric(balance["smd_weighted"]).abs()
        max_weighted_smd = (
            _as_finite_float(weighted_smd.max()) if not weighted_smd.empty else None
        )
        _append_check(
            checks,
            test="weighted_covariate_balance",
            flag=(
                "YELLOW"
                if max_weighted_smd is None or max_weighted_smd > max_abs_weighted_smd
                else "GREEN"
            ),
            value=max_weighted_smd,
            threshold=f"max |weighted SMD| <= {max_abs_weighted_smd:.3g}",
            message=(
                "At least one fitted cell remains imbalanced after CS weighting."
                if max_weighted_smd is None or max_weighted_smd > max_abs_weighted_smd
                else "Weighted covariate balance is within tolerance."
            ),
        )

    try:
        influence = did_influence_table(estimate)
    except ValueError as exc:
        _append_check(
            checks,
            test="unit_influence_concentration",
            flag="YELLOW",
            value=None,
            threshold=(
                f"top unit share <= {max_top_unit_influence_share:.3g}; "
                f"influence ESS >= {min_influence_ess:.3g}"
            ),
            message=str(exc),
        )
    else:
        top_share = _as_finite_float(influence["abs_influence_share"].max())
        influence_ess = _influence_ess(influence["abs_influence_score"])
        influence_flag = (
            "YELLOW"
            if top_share is None
            or influence_ess is None
            or top_share > max_top_unit_influence_share
            or influence_ess < min_influence_ess
            else "GREEN"
        )
        _append_check(
            checks,
            test="unit_influence_concentration",
            flag=influence_flag,
            value={"top_unit_share": top_share, "influence_ess": influence_ess},
            threshold=(
                f"top unit share <= {max_top_unit_influence_share:.3g}; "
                f"influence ESS >= {min_influence_ess:.3g}"
            ),
            message=(
                "The simple ATT is concentrated in too few unit-level influence contributions."
                if influence_flag != "GREEN"
                else "Unit-level influence concentration is within tolerance."
            ),
        )

    if estimate.cluster_col is None:
        _append_check(
            checks,
            test="cluster_influence_concentration",
            flag="GREEN",
            value="not_clustered",
            threshold="n/a",
            message="No cluster_col is set; cluster influence concentration is not applicable.",
        )
    elif data is None:
        _append_check(
            checks,
            test="cluster_influence_concentration",
            flag="YELLOW",
            value="not_checked",
            threshold="PanelDataDID supplied",
            message="PanelDataDID was not supplied; cluster influence concentration was not checked.",
        )
    else:
        try:
            cluster_influence = did_cluster_influence_table(data, estimate)
        except ValueError as exc:
            message = str(exc)
            missing_payload = "influence_scores" in message or "influence-score" in message
            _append_check(
                checks,
                test="cluster_influence_concentration",
                flag="YELLOW" if missing_payload else "RED",
                value=None,
                threshold=(
                    "diagnostic payload available"
                    if missing_payload
                    else "stable clusters"
                ),
                message=message,
            )
        else:
            top_cluster_share = _as_finite_float(
                cluster_influence["abs_influence_share"].max()
            )
            n_clusters = int(len(cluster_influence))
            cluster_flag = (
                "RED"
                if n_clusters < min_clusters
                else "YELLOW"
                if top_cluster_share is None
                or top_cluster_share > max_top_cluster_influence_share
                else "GREEN"
            )
            _append_check(
                checks,
                test="cluster_influence_concentration",
                flag=cluster_flag,
                value={
                    "n_clusters": n_clusters,
                    "top_cluster_share": top_cluster_share,
                },
                threshold=(
                    f"clusters >= {min_clusters}; "
                    f"top cluster share <= {max_top_cluster_influence_share:.3g}"
                ),
                message=(
                    "Clustered inference is fragile because too few clusters or one dominant cluster drive the score."
                    if cluster_flag != "GREEN"
                    else "Cluster influence concentration is within tolerance."
                ),
            )

    pre = cell_table.loc[
        ~cell_table["is_post_treatment"].fillna(False).astype(bool)
    ].copy()
    if pre.empty:
        _append_check(
            checks,
            test="fitted_pre_period_placebo",
            flag="YELLOW",
            value=None,
            threshold=f"max |t| <= {max_abs_pretrend_t_stat:.3g}",
            message="No fitted pre-period placebo cells are present; refit with include_pre_periods=True to check this.",
        )
    else:
        pre_abs_t = _finite_numeric(pre["abs_t_stat"]) if "abs_t_stat" in pre.columns else pd.Series(dtype=float)
        max_pre_t = _as_finite_float(pre_abs_t.max()) if not pre_abs_t.empty else None
        _append_check(
            checks,
            test="fitted_pre_period_placebo",
            flag=(
                "YELLOW"
                if max_pre_t is None or max_pre_t > max_abs_pretrend_t_stat
                else "GREEN"
            ),
            value=max_pre_t,
            threshold=f"max |t| <= {max_abs_pretrend_t_stat:.3g}",
            message=(
                "Fitted pre-period placebo cells show a large standardized deviation."
                if max_pre_t is None or max_pre_t > max_abs_pretrend_t_stat
                else "Fitted pre-period placebo cells are within the standardized threshold."
            ),
        )

    simple_weights = _simple_weight_rows(estimate)
    if simple_weights.empty:
        _append_check(
            checks,
            test="simple_aggregate_cell_weight",
            flag="YELLOW",
            value=None,
            threshold=f"max cell weight <= {max_simple_cell_weight_share:.3g}",
            message="Simple aggregate cell weights are unavailable.",
        )
    else:
        max_weight = _as_finite_float(simple_weights["weight"].abs().max())
        _append_check(
            checks,
            test="simple_aggregate_cell_weight",
            flag=(
                "YELLOW"
                if max_weight is None or max_weight > max_simple_cell_weight_share
                else "GREEN"
            ),
            value=max_weight,
            threshold=f"max cell weight <= {max_simple_cell_weight_share:.3g}",
            message=(
                "The simple aggregate ATT is dominated by one fitted cell."
                if max_weight is None or max_weight > max_simple_cell_weight_share
                else "Simple aggregate cell weights are not overly concentrated."
            ),
        )

    simple = estimate.aggregates["simple"].iloc[0]
    se = _as_finite_float(simple.get("se"))
    ci_lower = _as_finite_float(simple.get("ci_lower"))
    ci_upper = _as_finite_float(simple.get("ci_upper"))
    p_value = _as_finite_float(simple.get("p_value"))
    finite_inference = (
        se is not None
        and se >= 0.0
        and ci_lower is not None
        and ci_upper is not None
        and ci_lower <= ci_upper
        and p_value is not None
        and 0.0 <= p_value <= 1.0
    )
    _append_check(
        checks,
        test="simple_aggregate_inference_finite",
        flag="GREEN" if finite_inference else "RED",
        value={"se": se, "ci_lower": ci_lower, "ci_upper": ci_upper, "p_value": p_value},
        threshold="finite SE, ordered CI, p-value in [0, 1]",
        message=(
            "Simple aggregate inference outputs are finite and internally consistent."
            if finite_inference
            else "Simple aggregate inference outputs are not internally consistent."
        ),
    )

    bootstrap_replications = int(diagnostics.get("bootstrap_replications") or 0)
    has_bootstrap = bootstrap_replications > 0
    bootstrap_flag = (
        "GREEN"
        if has_bootstrap or not require_multiplier_bootstrap
        else "YELLOW"
    )
    _append_check(
        checks,
        test="multiplier_bootstrap_inference",
        flag=bootstrap_flag,
        value=bootstrap_replications,
        threshold=(
            "> 0 replications"
            if require_multiplier_bootstrap
            else "optional"
        ),
        message=(
            "Multiplier bootstrap inference is available."
            if has_bootstrap
            else "Asymptotic influence inference was used; set require_multiplier_bootstrap=True to enforce bootstrap robustness."
            if not require_multiplier_bootstrap
            else "Multiplier bootstrap inference was required but not used."
        ),
    )

    worst = (
        "RED"
        if any(row["flag"] == "RED" for row in checks)
        else "YELLOW"
        if any(row["flag"] == "YELLOW" for row in checks)
        else "GREEN"
    )
    if worst == "GREEN":
        message = "I can rely on the results: post-inference diagnostics did not find material fragility."
    elif worst == "YELLOW":
        message = "I would treat the results as directionally useful, but not fully robust until yellow diagnostics are addressed."
    else:
        message = "I would not rely on the results yet: at least one red post-inference diagnostic failed."
    overall = {
        "test": "overall_inference_reliability",
        "flag": worst,
        "value": worst.lower(),
        "threshold": "all checks GREEN for robust reliance",
        "message": message,
    }
    return pd.DataFrame([overall, *checks], columns=_CHECK_COLUMNS)


def run_did_inference_diagnostics(*args, **kwargs) -> pd.DataFrame:
    """Alias for :func:`run_did_post_inference_diagnostics`."""

    return run_did_post_inference_diagnostics(*args, **kwargs)


__all__ = [
    "did_post_inference_cell_table",
    "did_influence_table",
    "did_cluster_influence_table",
    "run_did_post_inference_diagnostics",
    "run_did_inference_diagnostics",
]
