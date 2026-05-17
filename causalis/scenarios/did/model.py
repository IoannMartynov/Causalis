from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Hashable, Literal, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import norm

from causalis.data_contracts.panel_did_estimate import CallawaySantAnnaDIDEstimate
from causalis.data_contracts.panel_data_did import ComparisonGroup, PanelDataDID


Estimator = Literal["dr", "aipw", "ipw"]
AggregateKind = Literal["simple", "cohort", "calendar", "event"]
BasePeriod = Literal["universal", "varying"]

_MODEL_NAME = "CallawaySantAnnaStaggeredDID"
_CONTROL_GROUPS = {"never_treated", "not_yet_treated", "not_yet_or_never"}
_ESTIMATORS = {"dr", "aipw", "ipw"}
_DEFAULT_PROPENSITY_CLIP = 1e-6
_DEFAULT_LOGIT_RIDGE = 1e-8
_DEFAULT_OPTIMIZER_TOL = 1e-8
_DEFAULT_OPTIMIZER_MAXITER = 1000
_DEFAULT_MAX_CONDITION_NUMBER = 1e8


@dataclass(frozen=True)
class _PreparedPanel:
    data: PanelDataDID
    df: pd.DataFrame
    unit_ids: list[Hashable]
    times: list[pd.Period]
    time_index: dict[pd.Period, int]
    outcome: pd.DataFrame
    clusters: Optional[np.ndarray]


@dataclass(frozen=True)
class _CellFit:
    row: dict[str, Any]
    influence_scores: np.ndarray
    unit_diagnostics: pd.DataFrame
    overlap: dict[str, Any]
    balance: pd.DataFrame


def _validate_alpha(alpha: float) -> float:
    value = float(alpha)
    if not np.isfinite(value) or not (0.0 < value < 1.0):
        raise ValueError("alpha must be finite and in (0, 1).")
    return value


def _validate_control_group(control_group: str) -> ComparisonGroup:
    if not isinstance(control_group, str):
        raise ValueError(
            "control_group must be one of 'never_treated', 'not_yet_treated', or 'not_yet_or_never'."
        )
    if control_group not in _CONTROL_GROUPS:
        raise ValueError(
            "control_group must be one of 'never_treated', 'not_yet_treated', or 'not_yet_or_never'."
        )
    return control_group  # type: ignore[return-value]


def _validate_estimator(estimator: str) -> Estimator:
    if not isinstance(estimator, str):
        raise ValueError("estimator must be one of 'dr', 'aipw', or 'ipw'.")
    normalized = estimator.lower()
    if normalized not in _ESTIMATORS:
        raise ValueError(
            "estimator must be one of 'dr', 'aipw', or 'ipw'. The OR point estimator is disabled "
            "until nuisance-estimation inference is implemented."
        )
    return normalized  # type: ignore[return-value]


def _validate_base_period(base_period: str) -> BasePeriod:
    if base_period not in {"universal", "varying"}:
        raise ValueError("base_period must be either 'universal' or 'varying'.")
    return base_period  # type: ignore[return-value]


def _validate_nonnegative_int(value: int, name: str) -> int:
    out = int(value)
    if out < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return out


def _validate_positive_int(value: int, name: str) -> int:
    out = int(value)
    if out <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return out


def _validate_positive_float(value: float, name: str) -> float:
    out = float(value)
    if not np.isfinite(out) or out <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return out


def _validate_nonnegative_float(value: float, name: str) -> float:
    out = float(value)
    if not np.isfinite(out) or out < 0.0:
        raise ValueError(f"{name} must be non-negative and finite.")
    return out


def _validate_propensity_clip(propensity_clip: float) -> float:
    value = float(propensity_clip)
    if not np.isfinite(value) or not (0.0 < value < 0.5):
        raise ValueError("propensity_clip must be finite and in (0, 0.5).")
    return value


def _normal_p_value(estimate: float, se: float) -> float:
    if se > 0.0:
        return float(2.0 * norm.sf(abs(estimate / se)))
    return 1.0 if abs(estimate) <= 1e-16 else float("nan")


def _effective_sample_size(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    denom = float(np.sum(weights**2))
    if denom <= 0.0:
        return float("nan")
    return float(np.sum(weights) ** 2 / denom)


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    denom = float(np.sum(weights))
    if denom <= 0.0:
        return float("nan")
    return float(np.sum(values * weights) / denom)


def _smd(numerator: float, treated_values: np.ndarray, control_values: np.ndarray) -> float:
    treated_var = float(np.var(treated_values, ddof=1)) if treated_values.size > 1 else 0.0
    control_var = float(np.var(control_values, ddof=1)) if control_values.size > 1 else 0.0
    pooled = float(np.sqrt(0.5 * (treated_var + control_var)))
    if pooled == 0.0:
        return 0.0 if abs(numerator) <= 1e-16 else float("inf")
    return float(numerator / pooled)


def _resolve_unit_clusters(data: PanelDataDID, df: pd.DataFrame, unit_ids: list[Hashable]) -> Optional[np.ndarray]:
    cluster_col = data.cluster_col
    if cluster_col is None:
        return None
    if cluster_col == data.unit_col:
        return np.asarray(unit_ids, dtype=object)

    cluster_counts = df.groupby(data.unit_col, sort=False)[cluster_col].nunique(dropna=False)
    unstable = cluster_counts[cluster_counts > 1]
    if not unstable.empty:
        raise ValueError(
            "cluster_col must be stable within unit for unit-level CSA influence-function inference; "
            f"unstable units: {list(unstable.index)}"
        )

    clusters = (
        df[[data.unit_col, cluster_col]]
        .drop_duplicates(subset=[data.unit_col])
        .set_index(data.unit_col)
        .loc[unit_ids, cluster_col]
        .to_numpy(dtype=object)
    )
    return clusters


def _prepare_panel(data: PanelDataDID) -> _PreparedPanel:
    if not isinstance(data, PanelDataDID):
        raise ValueError("Input must be a PanelDataDID object.")

    df = data.df_analysis()
    unit_ids = df[data.unit_col].drop_duplicates().tolist()
    times = list(data.analysis_times())
    outcome = (
        df.pivot(index=data.unit_col, columns=data.time_col, values=data.y)
        .reindex(index=unit_ids, columns=times)
        .astype(float)
    )
    clusters = _resolve_unit_clusters(data, df, unit_ids)
    return _PreparedPanel(
        data=data,
        df=df,
        unit_ids=unit_ids,
        times=times,
        time_index=data.time_to_index(),
        outcome=outcome,
        clusters=clusters,
    )


def _safe_observed_units(
    prepared: _PreparedPanel,
    candidates: set[Hashable],
    *,
    base_time: pd.Period,
    target_time: pd.Period,
) -> list[Hashable]:
    ordered = [unit for unit in prepared.unit_ids if unit in candidates]
    if not ordered:
        return []

    y = prepared.outcome.loc[ordered, [base_time, target_time]]
    complete = y.notna().all(axis=1)
    return y.index[complete].tolist()


def _comparison_units_with_anticipation(
    prepared: _PreparedPanel,
    target_time: pd.Period,
    *,
    control_group: ComparisonGroup,
    anticipation: int,
) -> set[Hashable]:
    target_idx = prepared.time_index[target_time] + anticipation
    first_by_unit = prepared.data.first_treatment_by_unit
    out: set[Hashable] = set()
    for unit in prepared.unit_ids:
        first_treatment = first_by_unit[unit]
        if control_group == "never_treated":
            include = first_treatment is None
        elif control_group == "not_yet_treated":
            include = first_treatment is not None and prepared.time_index[first_treatment] > target_idx
        else:
            include = first_treatment is None or prepared.time_index[first_treatment] > target_idx
        if include:
            out.add(unit)
    return out


def _design_at_base(
    prepared: _PreparedPanel,
    units: list[Hashable],
    base_time: pd.Period,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    covariates = list(prepared.data.covariates)
    intercept = np.ones((len(units), 1), dtype=float)
    if not covariates:
        return intercept, None

    base_rows = (
        prepared.df.loc[prepared.df[prepared.data.time_col] == base_time, [prepared.data.unit_col, *covariates]]
        .set_index(prepared.data.unit_col)
        .reindex(units)
    )
    x = base_rows[covariates].to_numpy(dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("Pre-treatment covariates must be observed and finite in each ATT(g,t) base period.")
    return np.column_stack([intercept, x]), x


def _fit_logistic_propensity(
    design: np.ndarray,
    d: np.ndarray,
    *,
    clip: float,
    ridge: float,
    tol: float,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray]:
    q = float(np.mean(d))
    if not (0.0 < q < 1.0):
        raise ValueError("Each ATT(g,t) cell must contain treated and comparison units.")

    if design.shape[1] == 1:
        gamma = np.asarray([logit(np.clip(q, clip, 1.0 - clip))], dtype=float)
        return gamma, np.full(design.shape[0], np.clip(q, clip, 1.0 - clip), dtype=float)

    def objective(gamma: np.ndarray) -> float:
        eta = design @ gamma
        penalty = 0.5 * ridge * float(np.sum(gamma[1:] ** 2))
        return float(np.mean(np.logaddexp(0.0, eta) - d * eta) + penalty)

    def gradient(gamma: np.ndarray) -> np.ndarray:
        eta = design @ gamma
        grad = design.T @ (expit(eta) - d) / design.shape[0]
        grad[1:] += ridge * gamma[1:]
        return grad

    result = minimize(
        objective,
        np.zeros(design.shape[1], dtype=float),
        jac=gradient,
        method="BFGS",
        options={"gtol": tol, "maxiter": max_iter},
    )
    grad_norm = float(np.linalg.norm(gradient(np.asarray(result.x, dtype=float)), ord=np.inf))
    if (not result.success) and grad_norm > max(1e-6, 10.0 * tol):
        raise RuntimeError(f"Propensity optimization failed to converge: {result.message}")

    gamma = np.asarray(result.x, dtype=float)
    if not np.isfinite(gamma).all():
        raise RuntimeError("Propensity optimization produced non-finite coefficients.")
    propensity = np.clip(expit(design @ gamma), clip, 1.0 - clip)
    return gamma, propensity


def _fit_outcome_regression(design: np.ndarray, delta_y: np.ndarray, control: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not bool(control.any()):
        raise ValueError("Outcome regression requires at least one comparison unit.")
    beta, *_ = np.linalg.lstsq(design[control], delta_y[control], rcond=None)
    beta = np.asarray(beta, dtype=float)
    if not np.isfinite(beta).all():
        raise RuntimeError("Outcome regression produced non-finite coefficients.")
    return beta, design @ beta


def _variance_from_scores(scores: np.ndarray, clusters: Optional[np.ndarray]) -> float:
    n = scores.size
    if clusters is None:
        return float(np.sum(scores**2) / (n**2))

    cluster_scores = (
        pd.DataFrame({"cluster": clusters, "score": scores})
        .groupby("cluster", sort=False, observed=True)["score"]
        .sum()
    )
    n_clusters = int(cluster_scores.shape[0])
    if n_clusters < 2:
        raise ValueError("Clustered inference requires at least two clusters.")
    centered = cluster_scores.to_numpy(dtype=float) - float(cluster_scores.mean())
    return float(n_clusters / (n_clusters - 1.0) * np.sum(centered**2) / (n**2))


def _draw_multiplier_weights(
    *,
    n_units: int,
    clusters: Optional[np.ndarray],
    replications: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if clusters is None:
        return rng.choice(np.asarray([-1.0, 1.0]), size=(replications, n_units))

    cluster_index = pd.Index(pd.Series(clusters).drop_duplicates().tolist())
    unit_cluster_pos = cluster_index.get_indexer(clusters)
    cluster_multipliers = rng.choice(np.asarray([-1.0, 1.0]), size=(replications, len(cluster_index)))
    return cluster_multipliers[:, unit_cluster_pos]


def _multiplier_shifts(scores: np.ndarray, multiplier_weights: np.ndarray) -> np.ndarray:
    if scores.ndim == 1:
        scores = scores[:, None]
    return multiplier_weights @ scores / scores.shape[0]


def _add_inference(
    table: pd.DataFrame,
    scores: np.ndarray,
    *,
    estimate_col: str,
    clusters: Optional[np.ndarray],
    alpha: float,
    bootstrap_replications: int,
    rng: np.random.Generator,
    multiplier_weights: Optional[np.ndarray] = None,
    simultaneous: bool = False,
) -> pd.DataFrame:
    out = table.copy()
    if scores.ndim == 1:
        scores = scores[:, None]
    if scores.shape[1] != len(out):
        raise ValueError("Influence-score columns must match table rows.")

    if bootstrap_replications > 0:
        if multiplier_weights is None:
            multiplier_weights = _draw_multiplier_weights(
                n_units=scores.shape[0],
                clusters=clusters,
                replications=bootstrap_replications,
                rng=rng,
            )
        shifts = _multiplier_shifts(scores, multiplier_weights)
        se = shifts.std(axis=0, ddof=1)
        inference = "clustered_multiplier_bootstrap" if clusters is not None else "multiplier_bootstrap"
    else:
        se = np.asarray([np.sqrt(max(_variance_from_scores(scores[:, idx], clusters), 0.0)) for idx in range(scores.shape[1])])
        inference = "clustered_influence" if clusters is not None else "influence"

    z = float(norm.ppf(1.0 - alpha / 2.0))
    estimates = out[estimate_col].to_numpy(dtype=float)
    out["se"] = se
    out["ci_lower"] = estimates - z * se
    out["ci_upper"] = estimates + z * se
    out["p_value"] = [_normal_p_value(est, std) for est, std in zip(estimates, se)]
    out["alpha"] = alpha
    out["is_significant"] = out["p_value"] < alpha
    out["inference"] = inference
    if bootstrap_replications > 0 and simultaneous:
        denom = np.where(se > 0.0, se, np.nan)
        max_abs_t = np.nanmax(np.abs(shifts / denom[None, :]), axis=1)
        critical_value = float(np.nanquantile(max_abs_t, 1.0 - alpha))
        out["simultaneous_critical_value"] = critical_value
        out["sim_ci_lower"] = estimates - critical_value * se
        out["sim_ci_upper"] = estimates + critical_value * se
    return out


def _overlap_summary(
    *,
    cell_id: int,
    cohort: pd.Period,
    time: pd.Period,
    event_time: int,
    propensity: np.ndarray,
    d: np.ndarray,
    control_weights: np.ndarray,
    treated_weights: np.ndarray,
) -> dict[str, Any]:
    treated = d == 1.0
    control = d == 0.0
    return {
        "cell_id": cell_id,
        "cohort": cohort,
        "time": time,
        "event_time": event_time,
        "min": float(np.min(propensity)),
        "max": float(np.max(propensity)),
        "mean": float(np.mean(propensity)),
        "treated_min": float(np.min(propensity[treated])),
        "treated_max": float(np.max(propensity[treated])),
        "control_min": float(np.min(propensity[control])),
        "control_max": float(np.max(propensity[control])),
        "treated_weight_ess": _effective_sample_size(treated_weights[treated]),
        "control_weight_ess": _effective_sample_size(control_weights[control]),
    }


def _balance_table(
    *,
    cell_id: int,
    cohort: pd.Period,
    time: pd.Period,
    event_time: int,
    x: Optional[np.ndarray],
    covariate_names: list[str],
    d: np.ndarray,
    control_weights: np.ndarray,
) -> pd.DataFrame:
    columns = [
        "cell_id",
        "cohort",
        "time",
        "event_time",
        "covariate",
        "treated_mean",
        "control_mean",
        "weighted_control_mean",
        "smd_unweighted",
        "smd_weighted",
    ]
    if x is None or not covariate_names:
        return pd.DataFrame(columns=columns)

    treated = d == 1.0
    control = d == 0.0
    rows: list[dict[str, Any]] = []
    for idx, covariate in enumerate(covariate_names):
        values = x[:, idx]
        treated_values = values[treated]
        control_values = values[control]
        treated_mean = float(np.mean(treated_values))
        control_mean = float(np.mean(control_values))
        weighted_control_mean = _weighted_mean(values[control], control_weights[control])
        rows.append(
            {
                "cell_id": cell_id,
                "cohort": cohort,
                "time": time,
                "event_time": event_time,
                "covariate": covariate,
                "treated_mean": treated_mean,
                "control_mean": control_mean,
                "weighted_control_mean": weighted_control_mean,
                "smd_unweighted": _smd(treated_mean - control_mean, treated_values, control_values),
                "smd_weighted": _smd(treated_mean - weighted_control_mean, treated_values, control_values),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _fit_cell(
    prepared: _PreparedPanel,
    *,
    cell_id: int,
    cohort: pd.Period,
    target_time: pd.Period,
    base_time: pd.Period,
    event_time: int,
    is_post_treatment: bool,
    n_treated_available: int,
    n_control_available: int,
    control_group: ComparisonGroup,
    estimator: Estimator,
    anticipation: int,
    propensity_clip: float,
    logit_ridge: float,
    optimizer_tol: float,
    optimizer_maxiter: int,
    min_treated_per_cell: int,
    min_control_per_cell: int,
    min_control_ess: float,
    max_propensity_clip_share: float,
    max_condition_number: float,
) -> Optional[_CellFit]:
    data = prepared.data
    cohort_candidates = set(data.cohort_units(cohort))
    comparison_candidates = _comparison_units_with_anticipation(
        prepared,
        target_time,
        control_group=control_group,
        anticipation=anticipation,
    )

    treated_units = _safe_observed_units(
        prepared,
        cohort_candidates,
        base_time=base_time,
        target_time=target_time,
    )
    control_units = _safe_observed_units(
        prepared,
        comparison_candidates,
        base_time=base_time,
        target_time=target_time,
    )
    if not treated_units or not control_units:
        return None

    sample_units = treated_units + control_units
    d = np.concatenate([np.ones(len(treated_units)), np.zeros(len(control_units))]).astype(float)
    y_post = prepared.outcome.loc[sample_units, target_time].to_numpy(dtype=float)
    y_pre = prepared.outcome.loc[sample_units, base_time].to_numpy(dtype=float)
    delta_y = y_post - y_pre
    if not np.isfinite(delta_y).all():
        raise ValueError("Outcome differences must be finite.")

    design, x = _design_at_base(prepared, sample_units, base_time)
    control = d == 0.0
    gamma_hat, propensity = _fit_logistic_propensity(
        design,
        d,
        clip=propensity_clip,
        ridge=logit_ridge,
        tol=optimizer_tol,
        max_iter=optimizer_maxiter,
    )
    odds_control = propensity * (1.0 - d) / (1.0 - propensity)
    odds_mean = float(np.mean(odds_control))
    if not np.isfinite(odds_mean) or odds_mean <= 0.0:
        raise ValueError("Normalized control propensity weights have zero or non-finite mean.")

    q = float(np.mean(d))
    treated_weights = d / q
    control_weights = odds_control / odds_mean

    if estimator in {"dr", "aipw"}:
        beta_hat, outcome_hat = _fit_outcome_regression(design, delta_y, control)
    else:
        beta_hat = np.zeros(design.shape[1], dtype=float)
        outcome_hat = np.zeros(delta_y.shape[0], dtype=float)

    residualized_delta = delta_y - outcome_hat
    att = float(np.mean((treated_weights - control_weights) * residualized_delta))
    raw_scores = (treated_weights - control_weights) * residualized_delta - treated_weights * att
    used_control_weights = control_weights

    control_design = design[control]
    control_rank = int(np.linalg.matrix_rank(control_design))
    n_parameters = int(design.shape[1])
    condition_number = float(np.linalg.cond(control_design)) if control_design.shape[0] else float("inf")
    treated_ess = _effective_sample_size(treated_weights[d == 1.0])
    control_ess = _effective_sample_size(control_weights[control])
    clip_share = float(np.mean((propensity <= propensity_clip) | (propensity >= 1.0 - propensity_clip)))
    flags = []
    if len(treated_units) < min_treated_per_cell:
        flags.append("low_treated_count")
    if len(control_units) < min_control_per_cell:
        flags.append("low_control_count")
    if control_ess < min_control_ess:
        flags.append("low_control_ess")
    if clip_share > max_propensity_clip_share:
        flags.append("high_propensity_clip_share")
    if control_rank < n_parameters:
        flags.append("rank_deficient_control_design")
    if condition_number > max_condition_number:
        flags.append("ill_conditioned_control_design")
    diagnostic_status = "red" if any(flag in flags for flag in {"rank_deficient_control_design"}) else "yellow" if flags else "green"

    n_total = len(prepared.unit_ids)
    n_cell = len(sample_units)
    unit_pos = {unit: idx for idx, unit in enumerate(prepared.unit_ids)}
    full_scores = np.zeros(n_total, dtype=float)
    for local_idx, unit in enumerate(sample_units):
        full_scores[unit_pos[unit]] = raw_scores[local_idx] * n_total / n_cell

    unit_diag = pd.DataFrame(
        {
            "cell_id": cell_id,
            "cohort": cohort,
            "time": target_time,
            "base_time": base_time,
            "event_time": event_time,
            "is_post_treatment": is_post_treatment,
            data.unit_col: sample_units,
            "is_treated_cohort": d.astype(int),
            "delta_y": delta_y,
            "propensity_score": propensity,
            "outcome_regression": outcome_hat,
            "treated_weight": treated_weights,
            "control_weight": used_control_weights,
            "influence_score": raw_scores,
        }
    )
    for idx, covariate in enumerate(data.covariates):
        if x is not None:
            unit_diag[covariate] = x[:, idx]

    row = {
        "cell_id": cell_id,
        "cohort": cohort,
        "time": target_time,
        "base_time": base_time,
        "event_time": event_time,
        "is_post_treatment": is_post_treatment,
        "att": att,
        "n_units": n_cell,
        "n_treated_available": n_treated_available,
        "n_treated_complete": len(treated_units),
        "n_control_available": n_control_available,
        "n_control_complete": len(control_units),
        "n_treated": len(treated_units),
        "n_control": len(control_units),
        "control_group": control_group,
        "estimator": estimator,
        "control_design_rank": control_rank,
        "n_parameters": n_parameters,
        "condition_number": condition_number,
        "treated_weight_ess": treated_ess,
        "control_weight_ess": control_ess,
        "propensity_clip_share": clip_share,
        "diagnostic_status": diagnostic_status,
        "diagnostic_flags": ",".join(flags),
    }
    overlap = _overlap_summary(
        cell_id=cell_id,
        cohort=cohort,
        time=target_time,
        event_time=event_time,
        propensity=propensity,
        d=d,
        control_weights=control_weights,
        treated_weights=treated_weights,
    )
    balance = _balance_table(
        cell_id=cell_id,
        cohort=cohort,
        time=target_time,
        event_time=event_time,
        x=x,
        covariate_names=list(data.covariates),
        d=d,
        control_weights=control_weights,
    )
    return _CellFit(
        row=row,
        influence_scores=full_scores,
        unit_diagnostics=unit_diag,
        overlap=overlap,
        balance=balance,
    )


def _fit_att_gt(
    prepared: _PreparedPanel,
    *,
    control_group: ComparisonGroup,
    estimator: Estimator,
    anticipation: int,
    base_period: BasePeriod,
    include_pre_periods: bool,
    propensity_clip: float,
    logit_ridge: float,
    optimizer_tol: float,
    optimizer_maxiter: int,
    min_treated_per_cell: int,
    min_control_per_cell: int,
    min_control_ess: float,
    max_propensity_clip_share: float,
    max_condition_number: float,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    scores: list[np.ndarray] = []
    unit_diag_parts: list[pd.DataFrame] = []
    overlap_rows: list[dict[str, Any]] = []
    balance_parts: list[pd.DataFrame] = []
    skipped_rows: list[dict[str, Any]] = []

    cell_id = 0
    support = prepared.data.att_gt_cells(
        control_group=control_group,
        anticipation=anticipation,
        base_period=base_period,
        include_pre_periods=include_pre_periods,
        include_unsupported=True,
    )
    for support_row in support.to_dict("records"):
        if not bool(support_row["is_supported"]):
            skipped_rows.append(support_row)
            continue
        cohort = support_row["cohort"]
        target_time = support_row["time"]
        base_time = support_row["base_time"]
        event_time = int(support_row["event_time"])
        try:
            fitted = _fit_cell(
                prepared,
                cell_id=cell_id,
                cohort=cohort,
                target_time=target_time,
                base_time=base_time,
                event_time=event_time,
                is_post_treatment=bool(support_row["is_post_treatment"]),
                n_treated_available=int(support_row["n_treated_available"]),
                n_control_available=int(support_row["n_control_available"]),
                control_group=control_group,
                estimator=estimator,
                anticipation=anticipation,
                propensity_clip=propensity_clip,
                logit_ridge=logit_ridge,
                optimizer_tol=optimizer_tol,
                optimizer_maxiter=optimizer_maxiter,
                min_treated_per_cell=min_treated_per_cell,
                min_control_per_cell=min_control_per_cell,
                min_control_ess=min_control_ess,
                max_propensity_clip_share=max_propensity_clip_share,
                max_condition_number=max_condition_number,
            )
        except Exception as exc:
            failed = dict(support_row)
            failed["is_supported"] = False
            failed["unsupported_reason"] = f"estimation_failed: {exc}"
            skipped_rows.append(failed)
            continue
        if fitted is None:
            skipped = dict(support_row)
            skipped["is_supported"] = False
            skipped["unsupported_reason"] = "missing_treated_or_control_complete_cases"
            skipped_rows.append(skipped)
            continue
        rows.append(fitted.row)
        scores.append(fitted.influence_scores)
        unit_diag_parts.append(fitted.unit_diagnostics)
        overlap_rows.append(fitted.overlap)
        if not fitted.balance.empty:
            balance_parts.append(fitted.balance)
        cell_id += 1

    if not rows:
        raise ValueError(
            "No estimable Callaway-Sant'Anna ATT(g,t) cells found for the requested "
            f"control_group={control_group!r}, estimator={estimator!r}, anticipation={anticipation}, "
            f"base_period={base_period!r}, and include_pre_periods={include_pre_periods!r}."
        )

    att_gt = pd.DataFrame(rows).sort_values(["cohort", "time"]).reset_index(drop=True)
    score_matrix = np.column_stack(scores)
    unit_diag = pd.concat(unit_diag_parts, ignore_index=True)
    overlap = pd.DataFrame(overlap_rows)
    balance = (
        pd.concat(balance_parts, ignore_index=True)
        if balance_parts
        else pd.DataFrame(
            columns=[
                "cell_id",
                "cohort",
                "time",
                "event_time",
                "covariate",
                "treated_mean",
                "control_mean",
                "weighted_control_mean",
                "smd_unweighted",
                "smd_weighted",
            ]
        )
    )
    order = att_gt["cell_id"].to_numpy(dtype=int)
    score_matrix = score_matrix[:, order]
    skipped = pd.DataFrame(skipped_rows, columns=support.columns)
    return att_gt, score_matrix, unit_diag, overlap, balance, support, skipped


def _aggregate_scores(
    att_gt: pd.DataFrame,
    score_matrix: np.ndarray,
    *,
    kind: AggregateKind,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    source = att_gt if kind == "event" else att_gt[att_gt["is_post_treatment"].astype(bool)]
    if source.empty:
        raise ValueError(f"No eligible ATT(g,t) cells are available for {kind!r} aggregation.")
    rows: list[dict[str, Any]] = []
    score_columns: list[np.ndarray] = []
    weight_rows: list[dict[str, Any]] = []

    def append_row(
        *,
        label: dict[str, Any],
        subset: pd.DataFrame,
        weights: np.ndarray,
    ) -> None:
        idx = subset.index.to_numpy(dtype=int)
        estimate = float(np.sum(weights * subset["att"].to_numpy(dtype=float)))
        agg_scores = score_matrix[:, idx] @ weights
        row = {
            "aggregate": kind,
            **label,
            "estimate": estimate,
            "n_cells": len(subset),
            "total_treated_weight": float(np.sum(subset["n_treated"].to_numpy(dtype=float))),
        }
        rows.append(row)
        score_columns.append(agg_scores)
        for cell_id, weight in zip(subset["cell_id"].tolist(), weights):
            weight_rows.append({"aggregate": kind, **label, "cell_id": cell_id, "weight": float(weight)})

    if kind == "simple":
        weights = source["n_treated"].to_numpy(dtype=float)
        weights = weights / float(np.sum(weights))
        append_row(label={}, subset=source, weights=weights)
    elif kind == "cohort":
        for cohort, subset in source.groupby("cohort", sort=True, observed=True):
            weights = np.full(len(subset), 1.0 / len(subset), dtype=float)
            append_row(label={"cohort": cohort}, subset=subset, weights=weights)
    elif kind == "calendar":
        for time, subset in source.groupby("time", sort=True, observed=True):
            weights = subset["n_treated"].to_numpy(dtype=float)
            weights = weights / float(np.sum(weights))
            append_row(label={"time": time}, subset=subset, weights=weights)
    elif kind == "event":
        for event_time, subset in source.groupby("event_time", sort=True, observed=True):
            weights = subset["n_treated"].to_numpy(dtype=float)
            weights = weights / float(np.sum(weights))
            append_row(label={"event_time": int(event_time)}, subset=subset, weights=weights)
    else:
        raise ValueError("kind must be one of 'simple', 'cohort', 'calendar', or 'event'.")

    table = pd.DataFrame(rows)
    scores = np.column_stack(score_columns)
    weights = pd.DataFrame(weight_rows)
    return table, scores, weights


class CallawaySantAnnaDID:
    r"""Callaway-Sant'Anna staggered-adoption DID estimator.

    The model estimates group-time average treatment effects :math:`ATT(g,t)` for
    cohorts (groups) first treated in period :math:`g` and calendar times :math:`t`.
    Estimation is performed using the doubly robust or inverse-probability weighting
    methods proposed by Callaway and Sant'Anna (2021).

    Parameters
    ----------
    estimator : {"dr", "ipw"}, default "dr"
        The estimator to use for :math:`ATT(g,t)` cells.
        - "dr": Doubly robust AIPW-style estimator.
        - "ipw": Inverse probability weighting estimator.
    control_group : {"not_yet_or_never", "never_treated"}, default "not_yet_or_never"
        Which units to use as the comparison group.
        - "not_yet_or_never": Includes units not yet treated by time :math:`t` and never-treated units.
        - "never_treated": Includes only never-treated units.
    anticipation : int, default 0
        Number of periods before actual treatment start where units are considered treated
        (e.g., due to announcement effects).
    base_period : {"universal", "varying"}, default "universal"
        Definition of the 'before' period in the DID comparison.
        - "universal": Uses :math:`g - 1 - \text{anticipation}` as the base for all :math:`t`.
        - "varying": Uses :math:`t - 1` as the base for all :math:`t`.
    include_pre_periods : bool, default False
        Whether to estimate :math:`ATT(g,t)` for :math:`t < g` (pre-treatment testing).
    alpha : float, default 0.05
        Significance level for confidence intervals.
    diagnostic_data : bool, default True
        Whether to store diagnostic information (overlap, balance, etc.).
    propensity_clip : float, default 1e-4
        Clipping threshold for propensity scores to avoid division by zero.
    logit_ridge : float, default 1e-4
        L2 regularization strength for the logistic propensity score model.
    optimizer_tol : float, default 1e-8
        Tolerance for the propensity score optimizer.
    optimizer_maxiter : int, default 1000
        Maximum iterations for the propensity score optimizer.
    min_treated_per_cell : int, default 30
        Minimum number of treated units required in a :math:`(g,t)` cell.
    min_control_per_cell : int, default 30
        Minimum number of control units required in a :math:`(g,t)` cell.
    min_control_ess : float, default 20.0
        Minimum effective sample size for control units in a :math:`(g,t)` cell.
    max_propensity_clip_share : float, default 0.05
        Maximum fraction of units that can be clipped before skipping a cell.
    max_condition_number : float, default 1e5
        Maximum condition number for covariate matrices.
    bootstrap_replications : int, default 0
        Number of multiplier bootstrap replications for simultaneous confidence bands.
        If 0, uses asymptotic normal approximation for pointwise intervals.
    random_state : int, optional, default None
        Random seed for bootstrap multipliers.

    Examples
    --------
    >>> from causalis.scenarios.did.dgp import generate_did_gamma_26
    >>> from causalis.scenarios.did.model import CallawaySantAnnaDID
    >>> # Generate synthetic staggered-adoption panel data
    >>> data = generate_did_gamma_26(
    ...     n_treated_units=50,
    ...     n_control_units=100,
    ...     seed=42,
    ... )
    >>> # Initialize and fit the model
    >>> model = CallawaySantAnnaDID(estimator="dr", control_group="not_yet_or_never")
    >>> model.fit(data)  # doctest: +ELLIPSIS
    CallawaySantAnnaDID(status='fitted', ...)
    >>> # Estimate effects and aggregate
    >>> results = model.estimate()
    >>> results.summary()  # doctest: +SKIP
    >>> # Access specific aggregations
    >>> event_study = results.event_study()
    >>> event_study.head()  # doctest: +SKIP

    Notes
    -----
    In a staggered adoption design, units :math:`i` are treated at different times :math:`G_i \in \{g_1, \dots, g_K, \infty\}`.
    The group-time average treatment effect is defined as:

    .. math::

        ATT(g,t) = \mathbb{E}[Y_t(g) - Y_t(\infty) \mid G=g]

    where :math:`Y_t(g)` is the potential outcome at time :math:`t` if treated at time :math:`g`,
    and :math:`Y_t(\infty)` is the potential outcome if never treated.

    This implementation follows Callaway and Sant'Anna (2021) and uses logistic regression
    for propensity scores and linear regression for outcome models (in "dr" mode).
    The doubly robust estimator for :math:`ATT(g,t)` is:

    .. math::

        \theta_{DR}(g,t) = \mathbb{E} \left[ \left( \frac{G_g}{\mathbb{E}[G_g]} - \frac{\frac{p_g(X) C}{1-p_g(X)}}{\mathbb{E}[\frac{p_g(X) C}{1-p_g(X)}]} \right) (Y_t - Y_{base} - m_{g,t}(X)) \right]

    where :math:`G_g = \mathbb{1}\{G=g\}`, :math:`C` is the control group indicator,
    :math:`p_g(X)` is the propensity score, and :math:`m_{g,t}(X)` is the outcome regression.

    Simultaneous confidence bands are computed using the multiplier bootstrap on the
    influence functions of the :math:`ATT(g,t)` estimates.
    """

    def __init__(
        self,
        *,
        estimator: Estimator = "dr",
        control_group: ComparisonGroup = "not_yet_or_never",
        anticipation: int = 0,
        base_period: BasePeriod = "universal",
        include_pre_periods: bool = False,
        alpha: float = 0.05,
        diagnostic_data: bool = True,
        propensity_clip: float = _DEFAULT_PROPENSITY_CLIP,
        logit_ridge: float = _DEFAULT_LOGIT_RIDGE,
        optimizer_tol: float = _DEFAULT_OPTIMIZER_TOL,
        optimizer_maxiter: int = _DEFAULT_OPTIMIZER_MAXITER,
        min_treated_per_cell: int = 30,
        min_control_per_cell: int = 30,
        min_control_ess: float = 20.0,
        max_propensity_clip_share: float = 0.05,
        max_condition_number: float = _DEFAULT_MAX_CONDITION_NUMBER,
        bootstrap_replications: int = 0,
        random_state: Optional[int] = None,
    ) -> None:
        self.estimator = _validate_estimator(estimator)
        self.control_group = _validate_control_group(control_group)
        self.anticipation = _validate_nonnegative_int(anticipation, "anticipation")
        self.base_period = _validate_base_period(base_period)
        self.include_pre_periods = bool(include_pre_periods)
        self.alpha = _validate_alpha(alpha)
        self.diagnostic_data = bool(diagnostic_data)
        self.propensity_clip = _validate_propensity_clip(propensity_clip)
        self.logit_ridge = _validate_nonnegative_float(logit_ridge, "logit_ridge")
        self.optimizer_tol = _validate_positive_float(optimizer_tol, "optimizer_tol")
        self.optimizer_maxiter = _validate_positive_int(optimizer_maxiter, "optimizer_maxiter")
        self.min_treated_per_cell = _validate_positive_int(min_treated_per_cell, "min_treated_per_cell")
        self.min_control_per_cell = _validate_positive_int(min_control_per_cell, "min_control_per_cell")
        self.min_control_ess = _validate_positive_float(min_control_ess, "min_control_ess")
        self.max_propensity_clip_share = _validate_nonnegative_float(
            max_propensity_clip_share,
            "max_propensity_clip_share",
        )
        self.max_condition_number = _validate_positive_float(max_condition_number, "max_condition_number")
        self.bootstrap_replications = _validate_nonnegative_int(bootstrap_replications, "bootstrap_replications")
        self.random_state = random_state

        self._data: Optional[PanelDataDID] = None
        self._prepared: Optional[_PreparedPanel] = None
        self._att_gt: Optional[pd.DataFrame] = None
        self._score_matrix: Optional[np.ndarray] = None
        self._unit_diagnostics: Optional[pd.DataFrame] = None
        self._overlap: Optional[pd.DataFrame] = None
        self._balance: Optional[pd.DataFrame] = None
        self._support: Optional[pd.DataFrame] = None
        self._skipped_cells: Optional[pd.DataFrame] = None
        self._is_fitted = False

    def fit(self, data: PanelDataDID) -> "CallawaySantAnnaDID":
        """
        Fit all supported :math:`ATT(g,t)` cells on a validated :class:`PanelDataDID`.

        This method prepares the panel data, identifies valid comparison groups for each
        cohort-time cell, and estimates the group-time average treatment effects.

        Parameters
        ----------
        data : PanelDataDID
            The validated panel data container.

        Returns
        -------
        CallawaySantAnnaDID
            The fitted model instance.
        """

        prepared = _prepare_panel(data)
        att_gt, scores, unit_diag, overlap, balance, support, skipped = _fit_att_gt(
            prepared,
            control_group=self.control_group,
            estimator=self.estimator,
            anticipation=self.anticipation,
            base_period=self.base_period,
            include_pre_periods=self.include_pre_periods,
            propensity_clip=self.propensity_clip,
            logit_ridge=self.logit_ridge,
            optimizer_tol=self.optimizer_tol,
            optimizer_maxiter=self.optimizer_maxiter,
            min_treated_per_cell=self.min_treated_per_cell,
            min_control_per_cell=self.min_control_per_cell,
            min_control_ess=self.min_control_ess,
            max_propensity_clip_share=self.max_propensity_clip_share,
            max_condition_number=self.max_condition_number,
        )

        self._data = data
        self._prepared = prepared
        self._att_gt = att_gt
        self._score_matrix = scores
        self._unit_diagnostics = unit_diag
        self._overlap = overlap
        self._balance = balance
        self._support = support
        self._skipped_cells = skipped
        self._is_fitted = True
        return self

    def _require_fitted(self) -> tuple[PanelDataDID, _PreparedPanel, pd.DataFrame, np.ndarray]:
        if (
            not self._is_fitted
            or self._data is None
            or self._prepared is None
            or self._att_gt is None
            or self._score_matrix is None
        ):
            raise RuntimeError("Model must be fitted with .fit(data) before calling .estimate().")
        return self._data, self._prepared, self._att_gt, self._score_matrix

    def estimate(
        self,
        *,
        alpha: Optional[float] = None,
        diagnostic_data: Optional[bool] = None,
        bootstrap_replications: Optional[int] = None,
        random_state: Optional[int] = None,
    ) -> CallawaySantAnnaDIDEstimate:
        """
        Aggregate group-time effects and perform inference.

        This method computes simple, cohort, calendar, and event-study aggregations
        from the fitted :math:`ATT(g,t)` cells. It also performs inference using either
        asymptotic normal approximation or the multiplier bootstrap.

        Parameters
        ----------
        alpha : float, optional
            Significance level. If None, uses the value from ``__init__``.
        diagnostic_data : bool, optional
            Whether to include diagnostic data in the result. If None, uses the
            value from ``__init__``.
        bootstrap_replications : int, optional
            Number of multiplier bootstrap replications. If None, uses the value
            from ``__init__``.
        random_state : int, optional
            Random seed for bootstrap. If None, uses the value from ``__init__``.

        Returns
        -------
        CallawaySantAnnaDIDEstimate
            The estimation results including aggregations and inference.
        """

        data, prepared, att_gt_raw, score_matrix = self._require_fitted()
        a = self.alpha if alpha is None else _validate_alpha(alpha)
        include_diagnostics = self.diagnostic_data if diagnostic_data is None else bool(diagnostic_data)
        b = self.bootstrap_replications if bootstrap_replications is None else _validate_nonnegative_int(
            bootstrap_replications, "bootstrap_replications"
        )
        seed = self.random_state if random_state is None else random_state
        rng = np.random.default_rng(seed)
        multiplier_weights = (
            _draw_multiplier_weights(
                n_units=score_matrix.shape[0],
                clusters=prepared.clusters,
                replications=b,
                rng=rng,
            )
            if b > 0
            else None
        )

        att_gt = _add_inference(
            att_gt_raw,
            score_matrix,
            estimate_col="att",
            clusters=prepared.clusters,
            alpha=a,
            bootstrap_replications=b,
            rng=rng,
            multiplier_weights=multiplier_weights,
            simultaneous=True,
        )

        aggregates: dict[str, pd.DataFrame] = {}
        weights_parts: list[pd.DataFrame] = []
        for kind in ("simple", "cohort", "calendar", "event"):
            table, agg_scores, weights = _aggregate_scores(att_gt_raw, score_matrix, kind=kind)  # type: ignore[arg-type]
            aggregates[kind] = _add_inference(
                table,
                agg_scores,
                estimate_col="estimate",
                clusters=prepared.clusters,
                alpha=a,
                bootstrap_replications=b,
                rng=rng,
                multiplier_weights=multiplier_weights,
                simultaneous=(kind == "event"),
            )
            weights_parts.append(weights)

        inference = (
            "clustered_multiplier_bootstrap"
            if prepared.clusters is not None and b > 0
            else "multiplier_bootstrap"
            if b > 0
            else "clustered_influence"
            if prepared.clusters is not None
            else "influence"
        )

        diagnostic_cols = [
            "cell_id",
            "cohort",
            "time",
            "base_time",
            "event_time",
            "is_post_treatment",
            "n_treated_available",
            "n_treated_complete",
            "n_control_available",
            "n_control_complete",
            "control_design_rank",
            "n_parameters",
            "condition_number",
            "treated_weight_ess",
            "control_weight_ess",
            "propensity_clip_share",
            "diagnostic_status",
            "diagnostic_flags",
        ]
        cell_diagnostics = att_gt_raw[diagnostic_cols].copy()
        support = self._support.copy() if self._support is not None else pd.DataFrame()
        skipped_cells = self._skipped_cells.copy() if self._skipped_cells is not None else pd.DataFrame()
        weights = pd.concat(weights_parts, ignore_index=True) if weights_parts else pd.DataFrame()
        diagnostics: dict[str, Any] = {
            "estimand": "average_post_effect",
            "ci_alpha": float(a),
            "diagnostic_data_requested": bool(include_diagnostics),
            "estimator": self.estimator,
            "control_group": self.control_group,
            "anticipation": int(self.anticipation),
            "base_period": self.base_period,
            "include_pre_periods": bool(self.include_pre_periods),
            "inference": inference,
            "bootstrap_replications": int(b),
            "cluster_col": data.cluster_col,
            "n_units": int(len(prepared.unit_ids)),
            "n_att_gt_cells": int(len(att_gt)),
            "n_post_treatment_att_gt_cells": int(att_gt["is_post_treatment"].sum()),
            "n_pre_period_att_gt_cells": int((~att_gt["is_post_treatment"]).sum()),
            "n_skipped_cells": int(len(skipped_cells)),
            "min_treated_per_cell": int(self.min_treated_per_cell),
            "min_control_per_cell": int(self.min_control_per_cell),
            "min_control_ess": float(self.min_control_ess),
            "max_propensity_clip_share": float(self.max_propensity_clip_share),
            "max_condition_number": float(self.max_condition_number),
            "support": support,
            "skipped_cells": skipped_cells,
            "cell_diagnostics": cell_diagnostics,
            "weights": weights,
        }
        if include_diagnostics:
            influence = pd.DataFrame(
                score_matrix,
                index=prepared.unit_ids,
                columns=att_gt_raw["cell_id"].tolist(),
            )
            influence.index.name = data.unit_col
            diagnostics.update(
                {
                    "unit_level": (
                        self._unit_diagnostics.copy()
                        if self._unit_diagnostics is not None
                        else pd.DataFrame()
                    ),
                    "overlap": self._overlap.copy() if self._overlap is not None else pd.DataFrame(),
                    "balance": self._balance.copy() if self._balance is not None else pd.DataFrame(),
                    "influence_scores": influence,
                }
            )

        return CallawaySantAnnaDIDEstimate(
            model=_MODEL_NAME,
            estimator=self.estimator,
            control_group=self.control_group,
            anticipation=self.anticipation,
            base_period=self.base_period,
            include_pre_periods=self.include_pre_periods,
            alpha=a,
            att_gt=att_gt,
            aggregates=aggregates,
            support=support,
            skipped_cells=skipped_cells,
            outcome=data.y,
            treatment=data.treated_time,
            unit_col=data.unit_col,
            time_col=data.time_col,
            covariates=list(data.covariates),
            cluster_col=data.cluster_col,
            inference=inference,
            diagnostics=diagnostics,
        )

    @property
    def is_fitted(self) -> bool:
        """Whether the model has been fitted on data."""
        return self._is_fitted

    @property
    def support_(self) -> pd.DataFrame:
        """A table of units and their inclusion status in various :math:`ATT(g,t)` cells."""
        if self._support is None:
            raise RuntimeError("Model must be fitted before support_ is available.")
        return self._support.copy()

    @property
    def skipped_cells_(self) -> pd.DataFrame:
        """A table of cohort-time cells that were skipped due to insufficient data or quality issues."""
        if self._skipped_cells is None:
            raise RuntimeError("Model must be fitted before skipped_cells_ is available.")
        return self._skipped_cells.copy()

    @property
    def cell_diagnostics_(self) -> pd.DataFrame:
        """Detailed diagnostic information for each :math:`ATT(g,t)` cell."""
        if self._att_gt is None:
            raise RuntimeError("Model must be fitted before cell_diagnostics_ is available.")
        cols = [
            "cell_id",
            "cohort",
            "time",
            "base_time",
            "event_time",
            "is_post_treatment",
            "n_treated_available",
            "n_treated_complete",
            "n_control_available",
            "n_control_complete",
            "control_design_rank",
            "n_parameters",
            "condition_number",
            "treated_weight_ess",
            "control_weight_ess",
            "propensity_clip_share",
            "diagnostic_status",
            "diagnostic_flags",
        ]
        return self._att_gt[cols].copy()

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "unfitted"
        return (
            f"{self.__class__.__name__}(status={status!r}, estimator={self.estimator!r}, "
            f"control_group={self.control_group!r}, anticipation={self.anticipation!r}, "
            f"base_period={self.base_period!r}, include_pre_periods={self.include_pre_periods!r}, "
            f"alpha={self.alpha!r})"
        )


__all__ = [
    "AggregateKind",
    "BasePeriod",
    "CallawaySantAnnaDID",
    "CallawaySantAnnaDIDEstimate",
    "Estimator",
]
