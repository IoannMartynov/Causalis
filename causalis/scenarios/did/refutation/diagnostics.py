from __future__ import annotations

from typing import Any, Hashable, Literal

import numpy as np
import pandas as pd

from causalis.data_contracts import PanelDataDID
from causalis.data_contracts.panel_data_did import ComparisonGroup


BasePeriod = Literal["universal", "varying"]

_CONTROL_GROUPS = {"never_treated", "not_yet_treated", "not_yet_or_never"}
_BASE_PERIODS = {"universal", "varying"}
_DEFAULT_MAX_CONDITION_NUMBER = 1e8


def _ensure_panel_data(data: PanelDataDID) -> PanelDataDID:
    if not isinstance(data, PanelDataDID):
        raise TypeError("data must be a PanelDataDID instance.")
    return data


def _validate_control_group(control_group: str) -> ComparisonGroup:
    if control_group not in _CONTROL_GROUPS:
        raise ValueError(
            "control_group must be one of 'never_treated', 'not_yet_treated', or 'not_yet_or_never'."
        )
    return control_group  # type: ignore[return-value]


def _validate_base_period(base_period: str) -> BasePeriod:
    if base_period not in _BASE_PERIODS:
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


def _validate_probability(value: float, name: str) -> float:
    out = float(value)
    if not np.isfinite(out) or not (0.0 <= out <= 1.0):
        raise ValueError(f"{name} must be finite and in [0, 1].")
    return out


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


def _ratio(numerator: Any, denominator: Any) -> float:
    try:
        num = float(numerator)
        den = float(denominator)
    except (TypeError, ValueError):
        return float("nan")
    if not np.isfinite(num) or not np.isfinite(den) or den <= 0.0:
        return float("nan")
    return float(num / den)


def _safe_smd(
    treated_values: np.ndarray,
    control_values: np.ndarray,
) -> float:
    treated = np.asarray(treated_values, dtype=float)
    control = np.asarray(control_values, dtype=float)
    if treated.size == 0 or control.size == 0:
        return float("nan")

    treated_mean = float(np.mean(treated))
    control_mean = float(np.mean(control))
    treated_var = float(np.var(treated, ddof=1)) if treated.size > 1 else 0.0
    control_var = float(np.var(control, ddof=1)) if control.size > 1 else 0.0
    pooled = float(np.sqrt(0.5 * (treated_var + control_var)))
    if pooled == 0.0:
        return 0.0 if abs(treated_mean - control_mean) <= 1e-16 else float("inf")
    return float((treated_mean - control_mean) / pooled)


def _comparison_units_at_time(
    data: PanelDataDID,
    time: pd.Period,
    *,
    control_group: ComparisonGroup,
    anticipation: int,
) -> list[Hashable]:
    time_index = data.time_to_index()
    target_idx = time_index[time] + anticipation
    first_by_unit = data.first_treatment_by_unit
    units = data.df_analysis()[data.unit_col].drop_duplicates().tolist()

    out: list[Hashable] = []
    for unit in units:
        first_treatment = first_by_unit[unit]
        if control_group == "never_treated":
            include = first_treatment is None
        elif control_group == "not_yet_treated":
            include = first_treatment is not None and time_index[first_treatment] > target_idx
        else:
            include = first_treatment is None or time_index[first_treatment] > target_idx
        if include:
            out.append(unit)
    return out


def _complete_pair_units(
    data: PanelDataDID,
    candidate_units: list[Hashable],
    *,
    base_time: pd.Period,
    target_time: pd.Period,
) -> list[Hashable]:
    if not candidate_units:
        return []
    df = data.df_analysis()
    base_units = set(
        df.loc[df[data.time_col] == base_time, data.unit_col].drop_duplicates().tolist()
    )
    target_units = set(
        df.loc[df[data.time_col] == target_time, data.unit_col].drop_duplicates().tolist()
    )
    observed = base_units.intersection(target_units)
    return [unit for unit in candidate_units if unit in observed]


def _delta_y(
    data: PanelDataDID,
    units: list[Hashable],
    *,
    base_time: pd.Period,
    target_time: pd.Period,
) -> np.ndarray:
    if not units:
        return np.empty(0, dtype=float)
    df = data.df_analysis()
    wide = (
        df.loc[
            df[data.unit_col].isin(units)
            & df[data.time_col].isin([base_time, target_time]),
            [data.unit_col, data.time_col, data.y],
        ]
        .pivot(index=data.unit_col, columns=data.time_col, values=data.y)
        .reindex(index=units, columns=[base_time, target_time])
    )
    return (
        wide[target_time].to_numpy(dtype=float)
        - wide[base_time].to_numpy(dtype=float)
    )


def _base_covariate_values(
    data: PanelDataDID,
    units: list[Hashable],
    covariate: str,
    *,
    base_time: pd.Period,
) -> np.ndarray:
    if not units:
        return np.empty(0, dtype=float)
    df = data.df_analysis()
    base = (
        df.loc[
            df[data.time_col] == base_time,
            [data.unit_col, covariate],
        ]
        .drop_duplicates(subset=[data.unit_col])
        .set_index(data.unit_col)
        .reindex(units)
    )
    return base[covariate].to_numpy(dtype=float)


def did_support_table(
    data: PanelDataDID,
    *,
    control_group: ComparisonGroup = "not_yet_or_never",
    anticipation: int = 0,
    base_period: BasePeriod = "universal",
    include_pre_periods: bool = False,
) -> pd.DataFrame:
    """
    Return Callaway & Sant'Anna ``ATT(g,t)`` support under the requested model policy.

    This function identifies the available cohort-time cells for estimation and
    verifies if there are enough units to form complete treated/control pairs.
    A unit is "complete" if it is observed in both the base period and the
    target period.

    Parameters
    ----------
    data : PanelDataDID
        The validated panel data object.
    control_group : {"not_yet_or_never", "never_treated"}, default "not_yet_or_never"
        The definition of the comparison group.
    anticipation : int, default 0
        Number of periods before treatment to exclude from the control group
        due to potential anticipation effects.
    base_period : {"universal", "varying"}, default "universal"
        Whether to use a fixed base period (universal) or a period-specific
        one (varying) for each target period.
    include_pre_periods : bool, default False
        Whether to include pre-treatment periods (useful for placebo tests).

    Returns
    -------
    pd.DataFrame
        A table of support metrics for each cohort-time cell:
            - ``cohort``: The treatment group.
            - ``time``: The calendar period.
            - ``base_time``: The period used as a baseline for the difference.
            - ``is_supported``: Whether the cell has sufficient data for estimation.
            - ``n_treated_complete``: Number of treated units observed in both periods.
            - ``n_control_complete``: Number of control units observed in both periods.
            - ``treated_completion_rate``: Share of cohort units that are complete.
            - ``control_completion_rate``: Share of control units that are complete.

    Notes
    -----
    The Callaway & Sant'Anna (2021) estimator requires that for each target
    parameter :math:`ATT(g,t)`, there exists a set of units in the comparison
    group that are also observed in the base period :math:`g-1` (or :math:`t-1`
    for varying base periods).

    Examples
    --------
    >>> from causalis.scenarios.did import generate_did_gamma_26
    >>> from causalis.scenarios.did.refutation import did_support_table
    >>> data = generate_did_gamma_26(n_units=100, n_periods=5, seed=42)
    >>> support = did_support_table(data, control_group="never_treated")
    >>> support[["cohort", "time", "is_supported", "n_treated_complete"]].head()
    """

    data = _ensure_panel_data(data)
    control_group = _validate_control_group(control_group)
    anticipation = _validate_nonnegative_int(anticipation, "anticipation")
    base_period = _validate_base_period(base_period)

    support = data.att_gt_cells(
        control_group=control_group,
        anticipation=anticipation,
        base_period=base_period,
        include_pre_periods=bool(include_pre_periods),
        include_unsupported=True,
    )
    out = support.reset_index(drop=True).copy()
    out.insert(0, "support_cell_id", np.arange(len(out), dtype=int))

    out["treated_completion_rate"] = [
        _ratio(num, den)
        for num, den in zip(out["n_treated_complete"], out["n_treated_available"])
    ]
    out["control_completion_rate"] = [
        _ratio(num, den)
        for num, den in zip(out["n_control_complete"], out["n_control_available"])
    ]
    out["control_to_treated_ratio"] = [
        _ratio(num, den)
        for num, den in zip(out["n_control_complete"], out["n_treated_complete"])
    ]
    out["cell_type"] = np.where(
        out["is_post_treatment"].fillna(False).astype(bool),
        "post",
        "pre",
    )
    return out


def raw_did_event_study_table(
    data: PanelDataDID,
    *,
    control_group: ComparisonGroup = "not_yet_or_never",
    anticipation: int = 0,
    base_period: BasePeriod = "varying",
    include_pre_periods: bool = True,
) -> pd.DataFrame:
    """
    Return unadjusted DID event-study cells from the validated panel.

    This function calculates simple mean differences between treated and control
    groups across different event-time periods. These are "raw" estimates
    without covariate adjustment or IPW/DR weighting.

    Parameters
    ----------
    data : PanelDataDID
        The validated panel data object.
    control_group : {"not_yet_or_never", "never_treated"}, default "not_yet_or_never"
        The definition of the comparison group.
    anticipation : int, default 0
        Number of periods before treatment to exclude.
    base_period : {"varying", "universal"}, default "varying"
        The base period policy for the event study.
    include_pre_periods : bool, default True
        Whether to include pre-treatment (placebo) periods.

    Returns
    -------
    pd.DataFrame
        A table of raw DID estimates:
            - ``event_time``: Periods relative to treatment (:math:`t - g`).
            - ``raw_did``: The unadjusted Difference-in-Differences estimate.
            - ``se``: Naive standard error of the mean difference.
            - ``t_stat``: t-statistic for the null of zero difference.
            - ``n_treated``, ``n_control``: Sample sizes in the cell.

    Notes
    -----
    The raw DID for a cohort-time cell :math:`(g, t)` is calculated as:

    .. math::
        \\Delta_{raw}(g,t) = [E[Y_t | G=g] - E[Y_{base} | G=g]] - [E[Y_t | C] - E[Y_{base} | C]]

    where :math:`C` is the comparison group. These are useful for visual
    inspection of parallel trends before applying more complex estimators.

    Examples
    --------
    >>> from causalis.scenarios.did import generate_did_gamma_26
    >>> from causalis.scenarios.did.refutation import raw_did_event_study_table
    >>> data = generate_did_gamma_26(n_units=200, n_periods=5, seed=42)
    >>> event_table = raw_did_event_study_table(data)
    >>> event_table[["event_time", "raw_did", "t_stat"]].head()
    """

    data = _ensure_panel_data(data)
    control_group = _validate_control_group(control_group)
    anticipation = _validate_nonnegative_int(anticipation, "anticipation")
    base_period = _validate_base_period(base_period)

    support = did_support_table(
        data,
        control_group=control_group,
        anticipation=anticipation,
        base_period=base_period,
        include_pre_periods=include_pre_periods,
    )
    rows: list[dict[str, Any]] = []
    for support_row in support.to_dict("records"):
        if not bool(support_row["is_supported"]):
            continue
        cohort = support_row["cohort"]
        target_time = support_row["time"]
        base_time = support_row["base_time"]
        if pd.isna(cohort) or pd.isna(target_time) or pd.isna(base_time):
            continue

        cohort_units = list(data.cohort_units(cohort))
        comparison_units = _comparison_units_at_time(
            data,
            target_time,
            control_group=control_group,
            anticipation=anticipation,
        )
        treated_units = _complete_pair_units(
            data,
            cohort_units,
            base_time=base_time,
            target_time=target_time,
        )
        control_units = _complete_pair_units(
            data,
            comparison_units,
            base_time=base_time,
            target_time=target_time,
        )
        treated_delta = _delta_y(
            data,
            treated_units,
            base_time=base_time,
            target_time=target_time,
        )
        control_delta = _delta_y(
            data,
            control_units,
            base_time=base_time,
            target_time=target_time,
        )
        if treated_delta.size == 0 or control_delta.size == 0:
            continue

        treated_mean = float(np.mean(treated_delta))
        control_mean = float(np.mean(control_delta))
        raw_did = float(treated_mean - control_mean)
        treated_var = (
            float(np.var(treated_delta, ddof=1)) if treated_delta.size > 1 else 0.0
        )
        control_var = (
            float(np.var(control_delta, ddof=1)) if control_delta.size > 1 else 0.0
        )
        se = float(
            np.sqrt(treated_var / treated_delta.size + control_var / control_delta.size)
        )
        if se > 0.0:
            t_stat = float(raw_did / se)
        elif abs(raw_did) <= 1e-16:
            t_stat = 0.0
        else:
            t_stat = float("inf")
        rows.append(
            {
                "support_cell_id": int(support_row["support_cell_id"]),
                "cohort": cohort,
                "time": target_time,
                "base_time": base_time,
                "event_time": int(support_row["event_time"]),
                "is_post_treatment": bool(support_row["is_post_treatment"]),
                "n_treated": int(treated_delta.size),
                "n_control": int(control_delta.size),
                "treated_mean_delta": treated_mean,
                "control_mean_delta": control_mean,
                "raw_did": raw_did,
                "se": se,
                "t_stat": t_stat,
            }
        )

    columns = [
        "support_cell_id",
        "cohort",
        "time",
        "base_time",
        "event_time",
        "is_post_treatment",
        "n_treated",
        "n_control",
        "treated_mean_delta",
        "control_mean_delta",
        "raw_did",
        "se",
        "t_stat",
    ]
    return pd.DataFrame(rows, columns=columns)


def did_covariate_balance_table(
    data: PanelDataDID,
    *,
    control_group: ComparisonGroup = "not_yet_or_never",
    anticipation: int = 0,
    base_period: BasePeriod = "universal",
    include_pre_periods: bool = False,
    post_only: bool = True,
) -> pd.DataFrame:
    """
    Return unweighted base-period covariate balance for Callaway & Sant'Anna cells.

    Calculates the standardized mean difference (SMD) for each covariate
    between the treated and control groups in the base period.

    Parameters
    ----------
    data : PanelDataDID
        The validated panel data object.
    control_group : {"not_yet_or_never", "never_treated"}, default "not_yet_or_never"
        The definition of the comparison group.
    anticipation : int, default 0
        Anticipation periods to exclude.
    base_period : {"universal", "varying"}, default "universal"
        Base period policy.
    include_pre_periods : bool, default False
        Whether to include pre-treatment cells.
    post_only : bool, default True
        If True, only checks balance for cells used in post-treatment estimation.

    Returns
    -------
    pd.DataFrame
        A balance table with columns:
            - ``covariate``: Name of the covariate.
            - ``treated_mean``: Average value in the treated group.
            - ``control_mean``: Average value in the control group.
            - ``smd``: Standardized Mean Difference.
            - ``abs_smd``: Absolute value of the SMD.

    Notes
    -----
    The Standardized Mean Difference is defined as:

    .. math::
        SMD = \\frac{\\bar{X}_{treated} - \\bar{X}_{control}}{\\sqrt{(s^2_{treated} + s^2_{control}) / 2}}

    Values of :math:`|SMD| > 0.1` or :math:`0.25` are often used as thresholds
    to indicate potential imbalance that requires adjustment.

    Examples
    --------
    >>> from causalis.scenarios.did import generate_did_gamma_26
    >>> from causalis.scenarios.did.refutation import did_covariate_balance_table
    >>> data = generate_did_gamma_26(n_units=200, n_periods=5, seed=42)
    >>> balance = did_covariate_balance_table(data)
    >>> balance[["covariate", "treated_mean", "control_mean", "abs_smd"]].head()
    """

    data = _ensure_panel_data(data)
    control_group = _validate_control_group(control_group)
    anticipation = _validate_nonnegative_int(anticipation, "anticipation")
    base_period = _validate_base_period(base_period)

    columns = [
        "support_cell_id",
        "cohort",
        "time",
        "base_time",
        "event_time",
        "is_post_treatment",
        "covariate",
        "n_treated",
        "n_control",
        "treated_mean",
        "control_mean",
        "mean_difference",
        "smd",
        "abs_smd",
    ]
    if not data.covariates:
        return pd.DataFrame(columns=columns)

    support = did_support_table(
        data,
        control_group=control_group,
        anticipation=anticipation,
        base_period=base_period,
        include_pre_periods=include_pre_periods,
    )
    if post_only:
        support = support[support["is_post_treatment"].fillna(False).astype(bool)]

    rows: list[dict[str, Any]] = []
    for support_row in support.to_dict("records"):
        if not bool(support_row["is_supported"]):
            continue
        cohort = support_row["cohort"]
        target_time = support_row["time"]
        base_time = support_row["base_time"]
        if pd.isna(cohort) or pd.isna(target_time) or pd.isna(base_time):
            continue

        cohort_units = list(data.cohort_units(cohort))
        comparison_units = _comparison_units_at_time(
            data,
            target_time,
            control_group=control_group,
            anticipation=anticipation,
        )
        treated_units = _complete_pair_units(
            data,
            cohort_units,
            base_time=base_time,
            target_time=target_time,
        )
        control_units = _complete_pair_units(
            data,
            comparison_units,
            base_time=base_time,
            target_time=target_time,
        )
        for covariate in data.covariates:
            treated_values = _base_covariate_values(
                data,
                treated_units,
                covariate,
                base_time=base_time,
            )
            control_values = _base_covariate_values(
                data,
                control_units,
                covariate,
                base_time=base_time,
            )
            treated_mean = float(np.mean(treated_values))
            control_mean = float(np.mean(control_values))
            smd = _safe_smd(treated_values, control_values)
            rows.append(
                {
                    "support_cell_id": int(support_row["support_cell_id"]),
                    "cohort": cohort,
                    "time": target_time,
                    "base_time": base_time,
                    "event_time": int(support_row["event_time"]),
                    "is_post_treatment": bool(support_row["is_post_treatment"]),
                    "covariate": covariate,
                    "n_treated": int(treated_values.size),
                    "n_control": int(control_values.size),
                    "treated_mean": treated_mean,
                    "control_mean": control_mean,
                    "mean_difference": float(treated_mean - control_mean),
                    "smd": smd,
                    "abs_smd": abs(smd) if np.isfinite(smd) else float("inf"),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def did_base_design_table(
    data: PanelDataDID,
    *,
    control_group: ComparisonGroup = "not_yet_or_never",
    anticipation: int = 0,
    base_period: BasePeriod = "universal",
    include_pre_periods: bool = False,
    post_only: bool = True,
) -> pd.DataFrame:
    """
    Return base-period control-design rank diagnostics for Callaway & Sant'Anna cells.

    Checks the numerical stability of the propensity score and outcome regression
    designs in the comparison group. High condition numbers or rank deficiency
    indicate potential multicollinearity or insufficient variation in covariates.

    Parameters
    ----------
    data : PanelDataDID
        The validated panel data object.
    control_group : {"not_yet_or_never", "never_treated"}, default "not_yet_or_never"
        The definition of the comparison group.
    anticipation : int, default 0
        Anticipation periods to exclude.
    base_period : {"universal", "varying"}, default "universal"
        Base period policy.
    include_pre_periods : bool, default False
        Whether to include pre-treatment cells.
    post_only : bool, default True
        If True, only checks cells used in post-treatment estimation.

    Returns
    -------
    pd.DataFrame
        A table of design diagnostics:
            - ``n_control``: Number of units in the control pool for the cell.
            - ``n_parameters``: Number of covariates including the intercept.
            - ``control_design_rank``: Matrix rank of the covariate design.
            - ``condition_number``: The L2 condition number of the design matrix.
            - ``is_rank_deficient``: Whether the matrix is not full rank.

    Examples
    --------
    >>> from causalis.scenarios.did import generate_did_gamma_26
    >>> from causalis.scenarios.did.refutation import did_base_design_table
    >>> data = generate_did_gamma_26(n_units=200, n_periods=5, seed=42)
    >>> design = did_base_design_table(data)
    >>> design[["n_control", "condition_number", "is_rank_deficient"]].head()
    """

    data = _ensure_panel_data(data)
    control_group = _validate_control_group(control_group)
    anticipation = _validate_nonnegative_int(anticipation, "anticipation")
    base_period = _validate_base_period(base_period)

    support = did_support_table(
        data,
        control_group=control_group,
        anticipation=anticipation,
        base_period=base_period,
        include_pre_periods=include_pre_periods,
    )
    if post_only:
        support = support[support["is_post_treatment"].fillna(False).astype(bool)]

    rows: list[dict[str, Any]] = []
    covariates = list(data.covariates)
    for support_row in support.to_dict("records"):
        if not bool(support_row["is_supported"]):
            continue
        cohort = support_row["cohort"]
        target_time = support_row["time"]
        base_time = support_row["base_time"]
        if pd.isna(cohort) or pd.isna(target_time) or pd.isna(base_time):
            continue

        comparison_units = _comparison_units_at_time(
            data,
            target_time,
            control_group=control_group,
            anticipation=anticipation,
        )
        control_units = _complete_pair_units(
            data,
            comparison_units,
            base_time=base_time,
            target_time=target_time,
        )
        intercept = np.ones((len(control_units), 1), dtype=float)
        if covariates:
            covariate_columns = [
                _base_covariate_values(
                    data,
                    control_units,
                    covariate,
                    base_time=base_time,
                )
                for covariate in covariates
            ]
            design = np.column_stack([intercept, *covariate_columns])
        else:
            design = intercept

        n_parameters = int(design.shape[1])
        rank = int(np.linalg.matrix_rank(design)) if design.size else 0
        condition_number = (
            float(np.linalg.cond(design))
            if design.shape[0] > 0 and design.shape[1] > 0
            else float("inf")
        )
        rows.append(
            {
                "support_cell_id": int(support_row["support_cell_id"]),
                "cohort": cohort,
                "time": target_time,
                "base_time": base_time,
                "event_time": int(support_row["event_time"]),
                "is_post_treatment": bool(support_row["is_post_treatment"]),
                "n_control": int(len(control_units)),
                "n_parameters": n_parameters,
                "control_design_rank": rank,
                "condition_number": condition_number,
                "is_rank_deficient": bool(rank < n_parameters),
            }
        )

    columns = [
        "support_cell_id",
        "cohort",
        "time",
        "base_time",
        "event_time",
        "is_post_treatment",
        "n_control",
        "n_parameters",
        "control_design_rank",
        "condition_number",
        "is_rank_deficient",
    ]
    return pd.DataFrame(rows, columns=columns)


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


def _append_min_count_check(
    checks: list[dict[str, Any]],
    *,
    test: str,
    values: pd.Series,
    threshold_value: int,
    metric_label: str,
) -> None:
    if values.empty:
        _append_check(
            checks,
            test=test,
            flag="RED",
            value=None,
            threshold=f">= {threshold_value}",
            message=f"No supported post-treatment cells are available to check {metric_label}.",
        )
        return
    value = int(pd.to_numeric(values, errors="coerce").min())
    _append_check(
        checks,
        test=test,
        flag="YELLOW" if value < threshold_value else "GREEN",
        value=value,
        threshold=f">= {threshold_value}",
        message=(
            f"Minimum complete {metric_label} per supported post cell is below the requested threshold."
            if value < threshold_value
            else f"Complete {metric_label} per supported post cell meets the requested threshold."
        ),
    )


def run_did_diagnostics(
    data: PanelDataDID,
    *,
    control_group: ComparisonGroup = "not_yet_or_never",
    anticipation: int = 0,
    base_period: BasePeriod = "universal",
    include_pre_periods: bool = False,
    min_treated_per_cell: int = 30,
    min_control_per_cell: int = 30,
    min_control_to_treated_ratio: float = 1.0,
    min_pair_completion_rate: float = 0.80,
    min_control_pool_retention: float = 0.25,
    max_unsupported_cell_share: float = 0.25,
    min_pre_periods: int = 2,
    max_abs_pretrend_t_stat: float = 2.0,
    max_abs_covariate_smd: float = 0.25,
    max_condition_number: float = _DEFAULT_MAX_CONDITION_NUMBER,
    min_clusters: int = 2,
) -> pd.DataFrame:
    """
    Run compact pre-fit diagnostics for Callaway & Sant'Anna estimation readiness.

    This function performs a battery of "smoke tests" on the data before fitting
    the model. It checks for sufficient sample size, parallel trends in pre-treatment
    periods, covariate balance, and numerical stability of the design.

    Parameters
    ----------
    data : PanelDataDID
        The validated panel data object.
    control_group : {"not_yet_or_never", "never_treated"}, default "not_yet_or_never"
        The definition of the comparison group.
    anticipation : int, default 0
        Anticipation periods to exclude.
    base_period : {"universal", "varying"}, default "universal"
        Base period policy.
    include_pre_periods : bool, default False
        Whether to include pre-treatment cells in the diagnostics.
    min_treated_per_cell : int, default 30
        Minimum number of treated units required in each ATT(g,t) cell.
    min_control_per_cell : int, default 30
        Minimum number of control units required in each ATT(g,t) cell.
    min_control_to_treated_ratio : float, default 1.0
        Minimum ratio of control units to treated units.
    min_pair_completion_rate : float, default 0.80
        Minimum share of units that must be observed in both base and target periods.
    min_control_pool_retention : float, default 0.25
        Minimum share of the original control pool that must be available for estimation.
    max_unsupported_cell_share : float, default 0.25
        Maximum allowable share of cohort-time cells that cannot be estimated.
    min_pre_periods : int, default 2
        Minimum number of pre-treatment periods required for placebo tests.
    max_abs_pretrend_t_stat : float, default 2.0
        Maximum absolute t-statistic allowed for raw pre-treatment differences.
    max_abs_covariate_smd : float, default 0.25
        Maximum allowable absolute SMD for covariates in the base period.
    max_condition_number : float, default 1e6
        Maximum allowable condition number for the control design matrix.
    min_clusters : int, default 2
        Minimum number of clusters required if clustering is used.

    Returns
    -------
    pd.DataFrame
        A diagnostic report with columns:
            - ``test``: Name of the diagnostic check.
            - ``flag``: Status (GREEN, YELLOW, RED).
            - ``value``: Observed value of the metric.
            - ``threshold``: The threshold used for the check.
            - ``message``: Descriptive result message.

    Examples
    --------
    >>> from causalis.scenarios.did import generate_did_gamma_26
    >>> from causalis.scenarios.did.refutation import run_did_diagnostics
    >>> data = generate_did_gamma_26(n_units=300, n_periods=5, seed=42)
    >>> report = run_did_diagnostics(data)
    >>> report[["test", "flag", "value"]]
    """

    data = _ensure_panel_data(data)
    control_group = _validate_control_group(control_group)
    anticipation = _validate_nonnegative_int(anticipation, "anticipation")
    base_period = _validate_base_period(base_period)
    min_treated_per_cell = _validate_positive_int(
        min_treated_per_cell,
        "min_treated_per_cell",
    )
    min_control_per_cell = _validate_positive_int(
        min_control_per_cell,
        "min_control_per_cell",
    )
    min_control_to_treated_ratio = _validate_positive_float(
        min_control_to_treated_ratio,
        "min_control_to_treated_ratio",
    )
    min_pair_completion_rate = _validate_probability(
        min_pair_completion_rate,
        "min_pair_completion_rate",
    )
    min_control_pool_retention = _validate_probability(
        min_control_pool_retention,
        "min_control_pool_retention",
    )
    max_unsupported_cell_share = _validate_probability(
        max_unsupported_cell_share,
        "max_unsupported_cell_share",
    )
    min_pre_periods = _validate_positive_int(min_pre_periods, "min_pre_periods")
    max_abs_pretrend_t_stat = _validate_positive_float(
        max_abs_pretrend_t_stat,
        "max_abs_pretrend_t_stat",
    )
    max_abs_covariate_smd = _validate_positive_float(
        max_abs_covariate_smd,
        "max_abs_covariate_smd",
    )
    max_condition_number = _validate_positive_float(
        max_condition_number,
        "max_condition_number",
    )
    min_clusters = _validate_positive_int(min_clusters, "min_clusters")

    support = did_support_table(
        data,
        control_group=control_group,
        anticipation=anticipation,
        base_period=base_period,
        include_pre_periods=include_pre_periods,
    )
    is_post = support["is_post_treatment"].fillna(False).astype(bool)
    post = support[is_post]
    supported_post = post[post["is_supported"].fillna(False).astype(bool)]

    checks: list[dict[str, Any]] = []
    n_supported_post = int(len(supported_post))
    _append_check(
        checks,
        test="requested_cs_post_support",
        flag="RED" if n_supported_post == 0 else "GREEN",
        value=n_supported_post,
        threshold=">= 1",
        message=(
            "No post-treatment ATT(g,t) cells are supported under the requested CS options."
            if n_supported_post == 0
            else "At least one post-treatment ATT(g,t) cell is supported under the requested CS options."
        ),
    )

    if post.empty:
        _append_check(
            checks,
            test="unsupported_post_cell_share",
            flag="YELLOW",
            value=None,
            threshold=f"<= {max_unsupported_cell_share:.3g}",
            message="No post-treatment support rows are available under the requested CS options.",
        )
    else:
        unsupported_share = float(1.0 - n_supported_post / len(post))
        _append_check(
            checks,
            test="unsupported_post_cell_share",
            flag=(
                "RED"
                if n_supported_post == 0
                else "YELLOW"
                if unsupported_share > max_unsupported_cell_share
                else "GREEN"
            ),
            value=unsupported_share,
            threshold=f"<= {max_unsupported_cell_share:.3g}",
            message=(
                "Too many requested post-treatment ATT(g,t) cells lack complete treated/control pairs."
                if unsupported_share > max_unsupported_cell_share
                else "Unsupported post-treatment cell share is within tolerance."
            ),
        )

    _append_min_count_check(
        checks,
        test="min_complete_treated_per_post_cell",
        values=supported_post["n_treated_complete"],
        threshold_value=min_treated_per_cell,
        metric_label="treated units",
    )
    _append_min_count_check(
        checks,
        test="min_complete_control_per_post_cell",
        values=supported_post["n_control_complete"],
        threshold_value=min_control_per_cell,
        metric_label="control units",
    )

    if supported_post.empty:
        _append_check(
            checks,
            test="min_control_to_treated_ratio",
            flag="RED",
            value=None,
            threshold=f">= {min_control_to_treated_ratio:.3g}",
            message="No supported post-treatment cells are available to check control-to-treated ratio.",
        )
        _append_check(
            checks,
            test="min_pair_completion_rate",
            flag="RED",
            value=None,
            threshold=f">= {min_pair_completion_rate:.3g}",
            message="No supported post-treatment cells are available to check complete-pair rates.",
        )
        _append_check(
            checks,
            test="control_pool_retention",
            flag="RED",
            value=None,
            threshold=f">= {min_control_pool_retention:.3g}",
            message="No supported post-treatment cells are available to check control-pool retention.",
        )
    else:
        min_ratio = _as_finite_float(supported_post["control_to_treated_ratio"].min())
        ratio_flag = (
            "YELLOW"
            if min_ratio is None or min_ratio < min_control_to_treated_ratio
            else "GREEN"
        )
        _append_check(
            checks,
            test="min_control_to_treated_ratio",
            flag=ratio_flag,
            value=min_ratio,
            threshold=f">= {min_control_to_treated_ratio:.3g}",
            message=(
                "At least one supported post cell has a thin comparison pool relative to its treated cohort."
                if ratio_flag != "GREEN"
                else "Control-to-treated ratios are within tolerance for supported post cells."
            ),
        )

        completion_values = pd.concat(
            [
                supported_post["treated_completion_rate"],
                supported_post["control_completion_rate"],
            ],
            ignore_index=True,
        )
        min_completion = _as_finite_float(
            pd.to_numeric(completion_values, errors="coerce").min()
        )
        completion_flag = (
            "YELLOW"
            if min_completion is None or min_completion < min_pair_completion_rate
            else "GREEN"
        )
        _append_check(
            checks,
            test="min_pair_completion_rate",
            flag=completion_flag,
            value=min_completion,
            threshold=f">= {min_pair_completion_rate:.3g}",
            message=(
                "At least one supported post cell loses many units between base and target periods."
                if completion_flag != "GREEN"
                else "Complete-pair rates are within tolerance for supported post cells."
            ),
        )

        control_counts = pd.to_numeric(
            supported_post["n_control_complete"],
            errors="coerce",
        )
        max_controls = _as_finite_float(control_counts.max())
        min_controls = _as_finite_float(control_counts.min())
        retention = (
            None
            if max_controls is None or min_controls is None or max_controls <= 0.0
            else float(min_controls / max_controls)
        )
        retention_flag = (
            "YELLOW"
            if retention is None or retention < min_control_pool_retention
            else "GREEN"
        )
        _append_check(
            checks,
            test="control_pool_retention",
            flag=retention_flag,
            value=retention,
            threshold=f">= {min_control_pool_retention:.3g}",
            message=(
                "The available comparison pool shrinks sharply across supported post cells."
                if retention_flag != "GREEN"
                else "Comparison-pool retention is within tolerance across supported post cells."
            ),
        )

    df = data.df_analysis()
    if data.cluster_col is None:
        _append_check(
            checks,
            test="cluster_readiness",
            flag="GREEN",
            value="not_requested",
            threshold="n/a",
            message="No cluster_col is set; CS inference will use unit-level influence scores.",
        )
    else:
        cluster_counts = df.groupby(data.unit_col, sort=False)[data.cluster_col].nunique(
            dropna=False
        )
        unstable_units = int((cluster_counts > 1).sum())
        n_clusters = int(df[data.cluster_col].nunique(dropna=False))
        if data.cluster_col != data.unit_col and unstable_units > 0:
            cluster_flag = "RED"
            cluster_message = (
                "cluster_col is not stable within unit, so clustered CS inference will fail."
            )
        elif n_clusters < min_clusters:
            cluster_flag = "RED"
            cluster_message = "cluster_col has too few clusters for clustered CS inference."
        else:
            cluster_flag = "GREEN"
            cluster_message = "cluster_col is stable within unit and has enough clusters."
        _append_check(
            checks,
            test="cluster_readiness",
            flag=cluster_flag,
            value={"n_clusters": n_clusters, "unstable_units": unstable_units},
            threshold=f"stable within unit and clusters >= {min_clusters}",
            message=cluster_message,
        )

    _append_check(
        checks,
        test="pre_period_depth",
        flag="YELLOW" if data.n_pre_periods < min_pre_periods else "GREEN",
        value=int(data.n_pre_periods),
        threshold=f">= {min_pre_periods}",
        message=(
            "Only one pre-period is contract-valid, but weak for pre-trend diagnostics."
            if data.n_pre_periods < min_pre_periods
            else "There are enough pre-periods for basic pre-trend diagnostics."
        ),
    )

    raw_event = raw_did_event_study_table(
        data,
        control_group=control_group,
        anticipation=anticipation,
        base_period="varying",
        include_pre_periods=True,
    )
    pre_event = raw_event[~raw_event["is_post_treatment"].fillna(False).astype(bool)]
    usable_t = pd.to_numeric(pre_event["t_stat"], errors="coerce").abs().dropna()
    if pre_event.empty:
        _append_check(
            checks,
            test="raw_pretrend_placebo",
            flag="YELLOW",
            value=None,
            threshold=f"max |t| <= {max_abs_pretrend_t_stat:.3g}",
            message="No raw pre-treatment placebo cells are estimable from the panel.",
        )
    elif usable_t.empty:
        _append_check(
            checks,
            test="raw_pretrend_placebo",
            flag="YELLOW",
            value=None,
            threshold=f"max |t| <= {max_abs_pretrend_t_stat:.3g}",
            message="Raw pre-treatment placebo cells exist, but t-statistics are undefined.",
        )
    else:
        max_abs_t = float(usable_t.max())
        _append_check(
            checks,
            test="raw_pretrend_placebo",
            flag="YELLOW" if max_abs_t > max_abs_pretrend_t_stat else "GREEN",
            value=max_abs_t,
            threshold=f"max |t| <= {max_abs_pretrend_t_stat:.3g}",
            message=(
                "Raw pre-treatment DID placebo cells show a large standardized deviation."
                if max_abs_t > max_abs_pretrend_t_stat
                else "Raw pre-treatment DID placebo cells are within the standardized threshold."
            ),
        )

    if not data.covariates:
        _append_check(
            checks,
            test="max_base_covariate_smd",
            flag="GREEN",
            value="no_covariates",
            threshold="n/a",
            message="No covariates were supplied; base-period covariate balance is not applicable.",
        )
    else:
        balance = did_covariate_balance_table(
            data,
            control_group=control_group,
            anticipation=anticipation,
            base_period=base_period,
            include_pre_periods=include_pre_periods,
            post_only=True,
        )
        finite_smd = pd.to_numeric(balance["abs_smd"], errors="coerce")
        finite_smd = finite_smd[np.isfinite(finite_smd)]
        if finite_smd.empty:
            _append_check(
                checks,
                test="max_base_covariate_smd",
                flag="YELLOW",
                value=None,
                threshold=f"<= {max_abs_covariate_smd:.3g}",
                message="No supported post-cell covariate balance rows are available.",
            )
        else:
            max_smd = float(finite_smd.max())
            _append_check(
                checks,
                test="max_base_covariate_smd",
                flag="YELLOW" if max_smd > max_abs_covariate_smd else "GREEN",
                value=max_smd,
                threshold=f"<= {max_abs_covariate_smd:.3g}",
                message=(
                    "At least one base-period covariate is imbalanced before CS weighting/regression."
                    if max_smd > max_abs_covariate_smd
                    else "Base-period covariate imbalance is within tolerance before CS fitting."
                ),
            )

    design = did_base_design_table(
        data,
        control_group=control_group,
        anticipation=anticipation,
        base_period=base_period,
        include_pre_periods=include_pre_periods,
        post_only=True,
    )
    if design.empty:
        _append_check(
            checks,
            test="base_control_design_rank",
            flag="RED",
            value=None,
            threshold="rank == n_parameters",
            message="No supported post-cell base control designs are available.",
        )
        _append_check(
            checks,
            test="base_control_design_condition",
            flag="RED",
            value=None,
            threshold=f"<= {max_condition_number:.3g}",
            message="No supported post-cell base control designs are available.",
        )
    else:
        deficient = int(design["is_rank_deficient"].sum())
        _append_check(
            checks,
            test="base_control_design_rank",
            flag="RED" if deficient > 0 else "GREEN",
            value=deficient,
            threshold="0 rank-deficient cells",
            message=(
                "At least one supported post-cell control design is rank-deficient."
                if deficient > 0
                else "Supported post-cell control designs have full column rank."
            ),
        )
        condition_values = pd.to_numeric(design["condition_number"], errors="coerce")
        max_condition = _as_finite_float(condition_values.max())
        condition_flag = (
            "YELLOW"
            if max_condition is None or max_condition > max_condition_number
            else "GREEN"
        )
        _append_check(
            checks,
            test="base_control_design_condition",
            flag=condition_flag,
            value=max_condition,
            threshold=f"<= {max_condition_number:.3g}",
            message=(
                "At least one supported post-cell control design is ill-conditioned."
                if condition_flag != "GREEN"
                else "Supported post-cell control-design condition numbers are within tolerance."
            ),
        )

    return pd.DataFrame(checks, columns=["test", "flag", "value", "threshold", "message"])


__all__ = [
    "did_support_table",
    "raw_did_event_study_table",
    "did_covariate_balance_table",
    "did_base_design_table",
    "run_did_diagnostics",
]
