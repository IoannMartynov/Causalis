from __future__ import annotations

from typing import Any, Dict, Hashable

import numpy as np
import pandas as pd

from causalis.data_contracts.panel_data_scm import PanelDataSCM
from causalis.data_contracts.panel_estimate import PanelEstimate
from causalis.scenarios.synthetic_control.model import ASCM


def _as_finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def _series_rmse(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return float(np.sqrt(np.mean(np.square(values))))


def _series_mean(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return float(np.mean(values))


def _series_max_abs(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return float(np.max(np.abs(values)))


def _rmspe_ratio(*, post_rmse: float | None, pre_rmse: float | None) -> float | None:
    if post_rmse is None or pre_rmse is None:
        return None
    if pre_rmse == 0.0:
        if post_rmse == 0.0:
            return None
        return float(np.inf)
    return float(post_rmse / pre_rmse)


def _rank_desc(values: pd.Series) -> np.ndarray:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    sort_key = np.where(np.isnan(arr), np.inf, -arr)
    order = np.argsort(sort_key, kind="mergesort")
    ranks = np.empty(arr.size, dtype=int)
    ranks[order] = np.arange(1, arr.size + 1, dtype=int)
    return ranks


def _validate_inputs(estimate: PanelEstimate, paneldata: PanelDataSCM) -> None:
    if not isinstance(estimate, PanelEstimate):
        raise TypeError("estimate must be a PanelEstimate instance.")
    if not isinstance(paneldata, PanelDataSCM):
        raise TypeError("paneldata must be a PanelDataSCM instance.")
    if estimate.treated_unit != paneldata.treated_unit:
        raise ValueError(
            "estimate.treated_unit must match paneldata.treated_unit "
            f"({estimate.treated_unit!r} != {paneldata.treated_unit!r})."
        )
    if estimate.treatment_start != paneldata.treatment_start:
        raise ValueError(
            "estimate.treatment_start must match paneldata.treatment_start "
            f"({estimate.treatment_start!r} != {paneldata.treatment_start!r})."
        )


def _build_placebo_panel(
    *,
    paneldata: PanelDataSCM,
    treated_unit: Hashable,
    treatment_start: Any,
) -> PanelDataSCM:
    df = paneldata.df_analysis().copy()
    df[paneldata.treated_time] = 0
    treated_mask = (df[paneldata.unit_col] == treated_unit) & (df[paneldata.time_col] >= treatment_start)
    df.loc[treated_mask, paneldata.treated_time] = 1
    return PanelDataSCM(
        df=df,
        y=paneldata.y,
        unit_col=paneldata.unit_col,
        time_col=paneldata.time_col,
        treated_time=paneldata.treated_time,
    )


def _fit_placebo_estimate(
    *,
    paneldata: PanelDataSCM,
    treated_unit: Hashable,
    treatment_start: Any,
    model_kwargs: Dict[str, Any] | None,
) -> PanelEstimate:
    placebo_panel = _build_placebo_panel(
        paneldata=paneldata,
        treated_unit=treated_unit,
        treatment_start=treatment_start,
    )
    model = ASCM(**(model_kwargs or {}))
    return model.fit(placebo_panel).estimate()


def _extract_average_att(placebo_estimate: PanelEstimate) -> float | None:
    diagnostics = dict(placebo_estimate.diagnostics or {})
    avg = _as_finite_float(diagnostics.get("average_att_estimate"))
    if avg is not None:
        return avg
    return _series_mean(placebo_estimate.effect_by_time)


def _extract_pre_fit_metric(placebo_estimate: PanelEstimate) -> float | None:
    diagnostics = dict(placebo_estimate.diagnostics or {})
    pre_fit = _as_finite_float(diagnostics.get("pre_rmse_augmented"))
    if pre_fit is not None:
        return pre_fit
    gap = placebo_estimate.observed_outcome - placebo_estimate.synthetic_outcome
    return _series_rmse(gap.loc[list(placebo_estimate.pre_times)])


def placebo_in_space_table(
    estimate: PanelEstimate,
    paneldata: PanelDataSCM,
    *,
    model_kwargs: Dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Build Abadie-style placebo-in-space RMSPE ratio table."""
    _validate_inputs(estimate, paneldata)

    df = paneldata.df_analysis().copy()
    units = pd.Index(df[paneldata.unit_col].unique()).tolist()
    rows: list[dict[str, Any]] = []
    for unit_id in units:
        if unit_id == estimate.treated_unit:
            unit_estimate = estimate
        else:
            unit_estimate = _fit_placebo_estimate(
                paneldata=paneldata,
                treated_unit=unit_id,
                treatment_start=paneldata.treatment_start,
                model_kwargs=model_kwargs,
            )
        gap = unit_estimate.observed_outcome - unit_estimate.synthetic_outcome
        pre_gap = gap.loc[list(unit_estimate.pre_times)]
        post_gap = gap.loc[list(unit_estimate.post_times)]

        pre_rmse = _series_rmse(pre_gap)
        post_rmse = _series_rmse(post_gap)
        rows.append(
            {
                "unit_id": unit_id,
                "pre_rmse": pre_rmse,
                "post_rmse": post_rmse,
                "post_pre_rmspe_ratio": _rmspe_ratio(post_rmse=post_rmse, pre_rmse=pre_rmse),
                "average_post_gap": _series_mean(post_gap),
                "max_abs_post_gap": _series_max_abs(post_gap),
                "is_actual_treated": bool(unit_id == paneldata.treated_unit),
            }
        )

    table = pd.DataFrame(rows)
    if table.empty:
        return pd.DataFrame(
            columns=[
                "unit_id",
                "pre_rmse",
                "post_rmse",
                "post_pre_rmspe_ratio",
                "average_post_gap",
                "max_abs_post_gap",
                "rank_post_pre_rmspe_ratio",
                "is_actual_treated",
            ]
        )

    table["rank_post_pre_rmspe_ratio"] = _rank_desc(table["post_pre_rmspe_ratio"]).astype(int)
    cols = [
        "unit_id",
        "pre_rmse",
        "post_rmse",
        "post_pre_rmspe_ratio",
        "average_post_gap",
        "max_abs_post_gap",
        "rank_post_pre_rmspe_ratio",
        "is_actual_treated",
    ]
    return table.loc[:, cols].sort_values("rank_post_pre_rmspe_ratio", kind="mergesort").reset_index(
        drop=True
    )


def placebo_in_time_table(
    estimate: PanelEstimate,
    paneldata: PanelDataSCM,
    *,
    model_kwargs: Dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Build CWZ-style placebo-in-time structural-break table."""
    _validate_inputs(estimate, paneldata)

    pre_times = list(paneldata.pre_times())
    placebo_starts = pre_times[1:]
    rows: list[dict[str, Any]] = []
    for placebo_start in placebo_starts:
        placebo_estimate = _fit_placebo_estimate(
            paneldata=paneldata,
            treated_unit=paneldata.treated_unit,
            treatment_start=placebo_start,
            model_kwargs=model_kwargs,
        )
        diagnostics = dict(placebo_estimate.diagnostics or {})
        p_value = _as_finite_float(diagnostics.get("average_att_p_value"))
        alpha = float(placebo_estimate.alpha)
        rows.append(
            {
                "placebo_treatment_start": placebo_start,
                "n_pre_before_placebo": int(len(placebo_estimate.pre_times)),
                "n_post_after_placebo": int(len(placebo_estimate.post_times)),
                "average_att_placebo": _extract_average_att(placebo_estimate),
                "ci_lower": _as_finite_float(diagnostics.get("average_att_ci_lower")),
                "ci_upper": _as_finite_float(diagnostics.get("average_att_ci_upper")),
                "p_value": p_value,
                "rejects_zero": bool(p_value < alpha) if p_value is not None else False,
                "pre_fit_metric": _extract_pre_fit_metric(placebo_estimate),
            }
        )

    table = pd.DataFrame(rows)
    if table.empty:
        return pd.DataFrame(
            columns=[
                "placebo_treatment_start",
                "n_pre_before_placebo",
                "n_post_after_placebo",
                "average_att_placebo",
                "ci_lower",
                "ci_upper",
                "p_value",
                "rejects_zero",
                "pre_fit_metric",
            ]
        )

    cols = [
        "placebo_treatment_start",
        "n_pre_before_placebo",
        "n_post_after_placebo",
        "average_att_placebo",
        "ci_lower",
        "ci_upper",
        "p_value",
        "rejects_zero",
        "pre_fit_metric",
    ]
    return table.loc[:, cols].sort_values("placebo_treatment_start", kind="mergesort").reset_index(
        drop=True
    )


def run_placebo_tests(
    estimate: PanelEstimate,
    paneldata: PanelDataSCM,
    *,
    model_kwargs: Dict[str, Any] | None = None,
) -> Dict[str, pd.DataFrame]:
    """Run placebo-in-space and placebo-in-time robustness tests."""
    _validate_inputs(estimate, paneldata)
    return {
        "placebo_in_space": placebo_in_space_table(
            estimate=estimate,
            paneldata=paneldata,
            model_kwargs=model_kwargs,
        ),
        "placebo_in_time": placebo_in_time_table(
            estimate=estimate,
            paneldata=paneldata,
            model_kwargs=model_kwargs,
        ),
    }


__all__ = [
    "placebo_in_space_table",
    "placebo_in_time_table",
    "run_placebo_tests",
]
