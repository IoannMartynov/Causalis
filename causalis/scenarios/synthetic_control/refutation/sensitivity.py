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


def _series_mean(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return float(np.mean(values))


def _series_rmse(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return float(np.sqrt(np.mean(np.square(values))))


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


def _extract_average_att(estimate: PanelEstimate) -> float | None:
    diagnostics = dict(estimate.diagnostics or {})
    avg = _as_finite_float(diagnostics.get("average_att_estimate"))
    if avg is not None:
        return avg
    return _series_mean(estimate.effect_by_time)


def _extract_pre_rmse(estimate: PanelEstimate) -> float | None:
    diagnostics = dict(estimate.diagnostics or {})
    pre_rmse = _as_finite_float(diagnostics.get("pre_rmse_augmented"))
    if pre_rmse is not None:
        return pre_rmse
    gap = estimate.observed_outcome - estimate.synthetic_outcome
    return _series_rmse(gap.loc[list(estimate.pre_times)])


def _extract_max_weight(estimate: PanelEstimate) -> float | None:
    diagnostics = dict(estimate.diagnostics or {})
    max_w = _as_finite_float(diagnostics.get("max_weight_augmented"))
    if max_w is not None:
        return max_w

    weights = np.asarray(list(estimate.donor_weights_augmented.values()), dtype=float)
    weights = weights[np.isfinite(weights)]
    if weights.size == 0:
        return None
    return float(np.max(weights))


def _effective_n_donors(estimate: PanelEstimate) -> float | None:
    weights = np.asarray(list(estimate.donor_weights_augmented.values()), dtype=float)
    weights = weights[np.isfinite(weights)]
    if weights.size == 0:
        return None
    herfindahl = float(np.sum(np.square(weights)))
    if herfindahl <= 0.0:
        return None
    return float(1.0 / herfindahl)


def _build_panel_without_donor(paneldata: PanelDataSCM, dropped_donor: Hashable) -> PanelDataSCM:
    df = paneldata.df_analysis().copy()
    df = df[df[paneldata.unit_col] != dropped_donor].copy()
    return PanelDataSCM(
        df=df,
        y=paneldata.y,
        unit_col=paneldata.unit_col,
        time_col=paneldata.time_col,
        treated_time=paneldata.treated_time,
    )


def leave_one_donor_out_sensitivity(
    estimate: PanelEstimate,
    paneldata: PanelDataSCM,
    *,
    model_kwargs: Dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Re-fit ASCM dropping each donor once and report leave-one-donor-out sensitivity."""
    _validate_inputs(estimate, paneldata)

    full_model_average_att = _extract_average_att(estimate)
    donors = list(paneldata.donor_pool())

    rows: list[dict[str, Any]] = []
    for donor in donors:
        row: dict[str, Any] = {
            "dropped_donor": donor,
            "average_att_reestimated": np.nan,
            "delta_vs_full_model": np.nan,
            "pre_rmse_reestimated": np.nan,
            "max_weight_after_refit": np.nan,
            "effective_n_donors_after_refit": np.nan,
        }
        try:
            reduced_panel = _build_panel_without_donor(paneldata, dropped_donor=donor)
            refit_estimate = ASCM(**(model_kwargs or {})).fit(reduced_panel).estimate()

            att_reestimated = _extract_average_att(refit_estimate)
            row["average_att_reestimated"] = att_reestimated
            row["delta_vs_full_model"] = (
                np.nan
                if att_reestimated is None or full_model_average_att is None
                else float(att_reestimated - full_model_average_att)
            )
            row["pre_rmse_reestimated"] = _extract_pre_rmse(refit_estimate)
            row["max_weight_after_refit"] = _extract_max_weight(refit_estimate)
            row["effective_n_donors_after_refit"] = _effective_n_donors(refit_estimate)
        except Exception:
            pass
        rows.append(row)

    table = pd.DataFrame(rows)
    if table.empty:
        return pd.DataFrame(
            columns=[
                "dropped_donor",
                "average_att_reestimated",
                "delta_vs_full_model",
                "pre_rmse_reestimated",
                "max_weight_after_refit",
                "effective_n_donors_after_refit",
            ]
        )

    cols = [
        "dropped_donor",
        "average_att_reestimated",
        "delta_vs_full_model",
        "pre_rmse_reestimated",
        "max_weight_after_refit",
        "effective_n_donors_after_refit",
    ]
    return table.loc[:, cols].reset_index(drop=True)


__all__ = ["leave_one_donor_out_sensitivity"]
