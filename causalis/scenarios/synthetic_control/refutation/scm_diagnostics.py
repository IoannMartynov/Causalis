from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from causalis.data_contracts.panel_data_scm import PanelDataSCM
from causalis.data_contracts.panel_estimate import PanelEstimate


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


def _series_rmse(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return float(np.sqrt(np.mean(np.square(values))))


def _series_mae(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return float(np.mean(np.abs(values)))


def _series_max_abs(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return float(np.max(np.abs(values)))


def _series_mean_signed(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return float(np.mean(values))


def _coerce_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        if value in (0, 1):
            return bool(value)
        return None
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"true", "t", "1", "yes", "y"}:
            return True
        if token in {"false", "f", "0", "no", "n"}:
            return False
    return None


def _coerce_string_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        out: list[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                out.append(text)
        return out
    return []


def _extract_placebo_atts(diagnostics: Dict[str, Any]) -> np.ndarray:
    raw = diagnostics.get("att_placebo_att_distribution")
    if raw is None:
        return np.asarray([], dtype=float)

    values = pd.to_numeric(pd.Series(list(raw)), errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    return values


def _optional_series_attr(estimate: PanelEstimate, attr_name: str) -> pd.Series | None:
    value = getattr(estimate, attr_name, None)
    return value if isinstance(value, pd.Series) else None


def _resolve_att_aug(estimate: PanelEstimate, diagnostics: Dict[str, Any]) -> float | None:
    for value in (
        getattr(estimate, "att", None),
        diagnostics.get("att"),
        diagnostics.get("average_att_estimate"),
    ):
        resolved = _as_finite_float(value)
        if resolved is not None:
            return resolved

    effect_series = pd.to_numeric(estimate.effect_by_time, errors="coerce")
    post_idx = effect_series.index.intersection(pd.Index(list(estimate.post_times)))
    effect_vals = effect_series.loc[post_idx].to_numpy(dtype=float)
    effect_vals = effect_vals[np.isfinite(effect_vals)]
    if effect_vals.size == 0:
        return None
    return float(np.mean(effect_vals))


def _resolve_att_sc(estimate: PanelEstimate, diagnostics: Dict[str, Any]) -> float | None:
    for value in (
        getattr(estimate, "att_sc", None),
        diagnostics.get("att_sc"),
    ):
        resolved = _as_finite_float(value)
        if resolved is not None:
            return resolved

    synthetic_sc = _optional_series_attr(estimate, "synthetic_outcome_sc")
    if synthetic_sc is None:
        return None

    observed = estimate.observed_outcome
    gap_sc = observed - synthetic_sc
    post_idx = gap_sc.index.intersection(pd.Index(list(estimate.post_times)))
    post_vals = pd.to_numeric(gap_sc.loc[post_idx], errors="coerce").to_numpy(dtype=float)
    post_vals = post_vals[np.isfinite(post_vals)]
    if post_vals.size == 0:
        return None
    return float(np.mean(post_vals))


def _missing_cell_fraction(paneldata: PanelDataSCM) -> float:
    df = paneldata.df_analysis().copy()
    if df.empty:
        return 0.0

    unit_idx = pd.Index(df[paneldata.unit_col].unique())
    time_idx = pd.Index(paneldata.analysis_times())
    full_grid = pd.MultiIndex.from_product(
        [unit_idx, time_idx],
        names=[paneldata.unit_col, paneldata.time_col],
    )
    y_full = (
        df.set_index([paneldata.unit_col, paneldata.time_col])[paneldata.y]
        .reindex(full_grid)
        .reset_index(drop=True)
    )
    missing = y_full.isna()
    return float(missing.mean())


def run_scm_diagnostics(
    estimate: PanelEstimate,
    paneldata: PanelDataSCM,
    *,
    output_dir: str | Path | None = None,
    filename_prefix: str = "scm_diagnostics",
    pre_tail_k: int = 3,
    dpi: int = 220,
) -> Dict[str, Any]:
    """Run compact SCM diagnostics and save the three v1 diagnostic plots."""
    if not isinstance(estimate, PanelEstimate):
        raise TypeError("estimate must be a PanelEstimate instance.")
    if not isinstance(paneldata, PanelDataSCM):
        raise TypeError("paneldata must be a PanelDataSCM instance.")
    if int(pre_tail_k) <= 0:
        raise ValueError("pre_tail_k must be a positive integer.")
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

    diagnostics = dict(estimate.diagnostics or {})
    _ = output_dir, filename_prefix, dpi  # plotting is handled by diagnostic_plots.py

    observed = estimate.observed_outcome.copy()
    synthetic_aug = estimate.synthetic_outcome.copy()
    synthetic_sc = _optional_series_attr(estimate, "synthetic_outcome_sc")
    gap_aug = observed - synthetic_aug
    gap_sc = (observed - synthetic_sc) if synthetic_sc is not None else None

    pre_times = list(estimate.pre_times)
    pre_idx = gap_aug.index.intersection(pd.Index(pre_times))
    k_eff = min(int(pre_tail_k), len(pre_times))
    if k_eff > 0:
        tail_pre_times = pre_times[-k_eff:]
        tail_pre_idx = gap_aug.index.intersection(pd.Index(tail_pre_times))
        mean_gap_last_k_pre_aug = _as_finite_float(gap_aug.loc[tail_pre_idx].mean())
        mean_gap_last_k_pre_sc = (
            _as_finite_float(gap_sc.loc[tail_pre_idx].mean()) if gap_sc is not None else None
        )
    else:
        mean_gap_last_k_pre_aug = None
        mean_gap_last_k_pre_sc = None

    pre_gap_aug = gap_aug.loc[pre_idx]
    pre_gap_sc = gap_sc.loc[pre_idx] if gap_sc is not None else None

    pre_rmse_sc = _as_finite_float(diagnostics.get("pre_rmse_sc"))
    if pre_rmse_sc is None:
        pre_rmse_sc = _as_finite_float(diagnostics.get("pre_rmse_scm"))
    if pre_rmse_sc is None and pre_gap_sc is not None:
        pre_rmse_sc = _series_rmse(pre_gap_sc)

    pre_rmse_aug = _as_finite_float(diagnostics.get("pre_rmse_augmented"))
    if pre_rmse_aug is None:
        pre_rmse_aug = _series_rmse(pre_gap_aug)
    pre_mae_aug = _as_finite_float(diagnostics.get("pre_mae_augmented"))
    if pre_mae_aug is None:
        pre_mae_aug = _series_mae(pre_gap_aug)
    max_abs_pre_gap_augmented = _as_finite_float(diagnostics.get("max_abs_pre_gap_augmented"))
    if max_abs_pre_gap_augmented is None:
        max_abs_pre_gap_augmented = _as_finite_float(diagnostics.get("max_abs_pre_gap"))
    if max_abs_pre_gap_augmented is None:
        max_abs_pre_gap_augmented = _series_max_abs(pre_gap_aug)
    mean_signed_pre_gap_augmented = _as_finite_float(diagnostics.get("mean_signed_pre_gap_augmented"))
    if mean_signed_pre_gap_augmented is None:
        mean_signed_pre_gap_augmented = _as_finite_float(diagnostics.get("mean_signed_pre_gap"))
    if mean_signed_pre_gap_augmented is None:
        mean_signed_pre_gap_augmented = _series_mean_signed(pre_gap_aug)

    donor_weights_sc = getattr(estimate, "donor_weights_sc", None)
    w_sc = (
        np.asarray(list(donor_weights_sc.values()), dtype=float)
        if isinstance(donor_weights_sc, dict)
        else np.asarray([], dtype=float)
    )
    w_aug = np.asarray(list(estimate.donor_weights_augmented.values()), dtype=float)
    w_aug_finite = w_aug[np.isfinite(w_aug)]
    max_weight_sc = _as_finite_float(diagnostics.get("max_weight_sc"))
    if max_weight_sc is None and w_sc.size > 0:
        max_weight_sc = float(np.max(w_sc))
    max_abs_weight_aug = _as_finite_float(diagnostics.get("max_abs_weight_augmented"))
    if max_abs_weight_aug is None and w_aug_finite.size > 0:
        max_abs_weight_aug = float(np.max(np.abs(w_aug_finite)))
    l1_norm_weight_aug = _as_finite_float(diagnostics.get("l1_norm_weights_augmented"))
    if l1_norm_weight_aug is None and w_aug_finite.size > 0:
        l1_norm_weight_aug = float(np.sum(np.abs(w_aug_finite)))
    sum_weights_augmented = _as_finite_float(diagnostics.get("sum_weights_augmented"))
    if sum_weights_augmented is None and w_aug_finite.size > 0:
        sum_weights_augmented = float(np.sum(w_aug_finite))

    w_sc_finite = w_sc[np.isfinite(w_sc)]
    herfindahl_weights_sc = float(np.sum(np.square(w_sc_finite))) if w_sc_finite.size > 0 else None
    effective_n_donors_sc = (
        float(1.0 / herfindahl_weights_sc)
        if herfindahl_weights_sc is not None and herfindahl_weights_sc > 0.0
        else None
    )

    abs_sum_aug = float(np.sum(np.abs(w_aug_finite))) if w_aug_finite.size > 0 else None
    if abs_sum_aug is not None and abs_sum_aug > 0.0:
        w_aug_abs_norm = np.abs(w_aug_finite) / abs_sum_aug
        herfindahl_abs_augmented = float(np.sum(np.square(w_aug_abs_norm)))
        effective_n_donors_abs_augmented = (
            float(1.0 / herfindahl_abs_augmented) if herfindahl_abs_augmented > 0.0 else None
        )
    else:
        herfindahl_abs_augmented = None
        effective_n_donors_abs_augmented = None

    n_negative_weights = int(np.sum(w_aug_finite < 0.0))
    if l1_norm_weight_aug is None or l1_norm_weight_aug <= 0.0:
        negative_weight_share = None
    else:
        negative_weight_share = float(
            np.sum(np.abs(w_aug_finite[w_aug_finite < 0.0])) / l1_norm_weight_aug
        )

    placebo_atts = _extract_placebo_atts(diagnostics)
    n_placebos = diagnostics.get("att_placebo_n")
    try:
        n_placebos_eff = int(n_placebos) if n_placebos is not None else int(placebo_atts.size)
    except (TypeError, ValueError):
        n_placebos_eff = int(placebo_atts.size)
    if n_placebos_eff < 0:
        n_placebos_eff = int(placebo_atts.size)

    min_possible_p = _as_finite_float(diagnostics.get("att_placebo_min_possible_p"))
    if min_possible_p is None:
        min_possible_p = _as_finite_float(diagnostics.get("pointwise_min_possible_p_value"))
    if min_possible_p is None and n_placebos_eff > 0:
        min_possible_p = float(1.0 / float(n_placebos_eff + 1))

    p_value_att = _as_finite_float(diagnostics.get("att_placebo_p_value"))
    if p_value_att is None:
        p_value_att = _as_finite_float(diagnostics.get("average_att_p_value"))

    ci_low_abs = _as_finite_float(diagnostics.get("att_placebo_ci_lower_absolute"))
    if ci_low_abs is None:
        ci_low_abs = _as_finite_float(diagnostics.get("average_att_ci_lower"))
    ci_high_abs = _as_finite_float(diagnostics.get("att_placebo_ci_upper_absolute"))
    if ci_high_abs is None:
        ci_high_abs = _as_finite_float(diagnostics.get("average_att_ci_upper"))

    placebo_ci_is_unbounded_raw = diagnostics.get("att_placebo_ci_is_unbounded")
    placebo_ci_is_unbounded = _coerce_bool(placebo_ci_is_unbounded_raw)
    if placebo_ci_is_unbounded is None:
        placebo_ci_is_unbounded = False

    is_robust_model = str(estimate.model) == "RobustSyntheticControl"
    missing_cell_fraction = _as_finite_float(diagnostics.get("missing_cell_fraction"))
    if missing_cell_fraction is None and is_robust_model:
        missing_cell_fraction = _missing_cell_fraction(paneldata)

    completion_converged_raw = diagnostics.get("completion_converged")
    completion_converged = _coerce_bool(completion_converged_raw)
    completion_effective_rank = diagnostics.get("completion_effective_rank")
    if completion_effective_rank is not None:
        try:
            completion_effective_rank = int(completion_effective_rank)
        except (TypeError, ValueError):
            completion_effective_rank = None

    treated_att_aug = _resolve_att_aug(estimate, diagnostics)
    if treated_att_aug is None:
        raise ValueError("Unable to resolve augmented ATT for diagnostics metrics.")

    slsqp_fallback_reasons = _coerce_string_list(diagnostics.get("slsqp_fallback_reasons"))
    slsqp_fallback_count_raw = diagnostics.get("slsqp_fallback_count")
    try:
        slsqp_fallback_count = int(slsqp_fallback_count_raw)
    except (TypeError, ValueError):
        slsqp_fallback_count = len(slsqp_fallback_reasons)
    slsqp_fallback_count = max(0, slsqp_fallback_count)

    suppressed_fit_warnings = _coerce_string_list(diagnostics.get("suppressed_fit_warnings"))
    if not suppressed_fit_warnings:
        suppressed_fit_warnings = _coerce_string_list(diagnostics.get("stability_warning_messages"))

    metrics: Dict[str, Any] = {
        "n_donors": int(len(estimate.donor_weights_augmented)),
        "n_pre": int(len(estimate.pre_times)),
        "n_post": int(len(estimate.post_times)),
        "pre_rmse_sc": pre_rmse_sc,
        "pre_rmse_aug": pre_rmse_aug,
        "pre_rmse_augmented": pre_rmse_aug,
        "pre_mae_augmented": pre_mae_aug,
        "max_abs_pre_gap": max_abs_pre_gap_augmented,
        "max_abs_pre_gap_augmented": max_abs_pre_gap_augmented,
        "mean_signed_pre_gap": mean_signed_pre_gap_augmented,
        "mean_signed_pre_gap_augmented": mean_signed_pre_gap_augmented,
        "att_sc": _resolve_att_sc(estimate, diagnostics),
        "att_aug": treated_att_aug,
        "max_weight_sc": max_weight_sc,
        "max_abs_weight_aug": max_abs_weight_aug,
        "max_abs_weight_augmented": max_abs_weight_aug,
        "l1_norm_weight_aug": l1_norm_weight_aug,
        "l1_norm_weights_augmented": l1_norm_weight_aug,
        "sum_weights_augmented": sum_weights_augmented,
        "effective_n_donors": effective_n_donors_sc,
        "effective_n_donors_sc": effective_n_donors_sc,
        "herfindahl_weights": herfindahl_weights_sc,
        "herfindahl_weights_sc": herfindahl_weights_sc,
        "effective_n_donors_abs_augmented": effective_n_donors_abs_augmented,
        "herfindahl_abs_augmented": herfindahl_abs_augmented,
        "negative_weight_share": negative_weight_share,
        "n_negative_weights": n_negative_weights,
        "cond_augmented_gram": _as_finite_float(diagnostics.get("cond_augmented_gram")),
        "n_placebos": n_placebos_eff,
        "min_possible_p": min_possible_p,
        "p_value_att": p_value_att,
        "ci_low_abs": ci_low_abs,
        "ci_high_abs": ci_high_abs,
        "placebo_ci_is_unbounded": placebo_ci_is_unbounded,
        "missing_cell_fraction": missing_cell_fraction,
        "completion_converged": completion_converged,
        "completion_effective_rank": completion_effective_rank,
        "mean_gap_last_k_pre_sc": mean_gap_last_k_pre_sc,
        "mean_gap_last_k_pre_aug": mean_gap_last_k_pre_aug,
        "pre_tail_k_used": int(k_eff),
        "slsqp_fallback_count": slsqp_fallback_count,
        "slsqp_fallback_reasons": slsqp_fallback_reasons,
        "suppressed_fit_warning_count": int(len(suppressed_fit_warnings)),
        "suppressed_fit_warnings": list(suppressed_fit_warnings),
    }

    return {"metrics": metrics, "warnings": list(suppressed_fit_warnings)}


__all__ = ["run_scm_diagnostics"]
