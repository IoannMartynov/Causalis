from __future__ import annotations

from pathlib import Path
from typing import Any

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


def _series_max_abs(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return float(np.max(np.abs(values)))


def _pre_variability_scale(values: np.ndarray, *, tol: float = 1e-10) -> float | None:
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return None

    scales: list[float] = []

    std = float(np.std(arr, ddof=1))
    if np.isfinite(std) and std > tol:
        scales.append(std)

    median = float(np.median(arr))
    mad = float(1.4826 * np.median(np.abs(arr - median)))
    if np.isfinite(mad) and mad > tol:
        scales.append(mad)

    q25, q75 = np.percentile(arr, [25.0, 75.0])
    iqr_scale = float((q75 - q25) / 1.349)  # Gaussian-consistent IQR scale.
    if np.isfinite(iqr_scale) and iqr_scale > tol:
        scales.append(iqr_scale)

    diffs = np.diff(arr)
    if diffs.size == 1:
        diff_scale = float(abs(diffs[0]) / np.sqrt(2.0))
    elif diffs.size > 1:
        diff_scale = float(np.std(diffs, ddof=1) / np.sqrt(2.0))
    else:
        diff_scale = np.nan
    if np.isfinite(diff_scale) and diff_scale > tol:
        scales.append(diff_scale)

    if not scales:
        return None
    return float(max(scales))


def _coerce_string_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        out: list[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                out.append(text)
        return out
    return []


def run_scm_diagnostics(
    estimate: PanelEstimate,
    paneldata: PanelDataSCM,
    *,
    output_dir: str | Path | None = None,
    filename_prefix: str = "scm_diagnostics",
    pre_tail_k: int = 3,
    dpi: int = 220,
) -> pd.DataFrame:
    """Run compact SCM diagnostics and return thresholded checks."""
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
    gap_aug = observed - synthetic_aug

    pre_times = list(estimate.pre_times)
    pre_idx = gap_aug.index.intersection(pd.Index(pre_times))
    k_eff = min(int(pre_tail_k), len(pre_times))
    if k_eff > 0:
        tail_pre_times = pre_times[-k_eff:]
        tail_pre_idx = gap_aug.index.intersection(pd.Index(tail_pre_times))
        mean_gap_last_k_pre_aug = _as_finite_float(gap_aug.loc[tail_pre_idx].mean())
    else:
        mean_gap_last_k_pre_aug = None

    pre_gap_aug = gap_aug.loc[pre_idx]

    pre_rmse_aug = _as_finite_float(diagnostics.get("pre_rmse_augmented"))
    if pre_rmse_aug is None:
        pre_rmse_aug = _series_rmse(pre_gap_aug)
    max_abs_pre_gap_augmented = _as_finite_float(diagnostics.get("max_abs_pre_gap_augmented"))
    if max_abs_pre_gap_augmented is None:
        max_abs_pre_gap_augmented = _as_finite_float(diagnostics.get("max_abs_pre_gap"))
    if max_abs_pre_gap_augmented is None:
        max_abs_pre_gap_augmented = _series_max_abs(pre_gap_aug)

    w_aug = np.asarray(list(estimate.donor_weights_augmented.values()), dtype=float)
    w_aug_finite = w_aug[np.isfinite(w_aug)]
    max_abs_weight_aug = _as_finite_float(diagnostics.get("max_abs_weight_augmented"))
    if max_abs_weight_aug is None and w_aug_finite.size > 0:
        max_abs_weight_aug = float(np.max(np.abs(w_aug_finite)))
    l1_norm_weight_aug = _as_finite_float(diagnostics.get("l1_norm_weights_augmented"))
    if l1_norm_weight_aug is None and w_aug_finite.size > 0:
        l1_norm_weight_aug = float(np.sum(np.abs(w_aug_finite)))
    weight_max_abs_warn_threshold = _as_finite_float(
        diagnostics.get("augmented_weight_max_abs_warn_threshold")
    )
    if weight_max_abs_warn_threshold is None or weight_max_abs_warn_threshold <= 0.0:
        weight_max_abs_warn_threshold = 2.0
    weight_l1_warn_threshold = _as_finite_float(
        diagnostics.get("augmented_weight_l1_warn_threshold")
    )
    if weight_l1_warn_threshold is None or weight_l1_warn_threshold <= 0.0:
        weight_l1_warn_threshold = 5.0

    if l1_norm_weight_aug is None or l1_norm_weight_aug <= 0.0:
        negative_weight_share = None
    else:
        negative_weight_share = float(
            np.sum(np.abs(w_aug_finite[w_aug_finite < 0.0])) / l1_norm_weight_aug
        )

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

    pre_observed_values = pd.to_numeric(observed.loc[pre_idx], errors="coerce").to_numpy(dtype=float)
    pre_scale = _pre_variability_scale(pre_observed_values)

    checks: list[dict[str, Any]] = []

    def _append_check(
        *,
        test: str,
        value: Any,
        threshold_value: float,
        mode: str,
        message_ok: str,
        message_flagged: str,
        fail_severity: str = "YELLOW",
        use_abs: bool = False,
    ) -> None:
        threshold = float(threshold_value)
        value_numeric = _as_finite_float(value)
        if mode == "le":
            threshold_label = f"<= {threshold:.4g}"
        elif mode == "ge":
            threshold_label = f">= {threshold:.4g}"
        elif mode == "eq":
            threshold_label = f"== {threshold:.4g}"
        else:
            raise ValueError(f"Unsupported comparison mode: {mode}")

        if value_numeric is None:
            checks.append(
                {
                    "test": test,
                    "flag": "YELLOW",
                    "value": value,
                    "threshold": "n/a",
                    "message": "Skipped: metric unavailable for this estimate.",
                }
            )
            return

        value_for_check = abs(value_numeric) if use_abs else value_numeric
        if mode == "le":
            flagged = bool(value_for_check > threshold)
        elif mode == "ge":
            flagged = bool(value_for_check < threshold)
        else:
            flagged = bool(abs(value_for_check - threshold) > 1e-12)

        checks.append(
            {
                "test": test,
                "flag": fail_severity if flagged else "GREEN",
                "value": value,
                "threshold": threshold_label,
                "message": message_flagged if flagged else message_ok,
            }
        )

    if pre_scale is None:
        checks.extend(
            [
                {
                    "test": "pre_rmse_augmented",
                    "flag": "YELLOW",
                    "value": pre_rmse_aug,
                    "threshold": "n/a",
                    "message": "Skipped: pre-period variability is near zero; scale-based threshold is undefined.",
                },
                {
                    "test": "max_abs_pre_gap_augmented",
                    "flag": "YELLOW",
                    "value": max_abs_pre_gap_augmented,
                    "threshold": "n/a",
                    "message": "Skipped: pre-period variability is near zero; scale-based threshold is undefined.",
                },
                {
                    "test": "mean_gap_last_k_pre_augmented",
                    "flag": "YELLOW",
                    "value": mean_gap_last_k_pre_aug,
                    "threshold": "n/a",
                    "message": "Skipped: pre-period variability is near zero; scale-based threshold is undefined.",
                },
            ]
        )
    else:
        _append_check(
            test="pre_rmse_augmented",
            value=pre_rmse_aug,
            threshold_value=0.20 * pre_scale,
            mode="le",
            message_ok="Pre-treatment RMSE is within tolerance.",
            message_flagged="Pre-treatment RMSE is high relative to pre-period volatility.",
        )
        _append_check(
            test="max_abs_pre_gap_augmented",
            value=max_abs_pre_gap_augmented,
            threshold_value=0.50 * pre_scale,
            mode="le",
            message_ok="Largest pre-treatment gap is within tolerance.",
            message_flagged="Largest pre-treatment gap is too large.",
        )
        _append_check(
            test="mean_gap_last_k_pre_augmented",
            value=mean_gap_last_k_pre_aug,
            threshold_value=0.25 * pre_scale,
            mode="le",
            use_abs=True,
            message_ok=f"Average gap in the last {k_eff} pre periods is centered near zero.",
            message_flagged=f"Average gap in the last {k_eff} pre periods indicates pre-trend drift.",
        )
    _append_check(
        test="max_abs_weight_augmented",
        value=max_abs_weight_aug,
        threshold_value=weight_max_abs_warn_threshold,
        mode="le",
        message_ok="No extreme augmented donor weight detected (model-aligned threshold).",
        message_flagged=(
            "At least one augmented donor weight is extreme relative to the model warning threshold."
        ),
    )
    _append_check(
        test="l1_norm_weights_augmented",
        value=l1_norm_weight_aug,
        threshold_value=weight_l1_warn_threshold,
        mode="le",
        message_ok="Total absolute augmented weight mass is controlled (model-aligned threshold).",
        message_flagged="Total absolute augmented weight mass is high relative to model warning threshold.",
    )
    _append_check(
        test="negative_weight_share",
        value=negative_weight_share,
        threshold_value=0.30,
        mode="le",
        message_ok="Negative-weight share is moderate.",
        message_flagged="Negative-weight share is high and may indicate extrapolation.",
    )
    _append_check(
        test="slsqp_fallback_count",
        value=slsqp_fallback_count,
        threshold_value=0.0,
        mode="eq",
        fail_severity="RED",
        message_ok="No optimizer fallback events recorded.",
        message_flagged=(
            f"Optimizer fallback occurred {slsqp_fallback_count} time(s): "
            f"{'; '.join(slsqp_fallback_reasons) if slsqp_fallback_reasons else 'no reason reported'}."
        ),
    )
    _append_check(
        test="suppressed_fit_warning_count",
        value=int(len(suppressed_fit_warnings)),
        threshold_value=0.0,
        mode="eq",
        fail_severity="RED",
        message_ok="No suppressed fit warnings were captured.",
        message_flagged=(
            f"Suppressed fit warnings detected ({len(suppressed_fit_warnings)}): "
            f"{'; '.join(suppressed_fit_warnings[:2])}."
        ),
    )

    return pd.DataFrame(checks, columns=["test", "flag", "value", "threshold", "message"])


__all__ = ["run_scm_diagnostics"]
