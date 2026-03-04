from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import norm

from causalis.data_contracts.multicausaldata import MultiCausalData

__all__ = [
    "sensitivity_analysis",
    "get_sensitivity_summary",
    "sensitivity_benchmark",
    "compute_bias_aware_ci",
]

_ESSENTIALLY_ZERO = 1e-32


def _is_model_like(obj: Any) -> bool:
    return hasattr(obj, "coef_") and hasattr(obj, "se_")


def _is_estimate_like(obj: Any) -> bool:
    return hasattr(obj, "value") and hasattr(obj, "diagnostic_data")


def _to_1d_float(x: Any, *, name: str = "value") -> np.ndarray:
    if x is None:
        raise ValueError(f"{name} must not be None.")
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        return arr.reshape(1)
    return arr.reshape(-1)


def _broadcast_to_length(x: Any, length: int, *, name: str) -> np.ndarray:
    arr = _to_1d_float(x, name=name)
    if arr.size == 1 and length > 1:
        return np.repeat(arr, length)
    if arr.size != length:
        raise ValueError(f"{name} must have length {length}, got {arr.size}.")
    return arr


def _ci_to_2d(ci: Any, j: int) -> np.ndarray:
    ci_arr = np.asarray(ci, dtype=float)
    if ci_arr.ndim == 1:
        if ci_arr.size != 2:
            raise ValueError("sampling_ci as 1D must have length 2.")
        return np.tile(ci_arr.reshape(1, 2), (j, 1))
    if ci_arr.ndim == 2:
        if ci_arr.shape == (j, 2):
            return ci_arr
        if ci_arr.shape == (2, j):
            return ci_arr.T
    raise ValueError(
        f"sampling_ci must have shape (2,), ({j}, 2), or (2, {j}); got {ci_arr.shape}."
    )


def _build_sampling_ci(theta: np.ndarray, se: np.ndarray, alpha: float) -> np.ndarray:
    z = float(norm.ppf(1 - alpha / 2.0))
    if not np.isfinite(z):
        return np.column_stack([theta, theta])
    return np.column_stack([theta - z * se, theta + z * se])


def _maybe_squeeze_scalar(x: Any, *, j: int) -> Any:
    if j != 1:
        return x
    if isinstance(x, tuple) and len(x) == 2:
        return float(x[0]), float(x[1])
    arr = np.asarray(x)
    if arr.ndim == 0:
        return float(arr)
    if arr.ndim == 1 and arr.size == 1:
        return float(arr[0])
    if arr.ndim == 2 and arr.shape == (1, 2):
        return float(arr[0, 0]), float(arr[0, 1])
    return x


def _squeeze_result_if_scalar(out: Dict[str, Any], *, j: int) -> Dict[str, Any]:
    if j != 1:
        return out

    squeezed = dict(out)
    for key in ("theta", "se", "max_bias_base", "max_bias", "bound_width", "nu2", "rv", "rva"):
        arr = np.asarray(squeezed[key], dtype=float).reshape(-1)
        squeezed[key] = float(arr[0]) if arr.size else np.nan

    for key in ("sampling_ci", "theta_bounds_cofounding", "bias_aware_ci"):
        arr = np.asarray(squeezed[key], dtype=float).reshape(1, 2)
        squeezed[key] = float(arr[0, 0]), float(arr[0, 1])

    labels = squeezed.get("contrast_labels")
    if isinstance(labels, (list, tuple, np.ndarray)) and len(labels) == 1:
        squeezed["contrast_labels"] = str(labels[0])

    return squeezed


def _stack_ci_bounds(lower: Any, upper: Any, *, j: int) -> np.ndarray:
    lo = _broadcast_to_length(lower, j, name="ci_lower_absolute")
    hi = _broadcast_to_length(upper, j, name="ci_upper_absolute")
    return np.column_stack([lo, hi])


def _extract_model_ci(model: Any, alpha: float) -> Any:
    confint_cached = getattr(model, "confint_", None)
    if confint_cached is not None:
        return confint_cached

    if not hasattr(model, "confint"):
        return None

    ci_obj = None
    try:
        ci_obj = model.confint(alpha=alpha)
    except TypeError:
        try:
            ci_obj = model.confint()
        except Exception:
            ci_obj = None
    except Exception:
        ci_obj = None

    if isinstance(ci_obj, pd.DataFrame):
        arr = ci_obj.to_numpy(dtype=float)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            return arr[:, :2]
    return ci_obj


def _resolve_input_context(effect_estimation: Any, context: Any = None) -> Dict[str, Any]:
    model = None
    estimate = None
    data = None
    diag = None
    effect_dict = effect_estimation if isinstance(effect_estimation, dict) else None

    if isinstance(effect_estimation, dict):
        model = effect_estimation.get("model")
        estimate_candidate = effect_estimation.get("estimate")
        if estimate_candidate is not None and _is_estimate_like(estimate_candidate):
            estimate = estimate_candidate

        data_candidate = effect_estimation.get("data", effect_estimation.get("data_contracts"))
        if isinstance(data_candidate, MultiCausalData):
            data = data_candidate

        diag = effect_estimation.get("diagnostic_data")
    elif _is_estimate_like(effect_estimation):
        estimate = effect_estimation
        diag = getattr(effect_estimation, "diagnostic_data", None)
    elif _is_model_like(effect_estimation):
        model = effect_estimation
    elif isinstance(effect_estimation, MultiCausalData):
        data = effect_estimation

    if context is not None:
        if isinstance(context, MultiCausalData):
            data = context
        elif _is_estimate_like(context):
            estimate = context
            diag = getattr(context, "diagnostic_data", diag)
        elif isinstance(context, dict):
            model = context.get("model", model)
            estimate_candidate = context.get("estimate")
            if estimate_candidate is not None and _is_estimate_like(estimate_candidate):
                estimate = estimate_candidate
            data_candidate = context.get("data", context.get("data_contracts"))
            if isinstance(data_candidate, MultiCausalData):
                data = data_candidate
            diag = context.get("diagnostic_data", diag)
        elif _is_model_like(context):
            model = context

    if diag is None and estimate is not None:
        diag = getattr(estimate, "diagnostic_data", None)

    if data is None and estimate is not None:
        data_candidate = getattr(estimate, "data", getattr(estimate, "data_contracts", None))
        if isinstance(data_candidate, MultiCausalData):
            data = data_candidate

    if model is not None:
        model_data = getattr(model, "data", getattr(model, "data_contracts", None))
        if data is None and isinstance(model_data, MultiCausalData):
            data = model_data
        if diag is None:
            diag = getattr(model, "diagnostic_data", None)

    return {
        "model": model,
        "estimate": estimate,
        "data": data,
        "diag": diag,
        "effect_dict": effect_dict,
    }


def _validate_estimate_data_alignment(
    *,
    estimate: Optional[Any],
    data: Optional[MultiCausalData],
) -> None:
    if estimate is None or data is None:
        return

    outcome_est = getattr(estimate, "outcome", None)
    if outcome_est is not None and str(outcome_est) != str(data.outcome):
        raise ValueError(
            "estimate.outcome must match data.outcome "
            f"({outcome_est!r} != {data.outcome!r})."
        )

    treatments_est = getattr(estimate, "treatment", None)
    if treatments_est is None:
        return

    est_names = [str(name) for name in list(treatments_est)]
    data_names = [str(name) for name in list(data.treatment_names)]
    if est_names and est_names != data_names:
        raise ValueError(
            "estimate.treatment must match data.treatment_names in the same order "
            f"({est_names!r} != {data_names!r})."
        )


def _labels_from_treatments(treatment_names: List[str], j: int) -> Optional[List[str]]:
    if len(treatment_names) < j + 1:
        return None
    baseline = str(treatment_names[0])
    return [f"{str(name)} vs {baseline}" for name in treatment_names[1 : j + 1]]


def _infer_contrast_labels(
    *,
    estimate: Optional[Any],
    model: Optional[Any],
    data: Optional[MultiCausalData],
    j: int,
) -> List[str]:
    if estimate is not None:
        contrast_labels = getattr(estimate, "contrast_labels", None)
        if contrast_labels is not None:
            labels = [str(label) for label in list(contrast_labels)]
            if len(labels) == j:
                return labels

        treatment_names = getattr(estimate, "treatment", None)
        if treatment_names is not None:
            labels = _labels_from_treatments([str(name) for name in list(treatment_names)], j)
            if labels is not None:
                return labels

    if data is not None:
        labels = _labels_from_treatments([str(name) for name in list(data.treatment_names)], j)
        if labels is not None:
            return labels

    if model is not None:
        model_data = getattr(model, "data", getattr(model, "data_contracts", None))
        if isinstance(model_data, MultiCausalData):
            labels = _labels_from_treatments([str(name) for name in list(model_data.treatment_names)], j)
            if labels is not None:
                return labels

    return [f"contrast_{idx + 1}" for idx in range(j)]


def _finalize_theta_se_ci(
    *,
    theta_raw: Any,
    se_raw: Any,
    ci_raw: Any,
    alpha: float,
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray], Union[Tuple[float, float], np.ndarray]]:
    theta_arr = _to_1d_float(theta_raw, name="theta")
    j = theta_arr.size
    z = float(norm.ppf(1 - alpha / 2.0))

    se_arr: Optional[np.ndarray] = None
    if se_raw is not None:
        se_arr = _broadcast_to_length(se_raw, j, name="se")

    ci_arr: Optional[np.ndarray] = None
    if ci_raw is not None:
        ci_arr = _ci_to_2d(ci_raw, j)

    if se_arr is None:
        if ci_arr is not None and np.isfinite(z) and z > 0.0:
            se_arr = np.maximum((ci_arr[:, 1] - theta_arr) / z, 0.0)
        else:
            se_arr = np.zeros(j, dtype=float)

    if ci_arr is None:
        ci_arr = _build_sampling_ci(theta_arr, se_arr, alpha=alpha)

    return (
        _maybe_squeeze_scalar(theta_arr, j=j),
        _maybe_squeeze_scalar(se_arr, j=j),
        _maybe_squeeze_scalar(ci_arr, j=j),
    )


def _pull_theta_se_ci(
    effect_estimation: Any,
    alpha: float,
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray], Union[Tuple[float, float], np.ndarray]]:
    if _is_estimate_like(effect_estimation):
        theta_raw = getattr(effect_estimation, "value", 0.0)
        theta_arr = _to_1d_float(theta_raw, name="value")
        j = theta_arr.size

        ci_low = getattr(effect_estimation, "ci_lower_absolute", None)
        ci_high = getattr(effect_estimation, "ci_upper_absolute", None)
        ci_raw = None
        if ci_low is not None and ci_high is not None:
            ci_raw = _stack_ci_bounds(ci_low, ci_high, j=j)

        options = getattr(effect_estimation, "model_options", {}) or {}
        se_raw = options.get("std_error") if isinstance(options, dict) else None

        return _finalize_theta_se_ci(theta_raw=theta_arr, se_raw=se_raw, ci_raw=ci_raw, alpha=alpha)

    if isinstance(effect_estimation, dict):
        model = effect_estimation.get("model")
        theta_raw = effect_estimation.get("coefficient")
        se_raw = effect_estimation.get("std_error")
        ci_raw = effect_estimation.get("confidence_interval")

        if theta_raw is None and model is not None and hasattr(model, "coef_"):
            theta_raw = getattr(model, "coef_")
        if se_raw is None and model is not None and hasattr(model, "se_"):
            se_raw = getattr(model, "se_")
        if se_raw is None:
            options = effect_estimation.get("model_options", {})
            if isinstance(options, dict):
                se_raw = options.get("std_error")
        if ci_raw is None and model is not None:
            ci_raw = _extract_model_ci(model, alpha=alpha)

        if theta_raw is None:
            theta_raw = 0.0

        return _finalize_theta_se_ci(theta_raw=theta_raw, se_raw=se_raw, ci_raw=ci_raw, alpha=alpha)

    if _is_model_like(effect_estimation):
        model = effect_estimation
        theta_raw = getattr(model, "coef_", 0.0)
        se_raw = getattr(model, "se_", 0.0)
        ci_raw = _extract_model_ci(model, alpha=alpha)
        return _finalize_theta_se_ci(theta_raw=theta_raw, se_raw=se_raw, ci_raw=ci_raw, alpha=alpha)

    return 0.0, 0.0, (0.0, 0.0)


def _extract_sensitivity_elements(diag: Any, model: Any) -> Optional[Dict[str, Any]]:
    if isinstance(diag, dict) and diag.get("sigma2") is not None:
        return {
            "sigma2": diag.get("sigma2"),
            "nu2": diag.get("nu2"),
            "psi_sigma2": diag.get("psi_sigma2"),
            "psi_nu2": diag.get("psi_nu2"),
            "riesz_rep": diag.get("riesz_rep"),
            "m_alpha": diag.get("m_alpha"),
            "psi": diag.get("psi"),
        }

    if diag is not None and getattr(diag, "sigma2", None) is not None:
        return {
            "sigma2": getattr(diag, "sigma2", None),
            "nu2": getattr(diag, "nu2", None),
            "psi_sigma2": getattr(diag, "psi_sigma2", None),
            "psi_nu2": getattr(diag, "psi_nu2", None),
            "riesz_rep": getattr(diag, "riesz_rep", None),
            "m_alpha": getattr(diag, "m_alpha", None),
            "psi": getattr(diag, "psi", None),
        }

    if model is not None and hasattr(model, "_sensitivity_element_est"):
        return model._sensitivity_element_est()

    return None


def _ensure_2d(a: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(a, dtype=float)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 1D or 2D; got shape {arr.shape}.")
    return arr


def _compute_sensitivity_bias_unified(
    sigma2: Union[float, np.ndarray],
    nu2: Union[float, np.ndarray],
    psi_sigma2: np.ndarray,
    psi_nu2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    sigma2_f = float(np.asarray(sigma2).reshape(()))
    psi_sigma2_arr = np.asarray(psi_sigma2, dtype=float).reshape(-1)
    n = psi_sigma2_arr.shape[0]

    nu2_arr = np.atleast_1d(np.asarray(nu2, dtype=float).reshape(-1))
    j = nu2_arr.size

    if sigma2_f <= 0.0:
        return np.zeros(j, dtype=float), np.zeros((n, j), dtype=float)

    psi_sigma2_arr = psi_sigma2_arr - float(np.mean(psi_sigma2_arr))
    psi_nu2_arr = np.asarray(psi_nu2, dtype=float)
    if psi_nu2_arr.ndim == 1:
        psi_nu2_arr = psi_nu2_arr.reshape(-1, 1)
    if psi_nu2_arr.shape != (n, j):
        raise ValueError(f"psi_nu2 must have shape ({n}, {j}), got {psi_nu2_arr.shape}.")
    psi_nu2_arr = psi_nu2_arr - psi_nu2_arr.mean(axis=0, keepdims=True)

    product = np.maximum(sigma2_f * nu2_arr, 0.0)
    max_bias = np.sqrt(product)

    psi_max_bias = np.zeros((n, j), dtype=float)
    for idx in range(j):
        if nu2_arr[idx] <= 0.0 or max_bias[idx] <= _ESSENTIALLY_ZERO:
            continue
        denom = 2.0 * max_bias[idx]
        psi_max_bias[:, idx] = (
            sigma2_f * psi_nu2_arr[:, idx] + nu2_arr[idx] * psi_sigma2_arr
        ) / denom

    return max_bias, psi_max_bias


def compute_sensitivity_bias(
    sigma2: Union[float, np.ndarray],
    nu2: Union[float, np.ndarray],
    psi_sigma2: np.ndarray,
    psi_nu2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    return _compute_sensitivity_bias_unified(sigma2, nu2, psi_sigma2, psi_nu2)


def compute_sensitivity_bias_local(
    sigma2: Union[float, np.ndarray],
    nu2: Union[float, np.ndarray],
    psi_sigma2: np.ndarray,
    psi_nu2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    return _compute_sensitivity_bias_unified(sigma2, nu2, psi_sigma2, psi_nu2)


def combine_nu2(
    m_alpha: np.ndarray,
    rr: np.ndarray,
    cf_y: float,
    r2_d: Union[float, np.ndarray],
    rho: Union[float, np.ndarray],
    use_signed_rr: bool = False,
) -> Tuple[Union[float, np.ndarray], np.ndarray]:
    cf_y = float(cf_y)

    if cf_y < 0.0:
        raise ValueError("cf_y must be >= 0.")

    m_alpha_arr = _ensure_2d(m_alpha, name="m_alpha")
    rr_arr = _ensure_2d(rr, name="rr")
    if m_alpha_arr.shape != rr_arr.shape:
        raise ValueError(
            "m_alpha and rr must have the same shape (n, K-1). "
            f"Got {m_alpha_arr.shape} and {rr_arr.shape}."
        )
    j = m_alpha_arr.shape[1]

    r2_d_arr = _broadcast_to_length(r2_d, j, name="r2_d")
    rho_arr = _broadcast_to_length(rho, j, name="rho")
    rho_arr = np.clip(rho_arr, -1.0, 1.0)

    if np.any(~np.isfinite(r2_d_arr)) or np.any(r2_d_arr < 0.0):
        raise ValueError("r2_d must be finite and >= 0.")
    if np.any(r2_d_arr >= 1.0):
        raise ValueError("r2_d must be < 1.0.")

    cf_d = r2_d_arr / (1.0 - r2_d_arr)
    a = np.sqrt(2.0 * np.maximum(m_alpha_arr, 0.0))
    b = rr_arr if use_signed_rr else np.abs(rr_arr)

    cross_scale = np.sqrt(cf_y * cf_d)
    base = (a * a) * cf_y + (b * b) * cf_d[None, :] + 2.0 * rho_arr[None, :] * cross_scale[None, :] * a * b
    base = np.maximum(base, 0.0)

    nu2 = base.mean(axis=0)
    psi_nu2 = base - nu2[None, :]
    if nu2.size == 1:
        return float(nu2[0]), psi_nu2.reshape(-1)
    return nu2, psi_nu2


def _combine_nu2_local(
    m_alpha: np.ndarray,
    rr: np.ndarray,
    cf_y: float,
    r2_d: Union[float, np.ndarray],
    rho: Union[float, np.ndarray],
    _unused: Any = None,
    use_signed_rr: bool = False,
) -> Tuple[Union[float, np.ndarray], np.ndarray]:
    return combine_nu2(m_alpha, rr, cf_y, r2_d, rho, use_signed_rr=use_signed_rr)


def pulltheta_se_ci(
    effect_estimation: Any,
    alpha: float,
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray], Union[Tuple[float, float], np.ndarray]]:
    return _pull_theta_se_ci(effect_estimation, alpha=alpha)


def compute_bias_aware_ci(
    effect_estimation: Dict[str, Any] | Any,
    _=None,
    cf_y: float = 0.0,
    r2_d: Union[float, np.ndarray] = 0.0,
    rho: Union[float, np.ndarray] = 1.0,
    H0: float = 0.0,
    alpha: float = 0.05,
    use_signed_rr: bool = False,
) -> Dict[str, Any]:
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
    if cf_y < 0.0:
        raise ValueError("cf_y must be >= 0.")

    ctx = _resolve_input_context(effect_estimation, context=_)
    model = ctx["model"]
    estimate = ctx["estimate"]
    data = ctx["data"]
    diag = ctx["diag"]

    if isinstance(effect_estimation, dict):
        has_legacy_stats = any(
            key in effect_estimation for key in ("coefficient", "std_error", "confidence_interval")
        )
        if model is None and estimate is None and not has_legacy_stats:
            raise TypeError(
                "effect_estimation must provide one of: "
                "`model`, `estimate`, or legacy coefficient/std_error fields."
            )
    elif not (_is_estimate_like(effect_estimation) or _is_model_like(effect_estimation)):
        raise TypeError(
            "effect_estimation must be a dict, MultiCausalEstimate-like object, "
            "or a fitted model exposing coef_/se_."
        )

    _validate_estimate_data_alignment(estimate=estimate, data=data)

    source_for_pull = effect_estimation
    if (
        isinstance(effect_estimation, dict)
        and estimate is not None
        and effect_estimation.get("coefficient") is None
        and effect_estimation.get("model") is None
    ):
        source_for_pull = estimate

    theta, se, sampling_ci = _pull_theta_se_ci(source_for_pull, alpha=alpha)
    theta_arr = _to_1d_float(theta, name="theta")
    j = theta_arr.size
    se_arr = _broadcast_to_length(se, j, name="se")
    sampling_ci_2d = _ci_to_2d(sampling_ci, j)
    r2_d_arr = _broadcast_to_length(r2_d, j, name="r2_d")
    rho_arr = _broadcast_to_length(rho, j, name="rho")
    rho_clip_arr = np.clip(rho_arr, -1.0, 1.0)

    if np.any(~np.isfinite(r2_d_arr)) or np.any(r2_d_arr < 0.0):
        raise ValueError("r2_d must be finite and >= 0.")
    if np.any(r2_d_arr >= 1.0):
        raise ValueError("r2_d must be < 1.0.")

    if data is not None:
        expected = len(data.treatment_names) - 1
        if expected != j:
            raise ValueError(
                "Number of effects does not match MultiCausalData treatment columns: "
                f"expected {expected}, got {j}."
            )

    if estimate is not None:
        treatment_names_est = getattr(estimate, "treatment", None)
        if treatment_names_est is not None:
            expected = len(list(treatment_names_est)) - 1
            if expected > 0 and expected != j:
                raise ValueError(
                    "Number of effects does not match MultiCausalEstimate treatment contrasts: "
                    f"expected {expected}, got {j}."
                )

    contrast_labels = _infer_contrast_labels(estimate=estimate, model=model, data=data, j=j)

    elems = _extract_sensitivity_elements(diag=diag, model=model)
    z = float(norm.ppf(1 - alpha / 2.0))

    sigma2 = np.nan
    nu2 = np.full(j, np.nan, dtype=float)
    max_bias_base = np.zeros(j, dtype=float)
    max_bias = np.zeros(j, dtype=float)
    bound_width = np.zeros(j, dtype=float)
    rv = np.full(j, np.nan, dtype=float)
    rva = np.full(j, np.nan, dtype=float)

    correction_scale: Optional[np.ndarray] = None
    use_signed_rr_effective = bool(use_signed_rr)
    psi_nu2_for_ci: Optional[np.ndarray] = None

    if elems is not None:
        sigma2 = float(np.asarray(elems.get("sigma2", np.nan)).reshape(()))
        nu2 = _broadcast_to_length(elems.get("nu2", np.full(j, np.nan, dtype=float)), j, name="nu2")
        psi_nu2_for_ci = elems.get("psi_nu2", None)

        if use_signed_rr_effective:
            m_alpha = elems.get("m_alpha", None)
            rr = elems.get("riesz_rep", None)
            if m_alpha is not None and rr is not None:
                nu2_signed, psi_nu2_signed = _combine_nu2_local(
                    m_alpha=m_alpha,
                    rr=rr,
                    cf_y=cf_y,
                    r2_d=r2_d_arr,
                    rho=rho_clip_arr,
                    use_signed_rr=True,
                )
                nu2 = _broadcast_to_length(nu2_signed, j, name="nu2")
                psi_nu2_for_ci = np.asarray(psi_nu2_signed, dtype=float)

                max_bias_base = np.sqrt(np.maximum(sigma2 * nu2, 0.0))
                max_bias = max_bias_base.copy()
                bound_width = max_bias.copy()
                correction_scale = np.ones(j, dtype=float)
            else:
                use_signed_rr_effective = False

        if not use_signed_rr_effective:
            bias_factor = np.zeros(j, dtype=float)
            if cf_y > 0.0:
                positive = r2_d_arr > 0.0
                bias_factor[positive] = np.sqrt(cf_y * r2_d_arr[positive] / (1.0 - r2_d_arr[positive]))
            max_bias_base = np.sqrt(np.maximum(sigma2 * nu2, 0.0))
            max_bias = max_bias_base * bias_factor
            bound_width = np.abs(rho_clip_arr) * max_bias
            correction_scale = np.abs(rho_clip_arr) * bias_factor

            delta_theta = np.abs(theta_arr - float(H0))
            denom_rv = np.abs(rho_clip_arr) * max_bias_base * np.sqrt(cf_y)

            good = np.isfinite(denom_rv) & (denom_rv > 1e-16) & (delta_theta > 0.0)
            D = np.zeros(j, dtype=float)
            D[good] = (delta_theta[good] / denom_rv[good]) ** 2
            rv[good] = D[good] / (1.0 + D[good])
            rv[delta_theta == 0.0] = 0.0

            delta_theta_a = np.maximum(delta_theta - z * se_arr, 0.0)
            good_a = np.isfinite(denom_rv) & (denom_rv > 1e-16) & (delta_theta_a > 0.0)
            Da = np.zeros(j, dtype=float)
            Da[good_a] = (delta_theta_a[good_a] / denom_rv[good_a]) ** 2
            rva[good_a] = Da[good_a] / (1.0 + Da[good_a])
            rva[delta_theta_a == 0.0] = 0.0
    else:
        delta_theta = np.abs(theta_arr - float(H0))
        rv[delta_theta == 0.0] = 0.0
        delta_theta_a = np.maximum(delta_theta - z * se_arr, 0.0)
        rva[delta_theta_a == 0.0] = 0.0

    theta_lower = theta_arr - bound_width
    theta_upper = theta_arr + bound_width
    theta_bounds = np.column_stack([theta_lower, theta_upper])

    bias_aware_ci = np.column_stack([theta_lower - z * se_arr, theta_upper + z * se_arr])
    bad_se = (~np.isfinite(se_arr)) | (se_arr < 0.0) | (not np.isfinite(z))
    if np.any(bad_se):
        bias_aware_ci[bad_se] = theta_bounds[bad_se]

    if (
        elems is not None
        and elems.get("psi") is not None
        and elems.get("psi_sigma2") is not None
        and psi_nu2_for_ci is not None
        and np.isfinite(z)
    ):
        psi = np.asarray(elems.get("psi"), dtype=float)
        if psi.ndim == 1:
            if j != 1:
                raise ValueError(f"psi must have shape (n, {j}), got {psi.shape}.")
            psi = psi.reshape(-1, 1)
        if psi.ndim != 2 or psi.shape[1] != j:
            raise ValueError(f"psi must have shape (n, {j}), got {psi.shape}.")
        n = psi.shape[0]

        psi_sigma2 = _broadcast_to_length(elems.get("psi_sigma2"), n, name="psi_sigma2")
        psi_nu2 = np.asarray(psi_nu2_for_ci, dtype=float)
        if psi_nu2.ndim == 1:
            if j != 1:
                raise ValueError(f"psi_nu2 must have shape (n, {j}), got {psi_nu2.shape}.")
            psi_nu2 = psi_nu2.reshape(-1, 1)
        if psi_nu2.shape != (n, j):
            raise ValueError(f"psi_nu2 must have shape ({n}, {j}), got {psi_nu2.shape}.")

        corr_scale = np.zeros(j, dtype=float) if correction_scale is None else correction_scale
        corr_scale = _broadcast_to_length(corr_scale, j, name="correction_scale")

        denom = 2.0 * np.sqrt(np.maximum(sigma2 * nu2, 0.0))
        numer = sigma2 * psi_nu2 + (nu2[None, :] * psi_sigma2[:, None])
        correction = np.zeros_like(psi, dtype=float)

        good = (
            np.isfinite(denom)
            & (denom > _ESSENTIALLY_ZERO)
            & np.isfinite(corr_scale)
            & (corr_scale > 0.0)
        )
        if np.any(good):
            correction[:, good] = corr_scale[good] * (numer[:, good] / denom[good])

        psi_plus = psi + correction
        psi_minus = psi - correction

        if n > 1:
            se_lower = np.sqrt(np.var(psi_minus, axis=0, ddof=1) / n)
            se_upper = np.sqrt(np.var(psi_plus, axis=0, ddof=1) / n)
        else:
            se_lower = se_arr.copy()
            se_upper = se_arr.copy()

        se_lower = np.where(np.isfinite(se_lower) & (se_lower >= 0.0), se_lower, se_arr)
        se_upper = np.where(np.isfinite(se_upper) & (se_upper >= 0.0), se_upper, se_arr)

        bias_aware_ci = np.column_stack([theta_lower - z * se_lower, theta_upper + z * se_upper])
        if np.any(bad_se):
            bias_aware_ci[bad_se] = theta_bounds[bad_se]

    out: Dict[str, Any] = {
        "theta": theta_arr,
        "se": se_arr,
        "alpha": float(alpha),
        "z": float(z),
        "H0": float(H0),
        "sampling_ci": sampling_ci_2d,
        "theta_bounds_cofounding": theta_bounds,
        "bias_aware_ci": bias_aware_ci,
        "max_bias_base": max_bias_base,
        "max_bias": max_bias,
        "bound_width": bound_width,
        "sigma2": float(sigma2),
        "nu2": nu2,
        "rv": rv,
        "rva": rva,
        "contrast_labels": contrast_labels,
        "params": {
            "cf_y": float(cf_y),
            "r2_d": _maybe_squeeze_scalar(r2_d_arr, j=j),
            "rho": _maybe_squeeze_scalar(rho_clip_arr, j=j),
            "use_signed_rr": bool(use_signed_rr_effective),
        },
    }
    return _squeeze_result_if_scalar(out, j=j)


def _labels_from_result(res: Dict[str, Any], j: int) -> List[str]:
    labels = res.get("contrast_labels")
    if isinstance(labels, str):
        labels = [labels]
    if isinstance(labels, (list, tuple, np.ndarray)):
        labels_list = [str(label) for label in list(labels)]
        if len(labels_list) == j:
            return labels_list
    return [f"contrast_{idx + 1}" for idx in range(j)]


def format_bias_aware_summary(res: Dict[str, Any], label: str | None = None) -> str:
    theta_arr = _to_1d_float(res.get("theta", 0.0), name="theta")
    j = theta_arr.size
    se_arr = _broadcast_to_length(res.get("se", 0.0), j, name="se")
    max_bias_arr = _broadcast_to_length(res.get("max_bias", 0.0), j, name="max_bias")
    nu2_arr = _broadcast_to_length(res.get("nu2", np.nan), j, name="nu2")
    max_bias_base_arr = _broadcast_to_length(res.get("max_bias_base", np.nan), j, name="max_bias_base")
    bound_width_arr = _broadcast_to_length(
        res.get("bound_width", max_bias_arr), j, name="bound_width"
    )

    sampling_ci = _ci_to_2d(res.get("sampling_ci", (0.0, 0.0)), j)
    theta_bounds = _ci_to_2d(res.get("theta_bounds_cofounding", (0.0, 0.0)), j)
    bias_aware_ci = _ci_to_2d(res.get("bias_aware_ci", (0.0, 0.0)), j)
    params = res.get("params", {})

    if j == 1:
        row_label = label or _labels_from_result(res, 1)[0]
        lines = []
        lines.append("================== Bias-aware Interval ==================")
        lines.append("")
        lines.append("------------------ Scenario          ------------------")
        lines.append(f"Significance Level: alpha={res.get('alpha', 0.05)}")
        lines.append(f"Null Hypothesis: H0={res.get('H0', 0.0)}")
        lines.append(
            "Sensitivity parameters: "
            f"cf_y={params.get('cf_y', 0.0)}; "
            f"r2_d={params.get('r2_d', 0.0)}, "
            f"rho={params.get('rho', 0.0)}, "
            f"use_signed_rr={params.get('use_signed_rr', False)}"
        )
        lines.append("")
        lines.append("------------------ Components        ------------------")
        lines.append(
            f"{'':>12} {'theta':>11} {'se':>11} {'z':>8} {'max_bias':>12} {'sigma2':>12} {'nu2':>12}"
        )
        lines.append(
            f"{row_label:>12} {theta_arr[0]:11.6f} {se_arr[0]:11.6f} {float(res.get('z', np.nan)):8.4f} "
            f"{max_bias_arr[0]:12.6f} {float(res.get('sigma2', np.nan)):12.6f} {nu2_arr[0]:12.6f}"
        )
        lines.append(f"Bound width (theta +/-): {bound_width_arr[0]:.6f}")
        if np.isfinite(max_bias_base_arr[0]):
            lines.append(f"Base sqrt(sigma2*nu2): {max_bias_base_arr[0]:.6f}")
        lines.append("")
        lines.append("------------------ Intervals         ------------------")
        lines.append(
            f"{'':>12} {'Sampling CI l':>14} {'Conf. θ l':>12} {'Bias-aware l':>14} "
            f"{'Bias-aware u':>14} {'Conf. θ u':>12} {'Sampling CI u':>14}"
        )
        lines.append(
            f"{row_label:>12} {sampling_ci[0, 0]:14.6f} {theta_bounds[0, 0]:12.6f} "
            f"{bias_aware_ci[0, 0]:14.6f} {bias_aware_ci[0, 1]:14.6f} "
            f"{theta_bounds[0, 1]:12.6f} {sampling_ci[0, 1]:14.6f}"
        )

        if "rv" in res and "rva" in res:
            rv_val = float(np.asarray(res["rv"], dtype=float).reshape(-1)[0])
            rva_val = float(np.asarray(res["rva"], dtype=float).reshape(-1)[0])
            lines.append("")
            lines.append("------------------ Robustness Values ------------------")
            lines.append(f"{'':>12} {'RV (%)':>15} {'RVa (%)':>15}")
            lines.append(f"{row_label:>12} {rv_val*100:15.6f} {rva_val*100:15.6f}")

        return "\n".join(lines)

    labels = _labels_from_result(res, j)
    df = pd.DataFrame(
        {
            "theta": theta_arr,
            "se": se_arr,
            "max_bias": max_bias_arr,
            "max_bias_base": max_bias_base_arr,
            "bound_width": bound_width_arr,
            "sigma2": float(res.get("sigma2", np.nan)),
            "nu2": nu2_arr,
            "sampling_ci_l": sampling_ci[:, 0],
            "sampling_ci_u": sampling_ci[:, 1],
            "theta_l": theta_bounds[:, 0],
            "theta_u": theta_bounds[:, 1],
            "bias_aware_ci_l": bias_aware_ci[:, 0],
            "bias_aware_ci_u": bias_aware_ci[:, 1],
        },
        index=labels,
    )

    rv_raw = res.get("rv")
    rva_raw = res.get("rva")
    if rv_raw is not None and rva_raw is not None:
        try:
            rv_arr = _broadcast_to_length(rv_raw, j, name="rv")
            rva_arr = _broadcast_to_length(rva_raw, j, name="rva")
            df["rv"] = rv_arr
            df["rva"] = rva_arr
        except Exception:
            pass

    lines = []
    lines.append("================== Bias-aware Interval ==================")
    lines.append("")
    lines.append("------------------ Scenario          ------------------")
    if label is not None:
        lines.append(f"Label: {label}")
    lines.append(f"Significance Level: alpha={res.get('alpha', 0.05)}")
    lines.append(f"Null Hypothesis: H0={res.get('H0', 0.0)}")
    lines.append(
        "Sensitivity parameters: "
        f"cf_y={params.get('cf_y', 0.0)}; "
        f"r2_d={params.get('r2_d', 0.0)}, "
        f"rho={params.get('rho', 0.0)}, "
        f"use_signed_rr={params.get('use_signed_rr', False)}"
    )
    lines.append("")
    lines.append(df.to_string(float_format=lambda value: f"{value: .6f}"))
    return "\n".join(lines)


def get_sensitivity_summary(
    effect_estimation: Dict[str, Any] | Any,
    _=None,
    label: Optional[str] = None,
) -> Optional[str]:
    ctx = _resolve_input_context(effect_estimation, context=_)
    estimate = ctx["estimate"]
    model = ctx["model"]
    data = ctx["data"]
    diag = ctx["diag"]

    res = None
    if isinstance(effect_estimation, dict):
        res = effect_estimation.get("bias_aware", None)

    if not isinstance(res, dict) or not res:
        if isinstance(diag, dict):
            res = diag.get("sensitivity_analysis")
        elif diag is not None:
            res = getattr(diag, "sensitivity_analysis", None)

    if not isinstance(res, dict) or not res:
        alpha_fallback = getattr(effect_estimation, "alpha", None)
        if alpha_fallback is None and isinstance(effect_estimation, dict):
            alpha_fallback = effect_estimation.get("alpha", None)
        try:
            alpha_fallback = float(alpha_fallback)
        except (TypeError, ValueError):
            alpha_fallback = 0.05
        if not (0.0 < alpha_fallback < 1.0):
            alpha_fallback = 0.05

        source_for_pull = estimate if estimate is not None else effect_estimation
        theta, se, sampling_ci = _pull_theta_se_ci(source_for_pull, alpha=alpha_fallback)
        theta_arr = _to_1d_float(theta, name="theta")
        j = theta_arr.size
        se_arr = _broadcast_to_length(se, j, name="se")
        sampling_ci_2d = _ci_to_2d(sampling_ci, j)
        z = float(norm.ppf(1 - alpha_fallback / 2.0))
        contrast_labels = _infer_contrast_labels(estimate=estimate, model=model, data=data, j=j)

        fallback = {
            "theta": theta_arr,
            "se": se_arr,
            "alpha": float(alpha_fallback),
            "z": z,
            "H0": 0.0,
            "sampling_ci": sampling_ci_2d,
            "theta_bounds_cofounding": np.column_stack([theta_arr, theta_arr]),
            "bias_aware_ci": _build_sampling_ci(theta_arr, se_arr, alpha=alpha_fallback),
            "max_bias_base": np.zeros(j, dtype=float),
            "max_bias": np.zeros(j, dtype=float),
            "bound_width": np.zeros(j, dtype=float),
            "sigma2": np.nan,
            "nu2": np.full(j, np.nan, dtype=float),
            "rv": np.full(j, np.nan, dtype=float),
            "rva": np.full(j, np.nan, dtype=float),
            "contrast_labels": contrast_labels,
            "params": {"cf_y": 0.0, "r2_d": 0.0, "rho": 0.0, "use_signed_rr": False},
        }
        res = _squeeze_result_if_scalar(fallback, j=j)

    return format_bias_aware_summary(res, label=label)


def sensitivity_benchmark(
    effect_estimation: Dict[str, Any] | Any,
    benchmarking_set: List[str],
    fit_args: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    model = None
    if isinstance(effect_estimation, dict):
        model = effect_estimation.get("model")
    elif hasattr(effect_estimation, "_model"):
        model = getattr(effect_estimation, "_model")
    elif _is_estimate_like(effect_estimation):
        diag = getattr(effect_estimation, "diagnostic_data", None)
        model = getattr(diag, "_model", None)
    elif _is_model_like(effect_estimation):
        model = effect_estimation

    if model is None:
        raise TypeError(
            "effect_estimation must be a dict with 'model', a MultiCausalEstimate with diagnostic_data._model, "
            "or a fitted MultiTreatmentIRM-like model."
        )

    required_attrs = [
        "data",
        "coef_",
        "se_",
        "_sensitivity_element_est",
        "g_hat_",
        "m_hat_",
    ]
    missing = [attr for attr in required_attrs if not hasattr(model, attr)]
    if missing:
        raise NotImplementedError(
            "Sensitivity benchmarking requires a fitted MultiTreatmentIRM-like model. "
            f"Missing attributes: {missing}"
        )

    if not isinstance(benchmarking_set, list):
        raise TypeError(
            f"benchmarking_set must be a list. {benchmarking_set} of type {type(benchmarking_set)} was passed."
        )
    if len(benchmarking_set) == 0:
        raise ValueError("benchmarking_set must not be empty.")
    if fit_args is not None and not isinstance(fit_args, dict):
        raise TypeError(f"fit_args must be a dict. {fit_args} of type {type(fit_args)} was passed.")

    data_long = getattr(model, "data", getattr(model, "data_contracts", None))
    if not isinstance(data_long, MultiCausalData):
        raise TypeError("model.data must be MultiCausalData for multi-treatment sensitivity benchmarking.")

    x_list_long = list(getattr(data_long, "confounders", []))
    if not set(benchmarking_set) <= set(x_list_long):
        raise ValueError(
            f"benchmarking_set must be a subset of features {x_list_long}. "
            f"{benchmarking_set} was passed."
        )

    x_list_short = [x for x in x_list_long if x not in benchmarking_set]
    if len(x_list_short) == 0:
        raise ValueError("After removing benchmarking_set there are no confounders left to fit the short model.")

    from causalis.scenarios.multi_unconfoundedness.model import MultiTreatmentIRM

    df_long = data_long.get_df()
    data_short = MultiCausalData(
        df=df_long,
        outcome=str(data_long.outcome),
        treatment_names=list(data_long.treatment_names),
        confounders=x_list_short,
        control_treatment=str(data_long.control_treatment),
        user_id=data_long.user_id,
    )

    irm_short = MultiTreatmentIRM(
        data=data_short,
        ml_g=model.ml_g,
        ml_m=model.ml_m,
        n_folds=int(getattr(model, "n_folds", 5)),
        n_rep=int(getattr(model, "n_rep", 1)),
        normalize_ipw=bool(getattr(model, "normalize_ipw", False)),
        trimming_rule=str(getattr(model, "trimming_rule", "truncate")),
        trimming_threshold=float(getattr(model, "trimming_threshold", 1e-2)),
        random_state=getattr(model, "random_state", None),
        n_jobs=int(getattr(model, "n_jobs", 1)),
    )

    irm_short.fit()
    estimate_args: Dict[str, Any] = dict(fit_args or {})
    if "score" not in estimate_args:
        estimate_args["score"] = getattr(model, "score", "ATE")
    if "diagnostic_data" not in estimate_args:
        # Benchmarking only needs short-model point estimates.
        estimate_args["diagnostic_data"] = False
    irm_short.estimate(**estimate_args)

    theta_long = np.asarray(model.coef_, dtype=float).reshape(-1)
    theta_short = np.asarray(irm_short.coef_, dtype=float).reshape(-1)
    if theta_long.size != theta_short.size:
        raise RuntimeError(
            "Long and short models returned a different number of contrasts: "
            f"{theta_long.size} vs {theta_short.size}."
        )

    y = np.asarray(getattr(model, "_y", df_long[data_long.outcome].to_numpy(dtype=float)), dtype=float).reshape(-1)
    d = np.asarray(
        getattr(model, "_d", df_long[list(data_long.treatments.columns)].to_numpy(dtype=int)),
        dtype=float,
    )
    g_hat = np.asarray(model.g_hat_, dtype=float)
    m_hat = np.asarray(model.m_hat_, dtype=float)
    z = df_long[benchmarking_set].to_numpy(dtype=float)

    if d.ndim != 2:
        raise RuntimeError(f"Expected one-hot treatment matrix with shape (n, K). Got shape {d.shape}.")
    n, k = d.shape
    if theta_long.size != k - 1:
        raise RuntimeError(
            f"Expected {k - 1} treatment contrasts from one-hot treatment matrix. Got {theta_long.size}."
        )

    # Outcome residual under observed arm.
    g_obs = np.sum(d * g_hat, axis=1)
    r_y = y - g_obs

    # Multi-arm treatment residuals as pairwise contrasts vs baseline.
    r_d = d - m_hat
    r_d_contrast = r_d[:, 1:] - r_d[:, [0]]

    def _ols_r2_and_fit(target: np.ndarray, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        target_arr = np.asarray(target, dtype=float)
        if target_arr.ndim == 1:
            target_arr = target_arr.reshape(-1, 1)
        features_arr = np.asarray(features, dtype=float)
        if features_arr.ndim == 1:
            features_arr = features_arr.reshape(-1, 1)

        features_centered = features_arr - np.nanmean(features_arr, axis=0, keepdims=True)
        feature_std = np.nanstd(features_centered, axis=0, ddof=0)
        valid_cols = np.isfinite(feature_std) & (feature_std > 1e-12)
        if not np.any(valid_cols):
            return np.zeros(target_arr.shape[1], dtype=float), np.zeros_like(target_arr)

        features_scaled = features_centered[:, valid_cols] / feature_std[valid_cols]
        features_scaled = np.nan_to_num(features_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        target_centered = target_arr - np.nanmean(target_arr, axis=0, keepdims=True)
        target_centered = np.nan_to_num(target_centered, nan=0.0, posinf=0.0, neginf=0.0)

        from numpy.linalg import lstsq

        beta, *_ = lstsq(features_scaled, target_centered, rcond=1e-12)
        target_hat = features_scaled @ beta

        denom = np.sum(target_centered * target_centered, axis=0)
        num = np.sum(target_hat * target_hat, axis=0)
        valid = np.isfinite(denom) & (denom > 1e-12) & np.isfinite(num) & (num >= 0.0)
        r2 = np.zeros(target_arr.shape[1], dtype=float)
        r2[valid] = np.clip(num[valid] / denom[valid], 0.0, 1.0)
        return r2, target_hat

    def _safe_corr(u: np.ndarray, v: np.ndarray) -> float:
        u_arr = np.asarray(u, dtype=float).reshape(-1)
        v_arr = np.asarray(v, dtype=float).reshape(-1)
        if u_arr.size != v_arr.size:
            raise ValueError("Correlation inputs must have the same length.")

        u_c = u_arr - np.mean(u_arr)
        v_c = v_arr - np.mean(v_arr)
        su = float(np.std(u_c))
        sv = float(np.std(v_c))
        if not (np.isfinite(su) and np.isfinite(sv)) or su <= 0.0 or sv <= 0.0:
            return 0.0
        val = float(np.corrcoef(u_c, v_c)[0, 1])
        return float(np.clip(val, -1.0, 1.0))

    r2_y_arr, yhat_u = _ols_r2_and_fit(r_y, z)
    r2_y = float(r2_y_arr[0])
    cf_y = float(r2_y / (1.0 - r2_y)) if r2_y < 1.0 else np.inf

    r2_d, dhat_u = _ols_r2_and_fit(r_d_contrast, z)
    rho = np.array([_safe_corr(yhat_u[:, 0], dhat_u[:, j]) for j in range(theta_long.size)], dtype=float)

    delta = theta_long - theta_short

    treatment_cols = list(data_long.treatments.columns)
    baseline = str(treatment_cols[0]) if treatment_cols else "d0"
    idx = [f"{str(name)} vs {baseline}" for name in treatment_cols[1:]]
    if len(idx) != theta_long.size:
        idx = [f"contrast_{j + 1}" for j in range(theta_long.size)]

    return pd.DataFrame(
        {
            "cf_y": np.repeat(cf_y, theta_long.size),
            "r2_y": np.repeat(r2_y, theta_long.size),
            "r2_d": r2_d,
            "rho": rho,
            "theta_long": theta_long,
            "theta_short": theta_short,
            "delta": delta,
        },
        index=idx,
    )


def _resolve_cf_y_alias(
    *,
    cf_y: Optional[float],
    r2_y: Optional[float],
) -> float:
    cf_y_value = 0.0 if cf_y is None else float(cf_y)
    if cf_y_value < 0.0:
        raise ValueError("cf_y must be >= 0.")

    if r2_y is None:
        return cf_y_value

    r2_y_value = float(r2_y)
    if not (0.0 <= r2_y_value < 1.0):
        raise ValueError("r2_y must be in [0, 1).")

    cf_y_from_r2 = r2_y_value / (1.0 - r2_y_value)

    if cf_y is None:
        return cf_y_from_r2

    if not np.isclose(cf_y_value, cf_y_from_r2, rtol=1e-10, atol=1e-12):
        raise ValueError(
            "cf_y and r2_y are both set but inconsistent. "
            f"Expected cf_y={cf_y_from_r2:.12g} from r2_y={r2_y_value:.12g}, got cf_y={cf_y_value:.12g}."
        )
    return cf_y_value


def sensitivity_analysis(
    effect_estimation: Dict[str, Any] | Any,
    _=None,
    cf_y: Optional[float] = None,
    r2_y: Optional[float] = None,
    r2_d: Union[float, np.ndarray] = 0.0,
    rho: Union[float, np.ndarray] = 1.0,
    H0: float = 0.0,
    alpha: float = 0.05,
    use_signed_rr: bool = False,
) -> Dict[str, Any]:
    cf_y_resolved = _resolve_cf_y_alias(cf_y=cf_y, r2_y=r2_y)

    res = compute_bias_aware_ci(
        effect_estimation,
        _=_,
        cf_y=cf_y_resolved,
        r2_d=r2_d,
        rho=rho,
        H0=H0,
        alpha=alpha,
        use_signed_rr=use_signed_rr,
    )

    ctx = _resolve_input_context(effect_estimation, context=_)
    diag = ctx["diag"]

    if isinstance(effect_estimation, dict):
        effect_estimation["bias_aware"] = res

    if isinstance(diag, dict):
        diag["sensitivity_analysis"] = res
    elif diag is not None:
        try:
            diag.sensitivity_analysis = res
        except Exception:
            pass

    return res
