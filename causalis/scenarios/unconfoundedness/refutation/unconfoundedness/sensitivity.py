r"""
Sensitivity analysis and benchmarking for unconfoundedness.

This module provides tools to assess the robustness of causal estimates to
potential unobserved confounding. It implements the sensitivity framework
based on partial $R^2$ as described in Cinelli & Hazlett (2020), adapted for
non-linear and semi-parametric models via influence functions.

The framework assumes the existence of an unobserved confounder $U$ and
quantifies the bias as a function of:
- $R^2_{Y \sim U | D, X}$: How much of the outcome variance $U$ explains.
- $R^2_{D \sim U | X}$: How much of the treatment variance $U$ explains.

The bias-aware estimate is then:

.. math::

    \hat{\tau}_{adj} = \hat{\tau} \pm \text{bias}(R^2_{Y \sim U}, R^2_{D \sim U})
"""
from __future__ import annotations

from typing import Dict, Any, Optional, List, Literal, Mapping

import numpy as np
import pandas as pd

from causalis.data_contracts.causal_diagnostic_data import UnconfoundednessDiagnosticData
from causalis.data_contracts.sensitivity_analysis_result import SensitivityAnalysisResult
from causalis.dgp.causaldata import CausalData
from causalis.scenarios.unconfoundedness._score_utils import _compute_ipw_components

__all__ = [
    "sensitivity_analysis",
    "sensitivity_benchmark",
    "sensitivity_benchmark_group",
    "run_sensitivity_protocol",
    "get_sensitivity_summary",
    "interpret_sensitivity_analysis",
]

# ---------------- Internals ----------------

_ESSENTIALLY_ZERO = 1e-32


def _resolve_sensitivity_label(
    effect_estimation: Dict[str, Any] | Any,
    *,
    model: Any | None = None,
) -> str:
    """Resolve the row label used for single-treatment sensitivity summaries."""
    treatment = getattr(effect_estimation, "treatment", None)
    if isinstance(treatment, str):
        return treatment

    if isinstance(effect_estimation, dict):
        estimate = effect_estimation.get("estimate")
        treatment = getattr(estimate, "treatment", None)
        if isinstance(treatment, str):
            return treatment
        model = effect_estimation.get("model", model)

    data_obj = getattr(model, "data", getattr(model, "data_contracts", None))
    treatment = getattr(data_obj, "treatment", None)
    if hasattr(treatment, "name"):
        return str(treatment.name)
    if isinstance(treatment, str):
        return treatment
    return "theta"


def _normalize_ate_atte_score(score: Any) -> str:
    """Normalize score aliases and enforce ATE/ATTE-only semantics."""
    score_u = str(score or "ATE").upper()
    if "ATT" in score_u:
        return "ATTE"
    if score_u == "ATE":
        return "ATE"
    raise ValueError(
        "Sensitivity analysis supports only score='ATE' or score='ATTE'. "
        f"Got {score!r}."
    )


def _resolve_sensitivity_score(effect_estimation: Any, model: Any, explicit_score: Any | None) -> str:
    """Resolve benchmark/sensitivity score without coupling to model.score=GATE state."""
    if explicit_score is not None:
        return _normalize_ate_atte_score(explicit_score)

    score_candidates: list[Any] = []

    if hasattr(effect_estimation, "estimand"):
        score_candidates.append(getattr(effect_estimation, "estimand", None))

    diag = getattr(effect_estimation, "diagnostic_data", None)
    if diag is not None and hasattr(diag, "score"):
        score_candidates.append(getattr(diag, "score", None))

    if hasattr(model, "score"):
        score_candidates.append(getattr(model, "score", None))

    for cand in score_candidates:
        if cand is None:
            continue
        try:
            return _normalize_ate_atte_score(cand)
        except ValueError:
            continue

    return "ATE"


def compute_irm_sensitivity_elements(
    *,
    model: Any,
    y: np.ndarray,
    d: np.ndarray,
    g0: np.ndarray,
    g1: np.ndarray,
    m_hat: np.ndarray,
    w: Optional[np.ndarray] = None,
    w_bar: Optional[np.ndarray] = None,
    psi: Optional[np.ndarray] = None,
    inv_m: Optional[np.ndarray] = None,
    inv_1m: Optional[np.ndarray] = None,
    score: Any = "ATE",
) -> dict[str, Any]:
    """Compute DoubleML-style sensitivity elements for binary-treatment IRM."""
    y_arr = np.asarray(y, dtype=float).ravel()
    d_arr = np.asarray(d, dtype=int).ravel()
    g0_arr = np.asarray(g0, dtype=float).ravel()
    g1_arr = np.asarray(g1, dtype=float).ravel()
    m_hat_arr = np.asarray(m_hat, dtype=float).ravel()

    n = y_arr.size
    if any(arr.size != n for arr in (d_arr, g0_arr, g1_arr, m_hat_arr)):
        raise ValueError("y, d, g0, g1, and m_hat must share the same sample size.")

    if w is None or w_bar is None:
        if model is None or not hasattr(model, "_get_weights"):
            raise RuntimeError("IRM sensitivity elements require model._get_weights when weights are omitted.")
        w, w_bar = model._get_weights(n=n, m_hat_adj=m_hat_arr, d=d_arr, score=score)

    w_arr = np.asarray(w, dtype=float).ravel()
    w_bar_arr = np.asarray(w_bar, dtype=float).ravel()
    if w_arr.size != n or w_bar_arr.size != n:
        raise ValueError("w and w_bar must share the same sample size as y.")

    if psi is None and model is not None:
        psi = getattr(model, "psi_", None)
    psi_arr = None if psi is None else np.asarray(psi, dtype=float)

    if inv_m is None or inv_1m is None:
        normalize_ipw = bool(getattr(model, "normalize_ipw", False)) if model is not None else False
        _, _, inv_m_arr, inv_1m_arr = _compute_ipw_components(
            d=d_arr,
            m_hat=m_hat_arr,
            normalize_ipw=normalize_ipw,
            score=score,
            warn=False,
        )
    else:
        inv_m_arr = np.asarray(inv_m, dtype=float).ravel()
        inv_1m_arr = np.asarray(inv_1m, dtype=float).ravel()

    if inv_m_arr.size != n or inv_1m_arr.size != n:
        raise ValueError("inv_m and inv_1m must share the same sample size as y.")

    sigma2_score_element = np.square(y_arr - d_arr * g1_arr - (1.0 - d_arr) * g0_arr)
    sigma2 = float(np.mean(sigma2_score_element))
    psi_sigma2 = sigma2_score_element - sigma2

    m_alpha = (w_bar_arr ** 2) * (inv_m_arr + inv_1m_arr)
    rr = w_bar_arr * (d_arr * inv_m_arr - (1.0 - d_arr) * inv_1m_arr)
    nu2_score_element = 2.0 * m_alpha - np.square(rr)
    nu2 = float(np.mean(nu2_score_element))
    psi_nu2 = nu2_score_element - nu2

    return {
        "sigma2": sigma2,
        "nu2": nu2,
        "psi_sigma2": psi_sigma2,
        "psi_nu2": psi_nu2,
        "riesz_rep": rr,
        "m_alpha": m_alpha,
        "psi": psi_arr,
    }


# ---------------- Core sensitivity primitives (public, legacy-compatible) ----------------

def _compute_sensitivity_bias_unified(
    sigma2: np.ndarray | float,
    nu2: np.ndarray | float,
    psi_sigma2: np.ndarray,
    psi_nu2: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Compute max bias and its influence function.

    max_bias = sqrt(max(sigma2 * nu2, 0)). Influence function via delta method.
    Returns zero IF on the boundary and an IF shaped like psi_sigma2 otherwise.

    Parameters
    ----------
    sigma2 : np.ndarray or float
        Variance of the outcome residuals.
    nu2 : np.ndarray or float
        Variance related to the Riesz representer.
    psi_sigma2 : np.ndarray
        Influence function for sigma2.
    psi_nu2 : np.ndarray
        Influence function for nu2.

    Returns
    -------
    max_bias : float
        The maximum bias.
    psi_max_bias : np.ndarray
        The influence function for the maximum bias.
    """
    sigma2_f = float(np.asarray(sigma2).reshape(()))
    nu2_f = float(np.asarray(nu2).reshape(()))
    if not (sigma2_f > 0.0 and nu2_f > 0.0):
        return 0.0, np.zeros_like(psi_sigma2, dtype=float)
    max_bias = float(np.sqrt(sigma2_f * nu2_f))
    denom = 2.0 * max_bias if max_bias > _ESSENTIALLY_ZERO else 1.0
    psi_sigma2 = np.asarray(psi_sigma2, float)
    psi_sigma2 = psi_sigma2 - float(np.mean(psi_sigma2))
    psi_nu2 = np.asarray(psi_nu2, float)
    psi_nu2 = psi_nu2 - float(np.mean(psi_nu2))
    psi_max_bias = (sigma2_f * psi_nu2 + nu2_f * psi_sigma2) / denom
    return max_bias, psi_max_bias

# Backward-compatible alias
def _compute_sensitivity_bias(
    sigma2: np.ndarray | float,
    nu2: np.ndarray | float,
    psi_sigma2: np.ndarray,
    psi_nu2: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Backward-compatible alias for _compute_sensitivity_bias_unified.

    Parameters
    ----------
    sigma2 : np.ndarray or float
        Variance of the outcome residuals.
    nu2 : np.ndarray or float
        Variance related to the Riesz representer.
    psi_sigma2 : np.ndarray
        Influence function for sigma2.
    psi_nu2 : np.ndarray
        Influence function for nu2.

    Returns
    -------
    tuple
        (max_bias, psi_max_bias)
    """
    return _compute_sensitivity_bias_unified(sigma2, nu2, psi_sigma2, psi_nu2)


def _combine_nu2(m_alpha: np.ndarray, rr: np.ndarray, r2_y: float, r2_d: float, rho: float) -> tuple[float, np.ndarray]:
    """Combine sensitivity levers into nu2 via per-unit quadratic form.

    nu2_i = (sqrt(2*m_alpha_i))^2 * cf_y + (|rr_i|)^2 * (r2_d/(1-r2_d)) + 2*rho*sqrt(cf_y*r2_d/(1-r2_d))*|rr_i|*sqrt(2*m_alpha_i)
    with cf_y = r2_y / (1 - r2_y).
    Returns (nu2, psi_nu2) with psi_nu2 centered.

    Note: we use abs(rr) for a conservative worst-case cross-term; the quadratic
    form is PSD for signed rr as well, but abs() avoids reductions when rr < 0.

    Parameters
    ----------
    m_alpha : np.ndarray
        Component for the representer variance.
    rr : np.ndarray
        Riesz representer.
    r2_y : float
        Sensitivity parameter for the outcome (R^2 form, R_Y^2; converted to odds form internally).
    r2_d : float
        Sensitivity parameter for the treatment (R^2 form, R_D^2).
    rho : float
        Correlation parameter.

    Returns
    -------
    nu2 : float
        The combined nu2 value.
    psi_nu2 : np.ndarray
        The centered influence function for nu2.
    """
    r2_y = float(r2_y)
    r2_d = float(r2_d)
    rho = float(np.clip(rho, -1.0, 1.0))
    if r2_y < 0 or r2_d < 0:
        raise ValueError("r2_y and r2_d must be >= 0.")
    if r2_y >= 1.0:
        raise ValueError("r2_y must be < 1.0.")
    if r2_d >= 1.0:
        raise ValueError("r2_d must be < 1.0.")
    
    cf_y = r2_y / (1.0 - r2_y)
    cf_d = r2_d / (1.0 - r2_d)
    a = np.sqrt(2.0 * np.maximum(np.asarray(m_alpha, dtype=float), 0.0))
    b = np.abs(np.asarray(rr, dtype=float))
    base = (a * a) * cf_y + (b * b) * cf_d + 2.0 * rho * np.sqrt(cf_y * cf_d) * a * b
    # numeric PSD clamp
    base = np.maximum(base, 0.0)
    nu2 = float(np.mean(base))
    psi_nu2 = base - nu2
    return nu2, psi_nu2


# ---------------- Bias-aware helpers (local variants + pullers) ----------------

def _combine_nu2_local(m_alpha: np.ndarray, rr: np.ndarray, r2_y: float, r2_d: float, rho: float, *, use_signed_rr: bool) -> tuple[float, np.ndarray]:
    """Nu^2 via per-unit quadratic form with optional sign-preserving rr.

    Parameters
    ----------
    m_alpha : np.ndarray
        Component for the representer variance.
    rr : np.ndarray
        Riesz representer.
    r2_y : float
        Sensitivity parameter for the outcome (R^2 form, R_Y^2).
    r2_d : float
        Sensitivity parameter for the treatment (R^2 form, R_D^2).
    rho : float
        Correlation parameter.
    use_signed_rr : bool
        Whether to use signed rr or absolute value.

    Returns
    -------
    nu2 : float
        The combined nu2 value.
    psi_nu2 : np.ndarray
        The centered influence function for nu2.
    """
    r2_y = float(r2_y); r2_d = float(r2_d); rho = float(np.clip(rho, -1.0, 1.0))
    if r2_y < 0 or r2_d < 0:
        raise ValueError("r2_y and r2_d must be >= 0.")
    if r2_y >= 1.0:
        raise ValueError("r2_y must be < 1.0.")
    if r2_d >= 1.0:
        raise ValueError("r2_d must be < 1.0.")
    
    cf_y = r2_y / (1.0 - r2_y)
    cf_d = r2_d / (1.0 - r2_d)
    a = np.sqrt(2.0 * np.maximum(np.asarray(m_alpha, float), 0.0))
    b = np.asarray(rr, float)
    if not use_signed_rr:
        b = np.abs(b)  # worst-case sign
    base = (a * a) * cf_y + (b * b) * cf_d + 2.0 * rho * np.sqrt(cf_y * cf_d) * a * b
    base = np.maximum(base, 0.0)
    nu2 = float(np.mean(base))
    psi_nu2 = base - nu2
    return nu2, psi_nu2




def _compute_sensitivity_bias_local(
    sigma2: float,
    nu2: float,
    psi_sigma2: np.ndarray,
    psi_nu2: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Backward-compatible wrapper delegating to unified helper.

    Parameters
    ----------
    sigma2 : float
        Variance of the outcome residuals.
    nu2 : float
        Variance related to the Riesz representer.
    psi_sigma2 : np.ndarray
        Influence function for sigma2.
    psi_nu2 : np.ndarray
        Influence function for nu2.

    Returns
    -------
    tuple
        (max_bias, psi_max_bias)
    """
    return _compute_sensitivity_bias_unified(sigma2, nu2, psi_sigma2, psi_nu2)


def _pull_theta_se_ci(effect_estimation: Any, alpha: float) -> tuple[float, float, tuple[float, float]]:
    """Robustly extract theta, se, and sampling CI from CausalEstimate, dict, or model.

    Parameters
    ----------
    effect_estimation : Any
        The effect estimation object (CausalEstimate, dict, or model).
    alpha : float
        Significance level.

    Returns
    -------
    theta : float
        The estimated effect.
    se : float
        The standard error.
    ci : tuple of float
        The confidence interval (lower, upper).
    """
    from scipy.stats import norm as _norm
    
    # 1. CausalEstimate
    if hasattr(effect_estimation, "value") and hasattr(effect_estimation, "ci_lower_absolute"):
        theta = float(effect_estimation.value)
        # Try to get SE from model_options
        opts = getattr(effect_estimation, "model_options", {})
        se = float(opts.get("std_error", 0.0))
        if se == 0.0 and hasattr(effect_estimation, "ci_upper_absolute"):
            # Fallback: back-calculate SE from CI if missing.
            # CI on CausalEstimate is tied to estimate.alpha, not the requested alpha.
            alpha_ci = float(getattr(effect_estimation, "alpha", alpha))
            z = float(_norm.ppf(1 - alpha_ci / 2.0))
            se = (float(effect_estimation.ci_upper_absolute) - theta) / z if z > 0 else 0.0
        ci = (float(effect_estimation.ci_lower_absolute), float(effect_estimation.ci_upper_absolute))
        return theta, se, ci

    # 2. Dict (legacy)
    if isinstance(effect_estimation, dict):
        model = effect_estimation.get('model')
        # theta
        try:
            theta = float(effect_estimation.get('coefficient', getattr(model, 'coef_', [0.0])[0]))
        except Exception:
            theta = 0.0
        # se
        try:
            se = float(effect_estimation.get('std_error', getattr(model, 'se_', [0.0])[0]))
        except Exception:
            se = 0.0
        # sampling CI
        ci = effect_estimation.get('confidence_interval', None)
        if ci is None and hasattr(model, 'confint'):
            try:
                ci_df = model.confint(alpha=alpha)
                if isinstance(ci_df, pd.DataFrame):
                    lower = None; upper = None
                    for col in ['ci_lower', f"{alpha/2*100:.1f} %", '2.5 %', '2.5%']:
                        if col in ci_df.columns:
                            lower = float(ci_df[col].iloc[0]); break
                    for col in ['ci_upper', f"{(1-alpha/2)*100:.1f} %", '97.5 %', '97.5%']:
                        if col in ci_df.columns:
                            upper = float(ci_df[col].iloc[0]); break
                    if lower is None or upper is None:
                        lower = float(ci_df.iloc[0, 0]); upper = float(ci_df.iloc[0, 1])
                    ci = (lower, upper)
            except Exception:
                pass
        if ci is None:
            z = _norm.ppf(1 - alpha / 2.0)
            ci = (theta - z*se, theta + z*se)
        return float(theta), float(se), (float(ci[0]), float(ci[1]))
    
    # 3. Model instance
    if hasattr(effect_estimation, "coef_") and hasattr(effect_estimation, "se_"):
        theta = float(effect_estimation.coef_[0])
        se = float(effect_estimation.se_[0])
        z = _norm.ppf(1 - alpha / 2.0)
        return theta, se, (theta - z*se, theta + z*se)

    return 0.0, 0.0, (0.0, 0.0)


# ---------------- Public API: bias-aware CI and text summaries ----------------

def compute_bias_aware_ci(
    effect_estimation: Dict[str, Any] | Any,
    *,
    r2_y: float,
    r2_d: float,
    rho: float = 1.0,
    H0: float = 0.0,
    alpha: float = 0.05,
    use_signed_rr: bool = False
) -> Dict[str, Any]:
    """Compute bias-aware confidence intervals.

    Returns a dict with:
      - theta, se, alpha, z
      - sampling_ci
      - theta_bounds_cofounding = [theta_lower, theta_upper] = theta ± bound_width
      - bias_aware_ci = [theta - (bound_width + z*se), theta + (bound_width + z*se)]
      - max_bias_base, max_bias, bound_width and components (sigma2, nu2)

    Parameters
    ----------
    effect_estimation : Dict[str, Any] or Any
        The effect estimation object.
    r2_y : float
        Sensitivity parameter for the outcome (R^2 form, R_Y^2).
    r2_d : float
        Sensitivity parameter for the treatment (R^2 form, R_D^2).
    rho : float, default 1.0
        Correlation parameter.
    H0 : float, default 0.0
        Null hypothesis for robustness values.
    alpha : float, default 0.05
        Significance level.
    use_signed_rr : bool, default False
        Whether to use signed rr in the quadratic combination of sensitivity components.
        If True and m_alpha/rr are available, the bias bound is computed via the
        per-unit quadratic form and RV/RVa are not reported.

    Returns
    -------
    dict
        Dictionary with bias-aware results.
    """
    from scipy.stats import norm as _norm

    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if r2_y < 0 or r2_d < 0:
        raise ValueError("r2_y and r2_d must be >= 0")
    if r2_y >= 1.0:
        raise ValueError("r2_y must be < 1.0 for the bias factor to be well-defined")
    if r2_d >= 1.0:
        raise ValueError("r2_d must be < 1.0 for the bias factor to be well-defined")

    if isinstance(effect_estimation, dict):
        if 'model' not in effect_estimation:
             raise TypeError("Pass the usual result dict with a fitted model under key 'model'.")
        effect_dict = effect_estimation
    elif hasattr(effect_estimation, "coef_") and hasattr(effect_estimation, "se_"):
        # Likely an IRM instance
        model = effect_estimation
        effect_dict = {'model': model}
    elif hasattr(effect_estimation, "value") and hasattr(effect_estimation, "diagnostic_data"):
        # CausalEstimate path
        effect_dict = {'model': None}
    else:
        raise TypeError("effect_estimation must be a dict, CausalEstimate, or an IRM-like model instance.")

    theta, se, sampling_ci = _pull_theta_se_ci(effect_estimation, alpha)
    z = float(_norm.ppf(1 - alpha / 2.0))

    model = effect_dict.get('model')
    # Try to extract elements from diagnostic data first
    diag = getattr(effect_estimation, 'diagnostic_data', effect_dict.get('diagnostic_data'))
    
    elems = None
    if diag is not None and hasattr(diag, "sigma2") and getattr(diag, "sigma2", None) is not None:
        elems = {
            "sigma2": diag.sigma2,
            "nu2": diag.nu2,
            "psi_sigma2": diag.psi_sigma2,
            "psi_nu2": diag.psi_nu2,
            "riesz_rep": diag.riesz_rep,
            "m_alpha": diag.m_alpha,
            "psi": diag.psi,
        }
    elif hasattr(model, "_sensitivity_element_est"):
        elems = model._sensitivity_element_est()

    # Default: no cofounding info → bias_aware = sampling CI
    max_bias_base = 0.0
    max_bias = 0.0
    sigma2 = np.nan; nu2 = np.nan
    rv = np.nan; rva = np.nan
    bound_width = 0.0
    correction_scale = None
    use_signed_rr_effective = bool(use_signed_rr)

    if elems:
        sigma2 = float(elems.get("sigma2", np.nan))
        nu2 = float(elems.get("nu2", np.nan))
        psi_sigma2 = elems.get("psi_sigma2", None)
        psi_nu2 = elems.get("psi_nu2", None)

        # Optional signed-rr quadratic form (uses r2_y/r2_d/rho internally).
        if use_signed_rr_effective:
            m_alpha = elems.get("m_alpha", None)
            rr = elems.get("riesz_rep", None)
            if m_alpha is not None and rr is not None:
                nu2, psi_nu2 = _combine_nu2_local(
                    m_alpha, rr, r2_y, r2_d, rho, use_signed_rr=True
                )
                max_bias_base = np.sqrt(max(sigma2 * nu2, 0.0))
                max_bias = max_bias_base
                bound_width = max_bias
                correction_scale = 1.0
                # RV/RVa are not defined under the signed-rr quadratic form
                rv = np.nan
                rva = np.nan
            else:
                use_signed_rr_effective = False

        if not use_signed_rr_effective:
            # Bias component: |rho| * sqrt(sigma2 * nu2) * sqrt(cf_y * r2_d / (1 - r2_d))
            cf_y = r2_y / (1.0 - r2_y)
            bias_factor = np.sqrt(cf_y * r2_d / (1.0 - r2_d))
            max_bias_base = np.sqrt(max(sigma2 * nu2, 0.0))
            max_bias = max_bias_base * bias_factor
            bound_width = abs(rho) * max_bias
            correction_scale = abs(rho) * bias_factor

            # Robustness Values (RV/RVa)
            # RV is the confounding strength that makes the bound include H0:
            # |theta - H0| = |rho| * sqrt(sigma2 * nu2) * RV / (1 - RV)
            delta_theta = abs(theta - H0)
            denom_rv = abs(rho) * max_bias_base
            if denom_rv > 1e-16 and delta_theta > 0:
                D = delta_theta / denom_rv
                rv = D / (1.0 + D)
            else:
                rv = 0.0 if delta_theta == 0 else np.nan

            delta_theta_a = max(abs(theta - H0) - z * se, 0.0)
            if denom_rv > 1e-16 and delta_theta_a > 0:
                Da = delta_theta_a / denom_rv
                rva = Da / (1.0 + Da)
            elif delta_theta_a == 0:
                rva = 0.0
            else:
                rva = np.nan
    else:
        # No cofounding info: keep max_bias=0; RV/RVa undefined unless delta=0
        delta_theta = abs(theta - H0)
        rv = 0.0 if delta_theta == 0 else np.nan
        delta_theta_a = max(abs(theta - H0) - z * se, 0.0)
        rva = 0.0 if delta_theta_a == 0 else np.nan

    # Bounds: theta ± bound_width (bound_width already includes rho/bias factors as applicable)
    theta_lower = float(theta) - float(bound_width)
    theta_upper = float(theta) + float(bound_width)

    # Graceful fallback: if se is non-finite, report cofounding bounds only
    if not (np.isfinite(se) and se >= 0.0 and np.isfinite(z)):
        bias_aware_ci = (theta_lower, theta_upper)
    elif elems and all(k in elems for k in ('psi', 'psi_sigma2', 'psi_nu2')):
        # Faithful inference for the bounds using orthogonal scores
        psi = np.asarray(elems['psi'])
        psi_sigma2 = np.asarray(psi_sigma2 if psi_sigma2 is not None else elems['psi_sigma2'])
        psi_nu2 = np.asarray(psi_nu2 if psi_nu2 is not None else elems['psi_nu2'])
        n = len(psi)

        if sigma2 * nu2 > 0 and correction_scale is not None:
            correction = (correction_scale / (2.0 * np.sqrt(sigma2 * nu2))) * (
                sigma2 * psi_nu2 + nu2 * psi_sigma2
            )
            psi_plus = psi + correction
            psi_minus = psi - correction
            se_lower = np.sqrt(np.var(psi_minus, ddof=1) / n)
            se_upper = np.sqrt(np.var(psi_plus, ddof=1) / n)
        else:
            se_lower = se
            se_upper = se

        bias_aware_ci = (
            float(theta_lower) - z * float(se_lower),
            float(theta_upper) + z * float(se_upper)
        )
    else:
        # bias-aware CI (approximate if scores not available)
        bias_aware_ci = (
            float(theta_lower) - z * float(se),
            float(theta_upper) + z * float(se),
        )

    return dict(
        theta=float(theta),
        se=float(se),
        alpha=float(alpha),
        z=z,
        H0=float(H0),
        sampling_ci=tuple(map(float, sampling_ci)),
        theta_bounds_cofounding=(float(theta_lower), float(theta_upper)),
        bias_aware_ci=tuple(map(float, bias_aware_ci)),
        max_bias_base=float(max_bias_base),
        max_bias=float(max_bias),
        bound_width=float(bound_width),
        sigma2=float(sigma2),
        nu2=float(nu2),
        rv=float(rv),
        rva=float(rva),
        params=dict(r2_y=float(r2_y), r2_d=float(r2_d), rho=float(np.clip(rho, -1.0, 1.0)), use_signed_rr=bool(use_signed_rr_effective)),
    )


def _summary_dataframe_from_result(
    res: Dict[str, Any],
    *,
    label: str | None = None,
) -> pd.DataFrame:
    """Build a simple two-column sensitivity summary."""
    def _round_value(value: Any) -> Any:
        if isinstance(value, list):
            return [round(float(item), 4) if pd.notna(item) else None for item in value]
        if pd.isna(value):
            return np.nan
        return round(float(value), 4)

    sampling_ci = tuple(map(float, res.get("sampling_ci", (np.nan, np.nan))))
    theta_bounds = tuple(map(float, res.get("theta_bounds_cofounding", (np.nan, np.nan))))
    bias_aware_ci = tuple(map(float, res.get("bias_aware_ci", (np.nan, np.nan))))
    return pd.DataFrame(
        {
            "statistics": [
                "bias_aware_ci",
                "theta",
                "sampling_ci",
                "rv",
                "rva",
                "se",
                "max_bias",
                "max_bias_base",
                "bound_width",
                "sigma2",
                "nu2",
            ],
            "value": [
                _round_value([bias_aware_ci[0], bias_aware_ci[1]]),
                _round_value([theta_bounds[0], float(res.get("theta", np.nan)), theta_bounds[1]]),
                _round_value([sampling_ci[0], sampling_ci[1]]),
                _round_value(float(res.get("rv", np.nan))),
                _round_value(float(res.get("rva", np.nan))),
                _round_value(float(res.get("se", np.nan))),
                _round_value(float(res.get("max_bias", np.nan))),
                _round_value(float(res.get("max_bias_base", np.nan))),
                _round_value(float(res.get("bound_width", np.nan))),
                _round_value(float(res.get("sigma2", np.nan))),
                _round_value(float(res.get("nu2", np.nan))),
            ],
        }
    )


def _format_summary_table(df: pd.DataFrame) -> str:
    """Render summary tables while formatting floats inside scalar/list values."""
    display_df = df.copy().astype(object)

    def _fmt(value: Any) -> Any:
        if isinstance(value, list):
            return [round(float(item), 6) if pd.notna(item) else None for item in value]
        if pd.isna(value):
            return ""
        if isinstance(value, (int, float, np.floating)):
            return f"{float(value):.6f}"
        return value

    for col in display_df.columns:
        display_df[col] = display_df[col].map(_fmt)
    return display_df.to_string()


def format_bias_aware_summary(res: Dict[str, Any], label: str | None = None) -> str:
    """Render a single, unified bias-aware summary string.

    Parameters
    ----------
    res : Dict[str, Any]
        The result dictionary from compute_bias_aware_ci.
    label : str, optional, default None
        The label for the estimand.

    Returns
    -------
    str
        Formatted summary string.
    """
    alpha = res['alpha']
    cf = res['params']
    summary_df = _summary_dataframe_from_result(res, label=label)

    lines = []
    lines.append("================== Bias-aware Interval ==================")
    lines.append("")
    lines.append("------------------ Scenario          ------------------")
    lines.append(f"Significance Level: alpha={alpha}")
    lines.append(f"Null Hypothesis: H0={res.get('H0', 0.0)}")
    lines.append(f"Sensitivity parameters: r2_y={cf['r2_y']}; r2_d={cf['r2_d']}, rho={cf['rho']}, use_signed_rr={cf['use_signed_rr']}")
    lines.append("")
    lines.append(_format_summary_table(summary_df))

    return "\n".join(lines)


def _wrap_sensitivity_result(
    res: Dict[str, Any],
    *,
    label: str | None = None,
) -> SensitivityAnalysisResult:
    """Attach rich summary behavior to a dict-compatible sensitivity result."""
    return SensitivityAnalysisResult(
        res,
        summary_builder=_summary_dataframe_from_result,
        text_summary_builder=format_bias_aware_summary,
        summary_kwargs={"label": label},
    )


# ---------------- Human-facing wrappers and legacy formatting ----------------

def _format_sensitivity_summary(
    summary: pd.DataFrame,
    r2_y: float,
    r2_d: float,
    rho: float,
    alpha: float
) -> str:
    """
    Format the sensitivity analysis summary into the expected output format.

    Parameters
    ----------
    summary : pd.DataFrame
        The sensitivity summary DataFrame
    r2_y : float
        Sensitivity parameter for the outcome equation (R^2 form, R_Y^2; converted to odds form internally)
    r2_d : float
        Sensitivity parameter for the treatment equation (R^2 form, R_D^2)
    rho : float
        Correlation parameter
    alpha : float
        Significance level

    Returns
    -------
    str
        Formatted sensitivity analysis report
    """
    # Create the formatted output
    output_lines = []
    output_lines.append("================== Sensitivity Analysis ==================")
    output_lines.append("")
    output_lines.append("------------------ Scenario          ------------------")
    output_lines.append(f"Significance Level: alpha={alpha}")
    output_lines.append(f"Sensitivity parameters: r2_y={r2_y}; r2_d={r2_d}, rho={rho}")
    output_lines.append("")

    # Bounds with CI section
    output_lines.append("------------------ Bounds with CI    ------------------")

    # Create header for the table
    header = f"{'':>6} {'CI lower':>11} {'theta lower':>12} {'theta':>15} {'theta upper':>12} {'CI upper':>13}"
    output_lines.append(header)

    # Extract values from summary DataFrame
    # The summary should contain bounds and confidence intervals
    lower_lbl = f"{alpha / 2 * 100:.1f} %"
    upper_lbl = f"{(1 - alpha / 2) * 100:.1f} %"
    for idx, row in summary.iterrows():
        # Format the row data_contracts - adjust column names based on actual output
        row_name = str(idx) if not isinstance(idx, str) else idx
        try:
            ci_lower = row.get('ci_lower', row.get(lower_lbl, row.get('2.5 %', row.get('2.5%', 0.0))))
            theta_lower = row.get('theta_lower', row.get('theta lower', row.get('lower_bound', row.get('lower', 0.0))))
            theta = row.get('theta', row.get('estimate', row.get('coef', 0.0)))
            theta_upper = row.get('theta_upper', row.get('theta upper', row.get('upper_bound', row.get('upper', 0.0))))
            ci_upper = row.get('ci_upper', row.get(upper_lbl, row.get('97.5 %', row.get('97.5%', 0.0))))
            row_str = f"{row_name:>6} {ci_lower:11.6f} {theta_lower:12.6f} {theta:15.6f} {theta_upper:12.6f} {ci_upper:13.6f}"
            output_lines.append(row_str)
        except (KeyError, AttributeError):
            # Fallback formatting if exact column names differ
            row_values = [f"{val:11.6f}" if isinstance(val, (int, float)) else f"{val:>11}"
                          for val in list(row.values)[:5]]
            row_str = f"{row_name:>6} " + " ".join(row_values)
            output_lines.append(row_str)

    output_lines.append("")

    # Robustness SNR proxy section
    output_lines.append("------------------ Robustness (risk proxy) -------------")

    # Create header for robustness values
    rob_header = f"{'':>6} {'H_0':>6} {'risk proxy (%)':>15} {'adj (%)':>8}"
    output_lines.append(rob_header)

    # Add robustness values if present, else placeholders
    for idx, row in summary.iterrows():
        row_name = str(idx) if not isinstance(idx, str) else idx
        try:
            h_0 = row.get('H_0', row.get('null_hypothesis', 0.0))
            rv = row.get('RV', row.get('robustness_value', 0.0))
            rva = row.get('RVa', row.get('robustness_value_adjusted', 0.0))
            rob_row = f"{row_name:>6} {h_0:6.1f} {rv:15.6f} {rva:8.6f}"
            output_lines.append(rob_row)
        except (KeyError, AttributeError):
            rob_row = f"{row_name:>6} {0.0:6.1f} {0.0:15.6f} {0.0:8.6f}"
            output_lines.append(rob_row)

    return "\n".join(output_lines)


def get_sensitivity_summary(
    effect_estimation: Dict[str, Any] | Any,
    *,
    label: Optional[str] = None,
) -> Optional[str]:
    """Render a single, unified bias-aware summary string.

    If bias-aware components are missing, shows a sampling-only variant with max_bias=0
    and then formats via `format_bias_aware_summary` for consistency.

    Parameters
    ----------
    effect_estimation : Dict[str, Any] or Any
        The effect estimation object.
    label : str, optional, default None
        The label for the estimand.

    Returns
    -------
    Optional[str]
        Formatted summary string or None if extraction fails.

    Notes
    -----
    The summary reports the point estimate, standard error, and confidence
    intervals under both the null (no unobserved confounding) and the assumed
    level of confounding ($R^2_Y, R^2_D$).

    It also includes the Robustness Value (RV), which is the minimum strength
    of confounding ($R^2_Y = R^2_D$) required to change the conclusion
    (e.g., make the estimate non-significant or zero).

    Examples
    --------
    >>> from causalis.scenarios.unconfoundedness.refutation.unconfoundedness import sensitivity_analysis, get_sensitivity_summary
    >>> # Assuming 'estimate' is a fitted CausalEstimate from IRM
    >>> res = sensitivity_analysis(estimate, r2_y=0.05, r2_d=0.05) # doctest: +SKIP
    >>> summary = get_sensitivity_summary(estimate) # doctest: +SKIP
    >>> print(summary) # doctest: +SKIP
    """
    if isinstance(effect_estimation, SensitivityAnalysisResult):
        return effect_estimation.text_summary()

    if isinstance(effect_estimation, dict):
        if 'model' not in effect_estimation:
            return None
        effect_dict = effect_estimation
    elif hasattr(effect_estimation, "coef_") and hasattr(effect_estimation, "se_"):
        # Likely an IRM instance
        effect_dict = {'model': effect_estimation}
    elif hasattr(effect_estimation, "value") and hasattr(effect_estimation, "diagnostic_data"):
        # CausalEstimate
        diag = effect_estimation.diagnostic_data
        bias_aware: Dict[str, Any] = {}
        if diag is not None:
            if isinstance(diag, UnconfoundednessDiagnosticData):
                bias_aware = diag.sensitivity_analysis or {}
            elif isinstance(diag, dict):
                bias_aware = diag.get("sensitivity_analysis") or diag.get("bias_aware") or {}
            else:
                bias_aware = (
                    getattr(diag, "sensitivity_analysis", None)
                    or getattr(diag, "bias_aware", None)
                    or {}
                )
        effect_dict = {
            'model': None,
            'diagnostic_data': diag,
            'bias_aware': bias_aware,
        }
    else:
        return None

    model = effect_dict['model']
    if label is None:
        label = _resolve_sensitivity_label(effect_estimation, model=model)

    res = effect_dict.get('bias_aware')

    # Build a sampling-only placeholder if needed.
    if not isinstance(res, dict) or not res:
        alpha_fallback = getattr(effect_estimation, "alpha", None)
        if alpha_fallback is None and isinstance(effect_estimation, dict):
            alpha_fallback = effect_estimation.get("alpha")
        try:
            alpha_fallback = float(alpha_fallback)
        except (TypeError, ValueError):
            alpha_fallback = 0.05
        if not (0.0 < alpha_fallback < 1.0):
            alpha_fallback = 0.05

        theta, se, ci = _pull_theta_se_ci(effect_estimation, alpha=alpha_fallback)
        from scipy.stats import norm
        z = float(norm.ppf(1 - alpha_fallback / 2.0))
        res = dict(
            theta=float(theta),
            se=float(se),
            alpha=float(alpha_fallback),
            z=z,
            H0=0.0,
            sampling_ci=(float(ci[0]), float(ci[1])),
            theta_bounds_cofounding=(float(theta), float(theta)),  # max_bias = 0
            bias_aware_ci=(float(theta - z * se), float(theta + z * se)),
            max_bias_base=0.0,
            max_bias=0.0,
            bound_width=0.0,
            sigma2=np.nan,
            nu2=np.nan,
            rv=np.nan,
            rva=np.nan,
            params=dict(r2_y=0.0, r2_d=0.0, rho=0.0, use_signed_rr=False),
        )

    # Single clean summary (reuse the one definitive formatter)
    return format_bias_aware_summary(res, label=label)


# ---------------- Benchmarking sensitivity (short vs long model) ----------------

_BENCHMARK_RESULT_COLUMNS = [
    "r2_y",
    "r2_d",
    "rho",
    "theta_long",
    "theta_short",
    "delta",
    "cf_y",
    "cf_d",
    "cf_y_raw",
    "cf_d_raw",
    "sigma2_long",
    "sigma2_short",
    "nu2_long",
    "nu2_short",
    "rho_raw",
    "rho_clipped",
    "cf_y_clipped",
    "cf_d_clipped",
    "rho_fallback",
    "boundary_calibration",
    "strengths_valid",
    "calibration_valid",
    "calibration_issue",
    "calibration_warning",
]


def _read_benchmark_sensitivity_elements(model: Any) -> tuple[float, float, str | None]:
    """Read the two scalar sensitivity elements needed for gain statistics."""
    try:
        elements = model._sensitivity_element_est()
        if not isinstance(elements, dict):
            raise TypeError("_sensitivity_element_est() did not return a dictionary")
        sigma2 = float(elements["sigma2"])
        nu2 = float(elements["nu2"])
    except (KeyError, TypeError, ValueError, RuntimeError, FloatingPointError) as exc:
        return np.nan, np.nan, f"sensitivity elements are unavailable: {exc}"
    return sigma2, nu2, None


def _resolve_benchmark_fold_assignments(
    model: Any,
    *,
    full_sample_size: int,
) -> tuple[np.ndarray | None, str | None]:
    """Resolve full-sample folds for aligned long/short benchmark refits.

    New IRM fits retain fold ids before applying the overlap policy. Older
    fitted objects may only expose ``folds_``; those ids are reusable when they
    still cover the full input sample. If a dropped legacy fit has a fixed
    random seed, leaving the assignments unset recreates the same stratified
    split on the unchanged full sample.
    """
    full_folds = getattr(model, "_full_sample_folds_", None)
    if full_folds is not None:
        full_folds_arr = np.asarray(full_folds, dtype=int).ravel()
        if full_folds_arr.size == full_sample_size:
            return full_folds_arr.copy(), None
        return None, (
            "stored full-sample cross-fitting assignments have inconsistent "
            f"length: expected {full_sample_size}, got {full_folds_arr.size}"
        )

    diagnostic_folds = getattr(model, "folds_", None)
    if diagnostic_folds is not None:
        diagnostic_folds_arr = np.asarray(diagnostic_folds, dtype=int).ravel()
        if diagnostic_folds_arr.size == full_sample_size:
            return diagnostic_folds_arr.copy(), None

    if getattr(model, "random_state", None) is not None:
        # The short model receives the same full sample, treatment vector,
        # n_folds, and seed, so StratifiedKFold recreates the long-model split.
        return None, None

    return None, (
        "long/short cross-fitting splits cannot be aligned because the long "
        "model has no full-sample fold assignments and random_state=None"
    )


def _calibrate_benchmark_gain_statistics(
    *,
    theta_long: float,
    theta_short: float,
    sigma2_long: float,
    sigma2_short: float,
    nu2_long: float,
    nu2_short: float,
    element_issue: str | None = None,
) -> dict[str, Any]:
    """Calibrate DoubleML-compatible gain statistics and Causalis R2 inputs.

    Raw ``cf_y`` and ``cf_d`` follow the long/short gain-statistic definitions.
    As in DoubleML, both are clipped to ``[0, 1]``. If the denominator for
    ``rho`` is not positive, ``rho`` falls back to the sign of the long/short
    effect shift (and therefore has magnitude one unless the shift is zero).
    The existing Causalis sensitivity API expects an R2 value for the outcome
    channel and converts it to odds internally, so ``r2_y`` is the inverse-odds
    representation of ``cf_y``. ``r2_d`` already has the required form.
    """
    result: dict[str, Any] = {
        "r2_y": np.nan,
        "r2_d": np.nan,
        "rho": np.nan,
        "cf_y": np.nan,
        "cf_d": np.nan,
        "cf_y_raw": np.nan,
        "cf_d_raw": np.nan,
        "sigma2_long": float(sigma2_long),
        "sigma2_short": float(sigma2_short),
        "nu2_long": float(nu2_long),
        "nu2_short": float(nu2_short),
        "rho_raw": np.nan,
        "rho_clipped": False,
        "cf_y_clipped": False,
        "cf_d_clipped": False,
        "rho_fallback": False,
        "boundary_calibration": False,
        "strengths_valid": False,
        "calibration_valid": False,
        "calibration_issue": None,
        "calibration_warning": None,
    }
    issues: list[str] = []
    calibration_warnings: list[str] = []
    if element_issue:
        issues.append(element_issue)

    scalar_values = np.asarray(
        [theta_long, theta_short, sigma2_long, sigma2_short, nu2_long, nu2_short],
        dtype=float,
    )
    if not np.isfinite(scalar_values).all():
        issues.append("long/short estimates and sensitivity elements must be finite")

    if np.isfinite(sigma2_long) and sigma2_long <= 0.0:
        issues.append("sigma2_long must be positive")
    if np.isfinite(nu2_long) and nu2_long <= 0.0:
        issues.append("nu2_long must be positive")
    if np.isfinite(nu2_short) and nu2_short <= 0.0:
        issues.append("nu2_short must be positive")

    gap_y = float(sigma2_short - sigma2_long)
    gap_d = float(nu2_long - nu2_short)

    if not issues:
        cf_y_raw = float(gap_y / sigma2_long)
        cf_d_raw = float(gap_d / nu2_short)
        result["cf_y_raw"] = cf_y_raw
        result["cf_d_raw"] = cf_d_raw
        if not np.isfinite(cf_y_raw):
            issues.append("cf_y_raw must be finite")
        if not np.isfinite(cf_d_raw):
            issues.append("cf_d_raw must be finite")

    if not issues:
        cf_y = float(np.clip(result["cf_y_raw"], 0.0, 1.0))
        cf_d = float(np.clip(result["cf_d_raw"], 0.0, 1.0))
        cf_y_clipped = bool(cf_y != result["cf_y_raw"])
        cf_d_clipped = bool(cf_d != result["cf_d_raw"])
        result.update(
            {
                "cf_y": cf_y,
                "cf_d": cf_d,
                "r2_y": float(cf_y / (1.0 + cf_y)),
                "r2_d": cf_d,
                "cf_y_clipped": cf_y_clipped,
                "cf_d_clipped": cf_d_clipped,
            }
        )
        if cf_y_clipped:
            calibration_warnings.append(
                f"cf_y_raw={result['cf_y_raw']:.6g} was clipped to {cf_y:.6g}"
            )
        if cf_d_clipped:
            calibration_warnings.append(
                f"cf_d_raw={result['cf_d_raw']:.6g} was clipped to {cf_d:.6g}"
            )

        if cf_d >= 1.0:
            issues.append(
                "DoubleML-clipped cf_d=1 is outside the finite Causalis sensitivity domain"
            )
        else:
            result["strengths_valid"] = True

        if gap_y > 0.0 and gap_d > 0.0:
            rho_denom = float(np.sqrt(gap_y * gap_d))
            rho_raw = float((theta_short - theta_long) / rho_denom)
            if not np.isfinite(rho_raw):
                issues.append("rho is not finite")
            else:
                rho = float(np.clip(rho_raw, -1.0, 1.0))
                result.update(
                    {
                        "rho_raw": rho_raw,
                        "rho": rho,
                        "rho_clipped": bool(
                            not np.isclose(rho, rho_raw, rtol=0.0, atol=0.0)
                        ),
                    }
                )
        else:
            rho = float(np.sign(theta_short - theta_long))
            result.update(
                {
                    "rho": rho,
                    "rho_fallback": True,
                }
            )
            calibration_warnings.append(
                "rho used the DoubleML boundary fallback because both gain "
                "components were not strictly positive"
            )

    result["boundary_calibration"] = bool(
        result["cf_y_clipped"]
        or result["cf_d_clipped"]
        or result["rho_fallback"]
    )
    result["calibration_valid"] = bool(
        not issues
        and result["strengths_valid"]
        and np.isfinite(result["rho"])
    )
    result["calibration_issue"] = "; ".join(dict.fromkeys(issues)) or None
    result["calibration_warning"] = (
        "; ".join(dict.fromkeys(calibration_warnings)) or None
    )
    return result


def _sensitivity_benchmark_refits(
    effect_estimation: Dict[str, Any] | Any,
    data: CausalData,
    benchmarking_set: List[str] | Literal["all"],
    fit_args: Optional[Dict[str, Any]] = None,
    *,
    grouped: bool = False,
) -> pd.DataFrame:
    r"""
    Benchmark confounders by refitting short IRMs that exclude either each
    requested confounder separately or all requested confounders as one group.

    This function performs a genuine short-model refit for every benchmark
    unit. Outcome strength, treatment/Riesz strength, and adversity are jointly
    calibrated from the long/short sensitivity elements and the actual change
    in the effect estimate.

    Parameters
    ----------
    effect_estimation : dict or Any
        Estimate/model container exposing a fitted IRM-like model.
    data : CausalData
        The causal dataset used for benchmarking. It must match the fitted long
        model on treatment, outcome, confounders, and row order.
    benchmarking_set : list[str] or "all"
        Confounders to benchmark one by one. Passing ``"all"`` benchmarks every
        confounder in ``data.confounders`` in that order.
    fit_args : dict, optional
        Additional keyword arguments passed to ``IRM.estimate(...)`` on each
        short model. If ``score`` is omitted, ATE/ATTE is inferred from the
        supplied estimate/model, and defaults to ATE. If ``store_diagnostics``
        or legacy ``diagnostic_data`` is omitted, the short benchmark refits use
        ``store_diagnostics=False`` by default.

    Returns
    -------
    pandas.DataFrame
        A long-form DataFrame with one row per benchmarked confounder. The
        original result columns are retained and element-based gain statistics,
        long/short sensitivity elements, and calibration audit fields are
        appended.

    Notes
    -----
    The gain statistics follow the long/short benchmarking construction and
    boundary handling used by DoubleML. Raw ``cf_y`` and ``cf_d`` are clipped
    to ``[0, 1]``. ``rho`` is estimated from the effect shift divided by the
    geometric mean of positive outcome- and Riesz-representer gains; if either
    gain is not positive, DoubleML's fallback uses the sign of the effect shift
    (magnitude one unless the shift is zero).

    Examples
    --------
    >>> from causalis.dgp import obs_linear_26_dataset
    >>> from causalis.scenarios.unconfoundedness.model import IRM
    >>> from causalis.scenarios.unconfoundedness.refutation import sensitivity_benchmark
    >>> # 1. Fit a model
    >>> data = obs_linear_26_dataset(n=1000, seed=42, return_causal_data=True)
    >>> irm = IRM(data=data).fit()
    >>> estimate = irm.estimate()
    >>> # 2. Benchmark specific confounders
    >>> benchmarks = sensitivity_benchmark(estimate, data, benchmarking_set=['x1', 'x2']) # doctest: +SKIP
    >>> print(benchmarks[['benchmark_confounder', 'r2_y', 'r2_d']]) # doctest: +SKIP
    """
    model = None
    if isinstance(effect_estimation, dict):
        model = effect_estimation.get("model")
    elif hasattr(effect_estimation, '_model'):
        model = getattr(effect_estimation, '_model')
    elif hasattr(effect_estimation, 'diagnostic_data'):
        diag = getattr(effect_estimation, 'diagnostic_data')
        model = getattr(diag, '_model', None)
    if model is None and hasattr(effect_estimation, 'coef_'):
        model = effect_estimation

    if model is None:
        raise TypeError(
            "effect_estimation must be a dict with 'model', a CausalEstimate, "
            "or a diagnostic_data object with a model reference."
        )

    if not isinstance(data, CausalData):
        raise TypeError(f"data must be a CausalData instance. Got {type(data)}.")

    required_attrs = ['data', 'coef_', 'se_', '_sensitivity_element_est']
    for attr in required_attrs:
        if not hasattr(model, attr):
            if attr == 'data' and hasattr(model, 'data_contracts'):
                continue
            raise NotImplementedError(
                "Sensitivity benchmarking requires a fitted IRM model with sensitivity elements. "
                f"Missing: {attr}"
            )

    model_data = getattr(model, "data", getattr(model, "data_contracts", None))
    if not isinstance(model_data, CausalData):
        raise TypeError("effect_estimation must resolve to a fitted IRM model with CausalData in model.data.")

    if fit_args is not None and not isinstance(fit_args, dict):
        raise TypeError(f"fit_args must be a dict. {fit_args} of type {type(fit_args)} was passed.")

    treatment_name = str(data.treatment_name)
    outcome_name = str(data.outcome_name)
    model_treatment_name = str(model_data.treatment_name)
    model_outcome_name = str(model_data.outcome_name)
    if treatment_name != model_treatment_name:
        raise ValueError(
            "data treatment column must match the fitted long model treatment column. "
            f"Got {treatment_name!r} and {model_treatment_name!r}."
        )
    if outcome_name != model_outcome_name:
        raise ValueError(
            "data outcome column must match the fitted long model outcome column. "
            f"Got {outcome_name!r} and {model_outcome_name!r}."
        )

    x_list_long = list(getattr(model_data, "confounders", []))
    data_confounders = list(data.confounders)
    if len(data_confounders) == 0:
        raise ValueError("data.confounders must not be empty.")
    if set(data_confounders) != set(x_list_long):
        raise ValueError(
            "data.confounders must match the fitted long model confounders as a set. "
            f"Got {data_confounders} and {x_list_long}."
        )

    if benchmarking_set == "all":
        benchmark_confounders = list(data_confounders)
    elif isinstance(benchmarking_set, list):
        if len(benchmarking_set) == 0:
            raise ValueError("benchmarking_set must not be empty.")
        deduped: list[str] = []
        seen: set[str] = set()
        for item in benchmarking_set:
            if not isinstance(item, str):
                raise TypeError(
                    "benchmarking_set must contain only strings. "
                    f"Found {type(item)}: {item!r}."
                )
            if item not in seen:
                deduped.append(item)
                seen.add(item)
        benchmark_confounders = deduped
    else:
        raise TypeError(
            "benchmarking_set must be a list of confounder names or the string 'all'. "
            f"Got {benchmarking_set!r} of type {type(benchmarking_set)}."
        )

    if not set(benchmark_confounders) <= set(data_confounders):
        raise ValueError(
            f"benchmarking_set must be a subset of data.confounders {data_confounders}. "
            f"{benchmark_confounders} was passed."
        )
    if not set(benchmark_confounders) <= set(x_list_long):
        raise ValueError(
            f"benchmarking_set must be a subset of long-model confounders {x_list_long}. "
            f"{benchmark_confounders} was passed."
        )

    if len(x_list_long) <= 1:
        raise ValueError("Benchmarking requires at least two confounders in the long model.")

    compare_cols = [outcome_name, treatment_name] + x_list_long
    input_cols = list(compare_cols)
    if data.user_id_name is not None:
        input_cols.append(str(data.user_id_name))
    df_long = model_data.get_df(columns=compare_cols)
    df_input = data.get_df(columns=input_cols)
    if df_input.shape[0] != df_long.shape[0]:
        raise ValueError(
            "data must have the same number of rows as the fitted long model. "
            f"Got {df_input.shape[0]} and {df_long.shape[0]}."
        )
    if not np.array_equal(
        df_input[compare_cols].to_numpy(dtype=object, copy=False),
        df_long.to_numpy(dtype=object, copy=False),
    ):
        raise ValueError("data must match the fitted long model sample and row order exactly.")

    from causalis.scenarios.unconfoundedness.model import IRM

    estimate_args_base: Dict[str, Any] = dict(fit_args or {})
    store_diagnostics_short = bool(
        estimate_args_base.pop("store_diagnostics", estimate_args_base.pop("diagnostic_data", False))
    )
    resolved_score = _resolve_sensitivity_score(
        effect_estimation=effect_estimation,
        model=model,
        explicit_score=estimate_args_base.get("score"),
    )
    estimate_args_base["score"] = resolved_score

    theta_long = float(model.coef_[0])
    sigma2_long, nu2_long, long_element_issue = _read_benchmark_sensitivity_elements(model)
    long_folds, split_issue = _resolve_benchmark_fold_assignments(
        model,
        full_sample_size=df_input.shape[0],
    )

    benchmark_groups = (
        [tuple(benchmark_confounders)]
        if grouped
        else [(confounder,) for confounder in benchmark_confounders]
    )

    rows: list[dict[str, Any]] = []
    for benchmark_group in benchmark_groups:
        group_set = set(benchmark_group)
        x_list_short = [x for x in data_confounders if x not in group_set]
        if len(x_list_short) == 0:
            if not grouped:
                raise ValueError(
                    f"Benchmarking confounder {benchmark_group[0]!r} would leave "
                    "no confounders for the short model."
                )
            raise ValueError(
                f"Benchmarking group {list(benchmark_group)!r} would leave no confounders "
                "for the short model."
            )

        data_short = CausalData(
            df=df_input,
            treatment=treatment_name,
            outcome=outcome_name,
            confounders=x_list_short,
            user_id=data.user_id_name,
        )
        irm_short = IRM(
            data=data_short,
            ml_g=model.ml_g,
            ml_m=model.ml_m,
            n_folds=getattr(model, 'n_folds', 4),
            n_rep=getattr(model, 'n_rep', 1),
            normalize_ipw=getattr(
                model,
                'normalize_ipw_effective_',
                getattr(model, 'normalize_ipw', False),
            ),
            overlap_policy=getattr(model, 'overlap_policy', 'clip'),
            overlap_threshold=getattr(model, 'overlap_threshold', 1e-2),
            weights=getattr(model, 'weights', None),
            random_state=getattr(model, 'random_state', None),
            n_jobs=getattr(model, 'n_jobs', 1),
        )
        if long_folds is not None:
            irm_short._fixed_fold_assignments_ = np.asarray(long_folds, dtype=int).copy()
        irm_short.fit(store_diagnostics=store_diagnostics_short)
        irm_short.estimate(**dict(estimate_args_base))

        theta_short = float(irm_short.coef_[0])
        sigma2_short, nu2_short, short_element_issue = (
            _read_benchmark_sensitivity_elements(irm_short)
        )
        element_issues = [
            issue
            for issue in (long_element_issue, short_element_issue, split_issue)
            if issue
        ]
        calibration = _calibrate_benchmark_gain_statistics(
            theta_long=theta_long,
            theta_short=theta_short,
            sigma2_long=sigma2_long,
            sigma2_short=sigma2_short,
            nu2_long=nu2_long,
            nu2_short=nu2_short,
            element_issue="; ".join(element_issues) or None,
        )
        benchmark_column = "benchmark_group" if grouped else "benchmark_confounder"
        benchmark_value = tuple(benchmark_group) if grouped else benchmark_group[0]

        rows.append(
            {
                benchmark_column: benchmark_value,
                **calibration,
                "theta_long": theta_long,
                "theta_short": theta_short,
                "delta": float(theta_long - theta_short),
            }
        )

    benchmark_column = "benchmark_group" if grouped else "benchmark_confounder"
    return pd.DataFrame(
        rows,
        columns=[benchmark_column, *_BENCHMARK_RESULT_COLUMNS],
    )


def sensitivity_benchmark(
    effect_estimation: Dict[str, Any] | Any,
    data: CausalData,
    benchmarking_set: List[str] | Literal["all"],
    fit_args: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    r"""
    Benchmark confounders one by one using genuine short-model IRM refits.

    Each requested confounder is excluded separately. The returned DataFrame
    therefore contains one row and one short-model estimate per confounder.
    Pass ``"all"`` to benchmark all confounders in ``data.confounders`` order.

    See Also
    --------
    sensitivity_benchmark_group
        Exclude several confounders together in one short-model refit.
    """
    return _sensitivity_benchmark_refits(
        effect_estimation,
        data,
        benchmarking_set,
        fit_args,
        grouped=False,
    )


def sensitivity_benchmark_group(
    effect_estimation: Dict[str, Any] | Any,
    data: CausalData,
    benchmarking_group: List[str],
    fit_args: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    r"""
    Benchmark a group of confounders using one genuine short-model IRM refit.

    All features in ``benchmarking_group`` are excluded together. Group-level
    confounding strengths and ``rho`` are calibrated jointly from long/short
    IRM sensitivity elements and the actual effect-estimate shift.

    Parameters
    ----------
    effect_estimation : dict or Any
        Estimate/model container exposing a fitted IRM-like model.
    data : CausalData
        The exact causal dataset and row order used to fit the long model.
    benchmarking_group : list[str]
        Non-empty group of confounders to exclude together. Duplicate names are
        ignored while preserving their first occurrence.
    fit_args : dict, optional
        Additional keyword arguments passed to ``IRM.estimate(...)`` for the
        short model. Short-model diagnostics are disabled by default.

    Returns
    -------
    pandas.DataFrame
        One-row DataFrame retaining ``benchmark_group``, ``r2_y``, ``r2_d``,
        ``rho``, ``theta_long``, ``theta_short``, and ``delta`` and appending
        element-based gain statistics and calibration audit fields. The group
        is stored as a tuple in ``benchmark_group``.

    Notes
    -----
    At least one long-model confounder must remain after removing the group.
    Runtime is normally dominated by the single short-model refit.
    """
    if not isinstance(benchmarking_group, list):
        raise TypeError(
            "benchmarking_group must be a list of confounder names. "
            f"Got {benchmarking_group!r} of type {type(benchmarking_group)}."
        )

    return _sensitivity_benchmark_refits(
        effect_estimation,
        data,
        benchmarking_group,
        fit_args,
        grouped=True,
    )


# ---------------- Main entry for producing textual sensitivity summary ----------------

def sensitivity_analysis(
    effect_estimation: Dict[str, Any] | Any,
    *,
    r2_y: float,
    r2_d: float,
    rho: float = 1.0,
    H0: float = 0.0,
    alpha: float = 0.05,
    use_signed_rr: bool = False,
) -> Dict[str, Any]:
    r"""Compute bias-aware components and cache them.

    This function turns a fitted estimate into a simple hidden-confounding
    stress test. In the default mode, the bound width is

    .. math::

        |\rho| \cdot \sqrt{\sigma^2 \nu^2}
        \cdot
        \sqrt{\frac{r2_y}{1-r2_y}\frac{r2_d}{1-r2_d}},

    so the reported confounding interval is

    .. math::

        [\theta - \text{bound\_width}, \theta + \text{bound\_width}].

    Here :math:`r2_y` controls how much residual outcome variation an omitted
    confounder could explain, :math:`r2_d` does the same for treatment
    assignment, and :math:`\rho` sets the sign and strength alignment between
    the two channels.

    Parameters
    ----------
    effect_estimation : Dict[str, Any] or Any
        The effect estimation object.
    r2_y : float
        Sensitivity parameter for the outcome (R^2 form, R_Y^2; converted to odds form internally).
    r2_d : float
        Sensitivity parameter for the treatment (R^2 form, R_D^2).
    rho : float, default 1.0
        Correlation parameter.
    H0 : float, default 0.0
        Null hypothesis for robustness values.
    alpha : float, default 0.05
        Significance level.
    use_signed_rr : bool, default False
        Whether to use signed rr in the quadratic combination of sensitivity components.
        If True and m_alpha/rr are available, the bias bound is computed via the
        per-unit quadratic form and RV/RVa are not reported.

    Returns
    -------
    dict
        Dictionary with bias-aware results:
          - theta, se, alpha, z
          - sampling_ci
          - theta_bounds_cofounding = (theta - bound_width, theta + bound_width)
          - bias_aware_ci = faithful CI for the bounds
          - max_bias and components (sigma2, nu2)
          - params (r2_y, r2_d, rho, use_signed_rr)

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    >>> from causalis.dgp import obs_linear_26_dataset
    >>> from causalis.scenarios.unconfoundedness.model import IRM
    >>> data = obs_linear_26_dataset(
    ...     n=1000,
    ...     seed=3141,
    ...     include_oracle=False,
    ...     return_causal_data=True,
    ... )
    >>> irm = IRM(
    ...     data=data,
    ...     ml_g=RandomForestRegressor(
    ...         n_estimators=200,
    ...         max_depth=6,
    ...         min_samples_leaf=5,
    ...         random_state=3141,
    ...     ),
    ...     ml_m=RandomForestClassifier(
    ...         n_estimators=200,
    ...         max_depth=6,
    ...         min_samples_leaf=5,
    ...         random_state=3141,
    ...     ),
    ...     n_folds=3,
    ...     random_state=3141,
    ... )
    >>> estimate = irm.fit().estimate(score="ATE")
    >>> out = sensitivity_analysis(estimate, r2_y=0.02, r2_d=0.02, rho=1.0)
    >>> out["theta_bounds_cofounding"]  # doctest: +SKIP
    >>> out["bias_aware_ci"]  # doctest: +SKIP
    """
    raw_res = compute_bias_aware_ci(
        effect_estimation,
        r2_y=r2_y,
        r2_d=r2_d,
        rho=rho,
        H0=H0,
        alpha=alpha,
        use_signed_rr=use_signed_rr
    )

    label = _resolve_sensitivity_label(
        effect_estimation,
        model=effect_estimation.get("model") if isinstance(effect_estimation, dict) else effect_estimation,
    )
    res = _wrap_sensitivity_result(raw_res, label=label)

    diag = None
    if isinstance(effect_estimation, dict):
        effect_estimation["bias_aware"] = res
        diag = effect_estimation.get("diagnostic_data")
    else:
        diag = getattr(effect_estimation, "diagnostic_data", None)

    if diag is not None:
        if isinstance(diag, dict):
            diag["sensitivity_analysis"] = res
        else:
            try:
                diag.sensitivity_analysis = res
            except Exception:
                pass

    return res


def interpret_sensitivity_analysis(
    effect_estimation: Dict[str, Any] | Any,
    *,
    r2_y: float,
    r2_d: float,
    rho: float = 1.0,
    H0: float = 0.0,
    alpha: float = 0.05,
    use_signed_rr: bool = False,
) -> Dict[str, Any]:
    """Run sensitivity analysis and return a structured interpretation.

    Parameters
    ----------
    effect_estimation : Dict[str, Any] or Any
        The effect estimation object.
    r2_y : float
        Sensitivity parameter for outcome residual confounding strength.
    r2_d : float
        Sensitivity parameter for treatment residual confounding strength.
    rho : float, default 1.0
        Correlation parameter for unobserved confounding.
    H0 : float, default 0.0
        Null hypothesis used for significance checks.
    alpha : float, default 0.05
        Significance level.
    use_signed_rr : bool, default False
        Whether to use signed rr in the quadratic sensitivity combination.

    Returns
    -------
    Dict[str, Any]
        Dictionary with:
          - raw: the output of ``sensitivity_analysis(...)``
          - interpretation: machine-readable interpretation fields
          - summary: compact human-readable interpretation

    Notes
    -----
    This function wraps `sensitivity_analysis` and provides a textual
    interpretation of the results, including whether the estimate remains
    significant under the assumed confounding.

    Examples
    --------
    >>> from causalis.scenarios.unconfoundedness.refutation.unconfoundedness import interpret_sensitivity_analysis
    >>> # Assuming 'estimate' is a fitted CausalEstimate
    >>> interpretation = interpret_sensitivity_analysis(estimate, r2_y=0.01, r2_d=0.01) # doctest: +SKIP
    >>> print(interpretation["summary"]) # doctest: +SKIP
    """
    res = sensitivity_analysis(
        effect_estimation,
        r2_y=r2_y,
        r2_d=r2_d,
        rho=rho,
        H0=H0,
        alpha=alpha,
        use_signed_rr=use_signed_rr,
    )

    def _ci_excludes_h0(ci: tuple[float, float], h0: float) -> bool:
        """Return whether a confidence interval excludes the null value."""
        lo, hi = float(ci[0]), float(ci[1])
        if not (np.isfinite(lo) and np.isfinite(hi)):
            return False
        return (lo > h0) or (hi < h0)

    theta = float(res.get("theta", np.nan))
    se = float(res.get("se", np.nan))
    sampling_ci = tuple(map(float, res.get("sampling_ci", (np.nan, np.nan))))
    bias_aware_ci = tuple(map(float, res.get("bias_aware_ci", (np.nan, np.nan))))
    confounding_bounds = tuple(map(float, res.get("theta_bounds_cofounding", (np.nan, np.nan))))
    rv = float(res.get("rv", np.nan))
    rva = float(res.get("rva", np.nan))

    if theta > H0:
        direction = "positive"
    elif theta < H0:
        direction = "negative"
    else:
        direction = "null"

    significant_no_confounding = _ci_excludes_h0(sampling_ci, H0)
    significant_with_assumed_confounding = _ci_excludes_h0(bias_aware_ci, H0)

    if not np.isfinite(rv):
        robustness_level = "not_available"
    elif rv < 0.02:
        robustness_level = "low"
    elif rv < 0.05:
        robustness_level = "moderate"
    else:
        robustness_level = "high"

    rv_str = "nan" if not np.isfinite(rv) else f"{rv:.4f}"
    rva_str = "nan" if not np.isfinite(rva) else f"{rva:.4f}"
    summary = (
        f"Effect estimate theta={theta:.4f} ({direction}), se={se:.4f}. "
        f"Sampling CI [{sampling_ci[0]:.4f}, {sampling_ci[1]:.4f}] "
        f"{'excludes' if significant_no_confounding else 'includes'} H0={H0:.4f}. "
        f"Under specified confounding, theta bounds are "
        f"[{confounding_bounds[0]:.4f}, {confounding_bounds[1]:.4f}] and bias-aware CI is "
        f"[{bias_aware_ci[0]:.4f}, {bias_aware_ci[1]:.4f}], which "
        f"{'excludes' if significant_with_assumed_confounding else 'includes'} H0. "
        f"RV={rv_str}, RVa={rva_str}, robustness={robustness_level}."
    )

    return {
        "raw": res,
        "interpretation": {
            "direction": direction,
            "significant_no_confounding": significant_no_confounding,
            "significant_with_assumed_confounding": significant_with_assumed_confounding,
            "robustness_level": robustness_level,
            "rv": rv,
            "rva": rva,
        },
        "summary": summary,
    }


# ---------------- Decision protocol for benchmarked sensitivity ----------------

_PROTOCOL_SCENARIO_COLUMNS = [
    "benchmark",
    "benchmark_group",
    "scenario",
    "multiplier",
    "r2_y",
    "r2_d",
    "rho",
    "theta_long",
    "theta_short",
    "delta",
    "relative_delta",
    "theta_lower",
    "theta_upper",
    "ci_lower",
    "ci_upper",
    "rv",
    "rva",
    "scenario_valid",
    "scenario_issue",
    "passed",
]


def _scale_partial_r2(r2: float, multiplier: float) -> float:
    """Scale a partial R-squared on the Cohen f-squared (odds) scale."""
    r2 = float(r2)
    if not np.isfinite(r2) or not (0.0 <= r2 < 1.0):
        raise ValueError(f"Benchmark partial R-squared must be in [0, 1). Got {r2!r}.")
    f2 = r2 / (1.0 - r2)
    scaled_f2 = float(multiplier) * f2
    return float(scaled_f2 / (1.0 + scaled_f2))


def _protocol_ci_passes(
    ci: tuple[float, float],
    *,
    direction: Literal["positive", "negative"],
    decision_threshold: float,
) -> bool:
    """Apply the pre-specified directional decision rule to an interval."""
    lo, hi = map(float, ci)
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return False
    if direction == "positive":
        return bool(lo > decision_threshold)
    return bool(hi < decision_threshold)


def _protocol_scenario(
    effect_estimation: Dict[str, Any] | Any,
    *,
    benchmark_name: str,
    benchmark_group: tuple[str, ...],
    benchmark_row: pd.Series,
    scenario: str,
    multiplier: float,
    r2_y: float,
    r2_d: float,
    rho: float,
    direction: Literal["positive", "negative"],
    decision_threshold: float,
    alpha: float,
) -> tuple[dict[str, Any], Dict[str, Any]]:
    """Evaluate one sensitivity scenario and return its table row and raw result."""
    interpreted = interpret_sensitivity_analysis(
        effect_estimation,
        r2_y=float(r2_y),
        r2_d=float(r2_d),
        rho=float(rho),
        H0=float(decision_threshold),
        alpha=float(alpha),
    )
    raw = interpreted["raw"]
    theta_bounds = tuple(map(float, raw["theta_bounds_cofounding"]))
    bias_aware_ci = tuple(map(float, raw["bias_aware_ci"]))
    theta_long = float(benchmark_row["theta_long"])
    delta = float(benchmark_row["delta"])
    relative_delta = np.nan if abs(theta_long) <= 1e-16 else abs(delta) / abs(theta_long)

    row = {
        "benchmark": benchmark_name,
        "benchmark_group": benchmark_group,
        "scenario": scenario,
        "multiplier": float(multiplier),
        "r2_y": float(r2_y),
        "r2_d": float(r2_d),
        "rho": float(rho),
        "theta_long": theta_long,
        "theta_short": float(benchmark_row["theta_short"]),
        "delta": delta,
        "relative_delta": float(relative_delta),
        "theta_lower": theta_bounds[0],
        "theta_upper": theta_bounds[1],
        "ci_lower": bias_aware_ci[0],
        "ci_upper": bias_aware_ci[1],
        "rv": float(raw.get("rv", np.nan)),
        "rva": float(raw.get("rva", np.nan)),
        "scenario_valid": True,
        "scenario_issue": None,
        "passed": _protocol_ci_passes(
            bias_aware_ci,
            direction=direction,
            decision_threshold=decision_threshold,
        ),
    }
    return row, dict(raw)


def _invalid_protocol_scenario(
    *,
    benchmark_name: str,
    benchmark_group: tuple[str, ...],
    benchmark_row: pd.Series,
    scenario: str,
    multiplier: float,
    r2_y: float,
    r2_d: float,
    rho: float,
    issue: str,
) -> tuple[dict[str, Any], Dict[str, Any]]:
    """Create a fail-closed scenario row when calibration cannot be evaluated."""
    theta_long = float(benchmark_row["theta_long"])
    delta = float(benchmark_row["delta"])
    relative_delta = np.nan if abs(theta_long) <= 1e-16 else abs(delta) / abs(theta_long)
    row = {
        "benchmark": benchmark_name,
        "benchmark_group": benchmark_group,
        "scenario": scenario,
        "multiplier": float(multiplier),
        "r2_y": float(r2_y),
        "r2_d": float(r2_d),
        "rho": float(rho),
        "theta_long": theta_long,
        "theta_short": float(benchmark_row["theta_short"]),
        "delta": delta,
        "relative_delta": float(relative_delta),
        "theta_lower": np.nan,
        "theta_upper": np.nan,
        "ci_lower": np.nan,
        "ci_upper": np.nan,
        "rv": np.nan,
        "rva": np.nan,
        "scenario_valid": False,
        "scenario_issue": issue,
        "passed": False,
    }
    return row, {"scenario_valid": False, "scenario_issue": issue}


def run_sensitivity_protocol(
    effect_estimation: Dict[str, Any] | Any,
    data: CausalData,
    *,
    benchmark_groups: Mapping[str, List[str]],
    decision_threshold: float,
    direction: Literal["positive", "negative"],
    alpha: float = 0.05,
    stress_multiplier: float = 2.0,
    preconditions_passed: bool = True,
    fit_args: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    r"""Run a binary, benchmark-calibrated sensitivity decision protocol.

    The primary decision uses one-times benchmark strength and the benchmark's
    estimated ``rho``. Every primary benchmark must pass. A benchmark passes
    when its bias-aware confidence interval lies strictly beyond the
    pre-specified practical threshold in ``direction``. The two-times and
    ``rho=1`` adversarial scenarios are reported as stress tests and do not
    change the primary PASS/FAIL decision.

    Parameters
    ----------
    effect_estimation : dict or Any
        Fitted IRM-like estimate accepted by :func:`sensitivity_analysis`.
    data : CausalData
        Exact data and row order used to fit the long model.
    benchmark_groups : mapping[str, list[str]]
        Non-empty mapping from meaningful benchmark labels to groups of
        observed pre-treatment confounders. Each group is omitted in its own
        genuine short-model refit.
    decision_threshold : float
        Pre-specified practically meaningful effect boundary. For a positive
        claim, the lower bias-aware CI endpoint must exceed this value. For a
        negative claim, the upper endpoint must be below it.
    direction : {"positive", "negative"}
        Direction of the practically meaningful causal claim.
    alpha : float, default 0.05
        Two-sided significance level used for bias-aware confidence intervals.
    stress_multiplier : float, default 2.0
        Strength multiplier for the secondary stress scenario. Each partial
        R-squared is multiplied on the odds/Cohen-f-squared scale.
    preconditions_passed : bool, default True
        External design gate covering the causal adjustment set, overlap,
        nuisance-model quality, and estimator stability. ``False`` forces the
        overall result to FAIL while still returning sensitivity diagnostics.
    fit_args : dict, optional
        Additional keyword arguments for each short-model estimate.

    Returns
    -------
    dict
        A report with ``status`` (``"PASS"`` or ``"FAIL"``), ``passed``,
        benchmark and scenario DataFrames, raw scenario details, warnings, and
        a standard human-readable conclusion. ``primary`` determines status;
        ``stress`` and ``adversarial`` describe the robustness margin.

    Notes
    -----
    The robustness values ``rv`` and ``rva`` are reported but are not used as
    universal pass/fail cutoffs. Benchmark selection and ``decision_threshold``
    must be justified before inspecting the effect estimate. ``rho`` for both
    singleton and multi-feature benchmarks is calibrated from long/short
    sensitivity elements. A value of ``+/-1`` in primary or stress can mean
    either that ``rho_raw`` saturated at the correlation boundary or that
    DoubleML's non-positive-gain fallback was used. Only the adversarial
    scenario forces ``rho=1`` independently of the benchmark effect shift.

    DoubleML-compatible boundary calibration clips non-positive raw gains to
    zero and keeps the scenario numerically evaluable. Raw values and boundary
    flags remain available in ``benchmarks`` for audit. Missing/non-finite
    sensitivity elements and an upper-bound ``cf_d=1`` still fail closed,
    because Causalis sensitivity bounds require finite inputs below one.

    The current IRM estimator supports ``n_rep=1`` only. Repeated cross-fitting
    stability must therefore be assessed outside this function until IRM adds
    repeated-split support.

    Examples
    --------
    >>> report = run_sensitivity_protocol(  # doctest: +SKIP
    ...     estimate,
    ...     data,
    ...     benchmark_groups={"engagement": ["sessions", "activity_days"]},
    ...     decision_threshold=0.0,
    ...     direction="positive",
    ... )
    >>> report["status"]  # doctest: +SKIP
    'PASS'
    >>> report["primary"][["benchmark", "ci_lower", "ci_upper", "passed"]]  # doctest: +SKIP
    """
    if direction not in {"positive", "negative"}:
        raise ValueError("direction must be either 'positive' or 'negative'.")
    if not np.isfinite(decision_threshold):
        raise ValueError("decision_threshold must be finite and pre-specified.")
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError("alpha must be in (0, 1).")
    if not np.isfinite(stress_multiplier) or float(stress_multiplier) <= 1.0:
        raise ValueError("stress_multiplier must be finite and greater than 1.")
    if not isinstance(preconditions_passed, (bool, np.bool_)):
        raise TypeError("preconditions_passed must be a boolean.")
    if not isinstance(benchmark_groups, Mapping):
        raise TypeError("benchmark_groups must be a mapping from labels to feature lists.")

    warnings: list[str] = []
    if not benchmark_groups:
        empty = pd.DataFrame(columns=_PROTOCOL_SCENARIO_COLUMNS)
        return {
            "status": "FAIL",
            "passed": False,
            "direction": direction,
            "decision_threshold": float(decision_threshold),
            "alpha": float(alpha),
            "stress_multiplier": float(stress_multiplier),
            "preconditions_passed": bool(preconditions_passed),
            "benchmarks": pd.DataFrame(),
            "primary": empty.copy(),
            "stress": empty.copy(),
            "adversarial": empty.copy(),
            "scenarios": empty.copy(),
            "details": {},
            "limited_margin": False,
            "warnings": ["At least one justified primary benchmark group is required."],
            "summary": (
                "FAIL: no primary benchmark group was supplied; hidden-confounding "
                "robustness is not established."
            ),
        }

    normalized_groups: list[tuple[str, List[str]]] = []
    for raw_name, group in benchmark_groups.items():
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise ValueError("Every benchmark label must be a non-empty string.")
        if not isinstance(group, list):
            raise TypeError(f"Benchmark {raw_name!r} must be a list of confounder names.")
        normalized_groups.append((raw_name.strip(), group))

    benchmark_frames: list[pd.DataFrame] = []
    primary_rows: list[dict[str, Any]] = []
    stress_rows: list[dict[str, Any]] = []
    adversarial_rows: list[dict[str, Any]] = []
    details: dict[str, dict[str, Dict[str, Any]]] = {}

    for benchmark_name, group in normalized_groups:
        benchmark = sensitivity_benchmark_group(
            effect_estimation,
            data,
            group,
            fit_args=fit_args,
        )
        benchmark = benchmark.copy()
        benchmark.insert(0, "benchmark", benchmark_name)
        benchmark_frames.append(benchmark)

        benchmark_row = benchmark.iloc[0]
        benchmark_group = tuple(benchmark_row["benchmark_group"])
        if bool(benchmark_row.get("rho_clipped", False)):
            warnings.append(
                f"Benchmark {benchmark_name!r} produced |rho_raw| > 1; "
                "the calibrated primary/stress rho was clipped to +/-1."
            )
        if bool(benchmark_row.get("boundary_calibration", False)):
            boundary_warning = str(
                benchmark_row.get("calibration_warning")
                or "DoubleML boundary clipping/fallback was used"
            )
            warnings.append(
                f"Benchmark {benchmark_name!r} used DoubleML boundary "
                f"calibration: {boundary_warning}."
            )

        r2_y = float(benchmark_row["r2_y"])
        r2_d = float(benchmark_row["r2_d"])
        rho = float(benchmark_row["rho"])
        strengths_valid = bool(benchmark_row.get("strengths_valid", True))
        calibration_valid = bool(benchmark_row.get("calibration_valid", True))
        calibration_issue = str(
            benchmark_row.get("calibration_issue")
            or "benchmark gain statistics could not be calibrated"
        )
        if not calibration_valid:
            warnings.append(f"Benchmark {benchmark_name!r} is not calibratable: {calibration_issue}.")

        if strengths_valid:
            stress_r2_y = _scale_partial_r2(r2_y, float(stress_multiplier))
            stress_r2_d = _scale_partial_r2(r2_d, float(stress_multiplier))
        else:
            stress_r2_y = np.nan
            stress_r2_d = np.nan

        # Evaluate secondary scenarios first so the final cached analysis on the
        # estimate corresponds to the primary scenario used for the decision.
        if calibration_valid:
            stress_row, stress_raw = _protocol_scenario(
                effect_estimation,
                benchmark_name=benchmark_name,
                benchmark_group=benchmark_group,
                benchmark_row=benchmark_row,
                scenario=f"{float(stress_multiplier):g}x",
                multiplier=float(stress_multiplier),
                r2_y=stress_r2_y,
                r2_d=stress_r2_d,
                rho=rho,
                direction=direction,
                decision_threshold=float(decision_threshold),
                alpha=float(alpha),
            )
        else:
            stress_row, stress_raw = _invalid_protocol_scenario(
                benchmark_name=benchmark_name,
                benchmark_group=benchmark_group,
                benchmark_row=benchmark_row,
                scenario=f"{float(stress_multiplier):g}x",
                multiplier=float(stress_multiplier),
                r2_y=stress_r2_y,
                r2_d=stress_r2_d,
                rho=rho,
                issue=calibration_issue,
            )

        if strengths_valid:
            adversarial_row, adversarial_raw = _protocol_scenario(
                effect_estimation,
                benchmark_name=benchmark_name,
                benchmark_group=benchmark_group,
                benchmark_row=benchmark_row,
                scenario="adversarial",
                multiplier=1.0,
                r2_y=r2_y,
                r2_d=r2_d,
                rho=1.0,
                direction=direction,
                decision_threshold=float(decision_threshold),
                alpha=float(alpha),
            )
        else:
            adversarial_row, adversarial_raw = _invalid_protocol_scenario(
                benchmark_name=benchmark_name,
                benchmark_group=benchmark_group,
                benchmark_row=benchmark_row,
                scenario="adversarial",
                multiplier=1.0,
                r2_y=r2_y,
                r2_d=r2_d,
                rho=1.0,
                issue=calibration_issue,
            )

        if calibration_valid:
            primary_row, primary_raw = _protocol_scenario(
                effect_estimation,
                benchmark_name=benchmark_name,
                benchmark_group=benchmark_group,
                benchmark_row=benchmark_row,
                scenario="primary",
                multiplier=1.0,
                r2_y=r2_y,
                r2_d=r2_d,
                rho=rho,
                direction=direction,
                decision_threshold=float(decision_threshold),
                alpha=float(alpha),
            )
        else:
            primary_row, primary_raw = _invalid_protocol_scenario(
                benchmark_name=benchmark_name,
                benchmark_group=benchmark_group,
                benchmark_row=benchmark_row,
                scenario="primary",
                multiplier=1.0,
                r2_y=r2_y,
                r2_d=r2_d,
                rho=rho,
                issue=calibration_issue,
            )

        stress_rows.append(stress_row)
        adversarial_rows.append(adversarial_row)
        primary_rows.append(primary_row)
        details[benchmark_name] = {
            "primary": primary_raw,
            "stress": stress_raw,
            "adversarial": adversarial_raw,
        }

    benchmarks_df = pd.concat(benchmark_frames, ignore_index=True)
    primary_df = pd.DataFrame(primary_rows, columns=_PROTOCOL_SCENARIO_COLUMNS)
    stress_df = pd.DataFrame(stress_rows, columns=_PROTOCOL_SCENARIO_COLUMNS)
    adversarial_df = pd.DataFrame(adversarial_rows, columns=_PROTOCOL_SCENARIO_COLUMNS)
    scenarios_df = pd.concat([primary_df, stress_df, adversarial_df], ignore_index=True)

    sensitivity_passed = bool(primary_df["passed"].all())
    overall_passed = bool(preconditions_passed) and sensitivity_passed
    stress_passed = bool(stress_df["passed"].all())
    adversarial_passed = bool(adversarial_df["passed"].all())
    limited_margin = overall_passed and not (stress_passed and adversarial_passed)
    status = "PASS" if overall_passed else "FAIL"
    invalid_primary = primary_df.loc[~primary_df["scenario_valid"], "benchmark"].tolist()
    if "boundary_calibration" in benchmarks_df.columns:
        boundary_benchmarks = benchmarks_df.loc[
            benchmarks_df["boundary_calibration"].astype(bool),
            "benchmark",
        ].tolist()
    else:
        boundary_benchmarks = []

    if not preconditions_passed:
        summary = (
            "FAIL: the external design/overlap/nuisance preconditions did not pass; "
            "the causal claim is not established."
        )
    elif invalid_primary:
        summary = (
            "FAIL: primary benchmark gain statistics could not be calibrated for "
            f"{invalid_primary}; the sensitivity gate fails closed."
        )
    elif not sensitivity_passed:
        failed = primary_df.loc[~primary_df["passed"], "benchmark"].tolist()
        summary = (
            "FAIL: under hidden confounding comparable to primary benchmark(s) "
            f"{failed}, the bias-aware confidence interval crosses the pre-specified "
            f"decision threshold {float(decision_threshold):g}; the causal claim is not robust."
        )
    else:
        margin = "limited" if limited_margin else "strong"
        boundary_note = (
            f" DoubleML boundary calibration was used for {boundary_benchmarks}."
            if boundary_benchmarks
            else ""
        )
        summary = (
            "PASS: the practically meaningful effect is robust to hidden confounding "
            "comparable to every primary benchmark. "
            f"The reported stress-test margin is {margin}."
            f"{boundary_note}"
        )

    model = effect_estimation.get("model") if isinstance(effect_estimation, dict) else effect_estimation
    if getattr(model, "n_rep", None) == 1:
        warnings.append(
            "IRM currently uses n_rep=1; repeated cross-fitting stability must be assessed separately."
        )

    return {
        "status": status,
        "passed": overall_passed,
        "direction": direction,
        "decision_threshold": float(decision_threshold),
        "alpha": float(alpha),
        "stress_multiplier": float(stress_multiplier),
        "preconditions_passed": bool(preconditions_passed),
        "benchmarks": benchmarks_df,
        "primary": primary_df,
        "stress": stress_df,
        "adversarial": adversarial_df,
        "scenarios": scenarios_df,
        "details": details,
        "limited_margin": limited_margin,
        "boundary_benchmarks": boundary_benchmarks,
        "warnings": warnings,
        "summary": summary,
    }
