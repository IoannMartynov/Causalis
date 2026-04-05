"""
Sensitivity functions refactored into a dedicated module.

This module centralizes bias-aware sensitivity helpers and the public
entry points used by refutation utilities for unconfoundedness.
"""
from __future__ import annotations

from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd

from causalis.data_contracts.causal_diagnostic_data import UnconfoundednessDiagnosticData

__all__ = [
    "sensitivity_analysis",
    "sensitivity_benchmark",
    "get_sensitivity_summary",
    "interpret_sensitivity_analysis",
]

# ---------------- Internals ----------------

_ESSENTIALLY_ZERO = 1e-32


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
    lbl = (label or 'theta').rjust(6)
    ci_l, ci_u = res['sampling_ci']
    th_l, th_u = res['theta_bounds_cofounding']
    bci_l, bci_u = res['bias_aware_ci']
    theta = res['theta']; se = res['se']
    alpha = res['alpha']; z = res['z']
    cf = res['params']
    bound_width = float(res.get('bound_width', res.get('max_bias', 0.0)))
    max_bias_base = float(res.get('max_bias_base', np.nan))

    lines = []
    lines.append("================== Bias-aware Interval ==================")
    lines.append("")
    lines.append("------------------ Scenario          ------------------")
    lines.append(f"Significance Level: alpha={alpha}")
    lines.append(f"Null Hypothesis: H0={res.get('H0', 0.0)}")
    lines.append(f"Sensitivity parameters: r2_y={cf['r2_y']}; r2_d={cf['r2_d']}, rho={cf['rho']}, use_signed_rr={cf['use_signed_rr']}")
    lines.append("")
    lines.append("------------------ Components        ------------------")
    lines.append(f"{'':>6} {'theta':>11} {'se':>11} {'z':>8} {'max_bias':>12} {'sigma2':>12} {'nu2':>12}")
    lines.append(f"{lbl} {theta:11.6f} {se:11.6f} {z:8.4f} {res['max_bias']:12.6f} {res['sigma2']:12.6f} {res['nu2']:12.6f}")
    lines.append(f"Bound width (theta +/-): {bound_width:.6f}")
    if np.isfinite(max_bias_base):
        lines.append(f"Base sqrt(sigma2*nu2): {max_bias_base:.6f}")
    lines.append("")
    lines.append("------------------ Intervals         ------------------")
    lines.append(f"{'':>6} {'Sampling CI lower':>18} {'Conf. θ lower':>16} {'Bias-aware lower':>18} {'Bias-aware upper':>18} {'Conf. θ upper':>16} {'Sampling CI upper':>20}")
    lines.append(f"{lbl} {ci_l:18.6f} {th_l:16.6f} {bci_l:18.6f} {bci_u:18.6f} {th_u:16.6f} {ci_u:20.6f}")
    
    if 'rv' in res and 'rva' in res:
        lines.append("")
        lines.append("------------------ Robustness Values ------------------")
        lines.append(f"{'':>6} {'RV (%)':>15} {'RVa (%)':>15}")
        lines.append(f"{lbl} {res['rv']*100:15.6f} {res['rva']*100:15.6f}")
    
    return "\n".join(lines)


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
    """
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
        if hasattr(effect_estimation, 'treatment') and isinstance(effect_estimation.treatment, str):
            label = effect_estimation.treatment
        else:
            # Check 'data' or 'data_contracts' for the treatment name
            data_obj = getattr(model, 'data', getattr(model, 'data_contracts', None))
            t = getattr(data_obj, 'treatment', None)
            label = getattr(t, 'name', None) or 'theta'

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

def sensitivity_benchmark(
    effect_estimation: Dict[str, Any],
    benchmarking_set: List[str],
    fit_args: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Computes a benchmark for a given set of features by refitting a short IRM model
    (excluding the provided features) and contrasting it with the original (long) model.
    Returns a DataFrame containing r2_y, r2_d, rho and the change in estimates.

    Parameters
    ----------
    effect_estimation : dict
        A dictionary containing the fitted IRM model under the key 'model'.
    benchmarking_set : list[str]
        List of confounder names to be used for benchmarking (to be removed in the short model).
    fit_args : dict, optional
        Legacy name for additional keyword arguments passed to `IRM.estimate(...)`
        on the short model. If `score` is omitted, ATE/ATTE is inferred from
        the supplied estimate/model, and defaults to ATE.
        If `store_diagnostics` or legacy `diagnostic_data` is omitted, the short
        benchmark model defaults to `store_diagnostics=False` for faster benchmarking.

    Returns
    -------
    pandas.DataFrame
        A one-row DataFrame indexed by the treatment name with columns:
        - r2_y, r2_d, rho: residual-based benchmarking strengths
        - theta_long, theta_short, delta: effect estimates and their change (long - short)
    """
    # Extract model from various possible inputs (dict, CausalEstimate, or DiagnosticData)
    model = None
    if isinstance(effect_estimation, dict):
        model = effect_estimation.get('model')
    elif hasattr(effect_estimation, '_model'):
        # DiagnosticData path (via private attribute)
        model = getattr(effect_estimation, '_model')
    elif hasattr(effect_estimation, 'diagnostic_data'):
        # CausalEstimate path
        diag = getattr(effect_estimation, 'diagnostic_data')
        model = getattr(diag, '_model', None)

    # Fallback: check if effect_estimation itself is the model
    if model is None and hasattr(effect_estimation, 'coef_'):
        model = effect_estimation

    if model is None:
        raise TypeError("effect_estimation must be a dict with 'model', a CausalEstimate, "
                        "or a diagnostic_data object with a model reference.")

    # Validate model type by attribute presence (duck-typing IRM)
    required_attrs = ['data', 'coef_', 'se_', '_sensitivity_element_est']
    for attr in required_attrs:
        if not hasattr(model, attr):
            # Fallback for data_contracts name
            if attr == 'data' and hasattr(model, 'data_contracts'):
                continue
            raise NotImplementedError(f"Sensitivity benchmarking requires a fitted IRM model with sensitivity elements. Missing: {attr}")

    # Extract current confounders
    try:
        x_list_long = list(getattr(model.data, 'confounders', []))
    except Exception as e:
        raise RuntimeError(f"Failed to access model data_contracts confounders: {e}")

    # input checks
    if not isinstance(benchmarking_set, list):
        raise TypeError(
            f"benchmarking_set must be a list. {str(benchmarking_set)} of type {type(benchmarking_set)} was passed."
        )
    if len(benchmarking_set) == 0:
        raise ValueError("benchmarking_set must not be empty.")
    if not set(benchmarking_set) <= set(x_list_long):
        raise ValueError(
            f"benchmarking_set must be a subset of features {str(x_list_long)}. "
            f"{str(benchmarking_set)} was passed."
        )
    if fit_args is not None and not isinstance(fit_args, dict):
        raise TypeError(f"fit_args must be a dict. {str(fit_args)} of type {type(fit_args)} was passed.")

    # Build short data_contracts excluding benchmarking features
    x_list_short = [x for x in x_list_long if x not in benchmarking_set]
    if len(x_list_short) == 0:
        raise ValueError("After removing benchmarking_set there are no confounders left to fit the short model.")

    # Create a shallow copy of the underlying DataFrame and build a new CausalData
    df_long = model.data.get_df()
    treatment_name = model.data.treatment.name
    outcome_name = model.data.outcome.name

    # Prefer in-scope names; fallback to import to avoid fragile self-import patterns
    try:
        CausalData  # type: ignore[name-defined]
        IRM  # type: ignore[name-defined]
    except NameError:
        from causalis.dgp.causaldata import CausalData
        from causalis.scenarios.unconfoundedness.model import IRM

    data_short = CausalData(df=df_long, treatment=treatment_name, outcome=outcome_name, confounders=x_list_short)

    # Clone/construct a short IRM with same hyperparameters/learners
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
        trimming_rule=getattr(model, 'trimming_rule', 'truncate'),
        trimming_threshold=getattr(model, 'trimming_threshold', 1e-2),
        weights=getattr(model, 'weights', None),
        random_state=getattr(model, 'random_state', None),
        n_jobs=getattr(model, 'n_jobs', 1),
    )

    estimate_args: Dict[str, Any] = dict(fit_args or {})
    store_diagnostics_short = bool(
        estimate_args.pop("store_diagnostics", estimate_args.pop("diagnostic_data", False))
    )

    # Fit short model.
    irm_short.fit(store_diagnostics=store_diagnostics_short)

    # Estimate score for short model (ATE/ATTE only, independent of model.score=GATE state).
    estimate_args["score"] = _resolve_sensitivity_score(
        effect_estimation=effect_estimation,
        model=model,
        explicit_score=estimate_args.get("score"),
    )
    irm_short.estimate(**estimate_args)

    # Long model stats
    theta_long = float(model.coef_[0])

    # Short model stats
    theta_short = float(irm_short.coef_[0])

    # Compute residual-based strengths on the long model
    y_cached = getattr(model, "_y", None)
    d_cached = getattr(model, "_d", None)
    y = np.asarray(
        df_long[outcome_name].to_numpy(dtype=float) if y_cached is None else y_cached,
        dtype=float,
    )
    d = np.asarray(
        df_long[treatment_name].to_numpy(dtype=float) if d_cached is None else d_cached,
        dtype=float,
    )
    m_hat = np.asarray(model.m_hat_, dtype=float)
    g0 = np.asarray(model.g0_hat_, dtype=float)
    g1 = np.asarray(model.g1_hat_, dtype=float)

    r_y = y - (d * g1 + (1.0 - d) * g0)
    r_d = d - m_hat

    def _center(a: np.ndarray) -> np.ndarray:
        """Center an array by its arithmetic mean."""
        return a - np.mean(a)

    def _center_w(a: np.ndarray, w: np.ndarray) -> np.ndarray:
        """Center an array by its weighted mean."""
        w = np.asarray(w, float)
        a = np.asarray(a, float)
        sw = float(np.sum(w))
        mu = float(np.sum(w * a)) / (sw if sw > 1e-12 else 1.0)
        return a - mu

    def _ols_r2_and_fit(yv: np.ndarray, Z: np.ndarray, w: Optional[np.ndarray] = None) -> tuple[float, np.ndarray]:
        """Stable (weighted) OLS on centered & standardized vars for R^2 and fitted component."""
        Z = np.asarray(Z, dtype=float)
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)
        if w is None:
            # Unweighted
            yv_c = _center(yv)
            Zc = Z - np.nanmean(Z, axis=0, keepdims=True)
            col_std = np.nanstd(Zc, axis=0, ddof=0)
            valid = np.isfinite(col_std) & (col_std > 1e-12)
            if not np.any(valid):
                return 0.0, np.zeros_like(yv_c)
            Zcs = Zc[:, valid] / col_std[valid]
            Zcs = np.nan_to_num(Zcs, nan=0.0, posinf=0.0, neginf=0.0)
            yv_c = np.nan_to_num(np.asarray(yv_c, float), nan=0.0, posinf=0.0, neginf=0.0)
            from numpy.linalg import lstsq
            beta, *_ = lstsq(Zcs, yv_c, rcond=1e-12)
            yhat = Zcs @ beta
            denom = float(np.dot(yv_c, yv_c))
            if not np.isfinite(denom) or denom <= 1e-12:
                return 0.0, np.zeros_like(yv_c)
            num = float(np.dot(yhat, yhat))
            if not np.isfinite(num) or num < 0.0:
                return 0.0, np.zeros_like(yv_c)
            r2 = float(np.clip(num / denom, 0.0, 1.0))
            return r2, yhat
        else:
            # Weighted
            w = np.asarray(w, float)
            # sanitize weights and features to avoid NaN/inf propagation
            w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
            Z = np.asarray(Z, float)
            Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
            sw = float(np.sum(w))
            if not np.isfinite(sw) or sw <= 1e-12:
                return 0.0, np.zeros_like(yv, dtype=float)
            yv_c = _center_w(yv, w)
            # Weighted column means
            muZ = (w[:, None] * Z).sum(axis=0) / sw
            Zc = Z - muZ
            # Weighted std per column
            var = (w[:, None] * (Zc * Zc)).sum(axis=0) / sw
            std = np.sqrt(np.maximum(var, 0.0))
            valid = np.isfinite(std) & (std > 1e-12)
            if not np.any(valid):
                return 0.0, np.zeros_like(yv_c)
            Zcs = Zc[:, valid] / std[valid]
            Zcs = np.nan_to_num(Zcs, nan=0.0, posinf=0.0, neginf=0.0)
            yv_c = np.nan_to_num(np.asarray(yv_c, float), nan=0.0, posinf=0.0, neginf=0.0)
            # Weighted least squares via sqrt(w)
            swr = np.sqrt(np.clip(w, 0.0, np.inf))
            Zsw = Zcs * swr[:, None]
            ysw = yv_c * swr
            from numpy.linalg import lstsq
            beta, *_ = lstsq(Zsw, ysw, rcond=1e-12)
            yhat = Zcs @ beta
            denom = float(np.dot(ysw, ysw))
            if not np.isfinite(denom) or denom <= 1e-12:
                return 0.0, np.zeros_like(yv_c)
            num = float(np.dot(swr * yhat, swr * yhat))
            if not np.isfinite(num) or num < 0.0:
                return 0.0, np.zeros_like(yv_c)
            r2 = float(np.clip(num / denom, 0.0, 1.0))
            return r2, yhat

    Z = df_long[benchmarking_set].to_numpy(dtype=float)
    # ATT weighting if applicable
    p = float(np.mean(d)) if (np.isfinite(np.mean(d)) and np.mean(d) > 0.0) else 1.0
    w_att = np.where(d > 0.5, 1.0 / max(p, 1e-12), 0.0)
    is_att = str(estimate_args["score"]).upper().startswith('ATT')
    weights = w_att if is_att else None

    R2y, yhat_u = _ols_r2_and_fit(r_y, Z, w=weights)
    R2d, dhat_u = _ols_r2_and_fit(r_d, Z, w=weights)
    r2_y = float(R2y)
    r2_d = float(R2d)

    def _safe_corr(u: np.ndarray, v: np.ndarray, w: Optional[np.ndarray] = None) -> float:
        """Compute a bounded correlation with optional weighting."""
        if w is None:
            u = _center(u); v = _center(v)
            su, sv = np.std(u), np.std(v)
            if not (np.isfinite(su) and np.isfinite(sv)) or su <= 0 or sv <= 0:
                return 0.0
            val = float(np.corrcoef(u, v)[0, 1])
            return float(np.clip(val, -1.0, 1.0))
        u = _center_w(u, w); v = _center_w(v, w)
        sw = float(np.sum(w))
        su = np.sqrt(max(0.0, float(np.sum(w * u * u)) / (sw if sw > 1e-12 else 1.0)))
        sv = np.sqrt(max(0.0, float(np.sum(w * v * v)) / (sw if sw > 1e-12 else 1.0)))
        sv = np.sqrt(max(0.0, float(np.sum(w * v * v)) / (sw if sw > 1e-12 else 1.0)))
        if su <= 0 or sv <= 0:
            return 0.0
        cov = float(np.sum(w * u * v)) / (sw if sw > 1e-12 else 1.0)
        val = cov / (su * sv)
        return float(np.clip(val, -1.0, 1.0))

    rho = _safe_corr(yhat_u, dhat_u, w=weights)

    delta = theta_long - theta_short

    df_benchmark = pd.DataFrame(
        {
            "r2_y": [r2_y],
            "r2_d": [r2_d],
            "rho": [rho],
            "theta_long": [theta_long],
            "theta_short": [theta_short],
            "delta": [delta],
        },
        index=[treatment_name],
    )
    return df_benchmark


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
    """Compute bias-aware components and cache them.

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
    """
    res = compute_bias_aware_ci(
        effect_estimation,
        r2_y=r2_y,
        r2_d=r2_d,
        rho=rho,
        H0=H0,
        alpha=alpha,
        use_signed_rr=use_signed_rr
    )

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
