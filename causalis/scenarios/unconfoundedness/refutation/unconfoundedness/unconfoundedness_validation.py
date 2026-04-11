"""Unconfoundedness diagnostics focused on covariate balance (SMD)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from causalis.data_contracts.causal_estimate import CausalEstimate
from causalis.dgp.causaldata import CausalData
from causalis.scenarios.unconfoundedness.refutation._shared import (
    _normalize_score,
    _validate_estimate_matches_data,
)


@dataclass
class _BalanceInputs:
    """Normalized inputs used by balance diagnostics."""
    x: np.ndarray
    d: np.ndarray
    m_hat: np.ndarray
    w_bar: Optional[np.ndarray]
    names: List[str]
    score: str
    normalize: bool


def _resolve_feature_names(names: Optional[List[str]], p: int) -> List[str]:
    """Return feature names or generate fallback labels for balance tables."""
    if names is not None:
        names_list = [str(name) for name in names]
        if len(names_list) == p:
            return names_list
    return [f"x{j + 1}" for j in range(p)]


def _extract_balance_inputs(
    *,
    data: CausalData,
    estimate: CausalEstimate,
    normalize: Optional[bool],
) -> _BalanceInputs:
    """Extract and validate balance-diagnostic arrays from estimate metadata."""
    _validate_estimate_matches_data(data=data, estimate=estimate, require_confounders=True)

    diagnostic_data = estimate.diagnostic_data
    if diagnostic_data is None:
        raise ValueError(
            "Missing estimate.diagnostic_data. "
            "Fit IRM with store_diagnostics=True and call estimate() first."
        )

    m_hat_raw = getattr(diagnostic_data, "m_hat", None)
    if m_hat_raw is None:
        raise ValueError("estimate.diagnostic_data must include `m_hat`.")

    x_raw = getattr(diagnostic_data, "x", None)
    if x_raw is None:
        x_raw = data.get_df()[list(data.confounders)].to_numpy(dtype=float)

    d_raw = getattr(diagnostic_data, "d", None)
    if d_raw is None:
        d_raw = data.get_df()[str(data.treatment_name)].to_numpy(dtype=float)

    score_raw = getattr(diagnostic_data, "score", estimate.estimand)
    score = _normalize_score(score_raw)

    normalize_used = normalize
    if normalize_used is None:
        normalize_used = getattr(diagnostic_data, "normalize_ipw", None)
    if normalize_used is None:
        normalize_used = bool(estimate.model_options.get("normalize_ipw", False))

    x = np.asarray(x_raw, dtype=float)
    d = np.asarray(d_raw, dtype=float).ravel()
    m_hat = np.asarray(m_hat_raw, dtype=float).ravel()

    if x.ndim != 2:
        raise ValueError("Confounder matrix must be 2D with shape (n, p).")

    n, p = x.shape
    if d.size != n or m_hat.size != n:
        raise ValueError("Confounders, treatment, and propensity predictions must share row count n.")

    w_bar: Optional[np.ndarray] = None
    if score == "ATE":
        w_bar_raw = getattr(diagnostic_data, "w_bar", None)
        if w_bar_raw is None:
            model_ref = getattr(diagnostic_data, "_model", None)
            if model_ref is not None and hasattr(model_ref, "_get_weights"):
                try:
                    _, w_bar_raw = model_ref._get_weights(
                        n=n,
                        m_hat_adj=m_hat,
                        d=d.astype(int),
                        score="ATE",
                    )
                except Exception:
                    w_bar_raw = None
        if w_bar_raw is not None:
            w_bar = np.asarray(w_bar_raw, dtype=float).ravel()
            if w_bar.size != n:
                raise ValueError(
                    "diagnostic_data.w_bar must match the sample size n "
                    f"(got {w_bar.size}, expected {n})."
                )
            if not np.all(np.isfinite(w_bar)):
                raise ValueError("diagnostic_data.w_bar must contain only finite values.")

    names: Optional[List[str]] = None
    if estimate.confounders:
        names = [str(name) for name in estimate.confounders]
    else:
        diag_names = getattr(diagnostic_data, "x_names", None)
        if diag_names is not None:
            names = [str(name) for name in list(diag_names)]
        else:
            names = [str(name) for name in data.confounders]

    # Canonical ATTE balance uses unnormalized ATT weights.
    if score == "ATTE":
        normalize_used = False

    return _BalanceInputs(
        x=x,
        d=d,
        m_hat=m_hat,
        w_bar=w_bar,
        names=_resolve_feature_names(names, p),
        score=score,
        normalize=bool(normalize_used),
    )


def _balance_smd(
    inputs: _BalanceInputs,
    *,
    threshold: float,
) -> Dict[str, Any]:
    """Compute weighted and unweighted standardized mean differences."""
    x = inputs.x
    d = inputs.d
    m_hat = np.clip(inputs.m_hat, 1e-12, 1.0 - 1e-12)

    n, p = x.shape

    if inputs.score == "ATE":
        target_w = inputs.w_bar if inputs.w_bar is not None else np.ones_like(d, dtype=float)
        w1 = target_w * d / m_hat
        w0 = target_w * (1.0 - d) / (1.0 - m_hat)
    else:
        w1 = d
        w0 = (1.0 - d) * (m_hat / (1.0 - m_hat))

    if inputs.normalize:
        w1_mean = float(np.mean(w1))
        w0_mean = float(np.mean(w0))
        w1 = w1 / (w1_mean if w1_mean != 0.0 else 1.0)
        w0 = w0 / (w0_mean if w0_mean != 0.0 else 1.0)

    s1 = float(np.sum(w1))
    s0 = float(np.sum(w0))
    if s1 <= 0.0 or s0 <= 0.0:
        raise RuntimeError("Degenerate weights: zero total mass in a pseudo-population.")

    mu1 = (w1[:, None] * x).sum(axis=0) / s1
    mu0 = (w0[:, None] * x).sum(axis=0) / s0

    var1 = (w1[:, None] * (x - mu1) ** 2).sum(axis=0) / s1
    var0 = (w0[:, None] * (x - mu0) ** 2).sum(axis=0) / s0
    s_pool = np.sqrt(0.5 * (np.maximum(var1, 0.0) + np.maximum(var0, 0.0)))

    smd_weighted = np.full(p, np.nan, dtype=float)
    zero_both = (var1 <= 1e-16) & (var0 <= 1e-16)
    diff = np.abs(mu1 - mu0)
    mask = (~zero_both) & (s_pool > 1e-16)
    smd_weighted[mask] = diff[mask] / s_pool[mask]
    smd_weighted[zero_both & (diff <= 1e-16)] = 0.0
    smd_weighted[zero_both & (diff > 1e-16)] = np.inf

    mask1 = d.astype(bool)
    mask0 = ~mask1
    if not np.any(mask1) or not np.any(mask0):
        smd_unweighted = np.full(p, np.nan, dtype=float)
    else:
        x1 = x[mask1]
        x0 = x[mask0]
        mu1_u = x1.mean(axis=0)
        mu0_u = x0.mean(axis=0)
        var1_u = x1.var(axis=0, ddof=0)
        var0_u = x0.var(axis=0, ddof=0)
        s_pool_u = np.sqrt(0.5 * (np.maximum(var1_u, 0.0) + np.maximum(var0_u, 0.0)))

        smd_unweighted = np.full(p, np.nan, dtype=float)
        zero_both_u = (var1_u <= 1e-16) & (var0_u <= 1e-16)
        diff_u = np.abs(mu1_u - mu0_u)
        mask_u = (~zero_both_u) & (s_pool_u > 1e-16)
        smd_unweighted[mask_u] = diff_u[mask_u] / s_pool_u[mask_u]
        smd_unweighted[zero_both_u & (diff_u <= 1e-16)] = 0.0
        smd_unweighted[zero_both_u & (diff_u > 1e-16)] = np.inf

    finite = np.isfinite(smd_weighted)
    if np.any(finite):
        frac_violations = float(np.mean(smd_weighted[finite] >= float(threshold)))
        smd_max = float(np.nanmax(smd_weighted[finite]))
        balance_pass = bool((frac_violations < 0.10) and (smd_max < 2.0 * float(threshold)))
    else:
        frac_violations = 0.0
        balance_pass = True
        smd_max = float("nan")

    return {
        "smd_weighted": smd_weighted,
        "smd_unweighted": smd_unweighted,
        "smd_max": smd_max,
        "frac_violations": frac_violations,
        "pass": balance_pass,
        "meta": {"n": int(n), "p": int(p)},
    }


def _grade(value: float, warn: float, strong: float) -> str:
    """Map a scalar diagnostic value to a traffic-light severity flag."""
    if value is None or not np.isfinite(value):
        return "NA"
    value_f = float(value)
    if value_f < warn:
        return "GREEN"
    if value_f < strong:
        return "YELLOW"
    return "RED"


def run_unconfoundedness_diagnostics(
    data: CausalData,
    estimate: CausalEstimate,
    *,
    threshold: float = 0.10,
    normalize: Optional[bool] = None,
    return_summary: bool = True,
) -> Dict[str, Any]:
    """Run covariate-balance diagnostics implied by unconfoundedness.

    The diagnostic compares the treated and control pseudo-populations induced
    by the estimated propensity score. For ATE, the effective weights are

    .. math::

        w_{1i} = \bar w_i \frac{D_i}{\hat m_i},
        \qquad
        w_{0i} = \bar w_i \frac{1-D_i}{1-\hat m_i},

    while for ATTE this implementation uses

    .. math::

        w_{1i} = D_i,
        \qquad
        w_{0i} = (1-D_i)\frac{\hat m_i}{1-\hat m_i}.

    For each confounder :math:`X_j`, the weighted standardized mean
    difference is

    .. math::

        \mathrm{SMD}_j =
        \frac{|\mu_{1j}^{(w)} - \mu_{0j}^{(w)}|}
        {\sqrt{(s_{1j}^{2,(w)} + s_{0j}^{2,(w)}) / 2}}.

    Smaller weighted SMDs are better. A common rule of thumb is to aim for
    :math:`|\mathrm{SMD}| < 0.10`.

    Parameters
    ----------
    data : CausalData
        Dataset used to fit the estimator.
    estimate : CausalEstimate
        Effect estimate with ``diagnostic_data`` containing propensity and, when
        available, weight information.
    threshold : float, default 0.10
        SMD threshold used for warnings and pass/fail summaries.
    normalize : bool, optional
        Override whether pseudo-population weights are mean-normalized.
    return_summary : bool, default True
        Include a compact summary table in the returned payload.

    Returns
    -------
    Dict[str, Any]
        Diagnostic report with weighted balance tables, severity flags, and an
        optional summary DataFrame.

    Raises
    ------
    ValueError
        If required diagnostic arrays are missing or have incompatible shapes.
    RuntimeError
        If balance weights collapse to zero total mass.

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
    >>> report = run_unconfoundedness_diagnostics(data, estimate)
    >>> report["balance"]["smd_max"]  # doctest: +SKIP
    >>> report["balance"]["worst_features"].head()  # doctest: +SKIP
    """
    inputs = _extract_balance_inputs(
        data=data,
        estimate=estimate,
        normalize=normalize,
    )
    balance = _balance_smd(inputs, threshold=threshold)

    smd = pd.Series(balance["smd_weighted"], index=inputs.names, dtype=float, name="SMD_weighted")
    smd_unweighted = pd.Series(
        balance["smd_unweighted"], index=inputs.names, dtype=float, name="SMD_unweighted"
    )
    worst = smd.sort_values(ascending=False).head(10)

    smd_warn = float(threshold)
    smd_strong = 2.0 * float(threshold)
    viol_warn = 0.10
    viol_strong = 0.25

    flags = {
        "balance_max_smd": _grade(float(balance["smd_max"]), smd_warn, smd_strong),
        "balance_violations": _grade(float(balance["frac_violations"]), viol_warn, viol_strong),
    }

    level = {"NA": -1, "GREEN": 0, "YELLOW": 1, "RED": 2}
    inv_level = {value: key for key, value in level.items()}
    overall_flag = inv_level[max(level.get(flag, -1) for flag in flags.values())]

    report: Dict[str, Any] = {
        "params": {
            "score": inputs.score,
            "normalize": inputs.normalize,
            "smd_threshold": float(threshold),
        },
        "balance": {
            "smd": smd,
            "smd_unweighted": smd_unweighted,
            "smd_max": float(balance["smd_max"]),
            "frac_violations": float(balance["frac_violations"]),
            "pass": bool(balance["pass"]),
            "worst_features": worst,
        },
        "flags": flags,
        "overall_flag": overall_flag,
        "meta": balance["meta"],
    }

    if return_summary:
        report["summary"] = pd.DataFrame(
            [
                {
                    "metric": "balance_max_smd",
                    "value": float(balance["smd_max"]),
                    "flag": flags["balance_max_smd"],
                },
                {
                    "metric": "balance_frac_violations",
                    "value": float(balance["frac_violations"]),
                    "flag": flags["balance_violations"],
                },
            ]
        )

    return report


__all__ = ["run_unconfoundedness_diagnostics"]
