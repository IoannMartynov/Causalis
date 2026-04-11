"""Score diagnostics focused on orthogonality and EIF stability."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from causalis.data_contracts.causal_estimate import CausalEstimate
from causalis.dgp.causaldata import CausalData
from causalis.scenarios.unconfoundedness._score_utils import (
    _normalize_ipw_terms,
    _resolve_ate_weights,
)
from causalis.scenarios.unconfoundedness.refutation._shared import (
    _normalize_score,
    _validate_estimate_matches_data,
)


def _resolve_trimming_threshold(
    trimming_threshold: Optional[float],
    diagnostic_data: Any,
    estimate: CausalEstimate,
) -> float:
    """Resolve the clipping threshold from explicit, diagnostic, or model state."""
    if trimming_threshold is not None:
        return float(trimming_threshold)

    trim_thr = getattr(diagnostic_data, "trimming_threshold", None)
    if trim_thr is None:
        trim_thr = estimate.model_options.get("trimming_threshold", None)
    if trim_thr is None:
        trim_thr = 0.01
    return float(trim_thr)


def _resolve_normalize_ipw(score: str, diagnostic_data: Any, estimate: CausalEstimate) -> bool:
    """Resolve whether normalized IPW should be used for diagnostics."""
    normalize_ipw = getattr(diagnostic_data, "normalize_ipw", None)
    if normalize_ipw is None:
        normalize_ipw = estimate.model_options.get("normalize_ipw", False)
    if score == "ATTE":
        return False
    return bool(normalize_ipw)


def _aipw_score_ate(
    y: np.ndarray,
    d: np.ndarray,
    g0: np.ndarray,
    g1: np.ndarray,
    m: np.ndarray,
    theta: float,
    trimming_threshold: float,
    *,
    normalize_ipw: bool,
    w: np.ndarray,
    w_bar: np.ndarray,
) -> np.ndarray:
    """Compute per-observation ATE score residuals."""
    m_clipped = np.clip(m, trimming_threshold, 1.0 - trimming_threshold)
    h1, h0 = _normalize_ipw_terms(d, m_clipped, normalize_ipw=normalize_ipw)
    u0 = y - g0
    u1 = y - g1
    psi_b = w * (g1 - g0) + w_bar * (u1 * h1 - u0 * h0)
    return psi_b - theta


def _aipw_score_atte(
    y: np.ndarray,
    d: np.ndarray,
    g0: np.ndarray,
    m: np.ndarray,
    theta: float,
    trimming_threshold: float,
) -> np.ndarray:
    """Compute per-observation ATTE score residuals."""
    m_clipped = np.clip(m, trimming_threshold, 1.0 - trimming_threshold)
    p_treated = float(np.mean(d))
    gamma = m_clipped / (1.0 - m_clipped)
    num = d * (y - g0 - theta) - (1.0 - d) * gamma * (y - g0)
    return num / (p_treated + 1e-12)


def _orthogonality_derivatives_ate(
    x_basis: np.ndarray,
    y: np.ndarray,
    d: np.ndarray,
    g0: np.ndarray,
    g1: np.ndarray,
    m: np.ndarray,
    trimming_threshold: float,
    *,
    normalize_ipw: bool,
    w: np.ndarray,
    w_bar: np.ndarray,
) -> pd.DataFrame:
    """Estimate finite-basis orthogonality derivatives for the ATE score."""
    n, b = x_basis.shape
    m_clipped = np.clip(m, trimming_threshold, 1.0 - trimming_threshold)
    h1, h0 = _normalize_ipw_terms(d, m_clipped, normalize_ipw=normalize_ipw)

    dg1_terms = x_basis * (w - w_bar * h1)[:, None]
    dg1 = dg1_terms.mean(axis=0)
    dg1_se = dg1_terms.std(axis=0, ddof=1) / np.sqrt(n)

    dg0_terms = x_basis * (-w + w_bar * h0)[:, None]
    dg0 = dg0_terms.mean(axis=0)
    dg0_se = dg0_terms.std(axis=0, ddof=1) / np.sqrt(n)

    u0 = y - g0
    u1 = y - g1
    m_summand = w_bar * (u1 * h1 / m_clipped + u0 * h0 / (1.0 - m_clipped))
    dm_terms = -x_basis * m_summand[:, None]
    dm = dm_terms.mean(axis=0)
    dm_se = dm_terms.std(axis=0, ddof=1) / np.sqrt(n)

    return pd.DataFrame(
        {
            "basis": np.arange(b),
            "d_g1": dg1,
            "se_g1": dg1_se,
            "t_g1": dg1 / np.maximum(dg1_se, 1e-12),
            "d_g0": dg0,
            "se_g0": dg0_se,
            "t_g0": dg0 / np.maximum(dg0_se, 1e-12),
            "d_m": dm,
            "se_m": dm_se,
            "t_m": dm / np.maximum(dm_se, 1e-12),
        }
    )


def _orthogonality_derivatives_atte(
    x_basis: np.ndarray,
    y: np.ndarray,
    d: np.ndarray,
    g0: np.ndarray,
    m: np.ndarray,
    trimming_threshold: float,
) -> pd.DataFrame:
    """Estimate finite-basis orthogonality derivatives for the ATTE score."""
    n, b = x_basis.shape
    p_treated = float(np.mean(d))
    m_clipped = np.clip(m, trimming_threshold, 1.0 - trimming_threshold)
    odds = m_clipped / (1.0 - m_clipped)

    dg0_terms = x_basis * (((1.0 - d) * odds - d) / (p_treated + 1e-12))[:, None]
    dg0 = dg0_terms.mean(axis=0)
    dg0_se = dg0_terms.std(axis=0, ddof=1) / np.sqrt(n)

    dm_terms = -x_basis * (((1.0 - d) * (y - g0)) / ((p_treated + 1e-12) * (1.0 - m_clipped) ** 2))[:, None]
    dm = dm_terms.mean(axis=0)
    dm_se = dm_terms.std(axis=0, ddof=1) / np.sqrt(n)

    return pd.DataFrame(
        {
            "basis": np.arange(b),
            "d_g1": np.zeros(b),
            "se_g1": np.zeros(b),
            "t_g1": np.zeros(b),
            "d_g0": dg0,
            "se_g0": dg0_se,
            "t_g0": dg0 / np.maximum(dg0_se, 1e-12),
            "d_m": dm,
            "se_m": dm_se,
            "t_m": dm / np.maximum(dm_se, 1e-12),
        }
    )


def _influence_summary(
    y: np.ndarray,
    d: np.ndarray,
    g0: np.ndarray,
    g1: np.ndarray,
    m: np.ndarray,
    theta: float,
    score: str,
    trimming_threshold: float,
    *,
    normalize_ipw: bool,
    w: np.ndarray,
    w_bar: np.ndarray,
    psi_override: Optional[np.ndarray] = None,
    k: int = 10,
) -> Dict[str, Any]:
    """Summarize tail behavior and most influential score contributions."""
    if psi_override is not None:
        psi = np.asarray(psi_override, dtype=float).ravel()
    elif score == "ATE":
        psi = _aipw_score_ate(
            y,
            d,
            g0,
            g1,
            m,
            theta,
            trimming_threshold,
            normalize_ipw=normalize_ipw,
            w=w,
            w_bar=w_bar,
        )
    else:
        psi = _aipw_score_atte(y, d, g0, m, theta, trimming_threshold)

    n = int(psi.size)
    se = float(np.std(psi, ddof=1) / np.sqrt(n)) if n > 1 else 0.0

    abs_psi = np.abs(psi)
    med = float(np.median(abs_psi)) if n > 0 else float("nan")
    p99 = float(np.quantile(abs_psi, 0.99)) if n > 0 else float("nan")

    var = float(np.var(psi, ddof=1)) if n > 1 else 0.0
    if var > 0.0 and np.isfinite(var):
        kurt = float(np.mean((psi - float(np.mean(psi))) ** 4) / (var ** 2 + 1e-12))
    else:
        kurt = float("nan")

    idx = np.argsort(-abs_psi)[: min(int(k), n)]
    top = pd.DataFrame(
        {
            "i": idx,
            "psi": psi[idx],
            "m": m[idx],
            "res_t": d[idx] * (y[idx] - g1[idx]),
            "res_c": (1.0 - d[idx]) * (y[idx] - g0[idx]),
        }
    )

    return {
        "se_plugin": se,
        "kurtosis": kurt,
        "p99_over_med": float(p99 / (med + 1e-12)) if np.isfinite(p99) and np.isfinite(med) else float("nan"),
        "top_influential": top,
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


def _two_sided_pvalue_from_t(t_stat: float) -> float:
    """Approximate a two-sided Gaussian p-value from a t-like statistic."""
    if not np.isfinite(t_stat):
        return float("nan")
    return float(math.erfc(abs(float(t_stat)) / math.sqrt(2.0)))


def _oos_moment_test_from_psi(
    psi_a: np.ndarray,
    psi_b: np.ndarray,
    folds: np.ndarray,
) -> Dict[str, Any]:
    """Compute out-of-sample moment checks using fold-wise leave-one-fold fits."""
    fold_rows = []
    psi_all: list[np.ndarray] = []

    fold_ids = np.asarray(folds).ravel()
    unique_folds = np.unique(fold_ids)

    for fold_value in unique_folds:
        test_mask = fold_ids == fold_value
        train_mask = ~test_mask
        n_test = int(np.sum(test_mask))
        n_train = int(np.sum(train_mask))

        if n_test == 0 or n_train == 0:
            continue

        mean_a_train = float(np.mean(psi_a[train_mask]))
        mean_b_train = float(np.mean(psi_b[train_mask]))
        if (not np.isfinite(mean_a_train)) or abs(mean_a_train) <= 1e-12 or (not np.isfinite(mean_b_train)):
            theta_minus_k = float("nan")
            psi_k = np.array([], dtype=float)
            mean_k = float("nan")
            var_k = float("nan")
        else:
            theta_minus_k = float(-mean_b_train / mean_a_train)
            psi_k = np.asarray(psi_b[test_mask] + psi_a[test_mask] * theta_minus_k, dtype=float).ravel()
            mean_k = float(np.mean(psi_k)) if psi_k.size > 0 else float("nan")
            var_k = float(np.var(psi_k, ddof=1)) if psi_k.size > 1 else float("nan")
            if psi_k.size > 0 and np.all(np.isfinite(psi_k)):
                psi_all.append(psi_k)

        fold_rows.append(
            {
                "fold": fold_value,
                "n": n_test,
                "theta_minus_k": theta_minus_k,
                "psi_mean": mean_k,
                "psi_var": var_k,
            }
        )

    fold_table = pd.DataFrame(fold_rows)
    required_cols = ["fold", "n", "theta_minus_k", "psi_mean", "psi_var"]
    if fold_table.empty:
        fold_table = pd.DataFrame(columns=required_cols)
    else:
        fold_table = fold_table[required_cols]

    t_fold = float("nan")
    if not fold_table.empty and np.all(np.isfinite(fold_table["psi_mean"].to_numpy(dtype=float))) and np.all(
        np.isfinite(fold_table["psi_var"].to_numpy(dtype=float))
    ):
        n_k = fold_table["n"].to_numpy(dtype=float)
        mean_k = fold_table["psi_mean"].to_numpy(dtype=float)
        var_k = fold_table["psi_var"].to_numpy(dtype=float)
        denom = float(np.sqrt(np.sum(n_k * var_k)))
        if denom > 0.0 and np.isfinite(denom):
            t_fold = float(np.sum(n_k * mean_k) / denom)

    t_strict = float("nan")
    if psi_all:
        psi_concat = np.concatenate(psi_all)
        if psi_concat.size > 1 and np.all(np.isfinite(psi_concat)):
            psi_mean_all = float(np.mean(psi_concat))
            psi_var_all = float(np.var(psi_concat, ddof=1))
            denom = float(np.sqrt(psi_var_all / psi_concat.size)) if psi_var_all > 0.0 else float("nan")
            if np.isfinite(denom) and denom > 0.0:
                t_strict = float(psi_mean_all / denom)

    return {
        "available": bool(not fold_table.empty and (np.isfinite(t_fold) or np.isfinite(t_strict))),
        "oos_tstat_fold": t_fold,
        "oos_tstat_strict": t_strict,
        "p_value_fold": _two_sided_pvalue_from_t(t_fold),
        "p_value_strict": _two_sided_pvalue_from_t(t_strict),
        "fold_table": fold_table,
    }


def run_score_diagnostics(
    data: CausalData,
    estimate: CausalEstimate,
    *,
    trimming_threshold: Optional[float] = None,
    n_basis_funcs: Optional[int] = None,
    return_summary: bool = True,
) -> Dict[str, Any]:
    """Run orthogonality and influence diagnostics for ATE or ATTE scores.

    The main object is the per-observation score contribution. For ATE, this
    diagnostic uses

    .. math::

        \hat\psi_i =
        w_i(\hat g_1(X_i) - \hat g_0(X_i))
        + \bar w_i
        \left[
        (Y_i - \hat g_1(X_i)) \frac{D_i}{\hat m_i}
        -
        (Y_i - \hat g_0(X_i)) \frac{1-D_i}{1-\hat m_i}
        \right]
        - \hat\theta.

    Good score behavior means:

    - the empirical score average is close to zero,
    - finite-basis derivatives with respect to nuisance parts are small,
    - the influence distribution is not driven by a tiny number of very large
      :math:`|\hat\psi_i|`.

    Parameters
    ----------
    data : CausalData
        Dataset used to fit the estimator.
    estimate : CausalEstimate
        Effect estimate with ``diagnostic_data`` containing nuisance predictions
        and optionally cached score arrays.
    trimming_threshold : float, optional
        Propensity clipping threshold. If omitted, the value is inferred from
        diagnostic or model metadata.
    n_basis_funcs : int, optional
        Number of simple basis functions used in orthogonality checks. Defaults
        to one intercept plus all available confounders.
    return_summary : bool, default True
        Include a compact summary table in the returned payload.

    Returns
    -------
    Dict[str, Any]
        Diagnostic report with orthogonality checks, influence summaries,
        optional out-of-sample tests, and a summary table.

    Raises
    ------
    ValueError
        If required diagnostic arrays are missing or have incompatible shapes.

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
    >>> report = run_score_diagnostics(data, estimate)
    >>> report["summary"]  # doctest: +SKIP
    >>> report["influence"]["top_influential"].head()  # doctest: +SKIP
    """
    _validate_estimate_matches_data(data=data, estimate=estimate, require_confounders=True)

    diagnostic_data = estimate.diagnostic_data
    if diagnostic_data is None:
        raise ValueError(
            "Missing estimate.diagnostic_data. "
            "Fit IRM with store_diagnostics=True and call estimate() first."
        )

    m_raw = getattr(diagnostic_data, "m_hat", None)
    g0_raw = getattr(diagnostic_data, "g0_hat", None)
    if m_raw is None or g0_raw is None:
        raise ValueError("estimate.diagnostic_data must include `m_hat` and `g0_hat`.")

    score = _normalize_score(getattr(diagnostic_data, "score", estimate.estimand))
    trimming_thr = _resolve_trimming_threshold(trimming_threshold, diagnostic_data, estimate)
    normalize_ipw = _resolve_normalize_ipw(score, diagnostic_data, estimate)

    y_raw = getattr(diagnostic_data, "y", None)
    if y_raw is None:
        y_raw = data.get_df()[str(data.outcome_name)].to_numpy(dtype=float)

    d_raw = getattr(diagnostic_data, "d", None)
    if d_raw is None:
        d_raw = data.get_df()[str(data.treatment_name)].to_numpy(dtype=float)

    g1_raw = getattr(diagnostic_data, "g1_hat", None)
    if g1_raw is None:
        g1_raw = np.asarray(g0_raw, dtype=float)

    x_raw = getattr(diagnostic_data, "x", None)
    if x_raw is None:
        x_raw = data.get_df()[list(data.confounders)].to_numpy(dtype=float)

    psi_raw = getattr(diagnostic_data, "psi", None)
    psi_b_raw = getattr(diagnostic_data, "psi_b", None)
    folds_raw = getattr(diagnostic_data, "folds", None)
    diag_w_raw = getattr(diagnostic_data, "w", None)
    diag_w_bar_raw = getattr(diagnostic_data, "w_bar", None)

    y = np.asarray(y_raw, dtype=float).ravel()
    d = (np.asarray(d_raw, dtype=float).ravel() > 0.5).astype(float)
    g0 = np.asarray(g0_raw, dtype=float).ravel()
    g1 = np.asarray(g1_raw, dtype=float).ravel()
    m = np.asarray(m_raw, dtype=float).ravel()
    x = np.asarray(x_raw, dtype=float)

    if x.ndim != 2:
        raise ValueError("Confounder matrix must be 2D with shape (n, p).")

    n = int(y.size)
    if any(arr.size != n for arr in (d, g0, g1, m)) or x.shape[0] != n:
        raise ValueError("All diagnostic arrays must have matching sample size n.")

    if score == "ATE" and (diag_w_raw is None or diag_w_bar_raw is None):
        model_ref = getattr(diagnostic_data, "_model", None)
        if model_ref is not None and hasattr(model_ref, "_get_weights"):
            try:
                w_model, w_bar_model = model_ref._get_weights(
                    n=n,
                    m_hat_adj=np.clip(m, trimming_thr, 1.0 - trimming_thr),
                    d=d.astype(int),
                    score="ATE",
                )
                if diag_w_raw is None:
                    diag_w_raw = w_model
                if diag_w_bar_raw is None:
                    diag_w_bar_raw = w_bar_model
            except Exception:
                pass

    if score == "ATE":
        w, w_bar = _resolve_ate_weights(n=n, w_raw=diag_w_raw, w_bar_raw=diag_w_bar_raw)
    else:
        p_treated = float(np.mean(d))
        w = d / (p_treated + 1e-12)
        w_bar = np.clip(m, trimming_thr, 1.0 - trimming_thr) / (p_treated + 1e-12)

    psi = None
    if psi_raw is not None:
        psi = np.asarray(psi_raw, dtype=float).ravel()
        if psi.size != n:
            psi = None
    psi_b = None
    if psi_b_raw is not None:
        psi_b = np.asarray(psi_b_raw, dtype=float).ravel()
        if psi_b.size != n:
            psi_b = None
    folds = None
    if folds_raw is not None:
        folds = np.asarray(folds_raw).ravel()
        if folds.size != n:
            folds = None

    # Build basis: constant + standardized first confounders (if available).
    if n_basis_funcs is None:
        n_basis_funcs = int(x.shape[1] + 1)
    n_covs = min(max(int(n_basis_funcs) - 1, 0), int(x.shape[1]))

    if n_covs > 0:
        x_sel = x[:, :n_covs]
        x_std = (x_sel - np.mean(x_sel, axis=0)) / (np.std(x_sel, axis=0) + 1e-8)
        x_basis = np.c_[np.ones(n), x_std]
    else:
        x_basis = np.ones((n, 1))

    finite_rows = (
        np.isfinite(y)
        & np.isfinite(d)
        & np.isfinite(g0)
        & np.isfinite(g1)
        & np.isfinite(m)
        & np.isfinite(w)
        & np.isfinite(w_bar)
        & np.all(np.isfinite(x_basis), axis=1)
    )
    if psi is not None:
        finite_rows = finite_rows & np.isfinite(psi)
    if psi_b is not None:
        finite_rows = finite_rows & np.isfinite(psi_b)
    if folds is not None:
        finite_rows = finite_rows & np.isfinite(np.asarray(folds, dtype=float))

    y = y[finite_rows]
    d = d[finite_rows]
    g0 = g0[finite_rows]
    g1 = g1[finite_rows]
    m = m[finite_rows]
    w = w[finite_rows]
    w_bar = w_bar[finite_rows]
    x_basis = x_basis[finite_rows]
    psi = psi[finite_rows] if psi is not None else None
    psi_b = psi_b[finite_rows] if psi_b is not None else None
    folds = folds[finite_rows] if folds is not None else None

    if score == "ATE":
        psi_a = -np.ones_like(d, dtype=float)
    else:
        psi_a = -w

    theta = float(estimate.value)
    influence = _influence_summary(
        y=y,
        d=d,
        g0=g0,
        g1=g1,
        m=m,
        theta=theta,
        score=score,
        trimming_threshold=trimming_thr,
        normalize_ipw=normalize_ipw,
        w=w,
        w_bar=w_bar,
        psi_override=psi,
    )

    if score == "ATE":
        ortho = _orthogonality_derivatives_ate(
            x_basis=x_basis,
            y=y,
            d=d,
            g0=g0,
            g1=g1,
            m=m,
            trimming_threshold=trimming_thr,
            normalize_ipw=normalize_ipw,
            w=w,
            w_bar=w_bar,
        )
    else:
        ortho = _orthogonality_derivatives_atte(
            x_basis=x_basis,
            y=y,
            d=d,
            g0=g0,
            m=m,
            trimming_threshold=trimming_thr,
        )

    max_t_g1 = float(np.nanmax(np.abs(ortho["t_g1"].to_numpy(dtype=float))))
    max_t_g0 = float(np.nanmax(np.abs(ortho["t_g0"].to_numpy(dtype=float))))
    max_t_m = float(np.nanmax(np.abs(ortho["t_m"].to_numpy(dtype=float))))

    thresholds = {
        "tail_ratio_warn": 10.0,
        "tail_ratio_strong": 20.0,
        "kurt_warn": 10.0,
        "kurt_strong": 30.0,
        "t_warn": 2.0,
        "t_strong": 4.0,
    }

    oos_moment_test = {
        "available": False,
        "oos_tstat_fold": float("nan"),
        "oos_tstat_strict": float("nan"),
        "p_value_fold": float("nan"),
        "p_value_strict": float("nan"),
        "fold_table": pd.DataFrame(columns=["fold", "n", "theta_minus_k", "psi_mean", "psi_var"]),
    }
    if psi_b is not None and folds is not None and psi_b.size == d.size and folds.size == d.size:
        oos_moment_test = _oos_moment_test_from_psi(psi_a=psi_a, psi_b=psi_b, folds=folds)

    oos_abs_t_candidates = [
        abs(float(oos_moment_test["oos_tstat_fold"])) if np.isfinite(oos_moment_test["oos_tstat_fold"]) else np.nan,
        abs(float(oos_moment_test["oos_tstat_strict"])) if np.isfinite(oos_moment_test["oos_tstat_strict"]) else np.nan,
    ]
    oos_abs_t = float(np.nanmax(oos_abs_t_candidates)) if np.any(np.isfinite(oos_abs_t_candidates)) else float("nan")

    flags = {
        "psi_tail_ratio": _grade(
            float(influence["p99_over_med"]),
            thresholds["tail_ratio_warn"],
            thresholds["tail_ratio_strong"],
        ),
        "psi_kurtosis": _grade(
            float(influence["kurtosis"]),
            thresholds["kurt_warn"],
            thresholds["kurt_strong"],
        ),
        "ortho_max_|t|_g1": _grade(max_t_g1, thresholds["t_warn"], thresholds["t_strong"]),
        "ortho_max_|t|_g0": _grade(max_t_g0, thresholds["t_warn"], thresholds["t_strong"]),
        "ortho_max_|t|_m": _grade(max_t_m, thresholds["t_warn"], thresholds["t_strong"]),
        "oos_moment": _grade(oos_abs_t, thresholds["t_warn"], thresholds["t_strong"]),
    }

    level = {"NA": -1, "GREEN": 0, "YELLOW": 1, "RED": 2}
    inv_level = {value: key for key, value in level.items()}
    overall_flag = inv_level[max(level.get(flag, -1) for flag in flags.values())]

    report: Dict[str, Any] = {
        "params": {
            "score": score,
            "trimming_threshold": float(trimming_thr),
            "normalize_ipw": bool(normalize_ipw),
        },
        "orthogonality_derivatives": ortho,
        "influence_diagnostics": influence,
        "oos_moment_test": oos_moment_test,
        "flags": flags,
        "thresholds": thresholds,
        "overall_flag": overall_flag,
        "meta": {
            "n": int(y.size),
            "score": score,
            "used_estimator_psi": bool(psi is not None),
            "uses_custom_weights": bool(score == "ATE" and (diag_w_raw is not None or diag_w_bar_raw is not None)),
        },
    }

    if return_summary:
        summary_rows = [
            {
                "metric": "psi_p99_over_med",
                "value": float(influence["p99_over_med"]),
                "flag": flags["psi_tail_ratio"],
            },
            {
                "metric": "psi_kurtosis",
                "value": float(influence["kurtosis"]),
                "flag": flags["psi_kurtosis"],
            },
        ]
        if score == "ATE":
            summary_rows.append({"metric": "max_|t|_g1", "value": max_t_g1, "flag": flags["ortho_max_|t|_g1"]})
        summary_rows.extend(
            [
                {"metric": "max_|t|_g0", "value": max_t_g0, "flag": flags["ortho_max_|t|_g0"]},
                {"metric": "max_|t|_m", "value": max_t_m, "flag": flags["ortho_max_|t|_m"]},
                {
                    "metric": "oos_max_abs_t",
                    "value": oos_abs_t,
                    "flag": flags["oos_moment"],
                },
            ]
        )
        summary = pd.DataFrame(summary_rows)
        report["summary"] = summary

    return report


__all__ = ["run_score_diagnostics"]
