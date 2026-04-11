"""Score diagnostics for multi-treatment unconfoundedness."""

from __future__ import annotations

import math
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from causalis.data_contracts.multicausal_estimate import MultiCausalEstimate
from causalis.data_contracts.multicausaldata import MultiCausalData


def _grade(value: float, warn: float, strong: float) -> str:
    if value is None or not np.isfinite(value):
        return "NA"
    v = float(value)
    if v < warn:
        return "GREEN"
    if v < strong:
        return "YELLOW"
    return "RED"


def _two_sided_pvalue_from_t(t_stat: float) -> float:
    if not np.isfinite(t_stat):
        return float("nan")
    return float(math.erfc(abs(float(t_stat)) / math.sqrt(2.0)))


def _validate_estimate_matches_data(data: MultiCausalData, estimate: MultiCausalEstimate) -> None:
    if str(estimate.outcome) != str(data.outcome):
        raise ValueError(
            "estimate.outcome must match data.outcome "
            f"({estimate.outcome!r} != {data.outcome!r})."
        )

    est_treatments = [str(name) for name in list(estimate.treatment)]
    data_treatments = [str(name) for name in list(data.treatment_names)]
    if est_treatments != data_treatments:
        raise ValueError(
            "estimate.treatment must match data.treatment_names in the same order "
            f"({est_treatments!r} != {data_treatments!r})."
        )


def _resolve_trimming_threshold(
    trimming_threshold: Optional[float],
    diagnostic_data: Any,
    estimate: MultiCausalEstimate,
) -> float:
    if trimming_threshold is not None:
        return float(trimming_threshold)

    trim_thr = getattr(diagnostic_data, "trimming_threshold", None)
    if trim_thr is None:
        trim_thr = estimate.model_options.get("trimming_threshold", None)
    if trim_thr is None:
        trim_thr = 0.01
    return float(trim_thr)


def _resolve_normalize_ipw(diagnostic_data: Any, estimate: MultiCausalEstimate) -> bool:
    normalize_ipw = getattr(diagnostic_data, "normalize_ipw", None)
    if normalize_ipw is None:
        normalize_ipw = estimate.model_options.get("normalize_ipw", False)
    return bool(normalize_ipw)


def _resolve_treatment_names(diag: Any, data: MultiCausalData, k: int) -> List[str]:
    names = [str(name) for name in list(data.treatment_names)]
    if len(names) == k:
        return names

    diag_names = getattr(diag, "treatment_names", None)
    if diag_names is None:
        diag_names = getattr(diag, "d_names", None)
    if diag_names is not None:
        diag_names = [str(name) for name in list(diag_names)]
        if len(diag_names) == k:
            return diag_names

    return [f"d_{idx}" for idx in range(k)]


def _comparison_labels(treatment_names: List[str]) -> List[str]:
    baseline = str(treatment_names[0])
    return [f"{name} vs {baseline}" for name in treatment_names[1:]]


def _normalize_rows_to_simplex(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    p = np.maximum(p, 1e-12)
    row_sums = p.sum(axis=1, keepdims=True)
    if np.any(np.isfinite(row_sums) & (row_sums <= 1e-12)):
        raise RuntimeError("Propensity matrix contains rows with zero total probability.")
    safe_sums = np.where(np.isfinite(row_sums) & (row_sums > 1e-12), row_sums, 1.0)
    return p / safe_sums


def _trim_multiclass_propensity(p: np.ndarray, trimming_threshold: float) -> np.ndarray:
    """Lower-trim multiclass propensity and renormalize rows to sum to 1."""
    p = np.asarray(p, dtype=float)
    if p.ndim != 2:
        raise ValueError(f"Propensity matrix must be 2D (n, K). Got shape {p.shape}.")
    n_treatments = p.shape[1]
    if n_treatments < 2:
        raise ValueError("Need at least 2 treatment columns for multiclass propensity.")

    thr = float(trimming_threshold)
    if not np.isfinite(thr) or not (0.0 <= thr < (1.0 / n_treatments)):
        raise ValueError(
            f"trimming_threshold must be finite and in [0, 1/K) for K={n_treatments}; got {thr}."
        )

    p_simplex = _normalize_rows_to_simplex(p)
    if thr <= 0.0:
        return p_simplex

    p_trim = np.maximum(p_simplex, thr)
    return _normalize_rows_to_simplex(p_trim)


def _build_basis(x: np.ndarray, n_basis_funcs: Optional[int]) -> np.ndarray:
    n, p = x.shape
    if n_basis_funcs is None:
        n_basis_funcs = int(p + 1)
    n_covs = min(max(int(n_basis_funcs) - 1, 0), int(p))

    if n_covs <= 0:
        return np.ones((n, 1), dtype=float)

    x_sel = x[:, :n_covs]
    x_std = (x_sel - np.mean(x_sel, axis=0)) / (np.std(x_sel, axis=0) + 1e-8)
    return np.c_[np.ones(n), x_std]


def _normalize_ipw_terms(d: np.ndarray, m: np.ndarray, *, normalize_ipw: bool) -> np.ndarray:
    h = d / m
    if normalize_ipw:
        h_mean = np.mean(h, axis=0, keepdims=True)
        h = h / np.where(h_mean != 0.0, h_mean, 1.0)
    return h


def _resolve_theta(value: Any, n_contrasts: int) -> np.ndarray:
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size == 1 and n_contrasts > 1:
        arr = np.repeat(arr, n_contrasts)
    if arr.size != n_contrasts:
        raise ValueError(
            f"estimate.value must have length {n_contrasts} (number of baseline contrasts), got {arr.size}."
        )
    return arr


def _compute_psi_from_nuisances(
    *,
    y: np.ndarray,
    d: np.ndarray,
    g_hat: np.ndarray,
    m: np.ndarray,
    theta: np.ndarray,
    normalize_ipw: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h = _normalize_ipw_terms(d=d, m=m, normalize_ipw=normalize_ipw)
    u = y[:, None] - g_hat

    psi_b = (
        (g_hat[:, 1:] - g_hat[:, [0]])
        + (u[:, 1:] * h[:, 1:])
        - (u[:, [0]] * h[:, [0]])
    )
    psi = psi_b - theta[None, :]
    return psi, psi_b, h, u


def _orthogonality_derivatives(
    *,
    x_basis: np.ndarray,
    d: np.ndarray,
    m: np.ndarray,
    h: np.ndarray,
    u: np.ndarray,
    comparison_labels: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n, b = x_basis.shape
    j = len(comparison_labels)

    rows: List[Dict[str, Any]] = []
    max_rows: List[Dict[str, Any]] = []

    for idx in range(j):
        k = idx + 1

        d_gk_terms = x_basis * (1.0 - h[:, k])[:, None]
        d_g0_terms = x_basis * (-1.0 + h[:, 0])[:, None]
        d_mk_terms = -x_basis * ((u[:, k] * d[:, k]) / (m[:, k] ** 2))[:, None]
        d_m0_terms = x_basis * ((u[:, 0] * d[:, 0]) / (m[:, 0] ** 2))[:, None]

        d_gk = d_gk_terms.mean(axis=0)
        d_g0 = d_g0_terms.mean(axis=0)
        d_mk = d_mk_terms.mean(axis=0)
        d_m0 = d_m0_terms.mean(axis=0)

        se_gk = d_gk_terms.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.full(b, np.nan)
        se_g0 = d_g0_terms.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.full(b, np.nan)
        se_mk = d_mk_terms.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.full(b, np.nan)
        se_m0 = d_m0_terms.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.full(b, np.nan)

        t_gk = d_gk / np.maximum(se_gk, 1e-12)
        t_g0 = d_g0 / np.maximum(se_g0, 1e-12)
        t_mk = d_mk / np.maximum(se_mk, 1e-12)
        t_m0 = d_m0 / np.maximum(se_m0, 1e-12)

        for basis_idx in range(b):
            rows.append(
                {
                    "comparison": comparison_labels[idx],
                    "basis": int(basis_idx),
                    "d_gk": float(d_gk[basis_idx]),
                    "se_gk": float(se_gk[basis_idx]),
                    "t_gk": float(t_gk[basis_idx]),
                    "d_g0": float(d_g0[basis_idx]),
                    "se_g0": float(se_g0[basis_idx]),
                    "t_g0": float(t_g0[basis_idx]),
                    "d_mk": float(d_mk[basis_idx]),
                    "se_mk": float(se_mk[basis_idx]),
                    "t_mk": float(t_mk[basis_idx]),
                    "d_m0": float(d_m0[basis_idx]),
                    "se_m0": float(se_m0[basis_idx]),
                    "t_m0": float(t_m0[basis_idx]),
                }
            )

        max_rows.append(
            {
                "comparison": comparison_labels[idx],
                "max_|t|_gk": float(np.nanmax(np.abs(t_gk))),
                "max_|t|_g0": float(np.nanmax(np.abs(t_g0))),
                "max_|t|_mk": float(np.nanmax(np.abs(t_mk))),
                "max_|t|_m0": float(np.nanmax(np.abs(t_m0))),
            }
        )

    max_df = pd.DataFrame(max_rows)
    max_df["max_|t|"] = np.nanmax(
        max_df[["max_|t|_gk", "max_|t|_g0", "max_|t|_mk", "max_|t|_m0"]].to_numpy(dtype=float),
        axis=1,
    )

    return pd.DataFrame(rows), max_df


def _influence_summary(
    *,
    psi: np.ndarray,
    m: np.ndarray,
    u: np.ndarray,
    comparison_labels: List[str],
    k_top: int = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n, j = psi.shape
    rows: List[Dict[str, Any]] = []
    top_rows: List[Dict[str, Any]] = []

    for idx in range(j):
        psi_j = psi[:, idx]

        se = float(np.std(psi_j, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
        abs_psi = np.abs(psi_j)
        med = float(np.median(abs_psi)) if n > 0 else float("nan")
        p99 = float(np.quantile(abs_psi, 0.99)) if n > 0 else float("nan")

        var = float(np.var(psi_j, ddof=1)) if n > 1 else float("nan")
        if np.isfinite(var) and var > 0.0:
            kurt = float(np.mean((psi_j - float(np.mean(psi_j))) ** 4) / (var ** 2 + 1e-12))
        else:
            kurt = float("nan")

        rows.append(
            {
                "comparison": comparison_labels[idx],
                "se_plugin": se,
                "kurtosis": kurt,
                "p99_over_med": float(p99 / (med + 1e-12)) if np.isfinite(p99) and np.isfinite(med) else float("nan"),
            }
        )

        top_idx = np.argsort(-abs_psi)[: min(int(k_top), n)]
        for i in top_idx:
            top_rows.append(
                {
                    "comparison": comparison_labels[idx],
                    "i": int(i),
                    "psi": float(psi_j[i]),
                    "m_k": float(m[i, idx + 1]),
                    "residual_k": float(u[i, idx + 1]),
                    "residual_0": float(u[i, 0]),
                }
            )

    return pd.DataFrame(rows), pd.DataFrame(top_rows)


def _oos_moment_test(
    *,
    psi_b: np.ndarray,
    folds: Optional[np.ndarray],
    comparison_labels: List[str],
) -> Dict[str, Any]:
    j = psi_b.shape[1]
    if folds is None:
        return {
            "available": False,
            "by_comparison": pd.DataFrame(
                {
                    "comparison": comparison_labels,
                    "oos_tstat_fold": [float("nan")] * j,
                    "oos_tstat_strict": [float("nan")] * j,
                    "p_value_fold": [float("nan")] * j,
                    "p_value_strict": [float("nan")] * j,
                }
            ),
            "fold_table": pd.DataFrame(columns=["comparison", "fold", "n", "theta_minus_k", "psi_mean", "psi_var"]),
        }

    folds_arr = np.asarray(folds).ravel()
    n = folds_arr.size
    if n != psi_b.shape[0]:
        return {
            "available": False,
            "by_comparison": pd.DataFrame(
                {
                    "comparison": comparison_labels,
                    "oos_tstat_fold": [float("nan")] * j,
                    "oos_tstat_strict": [float("nan")] * j,
                    "p_value_fold": [float("nan")] * j,
                    "p_value_strict": [float("nan")] * j,
                }
            ),
            "fold_table": pd.DataFrame(columns=["comparison", "fold", "n", "theta_minus_k", "psi_mean", "psi_var"]),
        }

    unique_folds = np.unique(folds_arr)
    fold_rows: List[Dict[str, Any]] = []
    by_comp_rows: List[Dict[str, Any]] = []

    for idx, comp in enumerate(comparison_labels):
        psi_b_j = psi_b[:, idx]
        psi_all: List[np.ndarray] = []

        for fold_value in unique_folds:
            test_mask = folds_arr == fold_value
            train_mask = ~test_mask
            n_test = int(np.sum(test_mask))
            n_train = int(np.sum(train_mask))

            if n_test == 0 or n_train == 0:
                continue

            theta_minus_k = float(np.mean(psi_b_j[train_mask]))
            psi_k = np.asarray(psi_b_j[test_mask] - theta_minus_k, dtype=float).ravel()
            psi_mean = float(np.mean(psi_k)) if psi_k.size > 0 else float("nan")
            psi_var = float(np.var(psi_k, ddof=1)) if psi_k.size > 1 else float("nan")
            if psi_k.size > 0 and np.all(np.isfinite(psi_k)):
                psi_all.append(psi_k)

            fold_rows.append(
                {
                    "comparison": comp,
                    "fold": int(fold_value),
                    "n": n_test,
                    "theta_minus_k": theta_minus_k,
                    "psi_mean": psi_mean,
                    "psi_var": psi_var,
                }
            )

        fold_df_comp = pd.DataFrame([row for row in fold_rows if row["comparison"] == comp])

        t_fold = float("nan")
        if not fold_df_comp.empty:
            n_k = fold_df_comp["n"].to_numpy(dtype=float)
            mean_k = fold_df_comp["psi_mean"].to_numpy(dtype=float)
            var_k = fold_df_comp["psi_var"].to_numpy(dtype=float)
            finite = np.isfinite(n_k) & np.isfinite(mean_k) & np.isfinite(var_k)
            if np.any(finite):
                denom = float(np.sqrt(np.sum(n_k[finite] * var_k[finite])))
                if denom > 0.0 and np.isfinite(denom):
                    t_fold = float(np.sum(n_k[finite] * mean_k[finite]) / denom)

        t_strict = float("nan")
        if psi_all:
            psi_concat = np.concatenate(psi_all)
            if psi_concat.size > 1 and np.all(np.isfinite(psi_concat)):
                psi_mean_all = float(np.mean(psi_concat))
                psi_var_all = float(np.var(psi_concat, ddof=1))
                denom = float(np.sqrt(psi_var_all / psi_concat.size)) if psi_var_all > 0.0 else float("nan")
                if np.isfinite(denom) and denom > 0.0:
                    t_strict = float(psi_mean_all / denom)

        by_comp_rows.append(
            {
                "comparison": comp,
                "oos_tstat_fold": t_fold,
                "oos_tstat_strict": t_strict,
                "p_value_fold": _two_sided_pvalue_from_t(t_fold),
                "p_value_strict": _two_sided_pvalue_from_t(t_strict),
            }
        )

    by_comp_df = pd.DataFrame(by_comp_rows)
    if by_comp_df.empty:
        by_comp_df = pd.DataFrame(
            {
                "comparison": comparison_labels,
                "oos_tstat_fold": [float("nan")] * j,
                "oos_tstat_strict": [float("nan")] * j,
                "p_value_fold": [float("nan")] * j,
                "p_value_strict": [float("nan")] * j,
            }
        )

    available = bool(np.any(np.isfinite(by_comp_df[["oos_tstat_fold", "oos_tstat_strict"]].to_numpy(dtype=float))))

    fold_table = pd.DataFrame(fold_rows)
    if fold_table.empty:
        fold_table = pd.DataFrame(columns=["comparison", "fold", "n", "theta_minus_k", "psi_mean", "psi_var"])

    return {
        "available": available,
        "by_comparison": by_comp_df,
        "fold_table": fold_table,
    }


def run_score_diagnostics(
    data: MultiCausalData,
    estimate: MultiCausalEstimate,
    *,
    trimming_threshold: Optional[float] = None,
    n_basis_funcs: Optional[int] = None,
    return_summary: bool = True,
) -> Dict[str, Any]:
    """Run score diagnostics for multi-treatment baseline contrasts."""

    if not isinstance(data, MultiCausalData):
        raise TypeError(f"data must be MultiCausalData, got {type(data).__name__}.")
    if not isinstance(estimate, MultiCausalEstimate):
        raise TypeError(
            f"estimate must be MultiCausalEstimate, got {type(estimate).__name__}."
        )

    _validate_estimate_matches_data(data=data, estimate=estimate)

    diag = estimate.diagnostic_data
    if diag is None:
        raise ValueError(
            "Missing estimate.diagnostic_data. "
            "Call estimate(diagnostic_data=True) first."
        )

    m_raw = getattr(diag, "m_hat", None)
    d_raw = getattr(diag, "d", None)
    g_hat_raw = getattr(diag, "g_hat", None)
    if m_raw is None or d_raw is None or g_hat_raw is None:
        raise ValueError("estimate.diagnostic_data must include `m_hat`, `d`, and `g_hat`.")

    score_raw = getattr(diag, "score", estimate.estimand)
    if str(score_raw).upper() != "ATE":
        raise ValueError(f"Only ATE is supported for multi-treatment score diagnostics. Got score={score_raw!r}.")

    y_raw = getattr(diag, "y", None)
    if y_raw is None:
        y_raw = data.get_df()[str(data.outcome)].to_numpy(dtype=float)

    x_raw = getattr(diag, "x", None)
    if x_raw is None:
        x_raw = data.get_df()[list(data.confounders)].to_numpy(dtype=float)

    y = np.asarray(y_raw, dtype=float).reshape(-1)
    d = np.asarray(d_raw, dtype=float)
    g_hat = np.asarray(g_hat_raw, dtype=float)
    m = np.asarray(m_raw, dtype=float)
    x = np.asarray(x_raw, dtype=float)

    if d.ndim != 2 or m.ndim != 2 or g_hat.ndim != 2:
        raise ValueError("`d`, `m_hat`, and `g_hat` must be 2D arrays of shape (n, K).")
    if d.shape != m.shape or d.shape != g_hat.shape:
        raise ValueError("`d`, `m_hat`, and `g_hat` must have matching shape (n, K).")
    if x.ndim != 2:
        raise ValueError("Confounder matrix must be 2D with shape (n, p).")

    n, k = d.shape
    if y.size != n or x.shape[0] != n:
        raise ValueError("All diagnostic arrays must share the same sample size n.")
    if k < 2:
        raise ValueError("Need at least 2 treatment columns for multi-treatment score diagnostics.")

    d = (d > 0.5).astype(float)
    m = _normalize_rows_to_simplex(m)

    comparison_labels = _comparison_labels(_resolve_treatment_names(diag=diag, data=data, k=k))
    j = len(comparison_labels)
    theta = _resolve_theta(estimate.value, n_contrasts=j)

    trimming_thr = _resolve_trimming_threshold(trimming_threshold, diag, estimate)
    normalize_ipw = _resolve_normalize_ipw(diag, estimate)
    m_diag = _trim_multiclass_propensity(m, trimming_thr)

    x_basis = _build_basis(x=x, n_basis_funcs=n_basis_funcs)

    psi_comp, psi_b_comp, _, u = _compute_psi_from_nuisances(
        y=y,
        d=d,
        g_hat=g_hat,
        m=m_diag,
        theta=theta,
        normalize_ipw=normalize_ipw,
    )

    normalize_ipw_for_orthogonality = bool(normalize_ipw)
    if normalize_ipw_for_orthogonality:
        warnings.warn(
            "Orthogonality derivatives are computed with normalize_ipw=False because "
            "m-derivatives for Hajek-normalized IPW are not implemented.",
            RuntimeWarning,
            stacklevel=2,
        )
        normalize_ipw_for_orthogonality = False
    h_ortho = _normalize_ipw_terms(
        d=d,
        m=m_diag,
        normalize_ipw=normalize_ipw_for_orthogonality,
    )

    psi_override = getattr(diag, "psi", None)
    used_estimator_psi = False
    if psi_override is not None:
        psi_override = np.asarray(psi_override, dtype=float)
        if psi_override.ndim == 1 and j == 1:
            psi_override = psi_override.reshape(-1, 1)
        if psi_override.shape == psi_comp.shape:
            psi = psi_override
            used_estimator_psi = True
        else:
            psi = psi_comp
    else:
        psi = psi_comp

    psi_b_override = getattr(diag, "psi_b", None)
    if psi_b_override is not None:
        psi_b_override = np.asarray(psi_b_override, dtype=float)
        if psi_b_override.ndim == 1 and j == 1:
            psi_b_override = psi_b_override.reshape(-1, 1)
        if psi_b_override.shape == psi_b_comp.shape:
            psi_b = psi_b_override
        else:
            psi_b = psi_b_comp
    else:
        psi_b = psi_b_comp

    folds = getattr(diag, "folds", None)
    folds_arr = None if folds is None else np.asarray(folds).reshape(-1)

    finite_rows = (
        np.isfinite(y)
        & np.all(np.isfinite(d), axis=1)
        & np.all(np.isfinite(g_hat), axis=1)
        & np.all(np.isfinite(m_diag), axis=1)
        & np.all(np.isfinite(x_basis), axis=1)
        & np.all(np.isfinite(psi), axis=1)
        & np.all(np.isfinite(psi_b), axis=1)
        & np.all(np.isfinite(h_ortho), axis=1)
    )
    if folds_arr is not None and folds_arr.size == n:
        finite_rows = finite_rows & np.isfinite(folds_arr.astype(float))

    y = y[finite_rows]
    d = d[finite_rows]
    g_hat = g_hat[finite_rows]
    m_diag = m_diag[finite_rows]
    x_basis = x_basis[finite_rows]
    psi = psi[finite_rows]
    psi_b = psi_b[finite_rows]
    h_ortho = h_ortho[finite_rows]
    u = u[finite_rows]
    folds_arr = folds_arr[finite_rows] if folds_arr is not None and folds_arr.size == n else None

    ortho_df, ortho_max = _orthogonality_derivatives(
        x_basis=x_basis,
        d=d,
        m=m_diag,
        h=h_ortho,
        u=u,
        comparison_labels=comparison_labels,
    )

    infl_df, top_influential = _influence_summary(
        psi=psi,
        m=m_diag,
        u=u,
        comparison_labels=comparison_labels,
    )

    oos = _oos_moment_test(psi_b=psi_b, folds=folds_arr, comparison_labels=comparison_labels)
    oos_df = oos["by_comparison"].copy()

    comp_diag = infl_df.merge(ortho_max, on="comparison", how="left")
    comp_diag = comp_diag.merge(oos_df, on="comparison", how="left")

    thresholds = {
        "tail_ratio_warn": 10.0,
        "tail_ratio_strong": 20.0,
        "kurt_warn": 10.0,
        "kurt_strong": 30.0,
        "t_warn": 2.0,
        "t_strong": 4.0,
    }

    def _worst_flag(flags: List[str]) -> str:
        level = {"NA": -1, "GREEN": 0, "YELLOW": 1, "RED": 2}
        inv_level = {v: k for k, v in level.items()}
        return inv_level[max(level.get(flag, -1) for flag in flags)]

    flag_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for row in comp_diag.to_dict(orient="records"):
        comparison = str(row["comparison"])
        flag_tail = _grade(float(row["p99_over_med"]), thresholds["tail_ratio_warn"], thresholds["tail_ratio_strong"])
        flag_kurt = _grade(float(row["kurtosis"]), thresholds["kurt_warn"], thresholds["kurt_strong"])
        flag_ortho = _grade(float(row["max_|t|"]), thresholds["t_warn"], thresholds["t_strong"])

        oos_abs_t = np.nanmax(
            [
                abs(float(row["oos_tstat_fold"])) if np.isfinite(row["oos_tstat_fold"]) else np.nan,
                abs(float(row["oos_tstat_strict"])) if np.isfinite(row["oos_tstat_strict"]) else np.nan,
            ]
        )
        flag_oos = _grade(float(oos_abs_t), thresholds["t_warn"], thresholds["t_strong"])

        overall_flag_comp = _worst_flag([flag_tail, flag_kurt, flag_ortho, flag_oos])

        flag_rows.append(
            {
                "comparison": comparison,
                "psi_tail_ratio": flag_tail,
                "psi_kurtosis": flag_kurt,
                "ortho_max_|t|": flag_ortho,
                "oos_moment": flag_oos,
                "overall_flag": overall_flag_comp,
            }
        )

        summary_rows.extend(
            [
                {"comparison": comparison, "metric": "se_plugin", "value": float(row["se_plugin"]), "flag": "NA"},
                {"comparison": comparison, "metric": "psi_p99_over_med", "value": float(row["p99_over_med"]), "flag": flag_tail},
                {"comparison": comparison, "metric": "psi_kurtosis", "value": float(row["kurtosis"]), "flag": flag_kurt},
                {"comparison": comparison, "metric": "max_|t|_gk", "value": float(row["max_|t|_gk"]), "flag": flag_ortho},
                {"comparison": comparison, "metric": "max_|t|_g0", "value": float(row["max_|t|_g0"]), "flag": flag_ortho},
                {"comparison": comparison, "metric": "max_|t|_mk", "value": float(row["max_|t|_mk"]), "flag": flag_ortho},
                {"comparison": comparison, "metric": "max_|t|_m0", "value": float(row["max_|t|_m0"]), "flag": flag_ortho},
                {"comparison": comparison, "metric": "max_|t|", "value": float(row["max_|t|"]), "flag": flag_ortho},
                {"comparison": comparison, "metric": "oos_tstat_fold", "value": float(row["oos_tstat_fold"]), "flag": flag_oos},
                {"comparison": comparison, "metric": "oos_tstat_strict", "value": float(row["oos_tstat_strict"]), "flag": flag_oos},
            ]
        )

    flags_by_comp = pd.DataFrame(flag_rows)

    if flags_by_comp.empty:
        global_flags = {
            "psi_tail_ratio": "NA",
            "psi_kurtosis": "NA",
            "ortho_max_|t|_gk": "NA",
            "ortho_max_|t|_g0": "NA",
            "ortho_max_|t|_mk": "NA",
            "ortho_max_|t|_m0": "NA",
            "ortho_max_|t|": "NA",
            "oos_moment": "NA",
        }
        overall_flag = "NA"
    else:
        global_flags = {
            "psi_tail_ratio": _worst_flag(flags_by_comp["psi_tail_ratio"].tolist()),
            "psi_kurtosis": _worst_flag(flags_by_comp["psi_kurtosis"].tolist()),
            "ortho_max_|t|": _worst_flag(flags_by_comp["ortho_max_|t|"].tolist()),
            "oos_moment": _worst_flag(flags_by_comp["oos_moment"].tolist()),
            "ortho_max_|t|_gk": _grade(float(np.nanmax(comp_diag["max_|t|_gk"].to_numpy(dtype=float))), thresholds["t_warn"], thresholds["t_strong"]),
            "ortho_max_|t|_g0": _grade(float(np.nanmax(comp_diag["max_|t|_g0"].to_numpy(dtype=float))), thresholds["t_warn"], thresholds["t_strong"]),
            "ortho_max_|t|_mk": _grade(float(np.nanmax(comp_diag["max_|t|_mk"].to_numpy(dtype=float))), thresholds["t_warn"], thresholds["t_strong"]),
            "ortho_max_|t|_m0": _grade(float(np.nanmax(comp_diag["max_|t|_m0"].to_numpy(dtype=float))), thresholds["t_warn"], thresholds["t_strong"]),
        }
        overall_flag = _worst_flag(list(global_flags.values()))

    report: Dict[str, Any] = {
        "params": {
            "score": "ATE",
            "trimming_threshold": float(trimming_thr),
            "normalize_ipw": bool(normalize_ipw),
            "orthogonality_normalize_ipw": bool(normalize_ipw_for_orthogonality),
        },
        "orthogonality_derivatives": ortho_df,
        "orthogonality_max_t": ortho_max,
        "influence_diagnostics": {
            "by_comparison": infl_df,
            "top_influential": top_influential,
        },
        "oos_moment_test": oos,
        "flags": global_flags,
        "flags_by_comparison": flags_by_comp,
        "thresholds": thresholds,
        "overall_flag": overall_flag,
        "meta": {
            "n": int(y.size),
            "K": int(k),
            "comparisons": list(comparison_labels),
            "used_estimator_psi": bool(used_estimator_psi),
            "orthogonality_derivatives_use_score_normalization": bool(
                normalize_ipw_for_orthogonality == normalize_ipw
            ),
        },
    }

    if return_summary:
        summary = pd.DataFrame(
            summary_rows,
            columns=["comparison", "metric", "value", "flag"],
        )
        report["summary"] = summary

    return report


__all__ = ["run_score_diagnostics"]
