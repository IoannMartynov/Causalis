"""Unconfoundedness diagnostics for multi-treatment settings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from causalis.data_contracts.multicausal_estimate import MultiCausalEstimate
from causalis.data_contracts.multicausaldata import MultiCausalData


@dataclass
class _BalanceInputs:
    x: np.ndarray
    d: np.ndarray
    m_hat: np.ndarray
    feature_names: List[str]
    treatment_names: List[str]
    score: str
    normalize: bool


def _grade(value: float, warn: float, strong: float) -> str:
    if value is None or not np.isfinite(value):
        return "NA"
    v = float(value)
    if v < warn:
        return "GREEN"
    if v < strong:
        return "YELLOW"
    return "RED"


def _grade_pass(value: bool) -> str:
    return "GREEN" if bool(value) else "RED"


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

    missing_confounders = [name for name in data.confounders if name not in data.get_df().columns]
    if missing_confounders:
        raise ValueError(
            "data.confounders are missing in data.get_df(): "
            + ", ".join(sorted(map(str, missing_confounders)))
        )


def _resolve_feature_names(diag: Any, data: MultiCausalData, p: int) -> List[str]:
    names = [str(name) for name in list(data.confounders)]
    if len(names) == p:
        return names

    diag_names = getattr(diag, "x_names", None)
    if diag_names is not None:
        diag_names = [str(name) for name in list(diag_names)]
        if len(diag_names) == p:
            return diag_names

    return [f"x{j + 1}" for j in range(p)]


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


def _extract_balance_inputs(
    *,
    data: MultiCausalData,
    estimate: MultiCausalEstimate,
    normalize: Optional[bool],
) -> _BalanceInputs:
    _validate_estimate_matches_data(data=data, estimate=estimate)

    diag = estimate.diagnostic_data
    if diag is None:
        raise ValueError(
            "Missing estimate.diagnostic_data. "
            "Call estimate(diagnostic_data=True) first."
        )

    m_hat_raw = getattr(diag, "m_hat", None)
    d_raw = getattr(diag, "d", None)
    if m_hat_raw is None or d_raw is None:
        raise ValueError("estimate.diagnostic_data must include `m_hat` and `d`.")

    x_raw = getattr(diag, "x", None)
    if x_raw is None:
        if not data.confounders:
            raise ValueError("MultiCausalData must include non-empty confounders.")
        x_raw = data.get_df()[list(data.confounders)].to_numpy(dtype=float)

    score_raw = getattr(diag, "score", estimate.estimand)
    score = str(score_raw).upper()
    if score not in {"ATE", "ATTE"}:
        raise ValueError(
            "Multi-treatment unconfoundedness diagnostics support only ATE or ATTE. "
            f"Got score={score_raw!r}."
        )

    normalize_used = normalize
    if normalize_used is None:
        normalize_used = getattr(diag, "normalize_ipw", None)
    if normalize_used is None:
        normalize_used = bool(estimate.model_options.get("normalize_ipw", False))
    if score == "ATTE":
        normalize_used = False

    x = np.asarray(x_raw, dtype=float)
    d = np.asarray(d_raw, dtype=float)
    m_hat = np.asarray(m_hat_raw, dtype=float)

    if x.ndim != 2:
        raise ValueError("Confounder matrix must be 2D with shape (n, p).")
    if d.ndim != 2 or m_hat.ndim != 2:
        raise ValueError("For multi-treatment diagnostics, `d` and `m_hat` must be 2D (n, K).")

    n, p = x.shape
    if d.shape[0] != n or m_hat.shape[0] != n:
        raise ValueError("Confounders, treatment matrix, and propensity matrix must share row count n.")
    if d.shape[1] != m_hat.shape[1]:
        raise ValueError("`d` and `m_hat` must have the same number of treatment columns.")

    k = d.shape[1]
    if k < 2:
        raise ValueError("Need at least 2 treatment columns for multi-treatment diagnostics.")

    feature_names = _resolve_feature_names(diag=diag, data=data, p=p)
    treatment_names = _resolve_treatment_names(diag=diag, data=data, k=k)

    return _BalanceInputs(
        x=x,
        d=d,
        m_hat=m_hat,
        feature_names=feature_names,
        treatment_names=treatment_names,
        score=score,
        normalize=bool(normalize_used),
    )


def _comparison_label(baseline: str, treatment: str) -> str:
    return f"{baseline} vs {treatment}"


def _smd_from_moments(
    *,
    mu_a: np.ndarray,
    var_a: np.ndarray,
    mu_b: np.ndarray,
    var_b: np.ndarray,
) -> np.ndarray:
    p = mu_a.shape[0]
    s_pool = np.sqrt(0.5 * (np.maximum(var_a, 0.0) + np.maximum(var_b, 0.0)))
    diff = np.abs(mu_a - mu_b)

    out = np.full(p, np.nan, dtype=float)
    zero_both = (var_a <= 1e-16) & (var_b <= 1e-16)
    ok = (~zero_both) & (s_pool > 1e-16)

    out[ok] = diff[ok] / s_pool[ok]
    out[zero_both & (diff <= 1e-16)] = 0.0
    out[zero_both & (diff > 1e-16)] = np.inf
    return out


def _balance_smd(inputs: _BalanceInputs, *, threshold: float) -> Dict[str, Any]:
    x = inputs.x
    d = inputs.d
    m_hat = np.clip(inputs.m_hat, 1e-12, 1.0 - 1e-12)

    _, p = x.shape
    k = d.shape[1]

    comparison_labels: List[str] = []
    smd_weighted_columns: List[np.ndarray] = []
    smd_unweighted_columns: List[np.ndarray] = []

    by_comparison_rows: List[Dict[str, Any]] = []

    baseline_name = str(inputs.treatment_names[0])

    for tr_idx in range(1, k):
        tr_name = str(inputs.treatment_names[tr_idx])
        comp = _comparison_label(baseline_name, tr_name)
        comparison_labels.append(comp)

        if inputs.score == "ATE":
            w_baseline = d[:, 0] / m_hat[:, 0]
            w_treated = d[:, tr_idx] / m_hat[:, tr_idx]
        else:
            w_baseline = d[:, 0] * (m_hat[:, tr_idx] / m_hat[:, 0])
            w_treated = d[:, tr_idx]

        if inputs.normalize:
            mean_baseline = float(np.mean(w_baseline))
            mean_treated = float(np.mean(w_treated))
            w_baseline = w_baseline / (mean_baseline if mean_baseline != 0.0 else 1.0)
            w_treated = w_treated / (mean_treated if mean_treated != 0.0 else 1.0)

        sw_baseline = float(np.sum(w_baseline))
        sw_treated = float(np.sum(w_treated))
        if sw_baseline <= 0.0 or sw_treated <= 0.0:
            raise RuntimeError(f"Degenerate weights in comparison {comp}: zero total mass.")

        mu_baseline = (w_baseline[:, None] * x).sum(axis=0) / sw_baseline
        mu_treated = (w_treated[:, None] * x).sum(axis=0) / sw_treated

        var_baseline = (w_baseline[:, None] * (x - mu_baseline) ** 2).sum(axis=0) / sw_baseline
        var_treated = (w_treated[:, None] * (x - mu_treated) ** 2).sum(axis=0) / sw_treated

        smd_weighted = _smd_from_moments(
            mu_a=mu_treated,
            var_a=var_treated,
            mu_b=mu_baseline,
            var_b=var_baseline,
        )
        smd_weighted_columns.append(smd_weighted)

        mask_baseline = d[:, 0].astype(bool)
        mask_treated = d[:, tr_idx].astype(bool)
        if not np.any(mask_baseline) or not np.any(mask_treated):
            smd_unweighted = np.full(p, np.nan, dtype=float)
        else:
            x_baseline = x[mask_baseline]
            x_treated = x[mask_treated]
            mu_baseline_u = x_baseline.mean(axis=0)
            mu_treated_u = x_treated.mean(axis=0)
            var_baseline_u = x_baseline.var(axis=0, ddof=0)
            var_treated_u = x_treated.var(axis=0, ddof=0)
            smd_unweighted = _smd_from_moments(
                mu_a=mu_treated_u,
                var_a=var_treated_u,
                mu_b=mu_baseline_u,
                var_b=var_baseline_u,
            )

        smd_unweighted_columns.append(smd_unweighted)

        finite = np.isfinite(smd_weighted)
        if np.any(finite):
            frac_viol = float(np.mean(smd_weighted[finite] >= float(threshold)))
            smd_max = float(np.nanmax(smd_weighted[finite]))
            passed = bool((np.all(smd_weighted[finite] < float(threshold))) and (frac_viol < 0.10))
        else:
            frac_viol = 0.0
            smd_max = float("nan")
            passed = True

        flag_max_smd = _grade(smd_max, float(threshold), 2.0 * float(threshold))
        flag_viol = _grade(frac_viol, 0.10, 0.25)
        level = {"NA": -1, "GREEN": 0, "YELLOW": 1, "RED": 2}
        inv_level = {v: k for k, v in level.items()}
        overall_flag = inv_level[max(level.get(flag_max_smd, -1), level.get(flag_viol, -1))]

        by_comparison_rows.append(
            {
                "comparison": comp,
                "smd_max": smd_max,
                "frac_violations": frac_viol,
                "pass": passed,
                "flag_max_smd": flag_max_smd,
                "flag_violations": flag_viol,
                "overall_flag": overall_flag,
            }
        )

    smd_weighted_df = pd.DataFrame(
        np.vstack(smd_weighted_columns).T,
        index=inputs.feature_names,
        columns=comparison_labels,
        dtype=float,
    )
    smd_unweighted_df = pd.DataFrame(
        np.vstack(smd_unweighted_columns).T,
        index=inputs.feature_names,
        columns=comparison_labels,
        dtype=float,
    )

    flat = smd_weighted_df.to_numpy().ravel()
    finite = np.isfinite(flat)
    if np.any(finite):
        frac_violations = float(np.mean(flat[finite] >= float(threshold)))
        smd_max = float(np.nanmax(flat[finite]))
        passed = bool((np.all(flat[finite] < float(threshold))) and (frac_violations < 0.10))
    else:
        frac_violations = 0.0
        smd_max = float("nan")
        passed = True

    worst_features = smd_weighted_df.max(axis=1).sort_values(ascending=False).head(10)

    return {
        "smd": smd_weighted_df,
        "smd_unweighted": smd_unweighted_df,
        "smd_max": smd_max,
        "frac_violations": frac_violations,
        "pass": passed,
        "worst_features": worst_features,
        "comparisons": comparison_labels,
        "by_comparison": pd.DataFrame(by_comparison_rows),
    }


def _build_long_summary(
    *,
    by_comparison: pd.DataFrame,
    overall_smd_max: float,
    overall_frac_violations: float,
    overall_pass: bool,
    threshold: float,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for row in by_comparison.to_dict(orient="records"):
        rows.append(
            {
                "comparison": str(row["comparison"]),
                "metric": "balance_max_smd",
                "value": float(row["smd_max"]),
                "flag": _grade(float(row["smd_max"]), float(threshold), 2.0 * float(threshold)),
            }
        )
        rows.append(
            {
                "comparison": str(row["comparison"]),
                "metric": "balance_frac_violations",
                "value": float(row["frac_violations"]),
                "flag": _grade(float(row["frac_violations"]), 0.10, 0.25),
            }
        )
        rows.append(
            {
                "comparison": str(row["comparison"]),
                "metric": "balance_pass",
                "value": bool(row["pass"]),
                "flag": _grade_pass(bool(row["pass"])),
            }
        )

    rows.append(
        {
            "comparison": "overall",
            "metric": "balance_max_smd",
            "value": float(overall_smd_max),
            "flag": _grade(float(overall_smd_max), float(threshold), 2.0 * float(threshold)),
        }
    )
    rows.append(
        {
            "comparison": "overall",
            "metric": "balance_frac_violations",
            "value": float(overall_frac_violations),
            "flag": _grade(float(overall_frac_violations), 0.10, 0.25),
        }
    )
    rows.append(
        {
            "comparison": "overall",
            "metric": "balance_pass",
            "value": bool(overall_pass),
            "flag": _grade_pass(bool(overall_pass)),
        }
    )

    return pd.DataFrame(rows, columns=["comparison", "metric", "value", "flag"])


def run_unconfoundedness_diagnostics(
    data: MultiCausalData,
    estimate: MultiCausalEstimate,
    *,
    threshold: float = 0.10,
    normalize: Optional[bool] = None,
    return_summary: bool = True,
) -> Dict[str, Any]:
    """Run multi-treatment unconfoundedness diagnostics from data and estimate.

    This implementation computes pairwise balance between baseline treatment 0
    and each active treatment k for either ATE or ATTE estimates.
    """
    if not isinstance(data, MultiCausalData):
        raise TypeError(f"data must be MultiCausalData, got {type(data).__name__}.")
    if not isinstance(estimate, MultiCausalEstimate):
        raise TypeError(
            f"estimate must be MultiCausalEstimate, got {type(estimate).__name__}."
        )

    inputs = _extract_balance_inputs(data=data, estimate=estimate, normalize=normalize)
    balance = _balance_smd(inputs, threshold=threshold)

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
        "balance": balance,
        "flags": flags,
        "overall_flag": overall_flag,
        "meta": {
            "n": int(inputs.x.shape[0]),
            "p": int(inputs.x.shape[1]),
            "K": int(inputs.d.shape[1]),
            "treatment_names": list(inputs.treatment_names),
            "baseline_treatment": inputs.treatment_names[0],
        },
    }

    if return_summary:
        report["summary"] = _build_long_summary(
            by_comparison=balance["by_comparison"],
            overall_smd_max=float(balance["smd_max"]),
            overall_frac_violations=float(balance["frac_violations"]),
            overall_pass=bool(balance["pass"]),
            threshold=float(threshold),
        )

    return report


def validate_unconfoundedness_balance(
    data: MultiCausalData,
    estimate: MultiCausalEstimate,
    *,
    threshold: float = 0.10,
    normalize: Optional[bool] = None,
) -> Dict[str, Any]:
    """Convenience wrapper returning the balance block only."""
    if not isinstance(data, MultiCausalData):
        raise TypeError(f"data must be MultiCausalData, got {type(data).__name__}.")
    if not isinstance(estimate, MultiCausalEstimate):
        raise TypeError(
            f"estimate must be MultiCausalEstimate, got {type(estimate).__name__}."
        )

    report = run_unconfoundedness_diagnostics(
        data=data,
        estimate=estimate,
        threshold=threshold,
        normalize=normalize,
        return_summary=False,
    )
    balance = dict(report["balance"])
    balance.update(
        {
            "score": report["params"]["score"],
            "normalized": report["params"]["normalize"],
            "threshold": report["params"]["smd_threshold"],
            "treatment_names": report["meta"]["treatment_names"],
            "baseline_treatment": report["meta"]["baseline_treatment"],
        }
    )
    return balance


__all__ = ["run_unconfoundedness_diagnostics", "validate_unconfoundedness_balance"]
