"""Overlap diagnostics for multi-treatment unconfoundedness."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from causalis.data_contracts.multicausal_estimate import MultiCausalEstimate
from causalis.data_contracts.multicausaldata import MultiCausalData

_DEFAULT_OVERLAP_THRESHOLDS: Dict[str, float] = {
    "edge_mass_warn_001": 0.02,
    "edge_mass_strong_001": 0.05,
    "ks_warn": 0.30,
    "ks_strong": 0.40,
    "auc_warn": 0.80,
    "auc_strong": 0.90,
    "ess_ratio_warn": 0.30,
    "ess_ratio_strong": 0.15,
    "clip_share_warn": 0.02,
    "clip_share_strong": 0.05,
}


def _grade(value: float, warn: float, strong: float) -> str:
    if value is None or not np.isfinite(value):
        return "NA"
    v = float(value)
    if v < warn:
        return "GREEN"
    if v < strong:
        return "YELLOW"
    return "RED"


def _grade_larger_better(value: float, warn: float, strong: float) -> str:
    if value is None or not np.isfinite(value):
        return "NA"
    v = float(value)
    if v >= warn:
        return "GREEN"
    if v >= strong:
        return "YELLOW"
    return "RED"


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


def _comparison_label(baseline: str, treatment: str) -> str:
    return f"{baseline} vs {treatment}"


def _ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    a = np.sort(np.asarray(a, dtype=float))
    b = np.sort(np.asarray(b, dtype=float))
    if a.size == 0 or b.size == 0:
        return float("nan")

    vals = np.sort(np.unique(np.concatenate([a, b])))
    cdf_a = np.searchsorted(a, vals, side="right") / a.size
    cdf_b = np.searchsorted(b, vals, side="right") / b.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


def _auc_mann_whitney(scores: np.ndarray, labels: np.ndarray) -> float:
    y = np.asarray(labels, dtype=bool)
    s = np.asarray(scores, dtype=float)
    pos = s[y]
    neg = s[~y]
    n_pos, n_neg = pos.size, neg.size
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, s.size + 1, dtype=float)

    sorted_scores = s[order]
    i = 0
    while i < sorted_scores.size:
        j = i
        while j < sorted_scores.size and sorted_scores[j] == sorted_scores[i]:
            j += 1
        if j - i > 1:
            avg_rank = (i + 1 + j) / 2.0
            ranks[order[i:j]] = avg_rank
        i = j

    r_pos = ranks[y].sum()
    return float((r_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _ess(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=float)
    s = float(np.sum(w))
    q = float(np.sum(w ** 2))
    return float((s * s) / q) if q > 0.0 else float("nan")


def _weight_tail_stats(weights: np.ndarray) -> Dict[str, float]:
    values = np.asarray(weights, dtype=float).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"q99": float("nan"), "median": float("nan")}
    try:
        q99 = np.quantile(values, 0.99, method="linear")
    except TypeError:
        q99 = np.quantile(values, 0.99)
    return {"q99": float(q99), "median": float(np.median(values))}


def _build_overlap_summary_long(by_comparison: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for row in by_comparison.to_dict(orient="records"):
        rows.append(
            {
                "comparison": str(row["comparison"]),
                "metric": "edge_0.01_below",
                "value": float(row["edge_0.01_below"]),
                "flag": str(row["flag_edge_001"]),
            }
        )
        rows.append(
            {
                "comparison": str(row["comparison"]),
                "metric": "edge_0.01_above",
                "value": float(row["edge_0.01_above"]),
                "flag": str(row["flag_edge_001"]),
            }
        )
        rows.append(
            {
                "comparison": str(row["comparison"]),
                "metric": "KS",
                "value": float(row["ks"]),
                "flag": str(row["flag_ks"]),
            }
        )
        rows.append(
            {
                "comparison": str(row["comparison"]),
                "metric": "AUC",
                "value": float(row["auc"]),
                "flag": str(row["flag_auc"]),
            }
        )
        rows.append(
            {
                "comparison": str(row["comparison"]),
                "metric": "ESS_treated_ratio",
                "value": float(row["ess_ratio_treated"]),
                "flag": str(row["flag_ess_treated"]),
            }
        )
        rows.append(
            {
                "comparison": str(row["comparison"]),
                "metric": "ESS_baseline_ratio",
                "value": float(row["ess_ratio_baseline"]),
                "flag": str(row["flag_ess_baseline"]),
            }
        )
        rows.append(
            {
                "comparison": str(row["comparison"]),
                "metric": "clip_m_total",
                "value": float(row["clip_m_total"]),
                "flag": str(row["flag_clip_m"]),
            }
        )
        rows.append(
            {
                "comparison": str(row["comparison"]),
                "metric": "overlap_pass",
                "value": bool(row["pass"]),
                "flag": str(row["overall_flag"]),
            }
        )
    return pd.DataFrame(rows, columns=["comparison", "metric", "value", "flag"])


def run_overlap_diagnostics(
    data: MultiCausalData,
    estimate: MultiCausalEstimate,
    *,
    thresholds: Optional[Dict[str, float]] = None,
    use_hajek: Optional[bool] = None,
    return_summary: bool = True,
    auc_flip_margin: float = 0.05,
) -> Dict[str, Any]:
    """Run multi-treatment overlap diagnostics from data and estimate.

    Diagnostics are computed pairwise between baseline treatment 0 and each
    active treatment k using pairwise conditional propensity
    P(D=k | X, D in {0, k}) as the comparison score.
    """
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

    m_diag_raw = getattr(diag, "m_hat", None)
    if m_diag_raw is None:
        raise ValueError("estimate.diagnostic_data must include `m_hat` and `d`.")
    m_raw = getattr(diag, "m_hat_raw", None)
    if m_raw is None:
        m_raw = m_diag_raw
    d_raw = getattr(diag, "d", None)
    if d_raw is None:
        raise ValueError("estimate.diagnostic_data must include `m_hat` and `d`.")

    m = np.asarray(m_raw, dtype=float)
    m_post = np.asarray(m_diag_raw, dtype=float)
    d = np.asarray(d_raw, dtype=float)
    if m.ndim != 2 or m_post.ndim != 2 or d.ndim != 2:
        raise ValueError("For multi-treatment overlap diagnostics, `m_hat` and `d` must be 2D (n, K).")
    if m.shape != d.shape or m_post.shape != d.shape:
        raise ValueError("`m_hat` and `d` must have the same shape (n, K).")

    n, k = m.shape
    if k < 2:
        raise ValueError("Need at least 2 treatment columns for overlap diagnostics.")

    score_raw = getattr(diag, "score", estimate.estimand)
    score = str(score_raw).upper()
    if score not in {"ATE", "ATTE"}:
        raise ValueError(
            "Multi-treatment overlap diagnostics support only ATE or ATTE. "
            f"Got score={score_raw!r}."
        )

    treatment_names = _resolve_treatment_names(diag=diag, data=data, k=k)
    baseline_name = str(treatment_names[0])

    if use_hajek is None:
        norm = getattr(diag, "normalize_ipw", None)
        if norm is None:
            norm = estimate.model_options.get("normalize_ipw", False)
        use_hajek = bool(norm)
    if score == "ATTE":
        use_hajek = False

    threshold_values = dict(_DEFAULT_OVERLAP_THRESHOLDS)
    if isinstance(thresholds, dict):
        threshold_values.update({str(key): float(val) for key, val in thresholds.items()})

    trimming_threshold = getattr(diag, "trimming_threshold", None)
    if trimming_threshold is None:
        trimming_threshold = estimate.model_options.get("trimming_threshold", None)
    if trimming_threshold is not None and np.isfinite(float(trimming_threshold)):
        trimming_threshold = float(trimming_threshold)
    else:
        trimming_threshold = None

    m = np.clip(m, 1e-12, 1.0 - 1e-12)
    m_post = np.clip(m_post, 1e-12, 1.0 - 1e-12)
    d_bool = d > 0.5

    comparison_rows: List[Dict[str, Any]] = []

    for tr_idx in range(1, k):
        tr_name = str(treatment_names[tr_idx])
        comparison = _comparison_label(baseline_name, tr_name)

        mask_baseline = d_bool[:, 0]
        mask_treated = d_bool[:, tr_idx]
        mask_pair = mask_baseline | mask_treated

        if not np.any(mask_baseline) or not np.any(mask_treated):
            raise ValueError(
                f"Both groups must have at least one observation for comparison {comparison}."
            )

        p_treated_all = m[:, tr_idx]
        p_baseline_all = m[:, 0]
        p_treated_pair = p_treated_all[mask_pair]
        p_baseline_pair = p_baseline_all[mask_pair]
        label_treated_pair = mask_treated[mask_pair].astype(int)
        label_baseline_pair = mask_baseline[mask_pair].astype(int)

        denom_pair_all = p_treated_all + p_baseline_all
        pair_score_all = np.clip(
            p_treated_all / np.where(denom_pair_all > 1e-12, denom_pair_all, 1.0),
            1e-12,
            1.0 - 1e-12,
        )
        pair_score_treated = pair_score_all[mask_treated]
        pair_score_baseline = pair_score_all[mask_baseline]
        pair_score_pair = pair_score_all[mask_pair]

        edge_low = float(np.mean(pair_score_pair < 0.01))
        edge_high = float(np.mean(pair_score_pair > 0.99))

        ks = _ks_statistic(pair_score_treated, pair_score_baseline)
        auc = _auc_mann_whitney(
            scores=np.concatenate(
                [pair_score_treated, pair_score_baseline]
            ),
            labels=np.concatenate(
                [
                    np.ones(int(np.sum(mask_treated)), dtype=int),
                    np.zeros(int(np.sum(mask_baseline)), dtype=int),
                ]
            ),
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            w_treated = np.where(label_treated_pair > 0, 1.0 / p_treated_pair, 0.0)
            w_baseline = np.where(label_baseline_pair > 0, 1.0 / p_baseline_pair, 0.0)

        treated_nonzero = w_treated[label_treated_pair > 0]
        baseline_nonzero = w_baseline[label_baseline_pair > 0]

        if use_hajek:
            treated_scale = float(np.mean(w_treated))
            baseline_scale = float(np.mean(w_baseline))
            if treated_nonzero.size > 0 and np.isfinite(treated_scale) and treated_scale != 0.0:
                treated_nonzero = treated_nonzero / treated_scale
            if baseline_nonzero.size > 0 and np.isfinite(baseline_scale) and baseline_scale != 0.0:
                baseline_nonzero = baseline_nonzero / baseline_scale

        n_treated_pair = int(treated_nonzero.size)
        n_baseline_pair = int(baseline_nonzero.size)

        ess_treated = _ess(treated_nonzero)
        ess_baseline = _ess(baseline_nonzero)
        ess_ratio_treated = (
            float(ess_treated / max(n_treated_pair, 1))
            if np.isfinite(ess_treated)
            else float("nan")
        )
        ess_ratio_baseline = (
            float(ess_baseline / max(n_baseline_pair, 1))
            if np.isfinite(ess_baseline)
            else float("nan")
        )

        tails_treated = _weight_tail_stats(treated_nonzero)
        tails_baseline = _weight_tail_stats(baseline_nonzero)

        if trimming_threshold is not None and 0.0 < trimming_threshold < 0.5:
            p_treated_pair_post = m_post[:, tr_idx][mask_pair]
            p_baseline_pair_post = m_post[:, 0][mask_pair]
            clip_t_mask = (
                (p_treated_pair_post <= trimming_threshold)
                | (p_treated_pair_post >= 1.0 - trimming_threshold)
            )
            clip_b_mask = (
                (p_baseline_pair_post <= trimming_threshold)
                | (p_baseline_pair_post >= 1.0 - trimming_threshold)
            )
            clip_t = float(np.mean(clip_t_mask))
            clip_b = float(np.mean(clip_b_mask))
            clip_total = float(np.mean(clip_t_mask | clip_b_mask))
        else:
            clip_t = float("nan")
            clip_b = float("nan")
            clip_total = float("nan")

        if edge_low > threshold_values["edge_mass_strong_001"] or edge_high > threshold_values["edge_mass_strong_001"]:
            flag_edge_001 = "RED"
        elif edge_low > threshold_values["edge_mass_warn_001"] or edge_high > threshold_values["edge_mass_warn_001"]:
            flag_edge_001 = "YELLOW"
        else:
            flag_edge_001 = "GREEN"

        flag_ks = _grade(float(ks), threshold_values["ks_warn"], threshold_values["ks_strong"])

        if np.isfinite(auc):
            separation = max(float(auc), float(1.0 - auc))
            flag_auc = _grade(separation, threshold_values["auc_warn"], threshold_values["auc_strong"])
            if auc < (0.5 - float(auc_flip_margin)) and flag_auc == "GREEN":
                flag_auc = "YELLOW"
        else:
            flag_auc = "NA"

        flag_ess_treated = _grade_larger_better(
            ess_ratio_treated,
            threshold_values["ess_ratio_warn"],
            threshold_values["ess_ratio_strong"],
        )
        flag_ess_baseline = _grade_larger_better(
            ess_ratio_baseline,
            threshold_values["ess_ratio_warn"],
            threshold_values["ess_ratio_strong"],
        )

        if np.isfinite(clip_total):
            flag_clip_m = _grade(
                clip_total,
                threshold_values["clip_share_warn"],
                threshold_values["clip_share_strong"],
            )
        else:
            flag_clip_m = "NA"

        level = {"NA": -1, "GREEN": 0, "YELLOW": 1, "RED": 2}
        inv_level = {value: key for key, value in level.items()}
        overall_flag = inv_level[
            max(
                level.get(flag_edge_001, -1),
                level.get(flag_ks, -1),
                level.get(flag_auc, -1),
                level.get(flag_ess_treated, -1),
                level.get(flag_ess_baseline, -1),
                level.get(flag_clip_m, -1),
            )
        ]

        comparison_rows.append(
            {
                "comparison": comparison,
                "n_pair": int(np.sum(mask_pair)),
                "n_treated": n_treated_pair,
                "n_baseline": n_baseline_pair,
                "edge_0.01_below": edge_low,
                "edge_0.01_above": edge_high,
                "ks": float(ks),
                "auc": float(auc),
                "ess_ratio_treated": ess_ratio_treated,
                "ess_ratio_baseline": ess_ratio_baseline,
                "tails_treated_q99": float(tails_treated["q99"]),
                "tails_treated_median": float(tails_treated["median"]),
                "tails_baseline_q99": float(tails_baseline["q99"]),
                "tails_baseline_median": float(tails_baseline["median"]),
                "clip_m_treated": clip_t,
                "clip_m_baseline": clip_b,
                "clip_m_total": clip_total,
                "flag_edge_001": flag_edge_001,
                "flag_ks": flag_ks,
                "flag_auc": flag_auc,
                "flag_ess_treated": flag_ess_treated,
                "flag_ess_baseline": flag_ess_baseline,
                "flag_clip_m": flag_clip_m,
                "overall_flag": overall_flag,
                "pass": overall_flag in {"GREEN", "NA"},
            }
        )

    by_comparison = pd.DataFrame(comparison_rows)
    level = {"NA": -1, "GREEN": 0, "YELLOW": 1, "RED": 2}
    inv_level = {value: key for key, value in level.items()}
    overall_flag = inv_level[
        max(level.get(flag, -1) for flag in by_comparison["overall_flag"].tolist())
    ]

    report: Dict[str, Any] = {
        "params": {
            "score": score,
            "use_hajek": bool(use_hajek),
            "thresholds": threshold_values,
            "auc_flip_margin": float(auc_flip_margin),
        },
        "overlap": {
            "by_comparison": by_comparison,
            "comparisons": by_comparison["comparison"].tolist(),
            "pass": bool(np.all(by_comparison["pass"].to_numpy(dtype=bool))),
        },
        "overall_flag": overall_flag,
        "meta": {
            "n": int(n),
            "K": int(k),
            "treatment_names": list(treatment_names),
            "baseline_treatment": baseline_name,
            "trimming_threshold": trimming_threshold,
            "propensity_source": "m_hat_raw" if getattr(diag, "m_hat_raw", None) is not None else "m_hat",
        },
    }

    if return_summary:
        report["summary"] = _build_overlap_summary_long(by_comparison=by_comparison)

    return report


__all__ = ["run_overlap_diagnostics"]
