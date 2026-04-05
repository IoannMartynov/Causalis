"""Overlap diagnostics focused on positivity and propensity calibration."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from causalis.data_contracts.causal_estimate import CausalEstimate
from causalis.dgp.causaldata import CausalData

_DEFAULT_THRESHOLDS: Dict[str, float] = {
    "edge_mass_warn_001": 0.02,
    "edge_mass_strong_001": 0.05,
    "edge_mass_warn_002": 0.05,
    "edge_mass_strong_002": 0.10,
    "ks_warn": 0.30,
    "ks_strong": 0.40,
    "auc_warn": 0.80,
    "auc_strong": 0.90,
    "ipw_relerr_warn": 0.05,
    "ipw_relerr_strong": 0.10,
    "ess_ratio_warn": 0.30,
    "ess_ratio_strong": 0.15,
    "clip_share_warn": 0.02,
    "clip_share_strong": 0.05,
    "tail_vs_med_warn": 10.0,
    "tail_vs_med_strong": 100.0,
    "ece_warn": 0.10,
    "ece_strong": 0.20,
    "slope_warn_lo": 0.8,
    "slope_warn_hi": 1.2,
    "slope_strong_lo": 0.6,
    "slope_strong_hi": 1.4,
    "intercept_warn": 0.2,
    "intercept_strong": 0.4,
}


def _validate_estimate_matches_data(data: CausalData, estimate: CausalEstimate) -> None:
    """Ensure an estimate is aligned with the supplied causal dataset."""
    if str(estimate.treatment) != str(data.treatment_name):
        raise ValueError(
            "estimate.treatment must match data.treatment_name "
            f"({estimate.treatment!r} != {data.treatment_name!r})."
        )

    if str(estimate.outcome) != str(data.outcome_name):
        raise ValueError(
            "estimate.outcome must match data.outcome_name "
            f"({estimate.outcome!r} != {data.outcome_name!r})."
        )


def _mask_finite_pairs(p: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop rows with non-finite values from aligned score/label arrays."""
    p = np.asarray(p, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if p.size == 0 or y.size == 0 or p.size != y.size:
        return p, y
    mask = np.isfinite(p) & np.isfinite(y)
    return p[mask], y[mask]


def _auc_mann_whitney(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute AUC via the Mann-Whitney rank statistic."""
    y = labels.astype(bool)
    pos = scores[y]
    neg = scores[~y]
    n1, n0 = pos.size, neg.size
    if n1 == 0 or n0 == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, scores.size + 1, dtype=float)

    s = scores[order]
    i = 0
    while i < s.size:
        j = i
        while j < s.size and s[j] == s[i]:
            j += 1
        if j - i > 1:
            avg = (i + 1 + j) / 2.0
            ranks[order[i:j]] = avg
        i = j

    r1 = ranks[y].sum()
    auc = (r1 - n1 * (n1 + 1) / 2.0) / (n1 * n0)
    return float(auc)


def _ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    """Compute the two-sample Kolmogorov-Smirnov statistic."""
    a = np.sort(np.asarray(a, dtype=float))
    b = np.sort(np.asarray(b, dtype=float))
    na, nb = a.size, b.size
    if na == 0 or nb == 0:
        return float("nan")

    vals = np.sort(np.unique(np.concatenate([a, b])))
    cdf_a = np.searchsorted(a, vals, side="right") / na
    cdf_b = np.searchsorted(b, vals, side="right") / nb
    return float(np.max(np.abs(cdf_a - cdf_b)))


def _ess(weights: np.ndarray) -> float:
    """Compute effective sample size for a vector of weights."""
    s = float(weights.sum())
    q = float((weights ** 2).sum())
    return float((s * s) / q) if q > 0.0 else float("nan")


def _weight_tail_stats(weights: np.ndarray) -> Dict[str, float]:
    """Summarize extreme-weight behavior using upper quantiles and maxima."""
    values = np.asarray(weights, dtype=float).ravel()
    if values.size == 0:
        return {
            "q95": float("nan"),
            "q99": float("nan"),
            "q999": float("nan"),
            "max": float("nan"),
            "median": float("nan"),
        }

    try:
        q95, q99, q999 = np.quantile(values, [0.95, 0.99, 0.999], method="linear")
    except TypeError:
        q95, q99, q999 = np.quantile(values, [0.95, 0.99, 0.999])

    return {
        "q95": float(q95),
        "q99": float(q99),
        "q999": float(q999),
        "max": float(np.max(values)),
        "median": float(np.median(values)),
    }


def _ece_binary(p: np.ndarray, y: np.ndarray, n_bins: int) -> float:
    """Compute expected calibration error for binary outcomes."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    p = np.clip(p, 0.0, 1.0)

    if p.size == 0 or y.size == 0 or p.size != y.size:
        return float("nan")

    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    sums = np.bincount(bins, weights=p, minlength=n_bins)
    hits = np.bincount(bins, weights=y, minlength=n_bins)
    cnts = np.bincount(bins, minlength=n_bins).astype(float)

    mask = cnts > 0
    if not np.any(mask):
        return float("nan")

    return float(
        np.average(
            np.abs(hits[mask] / cnts[mask] - sums[mask] / cnts[mask]),
            weights=cnts[mask] / cnts[mask].sum(),
        )
    )


def _logit(x: np.ndarray) -> np.ndarray:
    """Apply a numerically stable logit transform to probabilities."""
    clipped = np.clip(x, 1e-12, 1.0 - 1e-12)
    return np.log(clipped / (1.0 - clipped))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Apply a numerically stable logistic transform."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    return out


def _logistic_recalibration(
    p: np.ndarray,
    y: np.ndarray,
    *,
    max_iter: int = 100,
    tol: float = 1e-8,
    ridge: float = 1e-8,
) -> tuple[float, float]:
    """Fit intercept and slope for logistic recalibration of propensity scores."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)

    z = _logit(p)
    z_mean = float(np.mean(z))
    z_std = float(np.std(z))

    pi = float(np.clip(np.mean(y), 1e-6, 1.0 - 1e-6))
    base_intercept = float(np.log(pi / (1.0 - pi)))

    if (not np.isfinite(z_std)) or z_std < 1e-12:
        return base_intercept, 0.0

    z_stdized = (z - z_mean) / z_std
    x = np.column_stack([np.ones_like(z_stdized), z_stdized])

    theta = np.array([base_intercept, 1.0], dtype=float)

    for _ in range(max_iter):
        eta = np.clip(x @ theta, -40.0, 40.0)
        mu = np.clip(_sigmoid(eta), 1e-12, 1.0 - 1e-12)

        r = y - mu
        w = mu * (1.0 - mu)

        x_t_w = x.T * w
        h = x_t_w @ x
        g = x.T @ r

        h[0, 0] += ridge
        h[1, 1] += ridge

        try:
            step = np.linalg.solve(h, g)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(h) @ g

        max_step = np.max(np.abs(step))
        if not np.isfinite(max_step):
            break
        if max_step > 5.0:
            step = step * (5.0 / max_step)

        theta_new = theta + step
        if np.max(np.abs(step)) < tol:
            theta = theta_new
            break

        theta = np.clip(theta_new, -50.0, 50.0)

    beta = float(theta[1] / z_std)
    alpha = float(theta[0] - theta[1] * (z_mean / z_std))

    if not np.isfinite(alpha):
        alpha = base_intercept
    if not np.isfinite(beta):
        beta = 0.0

    return alpha, beta


def _calibration_report(p: np.ndarray, d: np.ndarray, *, n_bins: int) -> Dict[str, Any]:
    """Build calibration metrics, reliability bins, and severity flags."""
    p = np.asarray(p, dtype=float).ravel()
    d = np.asarray(d, dtype=int).ravel()

    p, d_f = _mask_finite_pairs(p, d)
    d = d_f.astype(int)

    if p.size == 0 or d.size == 0 or p.size != d.size:
        raise ValueError("Propensity and treatment arrays must be non-empty and aligned.")

    p = np.clip(p, 1e-12, 1.0 - 1e-12)

    auc = float(_auc_mann_whitney(p, d)) if np.unique(d).size == 2 else float("nan")
    brier = float(np.mean((p - d) ** 2))
    ece = float(_ece_binary(p, d, n_bins=n_bins))

    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    cnts = np.bincount(bins, minlength=n_bins).astype(int)
    sum_p = np.bincount(bins, weights=p, minlength=n_bins)
    sum_y = np.bincount(bins, weights=d, minlength=n_bins)

    with np.errstate(divide="ignore", invalid="ignore"):
        mean_p = np.where(cnts > 0, sum_p / cnts, np.nan)
        frac_pos = np.where(cnts > 0, sum_y / cnts, np.nan)
        abs_err = np.abs(frac_pos - mean_p)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rel_df = pd.DataFrame(
        {
            "bin": np.arange(n_bins),
            "lower": edges[:-1],
            "upper": edges[1:],
            "count": cnts,
            "mean_p": mean_p,
            "frac_pos": frac_pos,
            "abs_error": abs_err,
        }
    )

    alpha, beta = _logistic_recalibration(p, d)

    def _flag_ece(value: float) -> str:
        """Grade expected calibration error against warning thresholds."""
        if np.isnan(value):
            return "NA"
        if value > _DEFAULT_THRESHOLDS["ece_strong"]:
            return "RED"
        if value > _DEFAULT_THRESHOLDS["ece_warn"]:
            return "YELLOW"
        return "GREEN"

    def _flag_slope(value: float) -> str:
        """Grade logistic recalibration slope against warning thresholds."""
        if np.isnan(value):
            return "NA"
        if value < _DEFAULT_THRESHOLDS["slope_strong_lo"] or value > _DEFAULT_THRESHOLDS["slope_strong_hi"]:
            return "RED"
        if value < _DEFAULT_THRESHOLDS["slope_warn_lo"] or value > _DEFAULT_THRESHOLDS["slope_warn_hi"]:
            return "YELLOW"
        return "GREEN"

    def _flag_intercept(value: float) -> str:
        """Grade logistic recalibration intercept against warning thresholds."""
        if np.isnan(value):
            return "NA"
        if abs(value) > _DEFAULT_THRESHOLDS["intercept_strong"]:
            return "RED"
        if abs(value) > _DEFAULT_THRESHOLDS["intercept_warn"]:
            return "YELLOW"
        return "GREEN"

    return {
        "n": int(p.size),
        "n_bins": int(n_bins),
        "auc": auc,
        "brier": brier,
        "ece": ece,
        "reliability_table": rel_df,
        "recalibration": {"intercept": float(alpha), "slope": float(beta)},
        "flags": {
            "ece": _flag_ece(ece),
            "slope": _flag_slope(beta),
            "intercept": _flag_intercept(alpha),
        },
    }


def run_overlap_diagnostics(
    data: CausalData,
    estimate: CausalEstimate,
    *,
    thresholds: Optional[Dict[str, float]] = None,
    n_bins: int = 10,
    use_hajek: Optional[bool] = None,
    return_summary: bool = True,
    auc_flip_margin: float = 0.05,
) -> Dict[str, Any]:
    """Run overlap and calibration diagnostics for an estimated propensity model.

    Parameters
    ----------
    data : CausalData
        Dataset used to fit the estimator.
    estimate : CausalEstimate
        Effect estimate with ``diagnostic_data`` containing propensity-related
        arrays such as ``m_hat`` and ``d``.
    thresholds : dict, optional
        Optional threshold overrides keyed by metric name.
    n_bins : int, default 10
        Number of bins used for calibration summaries.
    use_hajek : bool, optional
        Whether to evaluate normalized IPW identities. If omitted, the value is
        inferred from diagnostic metadata.
    return_summary : bool, default True
        Include a compact tabular summary in the returned payload.
    auc_flip_margin : float, default 0.05
        Margin around 0.5 used when flagging reversed treated/control ranking.

    Returns
    -------
    Dict[str, Any]
        Diagnostic report containing edge-mass, calibration, weight-stability,
        and optional summary tables.

    Raises
    ------
    ValueError
        If required diagnostic arrays are missing or have incompatible shapes.
    """
    _validate_estimate_matches_data(data=data, estimate=estimate)

    threshold_values = dict(_DEFAULT_THRESHOLDS)
    if isinstance(thresholds, dict):
        threshold_values.update({str(k): float(v) for k, v in thresholds.items()})

    diagnostic_data = estimate.diagnostic_data
    if diagnostic_data is None:
        raise ValueError(
            "Missing estimate.diagnostic_data. "
            "Fit IRM with store_diagnostics=True and call estimate() first."
        )

    m_diag = getattr(diagnostic_data, "m_hat", None)
    if m_diag is None:
        raise ValueError("estimate.diagnostic_data must include `m_hat`.")
    m_raw = getattr(diagnostic_data, "m_hat_raw", None)
    if m_raw is None:
        m_raw = m_diag

    d_raw = getattr(diagnostic_data, "d", None)
    if d_raw is None:
        d_raw = data.get_df()[str(data.treatment_name)].to_numpy(dtype=float)

    m = np.asarray(m_raw, dtype=float).ravel()
    m_post = np.asarray(m_diag, dtype=float).ravel()
    d = (np.asarray(d_raw, dtype=float).ravel() > 0.5).astype(int)
    if m.size != d.size or m_post.size != d.size:
        raise ValueError("diagnostic_data.m_hat and treatment vector must have matching length.")

    finite = np.isfinite(m) & np.isfinite(m_post) & np.isfinite(d)
    m = m[finite]
    m_post = m_post[finite]
    d = d[finite]

    if m.size == 0:
        raise ValueError("No finite propensity/treatment pairs available for overlap diagnostics.")

    m = np.clip(m, 1e-12, 1.0 - 1e-12)
    m_post = np.clip(m_post, 1e-12, 1.0 - 1e-12)

    n = int(m.size)
    d_bool = d.astype(bool)
    n_treated = int(np.sum(d_bool))
    p1 = float(n_treated / n)

    if use_hajek is None:
        normalize_ipw = getattr(diagnostic_data, "normalize_ipw", None)
        if normalize_ipw is None:
            normalize_ipw = estimate.model_options.get("normalize_ipw", False)
        use_hajek = bool(normalize_ipw)

    eps1, eps2 = 0.01, 0.02
    edge_mass = {
        "share_below_001": float(np.mean(m < eps1)),
        "share_above_001": float(np.mean(m > 1.0 - eps1)),
        "share_below_002": float(np.mean(m < eps2)),
        "share_above_002": float(np.mean(m > 1.0 - eps2)),
        "min_m": float(np.min(m)),
        "max_m": float(np.max(m)),
    }

    if n_treated == 0 or n_treated == n:
        ks = float("nan")
        auc = float("nan")
    else:
        ks = _ks_statistic(m[d_bool], m[~d_bool])
        auc = _auc_mann_whitney(m, d)

    with np.errstate(divide="ignore", invalid="ignore"):
        w1 = np.where(d_bool, 1.0 / m, 0.0)
        w0 = np.where(~d_bool, 1.0 / (1.0 - m), 0.0)

    if use_hajek:
        if n_treated > 0:
            w1 = w1 / np.mean(w1[d_bool])
        if n_treated < n:
            w0 = w0 / np.mean(w0[~d_bool])

    if n_treated > 0:
        ess_w1 = float(_ess(w1[d_bool]))
        ess_ratio_w1 = float(ess_w1 / max(n_treated, 1))
    else:
        ess_w1 = float("nan")
        ess_ratio_w1 = float("nan")

    if n_treated < n:
        n_control = n - n_treated
        ess_w0 = float(_ess(w0[~d_bool]))
        ess_ratio_w0 = float(ess_w0 / max(n_control, 1))
    else:
        ess_w0 = float("nan")
        ess_ratio_w0 = float("nan")

    ate_ess = {
        "ess_w1": ess_w1,
        "ess_w0": ess_w0,
        "ess_ratio_w1": ess_ratio_w1,
        "ess_ratio_w0": ess_ratio_w0,
    }
    ate_tails = {
        "w1": _weight_tail_stats(w1[d_bool]),
        "w0": _weight_tail_stats(w0[~d_bool]),
    }
    att_rhs = float(n_treated)
    if n_treated > 0:
        with np.errstate(divide="ignore", invalid="ignore"):
            att_lhs = float(np.sum((~d_bool).astype(float) * (m / (1.0 - m))))
        att_rel_err = float(abs(att_lhs - att_rhs) / att_rhs)
    else:
        att_lhs = float(0.0)
        att_rel_err = float("inf")
    att_weights = {
        "lhs_sum": att_lhs,
        "rhs_sum": att_rhs,
        "rel_err": att_rel_err,
    }

    trim_thr = getattr(diagnostic_data, "trimming_threshold", None)
    if trim_thr is None:
        trim_thr = estimate.model_options.get("trimming_threshold", None)

    if trim_thr is not None and np.isfinite(float(trim_thr)) and 0.0 < float(trim_thr) < 0.5:
        trim_thr_f = float(trim_thr)
        clip_lower = float(np.mean(m_post <= trim_thr_f))
        clip_upper = float(np.mean(m_post >= 1.0 - trim_thr_f))
    else:
        clip_lower = float("nan")
        clip_upper = float("nan")

    clipping = {
        "m_clip_lower": clip_lower,
        "m_clip_upper": clip_upper,
    }

    calibration = _calibration_report(m, d, n_bins=int(n_bins))

    flags: Dict[str, str] = {}

    if edge_mass["share_below_001"] > threshold_values["edge_mass_strong_001"] or edge_mass["share_above_001"] > threshold_values["edge_mass_strong_001"]:
        flags["edge_mass_001"] = "RED"
    elif edge_mass["share_below_001"] > threshold_values["edge_mass_warn_001"] or edge_mass["share_above_001"] > threshold_values["edge_mass_warn_001"]:
        flags["edge_mass_001"] = "YELLOW"
    else:
        flags["edge_mass_001"] = "GREEN"

    if edge_mass["share_below_002"] > threshold_values["edge_mass_strong_002"] or edge_mass["share_above_002"] > threshold_values["edge_mass_strong_002"]:
        flags["edge_mass_002"] = "RED"
    elif edge_mass["share_below_002"] > threshold_values["edge_mass_warn_002"] or edge_mass["share_above_002"] > threshold_values["edge_mass_warn_002"]:
        flags["edge_mass_002"] = "YELLOW"
    else:
        flags["edge_mass_002"] = "GREEN"

    if not np.isfinite(ks):
        flags["ks"] = "NA"
    elif ks > threshold_values["ks_strong"]:
        flags["ks"] = "RED"
    elif ks > threshold_values["ks_warn"]:
        flags["ks"] = "YELLOW"
    else:
        flags["ks"] = "GREEN"

    if not np.isfinite(auc):
        flags["auc"] = "NA"
    else:
        sep = max(float(auc), float(1.0 - auc))
        if sep > threshold_values["auc_strong"]:
            flags["auc"] = "RED"
        elif sep > threshold_values["auc_warn"]:
            flags["auc"] = "YELLOW"
        else:
            flags["auc"] = "GREEN"
        if auc < (0.5 - float(auc_flip_margin)):
            flags["auc_flip_suspected"] = "YELLOW"

    if not np.isfinite(ess_ratio_w1):
        flags["ess_w1"] = "NA"
    elif ess_ratio_w1 < threshold_values["ess_ratio_strong"]:
        flags["ess_w1"] = "RED"
    elif ess_ratio_w1 < threshold_values["ess_ratio_warn"]:
        flags["ess_w1"] = "YELLOW"
    else:
        flags["ess_w1"] = "GREEN"

    if not np.isfinite(ess_ratio_w0):
        flags["ess_w0"] = "NA"
    elif ess_ratio_w0 < threshold_values["ess_ratio_strong"]:
        flags["ess_w0"] = "RED"
    elif ess_ratio_w0 < threshold_values["ess_ratio_warn"]:
        flags["ess_w0"] = "YELLOW"
    else:
        flags["ess_w0"] = "GREEN"

    for side in ("w1", "w0"):
        side_tails = ate_tails[side]
        med = float(side_tails.get("median", np.nan))
        if (not np.isfinite(med)) or med == 0.0:
            flags[f"tails_{side}"] = "NA"
            continue
        ratios = []
        for key in ("q95", "q99", "q999", "max"):
            value = float(side_tails.get(key, np.nan))
            if np.isfinite(value):
                ratios.append(value / med)
        if not ratios:
            flags[f"tails_{side}"] = "NA"
        elif any(ratio > threshold_values["tail_vs_med_strong"] for ratio in ratios):
            flags[f"tails_{side}"] = "RED"
        elif any(ratio > threshold_values["tail_vs_med_warn"] for ratio in ratios):
            flags[f"tails_{side}"] = "YELLOW"
        else:
            flags[f"tails_{side}"] = "GREEN"

    if np.isnan(att_rel_err):
        flags["att_identity"] = "NA"
    elif att_rel_err > threshold_values["ipw_relerr_strong"]:
        flags["att_identity"] = "RED"
    elif att_rel_err > threshold_values["ipw_relerr_warn"]:
        flags["att_identity"] = "YELLOW"
    else:
        flags["att_identity"] = "GREEN"

    if not np.isfinite(clip_lower) or not np.isfinite(clip_upper):
        flags["clip_m"] = "NA"
    else:
        clip_total = clip_lower + clip_upper
        if clip_total > threshold_values["clip_share_strong"]:
            flags["clip_m"] = "RED"
        elif clip_total > threshold_values["clip_share_warn"]:
            flags["clip_m"] = "YELLOW"
        else:
            flags["clip_m"] = "GREEN"

    flags["calibration_ece"] = calibration["flags"]["ece"]
    flags["calibration_slope"] = calibration["flags"]["slope"]
    flags["calibration_intercept"] = calibration["flags"]["intercept"]

    report: Dict[str, Any] = {
        "n": n,
        "n_treated": n_treated,
        "p1": p1,
        "edge_mass": edge_mass,
        "ks": float(ks),
        "auc": float(auc),
        "ate_ess": ate_ess,
        "ate_tails": ate_tails,
        "att_weights": att_weights,
        "clipping": clipping,
        "calibration": calibration,
        "flags": flags,
        "meta": {
            "use_hajek": bool(use_hajek),
            "propensity_source": "m_hat_raw" if getattr(diagnostic_data, "m_hat_raw", None) is not None else "m_hat",
            "thresholds": threshold_values,
            "n_bins": int(n_bins),
        },
    }

    if return_summary:
        def _safe_ratio(numerator: float, denominator: float) -> float:
            """Safely divide summary statistics while preserving ``nan`` on failure."""
            if (not np.isfinite(numerator)) or (not np.isfinite(denominator)) or denominator == 0.0:
                return float("nan")
            return float(numerator / denominator)

        clip_total = clip_lower + clip_upper if np.isfinite(clip_lower) and np.isfinite(clip_upper) else np.nan
        report["summary"] = pd.DataFrame(
            [
                {"metric": "edge_0.01_below", "value": edge_mass["share_below_001"], "flag": flags["edge_mass_001"]},
                {"metric": "edge_0.01_above", "value": edge_mass["share_above_001"], "flag": flags["edge_mass_001"]},
                {"metric": "edge_0.02_below", "value": edge_mass["share_below_002"], "flag": flags["edge_mass_002"]},
                {"metric": "edge_0.02_above", "value": edge_mass["share_above_002"], "flag": flags["edge_mass_002"]},
                {"metric": "KS", "value": float(ks), "flag": flags["ks"]},
                {"metric": "AUC", "value": float(auc), "flag": flags["auc"]},
                {"metric": "ESS_treated_ratio", "value": ess_ratio_w1, "flag": flags["ess_w1"]},
                {"metric": "ESS_control_ratio", "value": ess_ratio_w0, "flag": flags["ess_w0"]},
                {
                    "metric": "tails_w1_q99/med",
                    "value": _safe_ratio(float(ate_tails["w1"]["q99"]), float(ate_tails["w1"]["median"])),
                    "flag": flags["tails_w1"],
                },
                {
                    "metric": "tails_w0_q99/med",
                    "value": _safe_ratio(float(ate_tails["w0"]["q99"]), float(ate_tails["w0"]["median"])),
                    "flag": flags["tails_w0"],
                },
                {"metric": "ATT_identity_relerr", "value": att_rel_err, "flag": flags["att_identity"]},
                {"metric": "clip_m_total", "value": clip_total, "flag": flags["clip_m"]},
                {"metric": "calib_ECE", "value": float(calibration["ece"]), "flag": flags["calibration_ece"]},
                {
                    "metric": "calib_slope",
                    "value": float(calibration["recalibration"]["slope"]),
                    "flag": flags["calibration_slope"],
                },
                {
                    "metric": "calib_intercept",
                    "value": float(calibration["recalibration"]["intercept"]),
                    "flag": flags["calibration_intercept"],
                },
            ]
        )

    return report


__all__ = ["run_overlap_diagnostics"]
