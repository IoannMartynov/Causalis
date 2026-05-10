"""Post-inference diagnostics for binary instrumental-variable estimates."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm

from causalis.data_contracts.causal_diagnostic_data import IVDiagnosticData


_AUC_THRESHOLD = "GREEN <0.60; YELLOW 0.60-0.75; RED >0.75 by max(AUC, 1-AUC)"
_KS_THRESHOLD = "GREEN <=0.30; YELLOW >0.30-0.40; RED >0.40"
_ESS_THRESHOLD = "GREEN >=0.30; YELLOW 0.15-0.30; RED <0.15"
_FIRST_STAGE_THRESHOLD = (
    "GREEN F >= 10; YELLOW 4 <= F < 10; RED F < 4 or near-zero effect"
)
_FINITE_THRESHOLD = "GREEN when finite; RED when unavailable or invalid"


def _auc_mann_whitney(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute binary AUC via the Mann-Whitney rank statistic."""
    scores = np.asarray(scores, dtype=float).ravel()
    labels = np.asarray(labels, dtype=int).ravel().astype(bool)
    pos = scores[labels]
    neg = scores[~labels]
    n1, n0 = pos.size, neg.size
    if n1 == 0 or n0 == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, scores.size + 1, dtype=float)

    sorted_scores = scores[order]
    i = 0
    while i < sorted_scores.size:
        j = i
        while j < sorted_scores.size and sorted_scores[j] == sorted_scores[i]:
            j += 1
        if j - i > 1:
            ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j

    rank_sum_pos = float(ranks[labels].sum())
    return float((rank_sum_pos - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def _ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    """Compute the two-sample Kolmogorov-Smirnov statistic."""
    a = np.sort(np.asarray(a, dtype=float).ravel())
    b = np.sort(np.asarray(b, dtype=float).ravel())
    if a.size == 0 or b.size == 0:
        return float("nan")

    values = np.sort(np.unique(np.concatenate([a, b])))
    cdf_a = np.searchsorted(a, values, side="right") / a.size
    cdf_b = np.searchsorted(b, values, side="right") / b.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


def _ess(weights: np.ndarray) -> float:
    """Compute effective sample size for a vector of weights."""
    weights = np.asarray(weights, dtype=float).ravel()
    weight_sum = float(np.sum(weights))
    weight_sq_sum = float(np.sum(weights**2))
    return float((weight_sum * weight_sum) / weight_sq_sum) if weight_sq_sum > 0 else float("nan")


def _finite_flag(value: Any) -> str:
    """Return a green/red flag for finite scalar metrics."""
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return "RED"
    return "GREEN" if np.isfinite(value_f) else "RED"


def _flag_auc(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    separability = max(float(value), float(1.0 - value))
    if separability > 0.75:
        return "RED"
    if separability >= 0.60:
        return "YELLOW"
    return "GREEN"


def _flag_ks(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    if value > 0.40:
        return "RED"
    if value > 0.30:
        return "YELLOW"
    return "GREEN"


def _flag_ess(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    if value < 0.15:
        return "RED"
    if value < 0.30:
        return "YELLOW"
    return "GREEN"


def _flag_first_stage(
    *,
    f_stat: float,
    effect: float,
    weak_iv_threshold: float,
) -> str:
    if not np.isfinite(f_stat) or not np.isfinite(effect):
        return "RED"
    if abs(float(effect)) < float(weak_iv_threshold):
        return "RED"
    if f_stat >= 10.0:
        return "GREEN"
    if f_stat >= 4.0:
        return "YELLOW"
    return "RED"


def _resolve_diagnostic_data(result: Any) -> IVDiagnosticData:
    """Resolve an IV estimate or IV diagnostic payload."""
    if isinstance(result, IVDiagnosticData):
        return result

    diagnostic_data = getattr(result, "diagnostic_data", None)
    if diagnostic_data is None:
        raise ValueError(
            "Missing result.diagnostic_data. Fit IIVM and call estimate() before "
            "running IV diagnostics."
        )
    if not isinstance(diagnostic_data, IVDiagnosticData):
        raise TypeError("IV diagnostics require IVCausalEstimate.diagnostic_data.")
    return diagnostic_data


def _model_option(result: Any, key: str, default: Any) -> Any:
    options = getattr(result, "model_options", None)
    if isinstance(options, dict):
        return options.get(key, default)
    return default


def _clean_iv_arrays(
    diag: IVDiagnosticData,
    *,
    include_y: bool = False,
    include_x: bool = False,
) -> Tuple[np.ndarray, ...]:
    """Return finite, aligned IV arrays for diagnostics."""
    y = np.asarray(diag.y, dtype=float).ravel()
    d = np.asarray(diag.d, dtype=float).ravel()
    z = np.asarray(diag.z, dtype=float).ravel()
    arrays = [d, z]
    if include_y:
        arrays.insert(0, y)

    x = None
    if include_x:
        if diag.x is None:
            x = np.empty((z.size, 0), dtype=float)
        else:
            x = np.asarray(diag.x, dtype=float)
            if x.ndim == 1:
                x = x.reshape(-1, 1)
        if x.shape[0] != z.size:
            raise ValueError("diagnostic_data.x must align with y/d/z arrays.")

    lengths = {arr.size for arr in arrays}
    if len(lengths) != 1:
        raise ValueError("diagnostic_data y/d/z arrays must have matching length.")

    mask = np.ones(z.size, dtype=bool)
    for arr in arrays:
        mask &= np.isfinite(arr)
    if x is not None and x.size:
        mask &= np.all(np.isfinite(x), axis=1)

    cleaned = [arr[mask] for arr in arrays]
    if include_x:
        cleaned.append(x[mask])
    return tuple(cleaned)


def _row(metric: str, value: Any, flag: str, threshold: str, message: str) -> Dict[str, Any]:
    return {
        "metric": metric,
        "value": value,
        "flag": flag,
        "threshold": threshold,
        "message": message,
    }


def _table(rows: list[Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["metric", "value", "flag", "threshold", "message"])


def compute_instrument_overlap_diagnostics(diag: IVDiagnosticData) -> Dict[str, Any]:
    """Compute instrument propensity/overlap diagnostics from IV diagnostic data."""
    z = np.asarray(diag.z, dtype=float).ravel()
    p_source = "m_hat_raw" if diag.m_hat_raw is not None else "m_hat"
    p = np.asarray(diag.m_hat_raw if diag.m_hat_raw is not None else diag.m_hat, dtype=float).ravel()
    if z.size != p.size:
        raise ValueError("diagnostic_data.z and instrument propensity scores must match length.")

    finite = np.isfinite(z) & np.isfinite(p)
    z = (z[finite] > 0.5).astype(int)
    p = np.clip(p[finite], 1e-12, 1.0 - 1e-12)
    if z.size == 0:
        raise ValueError("No finite instrument propensity pairs are available.")

    z_bool = z.astype(bool)
    n_z1 = int(np.sum(z_bool))
    n_z0 = int(z.size - n_z1)
    if n_z1 == 0 or n_z0 == 0:
        auc = float("nan")
        ks = float("nan")
        ess_ratio_z1 = float("nan")
        ess_ratio_z0 = float("nan")
    else:
        auc = _auc_mann_whitney(p, z)
        ks = _ks_statistic(p[z_bool], p[~z_bool])
        weights_z1 = 1.0 / p[z_bool]
        weights_z0 = 1.0 / (1.0 - p[~z_bool])
        ess_ratio_z1 = float(_ess(weights_z1) / n_z1)
        ess_ratio_z0 = float(_ess(weights_z0) / n_z0)

    finite_ess_ratios = [
        value for value in (ess_ratio_z1, ess_ratio_z0) if np.isfinite(value)
    ]
    if finite_ess_ratios:
        ess_ratio = float(min(finite_ess_ratios))
    else:
        ess_ratio = float("nan")

    return {
        "n": int(z.size),
        "n_z1": n_z1,
        "n_z0": n_z0,
        "instrument_rate": float(np.mean(z)),
        "instrument_auc": float(auc),
        "instrument_auc_separability": float(max(auc, 1.0 - auc)) if np.isfinite(auc) else float("nan"),
        "instrument_propensity_ks": float(ks),
        "instrument_ess_ratio": ess_ratio,
        "instrument_ess_ratio_z1": float(ess_ratio_z1),
        "instrument_ess_ratio_z0": float(ess_ratio_z0),
        "propensity_source": p_source,
        "flags": {
            "instrument_auc": _flag_auc(float(auc)),
            "instrument_propensity_ks": _flag_ks(float(ks)),
            "instrument_ess_ratio": _flag_ess(float(ess_ratio)),
        },
    }


def _fit_ols_hc1(y: np.ndarray, design: np.ndarray) -> Any:
    """Fit OLS with HC1 robust covariance."""
    return sm.OLS(y, design).fit(cov_type="HC1")


def compute_first_stage_diagnostics(
    diag: IVDiagnosticData,
    *,
    weak_iv_threshold: float = 1e-2,
) -> Dict[str, Any]:
    """Compute controlled first-stage diagnostics: D ~ 1 + Z + X."""
    d, z, x = _clean_iv_arrays(diag, include_x=True)
    if d.size == 0:
        raise ValueError("No finite D/Z/X rows are available for first-stage diagnostics.")

    design_full = np.column_stack([np.ones(d.size, dtype=float), z, x])
    design_reduced = np.column_stack([np.ones(d.size, dtype=float), x])
    effect = se = pvalue = f_stat = partial_r2 = float("nan")
    failure = None

    try:
        full = _fit_ols_hc1(d, design_full)
        reduced = _fit_ols_hc1(d, design_reduced)
        effect = float(full.params[1])
        se = float(full.bse[1])
        pvalue = float(full.pvalues[1])
        t_stat = float(full.tvalues[1])
        f_stat = float(t_stat * t_stat) if np.isfinite(t_stat) else float("nan")
        denom = float(1.0 - reduced.rsquared)
        if np.isfinite(denom) and abs(denom) > 1e-15:
            partial_r2 = float((full.rsquared - reduced.rsquared) / denom)
    except Exception as exc:  # pragma: no cover - defensive against degenerate designs
        failure = str(exc)

    orthogonal_first_stage = float(np.mean(np.asarray(diag.phi_d, dtype=float))) if diag.phi_d is not None else float("nan")
    weak_flag = _flag_first_stage(
        f_stat=float(f_stat),
        effect=float(effect),
        weak_iv_threshold=float(weak_iv_threshold),
    )

    return {
        "n": int(d.size),
        "first_stage_effect": float(effect),
        "first_stage_se": float(se),
        "first_stage_pvalue": float(pvalue),
        "first_stage_f": float(f_stat),
        "partial_r2": float(partial_r2),
        "orthogonal_first_stage": orthogonal_first_stage,
        "weak_iv_flag": weak_flag,
        "weak_iv_threshold": float(weak_iv_threshold),
        "failure": failure,
        "flags": {
            "first_stage_effect": weak_flag,
            "first_stage_se": _finite_flag(se),
            "first_stage_pvalue": _finite_flag(pvalue),
            "first_stage_f": weak_flag,
            "partial_r2": _finite_flag(partial_r2),
            "orthogonal_first_stage": (
                "RED"
                if (not np.isfinite(orthogonal_first_stage))
                or abs(orthogonal_first_stage) < float(weak_iv_threshold)
                else "GREEN"
            ),
            "weak_iv_flag": weak_flag,
        },
    }


def compute_reduced_form_diagnostics(
    diag: IVDiagnosticData,
    *,
    late_value: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute simple reduced-form diagnostics: Y ~ 1 + Z."""
    y, _, z = _clean_iv_arrays(diag, include_y=True)
    if y.size == 0:
        raise ValueError("No finite Y/Z rows are available for reduced-form diagnostics.")

    design = np.column_stack([np.ones(y.size, dtype=float), z])
    effect = se = pvalue = float("nan")
    failure = None
    try:
        fit = _fit_ols_hc1(y, design)
        effect = float(fit.params[1])
        se = float(fit.bse[1])
        pvalue = float(fit.pvalues[1])
    except Exception as exc:  # pragma: no cover - defensive against degenerate designs
        failure = str(exc)

    orthogonal_reduced_form = float(np.mean(np.asarray(diag.phi_y, dtype=float))) if diag.phi_y is not None else float("nan")
    orthogonal_first_stage = float(np.mean(np.asarray(diag.phi_d, dtype=float))) if diag.phi_d is not None else float("nan")
    if np.isfinite(orthogonal_reduced_form) and np.isfinite(orthogonal_first_stage) and abs(orthogonal_first_stage) > 1e-15:
        late_ratio_check = float(orthogonal_reduced_form / orthogonal_first_stage)
    else:
        late_ratio_check = float("nan")

    return {
        "n": int(y.size),
        "reduced_form_effect": float(effect),
        "reduced_form_se": float(se),
        "reduced_form_pvalue": float(pvalue),
        "orthogonal_reduced_form": orthogonal_reduced_form,
        "late_ratio_check": late_ratio_check,
        "late_value": float(late_value) if late_value is not None else None,
        "failure": failure,
        "flags": {
            "reduced_form_effect": _finite_flag(effect),
            "reduced_form_se": _finite_flag(se),
            "reduced_form_pvalue": _finite_flag(pvalue),
            "orthogonal_reduced_form": _finite_flag(orthogonal_reduced_form),
            "late_ratio_check": _finite_flag(late_ratio_check),
        },
    }


def _instrument_overlap_payload(result: Any) -> Dict[str, Any]:
    diag = _resolve_diagnostic_data(result)
    if diag.instrument_overlap is None:
        diag.instrument_overlap = compute_instrument_overlap_diagnostics(diag)
    diag.diagnostics["instrument_overlap"] = diag.instrument_overlap
    return diag.instrument_overlap


def _first_stage_payload(result: Any) -> Dict[str, Any]:
    diag = _resolve_diagnostic_data(result)
    if diag.first_stage is None:
        diag.first_stage = compute_first_stage_diagnostics(
            diag,
            weak_iv_threshold=float(_model_option(result, "weak_iv_threshold", 1e-2)),
        )
    diag.diagnostics["first_stage"] = diag.first_stage
    return diag.first_stage


def _reduced_form_payload(result: Any) -> Dict[str, Any]:
    diag = _resolve_diagnostic_data(result)
    if diag.reduced_form is None:
        diag.reduced_form = compute_reduced_form_diagnostics(
            diag,
            late_value=getattr(result, "value", None),
        )
    diag.diagnostics["reduced_form"] = diag.reduced_form
    return diag.reduced_form


def instrument_overlap(result: Any) -> pd.DataFrame:
    """
    Return instrument propensity/overlap diagnostics for an IV result.

    Checks how well the instrument can be predicted from covariates and whether
    there is sufficient overlap in instrument assignment.

    Parameters
    ----------
    result : IIVM or IVCausalEstimate
        The fitted IV model or its estimation result.

    Returns
    -------
    pd.DataFrame
        A table with diagnostic metrics (AUC, KS, ESS ratio).

    Examples
    --------
    >>> from causalis.scenarios.iv.refutation import instrument_overlap
    >>> # Assuming 'result' is obtained from model.estimate()
    >>> instrument_overlap(result)
    """
    payload = _instrument_overlap_payload(result)
    flags = payload["flags"]
    return _table(
        [
            _row(
                "instrument_auc",
                payload["instrument_auc"],
                flags["instrument_auc"],
                _AUC_THRESHOLD,
                "How well covariates predict the instrument; high separability is suspicious.",
            ),
            _row(
                "instrument_propensity_ks",
                payload["instrument_propensity_ks"],
                flags["instrument_propensity_ks"],
                _KS_THRESHOLD,
                "Two-sample KS distance between Z=1 and Z=0 propensity distributions.",
            ),
            _row(
                "instrument_ess_ratio",
                payload["instrument_ess_ratio"],
                flags["instrument_ess_ratio"],
                _ESS_THRESHOLD,
                "Minimum observed-arm IPW effective-sample-size ratio for instrument groups.",
            ),
        ]
    )


def first_stage(result: Any) -> pd.DataFrame:
    """
    Return first-stage diagnostics for an IV result.

    Checks the strength of the relationship between the instrument and the treatment.
    A weak first stage (F-statistic < 10 or similar) can lead to biased and
    unstable IV estimates.

    Parameters
    ----------
    result : IIVM or IVCausalEstimate
        The fitted IV model or its estimation result.

    Returns
    -------
    pd.DataFrame
        A table with first-stage metrics (Effect, F-statistic, Partial R2, etc.).

    Examples
    --------
    >>> from causalis.scenarios.iv.refutation import first_stage
    >>> # Assuming 'result' is obtained from model.estimate()
    >>> first_stage(result)
    """
    payload = _first_stage_payload(result)
    flags = payload["flags"]
    weak_flag = payload["weak_iv_flag"]
    weak_message = "First stage is strong enough for IV inference." if weak_flag == "GREEN" else "First stage is weak or near zero; IV estimates may be unstable."
    return _table(
        [
            _row("first_stage_effect", payload["first_stage_effect"], flags["first_stage_effect"], _FIRST_STAGE_THRESHOLD, "Coefficient on Z in D ~ 1 + Z + X."),
            _row("first_stage_se", payload["first_stage_se"], flags["first_stage_se"], _FINITE_THRESHOLD, "HC1 robust standard error for the Z coefficient."),
            _row("first_stage_pvalue", payload["first_stage_pvalue"], flags["first_stage_pvalue"], _FINITE_THRESHOLD, "HC1 robust p-value for the Z coefficient."),
            _row("first_stage_f", payload["first_stage_f"], flags["first_stage_f"], _FIRST_STAGE_THRESHOLD, "Robust first-stage Wald F for the single instrument."),
            _row("partial_r2", payload["partial_r2"], flags["partial_r2"], _FINITE_THRESHOLD, "Partial R-squared added by Z after controlling for X."),
            _row("orthogonal_first_stage", payload["orthogonal_first_stage"], flags["orthogonal_first_stage"], _FIRST_STAGE_THRESHOLD, "Denominator of the orthogonal LATE score."),
            _row("weak_iv_flag", weak_flag, weak_flag, _FIRST_STAGE_THRESHOLD, weak_message),
        ]
    )


def reduced_form(result: Any) -> pd.DataFrame:
    """
    Return reduced-form sanity diagnostics for an IV result.

    The reduced form is the regression of the outcome on the instrument.
    Under the IV assumptions, the LATE is the ratio of the reduced-form
    effect to the first-stage effect.

    .. math::

        \theta = \frac{\mathbb{E}[Y|Z=1] - \mathbb{E}[Y|Z=0]}{\mathbb{E}[D|Z=1] - \mathbb{E}[D|Z=0]}

    Parameters
    ----------
    result : IIVM or IVCausalEstimate
        The fitted IV model or its estimation result.

    Returns
    -------
    pd.DataFrame
        A table with reduced-form metrics.

    Examples
    --------
    >>> from causalis.scenarios.iv.refutation import reduced_form
    >>> # Assuming 'result' is obtained from model.estimate()
    >>> reduced_form(result)
    """
    payload = _reduced_form_payload(result)
    flags = payload["flags"]
    return _table(
        [
            _row("reduced_form_effect", payload["reduced_form_effect"], flags["reduced_form_effect"], _FINITE_THRESHOLD, "Coefficient on Z in Y ~ 1 + Z."),
            _row("reduced_form_se", payload["reduced_form_se"], flags["reduced_form_se"], _FINITE_THRESHOLD, "HC1 robust standard error for the reduced-form effect."),
            _row("reduced_form_pvalue", payload["reduced_form_pvalue"], flags["reduced_form_pvalue"], _FINITE_THRESHOLD, "HC1 robust p-value; non-significance is not a failure by itself."),
            _row("orthogonal_reduced_form", payload["orthogonal_reduced_form"], flags["orthogonal_reduced_form"], _FINITE_THRESHOLD, "Numerator of the orthogonal LATE score."),
            _row("late_ratio_check", payload["late_ratio_check"], flags["late_ratio_check"], _FINITE_THRESHOLD, "Orthogonal reduced form divided by orthogonal first stage."),
        ]
    )


def instrument_overlap_plot(
    result: Any,
    *,
    bins: Any = "fd",
    ax: Optional[Any] = None,
    figsize: Tuple[float, float] = (8.0, 4.5),
    dpi: int = 150,
    save: Optional[str] = None,
) -> Any:
    """
    Plot instrument propensity distributions by observed instrument group.

    Visualizes the overlap of :math:`\mathbb{P}(Z=1|X)` between the :math:`Z=0` and
    :math:`Z=1` groups. Good overlap is essential for reliable IV estimation.

    Parameters
    ----------
    result : IIVM or IVCausalEstimate
        The fitted IV model or its estimation result.
    bins : str or int, default "fd"
        Binning strategy for histograms.
    ax : matplotlib.axes.Axes, optional
        Pre-existing axes to plot on.
    figsize : tuple, default (8.0, 4.5)
        Figure size.
    dpi : int, default 150
        Resolution.
    save : str, optional
        Path to save the figure.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    import matplotlib.pyplot as plt

    diag = _resolve_diagnostic_data(result)
    z = np.asarray(diag.z, dtype=float).ravel()
    p = np.asarray(diag.m_hat_raw if diag.m_hat_raw is not None else diag.m_hat, dtype=float).ravel()
    if z.size != p.size:
        raise ValueError("diagnostic_data.z and instrument propensity scores must match length.")

    finite = np.isfinite(z) & np.isfinite(p)
    z = (z[finite] > 0.5)
    p = np.clip(p[finite], 0.0, 1.0)
    p_z1 = p[z]
    p_z0 = p[~z]
    if p_z1.size == 0 or p_z0.size == 0:
        raise ValueError("Both Z=1 and Z=0 must be present for instrument overlap plotting.")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure

    ax.hist(p_z1, bins=bins, range=(0.0, 1.0), density=True, alpha=0.45, label=f"Z=1 (n={p_z1.size})", edgecolor="white", linewidth=0.6)
    ax.hist(p_z0, bins=bins, range=(0.0, 1.0), density=True, alpha=0.45, label=f"Z=0 (n={p_z0.size})", edgecolor="white", linewidth=0.6)
    ax.axvline(float(np.mean(p_z1)), color="C0", linewidth=1.6, linestyle="-")
    ax.axvline(float(np.mean(p_z0)), color="C1", linewidth=1.6, linestyle="--")
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Instrument propensity P(Z=1 | X)")
    ax.set_ylabel("Density")
    ax.set_title("Instrument Propensity Overlap")
    ax.legend(frameon=False)
    fig.tight_layout()

    if save is not None:
        fig.savefig(save, dpi=dpi, bbox_inches="tight")
    return fig


__all__ = [
    "compute_first_stage_diagnostics",
    "compute_instrument_overlap_diagnostics",
    "compute_reduced_form_diagnostics",
    "first_stage",
    "instrument_overlap",
    "instrument_overlap_plot",
    "reduced_form",
]
