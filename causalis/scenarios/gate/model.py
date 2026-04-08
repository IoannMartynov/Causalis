from __future__ import annotations

import warnings
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.utils.validation import check_is_fitted

from causalis.data_contracts.gate_estimate import GateEstimate


_SUPPORTED_COV_TYPES = {"HC0", "HC1", "HC2", "HC3"}
_GATE_GROUPS_REQUIRED_MSG = (
    "GATE requires pre-defined groups. Pass groups=... to estimate() or store gate_groups in CausalData."
)
_GATE_USER_ID_REQUIRED_MSG = (
    "GATE requires CausalData.user_id so subgroup definitions can be aligned to the fit-time observations."
)
_GATE_GROUPS_ALIGNMENT_MSG = (
    "GATE groups must be indexed by the fit-time observation ids, or by the original fit-time row index "
    "when that row-to-observation-id mapping is still unchanged. "
    "Fit the IRM with a stable DataFrame index or CausalData.user_id, and preserve that indexing on groups."
)


def _validate_gate_inputs(
    *,
    irm_model: Any,
    groups: Optional[pd.DataFrame | pd.Series],
    alpha: float,
    cov_type: str,
    cov_kwds: Optional[Dict[str, Any]],
) -> tuple[pd.DataFrame | pd.Series, str]:
    """Validate GATE estimation request and resolve fallback groups source."""
    check_is_fitted(irm_model, attributes=["g0_hat_", "g1_hat_", "m_hat_"])

    if not hasattr(irm_model, "_y") or not hasattr(irm_model, "_d"):
        raise RuntimeError("IRM model must be fitted before GATE estimation.")

    if not (0.0 < float(alpha) < 1.0):
        raise ValueError("alpha must be in (0,1)")

    cov_type_u = str(cov_type).upper()
    if cov_type_u not in _SUPPORTED_COV_TYPES:
        supported = ", ".join(sorted(_SUPPORTED_COV_TYPES))
        raise ValueError(f"cov_type must be one of {{{supported}}}. Got {cov_type!r}.")

    if cov_kwds is not None and not isinstance(cov_kwds, dict):
        raise TypeError("cov_kwds must be a dict or None.")

    groups_resolved = groups
    if groups_resolved is None:
        data_obj = getattr(irm_model, "data", None)
        groups_resolved = getattr(data_obj, "gate_groups", None) if data_obj is not None else None
    if groups_resolved is None:
        raise ValueError(_GATE_GROUPS_REQUIRED_MSG)

    data_obj = getattr(irm_model, "data", None)
    if data_obj is None or not getattr(data_obj, "user_id_name", None):
        raise ValueError(_GATE_USER_ID_REQUIRED_MSG)

    return groups_resolved, cov_type_u


def _is_default_positional_index(index: pd.Index, *, n_obs: int) -> bool:
    """Return whether an index is the default positional 0..n-1 labeling."""
    if len(index) != n_obs:
        return False
    if isinstance(index, pd.RangeIndex):
        return index.start == 0 and index.stop == n_obs and index.step == 1
    if not pd.api.types.is_integer_dtype(index.dtype):
        return False
    index_values = index.to_numpy(copy=False)
    return bool(np.array_equal(index_values, np.arange(n_obs, dtype=index_values.dtype)))


def _resolve_fit_observation_index(
    *,
    irm_model: Any,
    n_obs: int,
) -> pd.Index:
    """Resolve stable fit-time observation ids used to align group labels."""
    fit_index = getattr(irm_model, "_fit_index_", None)
    if fit_index is not None:
        fit_index_resolved = pd.Index(fit_index).copy()
    else:
        data_obj = getattr(irm_model, "data", None)
        if data_obj is None:
            raise RuntimeError(
                "IRM does not expose fit-time observation ids required for GATE group alignment. "
                "Refit the model with the current version."
            )
        user_id_name = getattr(data_obj, "user_id_name", None)
        if user_id_name:
            fit_index_resolved = pd.Index(data_obj.user_id.copy(), name=user_id_name)
        else:
            fit_index_resolved = pd.Index(data_obj.df.index.copy())

    if len(fit_index_resolved) != n_obs:
        raise RuntimeError("Stored fit-time observation ids have inconsistent length; refit the model.")

    if _is_default_positional_index(fit_index_resolved, n_obs=n_obs):
        raise ValueError(
            f"{_GATE_GROUPS_ALIGNMENT_MSG} "
            "The fitted sample currently only has a default positional index (0..n-1), which is ambiguous."
        )

    return fit_index_resolved


def _align_groups_to_fit_observations(
    groups: pd.DataFrame | pd.Series,
    *,
    irm_model: Any,
    n_obs: int,
) -> pd.DataFrame | pd.Series:
    """Align group labels to fit-time observation ids and reject ambiguous positional inputs."""
    if isinstance(groups, pd.Series):
        groups_df = groups.to_frame()
        original_is_series = True
        original_name = groups.name
    elif isinstance(groups, pd.DataFrame):
        groups_df = groups.copy()
        original_is_series = False
        original_name = None
    else:
        raise TypeError("groups must be a pandas Series or DataFrame.")

    if groups_df.shape[0] != n_obs:
        raise ValueError(f"groups must have {n_obs} rows, got {groups_df.shape[0]}.")

    fit_index = _resolve_fit_observation_index(irm_model=irm_model, n_obs=n_obs)

    if not groups_df.index.is_unique:
        raise ValueError("groups index must be unique so rows can be aligned to the fit-time sample.")

    if groups_df.index.equals(fit_index):
        aligned = groups_df
    elif fit_index.is_unique and set(groups_df.index) == set(fit_index):
        aligned = groups_df.reindex(fit_index)
    else:
        aligned = _align_groups_via_fit_row_index(groups_df=groups_df, irm_model=irm_model, fit_index=fit_index)
        if aligned is None:
            raise ValueError(
                f"{_GATE_GROUPS_ALIGNMENT_MSG} "
                "groups.index must either match the fit-time ids (or a permutation of them), "
                "or match the original fit-time row index while the current row-to-id mapping remains unchanged."
            )

    if original_is_series:
        return aligned.iloc[:, 0].rename(original_name)
    return aligned


def _align_groups_via_fit_row_index(
    *,
    groups_df: pd.DataFrame,
    irm_model: Any,
    fit_index: pd.Index,
) -> Optional[pd.DataFrame]:
    """Map row-indexed groups to fit-time observation ids when the row->id mapping is unchanged."""
    fit_row_index = getattr(irm_model, "_fit_row_index_", None)
    data_obj = getattr(irm_model, "data", None)
    user_id_name = getattr(data_obj, "user_id_name", None) if data_obj is not None else None

    if fit_row_index is None or data_obj is None or user_id_name is None:
        return None

    fit_row_index_resolved = pd.Index(fit_row_index).copy()
    if len(fit_row_index_resolved) != len(fit_index):
        raise RuntimeError("Stored fit-time row index has inconsistent length; refit the model.")
    if not fit_row_index_resolved.is_unique:
        return None

    if groups_df.index.equals(fit_row_index_resolved):
        groups_by_row = groups_df
    elif set(groups_df.index) == set(fit_row_index_resolved):
        groups_by_row = groups_df.reindex(fit_row_index_resolved)
    else:
        return None

    current_row_index = pd.Index(data_obj.df.index.copy())
    current_fit_ids = pd.Index(data_obj.user_id.copy(), name=user_id_name)
    if len(current_row_index) != len(fit_index) or len(current_fit_ids) != len(fit_index):
        return None
    if not current_row_index.is_unique:
        return None

    fit_mapping = pd.Series(np.asarray(fit_index, dtype=object), index=fit_row_index_resolved)
    current_mapping = pd.Series(np.asarray(current_fit_ids, dtype=object), index=current_row_index)
    if not current_mapping.reindex(fit_row_index_resolved).equals(fit_mapping):
        return None

    aligned = groups_by_row.copy()
    aligned.index = fit_index
    return aligned


def _coerce_groups_to_basis(
    groups: pd.DataFrame | pd.Series,
    *,
    n_obs: int,
) -> pd.DataFrame:
    """
    Convert user-supplied groups into a full GATE dummy basis.

    For strict GATE, the resulting basis must be mutually exclusive and exhaustive.
    """
    if isinstance(groups, pd.Series):
        groups_df = groups.to_frame()
    elif isinstance(groups, pd.DataFrame):
        groups_df = groups.copy()
    else:
        raise TypeError("groups must be a pandas Series or DataFrame.")

    if groups_df.shape[0] != n_obs:
        raise ValueError(f"groups must have {n_obs} rows, got {groups_df.shape[0]}.")
    if groups_df.shape[1] == 0:
        raise ValueError("groups must contain at least one column.")
    if not groups_df.columns.is_unique:
        raise ValueError("groups columns must be unique.")

    if groups_df.shape[1] == 1:
        group_col = groups_df.iloc[:, 0]
        if group_col.isna().any():
            raise ValueError("groups contains missing values; every observation must belong to exactly one group.")
        prefix = str(group_col.name) if group_col.name is not None else "group"
        basis = pd.get_dummies(group_col, prefix=prefix, prefix_sep="=", dtype=int)
        if basis.shape[1] == 0:
            raise ValueError("Unable to construct a non-empty GATE basis from groups.")
    else:
        basis_numeric = groups_df.copy()
        for col in basis_numeric.columns:
            basis_numeric[col] = pd.to_numeric(basis_numeric[col], errors="coerce")
        if basis_numeric.isna().any().any():
            raise ValueError("Multi-column groups must be binary indicators with values in {0,1}.")
        basis_arr = basis_numeric.to_numpy(dtype=float)
        if not np.isfinite(basis_arr).all():
            raise ValueError("Multi-column groups must contain finite values.")
        if not np.all((basis_arr == 0.0) | (basis_arr == 1.0)):
            raise ValueError("Multi-column groups must be binary indicators with values in {0,1}.")
        basis = basis_numeric.astype(int)

    row_sums = basis.sum(axis=1).to_numpy(dtype=int)
    if not np.all(row_sums == 1):
        raise ValueError(
            "Multi-column groups must be mutually exclusive and exhaustive (each row sum must equal 1). "
            "Overlapping basis columns correspond to generic BLP, not strict GATE."
        )

    col_sums = basis.sum(axis=0).to_numpy(dtype=int)
    if np.any(col_sums == 0):
        empty_cols = list(basis.columns[np.where(col_sums == 0)[0]])
        raise ValueError(f"Group indicator columns without observations are not estimable: {empty_cols}.")

    return basis.astype(float)


def _validate_gate_group_support(
    *,
    basis: pd.DataFrame,
    treatment: np.ndarray,
) -> None:
    """Require within-group treatment variation for strict GATE estimation."""
    invalid_groups: list[str] = []

    for group_name in basis.columns:
        mask = basis[group_name].to_numpy(dtype=bool)
        group_treatment = treatment[mask]
        if group_treatment.size == 0:
            continue
        n_treated = int(np.sum(group_treatment == 1))
        n_control = int(np.sum(group_treatment == 0))
        if n_treated == 0 or n_control == 0:
            invalid_groups.append(str(group_name))

    if invalid_groups:
        raise ValueError(
            "Each GATE group must contain at least one treated and one control observation. "
            f"Invalid groups: {invalid_groups}. "
            "Treatment-defined partitions such as groups=data['d'] are not valid GATE inputs."
        )


def _compute_gate_signal_from_irm(irm_model: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute canonical (unnormalized) cross-fitted orthogonal GATE signal from fitted IRM nuisances."""
    if hasattr(irm_model, "_resolve_estimation_targets"):
        y_raw, d_raw = irm_model._resolve_estimation_targets()
    elif hasattr(irm_model, "_resolve_estimation_sample"):
        _, y_raw, d_raw = irm_model._resolve_estimation_sample()
    else:
        y_raw = getattr(irm_model, "_y", None)
        d_raw = getattr(irm_model, "_d", None)
        if y_raw is None or d_raw is None:
            raise RuntimeError(
                "IRM does not expose estimate-time sample arrays. Refit with matching data available."
            )

    y = np.asarray(y_raw, dtype=float).reshape(-1)
    d = np.asarray(d_raw, dtype=float).reshape(-1)
    g0_hat = np.asarray(irm_model.g0_hat_, dtype=float).reshape(-1)
    g1_hat = np.asarray(irm_model.g1_hat_, dtype=float).reshape(-1)
    m_hat = np.asarray(irm_model.m_hat_, dtype=float).reshape(-1)

    n = y.shape[0]
    if not (d.shape[0] == n == g0_hat.shape[0] == g1_hat.shape[0] == m_hat.shape[0]):
        raise RuntimeError("Stored IRM arrays have inconsistent lengths; refit the model.")

    # Canonical DR signal for subgroup effects uses Horvitz-Thompson IPW terms.
    with np.errstate(divide="ignore", invalid="ignore"):
        h1 = d / m_hat
        h0 = (1.0 - d) / (1.0 - m_hat)
        phi = (g1_hat - g0_hat) + (y - g1_hat) * h1 - (y - g0_hat) * h0
    if not np.all(np.isfinite(phi)):
        raise RuntimeError("Computed GATE orthogonal signal contains non-finite values.")

    return phi, d, m_hat


def _estimate_gate_groupwise_inference(
    *,
    phi: np.ndarray,
    basis: pd.DataFrame,
    cov_type: str,
    alpha: float,
) -> tuple[
    list[str],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
]:
    """Closed-form GATE estimates and HCx variances for disjoint exhaustive dummy basis."""
    group_names = [str(col) for col in basis.columns]
    n_obs = int(phi.shape[0])
    k = len(group_names)

    n_group_counts = basis.sum(axis=0).to_numpy(dtype=int)
    mask_n1 = (n_group_counts == 1)
    if mask_n1.any():
        singleton_names = list(basis.columns[mask_n1])
        if cov_type in {"HC2", "HC3"}:
            warnings.warn(
                f"GATE groups {singleton_names} have only one observation (n=1). "
                f"{cov_type} covariance is undefined for singleton groups; their inference is set to NaN.",
                RuntimeWarning,
            )
        else:
            warnings.warn(
                f"GATE groups {singleton_names} have only one observation (n=1); "
                f"their inference is set to NaN.",
                RuntimeWarning,
            )

    hc1_scale = np.nan
    if cov_type == "HC1":
        denom = n_obs - k
        if denom <= 0:
            warnings.warn(
                "HC1 covariance scaling n/(n-k) is undefined because n <= k; falling back to HC0 scaling.",
                RuntimeWarning,
            )
            hc1_scale = 1.0
        else:
            hc1_scale = float(n_obs / denom)

    values = np.full(k, np.nan, dtype=float)
    variances = np.full(k, np.nan, dtype=float)

    for idx, group_name in enumerate(group_names):
        mask = basis[group_name].to_numpy(dtype=bool)
        n_group = int(np.sum(mask))
        if n_group <= 0:
            continue

        phi_group = phi[mask]
        beta_g = float(np.mean(phi_group))
        values[idx] = beta_g

        if n_group == 1:
            # Variance is not estimable for singleton groups.
            variances[idx] = np.nan
            continue

        residual = phi_group - beta_g
        sse = float(np.sum(residual ** 2))
        n_group_f = float(n_group)
        hc0_var = sse / (n_group_f ** 2)

        if cov_type == "HC0":
            variances[idx] = hc0_var
        elif cov_type == "HC1":
            variances[idx] = hc0_var * hc1_scale
        elif cov_type == "HC2":
            leverage_denom = 1.0 - (1.0 / n_group_f)
            variances[idx] = sse / ((n_group_f ** 2) * leverage_denom)
        elif cov_type == "HC3":
            leverage_denom = 1.0 - (1.0 / n_group_f)
            variances[idx] = sse / ((n_group_f ** 2) * (leverage_denom ** 2))
        else:
            raise ValueError(f"Unsupported cov_type: {cov_type!r}")

    std_errors = np.sqrt(variances)
    with np.errstate(divide="ignore", invalid="ignore"):
        wald_stats = values / std_errors
    p_values = 2.0 * norm.sf(np.abs(wald_stats))
    z_crit = float(norm.ppf(1.0 - (alpha / 2.0)))
    ci_lower = values - z_crit * std_errors
    ci_upper = values + z_crit * std_errors

    covariance = pd.DataFrame(np.diag(variances), index=group_names, columns=group_names)

    return (
        group_names,
        values,
        std_errors,
        wald_stats,
        p_values,
        ci_lower,
        ci_upper,
        covariance,
    )


def estimate_gate_from_irm(
    irm_model: Any,
    groups: Optional[pd.DataFrame | pd.Series],
    alpha: float = 0.05,
    cov_type: str = "HC3",
    cov_kwds: Optional[Dict[str, Any]] = None,
) -> GateEstimate:
    r"""Estimate Group Average Treatment Effects (GATEs) from a fitted IRM.

    Parameters
    ----------
    irm_model : fitted IRM
        Interactive Regression Model with stored cross-fitted nuisance predictions
        :math:`\hat g_0(X)`, :math:`\hat g_1(X)`, and :math:`\hat m(X)`.
        The model must already be fitted.
    groups : Optional[pd.DataFrame or pd.Series]
        Pre-specified subgroup definition.

        Accepted forms are:
        - A single Series / one-column DataFrame of group labels.
        - A multi-column dummy basis with binary indicators.

        For strict GATE, groups must define a mutually exclusive and exhaustive
        partition of the fitted sample. Each group must contain at least one
        treated and one control observation.

        GATE requires the fitted ``CausalData`` to define ``user_id``.
        Group rows are aligned to the IRM fit-time sample using those fit-time
        observation ids. In practice, ``groups.index`` must either exactly match
        those ids or be a permutation of them. As a convenience, when the model
        was fitted with ``user_id``, GATE also accepts groups indexed by the
        original fit-time row index if the current row-to-``user_id`` mapping is
        still unchanged.

        If None, falls back to ``irm_model.data.gate_groups`` when present.
    alpha : float, default 0.05
        Significance level for two-sided confidence intervals.
    cov_type : {"HC0", "HC1", "HC2", "HC3"}, default "HC3"
        Heteroskedasticity-robust covariance estimator used for subgroup
        inference. The implementation uses a closed-form saturated dummy-model
        covariance, equivalent to a no-intercept OLS regression on the GATE basis.
    cov_kwds : Optional[Dict[str, Any]], default None
        Additional covariance options requested by the caller.
        These are currently ignored because GATE uses the closed-form HCx
        covariance formulas implemented in this module.

    Returns
    -------
    GateEstimate
        Result contract containing subgroup effects, standard errors,
        Wald statistics, confidence intervals, covariance matrix, and
        group-level diagnostics. The returned object also supports
        ``contrast(...)`` and ``pairwise_summary(...)`` for formal
        post-estimation group comparisons.

    Examples
    --------
    >>> import pandas as pd
    >>> from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    >>> from causalis.dgp import obs_linear_26_dataset
    >>> from causalis.scenarios.unconfoundedness.model import IRM
    >>> data = obs_linear_26_dataset(
    ...     n=1000,
    ...     seed=3141,
    ...     include_oracle=False,
    ...     return_causal_data=True,
    ... )
    >>> groups = pd.Series(
    ...     (data.df["x1"] >= 0).map({True: "high_x1", False: "low_x1"}),
    ...     index=data.df["user_id"],
    ...     name="segment",
    ... )
    >>> irm = IRM(
    ...     data=data,
    ...     ml_g=RandomForestRegressor(n_estimators=200, max_depth=6, random_state=3141),
    ...     ml_m=RandomForestClassifier(n_estimators=200, max_depth=6, random_state=3141),
    ...     n_folds=3,
    ...     random_state=3141,
    ... )
    >>> irm.fit()
    >>> gate = estimate_gate_from_irm(irm, groups=groups)
    >>> gate.summary()  # doctest: +SKIP

    Notes
    -----
    This implementation targets strict subgroup effects under the same
    unconfoundedness and overlap assumptions used by IRM, with an additional
    requirement that subgroup membership is pre-treatment.

    Let :math:`G` denote a finite partition of the sample space and let
    :math:`\phi(W; \hat\eta)` be the cross-fitted doubly robust signal

    .. math::

        \phi =
        \hat g_1(X) - \hat g_0(X)
        + (Y - \hat g_1(X)) \frac{D}{\hat m(X)}
        - (Y - \hat g_0(X)) \frac{1-D}{1-\hat m(X)}.

    For a subgroup :math:`g`, the target estimand is

    .. math::

        \theta_g = \mathbb{E}[\phi(W; \eta_0) \mid G=g].

    With a mutually exclusive and exhaustive dummy basis, the estimator reduces
    to the groupwise sample mean of the orthogonal score:

    .. math::

        \hat\theta_g = \frac{1}{n_g} \sum_{i : G_i = g} \hat\phi_i.

    Inference is computed from the equivalent saturated no-intercept linear
    regression of :math:`\hat\phi` on the subgroup dummies. Because the design
    is block-diagonal, the covariance matrix is diagonal and each subgroup
    variance is available in closed form under HC0/HC1/HC2/HC3.

    ``normalize_ipw=True`` on the IRM is intentionally ignored for GATE so the
    estimator uses the canonical unnormalized orthogonal signal above.
    """
    groups_resolved, cov_type_u = _validate_gate_inputs(
        irm_model=irm_model,
        groups=groups,
        alpha=alpha,
        cov_type=cov_type,
        cov_kwds=cov_kwds,
    )

    normalize_ipw_requested = bool(irm_model._use_normalized_ipw(score="ATE", warn=False))
    if normalize_ipw_requested:
        warnings.warn(
            "normalize_ipw=True is ignored for GATE to preserve canonical unnormalized orthogonal signals.",
            RuntimeWarning,
        )
    if cov_kwds:
        warnings.warn(
            "cov_kwds are ignored for GATE; closed-form groupwise HCx covariance has no cov_kwds options.",
            RuntimeWarning,
        )

    phi, d, m_hat = _compute_gate_signal_from_irm(irm_model)
    groups_aligned = _align_groups_to_fit_observations(groups_resolved, irm_model=irm_model, n_obs=phi.shape[0])
    basis = _coerce_groups_to_basis(groups_aligned, n_obs=phi.shape[0])
    _validate_gate_group_support(basis=basis, treatment=d)

    (
        group_names,
        values,
        std_errors,
        wald_stats,
        p_values,
        ci_lower,
        ci_upper,
        covariance,
    ) = _estimate_gate_groupwise_inference(
        phi=phi,
        basis=basis,
        cov_type=cov_type_u,
        alpha=alpha,
    )

    n_group_list = []
    n_treated_list = []
    n_control_list = []
    share_treated_list = []
    mean_phi_list = []
    std_phi_list = []
    mean_propensity_list = []
    min_propensity_list = []
    max_propensity_list = []
    group_warning_messages: list[str] = []

    for group_name in group_names:
        mask = basis[group_name].to_numpy(dtype=bool)
        n_group = int(np.sum(mask))
        n_treated = int(np.sum(d[mask] == 1))
        n_control = int(np.sum(d[mask] == 0))
        share_treated = float(n_treated / n_group) if n_group > 0 else np.nan
        mean_phi = float(np.mean(phi[mask])) if n_group > 0 else np.nan
        std_phi = float(np.std(phi[mask], ddof=1)) if n_group > 1 else np.nan
        mean_prop = float(np.mean(m_hat[mask])) if n_group > 0 else np.nan
        min_prop = float(np.min(m_hat[mask])) if n_group > 0 else np.nan
        max_prop = float(np.max(m_hat[mask])) if n_group > 0 else np.nan

        if n_treated == 0 or n_control == 0:
            group_warning_messages.append(
                f"GATE group '{group_name}' has no treated or no control observations; inference may be unstable."
            )
        if n_group < 10:
            group_warning_messages.append(f"GATE group '{group_name}' has small support (n={n_group}).")
        if np.isfinite(min_prop) and np.isfinite(max_prop) and (min_prop < 0.05 or max_prop > 0.95):
            group_warning_messages.append(
                f"GATE group '{group_name}' has extreme propensity support (min={min_prop:.3f}, max={max_prop:.3f})."
            )

        n_group_list.append(n_group)
        n_treated_list.append(n_treated)
        n_control_list.append(n_control)
        share_treated_list.append(share_treated)
        mean_phi_list.append(mean_phi)
        std_phi_list.append(std_phi)
        mean_propensity_list.append(mean_prop)
        min_propensity_list.append(min_prop)
        max_propensity_list.append(max_prop)

    summary_table = pd.DataFrame(
        {
            "value": values,
            "std_error": std_errors,
            "wald_stat": wald_stats,
            "test_stat": wald_stats,
            "p_value": p_values,
            "is_significant": p_values < float(alpha),
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n_group": n_group_list,
            "n_treated": n_treated_list,
            "n_control": n_control_list,
            "share_treated": share_treated_list,
            "mean_phi": mean_phi_list,
            "std_phi": std_phi_list,
            "mean_propensity": mean_propensity_list,
            "min_propensity": min_propensity_list,
            "max_propensity": max_propensity_list,
        },
        index=group_names,
    )
    summary_table.index.name = "group"

    model_options = {
        "cov_type": cov_type_u,
        "cov_kwds": {},
        "cov_kwds_requested": {} if cov_kwds is None else dict(cov_kwds),
        "trimming_threshold": float(getattr(irm_model, "trimming_threshold", np.nan)),
        "normalize_ipw_requested": normalize_ipw_requested,
        "normalize_ipw_effective": False,
        "se_approx_hajek": False,
        "n_folds": int(getattr(irm_model, "n_folds", -1)),
        "random_state": getattr(irm_model, "random_state", None),
    }

    diagnostic_payload: Optional[Dict[str, Any]] = None
    if bool(getattr(irm_model, "store_diagnostics", True)):
        diagnostic_payload = {
            "orthogonal_signal": phi.copy(),
            "basis": basis.copy(),
            "group_diagnostics": summary_table[
                [
                    "n_group",
                    "n_treated",
                    "n_control",
                    "share_treated",
                    "mean_phi",
                    "std_phi",
                    "mean_propensity",
                    "min_propensity",
                    "max_propensity",
                ]
            ].copy(),
            "group_warning_messages": list(group_warning_messages),
        }

    return GateEstimate(
        estimand="GATE",
        model="IRM",
        group_names=group_names,
        values=values,
        std_errors=std_errors,
        test_stats=wald_stats,
        p_values=p_values,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        alpha=float(alpha),
        covariance=covariance,
        summary_table=summary_table,
        model_options=model_options,
        n_group=np.asarray(n_group_list, dtype=int),
        n_treated=np.asarray(n_treated_list, dtype=int),
        n_control=np.asarray(n_control_list, dtype=int),
        share_treated=np.asarray(share_treated_list, dtype=float),
        mean_phi=np.asarray(mean_phi_list, dtype=float),
        std_phi=np.asarray(std_phi_list, dtype=float),
        mean_propensity=np.asarray(mean_propensity_list, dtype=float),
        min_propensity=np.asarray(min_propensity_list, dtype=float),
        max_propensity=np.asarray(max_propensity_list, dtype=float),
        diagnostic_data=diagnostic_payload,
    )
