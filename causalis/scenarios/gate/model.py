"""Utilities for group-level IRM estimands such as GATE and GATET.

The public entry points in this module estimate subgroup effects from a fitted
binary-treatment IRM using pre-defined groups aligned to the fit-time sample.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.utils.validation import check_is_fitted

from causalis.data_contracts.gate_estimate import GateEstimate


_SUPPORTED_COV_TYPES = {"HC0", "HC1", "HC2", "HC3"}
_FIT_INDEX_SOURCE_USER_ID = "user_id"
_FIT_INDEX_SOURCE_ROW_INDEX = "row_index"


@dataclass(frozen=True)
class _GatePartition:
    """Compact representation of a strict subgroup partition."""

    group_names: list[str]
    codes: np.ndarray


def _groups_required_msg(estimand: str) -> str:
    return f"{estimand} requires pre-defined groups. Pass groups=... to estimate() or store gate_groups in CausalData."


def _user_id_required_msg(estimand: str) -> str:
    return f"{estimand} requires CausalData.user_id so subgroup definitions can be aligned to the fit-time observations."


def _groups_alignment_msg(estimand: str) -> str:
    return (
        f"{estimand} groups must be indexed by the fit-time observation ids, or by the original fit-time row index "
        "when that row-to-observation-id mapping is still unchanged. "
        "Fit the IRM with a stable DataFrame index or CausalData.user_id, and preserve that indexing on groups."
    )


def _validate_gate_inputs(
    *,
    irm_model: Any,
    groups: Optional[pd.DataFrame | pd.Series | np.ndarray],
    alpha: float,
    cov_type: str,
    cov_kwds: Optional[Dict[str, Any]],
    estimand: str = "GATE",
) -> tuple[pd.DataFrame | pd.Series | np.ndarray, str]:
    """Validate a subgroup-estimation request and resolve fallback groups."""
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
        raise ValueError(_groups_required_msg(estimand))

    data_obj = getattr(irm_model, "data", None)
    if data_obj is None or not getattr(data_obj, "user_id_name", None):
        raise ValueError(_user_id_required_msg(estimand))

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
    estimand: str = "GATE",
) -> tuple[pd.Index, str]:
    """Resolve stable fit-time observation ids used to align group labels."""
    data_obj = getattr(irm_model, "data", None)
    user_id_name = getattr(data_obj, "user_id_name", None) if data_obj is not None else None
    fit_index = getattr(irm_model, "_fit_index_", None)
    if fit_index is not None:
        fit_index_resolved = pd.Index(fit_index).copy()
        fit_index_source = (
            _FIT_INDEX_SOURCE_USER_ID
            if getattr(irm_model, "_fit_row_index_", None) is not None or fit_index_resolved.name == user_id_name
            else _FIT_INDEX_SOURCE_ROW_INDEX
        )
    else:
        if data_obj is None:
            raise RuntimeError(
                f"IRM does not expose fit-time observation ids required for {estimand} group alignment. "
                "Refit the model with the current version."
            )
        if user_id_name:
            fit_index_resolved = pd.Index(data_obj.user_id.copy(), name=user_id_name)
            fit_index_source = _FIT_INDEX_SOURCE_USER_ID
        else:
            fit_index_resolved = pd.Index(data_obj.df.index.copy())
            fit_index_source = _FIT_INDEX_SOURCE_ROW_INDEX

    if len(fit_index_resolved) != n_obs:
        raise RuntimeError("Stored fit-time observation ids have inconsistent length; refit the model.")

    if fit_index_source == _FIT_INDEX_SOURCE_ROW_INDEX and _is_default_positional_index(fit_index_resolved, n_obs=n_obs):
        raise ValueError(
            f"{_groups_alignment_msg(estimand)} "
            "The fitted sample currently only has a default positional index (0..n-1), which is ambiguous."
        )

    return fit_index_resolved, fit_index_source


def _align_groups_to_fit_observations(
    groups: pd.DataFrame | pd.Series | np.ndarray,
    *,
    irm_model: Any,
    n_obs: int,
    estimand: str = "GATE",
) -> pd.DataFrame | pd.Series:
    """Align group labels to fit-time observation ids and reject ambiguous positional inputs."""
    if isinstance(groups, np.ndarray):
        # When groups is a raw numpy array, we assume it is already aligned with
        # the original fit-time row index. We wrap it in a Series using that index
        # so the downstream alignment logic can correctly map it to the fit-time ids.
        fit_row_index = getattr(irm_model, "_fit_row_index_", None)
        if fit_row_index is None:
            data_obj = getattr(irm_model, "data", None)
            fit_row_index = getattr(data_obj.df, "index", None) if data_obj is not None else None

        if fit_row_index is not None and len(fit_row_index) == n_obs:
            groups = pd.Series(groups, index=fit_row_index)
        else:
            # If we can't find a reliable row index, we wrap it with the fit_index directly
            # assuming the user knows what they are doing (alignment by position).
            fit_index, _ = _resolve_fit_observation_index(irm_model=irm_model, n_obs=n_obs, estimand=estimand)
            groups = pd.Series(groups, index=fit_index)

    if isinstance(groups, pd.Series):
        groups_df = groups.to_frame()
        original_is_series = True
        original_name = groups.name
    elif isinstance(groups, pd.DataFrame):
        groups_df = groups.copy()
        original_is_series = False
        original_name = None
    else:
        raise TypeError("groups must be a pandas Series, DataFrame or numpy array.")

    if groups_df.shape[0] != n_obs:
        raise ValueError(f"groups must have {n_obs} rows, got {groups_df.shape[0]}.")

    fit_index, _ = _resolve_fit_observation_index(irm_model=irm_model, n_obs=n_obs, estimand=estimand)

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
                f"{_groups_alignment_msg(estimand)} "
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


def _coerce_groups_to_partition(
    groups: pd.DataFrame | pd.Series | np.ndarray,
    *,
    n_obs: int,
) -> _GatePartition:
    """Convert user-supplied groups into compact codes for a strict subgroup partition."""
    if isinstance(groups, np.ndarray):
        groups_df = pd.DataFrame(groups)
    elif isinstance(groups, pd.Series):
        groups_df = groups.to_frame()
    elif isinstance(groups, pd.DataFrame):
        groups_df = groups.copy()
    else:
        raise TypeError("groups must be a pandas Series, DataFrame or numpy array.")

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
        if isinstance(group_col.dtype, pd.CategoricalDtype):
            codes = group_col.cat.codes.to_numpy(dtype=int, copy=True)
            group_names = [f"{prefix}={category}" for category in group_col.cat.categories]
            group_counts = np.bincount(codes, minlength=len(group_names)).astype(int, copy=False)
            if np.any(group_counts == 0):
                empty_cols = [group_names[idx] for idx in np.flatnonzero(group_counts == 0)]
                raise ValueError(f"Group indicator columns without observations are not estimable: {empty_cols}.")
        else:
            codes, uniques = pd.factorize(group_col, sort=True)
            codes = np.asarray(codes, dtype=int)
            group_names = [f"{prefix}={value}" for value in uniques.tolist()]
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

        row_sums = basis_arr.sum(axis=1)
        if not np.all(row_sums == 1.0):
            raise ValueError(
                "Multi-column groups must be mutually exclusive and exhaustive (each row sum must equal 1). "
                "Overlapping basis columns correspond to generic BLP, not strict GATE."
            )

        group_names = [str(col) for col in basis_numeric.columns]
        group_counts = basis_arr.sum(axis=0).astype(int, copy=False)
        if np.any(group_counts == 0):
            empty_cols = [group_names[idx] for idx in np.flatnonzero(group_counts == 0)]
            raise ValueError(f"Group indicator columns without observations are not estimable: {empty_cols}.")
        codes = np.argmax(basis_arr, axis=1).astype(int, copy=False)

    return _GatePartition(group_names=list(group_names), codes=np.asarray(codes, dtype=int))


def _coerce_groups_to_basis(
    groups: pd.DataFrame | pd.Series | np.ndarray,
    *,
    n_obs: int,
) -> pd.DataFrame:
    """
    Convert user-supplied groups into a full strict subgroup dummy basis.

    For strict GATE/GATET, the resulting basis must be mutually exclusive and exhaustive.
    """
    if isinstance(groups, np.ndarray):
        groups_df = pd.DataFrame(groups)
    elif isinstance(groups, pd.Series):
        groups_df = groups.to_frame()
    elif isinstance(groups, pd.DataFrame):
        groups_df = groups.copy()
    else:
        raise TypeError("groups must be a pandas Series, DataFrame or numpy array.")

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
    y, d, g0_hat, g1_hat, m_hat = _resolve_irm_signal_inputs(irm_model)
    # Canonical DR signal for subgroup effects uses Horvitz-Thompson IPW terms.
    with np.errstate(divide="ignore", invalid="ignore"):
        h1 = d / m_hat
        h0 = (1.0 - d) / (1.0 - m_hat)
        phi = (g1_hat - g0_hat) + (y - g1_hat) * h1 - (y - g0_hat) * h0
    if not np.all(np.isfinite(phi)):
        raise RuntimeError("Computed GATE orthogonal signal contains non-finite values.")

    return phi, d, m_hat


def _resolve_irm_signal_inputs(irm_model: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Resolve aligned outcome, treatment, and nuisance arrays used by subgroup scores."""
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

    return y, d, g0_hat, g1_hat, m_hat


def _compute_gatet_signal_from_irm(irm_model: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Compute the canonical subgroup-ATT signal used by GATET.

    The signal is

    .. math::

        z =
        D \cdot (Y - \hat g_0(X))
        - \hat m(X) (1-D) \frac{Y - \hat g_0(X)}{1-\hat m(X)}.

    For pre-treatment groups :math:`G`, GATET is then identified by averaging
    :math:`z` over treated units within each subgroup. This is the subgroup ATT
    score, not the ordinary GATE score restricted to treated observations.
    """
    y, d, g0_hat, _, m_hat = _resolve_irm_signal_inputs(irm_model)
    u0 = y - g0_hat
    with np.errstate(divide="ignore", invalid="ignore"):
        z = d * u0 - (m_hat * (1.0 - d) * u0) / (1.0 - m_hat)
    if not np.all(np.isfinite(z)):
        raise RuntimeError("Computed GATET orthogonal signal contains non-finite values.")

    return z, d, m_hat


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


def _estimate_gate_groupwise_summary_from_partition(
    *,
    phi: np.ndarray,
    d: np.ndarray,
    m_hat: np.ndarray,
    partition: _GatePartition,
    cov_type: str,
    alpha: float,
) -> dict[str, Any]:
    """Compute strict GATE estimates and diagnostics from compact group codes."""
    group_names = list(partition.group_names)
    codes = np.asarray(partition.codes, dtype=int).reshape(-1)
    n_obs = int(phi.shape[0])
    k = len(group_names)

    if codes.shape[0] != n_obs:
        raise RuntimeError("Stored group partition has inconsistent length; recompute groups.")
    if k == 0:
        raise RuntimeError("Group partition must contain at least one group.")
    if np.any((codes < 0) | (codes >= k)):
        raise RuntimeError("Group partition contains invalid group codes.")

    n_group = np.bincount(codes, minlength=k).astype(int, copy=False)
    n_treated = np.rint(np.bincount(codes, weights=np.asarray(d, dtype=float), minlength=k)).astype(int)
    n_control = n_group - n_treated

    invalid_mask = (n_treated == 0) | (n_control == 0)
    if invalid_mask.any():
        invalid_groups = [group_names[idx] for idx in np.flatnonzero(invalid_mask)]
        raise ValueError(
            "Each GATE group must contain at least one treated and one control observation. "
            f"Invalid groups: {invalid_groups}. "
            "Treatment-defined partitions such as groups=data['d'] are not valid GATE inputs."
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

    n_group_f = n_group.astype(float, copy=False)
    sum_phi = np.bincount(codes, weights=phi, minlength=k)
    sum_phi2 = np.bincount(codes, weights=np.square(phi), minlength=k)
    values = sum_phi / n_group_f
    sse = sum_phi2 - (np.square(sum_phi) / n_group_f)
    tiny_negative_mask = np.isfinite(sse) & (sse < 0.0) & (np.abs(sse) < 1e-12)
    if tiny_negative_mask.any():
        sse[tiny_negative_mask] = 0.0
    if np.any(np.isfinite(sse) & (sse < 0.0)):
        raise RuntimeError("Computed negative within-group sum of squares; check GATE signal stability.")

    variances = np.full(k, np.nan, dtype=float)
    estimable_mask = n_group > 1
    hc0_var = np.full(k, np.nan, dtype=float)
    hc0_var[estimable_mask] = sse[estimable_mask] / np.square(n_group_f[estimable_mask])

    if cov_type == "HC0":
        variances[estimable_mask] = hc0_var[estimable_mask]
    elif cov_type == "HC1":
        variances[estimable_mask] = hc0_var[estimable_mask] * hc1_scale
    elif cov_type == "HC2":
        leverage = 1.0 - (1.0 / n_group_f[estimable_mask])
        variances[estimable_mask] = hc0_var[estimable_mask] / leverage
    elif cov_type == "HC3":
        leverage = 1.0 - (1.0 / n_group_f[estimable_mask])
        variances[estimable_mask] = hc0_var[estimable_mask] / np.square(leverage)
    else:
        raise ValueError(f"Unsupported cov_type: {cov_type!r}")

    std_errors = np.sqrt(variances)
    with np.errstate(divide="ignore", invalid="ignore"):
        wald_stats = values / std_errors
    p_values = 2.0 * norm.sf(np.abs(wald_stats))
    z_crit = float(norm.ppf(1.0 - (alpha / 2.0)))
    ci_lower = values - z_crit * std_errors
    ci_upper = values + z_crit * std_errors

    share_treated = n_treated / n_group_f
    std_phi = np.full(k, np.nan, dtype=float)
    std_phi[estimable_mask] = np.sqrt(sse[estimable_mask] / (n_group_f[estimable_mask] - 1.0))

    mean_propensity = np.bincount(codes, weights=m_hat, minlength=k) / n_group_f
    min_propensity = np.full(k, np.inf, dtype=float)
    max_propensity = np.full(k, -np.inf, dtype=float)
    np.minimum.at(min_propensity, codes, m_hat)
    np.maximum.at(max_propensity, codes, m_hat)

    covariance = pd.DataFrame(np.diag(variances), index=group_names, columns=group_names)

    return {
        "group_names": group_names,
        "values": values,
        "std_errors": std_errors,
        "wald_stats": wald_stats,
        "p_values": p_values,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "covariance": covariance,
        "n_group": n_group,
        "n_treated": n_treated,
        "n_control": n_control,
        "share_treated": share_treated,
        "mean_phi": values.copy(),
        "std_phi": std_phi,
        "mean_propensity": mean_propensity,
        "min_propensity": min_propensity,
        "max_propensity": max_propensity,
    }


def _estimate_gatet_groupwise_summary_from_partition(
    *,
    z: np.ndarray,
    d: np.ndarray,
    m_hat: np.ndarray,
    partition: _GatePartition,
    cov_type: str,
    alpha: float,
) -> dict[str, Any]:
    """Compute strict GATET estimates and diagnostics from compact group codes."""
    group_names = list(partition.group_names)
    codes = np.asarray(partition.codes, dtype=int).reshape(-1)
    n_obs = int(z.shape[0])
    k = len(group_names)

    if codes.shape[0] != n_obs:
        raise RuntimeError("Stored group partition has inconsistent length; recompute groups.")
    if k == 0:
        raise RuntimeError("Group partition must contain at least one group.")
    if np.any((codes < 0) | (codes >= k)):
        raise RuntimeError("Group partition contains invalid group codes.")

    d_float = np.asarray(d, dtype=float)
    n_group = np.bincount(codes, minlength=k).astype(int, copy=False)
    n_treated = np.rint(np.bincount(codes, weights=d_float, minlength=k)).astype(int)
    n_control = n_group - n_treated

    zero_treated_mask = n_treated == 0
    if zero_treated_mask.any():
        invalid_groups = [group_names[idx] for idx in np.flatnonzero(zero_treated_mask)]
        raise ValueError(
            "Each GATET group must contain at least one treated observation. "
            f"Invalid groups: {invalid_groups}. "
            "GATET targets treated-within-group effects, so zero-treated groups are not estimable."
        )

    zero_control_mask = n_control == 0
    group_warning_messages: list[str] = []
    if zero_control_mask.any():
        treated_only_groups = [group_names[idx] for idx in np.flatnonzero(zero_control_mask)]
        warning_msg = (
            "GATET groups "
            f"{treated_only_groups} have no control observations. "
            "Estimation remains defined because controls are borrowed through g0_hat and m_hat, "
            "but within-group overlap is degenerate."
        )
        warnings.warn(warning_msg, RuntimeWarning)
        group_warning_messages.append(warning_msg)

    singleton_mask = n_group == 1
    if singleton_mask.any():
        singleton_names = [group_names[idx] for idx in np.flatnonzero(singleton_mask)]
        if cov_type in {"HC2", "HC3"}:
            warnings.warn(
                f"GATET groups {singleton_names} have only one observation (n=1). "
                f"{cov_type} covariance is undefined for singleton groups; their inference is set to NaN.",
                RuntimeWarning,
            )
        else:
            warnings.warn(
                f"GATET groups {singleton_names} have only one observation (n=1); "
                "their inference is set to NaN.",
                RuntimeWarning,
            )

    one_treated_mask = n_treated == 1
    if cov_type in {"HC2", "HC3"} and one_treated_mask.any():
        one_treated_names = [group_names[idx] for idx in np.flatnonzero(one_treated_mask)]
        warnings.warn(
            f"GATET groups {one_treated_names} have only one treated observation. "
            f"{cov_type} covariance is undefined because treated leverage is 1; "
            "their inference is set to NaN.",
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

    n_group_f = n_group.astype(float, copy=False)
    n_treated_f = n_treated.astype(float, copy=False)
    share_treated = n_treated_f / n_group_f

    sum_z = np.bincount(codes, weights=z, minlength=k)
    values = sum_z / n_treated_f

    transformed_signal = z / share_treated[codes]
    sum_phi = np.bincount(codes, weights=transformed_signal, minlength=k)
    sum_phi2 = np.bincount(codes, weights=np.square(transformed_signal), minlength=k)
    sse_phi = sum_phi2 - (np.square(sum_phi) / n_group_f)
    tiny_negative_mask = np.isfinite(sse_phi) & (sse_phi < 0.0) & (np.abs(sse_phi) < 1e-12)
    if tiny_negative_mask.any():
        sse_phi[tiny_negative_mask] = 0.0
    if np.any(np.isfinite(sse_phi) & (sse_phi < 0.0)):
        raise RuntimeError("Computed negative within-group sum of squares; check GATET signal stability.")

    group_coef = values[codes]
    residual = z - (d_float * group_coef)
    u = residual
    sum_u2 = np.bincount(codes, weights=np.square(u), minlength=k)

    variances = np.full(k, np.nan, dtype=float)
    estimable_mask = n_group > 1
    hc0_var = np.full(k, np.nan, dtype=float)
    hc0_var[estimable_mask] = sum_u2[estimable_mask] / np.square(n_treated_f[estimable_mask])

    if cov_type == "HC0":
        variances[estimable_mask] = hc0_var[estimable_mask]
    elif cov_type == "HC1":
        variances[estimable_mask] = hc0_var[estimable_mask] * hc1_scale
    elif cov_type == "HC2":
        hcx_mask = estimable_mask & (n_treated > 1)
        obs_mask = hcx_mask[codes]
        adjusted_u2 = np.zeros(n_obs, dtype=float)
        leverage_denom = 1.0 - (d_float[obs_mask] / n_treated_f[codes[obs_mask]])
        adjusted_u2[obs_mask] = np.square(u[obs_mask]) / leverage_denom
        sum_adjusted_u2 = np.bincount(codes, weights=adjusted_u2, minlength=k)
        variances[hcx_mask] = sum_adjusted_u2[hcx_mask] / np.square(n_treated_f[hcx_mask])
    elif cov_type == "HC3":
        hcx_mask = estimable_mask & (n_treated > 1)
        obs_mask = hcx_mask[codes]
        adjusted_u2 = np.zeros(n_obs, dtype=float)
        leverage_denom = 1.0 - (d_float[obs_mask] / n_treated_f[codes[obs_mask]])
        adjusted_u2[obs_mask] = np.square(u[obs_mask]) / np.square(leverage_denom)
        sum_adjusted_u2 = np.bincount(codes, weights=adjusted_u2, minlength=k)
        variances[hcx_mask] = sum_adjusted_u2[hcx_mask] / np.square(n_treated_f[hcx_mask])
    else:
        raise ValueError(f"Unsupported cov_type: {cov_type!r}")

    std_errors = np.sqrt(variances)
    with np.errstate(divide="ignore", invalid="ignore"):
        wald_stats = values / std_errors
    p_values = 2.0 * norm.sf(np.abs(wald_stats))
    z_crit = float(norm.ppf(1.0 - (alpha / 2.0)))
    ci_lower = values - z_crit * std_errors
    ci_upper = values + z_crit * std_errors

    std_phi = np.full(k, np.nan, dtype=float)
    std_phi[estimable_mask] = np.sqrt(sse_phi[estimable_mask] / (n_group_f[estimable_mask] - 1.0))

    mean_propensity = np.bincount(codes, weights=m_hat, minlength=k) / n_group_f
    min_propensity = np.full(k, np.inf, dtype=float)
    max_propensity = np.full(k, -np.inf, dtype=float)
    np.minimum.at(min_propensity, codes, m_hat)
    np.maximum.at(max_propensity, codes, m_hat)

    covariance = pd.DataFrame(np.diag(variances), index=group_names, columns=group_names)

    return {
        "group_names": group_names,
        "values": values,
        "std_errors": std_errors,
        "wald_stats": wald_stats,
        "p_values": p_values,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "covariance": covariance,
        "n_group": n_group,
        "n_treated": n_treated,
        "n_control": n_control,
        "share_treated": share_treated,
        "mean_phi": sum_phi / n_group_f,
        "std_phi": std_phi,
        "mean_propensity": mean_propensity,
        "min_propensity": min_propensity,
        "max_propensity": max_propensity,
        "group_warning_messages": group_warning_messages,
        "orthogonal_signal": transformed_signal,
        "raw_signal": z,
    }


def _build_gate_estimate(
    *,
    estimand: str,
    irm_model: Any,
    alpha: float,
    cov_type_u: str,
    cov_kwds: Optional[Dict[str, Any]],
    partition: _GatePartition,
    stats: dict[str, Any],
    orthogonal_signal: np.ndarray,
    diagnostic_extra: Optional[Dict[str, Any]] = None,
) -> GateEstimate:
    """Build a GateEstimate result payload shared by GATE and GATET."""
    group_names = stats["group_names"]
    values = stats["values"]
    std_errors = stats["std_errors"]
    wald_stats = stats["wald_stats"]
    p_values = stats["p_values"]
    ci_lower = stats["ci_lower"]
    ci_upper = stats["ci_upper"]
    covariance = stats["covariance"]
    n_group = stats["n_group"]
    n_treated = stats["n_treated"]
    n_control = stats["n_control"]
    share_treated = stats["share_treated"]
    mean_phi = stats["mean_phi"]
    std_phi = stats["std_phi"]
    mean_propensity = stats["mean_propensity"]
    min_propensity = stats["min_propensity"]
    max_propensity = stats["max_propensity"]

    group_warning_messages = list(stats.get("group_warning_messages", []))
    for idx, group_name in enumerate(group_names):
        if n_group[idx] < 10:
            group_warning_messages.append(f"{estimand} group '{group_name}' has small support (n={int(n_group[idx])}).")
        if np.isfinite(min_propensity[idx]) and np.isfinite(max_propensity[idx]) and (
            min_propensity[idx] < 0.05 or max_propensity[idx] > 0.95
        ):
            group_warning_messages.append(
                f"{estimand} group '{group_name}' has extreme propensity support "
                f"(min={min_propensity[idx]:.3f}, max={max_propensity[idx]:.3f})."
            )

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
            "n_group": n_group,
            "n_treated": n_treated,
            "n_control": n_control,
            "share_treated": share_treated,
            "mean_phi": mean_phi,
            "std_phi": std_phi,
            "mean_propensity": mean_propensity,
            "min_propensity": min_propensity,
            "max_propensity": max_propensity,
        },
        index=group_names,
    )
    summary_table.index.name = "group"

    normalize_ipw_requested = bool(getattr(irm_model, "normalize_ipw", False))
    model_options = {
        "cov_type": cov_type_u,
        "covariance_structure": "diagonal",
        "cov_kwds": {},
        "cov_kwds_requested": {} if cov_kwds is None else dict(cov_kwds),
        "trimming_threshold": float(getattr(irm_model, "trimming_threshold", np.nan)),
        "normalize_ipw_requested": normalize_ipw_requested,
        "normalize_ipw_effective": False,
        "se_approx_hajek": False,
        "group_representation": "compact_codes",
        "n_folds": int(getattr(irm_model, "n_folds", -1)),
        "random_state": getattr(irm_model, "random_state", None),
    }

    diagnostic_payload: Optional[Dict[str, Any]] = None
    if bool(getattr(irm_model, "store_diagnostics", True)):
        diagnostic_payload = {
            "orthogonal_signal": np.asarray(orthogonal_signal, dtype=float).copy(),
            "group_codes": partition.codes.copy(),
            "group_names": list(group_names),
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
        if diagnostic_extra:
            for key, value in diagnostic_extra.items():
                if isinstance(value, np.ndarray):
                    diagnostic_payload[key] = value.copy()
                else:
                    diagnostic_payload[key] = value

    return GateEstimate(
        estimand=estimand,
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
        n_group=np.asarray(n_group, dtype=int),
        n_treated=np.asarray(n_treated, dtype=int),
        n_control=np.asarray(n_control, dtype=int),
        share_treated=np.asarray(share_treated, dtype=float),
        mean_phi=np.asarray(mean_phi, dtype=float),
        std_phi=np.asarray(std_phi, dtype=float),
        mean_propensity=np.asarray(mean_propensity, dtype=float),
        min_propensity=np.asarray(min_propensity, dtype=float),
        max_propensity=np.asarray(max_propensity, dtype=float),
        diagnostic_data=diagnostic_payload,
    )


def estimate_gate_from_irm(
    irm_model: Any,
    groups: Optional[pd.DataFrame | pd.Series | np.ndarray],
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
    groups : Optional[pd.DataFrame, pd.Series, or np.ndarray]
        Pre-specified subgroup definition.

        Accepted forms are:
        - A single Series / one-column DataFrame of group labels.
        - A multi-column dummy basis with binary indicators.
        - A numpy array of group labels (assumed aligned with fit-time rows).

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
        estimand="GATE",
    )

    normalize_ipw_requested = bool(getattr(irm_model, "normalize_ipw", False))
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
    groups_aligned = _align_groups_to_fit_observations(
        groups_resolved,
        irm_model=irm_model,
        n_obs=phi.shape[0],
        estimand="GATE",
    )
    partition = _coerce_groups_to_partition(groups_aligned, n_obs=phi.shape[0])
    stats = _estimate_gate_groupwise_summary_from_partition(
        phi=phi,
        d=d,
        m_hat=m_hat,
        partition=partition,
        cov_type=cov_type_u,
        alpha=alpha,
    )

    return _build_gate_estimate(
        estimand="GATE",
        irm_model=irm_model,
        alpha=alpha,
        cov_type_u=cov_type_u,
        cov_kwds=cov_kwds,
        partition=partition,
        stats=stats,
        orthogonal_signal=phi,
    )


def estimate_gatet_from_irm(
    irm_model: Any,
    groups: Optional[pd.DataFrame | pd.Series | np.ndarray],
    alpha: float = 0.05,
    cov_type: str = "HC3",
    cov_kwds: Optional[Dict[str, Any]] = None,
) -> GateEstimate:
    r"""Estimate Group Average Treatment Effects on the Treated (GATETs) from a fitted IRM.

    Parameters
    ----------
    irm_model : fitted IRM
        Interactive Regression Model with stored cross-fitted nuisance predictions
        :math:`\hat g_0(X)`, :math:`\hat g_1(X)`, and :math:`\hat m(X)`.
        The model must already be fitted.
    groups : Optional[pd.DataFrame, pd.Series, or np.ndarray]
        Pre-specified subgroup definition.

        Accepted forms are:
        - A single Series / one-column DataFrame of group labels.
        - A multi-column dummy basis with binary indicators.
        - A numpy array of group labels (assumed aligned with fit-time rows).

        For strict GATET, groups must define a mutually exclusive and exhaustive
        partition of the fitted sample. Each group must contain at least one
        treated observation. Groups with no controls are still accepted for
        GATET, but a warning is emitted because within-group overlap is
        degenerate and identification relies on :math:`\hat g_0(X)` and
        :math:`\hat m(X)` learned outside the subgroup.

        GATET requires the fitted ``CausalData`` to define ``user_id``.
        Group rows are aligned to the IRM fit-time sample using those fit-time
        observation ids. In practice, ``groups.index`` must either exactly match
        those ids or be a permutation of them. As a convenience, when the model
        was fitted with ``user_id``, GATET also accepts groups indexed by the
        original fit-time row index if the current row-to-``user_id`` mapping is
        still unchanged.

        If None, falls back to ``irm_model.data.gate_groups`` when present.
    alpha : float, default 0.05
        Significance level for two-sided confidence intervals.
    cov_type : {"HC0", "HC1", "HC2", "HC3"}, default "HC3"
        Heteroskedasticity-robust covariance estimator used for subgroup
        inference. The implementation uses a closed-form subgroupwise sandwich
        covariance equivalent to a no-intercept regression on subgroup dummies
        scaled by within-group treated shares.
    cov_kwds : Optional[Dict[str, Any]], default None
        Additional covariance options requested by the caller.
        These are currently ignored because GATET uses the closed-form HCx
        covariance formulas implemented in this module.

    Returns
    -------
    GateEstimate
        Result contract containing subgroup-treated effects, standard errors,
        confidence intervals, covariance matrix, and group-level diagnostics.
        The returned object also supports ``contrast(...)`` and
        ``pairwise_summary(...)`` for formal post-estimation group comparisons.
        Per-group ``summary()["is_significant"]`` tests each subgroup effect
        against zero; use formal contrasts to test whether subgroup effects
        differ from one another.

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
    >>> gatet = estimate_gatet_from_irm(irm, groups=groups)
    >>> gatet.summary()  # doctest: +SKIP

    Notes
    -----
    This implementation targets subgroup ATT under the same unconfoundedness
    and overlap assumptions used by IRM, with the additional requirement that
    subgroup membership is pre-treatment.

    Let :math:`G` denote a finite partition of the sample space. For subgroup
    :math:`g`, the target estimand is

    .. math::

        \theta_g^{\mathrm{GATET}}
        = \mathbb{E}[Y(1)-Y(0)\mid G=g, D=1].

    Equivalently, for a strict subgroup partition,

    .. math::

        \theta_g^{\mathrm{GATET}}
        = \mathbb{E}\left[
            \frac{D \mathbf{1}\{G=g\}}{\mathbb{P}(D=1, G=g)}
            (Y(1)-Y(0))
          \right].

    Using the fitted nuisances :math:`\hat g_0(X)` and :math:`\hat m(X)`, define
    the ATT-style orthogonal signal

    .. math::

        z =
        D \cdot (Y - \hat g_0(X))
        - \hat m(X)(1-D)\frac{Y-\hat g_0(X)}{1-\hat m(X)}.

    Then the subgroup ATT can be written as

    .. math::

        \hat\theta_g^{\mathrm{GATET}}
        = \frac{1}{n_{\mathrm{treated}, g}}
          \sum_{i:G_i=g} z_i,

    where :math:`n_{\mathrm{treated}, g}` is the number of treated observations
    in subgroup :math:`g`. In particular, the implementation changes the score
    itself to the subgroup-ATT signal above; it is not computed by first
    building ordinary GATE scores and then averaging them only over treated
    units in subgroup :math:`g`.

    Inference is based on the subgroup moment residual

    .. math::

        u_{ig} = \mathbf{1}\{G_i=g\}\,(z_i - D_i \hat\theta_g),

    with diagonal HC0/HC1/HC2/HC3 covariance formulas computed in closed form.

    As a sanity check, when the groups form an exhaustive partition,

    .. math::

        \mathrm{ATTE}
        = \sum_g \mathbb{P}(G=g \mid D=1)\,
          \theta_g^{\mathrm{GATET}}.

    ``normalize_ipw=True`` on the IRM is intentionally ignored for GATET so the
    estimator uses the canonical unnormalized subgroup-ATT signal above.
    """
    groups_resolved, cov_type_u = _validate_gate_inputs(
        irm_model=irm_model,
        groups=groups,
        alpha=alpha,
        cov_type=cov_type,
        cov_kwds=cov_kwds,
        estimand="GATET",
    )

    normalize_ipw_requested = bool(getattr(irm_model, "normalize_ipw", False))
    if normalize_ipw_requested:
        warnings.warn(
            "normalize_ipw=True is ignored for GATET to preserve canonical unnormalized orthogonal signals.",
            RuntimeWarning,
        )
    if cov_kwds:
        warnings.warn(
            "cov_kwds are ignored for GATET; closed-form groupwise HCx covariance has no cov_kwds options.",
            RuntimeWarning,
        )

    z, d, m_hat = _compute_gatet_signal_from_irm(irm_model)
    groups_aligned = _align_groups_to_fit_observations(
        groups_resolved,
        irm_model=irm_model,
        n_obs=z.shape[0],
        estimand="GATET",
    )
    partition = _coerce_groups_to_partition(groups_aligned, n_obs=z.shape[0])
    stats = _estimate_gatet_groupwise_summary_from_partition(
        z=z,
        d=d,
        m_hat=m_hat,
        partition=partition,
        cov_type=cov_type_u,
        alpha=alpha,
    )

    orthogonal_signal = np.asarray(stats["orthogonal_signal"], dtype=float)
    raw_signal = np.asarray(stats["raw_signal"], dtype=float)
    stats = dict(stats)
    stats.pop("orthogonal_signal", None)
    stats.pop("raw_signal", None)

    return _build_gate_estimate(
        estimand="GATET",
        irm_model=irm_model,
        alpha=alpha,
        cov_type_u=cov_type_u,
        cov_kwds=cov_kwds,
        partition=partition,
        stats=stats,
        orthogonal_signal=orthogonal_signal,
        diagnostic_extra={"raw_treated_signal": raw_signal},
    )
