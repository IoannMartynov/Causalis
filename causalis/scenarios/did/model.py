from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Hashable, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm

from causalis.data_contracts.panel_data_did import PanelDataDID
from causalis.data_contracts.panel_did_estimate import PanelDIDDiagnosticData, PanelDIDEstimate


_MODEL_NAME = "SantAnnaZhaoImprovedPanelDRDID"
_DEFAULT_PROPENSITY_CLIP = 1e-8
_DEFAULT_OPTIMIZER_TOL = 1e-8
_DEFAULT_OPTIMIZER_MAXITER = 1000


@dataclass(frozen=True)
class _PreparedPanel:
    unit_ids: list[Hashable]
    pre_time: Any
    post_time: Any
    treatment_start: Any
    delta_y: np.ndarray
    d: np.ndarray
    x: Optional[np.ndarray]
    design: np.ndarray
    covariate_names: list[str]
    clusters: Optional[np.ndarray]


@dataclass(frozen=True)
class _FittedPanel:
    prepared: _PreparedPanel
    gamma_hat: np.ndarray
    propensity_score: np.ndarray
    beta_hat: np.ndarray
    control_outcome_evolution: np.ndarray
    treated_weights: np.ndarray
    control_weights: np.ndarray
    att: float
    influence_scores: np.ndarray
    variance: float
    cluster_scores: Optional[pd.Series]


def _validate_alpha(alpha: float) -> float:
    value = float(alpha)
    if not np.isfinite(value) or not (0.0 < value < 1.0):
        raise ValueError("alpha must be finite and in (0, 1).")
    return value


def _validate_propensity_clip(propensity_clip: float) -> float:
    value = float(propensity_clip)
    if not np.isfinite(value) or not (0.0 < value < 0.5):
        raise ValueError("propensity_clip must be finite and in (0, 0.5).")
    return value


def _append_unique(items: list[str], value: Optional[str]) -> None:
    if value is not None and value not in items:
        items.append(value)


def _resolve_unit_clusters(data: PanelDataDID, df: pd.DataFrame, unit_ids: list[Hashable]) -> Optional[np.ndarray]:
    cluster_col = data.cluster_col
    if cluster_col is None:
        return None
    if cluster_col == data.unit_col:
        return np.asarray(unit_ids, dtype=object)

    cluster_counts = df.groupby(data.unit_col, sort=False)[cluster_col].nunique(dropna=False)
    unstable = cluster_counts[cluster_counts > 1]
    if not unstable.empty:
        raise ValueError(
            "cluster_col must be stable within unit across the pre and post rows used by SantAnnaZhaoDID; "
            f"unstable units: {list(unstable.index)}"
        )

    clusters = (
        df[[data.unit_col, cluster_col]]
        .drop_duplicates(subset=[data.unit_col])
        .set_index(data.unit_col)
        .loc[unit_ids, cluster_col]
        .to_numpy(dtype=object)
    )
    return clusters


def _prepare_panel(data: PanelDataDID) -> _PreparedPanel:
    if not isinstance(data, PanelDataDID):
        raise ValueError("Input must be a PanelDataDID object.")
    if data.design_type != "canonical_2x2":
        raise ValueError(
            "SantAnnaZhaoDID implements the Sant'Anna-Zhao improved panel DR DiD estimator for canonical 2x2 "
            "PanelDataDID inputs only. Multi-period simultaneous-adoption panels are not collapsed."
        )

    pre_time = data.pre_times()[0]
    post_time = data.post_times()[0]
    df = data.df_analysis()

    pre = df[df[data.time_col] == pre_time].copy()
    post = df[df[data.time_col] == post_time].copy()

    covariate_names = list(data.covariates)
    pre_cols = [data.unit_col, data.y]
    pre_cols.extend(covariate_names)
    _append_unique(pre_cols, data.cluster_col)

    post_cols = [data.unit_col, data.y]
    pre_unit = pre[pre_cols].rename(columns={data.y: "_did_y_pre"})
    post_unit = post[post_cols].rename(columns={data.y: "_did_y_post"})
    merged = pre_unit.merge(
        post_unit,
        on=data.unit_col,
        how="outer",
        indicator=True,
        validate="one_to_one",
    )

    incomplete = merged.loc[merged["_merge"] != "both", data.unit_col].tolist()
    if incomplete:
        raise ValueError(
            "SantAnnaZhaoDID requires the same units to be observed once before and once after treatment; "
            f"incomplete units: {incomplete}"
        )

    unit_ids = merged[data.unit_col].tolist()
    treated_units = set(data.treated_units)
    d = np.asarray([unit in treated_units for unit in unit_ids], dtype=float)
    if not np.any(d == 1.0) or not np.any(d == 0.0):
        raise ValueError("PanelDataDID must contain at least one treated and one control unit.")

    y_pre = merged["_did_y_pre"].to_numpy(dtype=float)
    y_post = merged["_did_y_post"].to_numpy(dtype=float)
    delta_y = y_post - y_pre
    if not np.isfinite(delta_y).all():
        raise ValueError("Outcome differences must be finite.")

    if covariate_names:
        x = merged[covariate_names].to_numpy(dtype=float)
        if not np.isfinite(x).all():
            raise ValueError("Pre-treatment covariates must be finite.")
    else:
        x = None

    intercept = np.ones((len(unit_ids), 1), dtype=float)
    design = intercept if x is None else np.column_stack([intercept, x])
    clusters = _resolve_unit_clusters(data, df, unit_ids)

    return _PreparedPanel(
        unit_ids=unit_ids,
        pre_time=pre_time,
        post_time=post_time,
        treatment_start=data.treatment_start,
        delta_y=delta_y,
        d=d,
        x=x,
        design=design,
        covariate_names=covariate_names,
        clusters=clusters,
    )


def _safe_exp(values: np.ndarray) -> np.ndarray:
    return np.exp(np.clip(values, -700.0, 700.0))


def _ipt_objective_and_gradient(gamma: np.ndarray, x: np.ndarray, d: np.ndarray) -> tuple[float, np.ndarray]:
    eta = x @ gamma
    exp_eta = _safe_exp(eta)
    residual = (1.0 - d) * exp_eta - d
    objective = float(np.mean((1.0 - d) * exp_eta - d * eta))
    gradient = x.T @ residual / x.shape[0]
    return objective, gradient


def _solve_ipt_gamma(
    x: np.ndarray,
    d: np.ndarray,
    *,
    tol: float = _DEFAULT_OPTIMIZER_TOL,
    max_iter: int = _DEFAULT_OPTIMIZER_MAXITER,
) -> np.ndarray:
    initial = np.zeros(x.shape[1], dtype=float)

    def objective(gamma: np.ndarray) -> float:
        value, _ = _ipt_objective_and_gradient(gamma, x, d)
        return value

    def gradient(gamma: np.ndarray) -> np.ndarray:
        _, grad = _ipt_objective_and_gradient(gamma, x, d)
        return grad

    result = minimize(
        objective,
        initial,
        jac=gradient,
        method="BFGS",
        options={"gtol": tol, "maxiter": int(max_iter)},
    )
    grad_norm = float(np.linalg.norm(gradient(np.asarray(result.x, dtype=float)), ord=np.inf))
    if (not result.success) and grad_norm > max(1e-6, 10.0 * tol):
        raise RuntimeError(f"IPT propensity optimization failed to converge: {result.message}")

    gamma_hat = np.asarray(result.x, dtype=float)
    if not np.isfinite(gamma_hat).all():
        raise RuntimeError("IPT propensity optimization produced non-finite coefficients.")
    return gamma_hat


def _weighted_least_squares(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
    if x.shape[0] == 0:
        raise ValueError("Control outcome evolution WLS requires at least one control unit.")
    if not np.isfinite(weights).all() or np.any(weights <= 0.0):
        raise ValueError("Control outcome evolution WLS weights must be positive and finite.")

    sqrt_w = np.sqrt(weights)
    x_weighted = x * sqrt_w[:, None]
    y_weighted = y * sqrt_w
    beta_hat, *_ = np.linalg.lstsq(x_weighted, y_weighted, rcond=None)
    beta_hat = np.asarray(beta_hat, dtype=float)
    if not np.isfinite(beta_hat).all():
        raise RuntimeError("Control outcome evolution WLS produced non-finite coefficients.")
    return beta_hat


def _effective_sample_size(weights: np.ndarray) -> float:
    positive = np.asarray(weights, dtype=float)
    denom = float(np.sum(positive ** 2))
    if denom <= 0.0:
        return float("nan")
    return float((np.sum(positive) ** 2) / denom)


def _overlap_summary(e_hat: np.ndarray, d: np.ndarray, w1: np.ndarray, w0: np.ndarray, clip: float) -> dict[str, Any]:
    treated = d == 1.0
    control = d == 0.0
    return {
        "propensity_clip": float(clip),
        "min": float(np.min(e_hat)),
        "max": float(np.max(e_hat)),
        "mean": float(np.mean(e_hat)),
        "treated_min": float(np.min(e_hat[treated])),
        "treated_max": float(np.max(e_hat[treated])),
        "control_min": float(np.min(e_hat[control])),
        "control_max": float(np.max(e_hat[control])),
        "treated_weight_ess": _effective_sample_size(w1[treated]),
        "control_weight_ess": _effective_sample_size(w0[control]),
    }


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    total = float(np.sum(weights))
    if total <= 0.0:
        return float("nan")
    return float(np.sum(values * weights) / total)


def _smd(numerator: float, treated_values: np.ndarray, control_values: np.ndarray) -> float:
    var_t = float(np.var(treated_values, ddof=1)) if treated_values.size > 1 else 0.0
    var_c = float(np.var(control_values, ddof=1)) if control_values.size > 1 else 0.0
    pooled = float(np.sqrt(0.5 * (var_t + var_c)))
    if pooled == 0.0:
        return 0.0 if abs(numerator) <= 1e-16 else float("inf")
    return float(numerator / pooled)


def _balance_table(
    x: Optional[np.ndarray],
    covariate_names: list[str],
    d: np.ndarray,
    w1: np.ndarray,
    w0: np.ndarray,
) -> pd.DataFrame:
    columns = [
        "treated_mean",
        "control_mean",
        "weighted_treated_mean",
        "weighted_control_mean",
        "smd_unweighted",
        "smd_weighted",
    ]
    if x is None or not covariate_names:
        return pd.DataFrame(columns=columns)

    treated = d == 1.0
    control = d == 0.0
    rows: list[dict[str, float]] = []
    for idx, name in enumerate(covariate_names):
        values = x[:, idx]
        treated_values = values[treated]
        control_values = values[control]
        treated_mean = float(np.mean(treated_values))
        control_mean = float(np.mean(control_values))
        weighted_treated_mean = _weighted_mean(values, w1)
        weighted_control_mean = _weighted_mean(values, w0)
        rows.append(
            {
                "covariate": name,
                "treated_mean": treated_mean,
                "control_mean": control_mean,
                "weighted_treated_mean": weighted_treated_mean,
                "weighted_control_mean": weighted_control_mean,
                "smd_unweighted": _smd(treated_mean - control_mean, treated_values, control_values),
                "smd_weighted": _smd(weighted_treated_mean - weighted_control_mean, treated_values, control_values),
            }
        )
    return pd.DataFrame(rows).set_index("covariate")[columns]


def _cluster_variance(
    psi: np.ndarray,
    clusters: Optional[np.ndarray],
) -> tuple[float, Optional[pd.Series]]:
    n = psi.size
    if clusters is None:
        return float(np.sum(psi ** 2) / (n ** 2)), None

    cluster_scores = (
        pd.DataFrame({"cluster": clusters, "psi": psi})
        .groupby("cluster", sort=False, observed=True)["psi"]
        .sum()
    )
    g = int(cluster_scores.shape[0])
    if g < 2:
        raise ValueError("Clustered inference requires at least two clusters.")
    centered = cluster_scores.to_numpy(dtype=float) - float(cluster_scores.mean())
    var_hat = float(g / (g - 1.0) * np.sum(centered ** 2) / (n ** 2))
    return var_hat, cluster_scores


def _normal_p_value(att: float, se: float) -> float:
    if se > 0.0:
        return float(2.0 * norm.sf(abs(att / se)))
    return 1.0 if abs(att) <= 1e-16 else 0.0


def _validate_optimizer_maxiter(optimizer_maxiter: int) -> int:
    value = int(optimizer_maxiter)
    if value <= 0:
        raise ValueError("optimizer_maxiter must be a positive integer.")
    return value


def _validate_optimizer_tol(optimizer_tol: float) -> float:
    value = float(optimizer_tol)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("optimizer_tol must be positive and finite.")
    return value


def _fit_prepared_panel(
    prepared: _PreparedPanel,
    *,
    propensity_clip: float,
    optimizer_tol: float,
    optimizer_maxiter: int,
) -> _FittedPanel:
    x = prepared.design
    d = prepared.d
    delta_y = prepared.delta_y
    n = delta_y.size

    gamma_hat = _solve_ipt_gamma(x, d, tol=optimizer_tol, max_iter=optimizer_maxiter)
    e_raw = expit(x @ gamma_hat)
    e_hat = np.clip(e_raw, propensity_clip, 1.0 - propensity_clip)
    if not np.isfinite(e_hat).all() or np.any(e_hat <= 0.0) or np.any(e_hat >= 1.0):
        raise RuntimeError("Estimated propensity scores must be finite and strictly inside (0, 1).")

    control = d == 0.0
    w_wls = e_hat[control] / (1.0 - e_hat[control])
    beta_hat = _weighted_least_squares(x[control], delta_y[control], w_wls)
    m0_delta_hat = x @ beta_hat

    q_hat = float(np.mean(d))
    if q_hat <= 0.0 or q_hat >= 1.0:
        raise ValueError("Treatment share must be strictly between 0 and 1.")
    w1 = d / q_hat

    odds_control = e_hat * (1.0 - d) / (1.0 - e_hat)
    odds_mean = float(np.mean(odds_control))
    if not np.isfinite(odds_mean) or odds_mean <= 0.0:
        raise ValueError("Normalized control odds weights have zero or non-finite mean.")
    w0 = odds_control / odds_mean

    residualized_delta = delta_y - m0_delta_hat
    att = float(np.mean((w1 - w0) * residualized_delta))
    psi = (w1 - w0) * residualized_delta - w1 * att
    var_hat, cluster_scores = _cluster_variance(psi, prepared.clusters)

    return _FittedPanel(
        prepared=prepared,
        gamma_hat=gamma_hat,
        propensity_score=e_hat,
        beta_hat=beta_hat,
        control_outcome_evolution=m0_delta_hat,
        treated_weights=w1,
        control_weights=w0,
        att=att,
        influence_scores=psi,
        variance=var_hat,
        cluster_scores=cluster_scores,
    )


class SantAnnaZhaoDID:
    """Sant'Anna-Zhao improved panel doubly robust DID estimator.

    The model accepts only canonical 2x2 :class:`PanelDataDID` inputs: the same
    units observed once before and once after simultaneous adoption. The first
    stages are inverse probability tilting for the propensity score and
    control-group WLS for outcome evolution.
    """

    def __init__(
        self,
        *,
        alpha: float = 0.05,
        diagnostic_data: bool = True,
        propensity_clip: float = _DEFAULT_PROPENSITY_CLIP,
        optimizer_tol: float = _DEFAULT_OPTIMIZER_TOL,
        optimizer_maxiter: int = _DEFAULT_OPTIMIZER_MAXITER,
    ) -> None:
        self.alpha = _validate_alpha(alpha)
        self.diagnostic_data = bool(diagnostic_data)
        self.propensity_clip = _validate_propensity_clip(propensity_clip)
        self.optimizer_tol = _validate_optimizer_tol(optimizer_tol)
        self.optimizer_maxiter = _validate_optimizer_maxiter(optimizer_maxiter)

        self._data: Optional[PanelDataDID] = None
        self._fitted: Optional[_FittedPanel] = None
        self._is_fitted = False

    def fit(self, data: PanelDataDID) -> "SantAnnaZhaoDID":
        """Fit IPT and WLS first stages on a canonical 2x2 PanelDataDID object."""

        prepared = _prepare_panel(data)
        self._fitted = _fit_prepared_panel(
            prepared,
            propensity_clip=self.propensity_clip,
            optimizer_tol=self.optimizer_tol,
            optimizer_maxiter=self.optimizer_maxiter,
        )
        self._data = data
        self._is_fitted = True
        return self

    def _require_fitted(self) -> _FittedPanel:
        if not self._is_fitted or self._fitted is None or self._data is None:
            raise RuntimeError("Model must be fitted with .fit(data) before calling .estimate().")
        return self._fitted

    def estimate(
        self,
        *,
        alpha: Optional[float] = None,
        diagnostic_data: Optional[bool] = None,
    ) -> PanelDIDEstimate:
        """Return the fitted ATT estimate and influence-function inference."""

        fitted = self._require_fitted()
        data = self._data
        if data is None:
            raise RuntimeError("Model must be fitted with .fit(data) before calling .estimate().")

        a = self.alpha if alpha is None else _validate_alpha(alpha)
        include_diagnostics = self.diagnostic_data if diagnostic_data is None else bool(diagnostic_data)
        prepared = fitted.prepared
        se = float(np.sqrt(max(fitted.variance, 0.0)))
        z = float(norm.ppf(1.0 - a / 2.0))
        ci_lower = float(fitted.att - z * se)
        ci_upper = float(fitted.att + z * se)
        p_value = _normal_p_value(fitted.att, se)

        diag = None
        if include_diagnostics:
            diag = PanelDIDDiagnosticData(
                unit_ids=prepared.unit_ids,
                d=prepared.d,
                delta_y=prepared.delta_y,
                x=prepared.x,
                covariate_names=prepared.covariate_names,
                propensity_score=fitted.propensity_score,
                control_outcome_evolution=fitted.control_outcome_evolution,
                treated_weights=fitted.treated_weights,
                control_weights=fitted.control_weights,
                influence_scores=fitted.influence_scores,
                gamma_hat=fitted.gamma_hat,
                beta_hat=fitted.beta_hat,
                overlap=_overlap_summary(
                    fitted.propensity_score,
                    prepared.d,
                    fitted.treated_weights,
                    fitted.control_weights,
                    self.propensity_clip,
                ),
                balance=_balance_table(
                    prepared.x,
                    prepared.covariate_names,
                    prepared.d,
                    fitted.treated_weights,
                    fitted.control_weights,
                ),
                cluster_scores=fitted.cluster_scores,
            )

        return PanelDIDEstimate(
            model=_MODEL_NAME,
            treatment_start=prepared.treatment_start,
            pre_time=prepared.pre_time,
            post_time=prepared.post_time,
            att=fitted.att,
            se=se,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            p_value=p_value,
            is_significant=bool(p_value < a),
            alpha=a,
            n_units=len(prepared.delta_y),
            n_treated=int(np.sum(prepared.d == 1.0)),
            n_control=int(np.sum(prepared.d == 0.0)),
            treatment_mean_delta=float(np.mean(prepared.delta_y[prepared.d == 1.0])),
            control_mean_delta=float(np.mean(prepared.delta_y[prepared.d == 0.0])),
            outcome=data.y,
            treatment=data.treated_time,
            covariates=list(data.covariates),
            cluster_col=data.cluster_col,
            inference="clustered_influence" if prepared.clusters is not None else "influence",
            diagnostic_data=diag,
        )

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "unfitted"
        return f"{self.__class__.__name__}(status='{status}', alpha={self.alpha!r})"


__all__ = ["SantAnnaZhaoDID"]
