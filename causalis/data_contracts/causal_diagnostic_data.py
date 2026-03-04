from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np
from pydantic import BaseModel, ConfigDict

# NOTE:
# `RegressionChecks` currently exists under `tests/` upstream and is not packaged in wheels.
# Keep this as a runtime-safe alias until the type is moved into the public package.
RegressionChecks = Any


class DiagnosticData(BaseModel):
    """Base class for all diagnostic data_contracts."""

    model_config = ConfigDict(arbitrary_types_allowed=True)


class UnconfoundednessDiagnosticData(DiagnosticData):
    """Fields common to all models assuming unconfoundedness."""

    m_hat: np.ndarray  # Propensity scores
    m_hat_raw: Optional[np.ndarray] = None  # Raw (pre-clipping) propensity scores when available
    d: np.ndarray  # Treatment indicators
    y: Optional[np.ndarray] = None  # Outcomes
    x: Optional[np.ndarray] = None  # Confounders (for balance checks)
    g0_hat: Optional[np.ndarray] = None  # Estimated outcome under control
    g1_hat: Optional[np.ndarray] = None  # Estimated outcome under treatment
    w: Optional[np.ndarray] = None  # Score target weights used in estimation
    w_bar: Optional[np.ndarray] = None  # Representer weights used in estimation
    psi_b: Optional[np.ndarray] = None  # Orthogonal signal (for DML)
    folds: Optional[np.ndarray] = None  # Cross-fitting folds
    trimming_threshold: float = 0.0
    normalize_ipw: Optional[bool] = None

    # Sensitivity elements (DoubleML-style)
    sigma2: Optional[float] = None
    nu2: Optional[float] = None
    psi_sigma2: Optional[np.ndarray] = None
    psi_nu2: Optional[np.ndarray] = None
    riesz_rep: Optional[np.ndarray] = None
    m_alpha: Optional[np.ndarray] = None
    psi: Optional[np.ndarray] = None
    score: Optional[str] = None  # ATE or ATTE
    sensitivity_analysis: Optional[Dict[str, Any]] = None
    score_plot_cache: Optional[Dict[str, Any]] = None
    residual_plot_cache: Optional[Dict[str, Any]] = None


class MultiUnconfoundednessDiagnosticData(DiagnosticData):
    """Fields common to all models assuming unconfoundedness with multi_unconfoundedness."""

    m_hat: np.ndarray  # Propensity scores
    m_hat_raw: Optional[np.ndarray] = None  # Raw (pre-trimming) propensity scores when available
    d: np.ndarray  # Treatments indicators
    y: Optional[np.ndarray] = None  # Outcomes
    x: Optional[np.ndarray] = None  # Confounders (for balance checks)
    g_hat: Optional[np.ndarray] = None  # Estimated outcome under control
    psi_b: Optional[np.ndarray] = None  # Orthogonal signal (for DML)
    folds: Optional[np.ndarray] = None  # Cross-fitting folds
    trimming_threshold: float = 0.0
    normalize_ipw: Optional[bool] = None

    # Sensitivity elements (DoubleML-style)
    sigma2: Union[float, np.ndarray] = None
    nu2: Optional[np.ndarray] = None
    psi_sigma2: Optional[np.ndarray] = None
    psi_nu2: Optional[np.ndarray] = None
    riesz_rep: Optional[np.ndarray] = None
    m_alpha: Optional[np.ndarray] = None
    psi: Optional[np.ndarray] = None
    score: Optional[str] = None  # ATE or ATTE
    sensitivity_analysis: Optional[Dict[str, Any]] = None
    residual_plot_cache: Optional[Dict[str, Any]] = None


class DiffInMeansDiagnosticData(DiagnosticData):
    """Diagnostic data_contracts for Difference-in-Means model."""

    pass


class CUPEDDiagnosticData(DiagnosticData):
    """Diagnostic data_contracts for CUPED-style (Lin-interacted OLS) adjustment."""

    ate_naive: float
    se_naive: float
    se_reduction_pct_same_cov: float
    r2_naive: float
    r2_adj: float
    beta_covariates: np.ndarray
    gamma_interactions: np.ndarray
    covariate_outcome_corr: Optional[np.ndarray] = None
    covariates: List[str]
    adj_type: str
    regression_checks: Optional[RegressionChecks] = None
