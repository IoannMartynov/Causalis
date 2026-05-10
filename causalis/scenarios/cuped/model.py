r"""
Controlled-experiment Using Pre-Experiment Data (CUPED) scenario.

This module provides the `CUPEDModel` which implements regression-adjusted
inference for Randomized Controlled Trials (RCTs) using pre-treatment covariates.
It primarily implements the Lin (2013) fully interacted OLS specification,
which is a robust generalization of the canonical CUPED estimator.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Optional, Sequence, List, Literal, Tuple

import numpy as np
import pandas as pd

from causalis.dgp.causaldata import CausalData
from causalis.data_contracts.causal_estimate import CausalEstimate
from causalis.data_contracts.causal_diagnostic_data import CUPEDDiagnosticData
from causalis.scenarios.cuped.refutation.config import CUPEDRefutationConfig
from causalis.scenarios.cuped.refutation.regression_checks import (
    RegressionChecks,
    design_matrix_checks,
    regression_assumptions_table_from_checks,
    run_regression_checks,
)

try:
    import statsmodels.api as sm
except Exception as e:  # pragma: no cover
    raise ImportError(
        "CUPEDModel requires statsmodels. Install with: pip install statsmodels"
    ) from e


class CUPEDModel:
    r"""
    CUPED-style regression adjustment estimator for ATE/ITT in randomized experiments.

    The CUPED estimator uses pre-experiment data (covariates) to reduce the
    variance of the treatment effect estimate without introducing bias. While
    the canonical CUPED estimator uses a single variance-reduction parameter $\theta$,
    this implementation follows Lin (2013) and uses a fully interacted OLS
    specification.

    Notes
    -----
    The canonical CUPED adjusted outcome is defined as:

    .. math::

        Y_{cuped} = Y - \theta (X - E[X])

    where $\theta = \frac{Cov(Y, X)}{Var(X)}$ minimizes $Var(Y_{cuped})$.

    This model implements the Lin (2013) specification, which is equivalent to
    saturated OLS and robust to heterogeneous treatment effects:

    .. math::

        Y = \alpha + \tau D + \beta (X - \bar{X}) + \gamma D(X - \bar{X}) + \epsilon

    where:
    - $D$ is the binary treatment indicator ($D=1$ for treatment, $D=0$ for control).
    - $X$ are the pre-treatment covariates (centered globally).
    - $\tau$ is the Average Treatment Effect (ATE).

    Centering covariates at their global mean $\bar{X}$ ensures that the
    coefficient $\tau$ on the treatment indicator $D$ directly estimates the ATE.

    Examples
    --------
    >>> from causalis.scenarios.cuped.dgp import generate_cuped_tweedie_26
    >>> from causalis.scenarios.cuped.model import CUPEDModel
    >>> from causalis.data_contracts import CausalData
    >>> # Generate synthetic data with pre-treatment covariate
    >>> data = generate_cuped_tweedie_26(seed=42, return_causal_data=False)
    >>> causaldata = CausalData(
    ...     df=data,
    ...     treatment='d',
    ...     outcome='y',
    ...     confounders=['y_pre']
    ... )
    >>> # Fit CUPED model adjusting for 'y_pre'
    >>> model = CUPEDModel().fit(causaldata, covariates=['y_pre'])
    >>> # Estimate ATE
    >>> estimate = model.estimate()
    >>> print(f"ATE: {estimate.value:.4f}")
    ATE: 0.6937
    >>> print(f"P-value: {estimate.p_value:.4f}")
    P-value: 0.0000

    Parameters
    ----------
    cov_type : str, default="HC2"
        Covariance estimator passed to statsmodels (e.g., "nonrobust", "HC0", "HC1", "HC2", "HC3").
        Note: for cluster-randomized designs, use cluster-robust SEs (not implemented here).
    alpha : float, default=0.05
        Significance level for confidence intervals.
    use_t : bool | None, default=None
        If bool, passed to statsmodels `.fit(..., use_t=use_t)` directly.
        If None, automatic policy is used: for robust HC* covariances,
        `use_t=True` when `n < use_t_auto_n_threshold`, else `False`.
        For non-robust covariance, `use_t=True`.
    use_t_auto_n_threshold : int, default=5000
        Sample-size threshold for automatic `use_t` selection when `use_t=None`
        and covariance is HC* robust.
    relative_ci_method : {"delta", "bootstrap"}, default="delta"
        Method for relative CI of `100 * tau / denominator`.
        - "delta": joint delta method that accounts for covariance between the
          adjusted ATE and the selected denominator.
        - "bootstrap": percentile bootstrap CI on the relative effect.
    relative_denominator : {"adjusted_control", "raw_control"}, default="adjusted_control"
        Denominator used for relative effects.
        - "adjusted_control": model-implied control mean at the full-sample covariate mean.
        - "raw_control": observed control-group outcome mean.
    relative_ci_bootstrap_draws : int, default=1000
        Number of bootstrap resamples used when `relative_ci_method="bootstrap"`.
    relative_ci_bootstrap_seed : int | None, default=None
        RNG seed used for bootstrap relative CI.
    refutation_config : CUPEDRefutationConfig | None, default=None
        Grouped configuration for regression checks, refutation thresholds, and
        check actions.
    covariate_variance_min : float, default=1e-12
        Minimum variance threshold for retaining a CUPED covariate. Covariates with
        variance less than or equal to this threshold are dropped before fitting.

    Notes
    -----
    - Validity requires covariates be pre-treatment. Post-treatment covariates can bias estimates.
    - Covariates are globally centered over the full sample only. This centering
      convention is required so the treatment coefficient in the Lin specification
      remains the ATE/ITT.
    - The Lin (2013) specification is recommended as a robust regression-adjustment default
      in RCTs.
    """

    def __init__(
        self,
        cov_type: str = "HC2",
        alpha: float = 0.05,
        use_t: Optional[bool] = None,
        use_t_auto_n_threshold: int = 5000,
        relative_ci_method: Literal["delta", "bootstrap"] = "delta",
        relative_denominator: Literal["adjusted_control", "raw_control"] = "adjusted_control",
        relative_ci_bootstrap_draws: int = 1000,
        relative_ci_bootstrap_seed: Optional[int] = None,
        refutation_config: Optional[CUPEDRefutationConfig] = None,
        covariate_variance_min: float = 1e-12,
    ) -> None:
        self.cov_type = str(cov_type)
        self.alpha = float(alpha)
        self.center_covariates = True
        self.centering_scope: Literal["global"] = "global"
        self.adjustment: Literal["lin"] = "lin"
        self.use_t = None if use_t is None else bool(use_t)
        self.use_t_auto_n_threshold = int(use_t_auto_n_threshold)
        if self.use_t_auto_n_threshold <= 0:
            raise ValueError("use_t_auto_n_threshold must be a positive integer.")
        if relative_ci_method not in {"delta", "bootstrap"}:
            raise ValueError(
                "relative_ci_method must be one of {'delta', 'bootstrap'}."
            )
        self.relative_ci_method: Literal["delta", "bootstrap"] = relative_ci_method
        if relative_denominator not in {"raw_control", "adjusted_control"}:
            raise ValueError(
                "relative_denominator must be one of {'raw_control', 'adjusted_control'}."
            )
        self.relative_denominator: Literal["adjusted_control", "raw_control"] = relative_denominator
        self.relative_ci_bootstrap_draws = int(relative_ci_bootstrap_draws)
        if self.relative_ci_bootstrap_draws <= 0:
            raise ValueError("relative_ci_bootstrap_draws must be a positive integer.")
        self.relative_ci_bootstrap_seed = relative_ci_bootstrap_seed
        self.covariate_variance_min = float(covariate_variance_min)
        if self.covariate_variance_min < 0.0:
            raise ValueError("covariate_variance_min must be non-negative.")
        if refutation_config is None:
            self.refutation_config = CUPEDRefutationConfig()
        elif isinstance(refutation_config, CUPEDRefutationConfig):
            self.refutation_config = refutation_config
        else:
            raise TypeError("refutation_config must be CUPEDRefutationConfig or None.")
        if not (0.0 < self.alpha < 1.0):
            raise ValueError("alpha must be in (0, 1).")

        self._is_fitted: bool = False
        self._result: Any = None
        self._result_naive: Any = None
        self._use_t_effective: Optional[bool] = None
        self._covariate_names: List[str] = []
        self._dropped_covariates: List[str] = []
        self._p: int = 0  # number of covariates used
        self._data: Optional[CausalData] = None
        self._regression_checks: Optional[RegressionChecks] = None
        self._regression_assumptions_table: Optional[pd.DataFrame] = None

    def fit(
        self,
        data: CausalData,
        covariates: Optional[Sequence[str]] = None,
        run_checks: Optional[bool] = None,
    ) -> CUPEDModel:
        """
        Fit CUPED-style regression adjustment (Lin-interacted OLS) on a CausalData object.

        Parameters
        ----------
        data : CausalData
            Validated dataset with columns: outcome (post), treatment, and confounders (pre covariates).
        covariates : Sequence[str], required
            Explicit subset of `data_contracts.confounders_names` to use as CUPED covariates.
            Pass `[]` for an unadjusted (naive) fit.
        run_checks : bool | None, optional
            Override whether regression checks are computed in this fit call.
            If ``None``, uses ``self.refutation_config.run_regression_checks``.

        Returns
        -------
        CUPEDModel
            Fitted estimator.

        Raises
        ------
        ValueError
            If `covariates` is omitted, not a sequence of strings, contains columns missing from the
            DataFrame, contains columns outside `data_contracts.confounders_names`,
            or the design matrix is rank deficient.
        """
        df = data.df
        y_name = data.outcome_name
        t_name = data.treatment_name

        # Choose covariates used for adjustment
        if covariates is None:
            raise ValueError(
                "covariates must be provided explicitly as a sequence of pre-treatment columns; "
                "pass [] for naive (no CUPED covariates)."
            )
        if isinstance(covariates, (str, bytes)) or not isinstance(covariates, Sequence):
            raise ValueError(
                "covariates must be a sequence of column names (Sequence[str]); "
                f"got {type(covariates).__name__}."
            )
        x_names = list(covariates)
        if not all(isinstance(c, str) for c in x_names):
            bad_types = sorted({type(c).__name__ for c in x_names if not isinstance(c, str)})
            raise ValueError(
                "covariates must contain only strings; "
                f"found non-string types: {bad_types}."
            )
        duplicate_covariates = sorted([name for name, count in Counter(x_names).items() if count > 1])
        if duplicate_covariates:
            raise ValueError(
                "covariates must not contain duplicates; "
                f"found duplicates: {duplicate_covariates}."
            )

        missing = [c for c in x_names if c not in df.columns]
        if missing:
            raise ValueError(f"CUPED covariates not found in data_contracts.df: {missing}")
        allowed_confounders = set(data.confounders_names)
        not_in_contract = [c for c in x_names if c not in allowed_confounders]
        if not_in_contract:
            raise ValueError(
                "CUPED covariates must be a subset of data_contracts.confounders_names; "
                f"not allowed: {not_in_contract}"
            )

        y = df[y_name].astype(float)
        d = df[t_name].astype(float).to_numpy(dtype=float)

        if len(x_names) > 0:
            x_df = df[x_names].astype(float)
            x_df, dropped = self._drop_near_zero_variance_covariates(
                covariates=x_df,
                variance_min=self.covariate_variance_min,
            )
            if dropped:
                self._check_signal(
                    "Dropped near-zero variance CUPED covariates: "
                    f"{dropped} (variance <= {self.covariate_variance_min:.3e}).",
                )
            x_names = list(x_df.columns)
            self._dropped_covariates = dropped
        else:
            self._dropped_covariates = []

        n = len(y)
        if n == 0:
            raise ValueError("CUPEDModel requires at least one observation.")
        cfg = self.refutation_config
        do_checks = cfg.run_regression_checks if run_checks is None else bool(run_checks)
        self._regression_checks = None
        self._regression_assumptions_table = None

        # Global (full-sample) centering only. Do not center within treatment groups.
        if len(x_names) > 0:
            Xc = self._center_covariates_global(x_df)
            centered_names = [f"{c}__centered" for c in x_names]
            Xc.columns = centered_names
            p = Xc.shape[1]
        else:
            Xc = pd.DataFrame(index=df.index)
            centered_names = []
            p = 0

        # Design matrix with explicit names: [intercept, D, Xc, D*Xc]
        design = pd.DataFrame(
            {"intercept": np.ones(n, dtype=float), t_name: d},
            index=df.index,
        )
        if p > 0:
            for raw_name, centered_name in zip(x_names, centered_names):
                centered_values = Xc[centered_name].to_numpy(dtype=float)
                design[centered_name] = centered_values
                design[f"{t_name}:{raw_name}"] = d * centered_values

        k_design, rank_design, full_rank_design, cond_number = design_matrix_checks(design)
        if not full_rank_design:
            raise ValueError(
                f"Design matrix is rank deficient: rank={rank_design}, k={k_design}. "
                "Likely perfect multicollinearity from duplicate covariates/interactions."
            )
        if not np.isfinite(cond_number) or cond_number > cfg.condition_number_warn_threshold:
            self._check_signal(
                "CUPED design matrix is ill-conditioned "
                f"(condition_number={cond_number:.3e}, "
                f"threshold={cfg.condition_number_warn_threshold:.3e}). "
                "Inference may be unstable.",
            )

        # Fit adjusted model with requested covariance estimator
        use_t_fit = self._resolve_use_t(n=n)
        model = sm.OLS(y, design)
        self._result = model.fit(cov_type=self.cov_type, use_t=use_t_fit)

        # Fit naive model: Y ~ 1 + D
        design_naive = pd.DataFrame(
            {"intercept": np.ones(n, dtype=float), t_name: d},
            index=df.index,
        )
        model_naive = sm.OLS(y, design_naive)
        self._result_naive = model_naive.fit(cov_type=self.cov_type, use_t=use_t_fit)
        self._use_t_effective = use_t_fit

        if do_checks:
            self._regression_checks = run_regression_checks(
                y=y,
                design=design,
                result=self._result,
                result_naive=self._result_naive,
                cov_type=self.cov_type,
                use_t_fit=use_t_fit,
                corr_near_one_tol=cfg.corr_near_one_tol,
                tiny_one_minus_h_tol=cfg.tiny_one_minus_h_tol,
                winsor_q=cfg.winsor_q,
            )
            bse_treat = float(np.asarray(self._result.bse, dtype=float)[1])
            self._regression_assumptions_table = regression_assumptions_table_from_checks(
                checks=self._regression_checks,
                cov_type=self.cov_type,
                condition_number_warn_threshold=cfg.condition_number_warn_threshold,
                vif_warn_threshold=cfg.vif_warn_threshold,
                tiny_one_minus_h_tol=cfg.tiny_one_minus_h_tol,
                winsor_reference_se=bse_treat,
            )
            self._signal_assumption_flags(
                table=self._regression_assumptions_table,
            )

        self._covariate_names = x_names
        self._p = p
        self._data = data
        self._is_fitted = True
        return self

    def estimate(self, alpha: Optional[float] = None, diagnostic_data: bool = True) -> CausalEstimate:
        """
        Return the adjusted ATE/ITT estimate and inference.

        Parameters
        ----------
        alpha : float, optional
            Override the instance significance level for confidence intervals.
        diagnostic_data : bool, default True
            Whether to include diagnostic data_contracts in the result.

        Returns
        -------
        CausalEstimate
            A results object containing effect estimates and inference.
        """
        self._require_fitted()

        a = self._validate_alpha(self.alpha if alpha is None else alpha)

        # Coef index: 0 intercept, 1 treatment, then covariates / interactions
        params = np.asarray(self._result.params, dtype=float)
        bse = np.asarray(self._result.bse, dtype=float)
        pvalues = np.asarray(self._result.pvalues, dtype=float)

        tau = float(params[1])
        se = float(bse[1])
        p_value = float(pvalues[1])

        ci = self._result.conf_int(alpha=a)
        ci_arr = np.asarray(ci, dtype=float)
        ci_low = float(ci_arr[1, 0])
        ci_high = float(ci_arr[1, 1])

        # Relative effect: adjusted ATE divided by the configured denominator.
        # By default this preserves the historical business-reporting convention:
        # adjusted numerator over the raw observed control mean.
        y_internal = np.asarray(self._result.model.endog, dtype=float)
        design_internal = np.asarray(self._result.model.exog, dtype=float)
        d_internal = np.asarray(design_internal[:, 1], dtype=float)
        treated_mask = d_internal == 1.0
        control_mask = d_internal == 0.0
        mu_t = float(np.mean(y_internal[treated_mask])) if np.any(treated_mask) else np.nan
        mu_c = float(np.mean(y_internal[control_mask])) if np.any(control_mask) else np.nan
        relative_denominator_value = self._relative_denominator_value(
            params=params,
            y=y_internal,
            control_mask=control_mask,
        )

        tau_rel = np.nan
        ci_low_rel = np.nan
        ci_high_rel = np.nan

        crit = self._critical_from_ci(tau=tau, se=se, ci_low=ci_low, ci_high=ci_high)
        if self._is_valid_relative_denominator(relative_denominator_value) and np.isfinite(crit):
            tau_rel = 100.0 * tau / relative_denominator_value
            if self.relative_ci_method == "bootstrap":
                # Bootstrap jointly captures uncertainty in tau and the denominator.
                ci_low_rel, ci_high_rel = self._relative_ci_bootstrap(alpha=a)
            else:
                ci_low_rel, ci_high_rel = self._relative_ci_delta(
                    tau=tau,
                    denominator=relative_denominator_value,
                    y=y_internal,
                    design=design_internal,
                    control_mask=control_mask,
                    crit=crit,
                )

        diag = None
        if diagnostic_data:
            params_naive = np.asarray(self._result_naive.params, dtype=float)
            bse_naive = np.asarray(self._result_naive.bse, dtype=float)
            ate_naive = float(params_naive[1])
            se_naive = float(bse_naive[1])
            se_adj = float(bse[1])
            if se_naive > 0.0:
                var_red = 1.0 - (se_adj ** 2) / (se_naive ** 2)
                se_red = 1.0 - se_adj / se_naive
                variance_reduction_pct = float(100.0 * var_red)
                se_reduction_pct = float(100.0 * se_red)
            else:
                variance_reduction_pct = np.nan
                se_reduction_pct = np.nan

            r2_naive = float(self._result_naive.rsquared) if hasattr(self._result_naive, "rsquared") else np.nan
            r2_adj = float(self._result.rsquared) if hasattr(self._result, "rsquared") else np.nan

            p = self._p
            if p == 0:
                beta_cov = np.zeros((0,), dtype=float)
                gamma_cov = np.zeros((0,), dtype=float)
                cov_outcome_corr = np.zeros((0,), dtype=float)
            else:
                # Extract by explicit design names; do not rely on positional blocks.
                exog_names = list(self._result.model.exog_names)
                beta_cov, gamma_cov = self._extract_beta_gamma_by_name(
                    params=params,
                    exog_names=exog_names,
                    treatment_name=str(self._data.treatment_name) if self._data is not None else "treatment",
                )
                cov_raw = self._data.df[list(self._covariate_names)].to_numpy(dtype=float)
                cov_outcome_corr = self._covariate_corr_with_outcome(cov_raw=cov_raw, y=y_internal)

            diag = CUPEDDiagnosticData(
                ate_naive=ate_naive,
                se_naive=se_naive,
                variance_reduction_pct_same_cov=(
                    float(variance_reduction_pct) if np.isfinite(variance_reduction_pct) else np.nan
                ),
                standard_error_reduction_pct_same_cov=(
                    float(se_reduction_pct) if np.isfinite(se_reduction_pct) else np.nan
                ),
                r2_naive=float(r2_naive) if np.isfinite(r2_naive) else np.nan,
                r2_adj=float(r2_adj) if np.isfinite(r2_adj) else np.nan,
                beta_covariates=beta_cov,
                gamma_interactions=gamma_cov,
                covariate_outcome_corr=cov_outcome_corr,
                covariates=list(self._covariate_names),
                adj_type=self.adjustment,
                regression_checks=self._regression_checks,
            )

        return CausalEstimate(
            estimand="ATE",
            model="CUPEDModel",
            model_options={
                "cov_type": self.cov_type,
                "use_t": bool(self._use_t_effective),
                "centering_scope": self.centering_scope,
                "relative_ci_method": self.relative_ci_method,
                "relative_denominator": self.relative_denominator,
                "dropped_covariates": list(self._dropped_covariates),
                "refutation_config": self.refutation_config.to_model_options(),
            },
            value=tau,
            ci_upper_absolute=ci_high,
            ci_lower_absolute=ci_low,
            value_relative=tau_rel,
            ci_upper_relative=ci_high_rel,
            ci_lower_relative=ci_low_rel,
            alpha=a,
            p_value=p_value,
            is_significant=bool(p_value < a),
            n_treated=int(np.sum(self._result.model.exog[:, 1] == 1)),
            n_control=int(np.sum(self._result.model.exog[:, 1] == 0)),
            treatment_mean=mu_t,
            control_mean=mu_c,
            outcome=str(self._data.outcome_name) if self._data is not None else "outcome",
            treatment=str(self._data.treatment_name) if self._data is not None else "treatment",
            confounders=list(self._covariate_names),
            diagnostic_data=diag,
        )

    def summary_dict(self, alpha: Optional[float] = None) -> Dict[str, Any]:
        """
        Convenience JSON/logging output.

        Parameters
        ----------
        alpha : float, optional
            Override the instance significance level for confidence intervals.

        Returns
        -------
        dict
            Dictionary with estimates, inference, and refutation checks.
        """
        eff = self.estimate(alpha=alpha)
        diag: CUPEDDiagnosticData = eff.diagnostic_data
        return {
            "method": "CUPED-style regression adjustment (Lin (2013) fully interacted OLS)",
            "adjustment": diag.adj_type,
            "ate": eff.value,
            "ate_relative_%": eff.value_relative,
            "p_value": eff.p_value,
            "ci_low": eff.ci_lower_absolute,
            "ci_high": eff.ci_upper_absolute,
            "ci_low_relative": eff.ci_lower_relative,
            "ci_high_relative": eff.ci_upper_relative,
            "alpha": eff.alpha,
            "nobs": eff.n_treated + eff.n_control,
            "cov_type": self.cov_type,
            "use_t": bool(self._use_t_effective),
            "centering_scope": self.centering_scope,
            "relative_ci_method": self.relative_ci_method,
            "relative_denominator": self.relative_denominator,
            "dropped_covariates": list(self._dropped_covariates),
            "refutation_config": self.refutation_config.to_model_options(),
            "ate_naive": diag.ate_naive,
            "se_naive": diag.se_naive,
            "variance_reduction_pct_same_cov": diag.variance_reduction_pct_same_cov,
            "standard_error_reduction_pct_same_cov": diag.standard_error_reduction_pct_same_cov,
            "r2_naive": diag.r2_naive,
            "r2_adj": diag.r2_adj,
            "covariates": diag.covariates,
            "beta_covariates": diag.beta_covariates.tolist(),
            "gamma_interactions": diag.gamma_interactions.tolist(),
            "covariate_outcome_corr": (
                diag.covariate_outcome_corr.tolist() if diag.covariate_outcome_corr is not None else None
            ),
            "regression_checks": (
                diag.regression_checks.model_dump() if diag.regression_checks is not None else None
            ),
            "regression_assumptions": (
                self._regression_assumptions_table.to_dict(orient="records")
                if self._regression_assumptions_table is not None
                else None
            ),
        }

    def assumptions_table(self) -> Optional[pd.DataFrame]:
        """Return fitted regression assumptions table (GREEN/YELLOW/RED) when available."""
        self._require_fitted()
        if self._regression_assumptions_table is None:
            return None
        return self._regression_assumptions_table.copy()

    def _signal_assumption_flags(
        self,
        table: pd.DataFrame,
        skip_test_ids: Optional[set[str]] = None,
    ) -> None:
        """Emit refutation signals from GREEN/YELLOW/RED assumption table."""
        self.refutation_config.signal_assumption_flags(
            table=table,
            skip_test_ids=skip_test_ids,
        )

    def _check_signal(self, msg: str) -> None:
        """Emit exception according to configured refutation action."""
        self.refutation_config.signal_message(msg)

    @staticmethod
    def _validate_alpha(alpha: float) -> float:
        value = float(alpha)
        if not (0.0 < value < 1.0):
            raise ValueError("alpha must be in (0, 1).")
        return value

    @staticmethod
    def _center_covariates_global(covariates: pd.DataFrame) -> pd.DataFrame:
        """
        Center covariates over the full sample.

        Parameters
        ----------
        covariates : pandas.DataFrame
            Covariate matrix to center.

        Returns
        -------
        pandas.DataFrame
            Centered covariate matrix.

        Notes
        -----
        This method intentionally does not use treatment groups in centering.
        """
        return covariates - covariates.mean(axis=0)

    @staticmethod
    def _drop_near_zero_variance_covariates(
        covariates: pd.DataFrame, variance_min: float
    ) -> tuple[pd.DataFrame, List[str]]:
        """
        Remove covariates with near-zero variance.

        Parameters
        ----------
        covariates : pandas.DataFrame
            Candidate covariates for CUPED adjustment.
        variance_min : float
            Minimum variance threshold. Columns with variance less than or equal to
            this threshold are dropped.

        Returns
        -------
        tuple[pandas.DataFrame, list[str]]
            A tuple `(kept_covariates, dropped_columns)`.
        """
        if covariates.shape[1] == 0:
            return covariates, []
        variances = covariates.var(axis=0, ddof=0)
        keep_mask = variances > variance_min
        keep_cols = variances.index[keep_mask].tolist()
        drop_cols = variances.index[~keep_mask].tolist()
        return covariates[keep_cols], drop_cols

    @staticmethod
    def _critical_from_ci(tau: float, se: float, ci_low: float, ci_high: float) -> float:
        """
        Recover the critical value implied by a symmetric CI around an estimate.

        Parameters
        ----------
        tau : float
            Point estimate.
        se : float
            Standard error of the estimate.
        ci_low : float
            Lower CI bound.
        ci_high : float
            Upper CI bound.

        Returns
        -------
        float
            Implied critical value. Returns `np.nan` when inputs are invalid.
        """
        if not np.isfinite(se) or se <= 0.0:
            return np.nan
        if not np.isfinite(ci_low) or not np.isfinite(ci_high):
            return np.nan
        up = abs(ci_high - tau)
        down = abs(tau - ci_low)
        return float(max(up, down) / se)

    def _relative_denominator_value(
        self,
        params: np.ndarray,
        y: np.ndarray,
        control_mask: np.ndarray,
    ) -> float:
        """Return the configured denominator for relative effect reporting."""
        if self.relative_denominator == "adjusted_control":
            return float(params[0]) if params.size > 0 else np.nan
        return float(np.mean(y[control_mask])) if np.any(control_mask) else np.nan

    @staticmethod
    def _is_valid_relative_denominator(value: float) -> bool:
        """Return True when a relative-effect denominator is finite and away from zero."""
        return bool(np.isfinite(value) and abs(float(value)) > 1e-12)

    def _relative_ci_delta(
        self,
        tau: float,
        denominator: float,
        y: np.ndarray,
        design: np.ndarray,
        control_mask: np.ndarray,
        crit: float,
    ) -> Tuple[float, float]:
        """Joint delta-method CI for `100 * tau / denominator`."""
        if self.relative_denominator == "adjusted_control":
            cov = np.asarray(self._result.cov_params(), dtype=float)
            if cov.shape[0] < 2 or cov.shape[1] < 2:
                return np.nan, np.nan
            grad = np.zeros((cov.shape[0],), dtype=float)
            grad[0] = -100.0 * tau / (denominator ** 2)
            grad[1] = 100.0 / denominator
            var_rel = float(grad @ cov @ grad)
            tau_rel = 100.0 * tau / denominator
            return self._relative_ci_from_variance(
                tau_rel=tau_rel,
                var_rel=var_rel,
                crit=crit,
            )

        var_tau = float(np.asarray(self._result.cov_params(), dtype=float)[1, 1])
        var_mu = self._raw_control_mean_variance(y=y, control_mask=control_mask)
        cov_tau_mu = self._cov_tau_raw_control_mean(
            y=y,
            design=design,
            control_mask=control_mask,
            var_tau_reference=var_tau,
        )
        return self._relative_ci_from_delta_components(
            tau=tau,
            denominator=denominator,
            var_tau=var_tau,
            var_denominator=var_mu,
            cov_tau_denominator=cov_tau_mu,
            crit=crit,
        )

    @staticmethod
    def _raw_control_mean_variance(y: np.ndarray, control_mask: np.ndarray) -> float:
        """Return the usual sample variance estimate for the raw control mean."""
        n_control = int(np.sum(control_mask))
        if n_control <= 1:
            return np.nan
        return float(np.var(y[control_mask], ddof=1)) / n_control

    def _cov_tau_raw_control_mean(
        self,
        y: np.ndarray,
        design: np.ndarray,
        control_mask: np.ndarray,
        var_tau_reference: float,
    ) -> float:
        """
        Estimate covariance between the adjusted treatment coefficient and raw
        control mean via empirical influence contributions.
        """
        n_control = int(np.sum(control_mask))
        if n_control <= 1:
            return np.nan

        try:
            z = np.asarray(design, dtype=float)
            resid = np.asarray(self._result.resid, dtype=float)
            xtx_inv = np.linalg.pinv(z.T @ z)
        except Exception:
            return np.nan

        if z.shape[0] != y.shape[0] or resid.shape[0] != y.shape[0] or z.shape[1] < 2:
            return np.nan

        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            tau_weights = z @ xtx_inv[:, 1]
            tau_if = tau_weights * resid * self._hc_influence_scale(z=z)

        var_tau_if = float(np.sum(tau_if ** 2))
        if np.isfinite(var_tau_reference) and var_tau_reference >= 0.0 and var_tau_if > 0.0:
            tau_if = tau_if * float(np.sqrt(var_tau_reference / var_tau_if))

        mu_c = float(np.mean(y[control_mask]))
        mu_if = np.zeros_like(y, dtype=float)
        mu_if[control_mask] = (y[control_mask] - mu_c) / n_control
        cov_tau_mu = float(np.sum(tau_if * mu_if))
        return cov_tau_mu if np.isfinite(cov_tau_mu) else np.nan

    def _hc_influence_scale(self, z: np.ndarray) -> np.ndarray:
        """Return residual scaling matching common statsmodels HC covariance types."""
        n, k = z.shape
        cov_upper = str(self.cov_type).strip().upper()
        scale = np.ones((n,), dtype=float)
        if cov_upper == "HC1":
            denom = max(n - k, 1)
            scale *= np.sqrt(float(n) / float(denom))
        elif cov_upper in {"HC2", "HC3"}:
            xtx_inv = np.linalg.pinv(z.T @ z)
            h = np.einsum("ij,jk,ik->i", z, xtx_inv, z)
            one_minus_h = np.maximum(1.0 - np.clip(h, 0.0, 1.0), 1e-15)
            if cov_upper == "HC2":
                scale /= np.sqrt(one_minus_h)
            else:
                scale /= one_minus_h
        return scale

    def _relative_ci_from_delta_components(
        self,
        tau: float,
        denominator: float,
        var_tau: float,
        var_denominator: float,
        cov_tau_denominator: float,
        crit: float,
    ) -> Tuple[float, float]:
        """Build a relative CI from numerator/denominator delta-method pieces."""
        d_tau = 100.0 / denominator
        d_denominator = -100.0 * tau / (denominator ** 2)
        var_rel = (
            (d_tau ** 2) * var_tau
            + (d_denominator ** 2) * var_denominator
            + 2.0 * d_tau * d_denominator * cov_tau_denominator
        )
        tau_rel = 100.0 * tau / denominator
        return self._relative_ci_from_variance(
            tau_rel=tau_rel,
            var_rel=var_rel,
            crit=crit,
        )

    @staticmethod
    def _relative_ci_from_variance(
        tau_rel: float,
        var_rel: float,
        crit: float,
    ) -> Tuple[float, float]:
        """Build a symmetric confidence interval from a relative-effect variance."""
        if not np.isfinite(var_rel):
            return np.nan, np.nan
        se_rel = float(np.sqrt(max(var_rel, 0.0)))
        ci_low_rel = float(tau_rel - crit * se_rel)
        ci_high_rel = float(tau_rel + crit * se_rel)
        if ci_low_rel > ci_high_rel:
            ci_low_rel, ci_high_rel = ci_high_rel, ci_low_rel
        return ci_low_rel, ci_high_rel

    def _relative_ci_bootstrap(self, alpha: float) -> Tuple[float, float]:
        """
        Compute percentile bootstrap CI for relative effect.

        Parameters
        ----------
        alpha : float
            Significance level for two-sided CI.

        Returns
        -------
        tuple[float, float]
            Relative CI `(lower, upper)` on percent scale. Returns `(np.nan, np.nan)`
            if too few valid bootstrap samples are available.
        """
        if self._data is None:
            return np.nan, np.nan

        df = self._data.df
        y_name = self._data.outcome_name
        t_name = self._data.treatment_name
        n = len(df)
        if n <= 1:
            return np.nan, np.nan

        rng = np.random.default_rng(self.relative_ci_bootstrap_seed)
        rel_samples: List[float] = []
        x_names = list(self._covariate_names)
        original_d = df[t_name].to_numpy(dtype=float)
        control_idx = np.flatnonzero(original_d == 0.0)
        treated_idx = np.flatnonzero(original_d == 1.0)
        if control_idx.size == 0 or treated_idx.size == 0:
            return np.nan, np.nan

        for _ in range(self.relative_ci_bootstrap_draws):
            # Resample within arms so every bootstrap refit preserves the RCT split.
            idx_control = rng.choice(control_idx, size=control_idx.size, replace=True)
            idx_treated = rng.choice(treated_idx, size=treated_idx.size, replace=True)
            idx = np.concatenate([idx_control, idx_treated])
            df_b = df.iloc[idx]
            d_b = df_b[t_name].to_numpy(dtype=float)
            c_mask = d_b == 0.0
            t_mask = d_b == 1.0
            if not np.any(c_mask) or not np.any(t_mask):
                continue

            y_b = df_b[y_name].astype(float)
            design_b = pd.DataFrame(
                {"intercept": np.ones(len(df_b), dtype=float), t_name: d_b},
                index=df_b.index,
            )
            if len(x_names) > 0:
                X_b = df_b[x_names].astype(float)
                Xc_b = self._center_covariates_global(X_b)
                for raw_name in x_names:
                    centered_values = Xc_b[raw_name].to_numpy(dtype=float)
                    design_b[f"{raw_name}__centered"] = centered_values
                    design_b[f"{t_name}:{raw_name}"] = d_b * centered_values

            try:
                # Use plain OLS in bootstrap re-fits for robust, stable resampling.
                res_b = sm.OLS(y_b, design_b).fit()
            except Exception:
                continue

            tau_b = float(np.asarray(res_b.params, dtype=float)[1])
            if self.relative_denominator == "adjusted_control":
                denom_b = float(np.asarray(res_b.params, dtype=float)[0])
            else:
                denom_b = float(np.mean(y_b.to_numpy(dtype=float)[c_mask]))
            if not self._is_valid_relative_denominator(denom_b):
                continue
            rel_b = 100.0 * tau_b / denom_b
            if np.isfinite(rel_b):
                rel_samples.append(rel_b)

        if len(rel_samples) < 20:
            return np.nan, np.nan
        q_low, q_high = np.quantile(np.asarray(rel_samples, dtype=float), [alpha / 2.0, 1.0 - alpha / 2.0])
        low = float(q_low)
        high = float(q_high)
        if low > high:
            low, high = high, low
        return low, high

    def _resolve_use_t(self, n: int) -> bool:
        """
        Resolve effective `use_t` flag given covariance type and sample size.

        Parameters
        ----------
        n : int
            Number of observations in the fitted sample.

        Returns
        -------
        bool
            Effective `use_t` passed to statsmodels fit.
        """
        if self.use_t is not None:
            return bool(self.use_t)
        cov_upper = str(self.cov_type).strip().upper()
        if cov_upper.startswith("HC"):
            return bool(n < self.use_t_auto_n_threshold)
        return True

    @staticmethod
    def _covariate_corr_with_outcome(cov_raw: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute per-covariate Pearson correlation with outcome.

        Parameters
        ----------
        cov_raw : numpy.ndarray
            Covariate matrix of shape `(n, p)`.
        y : numpy.ndarray
            Outcome vector of shape `(n,)`.

        Returns
        -------
        numpy.ndarray
            Correlation vector of shape `(p,)`. Entries are `np.nan` for
            degenerate columns.
        """
        p = int(cov_raw.shape[1])
        if p == 0:
            return np.zeros((0,), dtype=float)
        out = np.full((p,), np.nan, dtype=float)
        y_std = float(np.std(y, ddof=0))
        if not np.isfinite(y_std) or y_std <= 0.0:
            return out
        for j in range(p):
            xj = cov_raw[:, j]
            x_std = float(np.std(xj, ddof=0))
            if np.isfinite(x_std) and x_std > 0.0:
                out[j] = float(np.corrcoef(xj, y)[0, 1])
        return out

    @staticmethod
    def _extract_beta_gamma_by_name(
        params: np.ndarray, exog_names: List[str], treatment_name: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract Lin main-effect and interaction coefficients by design names.

        Parameters
        ----------
        params : numpy.ndarray
            Full parameter vector from fitted model.
        exog_names : list[str]
            Exogenous column names in fitted design order.
        treatment_name : str
            Treatment column name used in interaction prefixes.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            `(beta_covariates, gamma_interactions)` arrays in design order.
        """
        beta_idx = [i for i, name in enumerate(exog_names) if str(name).endswith("__centered")]
        gamma_prefix = f"{treatment_name}:"
        gamma_idx = [i for i, name in enumerate(exog_names) if str(name).startswith(gamma_prefix)]
        beta_cov = np.asarray(params[beta_idx], dtype=float) if beta_idx else np.zeros((0,), dtype=float)
        gamma_cov = np.asarray(params[gamma_idx], dtype=float) if gamma_idx else np.zeros((0,), dtype=float)
        return beta_cov, gamma_cov

    def _require_fitted(self) -> None:
        if not self._is_fitted or self._result is None:
            raise RuntimeError(
                "CUPEDModel is not fitted. "
                "Call .fit(causaldata, covariates=[...]) first."
            )

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "unfitted"
        use_t_repr = "auto" if self.use_t is None else str(self.use_t)
        return (
            f"{self.__class__.__name__}("
            f"cov_type='{self.cov_type}', alpha={self.alpha}, "
            f"centering_scope='{self.centering_scope}', "
            f"relative_ci_method='{self.relative_ci_method}', "
            f"relative_denominator='{self.relative_denominator}', "
            f"covariate_variance_min={self.covariate_variance_min}, "
            f"refutation_config={self.refutation_config!r}, "
            f"use_t={use_t_repr}, use_t_auto_n_threshold={self.use_t_auto_n_threshold}, "
            f"use_t_effective={self._use_t_effective}, status='{status}')"
        )
