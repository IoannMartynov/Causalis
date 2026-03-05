from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Optional, Sequence, List, Literal, Tuple

import numpy as np
import pandas as pd

from causalis.dgp.causaldata import CausalData
from causalis.data_contracts.causal_estimate import CausalEstimate
from causalis.data_contracts.causal_diagnostic_data import CUPEDDiagnosticData
from causalis.scenarios.cuped.diagnostics.regression_checks import (
    FLAG_GREEN,
    FLAG_RED,
    FLAG_YELLOW,
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
    """
    CUPED-style regression adjustment estimator for ATE/ITT in randomized experiments.

    Fits an outcome regression with pre-treatment covariates (always centered
    over the full sample, never within treatment groups)
    implemented as Lin (2013) fully interacted OLS:

        Y ~ 1 + D + X^c + D * X^c

    The reported effect is the coefficient on D, with robust covariance as requested.
    This specification ensures the coefficient on D is the ATE/ITT even if the
    treatment effect is heterogeneous with respect to covariates.
    This is broader than canonical single-theta CUPED (`Y - theta*(X - mean(X))`).

    Parameters
    ----------
    cov_type : str, default="HC2"
        Covariance estimator passed to statsmodels (e.g., "nonrobust", "HC0", "HC1", "HC2", "HC3").
        Note: for cluster-randomized designs, use cluster-robust SEs (not implemented here).
    alpha : float, default=0.05
        Significance level for confidence intervals.
    strict_binary_treatment : bool, default=True
        If True, require treatment to be binary {0,1}.
    use_t : bool | None, default=None
        If bool, passed to statsmodels `.fit(..., use_t=use_t)` directly.
        If None, automatic policy is used: for robust HC* covariances,
        `use_t=True` when `n < use_t_auto_n_threshold`, else `False`.
        For non-robust covariance, `use_t=True`.
    use_t_auto_n_threshold : int, default=5000
        Sample-size threshold for automatic `use_t` selection when `use_t=None`
        and covariance is HC* robust.
    relative_ci_method : {"delta_nocov", "bootstrap"}, default="delta_nocov"
        Method for relative CI of `100 * tau / mu_c`.
        - "delta_nocov": delta method using robust `Var(tau)` and `Var(mu_c)` while
          setting `Cov(tau, mu_c)=0` (safe fallback without unsupported hybrid IF covariance).
        - "bootstrap": percentile bootstrap CI on the relative effect.
    relative_ci_bootstrap_draws : int, default=1000
        Number of bootstrap resamples used when `relative_ci_method="bootstrap"`.
    relative_ci_bootstrap_seed : int | None, default=None
        RNG seed used for bootstrap relative CI.
    covariate_variance_min : float, default=1e-12
        Minimum variance threshold for retaining a CUPED covariate. Covariates with
        variance less than or equal to this threshold are dropped before fitting.
    condition_number_warn_threshold : float, default=1e8
        Trigger diagnostics signal when the design matrix condition number exceeds this threshold.
    run_regression_checks : bool, default=True
        Whether to compute regression diagnostics payload during ``fit()``.
    check_action : {"ignore", "raise"}, default="ignore"
        Action used when a diagnostics threshold is violated.
    raise_on_yellow : bool, default=False
        When ``check_action="raise"``, also raise on YELLOW assumption flags.
    corr_near_one_tol : float, default=1e-10
        Correlation tolerance used to mark near-duplicate centered covariates.
    vif_warn_threshold : float, default=20.0
        VIF threshold that triggers a diagnostics signal.
    winsor_q : float | None, default=0.01
        Quantile used for winsor sensitivity refit. Set ``None`` to disable.
    tiny_one_minus_h_tol : float, default=1e-8
        Threshold for flagging near-degenerate ``1 - leverage`` terms in HC2/HC3.

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
        strict_binary_treatment: bool = True,
        use_t: Optional[bool] = None,
        use_t_auto_n_threshold: int = 5000,
        relative_ci_method: Literal["delta_nocov", "bootstrap"] = "delta_nocov",
        relative_ci_bootstrap_draws: int = 1000,
        relative_ci_bootstrap_seed: Optional[int] = None,
        covariate_variance_min: float = 1e-12,
        condition_number_warn_threshold: float = 1e8,
        run_regression_checks: bool = True,
        check_action: Literal["ignore", "raise"] = "ignore",
        raise_on_yellow: bool = False,
        corr_near_one_tol: float = 1e-10,
        vif_warn_threshold: float = 20.0,
        winsor_q: Optional[float] = 0.01,
        tiny_one_minus_h_tol: float = 1e-8,
    ) -> None:
        self.cov_type = str(cov_type)
        self.alpha = float(alpha)
        self.center_covariates = True
        self.centering_scope: Literal["global"] = "global"
        self.strict_binary_treatment = bool(strict_binary_treatment)
        self.adjustment: Literal["lin"] = "lin"
        self.use_t = None if use_t is None else bool(use_t)
        self.use_t_auto_n_threshold = int(use_t_auto_n_threshold)
        if self.use_t_auto_n_threshold <= 0:
            raise ValueError("use_t_auto_n_threshold must be a positive integer.")
        if relative_ci_method not in {"delta_nocov", "bootstrap"}:
            raise ValueError("relative_ci_method must be one of {'delta_nocov', 'bootstrap'}.")
        self.relative_ci_method: Literal["delta_nocov", "bootstrap"] = relative_ci_method
        self.relative_ci_bootstrap_draws = int(relative_ci_bootstrap_draws)
        if self.relative_ci_bootstrap_draws <= 0:
            raise ValueError("relative_ci_bootstrap_draws must be a positive integer.")
        self.relative_ci_bootstrap_seed = relative_ci_bootstrap_seed
        self.covariate_variance_min = float(covariate_variance_min)
        if self.covariate_variance_min < 0.0:
            raise ValueError("covariate_variance_min must be non-negative.")
        self.condition_number_warn_threshold = float(condition_number_warn_threshold)
        if self.condition_number_warn_threshold <= 0.0:
            raise ValueError("condition_number_warn_threshold must be positive.")
        self.run_regression_checks = bool(run_regression_checks)
        if check_action not in {"ignore", "raise"}:
            raise ValueError("check_action must be one of {'ignore', 'raise'}.")
        self.check_action: Literal["ignore", "raise"] = check_action
        self.raise_on_yellow = bool(raise_on_yellow)
        self.corr_near_one_tol = float(corr_near_one_tol)
        if self.corr_near_one_tol < 0.0:
            raise ValueError("corr_near_one_tol must be non-negative.")
        self.vif_warn_threshold = float(vif_warn_threshold)
        if self.vif_warn_threshold <= 0.0:
            raise ValueError("vif_warn_threshold must be positive.")
        if winsor_q is None:
            self.winsor_q = None
        else:
            self.winsor_q = float(winsor_q)
            if not (0.0 < self.winsor_q < 0.5):
                raise ValueError("winsor_q must be in (0, 0.5) when provided.")
        self.tiny_one_minus_h_tol = float(tiny_one_minus_h_tol)
        if self.tiny_one_minus_h_tol <= 0.0:
            raise ValueError("tiny_one_minus_h_tol must be positive.")
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
            If ``None``, uses ``self.run_regression_checks``.

        Returns
        -------
        CUPEDModel
            Fitted estimator.

        Raises
        ------
        ValueError
            If `covariates` is omitted, not a sequence of strings, contains columns missing from the
            DataFrame, contains columns outside `data_contracts.confounders_names`,
            treatment is not binary when `strict_binary_treatment=True`,
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

        if self.strict_binary_treatment:
            uniq = np.unique(d)
            uniq_r = set(np.round(uniq, 12).tolist())
            if not (len(uniq_r) == 2 and uniq_r.issubset({0.0, 1.0})):
                raise ValueError(
                    "Treatment must be binary {0,1} when strict_binary_treatment=True. "
                    f"Found unique values: {uniq.tolist()}"
                )

        n = len(y)
        if n == 0:
            raise ValueError("CUPEDModel requires at least one observation.")
        do_checks = self.run_regression_checks if run_checks is None else bool(run_checks)
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
        if not np.isfinite(cond_number) or cond_number > self.condition_number_warn_threshold:
            self._check_signal(
                "CUPED design matrix is ill-conditioned "
                f"(condition_number={cond_number:.3e}, "
                f"threshold={self.condition_number_warn_threshold:.3e}). "
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
                corr_near_one_tol=self.corr_near_one_tol,
                tiny_one_minus_h_tol=self.tiny_one_minus_h_tol,
                winsor_q=self.winsor_q,
            )
            bse_treat = float(np.asarray(self._result.bse, dtype=float)[1])
            self._regression_assumptions_table = regression_assumptions_table_from_checks(
                checks=self._regression_checks,
                cov_type=self.cov_type,
                condition_number_warn_threshold=self.condition_number_warn_threshold,
                vif_warn_threshold=self.vif_warn_threshold,
                tiny_one_minus_h_tol=self.tiny_one_minus_h_tol,
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

        # Relative effect: tau_rel = 100*tau/mu_c
        y_internal = np.asarray(self._result.model.endog, dtype=float)
        design_internal = np.asarray(self._result.model.exog, dtype=float)
        d_internal = np.asarray(design_internal[:, 1], dtype=float)
        treated_mask = d_internal == 1.0
        control_mask = d_internal == 0.0
        mu_t = float(np.mean(y_internal[treated_mask])) if np.any(treated_mask) else np.nan
        mu_c = float(np.mean(y_internal[control_mask])) if np.any(control_mask) else np.nan

        tau_rel = np.nan
        ci_low_rel = np.nan
        ci_high_rel = np.nan

        crit = self._critical_from_ci(tau=tau, se=se, ci_low=ci_low, ci_high=ci_high)
        if np.isfinite(mu_c) and mu_c != 0.0 and np.isfinite(crit):
            tau_rel = 100.0 * tau / mu_c
            if self.relative_ci_method == "bootstrap":
                # Bootstrap jointly captures uncertainty in tau and mu_c
                # without requiring an analytic covariance derivation.
                ci_low_rel, ci_high_rel = self._relative_ci_bootstrap(alpha=a)
            else:
                # Safe delta fallback: Cov(tau, mu_c) is not estimated via unsupported hybrid IF.
                var_tau = float(np.asarray(self._result.cov_params(), dtype=float)[1, 1])
                n_control = int(np.sum(control_mask))
                var_mu = (
                    float(np.var(y_internal[control_mask], ddof=1)) / n_control
                    if n_control > 1
                    else np.nan
                )
                d_tau = 100.0 / mu_c
                d_mu = -100.0 * tau / (mu_c ** 2)
                var_rel = (d_tau ** 2) * var_tau + (d_mu ** 2) * var_mu

                if np.isfinite(var_rel):
                    se_rel = float(np.sqrt(max(var_rel, 0.0)))
                    ci_low_rel = float(tau_rel - crit * se_rel)
                    ci_high_rel = float(tau_rel + crit * se_rel)
                    if ci_low_rel > ci_high_rel:
                        ci_low_rel, ci_high_rel = ci_high_rel, ci_low_rel

        diag = None
        if diagnostic_data:
            params_naive = np.asarray(self._result_naive.params, dtype=float)
            bse_naive = np.asarray(self._result_naive.bse, dtype=float)
            ate_naive = float(params_naive[1])
            se_naive = float(bse_naive[1])
            se_adj = float(bse[1])
            if se_naive > 0.0:
                se_red = 1.0 - (se_adj ** 2) / (se_naive ** 2)
                se_red_pct = float(100.0 * se_red)
            else:
                se_red_pct = np.nan

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
                se_reduction_pct_same_cov=float(se_red_pct) if np.isfinite(se_red_pct) else np.nan,
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
                "dropped_covariates": list(self._dropped_covariates),
                "run_regression_checks": self.run_regression_checks,
                "check_action": self.check_action,
                "raise_on_yellow": self.raise_on_yellow,
                "winsor_q": self.winsor_q,
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
            Dictionary with estimates, inference, and diagnostics.
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
            "dropped_covariates": list(self._dropped_covariates),
            "run_regression_checks": self.run_regression_checks,
            "check_action": self.check_action,
            "raise_on_yellow": self.raise_on_yellow,
            "winsor_q": self.winsor_q,
            "ate_naive": diag.ate_naive,
            "se_naive": diag.se_naive,
            "se_reduction_pct_same_cov": diag.se_reduction_pct_same_cov,
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
        """Emit diagnostics signals from GREEN/YELLOW/RED assumption table."""
        if self.check_action == "ignore":
            return

        skip = set(skip_test_ids or set())
        for _, row in table.iterrows():
            test_id = str(row.get("test_id", ""))
            if test_id in skip:
                continue

            flag = str(row.get("flag", FLAG_GREEN)).upper()
            if flag == FLAG_GREEN:
                continue

            test_name = str(row.get("test", "assumption"))
            msg = str(row.get("message", "diagnostic check failed"))
            text = f"{test_name}: {msg}"

            should_raise = flag == FLAG_RED or (self.raise_on_yellow and flag == FLAG_YELLOW)
            if self.check_action == "raise" and should_raise:
                raise ValueError(text)

    def _check_signal(self, msg: str) -> None:
        """Emit exception according to configured diagnostics action."""
        if self.check_action == "ignore":
            return
        raise ValueError(msg)

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

    def _relative_ci_bootstrap(self, alpha: float) -> Tuple[float, float]:
        """
        Compute percentile bootstrap CI for relative effect `100 * tau / mu_c`.

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

        for _ in range(self.relative_ci_bootstrap_draws):
            # Resample units with replacement.
            idx = rng.integers(0, n, size=n)
            df_b = df.iloc[idx]
            d_b = df_b[t_name].to_numpy(dtype=float)
            c_mask = d_b == 0.0
            t_mask = d_b == 1.0
            if not np.any(c_mask) or not np.any(t_mask):
                continue

            y_b = df_b[y_name].astype(float)
            mu_c_b = float(np.mean(y_b.to_numpy(dtype=float)[c_mask]))
            if (not np.isfinite(mu_c_b)) or mu_c_b == 0.0:
                continue

            design_b = pd.DataFrame(
                {"intercept": np.ones(n, dtype=float), t_name: d_b},
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
            rel_b = 100.0 * tau_b / mu_c_b
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
            f"strict_binary_treatment={self.strict_binary_treatment}, "
            f"centering_scope='{self.centering_scope}', "
            f"relative_ci_method='{self.relative_ci_method}', "
            f"covariate_variance_min={self.covariate_variance_min}, "
            f"condition_number_warn_threshold={self.condition_number_warn_threshold}, "
            f"run_regression_checks={self.run_regression_checks}, "
            f"check_action='{self.check_action}', "
            f"raise_on_yellow={self.raise_on_yellow}, "
            f"corr_near_one_tol={self.corr_near_one_tol}, "
            f"vif_warn_threshold={self.vif_warn_threshold}, "
            f"winsor_q={self.winsor_q}, "
            f"tiny_one_minus_h_tol={self.tiny_one_minus_h_tol}, "
            f"use_t={use_t_repr}, use_t_auto_n_threshold={self.use_t_auto_n_threshold}, "
            f"use_t_effective={self._use_t_effective}, status='{status}')"
        )
