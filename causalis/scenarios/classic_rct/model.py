r"""
Difference-in-Means (RCT) scenario.

This module provides the `DiffInMeans` model, which implements standard inference
methods for Randomized Controlled Trials (RCTs). It supports absolute and relative
treatment effect estimation using t-tests, permutation tests, and z-tests for
proportions.
"""

from __future__ import annotations

from typing import Any, Optional, Literal

from causalis.dgp.causaldata import CausalData
from causalis.data_contracts.causal_estimate import CausalEstimate
from causalis.data_contracts.causal_diagnostic_data import DiffInMeansDiagnosticData
from .inference import ttest, welch_permutation_t_test, conversion_ztest


class DiffInMeans:
    r"""
    Difference-in-means model for Randomized Controlled Trials (RCT).

    The difference-in-means estimator is the simplest way to estimate the
    Average Treatment Effect (ATE) in a randomized experiment. Because
    treatment assignment is random, the difference in sample means between
    the treated and control groups is an unbiased estimator of the ATE.

    Notes
    -----
    The Average Treatment Effect (ATE) is defined as:

    .. math::

        \tau = E[Y(1) - Y(0)]

    In an RCT, $D \perp (Y(0), Y(1))$, so:

    .. math::

        \tau = E[Y | D=1] - E[Y | D=0]

    The estimator implemented here is the simple difference in sample means:

    .. math::

        \hat{\tau} = \frac{1}{n_1} \sum_{i: D_i=1} Y_i - \frac{1}{n_0} \sum_{i: D_i=0} Y_i

    Standard errors and confidence intervals are computed using Welch's t-test
    by default, which does not assume equal variances between groups:

    .. math::

        SE(\hat{\tau}) = \sqrt{\frac{s_1^2}{n_1} + \frac{s_0^2}{n_0}}

    where $s_g^2$ is the sample variance in group $g$.

    Examples
    --------
    >>> from causalis.scenarios.classic_rct.dgp import generate_classic_rct_26
    >>> from causalis.scenarios.classic_rct.model import DiffInMeans
    >>> # Generate synthetic RCT data
    >>> data = generate_classic_rct_26(seed=42)
    >>> # Fit the model
    >>> model = DiffInMeans().fit(data)
    >>> # Estimate ATE using standard t-test
    >>> estimate = model.estimate(method="ttest")
    >>> print(f"ATE: {estimate.value:.4f}")
    ATE: 0.0339
    >>> print(f"P-value: {estimate.p_value:.4f}")
    P-value: 0.0000

    Attributes
    ----------
    data : CausalData or None
        The dataset used for fitting and estimation.
    """

    def __init__(self) -> None:
        self.data: Optional[CausalData] = None
        self._is_fitted: bool = False

    def fit(self, data: CausalData) -> DiffInMeans:
        """
        Fit the model by storing the CausalData object.

        In the `DiffInMeans` scenario, "fitting" primarily involves validating
        the data structure and storing it for subsequent estimation.

        Parameters
        ----------
        data : CausalData
            The CausalData object containing treatment and outcome variables.
            Treatment must be binary (0/1).

        Returns
        -------
        DiffInMeans
            The fitted model instance.

        Raises
        ------
        ValueError
            If the input is not a `CausalData` object.
        """
        if not isinstance(data, CausalData):
            raise ValueError("Input must be a CausalData object.")
        self.data = data
        self._is_fitted = True
        return self

    def estimate(
        self,
        method: Literal[
            "ttest", "welch_permutation_t_test", "conversion_ztest"
        ] = "ttest",
        alpha: float = 0.05,
        diagnostic_data: bool = True,
        **kwargs: Any,
    ) -> CausalEstimate:
        r"""
        Compute the treatment effect using the specified method.

        Parameters
        ----------
        method : {"ttest", "welch_permutation_t_test", "conversion_ztest"}, default "ttest"
            The inference method to use:
            - "ttest": Standard Welch's two-sample t-test (handles unequal variances).
            - "welch_permutation_t_test": Permutation-based p-value using the Welch statistic.
            - "conversion_ztest": Z-test for proportions, suitable for binary (0/1) outcomes.
        alpha : float, default 0.05
            Significance level for calculating confidence intervals.
        diagnostic_data : bool, default True
            Whether to include diagnostic data in the result (e.g., covariate balance if confounders exist).
        **kwargs : Any
            Additional arguments passed to the underlying inference function:
            - For "welch_permutation_t_test": `B` (iterations), `alternative`, `seed`.
            - For "conversion_ztest": `ci_method`, `se_for_test`.

        Returns
        -------
        CausalEstimate
            A results object containing:
            - `value`: The estimated ATE (absolute difference).
            - `p_value`: The statistical significance.
            - `ci_lower_absolute`, `ci_upper_absolute`: Confidence interval bounds.
            - `value_relative`: The relative effect (% lift).

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        ValueError
            If an unsupported method is specified.
        """
        if not self._is_fitted or self.data is None:
            raise RuntimeError(
                "Model must be fitted with .fit(data_contracts) before calling .estimate()."
            )

        a = float(alpha)

        if method == "ttest":
            res = ttest(self.data, alpha=a)
        elif method in ["welch_permutation_t_test"]:
            res = welch_permutation_t_test(self.data, alpha=a, **kwargs)
        elif method in ["conversion_ztest"]:
            res = conversion_ztest(self.data, alpha=a, **kwargs)
        else:
            raise ValueError(
                f"Unknown method: {method}. "
                "Supported: 'ttest', 'welch_permutation_t_test', 'conversion_ztest'."
            )

        diag_data = None
        conf = self.data.confounders
        has_conf = conf is not None and (
            len(conf) > 0 if not hasattr(conf, "empty") else not conf.empty
        )
        if diagnostic_data and has_conf:
            diag_data = DiffInMeansDiagnosticData()

        treated_outcome = self.data.outcome[self.data.treatment == 1].dropna()
        control_outcome = self.data.outcome[self.data.treatment == 0].dropna()

        return CausalEstimate(
            estimand="ATE",
            model="DiffInMeans",
            model_options={"method": method, "alpha": a, **kwargs},
            value=res["absolute_difference"],
            ci_upper_absolute=res["absolute_ci"][1],
            ci_lower_absolute=res["absolute_ci"][0],
            value_relative=res.get("relative_difference"),
            ci_upper_relative=(
                res.get("relative_ci")[1]
                if isinstance(res.get("relative_ci"), (tuple, list))
                else None
            ),
            ci_lower_relative=(
                res.get("relative_ci")[0]
                if isinstance(res.get("relative_ci"), (tuple, list))
                else None
            ),
            alpha=a,
            p_value=res["p_value"],
            is_significant=bool(res["p_value"] < a),
            n_treated=int(self.data.outcome[self.data.treatment == 1].shape[0]),
            n_control=int(self.data.outcome[self.data.treatment == 0].shape[0]),
            treatment_mean=float(treated_outcome.mean()),
            control_mean=float(control_outcome.mean()),
            outcome=self.data.outcome_name,
            treatment=self.data.treatment_name,
            confounders=self.data.confounders,
            diagnostic_data=diag_data,
        )

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "unfitted"
        return f"DiffInMeans(status='{status}')"
