"""
T-test inference for Diff_in_Means model
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any

from causalis.data_contracts import CausalData


def ttest(data: CausalData, alpha: float = 0.05) -> Dict[str, Any]:
    r"""
    Perform a Welch two-sample t-test comparing outcomes between groups.

    The Welch t-test (also known as the unequal variances t-test) is used to
    test the hypothesis that two populations have equal means. It is more
    robust than Student's t-test when the two samples have unequal variances
    and/or unequal sample sizes.

    Notes
    -----
    The Welch t-statistic is calculated as:

    .. math::

        t = \frac{\bar{Y}_1 - \bar{Y}_0}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_0^2}{n_0}}}

    The degrees of freedom $\nu$ are approximated using the Welch-Satterthwaite
    equation:

    .. math::

        \nu \approx \frac{\left(\frac{s_1^2}{n_1} + \frac{s_0^2}{n_0}\right)^2}
        {\frac{(s_1^2/n_1)^2}{n_1-1} + \frac{(s_0^2/n_0)^2}{n_0-1}}

    For the relative difference (percent lift), the variance is estimated
    using the Delta method:

    .. math::

        Var\left(\frac{\bar{Y}_1}{\bar{Y}_0}\right) \approx
        \frac{1}{\bar{Y}_0^2} Var(\bar{Y}_1) + \frac{\bar{Y}_1^2}{\bar{Y}_0^4} Var(\bar{Y}_0)

    Examples
    --------
    >>> from causalis.scenarios.classic_rct.dgp import generate_classic_rct_26
    >>> from causalis.scenarios.classic_rct.inference.ttest import ttest
    >>> data = generate_classic_rct_26(n=2000, seed=42)
    >>> results = ttest(data)
    >>> print(f"P-value: {results['p_value']:.4f}")
    0.1685

    Parameters
    ----------
    data : CausalData
        The CausalData object containing treatment and outcome variables.
    alpha : float, default 0.05
        The significance level for calculating confidence intervals.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing:
        - `p_value`: Welch t-test p-value.
        - `absolute_difference`: $\bar{Y}_1 - \bar{Y}_0$.
        - `absolute_ci`: (lower, upper) CI for absolute difference.
        - `relative_difference`: Percent change relative to control.
        - `relative_se`: Delta-method standard error for the relative difference.
        - `relative_ci`: (lower, upper) CI for relative difference.
    """
    treatment_var = data.treatment
    outcome_var = data.outcome

    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1 (exclusive)")

    # Ensure binary treatment coded exactly as {0,1}
    vals = set(pd.unique(treatment_var.dropna()))
    if vals != {0, 1}:
        raise ValueError("Treatment variable must be coded as {0,1} (exactly).")

    # Build groups and drop missing outcomes
    control_data = outcome_var[treatment_var == 0].dropna()
    treatment_data = outcome_var[treatment_var == 1].dropna()

    n1 = int(len(treatment_data))
    n2 = int(len(control_data))
    if n1 < 2 or n2 < 2:
        raise ValueError("Not enough observations in one of the groups for t-test (need at least 2 per group)")

    # Welch t-test
    res = stats.ttest_ind(treatment_data, control_data, equal_var=False, nan_policy="raise")
    p_value = float(res.pvalue)

    # Means and variances
    control_mean = float(control_data.mean())
    treatment_mean = float(treatment_data.mean())
    s1_squared = float(treatment_data.var(ddof=1))
    s0_squared = float(control_data.var(ddof=1))

    # Absolute difference
    absolute_diff = treatment_mean - control_mean
    se_diff = float(np.sqrt(s1_squared / n1 + s0_squared / n2))

    # Welch df for absolute CI (use SciPy df if present; else compute)
    if hasattr(res, "df") and res.df is not None:
        df_abs = float(res.df)
    else:
        v1_abs = s1_squared / n1
        v0_abs = s0_squared / n2
        denom_abs = (v1_abs**2) / (n1 - 1) + (v0_abs**2) / (n2 - 1)
        df_abs = float(((v1_abs + v0_abs) ** 2) / denom_abs) if denom_abs > 0 else np.nan

    # Selects t or normal critical value based on degrees of freedom
    if np.isfinite(df_abs) and df_abs > 0:
        tcrit_abs = float(stats.t.ppf(1 - alpha / 2, df_abs))
    else:
        tcrit_abs = float(stats.norm.ppf(1 - alpha / 2))

    moe_abs = tcrit_abs * se_diff
    absolute_ci = (absolute_diff - moe_abs, absolute_diff + moe_abs)

    # Signed relative percent change via delta method
    # Guard against instability when control_mean is near 0.
    # Use a scale based on the outcome magnitude to define "near 0" robustly.
    outcome_scale = float(np.nanmean(np.abs(outcome_var.dropna()))) if not outcome_var.dropna().empty else 1.0
    eps = 1e-12 * max(1.0, outcome_scale)

    # Computes relative difference and confidence interval; guards instability
    if (not np.isfinite(control_mean)) or abs(control_mean) < eps:
        relative_diff = np.inf if absolute_diff > 0 else -np.inf if absolute_diff < 0 else 0.0
        relative_se = np.nan
        relative_ci = (np.nan, np.nan)
    else:
        relative_diff = (treatment_mean / control_mean - 1.0) * 100.0

        # Delta-method variance for r_hat/100
        v1 = s1_squared / n1
        v0 = s0_squared / n2
        w1 = (1.0 / control_mean) ** 2
        w0 = (treatment_mean / (control_mean ** 2)) ** 2

        var_rel_scaled = float(max(w1 * v1 + w0 * v0, 0.0))  # numeric guard
        relative_se = 100.0 * float(np.sqrt(var_rel_scaled))

        # Satterthwaite df for weighted sum variance
        denom_rel = (w1 * v1) ** 2 / (n1 - 1) + (w0 * v0) ** 2 / (n2 - 1)
        df_rel = float(((w1 * v1 + w0 * v0) ** 2) / denom_rel) if denom_rel > 0 else np.nan

        # Computes t‑critical value based on degrees of freedom
        if np.isfinite(df_rel) and df_rel > 0:
            tcrit_rel = float(stats.t.ppf(1 - alpha / 2, df_rel))
        else:
            tcrit_rel = float(stats.norm.ppf(1 - alpha / 2))

        moe_rel = tcrit_rel * relative_se
        relative_ci = (relative_diff - moe_rel, relative_diff + moe_rel)

    return {
        "p_value": float(p_value),
        "absolute_difference": float(absolute_diff),
        "absolute_ci": (float(absolute_ci[0]), float(absolute_ci[1])),
        "relative_difference": float(relative_diff),
        "relative_se": float(relative_se) if np.isfinite(relative_se) else relative_se,
        "relative_ci": (float(relative_ci[0]), float(relative_ci[1])),
    }
