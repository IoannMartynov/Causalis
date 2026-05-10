"""
Two-proportion z-test

Compares conversion rates between treated (D=1) and control (D=0) groups.
Returns p-value, absolute/relative differences, and their confidence intervals
"""

from typing import Dict, Any, Literal, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from causalis.dgp.causaldata import CausalData


def conversion_ztest(
    data: CausalData,
    alpha: float = 0.05,
    ci_method: Literal["newcombe", "wald_unpooled", "wald_pooled"] = "newcombe",
    se_for_test: Literal["pooled", "unpooled"] = "pooled",
) -> Dict[str, Any]:
    r"""
    Perform a two-proportion z-test on a CausalData object with a binary outcome.

    The z-test for proportions is used to compare the conversion rates of two
    independent groups. It assumes that the number of successes and failures
    in each group is sufficiently large (typically $n \cdot p > 5$ and
    $n \cdot (1-p) > 5$).

    Notes
    -----
    The z-statistic for testing $H_0: p_1 = p_0$ is calculated as:

    .. math::

        z = \frac{\hat{p}_1 - \hat{p}_0}{SE}

    By default (`se_for_test="pooled"`), the pooled standard error is used:

    .. math::

        SE_{pooled} = \sqrt{\hat{p}(1-\hat{p})\left(\frac{1}{n_1} + \frac{1}{n_0}\right)}

    where $\hat{p} = \frac{x_1 + x_0}{n_1 + n_0}$ is the pooled proportion.

    Confidence intervals for the difference $p_1 - p_0$ can be calculated using
    several methods. The "newcombe" method (Newcombe's hybrid score interval)
    is generally recommended as it performs better than the Wald interval
    when proportions are near 0 or 1.

    Examples
    --------
    >>> from causalis.scenarios.classic_rct.dgp import generate_classic_rct_26
    >>> from causalis.scenarios.classic_rct.inference.conversion_ztest import conversion_ztest
    >>> data = generate_classic_rct_26(n=2000, seed=42)
    >>> results = conversion_ztest(data)
    >>> print(f"Conversion Rate (Control): {data.df[data.df['d']==0]['conversion'].mean():.4f}")
    0.1349
    >>> print(f"P-value: {results['p_value']:.4f}")
    0.1688

    Parameters
    ----------
    data : CausalData
        The CausalData object containing treatment and outcome variables.
    alpha : float, default 0.05
        The significance level for calculating confidence intervals.
    ci_method : {"newcombe", "wald_unpooled", "wald_pooled"}, default "newcombe"
        Method for calculating the confidence interval for the absolute difference.
    se_for_test : {"pooled", "unpooled"}, default "pooled"
        Method for calculating the standard error for the z-test p-value.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing:
        - `p_value`: Two-sided p-value from the z-test.
        - `absolute_difference`: Difference in conversion rates ($p_1 - p_0$).
        - `absolute_ci`: (lower, upper) CI for the absolute difference.
        - `relative_difference`: Percentage change relative to control.
        - `relative_ci`: (lower, upper) CI for the relative difference (delta method).
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1 (exclusive)")

    treatment_var = data.treatment
    outcome_var = data.outcome

    if not isinstance(treatment_var, pd.Series) or treatment_var.empty:
        raise ValueError("CausalData must have a non-empty treatment Series")
    if not isinstance(outcome_var, pd.Series) or outcome_var.empty:
        raise ValueError("CausalData must have a non-empty outcome Series")

    # Pairwise drop missing (prevents denominator/numerator mismatch)
    df = pd.concat(
        [treatment_var.rename("D"), outcome_var.rename("Y")],
        axis=1
    ).dropna()

    if df.empty:
        raise ValueError("No non-missing (treatment, outcome) pairs available")

    # Strict 0/1 validation
    d_set = set(pd.unique(df["D"]))
    y_set = set(pd.unique(df["Y"]))

    # allow bools (True/False) since they are 1/0 in Python
    if not d_set.issubset({0, 1, False, True}):
        raise ValueError("Treatment must be binary coded as 0/1 (or False/True)")
    if not y_set.issubset({0, 1, False, True}):
        raise ValueError("Outcome must be binary coded as 0/1 (or False/True)")

    # Convert to int 0/1 to avoid surprises
    df["D"] = df["D"].astype(int)
    df["Y"] = df["Y"].astype(int)

    if set(pd.unique(df["D"])) != {0, 1}:
        raise ValueError("Treatment must contain both 0 and 1")

    control = df.loc[df["D"] == 0, "Y"]
    treat = df.loc[df["D"] == 1, "Y"]

    n0 = int(control.shape[0])
    n1 = int(treat.shape[0])
    if n0 < 1 or n1 < 1:
        raise ValueError("Need at least 1 observation per group")

    x0 = float(control.sum())
    x1 = float(treat.sum())

    p0 = x0 / n0
    p1 = x1 / n1
    absolute_diff = float(p1 - p0)

    z_crit = float(stats.norm.ppf(1 - alpha / 2))

    # 1) p-value (two-sided)
    if se_for_test == "pooled":
        p_pool = (x0 + x1) / (n0 + n1)
        se_test = float(np.sqrt(p_pool * (1 - p_pool) * (1 / n0 + 1 / n1)))
    else:
        se_test = float(np.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0))

    if se_test == 0.0:
        p_value = 1.0 if absolute_diff == 0.0 else 0.0
    else:
        z_stat = float(absolute_diff / se_test)
        p_value = float(2 * (1 - stats.norm.cdf(abs(z_stat))))

    # 2) Absolute CI
    def wilson_ci(p: float, n: int, z: float) -> Tuple[float, float]:
        den = 1.0 + (z**2) / n
        center = (p + (z**2) / (2 * n)) / den
        half = (z * np.sqrt(p * (1 - p) / n + (z**2) / (4 * n**2))) / den
        return float(center - half), float(center + half)

    if ci_method == "newcombe":
        l0, u0 = wilson_ci(p0, n0, z_crit)
        l1, u1 = wilson_ci(p1, n1, z_crit)
        absolute_ci = (float(l1 - u0), float(u1 - l0))
    elif ci_method == "wald_pooled":
        p_pool = (x0 + x1) / (n0 + n1)
        se_ci = float(np.sqrt(p_pool * (1 - p_pool) * (1 / n0 + 1 / n1)))
        margin = z_crit * se_ci
        absolute_ci = (absolute_diff - margin, absolute_diff + margin)
    else:  # wald_unpooled
        se_ci = float(np.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0))
        margin = z_crit * se_ci
        absolute_ci = (absolute_diff - margin, absolute_diff + margin)

    # 3) Relative effect (% lift) and CI via delta method
    # lift = (p1/p0 - 1) * 100
    eps = 1e-12
    if (not np.isfinite(p0)) or abs(p0) < eps:
        relative_diff = np.inf if p1 > 0 else 0.0
        relative_ci = (np.nan, np.nan)
    else:
        rr = p1 / p0
        relative_diff = float((rr - 1.0) * 100.0)

        v1 = p1 * (1 - p1) / n1
        v0 = p0 * (1 - p0) / n0
        w1 = (1.0 / p0) ** 2
        w0 = (p1 / (p0 ** 2)) ** 2
        var_rel_scaled = float(max(w1 * v1 + w0 * v0, 0.0))
        relative_se = 100.0 * float(np.sqrt(var_rel_scaled))
        moe = z_crit * relative_se
        relative_ci = (relative_diff - moe, relative_diff + moe)

    return {
        "p_value": float(p_value),
        "absolute_difference": float(absolute_diff),
        "absolute_ci": (float(absolute_ci[0]), float(absolute_ci[1])),
        "relative_difference": float(relative_diff),
        "relative_ci": (float(relative_ci[0]), float(relative_ci[1])),
    }
