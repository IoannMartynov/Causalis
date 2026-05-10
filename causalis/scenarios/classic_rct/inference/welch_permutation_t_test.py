"""
Welch permutation t-test inference for the classic RCT scenario.

The p-value is computed from a permutation distribution of the Welch
studentized statistic. Confidence intervals use the usual Welch-Satterthwaite
approximation so the result can still populate the shared CausalEstimate
contract.
"""

from typing import Any, Dict, Literal, Optional

import numpy as np
import pandas as pd
from scipy import stats

from causalis.dgp.causaldata import CausalData


Alternative = Literal["two-sided", "greater", "less"]


def _welch_t_stat(x: np.ndarray, y: np.ndarray) -> float:
    nx = int(x.shape[0])
    ny = int(y.shape[0])
    mean_x = float(np.mean(x))
    mean_y = float(np.mean(y))
    var_x = float(np.var(x, ddof=1))
    var_y = float(np.var(y, ddof=1))

    se = float(np.sqrt(var_x / nx + var_y / ny))
    diff = mean_x - mean_y
    if se == 0:
        if diff == 0:
            return 0.0
        return float(np.inf if diff > 0 else -np.inf)
    return float(diff / se)


def _welch_df(var_x: float, nx: int, var_y: float, ny: int) -> float:
    vx = var_x / nx
    vy = var_y / ny
    denom = (vx**2) / (nx - 1) + (vy**2) / (ny - 1)
    if denom <= 0:
        return float("nan")
    return float(((vx + vy) ** 2) / denom)


def _relative_effect_and_ci(
    treated_mean: float,
    control_mean: float,
    treated_var: float,
    control_var: float,
    n_treated: int,
    n_control: int,
    alpha: float,
    outcome_scale: float,
) -> tuple[float, tuple[float, float]]:
    absolute_diff = treated_mean - control_mean
    eps = 1e-12 * max(1.0, outcome_scale)

    if (not np.isfinite(control_mean)) or abs(control_mean) < eps:
        relative_diff = (
            np.inf if absolute_diff > 0 else -np.inf if absolute_diff < 0 else 0.0
        )
        return float(relative_diff), (float("nan"), float("nan"))

    relative_diff = (treated_mean / control_mean - 1.0) * 100.0
    vt = treated_var / n_treated
    vc = control_var / n_control
    wt = (1.0 / control_mean) ** 2
    wc = (treated_mean / (control_mean**2)) ** 2
    var_rel_scaled = float(max(wt * vt + wc * vc, 0.0))
    relative_se = 100.0 * float(np.sqrt(var_rel_scaled))

    denom = (wt * vt) ** 2 / (n_treated - 1) + (wc * vc) ** 2 / (n_control - 1)
    df_rel = float(((wt * vt + wc * vc) ** 2) / denom) if denom > 0 else float("nan")
    if np.isfinite(df_rel) and df_rel > 0:
        crit = float(stats.t.ppf(1 - alpha / 2, df_rel))
    else:
        crit = float(stats.norm.ppf(1 - alpha / 2))

    moe = crit * relative_se
    return float(relative_diff), (
        float(relative_diff - moe),
        float(relative_diff + moe),
    )


def welch_permutation_t_test(
    data: CausalData,
    alpha: float = 0.05,
    B: int = 10_000,
    alternative: Alternative = "two-sided",
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    r"""
    Run a Welch permutation t-test comparing treated and control outcomes.

    This test uses the Welch t-statistic as the test statistic but calculates
    the p-value using a permutation distribution rather than the t-distribution.
    This is useful when the normality assumption of the t-test is suspect,
    while still being robust to unequal variances between groups.

    Notes
    -----
    The permutation p-value is calculated as:

    .. math::

        p = \frac{\sum_{b=1}^B I(|t^*_b| \ge |t_{obs}|) + 1}{B + 1}

    where $t_{obs}$ is the observed Welch t-statistic and $t^*_b$ are the
    t-statistics calculated from $B$ random permutations of the treatment labels.
    The addition of 1 in the numerator and denominator is a standard correction
    to ensure the test is valid (never returning a p-value of exactly 0).

    Confidence intervals for the absolute and relative differences are still
    calculated using the Welch-Satterthwaite and Delta method approximations
    respectively, to remain consistent with other inference methods.

    Examples
    --------
    >>> from causalis.scenarios.classic_rct.dgp import generate_classic_rct_26
    >>> from causalis.scenarios.classic_rct.inference.welch_permutation_t_test import welch_permutation_t_test
    >>> data = generate_classic_rct_26(n=1000, seed=42)
    >>> results = welch_permutation_t_test(data, B=1000, seed=42)
    >>> print(f"Permutation P-value: {results['p_value']:.4f}")
    0.2607

    Parameters
    ----------
    data : CausalData
        The CausalData object containing treatment and outcome variables.
    alpha : float, default 0.05
        Significance level for the theoretical Welch confidence interval.
    B : int, default 10000
        Number of Monte Carlo label permutations.
    alternative : {"two-sided", "greater", "less"}, default "two-sided"
        Alternative hypothesis for the permutation p-value.
    seed : int, optional
        Random seed for reproducible permutations.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing the permutation p-value, observed Welch
        statistic, absolute and relative differences, confidence intervals,
        number of permutations, and alternative.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1 (exclusive)")
    if not isinstance(B, int) or B <= 0:
        raise ValueError("B must be a positive integer")
    if alternative not in {"two-sided", "greater", "less"}:
        raise ValueError("alternative must be one of: 'two-sided', 'greater', 'less'")

    treatment = data.treatment
    outcome = data.outcome

    if not isinstance(treatment, pd.Series) or treatment.empty:
        raise ValueError("causaldata object must have a treatment variable defined")
    if not isinstance(outcome, pd.Series) or outcome.empty:
        raise ValueError("causaldata object must have a outcome variable defined")

    vals = set(pd.unique(treatment.dropna()))
    if vals != {0, 1}:
        raise ValueError("Treatment variable must be coded as {0,1} (exactly).")

    control = np.asarray(outcome[treatment == 0].dropna(), dtype=float)
    treated = np.asarray(outcome[treatment == 1].dropna(), dtype=float)
    control = control[np.isfinite(control)]
    treated = treated[np.isfinite(treated)]

    n_control = int(control.shape[0])
    n_treated = int(treated.shape[0])
    if n_control < 2 or n_treated < 2:
        raise ValueError("Need at least 2 finite outcome observations per group")

    control_mean = float(np.mean(control))
    treated_mean = float(np.mean(treated))
    control_var = float(np.var(control, ddof=1))
    treated_var = float(np.var(treated, ddof=1))

    absolute_diff = treated_mean - control_mean
    se_diff = float(np.sqrt(treated_var / n_treated + control_var / n_control))
    df_abs = _welch_df(treated_var, n_treated, control_var, n_control)
    if np.isfinite(df_abs) and df_abs > 0:
        crit_abs = float(stats.t.ppf(1 - alpha / 2, df_abs))
    else:
        crit_abs = float(stats.norm.ppf(1 - alpha / 2))
    moe_abs = crit_abs * se_diff
    absolute_ci = (float(absolute_diff - moe_abs), float(absolute_diff + moe_abs))

    t_obs = _welch_t_stat(treated, control)
    z = np.concatenate([treated, control])
    rng = np.random.default_rng(seed)

    extreme_count = 0
    for _ in range(B):
        z_perm = rng.permutation(z)
        treated_perm = z_perm[:n_treated]
        control_perm = z_perm[n_treated:]
        t_perm = _welch_t_stat(treated_perm, control_perm)

        if alternative == "two-sided":
            extreme_count += int(abs(t_perm) >= abs(t_obs))
        elif alternative == "greater":
            extreme_count += int(t_perm >= t_obs)
        else:
            extreme_count += int(t_perm <= t_obs)

    p_value = (extreme_count + 1) / (B + 1)

    outcome_scale = float(np.mean(np.abs(np.concatenate([control, treated]))))
    relative_diff, relative_ci = _relative_effect_and_ci(
        treated_mean=treated_mean,
        control_mean=control_mean,
        treated_var=treated_var,
        control_var=control_var,
        n_treated=n_treated,
        n_control=n_control,
        alpha=alpha,
        outcome_scale=outcome_scale,
    )

    return {
        "p_value": float(p_value),
        "t_obs": float(t_obs),
        "absolute_difference": float(absolute_diff),
        "absolute_ci": (float(absolute_ci[0]), float(absolute_ci[1])),
        "relative_difference": float(relative_diff),
        "relative_ci": (float(relative_ci[0]), float(relative_ci[1])),
        "B": B,
        "alternative": alternative,
    }
