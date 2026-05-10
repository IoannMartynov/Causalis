r"""
Data Generating Processes (DGPs) for the CUPED scenario.

This module provides pre-configured synthetic datasets for testing CUPED and
other regression-adjusted estimators. It includes complex, skewed outcomes
(Tweedie) and binary outcomes, both featuring pre-treatment covariates
calibrated to specific target correlations.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, Union
from causalis.dgp.causaldata import CausalData
from causalis.dgp.causaldata.functional import make_cuped_tweedie, generate_cuped_binary
from causalis.dgp.causaldata.preperiod import PreCorrSpec, add_preperiod_covariate, corr_on_scale

_TWEEDIE_PRE_RESERVED_NAMES = {
    "y",
    "d",
    "tenure_months",
    "avg_sessions_week",
    "spend_last_month",
    "discount_rate",
    "platform_ios",
    "platform_web",
    "m",
    "m_obs",
    "tau_link",
    "g0",
    "g1",
    "cate",
    "user_id",
    "_latent_A",
}


def _clone_pre_spec_with_target(
    spec: Optional[PreCorrSpec],
    target_corr: float
) -> PreCorrSpec:
    if spec is None:
        return PreCorrSpec(target_corr=target_corr)
    return PreCorrSpec(
        target_corr=target_corr,
        transform=spec.transform,
        winsor_q=spec.winsor_q,
        method=spec.method,
        sigma_lo=spec.sigma_lo,
        sigma_hi=spec.sigma_hi,
        sigma_tol=spec.sigma_tol,
        max_iter=spec.max_iter,
    )


def _resolve_second_pre_target(base_target: float, explicit_target: Optional[float]) -> float:
    if explicit_target is not None:
        return float(explicit_target)
    # Default to a slightly weaker second pre signal to reduce collinearity.
    return float(max(0.0, min(base_target, min(0.72, base_target - 0.10))))


def _pre_transform(x: np.ndarray, transform: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    if transform == "none":
        return arr
    if transform == "log1p":
        return np.log1p(np.clip(arr, 0.0, None))
    if transform == "rank":
        return pd.Series(arr).rank(method="average").to_numpy(dtype=float)
    raise ValueError("transform must be one of: 'none', 'log1p', 'rank'.")


def generate_cuped_tweedie_26(
    n: int = 20000,
    seed: int = 42,
    add_pre: bool = True,
    pre_name: str = "y_pre",
    pre_name_2: Optional[str] = None,
    pre_target_corr: float = 0.82,
    pre_target_corr_2: Optional[float] = None,
    pre_spec: Optional[PreCorrSpec] = None,
    include_oracle: bool = False,
    return_causal_data: bool = True,
    theta_log: float = 0.38
) -> Union[pd.DataFrame, CausalData]:
    r"""
    Gold standard Tweedie-like DGP with mixed marginals and structured HTE.

    This DGP generates a dataset representing a typical e-commerce scenario with
    many zeros and a heavy right tail (e.g., revenue). It features pre-treatment
    covariates ('y_pre', 'y_pre_2') that are calibrated to have a specific
    correlation with the post-treatment outcome.

    Notes
    -----
    The outcome $Y$ follows a Tweedie-like distribution where the mean is a
    function of treatment $D$ and latent factors $A$ and $X$:

    .. math::

        E[Y | D, A, X] = \exp(\alpha + \theta D + \beta X + \gamma D X + A)

    The pre-period covariate $X_{pre}$ (e.g., `y_pre`) is generated to satisfy:

    .. math::

        Corr(X_{pre}, Y | D=0) \approx \rho

    where $\rho$ is the `pre_target_corr`. This allows for realistic testing of
    variance reduction techniques.

    Examples
    --------
    >>> from causalis.scenarios.cuped.dgp import generate_cuped_tweedie_26
    >>> # Generate data with two pre-period covariates
    >>> data = generate_cuped_tweedie_26(n=1000, seed=42, return_causal_data=False)
    >>> print(data.columns.tolist())
    ['y', 'd', 'tenure_months', 'avg_sessions_week', 'spend_last_month', 'discount_rate', 'platform_ios', 'platform_web', 'y_pre', 'y_pre_2']

    Parameters
    ----------
    n : int, default=10000
        Number of samples to generate.
    seed : int, default=42
        Random seed.
    add_pre : bool, default=True
        Whether to add pre-period covariates.
    pre_name : str, default="y_pre"
        Name of the first pre-period covariate column.
    pre_name_2 : str, optional
        Name of the second pre-period covariate column.
        Defaults to `f"{pre_name}_2"`.
    pre_target_corr : float, default=0.82
        Target correlation between the first pre covariate and post-outcome y in control group.
    pre_target_corr_2 : float, optional
        Target correlation for the second pre covariate. Defaults to a
        moderate value based on `pre_target_corr` to reduce collinearity.
    pre_spec : PreCorrSpec, optional
        Detailed specification for pre-period calibration (transform, method, etc.).
    include_oracle : bool, default=False
        Whether to include oracle ground-truth columns like 'cate', 'propensity', etc.
    return_causal_data : bool, default=True
        Whether to return a CausalData object.
    theta_log : float, default=0.38
        The log-uplift theta parameter for the treatment effect.

    Returns
    -------
    pd.DataFrame or CausalData
    """
    if not add_pre:
        return make_cuped_tweedie(
            n=n,
            seed=seed,
            add_pre=False,
            pre_name=pre_name,
            pre_target_corr=pre_target_corr,
            pre_spec=pre_spec,
            include_oracle=include_oracle,
            return_causal_data=return_causal_data,
            theta_log=theta_log
        )

    second_pre_name = pre_name_2 or f"{pre_name}_2"
    if second_pre_name == pre_name:
        raise ValueError("pre_name_2 must be different from pre_name.")
    for arg_name, col_name in (("pre_name", pre_name), ("pre_name_2", second_pre_name)):
        if col_name in _TWEEDIE_PRE_RESERVED_NAMES:
            raise ValueError(
                f"{arg_name}='{col_name}' collides with an existing generated column; "
                "choose a different pre-period column name."
            )

    # Build with oracle columns to access the shared latent driver used by y and y_pre.
    # We remove oracle columns later if include_oracle=False.
    df = make_cuped_tweedie(
        n=n,
        seed=seed,
        add_pre=True,
        pre_name=pre_name,
        pre_target_corr=pre_target_corr,
        pre_spec=pre_spec,
        include_oracle=True,
        return_causal_data=False,
        theta_log=theta_log
    )

    base_target = pre_spec.target_corr if pre_spec is not None else pre_target_corr
    second_target = _resolve_second_pre_target(base_target, pre_target_corr_2)
    spec_2 = _clone_pre_spec_with_target(pre_spec, target_corr=second_target)

    if "_latent_A" not in df.columns:
        raise RuntimeError(
            "Internal error: make_cuped_tweedie did not expose '_latent_A' required "
            "to build the second pre-period covariate."
        )

    rng = np.random.default_rng(seed + 2602)

    def _z(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1)
        mu = float(x.mean())
        sd = float(x.std())
        if sd < 1e-12:
            return x - mu
        return (x - mu) / sd

    y_post = df["y"].to_numpy(dtype=float)
    ctrl = (df["d"].to_numpy(dtype=float) == 0)
    latent_a = df["_latent_A"].to_numpy(dtype=float)
    tenure = _z(np.log1p(df["tenure_months"].to_numpy(dtype=float)))
    spend = _z(np.log1p(df["spend_last_month"].to_numpy(dtype=float)))
    sessions = _z(np.log1p(df["avg_sessions_week"].to_numpy(dtype=float)))
    discount = _z(df["discount_rate"].to_numpy(dtype=float))
    a = _z(latent_a)

    # A second noisy measurement of the same latent driver (A), with slightly
    # different nonlinear structure than y_pre.
    zi_score = 0.20 + 1.05 * a + 0.10 * tenure - 0.08 * discount
    p_pos_2 = 1.0 / (1.0 + np.exp(-np.clip(zi_score, -20.0, 20.0)))
    is_pos_2 = rng.binomial(1, p_pos_2, size=n).astype(float)

    loc_2 = 1.90 + 0.95 * a + 0.10 * sessions + 0.08 * spend
    mu_2 = np.exp(np.clip(loc_2, -20.0, 20.0))
    shape_2 = 2.3
    pre2_base = is_pos_2 * rng.gamma(shape=shape_2, scale=mu_2 / shape_2, size=n)

    c0_latent = corr_on_scale(
        pre2_base[ctrl],
        y_post[ctrl],
        transform=spec_2.transform,
        winsor_q=spec_2.winsor_q,
        method=spec_2.method
    )

    # If latent-only signal cannot hit the requested target, blend with y_pre
    # just enough to make the target feasible before sigma calibration.
    if c0_latent < second_target:
        pre1 = df[pre_name].to_numpy(dtype=float)
        best_corr = c0_latent
        best_base = pre2_base
        for w_pre1 in np.linspace(0.10, 1.00, 19):
            candidate = (1.0 - w_pre1) * pre2_base + w_pre1 * pre1
            cand_corr = corr_on_scale(
                candidate[ctrl],
                y_post[ctrl],
                transform=spec_2.transform,
                winsor_q=spec_2.winsor_q,
                method=spec_2.method
            )
            if cand_corr > best_corr:
                best_corr = cand_corr
                best_base = candidate
            if cand_corr >= second_target:
                best_base = candidate
                break
        pre2_base = best_base

    def base_builder(df_in: pd.DataFrame) -> np.ndarray:
        _ = df_in  # keep signature; pre2_base is precomputed for reproducibility
        return np.asarray(pre2_base, dtype=float)

    df = add_preperiod_covariate(
        df=df,
        y_col="y",
        d_col="d",
        pre_name=second_pre_name,
        base_builder=base_builder,
        spec=spec_2,
        rng=rng
    )

    if not include_oracle:
        drop_cols = [c for c in ("m", "m_obs", "tau_link", "g0", "g1", "cate", "_latent_A") if c in df.columns]
        if drop_cols:
            df = df.drop(columns=drop_cols)

    if not return_causal_data:
        return df

    exclude = {"y", "d", "m", "m_obs", "tau_link", "g0", "g1", "cate", "user_id", "_latent_A"}
    confounder_cols = [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]
    return CausalData(
        df=df,
        treatment="d",
        outcome="y",
        confounders=confounder_cols
    )


def make_cuped_binary_26(
    n: int = 10000,
    seed: int = 42,
    add_pre: bool = True,
    pre_name: str = "y_pre",
    pre_target_corr: float = 0.65,
    pre_spec: Optional[PreCorrSpec] = None,
    include_oracle: bool = True,
    return_causal_data: bool = True,
    theta_logit: float = 0.38
) -> Union[pd.DataFrame, CausalData]:
    r"""
    Binary CUPED benchmark with richer confounders and structured HTE.

    This DGP generates a binary outcome (e.g., conversion) with a calibrated
    pre-period covariate.

    Notes
    -----
    The outcome $Y$ is binary and follows:

    .. math::

        P(Y=1 | D, X) = \text{logit}^{-1}(\alpha + \theta D + \beta X)

    The pre-period covariate is calibrated such that its correlation with $Y$
    in the control group matches `pre_target_corr`.

    Examples
    --------
    >>> from causalis.scenarios.cuped.dgp import make_cuped_binary_26
    >>> data = make_cuped_binary_26(n=1000, seed=42, return_causal_data=False)
    >>> print(data.columns.tolist()[:7])
    ['y', 'd', 'tenure_months', 'spend_last_month', 'discount_rate', 'support_tickets', 'email_open_rate']

    Parameters
    ----------
    n : int, default=10000
        Number of samples to generate.
    seed : int, default=42
        Random seed.
    add_pre : bool, default=True
        Whether to add a pre-period covariate 'y_pre'.
    pre_name : str, default="y_pre"
        Name of the pre-period covariate column.
    pre_target_corr : float, default=0.65
        Target correlation between y_pre and post-outcome y in the control group.
    pre_spec : PreCorrSpec, optional
        Detailed specification for pre-period calibration (transform, method, etc.).
    include_oracle : bool, default=True
        Whether to include oracle columns like 'cate', 'g0', and 'g1'.
    return_causal_data : bool, default=True
        Whether to return a CausalData object.
    theta_logit : float, default=0.38
        Baseline log-odds uplift scale for heterogeneous treatment effects.

    Returns
    -------
    pd.DataFrame or CausalData
    """
    return generate_cuped_binary(
        n=n,
        seed=seed,
        add_pre=add_pre,
        pre_name=pre_name,
        pre_target_corr=pre_target_corr,
        pre_spec=pre_spec,
        include_oracle=include_oracle,
        return_causal_data=return_causal_data,
        theta_logit=theta_logit
    )
