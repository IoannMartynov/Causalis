r"""
Data Generating Processes (DGPs) for the Classic RCT scenario.

This module provides pre-configured synthetic datasets representing Randomized
Controlled Trials (RCTs). It includes common experimental setups such as
binary outcomes (conversions) and continuous/skewed outcomes (revenue).
"""

from __future__ import annotations
import uuid
import pandas as pd
import numpy as np
from typing import Optional, List, Union, Dict
from causalis.dgp.causaldata import CausalData
from causalis.dgp.causaldata.functional import generate_classic_rct, classic_rct_gamma
from causalis.dgp.base import _deterministic_ids

def generate_classic_rct_26(
    seed: int = 42,
    add_pre: bool = False,
    beta_y: Optional[Union[List[float], np.ndarray]] = None,
    outcome_depends_on_x: bool = True,
    include_oracle: bool = False,
    return_causal_data: bool = True,
    *,
    n: int = 10000,
    split: float = 0.5,
    outcome_params: Optional[Dict] = None,
    add_ancillary: bool = False,
    deterministic_ids: bool = True,
    **kwargs
):
    r"""
    A pre-configured classic RCT dataset with a binary (conversion) outcome.

    The dataset includes three binary confounders: `platform_ios`,
    `country_usa`, and `source_paid`. Treatment assignment is completely
    random and independent of these confounders.

    Notes
    -----
    The outcome $Y$ (conversion) is generated using a logistic link function:

    .. math::

        P(Y=1 | D, X) = \text{logit}^{-1}(\alpha_y + \theta D + \beta_y X)

    where:
    - $D$ is the binary treatment indicator ($D=1$ for treatment, $D=0$ for control).
    - $X$ are the confounders (platform, country, source).
    - $\alpha_y$ is the baseline log-odds of conversion.
    - $\theta$ is the treatment effect on the log-odds scale.
    - $\beta_y$ are the coefficients for the confounders.

    The default parameters set the baseline control rate to ~10% and the
    treatment rate to ~11% (marginal rates may vary due to $X$).

    Examples
    --------
    >>> from causalis.scenarios.classic_rct.dgp import generate_classic_rct_26
    >>> data = generate_classic_rct_26(n=5000, seed=42)
    >>> print(data.df["conversion"].mean())  # Approximately 0.22
    0.217
    >>> print(data.df.columns.tolist())
    ['user_id', 'conversion', 'd', 'platform_ios', 'country_usa', 'source_paid']

    Parameters
    ----------
    seed : int, default 42
        Random seed for reproducibility.
    add_pre : bool, default False
        Whether to generate a pre-period covariate ('y_pre') and include prognostic signal from X.
    beta_y : array-like, optional
        Linear coefficients for confounders in the outcome model. Default is [0.6, 0.4, 0.8].
    outcome_depends_on_x : bool, default True
        Whether to add default effects for confounders if beta_y is None.
    include_oracle : bool, default False
        Whether to include oracle ground-truth columns like 'cate', 'propensity', etc.
    return_causal_data : bool, default True
        Whether to return a `CausalData` object or a `pd.DataFrame`.
    n : int, default 10000
        Number of samples to generate.
    split : float, default 0.5
        Proportion of samples assigned to the treatment group.
    outcome_params : dict, optional
        Binary outcome parameters, e.g. {"p": {"A": 0.10, "B": 0.11}}.
    add_ancillary : bool, default False
        Whether to add standard ancillary columns (age, platform, etc.).
    deterministic_ids : bool, default True
        Whether to generate deterministic user IDs.
    **kwargs : Any
        Additional arguments passed to the underlying `generate_classic_rct`.

    Returns
    -------
    CausalData or pd.DataFrame
        The generated dataset.
    """
    if outcome_params is None:
        outcome_params = {"p": {"A": 0.10, "B": 0.11}}
    df = generate_classic_rct(
        n=n,
        split=split,
        random_state=seed,
        outcome_params=outcome_params,
        add_pre=add_pre,
        beta_y=beta_y,
        outcome_depends_on_x=outcome_depends_on_x,
        include_oracle=include_oracle,
        add_ancillary=add_ancillary,
        deterministic_ids=deterministic_ids,
        return_causal_data=False,
        **kwargs
    )
    if "user_id" not in df.columns:
        rng = np.random.default_rng(seed)
        if deterministic_ids:
            user_ids = _deterministic_ids(rng, len(df))
        else:
            user_ids = [uuid.uuid4().hex[:5] for _ in range(len(df))]
        df.insert(0, "user_id", user_ids)
    # The requirement asks for outcome - conversion(binary)
    df = df.rename(columns={"y": "conversion"})

    if not return_causal_data:
        return df

    exclude = {"conversion", "d", "m", "m_obs", "tau_link", "g0", "g1", "cate", "user_id"}
    confounders = [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]

    return CausalData(
        df=df,
        treatment="d",
        outcome="conversion",
        confounders=confounders,
        user_id="user_id"
    )


def classic_rct_gamma_26(
    seed: int = 42,
    add_pre: bool = False,
    beta_y: Optional[Union[List[float], np.ndarray]] = None,
    outcome_depends_on_x: bool = True,
    include_oracle: bool = False,
    return_causal_data: bool = True,
    *,
    n: int = 10000,
    split: float = 0.5,
    outcome_params: Optional[Dict] = None,
    add_ancillary: bool = True,
    deterministic_ids: bool = True,
    **kwargs
):
    r"""
    A pre-configured classic RCT dataset with a Gamma-distributed outcome.

    The dataset is designed to represent skewed metrics like revenue or spend.
    It includes three binary confounders: `platform_ios`, `country_usa`, and
    `source_paid`.

    Notes
    -----
    The outcome $Y$ is generated from a Gamma distribution using a log link for the mean:

    .. math::

        Y \sim \text{Gamma}(\kappa, \mu(D, X) / \kappa)

    where:

    .. math::

        \log(\mu(D, X)) = \alpha_y + \theta D + \beta_y X

    and $\kappa$ is the shape parameter. This implies multiplicative treatment
    effects on the mean scale.

    Examples
    --------
    >>> from causalis.scenarios.classic_rct.dgp import classic_rct_gamma_26
    >>> data = classic_rct_gamma_26(n=5000, seed=42)
    >>> print(f"{data.df['y'].mean():.2f}")  # Approximately 47.6
    47.58
    >>> print(data.df.columns.tolist())
    ['user_id', 'y', 'd', 'platform_ios', 'country_usa', 'source_paid', 'age', 'platform_android', 'platform_ios_anc', 'platform_web']

    Parameters
    ----------
    seed : int, default 42
        Random seed for reproducibility.
    add_pre : bool, default False
        Whether to generate a pre-period covariate ('y_pre').
    beta_y : array-like, optional
        Linear coefficients for confounders in the outcome model. Default is [0.25, 0.20, 0.45].
    outcome_depends_on_x : bool, default True
        Whether to add default effects for confounders if beta_y is None.
    include_oracle : bool, default False
        Whether to include oracle ground-truth columns like 'cate', 'propensity', etc.
    return_causal_data : bool, default True
        Whether to return a `CausalData` object or a `pd.DataFrame`.
    n : int, default 10000
        Number of samples to generate.
    split : float, default 0.5
        Proportion of samples assigned to the treatment group.
    outcome_params : dict, optional
        Gamma outcome parameters, e.g. {"shape": 2.0, "scale": {"A": 15.0, "B": 16.5}}.
    add_ancillary : bool, default True
        Whether to add standard ancillary columns (age, platform, etc.).
    deterministic_ids : bool, default True
        Whether to generate deterministic user IDs.
    **kwargs : Any
        Additional arguments passed to the underlying `classic_rct_gamma`.

    Returns
    -------
    CausalData or pd.DataFrame
        The generated dataset.
    """
    if outcome_params is None:
        outcome_params = {"shape": 2.0, "scale": {"A": 15.0, "B": 16.5}}
    df = classic_rct_gamma(
        n=n,
        split=split,
        random_state=seed,
        outcome_params=outcome_params,
        add_pre=add_pre,
        beta_y=beta_y,
        outcome_depends_on_x=outcome_depends_on_x,
        include_oracle=include_oracle,
        add_ancillary=add_ancillary,
        deterministic_ids=deterministic_ids,
        return_causal_data=False,
        **kwargs
    )
    if "user_id" not in df.columns:
        rng = np.random.default_rng(seed)
        if deterministic_ids:
            user_ids = _deterministic_ids(rng, len(df))
        else:
            user_ids = [uuid.uuid4().hex[:5] for _ in range(len(df))]
        df.insert(0, "user_id", user_ids)

    if not return_causal_data:
        return df

    exclude = {"y", "d", "m", "m_obs", "tau_link", "g0", "g1", "cate", "user_id"}
    confounders = [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]

    return CausalData(
        df=df,
        treatment="d",
        outcome="y",
        confounders=confounders,
        user_id="user_id"
    )
