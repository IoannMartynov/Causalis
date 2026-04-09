"""
High-level generators for multi-arm treatment datasets.

The helpers in this module build datasets where treatment assignment is a full
one-hot encoding over multiple arms, with arm-specific structural effects and
optional oracle columns.

Notes
-----
The first treatment column is treated as the control arm by convention. When
``return_causal_data=True``, the returned :class:`MultiCausalData` object uses
that first arm as ``control_treatment``.

Examples
--------
>>> from causalis.dgp.multicausaldata.functional import generate_multitreatment
>>> data = generate_multitreatment(
...     n=1000,
...     n_treatments=3,
...     theta=[0.0, 0.5, 1.0],
...     return_causal_data=True,
... )
>>> data.treatment_names
['d_0', 'd_1', 'd_2']
"""

from __future__ import annotations

import pandas as pd
from typing import Optional, Union, List, Dict, Any

from causalis.data_contracts.multicausaldata import MultiCausalData
from .base import MultiCausalDatasetGenerator


def generate_multitreatment(
    n: int = 10_000,
    n_treatments: int = 3,
    outcome_type: str = "continuous",
    sigma_y: float = 1.0,
    alpha_y: float = 0.0,
    gamma_shape: float = 2.0,
    tau: Optional[Any] = None,
    target_d_rate: Optional[Union[List[float], Any]] = None,
    confounder_specs: Optional[List[Dict[str, Any]]] = None,
    beta_y: Optional[Any] = None,
    beta_d: Optional[Any] = None,
    theta: Optional[Any] = None,
    random_state: Optional[int] = 42,
    k: int = 0,
    x_sampler: Optional[Any] = None,
    use_copula: bool = False,
    copula_corr: Optional[Any] = None,
    include_oracle: bool = True,
    return_causal_data: bool = False,
    d_names: Optional[List[str]] = None,
) -> Union[pd.DataFrame, MultiCausalData]:
    """
    Generate a multi-treatment dataset using MultiCausalDatasetGenerator.

    Parameters
    ----------
    n : int, default=10_000
        Number of samples.
    n_treatments : int, default=3
        Number of treatment classes (including control).
    outcome_type : {"continuous", "binary", "poisson", "gamma"}, default="continuous"
        Outcome family.
    sigma_y : float, default=1.0
        Noise level for continuous outcomes.
    alpha_y : float, default=0.0
        Outcome intercept on the structural link scale.
    gamma_shape : float, default=2.0
        Shape parameter when `outcome_type="gamma"`.
    tau : callable or list of callables, optional
        Heterogeneous treatment effects per class. When provided with `theta`,
        the structural effect is additive on link scale (`theta + tau(X)`).
    target_d_rate : array-like, optional
        Target marginal class probabilities (length K).
    confounder_specs : list of dict, optional
        Schema for confounder distributions.
    beta_y : array-like, optional
        Linear coefficients for outcome model.
    beta_d : array-like, optional
        Linear coefficients for treatment model.
    theta : float or array-like, optional
        Constant treatment effects per class (adds to `tau` when both are provided).
    random_state : int, optional
        Random seed.
    k : int, default=0
        Number of confounders if confounder_specs is None.
    x_sampler : callable, optional
        Custom sampler for confounders.
    use_copula : bool, default=False
        Whether to use Gaussian copula sampling for confounders.
    copula_corr : array-like, optional
        Correlation matrix used when `use_copula=True`.
    include_oracle : bool, default=True
        Whether to include oracle columns.
    return_causal_data : bool, default=False
        Whether to return a MultiCausalData object.
    d_names : list of str, optional
        Names of treatment columns.

    Returns
    -------
    pd.DataFrame or MultiCausalData

    Notes
    -----
    Treatment columns form a mutually exclusive one-hot representation:

    - exactly one treatment column equals ``1`` for each row;
    - ``d_names[0]`` is the control arm;
    - arm-specific structural effects are defined on the outcome link scale and
      are mapped back to the natural outcome scale in oracle outputs.

    Examples
    --------
    >>> from causalis.dgp.multicausaldata.functional import generate_multitreatment
    >>> df = generate_multitreatment(
    ...     n=500,
    ...     n_treatments=4,
    ...     outcome_type="continuous",
    ...     theta=[0.0, 0.2, 0.5, 0.8],
    ... )
    >>> treatment_cols = ["d_0", "d_1", "d_2", "d_3"]
    >>> int(df[treatment_cols].sum(axis=1).eq(1).all())
    1
    """
    gen = MultiCausalDatasetGenerator(
        n_treatments=n_treatments,
        d_names=d_names,
        outcome_type=outcome_type,
        sigma_y=sigma_y,
        alpha_y=alpha_y,
        gamma_shape=gamma_shape,
        tau=tau,
        target_d_rate=target_d_rate,
        confounder_specs=confounder_specs,
        beta_y=beta_y,
        beta_d=beta_d,
        theta=theta,
        seed=random_state,
        k=int(k),
        x_sampler=x_sampler,
        use_copula=use_copula,
        copula_corr=copula_corr,
        include_oracle=include_oracle,
    )
    df = gen.generate(n)

    if not return_causal_data:
        return df

    return MultiCausalData(
        df=df,
        outcome="y",
        treatment_names=gen.d_names,
        confounders=gen.confounder_names_,
        control_treatment=gen.d_names[0],
    )
