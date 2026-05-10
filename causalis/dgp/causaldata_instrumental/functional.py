"""
High-level helpers for instrumental-variable synthetic datasets.

Use this module when you want a ready-made binary-instrument, binary-treatment
dataset compatible with :class:`causalis.data_contracts.iv_causal_data.IVCausalData`.
For lower-level structural control, instantiate
:class:`causalis.dgp.causaldata_instrumental.base.InstrumentalGenerator`
directly.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from causalis.data_contracts.iv_causal_data import IVCausalData
from causalis.dgp.base import _add_ancillary_info

from .base import InstrumentalGenerator

_IV_ORACLE_COLS = {
    "m",
    "r_obs",
    "r_z0",
    "r_z1",
    "g_z0",
    "g_z1",
    "iv_first_stage",
    "iv_reduced_form",
    "late_x",
    "late",
    "tau_link",
    "g_d0",
    "g_d1",
    "cate",
}


def generate_iv_data(
    n: int = 1_000,
    *,
    outcome_type: str = "continuous",
    theta: float = 1.0,
    tau: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    sigma_y: float = 1.0,
    alpha_y: float = 0.0,
    gamma_shape: float = 2.0,
    first_stage: float = 1.25,
    alpha_d: float = -0.2,
    alpha_z: float = 0.0,
    target_d_rate: Optional[float] = None,
    target_z_rate: Optional[float] = 0.5,
    confounder_specs: Optional[List[Dict[str, Any]]] = None,
    beta_y: Optional[Union[List[float], np.ndarray]] = None,
    beta_d: Optional[Union[List[float], np.ndarray]] = None,
    beta_z: Optional[Union[List[float], np.ndarray]] = None,
    g_y: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    g_d: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    g_z: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    u_strength_d: float = 0.8,
    u_strength_y: float = 0.8,
    propensity_sharpness: float = 1.0,
    instrument_sharpness: float = 1.0,
    random_state: Optional[int] = 42,
    k: int = 2,
    x_sampler: Optional[Callable[[int, int, int], np.ndarray]] = None,
    use_copula: bool = False,
    copula_corr: Optional[np.ndarray] = None,
    include_oracle: bool = True,
    return_causal_data: bool = False,
    instrument_name: str = "z",
    add_ancillary: bool = False,
    deterministic_ids: bool = False,
) -> Union[pd.DataFrame, IVCausalData]:
    """
    Generate a synthetic instrumental-variable dataset.

    Parameters
    ----------
    n : int, default=1000
        Number of samples.
    outcome_type : {"continuous", "binary", "poisson", "gamma"}, default="continuous"
        Outcome family.
    theta : float, default=1.0
        Constant treatment effect on the structural outcome scale.
    first_stage : float, default=1.25
        Additive log-odds effect of the instrument on treatment.
    target_z_rate : float, optional
        Target marginal instrument rate.
    target_d_rate : float, optional
        Target marginal treatment rate after instrument assignment.
    u_strength_d, u_strength_y : float, default=0.8
        Latent confounding strengths in treatment and outcome.
    return_causal_data : bool, default=False
        If True, return a validated :class:`IVCausalData` object.
    instrument_name : str, default="z"
        Instrument column name.

    Returns
    -------
    pandas.DataFrame or IVCausalData
        Synthetic IV dataset or validated IV data contract.

    Examples
    --------
    >>> from causalis.dgp.causaldata_instrumental import generate_iv_data
    >>> data = generate_iv_data(n=500, return_causal_data=True)
    >>> data.instruments
    ['z']
    """
    gen = InstrumentalGenerator(
        theta=theta,
        tau=tau,
        beta_y=None if beta_y is None else np.asarray(beta_y, dtype=float),
        beta_d=None if beta_d is None else np.asarray(beta_d, dtype=float),
        g_y=g_y,
        g_d=g_d,
        alpha_y=alpha_y,
        alpha_d=alpha_d,
        sigma_y=sigma_y,
        outcome_type=outcome_type,
        confounder_specs=confounder_specs,
        k=int(k),
        x_sampler=x_sampler,
        use_copula=use_copula,
        copula_corr=copula_corr,
        target_d_rate=target_d_rate,
        u_strength_d=u_strength_d,
        u_strength_y=u_strength_y,
        propensity_sharpness=propensity_sharpness,
        gamma_shape=gamma_shape,
        include_oracle=include_oracle,
        seed=random_state,
        instrument_name=instrument_name,
        first_stage=first_stage,
        beta_z=None if beta_z is None else np.asarray(beta_z, dtype=float),
        g_z=g_z,
        alpha_z=alpha_z,
        target_z_rate=target_z_rate,
        instrument_sharpness=instrument_sharpness,
    )
    df = gen.generate(n)

    if add_ancillary:
        rng = np.random.default_rng(random_state)
        exclude = {"y", "d", instrument_name, *_IV_ORACLE_COLS}
        x_cols = [c for c in df.columns if c not in exclude]
        df = _add_ancillary_info(df, int(n), rng, deterministic_ids, x_cols)

    if not return_causal_data:
        return _order_columns(df, instrument_name=instrument_name)

    exclude = {"y", "d", instrument_name, "user_id", *_IV_ORACLE_COLS}
    confounder_cols = [
        c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]
    return IVCausalData.from_df(
        df,
        treatment="d",
        outcome="y",
        instruments=instrument_name,
        confounders=confounder_cols,
        user_id="user_id" if "user_id" in df.columns else None,
    )


def _order_columns(df: pd.DataFrame, *, instrument_name: str) -> pd.DataFrame:
    """Return columns in a stable core/confounder/oracle order."""
    all_cols = list(df.columns)
    core = [c for c in ["user_id", "y", "d", instrument_name] if c in all_cols]
    oracle = [c for c in all_cols if c in _IV_ORACLE_COLS]
    confounders = [c for c in all_cols if c not in set(core + oracle)]
    return df[core + confounders + oracle]

