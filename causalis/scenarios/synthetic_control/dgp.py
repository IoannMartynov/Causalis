from __future__ import annotations

from typing import Optional, Union

import pandas as pd

from causalis.data_contracts.panel_data_scm import PanelDataSCM
from causalis.dgp.panel_data_scm.functional import (
    generate_scm_gamma_26_data,
    generate_scm_poisson_26_data,
)

PanelOutput = Union[pd.DataFrame, PanelDataSCM]


def generate_scm_gamma_26(
    seed: int = 42,
    return_panel_data: bool = True,
    include_oracles: bool = False,
    n_donors: int = 30,
    n_pre_periods: Optional[int] = None,
    n_post_periods: Optional[int] = None,
    treatment_effect_rate: float = 0.12,
    treatment_effect_slope: float = 0.01,
    missing_outcome_frac: float = 0.0,
    **advanced_params,
) -> PanelOutput:
    """
    Generate realistic Gamma synthetic-control panel data.

    Parameters
    ----------
    seed : int, default=42
        Random seed.
    return_panel_data : bool, default=True
        If True, return a :class:`~causalis.data_contracts.panel_data_scm.PanelDataSCM`
        object. If False, return a pandas DataFrame.
    include_oracles : bool, default=False
        Whether to include oracle truth columns in the returned data:
        `is_treated_unit`, `y_cf`, `tau_realized_true`, `mu_cf`,
        `mu_treated`, `tau_mean_true`.
        Scenario-level outputs always exclude synthetic covariates
        `exposure`, `macro_index`, `seasonality_index`.
    n_donors : int, default=30
        Number of donor units.
    n_pre_periods : int or None, default=None
        Number of pre-treatment periods. Preferred explicit horizon control.
        When both `n_pre_periods` and `n_post_periods` are omitted, scenario
        defaults are used (`36` pre, `12` post). The generated panel includes
        one explicit intervention-anchor period, so each unit has
        ``n_pre_periods + 1 + n_post_periods`` rows.
    n_post_periods : int or None, default=None
        Number of post-treatment periods. Must be provided together with
        `n_pre_periods` when using explicit horizon control.
    treatment_effect_rate : float, default=0.12
        Long-run post-treatment relative effect scale. The first post period is
        attenuated by a ramp factor ``1 - exp(-1 / 2.5)`` (about 0.33x when slope
        is zero).
    treatment_effect_slope : float, default=0.01
        Linear slope of the post-treatment relative effect path.
    missing_outcome_frac : float, default=0.0
        Fraction of outcomes to mask as missing in the base generator.
    **advanced_params
        Forwarded to :func:`causalis.dgp.panel_data_scm.generate_scm_gamma_data`.
        Common advanced knobs include `time_start`, `calendar_start`,
        and latent/missingness configuration.

    Returns
    -------
    pandas.DataFrame or PanelDataSCM
        Long panel data for SCM experiments.

    Notes
    -----
    Time-axis semantics:

    - `n_pre_periods`: number of periods strictly before the intervention anchor.
    - One explicit intervention-anchor period is included in the output.
    - `n_post_periods`: number of periods strictly after the intervention anchor.
    - `time_start`: offset for the first `calendar_time` period relative to
      `calendar_start` (default `calendar_start="2000-01"` and `time_start=1`).
    - `treated_time`: explicit 0/1 treatment-assignment indicator in returned
      data (`1` only for treated-unit rows at/after the first treated period;
      `0` otherwise).
    - `PanelDataSCM` is built with required fields only:
      `df`, `y`, `unit_col`, `time_col`, `treated_time`.
    - When `return_panel_data=True`, all contract metadata is derived from the
      final `treated_time` path. Because this scenario keeps one explicit anchor
      period in the panel, contract-level pre periods are
      ``n_pre_periods + 1`` and post periods are ``n_post_periods``.
    - With this function's default arguments, the explicit values are:
      ``n_pre_periods=36``, ``n_post_periods=12``, ``calendar_start='2000-01'``,
      ``time_start=1``, first treated period at ``Period('2003-02', 'M')``,
      and intervention anchor at ``Period('2003-01', 'M')``.
    """
    return generate_scm_gamma_26_data(
        seed=seed,
        return_panel_data=return_panel_data,
        include_oracles=include_oracles,
        n_donors=n_donors,
        n_pre_periods=n_pre_periods,
        n_post_periods=n_post_periods,
        treatment_effect_rate=treatment_effect_rate,
        treatment_effect_slope=treatment_effect_slope,
        missing_outcome_frac=missing_outcome_frac,
        advanced_params=advanced_params,
    )


def generate_scm_poisson_26(
    seed: int = 42,
    return_panel_data: bool = True,
    include_oracles: bool = False,
    n_donors: int = 10,
    n_pre_periods: Optional[int] = None,
    n_post_periods: Optional[int] = None,
    treatment_effect_rate: float = 0.10,
    treatment_effect_slope: float = 0.005,
    donor_missing_block_frac: float = 0.08,
    **advanced_params,
) -> PanelOutput:
    """
    Generate realistic Poisson synthetic-control panel data.

    Parameters
    ----------
    seed : int, default=42
        Random seed.
    return_panel_data : bool, default=True
        If True, return a :class:`~causalis.data_contracts.panel_data_scm.PanelDataSCM`
        object. If False, return a pandas DataFrame.
    include_oracles : bool, default=False
        Whether to include oracle truth columns in the returned data:
        `is_treated_unit`, `y_cf`, `tau_realized_true`, `mu_cf`,
        `mu_treated`, `tau_mean_true`.
        Scenario-level outputs always exclude synthetic covariates
        `exposure`, `macro_index`, `seasonality_index`.
    n_donors : int, default=10
        Number of donor units.
    n_pre_periods : int or None, default=None
        Number of pre-treatment periods. Preferred explicit horizon control.
        When both `n_pre_periods` and `n_post_periods` are omitted, scenario
        defaults are used (`36` pre, `12` post). The generated panel includes
        one explicit intervention-anchor period, so each unit has
        ``n_pre_periods + 1 + n_post_periods`` rows.
    n_post_periods : int or None, default=None
        Number of post-treatment periods. Must be provided together with
        `n_pre_periods` when using explicit horizon control.
    treatment_effect_rate : float, default=0.10
        Long-run post-treatment relative effect scale. The first post period is
        attenuated by a ramp factor ``1 - exp(-1 / 2.5)`` (about 0.33x when slope
        is zero).
    treatment_effect_slope : float, default=0.005
        Linear slope of the post-treatment relative effect path.
    donor_missing_block_frac : float, default=0.08
        Fraction of donor-only rows to mask via contiguous missing-time blocks.
    **advanced_params
        Forwarded to :func:`causalis.dgp.panel_data_scm.generate_scm_poisson_data`.
        Common advanced knobs include `time_start`, `calendar_start`,
        and latent/missingness configuration.

    Returns
    -------
    pandas.DataFrame or PanelDataSCM
        Long panel data for SCM experiments.

    Notes
    -----
    Time-axis semantics:

    - `n_pre_periods`: number of periods strictly before the intervention anchor.
    - One explicit intervention-anchor period is included in the output.
    - `n_post_periods`: number of periods strictly after the intervention anchor.
    - `time_start`: offset for the first `calendar_time` period relative to
      `calendar_start` (default `calendar_start="2000-01"` and `time_start=1`).
    - `treated_time`: explicit 0/1 treatment-assignment indicator in returned
      data (`1` only for treated-unit rows at/after the first treated period;
      `0` otherwise).
    - `PanelDataSCM` is built with required fields only:
      `df`, `y`, `unit_col`, `time_col`, `treated_time`.
    - When `return_panel_data=True`, all contract metadata is derived from the
      final `treated_time` path. Because this scenario keeps one explicit anchor
      period in the panel, contract-level pre periods are
      ``n_pre_periods + 1`` and post periods are ``n_post_periods``.
    - With this function's default arguments, the explicit values are:
      ``n_pre_periods=36``, ``n_post_periods=12``, ``calendar_start='2000-01'``,
      ``time_start=1``, first treated period at ``Period('2003-02', 'M')``,
      and intervention anchor at ``Period('2003-01', 'M')``.
    """
    return generate_scm_poisson_26_data(
        seed=seed,
        return_panel_data=return_panel_data,
        include_oracles=include_oracles,
        n_donors=n_donors,
        n_pre_periods=n_pre_periods,
        n_post_periods=n_post_periods,
        treatment_effect_rate=treatment_effect_rate,
        treatment_effect_slope=treatment_effect_slope,
        donor_missing_block_frac=donor_missing_block_frac,
        advanced_params=advanced_params,
    )
