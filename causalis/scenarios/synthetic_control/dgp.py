from __future__ import annotations

from typing import Optional, Union

import pandas as pd

from causalis.data_contracts.panel_data_scm import PanelDataSCM
from causalis.dgp.panel_data_scm.functional import (
    generate_scm_gamma_26_data,
    generate_scm_poisson_26_data,
)

PanelOutput = Union[pd.DataFrame, PanelDataSCM]


def _generate_scm26(
        *,
        distribution: str,
        seed: int,
        return_panel_data: bool,
        include_oracles: bool,
        n_donors: int,
        n_pre_periods: Optional[int],
        n_post_periods: Optional[int],
        treatment_effect_rate: float,
        treatment_effect_slope: float,
        advanced_params: dict,
) -> PanelOutput:
    if distribution == "gamma":
        return generate_scm_gamma_26_data(
            seed=seed,
            return_panel_data=return_panel_data,
            include_oracles=include_oracles,
            n_donors=n_donors,
            n_pre_periods=n_pre_periods,
            n_post_periods=n_post_periods,
            treatment_effect_rate=treatment_effect_rate,
            treatment_effect_slope=treatment_effect_slope,
            advanced_params=advanced_params,
        )
    if distribution == "poisson":
        return generate_scm_poisson_26_data(
            seed=seed,
            return_panel_data=return_panel_data,
            include_oracles=include_oracles,
            n_donors=n_donors,
            n_pre_periods=n_pre_periods,
            n_post_periods=n_post_periods,
            treatment_effect_rate=treatment_effect_rate,
            treatment_effect_slope=treatment_effect_slope,
            advanced_params=advanced_params,
        )
    raise ValueError(f"Unsupported SCM26 distribution: {distribution!r}.")


def generate_scm_gamma_26(
        seed: int = 42,
        return_panel_data: bool = True,
        include_oracles: bool = False,
        n_donors: int = 40,
        n_pre_periods: Optional[int] = 36,
        n_post_periods: Optional[int] = 6,
        treatment_effect_rate: float = 0.10,
        treatment_effect_slope: float = 0.002,
        **advanced_params,
) -> PanelOutput:
    r"""
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
    n_donors : int, default=40
        Number of donor units.
    n_pre_periods : int or None, default=36
        Number of pre-treatment periods. Preferred explicit horizon control.
        When both `n_pre_periods` and `n_post_periods` are omitted, scenario
        defaults are used (`36` pre, `6` post). The generated panel includes
        one explicit intervention-anchor period, so each unit has
        ``n_pre_periods + 1 + n_post_periods`` rows.
    n_post_periods : int or None, default=6
        Number of post-treatment periods. Must be provided together with
        `n_pre_periods` when using explicit horizon control.
    treatment_effect_rate : float, default=0.10
        Long-run post-treatment relative effect scale. The first post period is
        attenuated by a ramp factor ``1 - exp(-1 / 2.5)`` (about 0.33x when slope
        is zero).
    treatment_effect_slope : float, default=0.002
        Linear slope of the post-treatment relative effect path.
    **advanced_params
        Forwarded to :func:`causalis.dgp.panel_data_scm.generate_scm_gamma_data`.
        Common advanced knobs include `time_start`, `calendar_start`,
        and latent-factor configuration. Defaults used by this wrapper are
        `gamma_shape=120`, `donor_noise_std_log=0.03`,
        `common_factor_std_log=0.03`, `latent_factor_std_log=0.0`,
        and `prefit_mismatch_std_log=0.0`.

    Returns
    -------
    pandas.DataFrame or PanelDataSCM
        Long panel data for SCM experiments.

    Notes
    -----
    DGP Math:
    The data follows a hierarchical log-linear model for the mean: math:`\mu`.
    For each donor unit :math:`j` at time :math:`t`, the mean is :math:`\mu_{tj} = E_{tj} \cdot \exp(\eta_{tj})`
    where :math:`E_{tj}` is exposure (with growth and noise) and :math:`\eta_{tj}` includes seasonality,
    common factors (macro index), latent factors, and unit-specific noise.
    Outcomes are sampled as :math:`y_{tj} \sim \text{Gamma}(k, \mu_{tj}/k)`, where :math:`k` is `gamma_shape`.

    The treated unit's counterfactual mean :math:`\mu_{t, cf}` is a weighted combination of donors
    (via Dirichlet weights) with a potential pre-fit mismatch. The realized treated outcome
    is :math:`y_{t, treated} = y_{t, cf} \cdot (1 + \tau_t^{rate})`, where :math:`\tau_t^{rate}` is the
    relative treatment effect.

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
      ``n_pre_periods=36``, ``n_post_periods=6``, ``calendar_start='2000-01'``,
      ``time_start=1``, first treated period at ``Period('2003-02', 'M')``,
      and intervention anchor at ``Period('2003-01', 'M')``.
    """
    default_advanced_params = dict(
        gamma_shape=120,
        donor_noise_std_log=0.03,
        common_factor_std_log=0.03,
        latent_factor_std_log=0.0,
        prefit_mismatch_std_log=0.0,
    )
    merged_advanced_params = {**default_advanced_params, **advanced_params}

    return _generate_scm26(
        distribution="gamma",
        seed=seed,
        return_panel_data=return_panel_data,
        include_oracles=include_oracles,
        n_donors=n_donors,
        n_pre_periods=n_pre_periods,
        n_post_periods=n_post_periods,
        treatment_effect_rate=treatment_effect_rate,
        treatment_effect_slope=treatment_effect_slope,
        advanced_params=merged_advanced_params,
    )


def generate_scm_poisson_26(
        seed: int = 42,
        return_panel_data: bool = True,
        include_oracles: bool = False,
        n_donors: int = 20,
        n_pre_periods: Optional[int] = 180,
        n_post_periods: Optional[int] = 4,
        treatment_effect_rate: float = 0.15,
        treatment_effect_slope: float = 0.0005,
        **advanced_params,
) -> PanelOutput:
    r"""
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
    n_donors : int, default=20
        Number of donor units.
    n_pre_periods : int or None, default=180
        Number of pre-treatment periods. Preferred explicit horizon control.
        When both `n_pre_periods` and `n_post_periods` are omitted, scenario
        defaults are used (`36` pre, `6` post). The generated panel includes
        one explicit intervention-anchor period, so each unit has
        ``n_pre_periods + 1 + n_post_periods`` rows.
    n_post_periods : int or None, default=4
        Number of post-treatment periods. Must be provided together with
        `n_pre_periods` when using explicit horizon control.
    treatment_effect_rate : float, default=0.15
        Long-run post-treatment relative effect scale. The first post period is
        attenuated by a ramp factor ``1 - exp(-1 / 2.5)`` (about 0.33x when slope
        is zero).
    treatment_effect_slope : float, default=0.0005
        Linear slope of the post-treatment relative effect path.
    **advanced_params
        Forwarded to :func:`causalis.dgp.panel_data_scm.generate_scm_poisson_data`.
        Common advanced knobs include `time_start`, `calendar_start`,
        and latent-factor configuration. Defaults used by this wrapper are
        `donor_noise_std_log=0.02`, `common_factor_std_log=0.02`,
        `latent_factor_std_log=0.0`, and `prefit_mismatch_std_log=0.0`.

    Returns
    -------
    pandas.DataFrame or PanelDataSCM
        Long panel data for SCM experiments.

    Notes
    -----
    DGP Math:
    The data follows a hierarchical log-linear model for the mean :math:`\mu`.
    For each donor unit :math:`j` at time :math:`t`, the mean is :math:`\mu_{tj} = E_{tj} \cdot \exp(\eta_{tj})`
    where :math:`E_{tj}` is exposure and :math:`\eta_{tj}` includes seasonality, common factors,
    latent factors, and unit-specific noise.
    Outcomes are sampled as :math:`y_{tj} \sim \text{Poisson}(\mu_{tj})`.

    The treated unit's counterfactual mean :math:`\mu_{t, cf}` is a weighted combination of donors.
    The realized treated outcome :math:`y_{t, treated}` is sampled from a Poisson distribution
    coupled with the counterfactual :math:`y_{t, cf}` via a thinning/superposition property to
    maintain exact marginals while ensuring the realized effect is driven by the multiplier.

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
      ``n_pre_periods=180``, ``n_post_periods=4``, ``calendar_start='2000-01'``,
      ``time_start=1``, first treated period at ``Period('2003-02', 'M')``,
      and intervention anchor at ``Period('2003-01', 'M')``.
    """
    default_advanced_params = dict(
        donor_noise_std_log=0.02,
        common_factor_std_log=0.02,
        latent_factor_std_log=0.0,
        prefit_mismatch_std_log=0.0,
    )
    merged_advanced_params = {**default_advanced_params, **advanced_params}

    return _generate_scm26(
        distribution="poisson",
        seed=seed,
        return_panel_data=return_panel_data,
        include_oracles=include_oracles,
        n_donors=n_donors,
        n_pre_periods=n_pre_periods,
        n_post_periods=n_post_periods,
        treatment_effect_rate=treatment_effect_rate,
        treatment_effect_slope=treatment_effect_slope,
        advanced_params=merged_advanced_params,
    )
