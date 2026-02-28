from __future__ import annotations

from typing import Optional, Union

import pandas as pd

from causalis.data_contracts.panel_data_scm import PanelDataSCM
from causalis.dgp.panel_data_scm import generate_scm_gamma_data, generate_scm_poisson_data


_ORACLE_COLS = (
    "is_treated_unit",
    "y_cf",
    "tau_realized_true",
    "mu_cf",
    "mu_treated",
    "tau_mean_true",
)
_COVARIATE_COLS = ("exposure", "macro_index", "seasonality_index")
_DEFAULT_SCM26_PRE_PERIODS = 36
_DEFAULT_SCM26_POST_PERIODS = 12


def _resolve_scm26_periods(
    *,
    n_pre_periods: Optional[int],
    n_post_periods: Optional[int],
) -> tuple[int, int]:
    if n_pre_periods is None and n_post_periods is None:
        return _DEFAULT_SCM26_PRE_PERIODS, _DEFAULT_SCM26_POST_PERIODS
    if n_pre_periods is None or n_post_periods is None:
        raise ValueError(
            "Provide both n_pre_periods and n_post_periods, or omit both to use scenario defaults."
        )
    return int(n_pre_periods), int(n_post_periods)


def _expand_periods_with_intervention_anchor(
    *,
    n_pre_periods: int,
    n_post_periods: int,
) -> tuple[int, int]:
    """Map wrapper periods to low-level generator periods.

    Scenario wrappers expose an explicit intervention-anchor period, so each
    unit has ``n_pre_periods + 1 + n_post_periods`` rows.
    """
    return int(n_pre_periods) + 1, int(n_post_periods)


def _apply_anchor_period_windows(
    out: Union[pd.DataFrame, PanelDataSCM],
    *,
    n_pre_periods: int,
    n_post_periods: int,
) -> Union[pd.DataFrame, PanelDataSCM]:
    """Expose pre/post windows around an explicit intervention-anchor period."""
    if isinstance(out, pd.DataFrame):
        return out

    intervention_anchor = out.treatment_start - 1
    time_values = sorted(pd.Index(out.df[out.time_col].unique()).tolist())
    pre_periods = [t for t in time_values if t < intervention_anchor]
    post_periods = [t for t in time_values if t > intervention_anchor]
    if len(pre_periods) != int(n_pre_periods) or len(post_periods) != int(n_post_periods):
        raise RuntimeError(
            "Internal period mapping error: expected "
            f"{n_pre_periods} pre and {n_post_periods} post periods, got "
            f"{len(pre_periods)} pre and {len(post_periods)} post."
        )

    return out.model_copy(
        update={
            "pre_periods": pre_periods,
            "post_periods": post_periods,
        }
    )


def _with_treatment_start_column(
    out: Union[pd.DataFrame, PanelDataSCM],
    *,
    n_pre_periods: int,
) -> Union[pd.DataFrame, PanelDataSCM]:
    """Ensure DataFrame outputs expose treatment_start like PanelDataSCM outputs."""
    if not isinstance(out, pd.DataFrame):
        return out
    if "treatment_start" in out.columns:
        return out
    if "calendar_time" not in out.columns:
        raise RuntimeError("Expected 'calendar_time' column to derive treatment_start.")

    time_values = sorted(pd.Index(out["calendar_time"].unique()).tolist())
    post_start_idx = int(n_pre_periods) + 1
    if post_start_idx >= len(time_values):
        raise RuntimeError(
            "Internal period mapping error: cannot derive treatment_start from calendar_time."
        )

    out_df = out.copy()
    out_df["treatment_start"] = time_values[post_start_idx]
    return out_df


def _apply_include_oracles(
    out: Union[pd.DataFrame, PanelDataSCM],
    *,
    include_oracles: bool,
) -> Union[pd.DataFrame, PanelDataSCM]:
    drop_cols = list(_COVARIATE_COLS)
    if not include_oracles:
        drop_cols.extend(_ORACLE_COLS)
    if isinstance(out, pd.DataFrame):
        return out.drop(columns=drop_cols, errors="ignore")
    df = out.df.drop(columns=drop_cols, errors="ignore")
    covariate_cols = tuple(col for col in out.covariate_cols if col in df.columns)
    return out.model_copy(update={"df": df, "covariate_cols": covariate_cols})


def generate_scm_gamma_26(
    n: Optional[int] = None,
    seed: int = 42,
    return_panel_data: bool = True,
    include_oracles: bool = False,
    n_donors: int = 10,
    n_pre_periods: Optional[int] = 24,
    n_post_periods: Optional[int] = 3,
    treatment_effect_rate: float = 0.12,
    treatment_effect_slope: float = 0.01,
    missing_outcome_frac: float = 0.0,
    **advanced_params,
) -> Union[pd.DataFrame, PanelDataSCM]:
    """
    Generate realistic Gamma synthetic-control panel data.

    Parameters
    ----------
    n : int or None, default=None
        Legacy compatibility argument. Scenario defaults no longer infer periods
        from `n`; default horizon is controlled by `n_pre_periods` and
        `n_post_periods`.
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
    n_pre_periods : int or None, default=24
        Number of pre-treatment periods. Preferred explicit horizon control.
        When both `n_pre_periods` and `n_post_periods` are omitted, scenario
        defaults are used (`36` pre, `12` post). The generated panel includes
        one explicit intervention-anchor period, so each unit has
        ``n_pre_periods + 1 + n_post_periods`` rows.
    n_post_periods : int or None, default=3
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
        Common advanced knobs include `n_pre_periods`, `n_post_periods`,
        and `time_start`.

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
    - `treatment_start`: first treated/post period in the returned panel
      (the intervention anchor is one period earlier).
    - With this function's default arguments, the explicit values are:
      ``n_pre_periods=24``, ``n_post_periods=3``, ``calendar_start='2000-01'``,
      ``time_start=1``, ``treatment_start=Period('2002-02', 'M')``,
      intervention anchor at ``Period('2002-01', 'M')``.
    """
    n_pre_periods, n_post_periods = _resolve_scm26_periods(
        n_pre_periods=n_pre_periods,
        n_post_periods=n_post_periods,
    )
    n_pre_effective, n_post_effective = _expand_periods_with_intervention_anchor(
        n_pre_periods=n_pre_periods,
        n_post_periods=n_post_periods,
    )
    n_hint = (
        int((n_donors + 1) * (n_pre_effective + n_post_effective))
        if n is None
        else int(n)
    )
    out = generate_scm_gamma_data(
        n=n_hint,
        seed=seed,
        return_panel_data=return_panel_data,
        n_donors=n_donors,
        n_pre_periods=n_pre_effective,
        n_post_periods=n_post_effective,
        treatment_effect_rate=treatment_effect_rate,
        treatment_effect_slope=treatment_effect_slope,
        missing_outcome_frac=missing_outcome_frac,
        **advanced_params,
    )
    out = _apply_anchor_period_windows(
        out,
        n_pre_periods=n_pre_periods,
        n_post_periods=n_post_periods,
    )
    out = _with_treatment_start_column(out, n_pre_periods=n_pre_periods)
    return _apply_include_oracles(out, include_oracles=include_oracles)


def generate_scm_poisson_26(
    n: Optional[int] = None,
    seed: int = 42,
    return_panel_data: bool = True,
    include_oracles: bool = False,
    n_donors: int = 10,
    n_pre_periods: Optional[int] = 24,
    n_post_periods: Optional[int] = 3,
    treatment_effect_rate: float = 0.10,
    treatment_effect_slope: float = 0.005,
    donor_missing_block_frac: float = 0.08,
    **advanced_params,
) -> Union[pd.DataFrame, PanelDataSCM]:
    """
    Generate realistic Poisson synthetic-control panel data.

    Parameters
    ----------
    n : int or None, default=None
        Legacy compatibility argument. Scenario defaults no longer infer periods
        from `n`; default horizon is controlled by `n_pre_periods` and
        `n_post_periods`.
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
    n_donors : int, default=8
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
        Common advanced knobs include `n_pre_periods`, `n_post_periods`,
        and `time_start`.

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
    - `treatment_start`: intervention boundary period, computed as
      ``calendar_start + (time_start - 1) + n_pre_periods``.
    - With this function's default arguments, the explicit values are:
      ``n_pre_periods=36``, ``n_post_periods=12``, ``calendar_start='2000-01'``,
      ``time_start=1``, ``treatment_start=Period('2003-01', 'M')``,
      first post period at ``Period('2003-02', 'M')``.
    """
    n_pre_periods, n_post_periods = _resolve_scm26_periods(
        n_pre_periods=n_pre_periods,
        n_post_periods=n_post_periods,
    )
    n_pre_effective, n_post_effective = _expand_periods_with_intervention_anchor(
        n_pre_periods=n_pre_periods,
        n_post_periods=n_post_periods,
    )
    n_hint = (
        int((n_donors + 1) * (n_pre_effective + n_post_effective))
        if n is None
        else int(n)
    )
    out = generate_scm_poisson_data(
        n=n_hint,
        seed=seed,
        return_panel_data=return_panel_data,
        n_donors=n_donors,
        n_pre_periods=n_pre_effective,
        n_post_periods=n_post_effective,
        treatment_effect_rate=treatment_effect_rate,
        treatment_effect_slope=treatment_effect_slope,
        donor_missing_block_frac=donor_missing_block_frac,
        **advanced_params,
    )
    out = _apply_anchor_period_windows(
        out,
        n_pre_periods=n_pre_periods,
        n_post_periods=n_post_periods,
    )
    out = _with_treatment_start_column(out, n_pre_periods=n_pre_periods)
    return _apply_include_oracles(out, include_oracles=include_oracles)
