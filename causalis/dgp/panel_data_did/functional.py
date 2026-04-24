"""
Panel-data DGP wrappers for canonical/simultaneous-adoption DID benchmarks.

This module exposes high-level helpers that build realistic long-format panel
datasets with multiple treated units, never-treated controls, a common adoption
date, covariates, cluster labels, and optional oracle counterfactual/effect
columns.

Examples
--------
>>> from causalis.dgp.panel_data_did.functional import generate_did_data
>>> panel = generate_did_data(
...     n_treated_units=4,
...     n_control_units=8,
...     n_pre_periods=12,
...     n_post_periods=4,
...     return_panel_data=True,
... )
>>> panel.design_type
'simultaneous_adoption'
>>> {"y_cf", "tau_mean_true"}.issubset(panel.df.columns)
True
"""

from __future__ import annotations

from typing import Any, Optional, Union

import pandas as pd

from causalis.data_contracts.panel_data_did import PanelDataDID
from .base import PanelDIDGenerator, PanelDIDGeneratorConfig

PanelOutput = Union[pd.DataFrame, PanelDataDID]

_DID_ORACLE_COLS = (
    "y_cf",
    "tau_realized_true",
    "mu_cf",
    "mu_treated",
    "tau_mean_true",
)
_DID_INTERNAL_ORACLE_COLS = ("tau_rate_true",)
_DID_COVARIATE_COLS = PanelDIDGenerator.covariate_cols
_DEFAULT_DID_PRE_PERIODS = 24
_DEFAULT_DID_POST_PERIODS = 8


def _infer_pre_post_periods(
    *,
    n: int,
    n_treated_units: int,
    n_control_units: int,
    pre_share: float = 0.75,
    min_periods: int = 12,
    min_post_periods: int = 4,
) -> tuple[int, int]:
    if n <= 0:
        raise ValueError("n must be > 0.")
    if n_treated_units < 1:
        raise ValueError("n_treated_units must be >= 1.")
    if n_control_units < 1:
        raise ValueError("n_control_units must be >= 1.")
    if not (0.0 < pre_share < 1.0):
        raise ValueError("pre_share must be in (0, 1).")

    n_units = n_treated_units + n_control_units
    periods = int(max(min_periods, round(n / n_units)))
    n_post = int(max(min_post_periods, round((1.0 - pre_share) * periods)))
    n_post = min(n_post, periods - 1)
    n_pre = periods - n_post
    return n_pre, n_post


def _resolve_pre_post_periods(
    *,
    n: int,
    n_treated_units: int,
    n_control_units: int,
    n_pre_periods: Optional[int],
    n_post_periods: Optional[int],
) -> tuple[int, int]:
    has_pre = n_pre_periods is not None
    has_post = n_post_periods is not None
    if has_pre and has_post:
        return int(n_pre_periods), int(n_post_periods)
    if has_pre != has_post:
        raise ValueError(
            "Provide both n_pre_periods and n_post_periods, or provide neither to infer from n."
        )
    return _infer_pre_post_periods(
        n=n,
        n_treated_units=n_treated_units,
        n_control_units=n_control_units,
    )


def _merge_config_with_locked_params(
    *,
    advanced_params: dict[str, Any],
    locked_params: dict[str, Any],
    wrapper_name: str,
) -> dict[str, Any]:
    conflicting = [
        key
        for key, provided_value in advanced_params.items()
        if key in locked_params and provided_value != locked_params[key]
    ]
    if conflicting:
        conflicting_str = ", ".join(sorted(conflicting))
        raise ValueError(
            f"{wrapper_name} does not allow overriding [{conflicting_str}] via advanced_params; "
            "use explicit top-level arguments instead."
        )
    merged = dict(advanced_params)
    merged.update(locked_params)
    return merged


def _build_generator_config(
    *,
    advanced_params: dict[str, Any],
    locked_params: dict[str, Any],
    wrapper_name: str,
) -> PanelDIDGeneratorConfig:
    config_params = _merge_config_with_locked_params(
        advanced_params=advanced_params,
        locked_params=locked_params,
        wrapper_name=wrapper_name,
    )
    return PanelDIDGeneratorConfig(**config_params)


def _rebuild_panel_with_df(df: pd.DataFrame) -> PanelDataDID:
    return PanelDIDGenerator._build_panel_contract(df)


def _hide_internal_oracle_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=list(_DID_INTERNAL_ORACLE_COLS), errors="ignore")


def _apply_include_oracle(df: pd.DataFrame, *, include_oracle: bool) -> pd.DataFrame:
    if include_oracle:
        return df
    return df.drop(columns=list(_DID_ORACLE_COLS), errors="ignore")


def _reorder_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    lead = [
        "unit_id",
        "region",
        "segment",
        "is_treated_unit",
        "treated_time",
        "calendar_time",
        "observed",
        "y",
        "y_cf",
        "tau_realized_true",
        "mu_cf",
        "mu_treated",
        "tau_mean_true",
    ]
    ordered = [c for c in lead if c in df.columns]
    ordered.extend(c for c in _DID_COVARIATE_COLS if c in df.columns and c not in ordered)
    ordered.extend(c for c in df.columns if c not in ordered)
    return df.loc[:, ordered]


def _finalize_output_df(df: pd.DataFrame, *, include_oracle: bool) -> pd.DataFrame:
    out = _hide_internal_oracle_columns(df)
    out = _apply_include_oracle(out, include_oracle=include_oracle)
    return _reorder_output_columns(out)


def _finalize_output(out: PanelOutput, *, include_oracle: bool) -> PanelOutput:
    if isinstance(out, pd.DataFrame):
        return _finalize_output_df(out, include_oracle=include_oracle)
    return _rebuild_panel_with_df(_finalize_output_df(out.df, include_oracle=include_oracle))


def generate_did_data(
    n_treated_units: int = 20,
    n_control_units: int = 60,
    n_pre_periods: int = 24,
    n_post_periods: int = 12,
    treatment_effect_rate: float = 0.08,
    treatment_effect_slope: float = 0.0,
    time_start: int = 1,
    calendar_start: str = "2021-01",
    time_freq: str = "M",
    treated_prefix: str = "treated_market_",
    control_prefix: str = "control_market_",
    random_state: Optional[int] = 42,
    return_panel_data: bool = True,
    include_oracle: bool = True,
    **advanced_params: Any,
) -> PanelOutput:
    """
    Generate a realistic Gaussian DID panel with a common adoption date.

    The outcome is a continuous marketplace-style metric generated from unit
    size, demand seasonality, macro shocks, competition, average order value,
    serially correlated unit noise, and a post-adoption relative lift for all
    treated units. Oracle columns expose the untreated counterfactual and true
    effect on each treated post-treatment row.
    """
    locked_params = dict(
        outcome_distribution="gaussian",
        n_treated_units=n_treated_units,
        n_control_units=n_control_units,
        n_pre_periods=n_pre_periods,
        n_post_periods=n_post_periods,
        treatment_effect_rate=treatment_effect_rate,
        treatment_effect_slope=treatment_effect_slope,
        time_start=time_start,
        calendar_start=calendar_start,
        time_freq=time_freq,
        treated_prefix=treated_prefix,
        control_prefix=control_prefix,
        random_state=random_state,
        return_panel_data=return_panel_data,
    )
    config = _build_generator_config(
        advanced_params=advanced_params,
        locked_params=locked_params,
        wrapper_name="generate_did_data",
    )
    out = PanelDIDGenerator(config).generate()
    return _finalize_output(out, include_oracle=include_oracle)


def generate_did_gamma_data(
    n: int = 960,
    seed: int = 42,
    return_panel_data: bool = True,
    include_oracle: bool = True,
    n_treated_units: int = 20,
    n_control_units: int = 60,
    treatment_effect_rate: float = 0.08,
    treatment_effect_slope: float = 0.0,
    n_pre_periods: Optional[int] = None,
    n_post_periods: Optional[int] = None,
    **advanced_params: Any,
) -> PanelOutput:
    """
    Generate a realistic Gamma DID panel for positive continuous outcomes.

    Preferred usage is explicit `n_pre_periods` and `n_post_periods`. If both
    are omitted, they are inferred from `n`.
    """
    n_pre_periods, n_post_periods = _resolve_pre_post_periods(
        n=n,
        n_treated_units=n_treated_units,
        n_control_units=n_control_units,
        n_pre_periods=n_pre_periods,
        n_post_periods=n_post_periods,
    )
    locked_params = dict(
        outcome_distribution="gamma",
        n_treated_units=n_treated_units,
        n_control_units=n_control_units,
        n_pre_periods=n_pre_periods,
        n_post_periods=n_post_periods,
        treatment_effect_rate=treatment_effect_rate,
        treatment_effect_slope=treatment_effect_slope,
        random_state=seed,
        return_panel_data=return_panel_data,
    )
    config = _build_generator_config(
        advanced_params=advanced_params,
        locked_params=locked_params,
        wrapper_name="generate_did_gamma_data",
    )
    out = PanelDIDGenerator(config).generate()
    return _finalize_output(out, include_oracle=include_oracle)


def generate_did_poisson_data(
    n: int = 960,
    seed: int = 42,
    return_panel_data: bool = True,
    include_oracle: bool = True,
    n_treated_units: int = 20,
    n_control_units: int = 60,
    treatment_effect_rate: float = 0.08,
    treatment_effect_slope: float = 0.0,
    n_pre_periods: Optional[int] = None,
    n_post_periods: Optional[int] = None,
    **advanced_params: Any,
) -> PanelOutput:
    """
    Generate a realistic Poisson DID panel for count outcomes.

    Preferred usage is explicit `n_pre_periods` and `n_post_periods`. If both
    are omitted, they are inferred from `n`.
    """
    n_pre_periods, n_post_periods = _resolve_pre_post_periods(
        n=n,
        n_treated_units=n_treated_units,
        n_control_units=n_control_units,
        n_pre_periods=n_pre_periods,
        n_post_periods=n_post_periods,
    )
    locked_params = dict(
        outcome_distribution="poisson",
        n_treated_units=n_treated_units,
        n_control_units=n_control_units,
        n_pre_periods=n_pre_periods,
        n_post_periods=n_post_periods,
        treatment_effect_rate=treatment_effect_rate,
        treatment_effect_slope=treatment_effect_slope,
        random_state=seed,
        return_panel_data=return_panel_data,
    )
    config = _build_generator_config(
        advanced_params=advanced_params,
        locked_params=locked_params,
        wrapper_name="generate_did_poisson_data",
    )
    out = PanelDIDGenerator(config).generate()
    return _finalize_output(out, include_oracle=include_oracle)


def generate_did_gamma_26_data(
    *,
    seed: int = 42,
    return_panel_data: bool = True,
    include_oracles: bool = True,
    n_treated_units: int = 20,
    n_control_units: int = 60,
    n_pre_periods: Optional[int] = None,
    n_post_periods: Optional[int] = None,
    treatment_effect_rate: float = 0.08,
    treatment_effect_slope: float = 0.0,
    advanced_params: Optional[dict[str, Any]] = None,
) -> PanelOutput:
    """Scenario-style Gamma DID wrapper with Causalis 26 naming."""
    pre = _DEFAULT_DID_PRE_PERIODS if n_pre_periods is None else int(n_pre_periods)
    post = _DEFAULT_DID_POST_PERIODS if n_post_periods is None else int(n_post_periods)
    params = {} if advanced_params is None else dict(advanced_params)
    return generate_did_gamma_data(
        seed=seed,
        return_panel_data=return_panel_data,
        include_oracle=include_oracles,
        n_treated_units=n_treated_units,
        n_control_units=n_control_units,
        n_pre_periods=pre,
        n_post_periods=post,
        treatment_effect_rate=treatment_effect_rate,
        treatment_effect_slope=treatment_effect_slope,
        **params,
    )


def generate_did_poisson_26_data(
    *,
    seed: int = 42,
    return_panel_data: bool = True,
    include_oracles: bool = True,
    n_treated_units: int = 20,
    n_control_units: int = 60,
    n_pre_periods: Optional[int] = None,
    n_post_periods: Optional[int] = None,
    treatment_effect_rate: float = 0.08,
    treatment_effect_slope: float = 0.0,
    advanced_params: Optional[dict[str, Any]] = None,
) -> PanelOutput:
    """Scenario-style Poisson DID wrapper with Causalis 26 naming."""
    pre = _DEFAULT_DID_PRE_PERIODS if n_pre_periods is None else int(n_pre_periods)
    post = _DEFAULT_DID_POST_PERIODS if n_post_periods is None else int(n_post_periods)
    params = {} if advanced_params is None else dict(advanced_params)
    return generate_did_poisson_data(
        seed=seed,
        return_panel_data=return_panel_data,
        include_oracle=include_oracles,
        n_treated_units=n_treated_units,
        n_control_units=n_control_units,
        n_pre_periods=pre,
        n_post_periods=post,
        treatment_effect_rate=treatment_effect_rate,
        treatment_effect_slope=treatment_effect_slope,
        **params,
    )
