from __future__ import annotations

from typing import Any, Hashable, Literal, Optional, Union

import numpy as np
import pandas as pd

from causalis.data_contracts.panel_data_scm import PanelDataSCM
from .base import PanelSCMGenerator, PanelSCMGeneratorConfig


def _infer_pre_post_periods(
    *,
    n: int,
    n_donors: int,
    pre_share: float = 0.75,
    min_periods: int = 12,
    min_post_periods: int = 4,
) -> tuple[int, int]:
    if n <= 0:
        raise ValueError("n must be > 0.")
    if n_donors < 1:
        raise ValueError("n_donors must be >= 1.")
    if not (0.0 < pre_share < 1.0):
        raise ValueError("pre_share must be in (0, 1).")

    n_units = n_donors + 1
    periods = int(max(min_periods, round(n / n_units)))
    n_post = int(max(min_post_periods, round((1.0 - pre_share) * periods)))
    n_post = min(n_post, periods - 1)
    n_pre = periods - n_post
    return n_pre, n_post


def _resolve_pre_post_periods(
    *,
    n: int,
    n_donors: int,
    n_pre_periods: Optional[int],
    n_post_periods: Optional[int],
) -> tuple[int, int]:
    """Resolve panel horizon from explicit periods or inferred total-size target."""
    has_pre = n_pre_periods is not None
    has_post = n_post_periods is not None
    if has_pre and has_post:
        return int(n_pre_periods), int(n_post_periods)
    if has_pre != has_post:
        raise ValueError(
            "Provide both n_pre_periods and n_post_periods, or provide neither to infer from n."
        )
    return _infer_pre_post_periods(n=n, n_donors=n_donors)


def _inject_donor_missing_periods(
    *,
    df: pd.DataFrame,
    treated_unit: Hashable,
    random_state: int,
    donor_missing_block_frac: float,
    donor_missing_block_min_len: int,
    donor_missing_block_max_len: Optional[int],
) -> pd.DataFrame:
    """Inject contiguous missing-outcome periods for donor units only."""
    if donor_missing_block_frac <= 0.0:
        return df
    if not (0.0 <= donor_missing_block_frac < 1.0):
        raise ValueError("donor_missing_block_frac must be in [0, 1).")
    if donor_missing_block_min_len < 1:
        raise ValueError("donor_missing_block_min_len must be >= 1.")
    if donor_missing_block_max_len is not None and donor_missing_block_max_len < 1:
        raise ValueError("donor_missing_block_max_len must be >= 1 when provided.")
    if (
        donor_missing_block_max_len is not None
        and donor_missing_block_max_len < donor_missing_block_min_len
    ):
        raise ValueError("donor_missing_block_max_len must be >= donor_missing_block_min_len.")

    out = df.copy()
    donors = [u for u in out["unit_id"].unique().tolist() if u != treated_unit]
    if not donors:
        return out

    rng = np.random.default_rng(random_state)
    donor_mask = out["unit_id"] != treated_unit
    n_target = int(round(donor_missing_block_frac * int(donor_mask.sum())))
    if n_target <= 0:
        return out

    donor_idx = {
        unit: out[out["unit_id"] == unit].sort_values("calendar_time").index.to_numpy(dtype=int)
        for unit in donors
    }
    protected_set = {int(idx_arr[0]) for idx_arr in donor_idx.values() if idx_arr.size > 0}
    miss_set: set[int] = set()
    n_tries = max(100, 25 * n_target)
    for _ in range(n_tries):
        if len(miss_set) >= n_target:
            break
        unit = donors[int(rng.integers(0, len(donors)))]
        idx = donor_idx[unit]
        n_unit = int(idx.size)
        if n_unit <= 1:
            continue
        min_len = int(min(max(1, donor_missing_block_min_len), n_unit))
        max_len_candidate = n_unit if donor_missing_block_max_len is None else int(donor_missing_block_max_len)
        max_len = int(min(max_len_candidate, n_unit))
        if max_len < min_len:
            continue

        block_len = int(rng.integers(min_len, max_len + 1))
        start = int(rng.integers(0, n_unit - block_len + 1))
        for idx_i in idx[start : start + block_len]:
            idx_int = int(idx_i)
            if idx_int in protected_set:
                continue
            miss_set.add(idx_int)
            if len(miss_set) >= n_target:
                break

    if miss_set:
        out.loc[list(miss_set), "y"] = np.nan
    return out


def _hide_internal_oracle_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop internal oracle diagnostics that should not be exposed in DGP outputs."""
    return df.drop(columns=["tau_rate_true"], errors="ignore")


def _reorder_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Apply a canonical output schema order for public DGP returns."""
    lead = [
        "unit_id",
        "is_treated_unit",
        "calendar_time",
        "observed",
        "y",
        "y_cf",
        "tau_realized_true",
    ]
    ordered = [c for c in lead if c in df.columns]
    ordered.extend(c for c in df.columns if c not in ordered)
    return df.loc[:, ordered]


def _finalize_output_df(df: pd.DataFrame) -> pd.DataFrame:
    """Hide internal diagnostics and enforce public column ordering."""
    return _reorder_output_columns(_hide_internal_oracle_columns(df))


def _merge_config_with_locked_params(
    *,
    advanced_params: dict[str, Any],
    locked_params: dict[str, Any],
    wrapper_name: str,
) -> dict[str, Any]:
    """Merge advanced params while preserving wrapper-level API intent."""
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


def generate_scm_data(
    n_donors: int = 5,
    n_pre_periods: int = 20,
    n_post_periods: int = 10,
    treatment_effect: float = 2.0,
    treatment_effect_slope: float = 0.0,
    donor_noise_std: float = 0.20,
    treated_noise_std: float = 0.10,
    common_factor_std: float = 0.15,
    time_start: int = 1,
    treated_unit: Hashable = "treated",
    donor_prefix: str = "donor_",
    random_state: Optional[int] = 42,
    missing_outcome_frac: float = 0.0,
    missing_cell_frac: float = 0.0,
    return_panel_data: bool = True,
    dirichlet_alpha: float = 1.0,
    rho_common: float = 0.0,
    rho_donor: float = 0.0,
    n_latent_factors: int = 0,
    latent_factor_std: float = 0.20,
    latent_loading_std: float = 0.35,
    rho_latent: float = 0.0,
    prefit_mismatch_std: float = 0.0,
    rho_prefit_mismatch: float = 0.0,
    missing_block_frac: float = 0.0,
    missing_block_min_len: int = 2,
    missing_block_max_len: Optional[int] = None,
    protect_treated_pre: bool = False,
    protect_treated_post: bool = False,
    treatment_effect_mode: Literal["additive", "multiplicative"] = "additive",
) -> Union[pd.DataFrame, PanelDataSCM]:
    """Medium-level wrapper for Gaussian SCM panel generation."""
    config = PanelSCMGeneratorConfig(
        outcome_distribution="gaussian",
        n_donors=n_donors,
        n_pre_periods=n_pre_periods,
        n_post_periods=n_post_periods,
        treatment_effect=treatment_effect,
        treatment_effect_slope=treatment_effect_slope,
        donor_noise_std=donor_noise_std,
        treated_noise_std=treated_noise_std,
        common_factor_std=common_factor_std,
        time_start=time_start,
        treated_unit=treated_unit,
        donor_prefix=donor_prefix,
        random_state=random_state,
        missing_outcome_frac=missing_outcome_frac,
        missing_cell_frac=missing_cell_frac,
        return_panel_data=return_panel_data,
        dirichlet_alpha=dirichlet_alpha,
        rho_common=rho_common,
        rho_donor=rho_donor,
        n_latent_factors=n_latent_factors,
        latent_factor_std=latent_factor_std,
        latent_loading_std=latent_loading_std,
        rho_latent=rho_latent,
        prefit_mismatch_std=prefit_mismatch_std,
        rho_prefit_mismatch=rho_prefit_mismatch,
        missing_block_frac=missing_block_frac,
        missing_block_min_len=missing_block_min_len,
        missing_block_max_len=missing_block_max_len,
        protect_treated_pre=protect_treated_pre,
        protect_treated_post=protect_treated_post,
        treatment_effect_mode=treatment_effect_mode,
    )
    out = PanelSCMGenerator(config).generate()
    if isinstance(out, pd.DataFrame):
        return _finalize_output_df(out)
    return out.model_copy(update={"df": _finalize_output_df(out.df)})


def generate_scm_gamma_data(
    n: int = 432,
    seed: int = 42,
    return_panel_data: bool = True,
    n_donors: int = 8,
    treatment_effect_rate: float = 0.12,
    treatment_effect_slope: float = 0.01,
    missing_outcome_frac: float = 0.0,
    n_pre_periods: Optional[int] = None,
    n_post_periods: Optional[int] = None,
    **advanced_params,
) -> Union[pd.DataFrame, PanelDataSCM]:
    """
    Medium-level wrapper for realistic Gamma SCM panel generation.

    Preferred usage is explicit `n_pre_periods` and `n_post_periods`. If both
    are omitted, they are inferred from `n`.
    The post-treatment effect path uses a ramp-in: at the first post period, the
    effective relative lift is
    `treatment_effect_rate * (1 - exp(-1 / 2.5))` (about 0.33x of the parameter
    when slope is zero).
    """
    n_pre_periods, n_post_periods = _resolve_pre_post_periods(
        n=n,
        n_donors=n_donors,
        n_pre_periods=n_pre_periods,
        n_post_periods=n_post_periods,
    )

    locked_params = dict(
        outcome_distribution="gamma",
        n_donors=n_donors,
        n_pre_periods=n_pre_periods,
        n_post_periods=n_post_periods,
        treatment_effect_rate=treatment_effect_rate,
        treatment_effect_slope=treatment_effect_slope,
        random_state=seed,
        missing_outcome_frac=missing_outcome_frac,
        return_panel_data=return_panel_data,
    )
    config_params = _merge_config_with_locked_params(
        advanced_params=advanced_params,
        locked_params=locked_params,
        wrapper_name="generate_scm_gamma_data",
    )

    config = PanelSCMGeneratorConfig(**config_params)
    out = PanelSCMGenerator(config).generate()
    if isinstance(out, pd.DataFrame):
        return _finalize_output_df(out)
    return out.model_copy(update={"df": _finalize_output_df(out.df)})


def generate_scm_poisson_data(
    n: int = 432,
    seed: int = 42,
    return_panel_data: bool = True,
    n_donors: int = 8,
    treatment_effect_rate: float = 0.10,
    treatment_effect_slope: float = 0.005,
    donor_missing_block_frac: float = 0.08,
    donor_missing_block_min_len: int = 2,
    donor_missing_block_max_len: Optional[int] = 4,
    n_pre_periods: Optional[int] = None,
    n_post_periods: Optional[int] = None,
    **advanced_params,
) -> Union[pd.DataFrame, PanelDataSCM]:
    """
    Medium-level wrapper for realistic Poisson SCM panel generation.

    Preferred usage is explicit `n_pre_periods` and `n_post_periods`. If both
    are omitted, they are inferred from `n`. Default behavior injects donor-only
    missing periods, keeping treated post periods observed so
    RobustSyntheticControl can be exercised reliably.
    The post-treatment effect path uses a ramp-in: at the first post period, the
    effective relative lift is
    `treatment_effect_rate * (1 - exp(-1 / 2.5))` (about 0.33x of the parameter
    when slope is zero).
    """
    n_pre_periods, n_post_periods = _resolve_pre_post_periods(
        n=n,
        n_donors=n_donors,
        n_pre_periods=n_pre_periods,
        n_post_periods=n_post_periods,
    )

    locked_params = dict(
        outcome_distribution="poisson",
        n_donors=n_donors,
        n_pre_periods=n_pre_periods,
        n_post_periods=n_post_periods,
        treatment_effect_rate=treatment_effect_rate,
        treatment_effect_slope=treatment_effect_slope,
        random_state=seed,
        # We inject donor-only missingness below.
        missing_outcome_frac=0.0,
        missing_cell_frac=0.0,
        missing_block_frac=0.0,
        return_panel_data=False,
        protect_treated_post=True,
    )
    config_params = _merge_config_with_locked_params(
        advanced_params=advanced_params,
        locked_params=locked_params,
        wrapper_name="generate_scm_poisson_data",
    )

    config = PanelSCMGeneratorConfig(**config_params)
    df = PanelSCMGenerator(config).generate(return_panel_data=False)

    df = _inject_donor_missing_periods(
        df=df,
        treated_unit=config.treated_unit,
        random_state=seed + 11_813,
        donor_missing_block_frac=donor_missing_block_frac,
        donor_missing_block_min_len=donor_missing_block_min_len,
        donor_missing_block_max_len=donor_missing_block_max_len,
    )
    df = df.sort_values(["unit_id", "calendar_time"]).reset_index(drop=True)
    df["observed"] = (~df["y"].isna()).astype(int)
    df = _finalize_output_df(df)

    if not return_panel_data:
        return df

    donor_names = [f"{config.donor_prefix}{j + 1}" for j in range(config.n_donors)]
    treatment_start = (
        pd.Period(config.calendar_start, freq=config.time_freq)
        + (int(config.time_start) - 1 + int(config.n_pre_periods))
    )
    panel = PanelDataSCM(
        df=df,
        unit_col="unit_id",
        time_col="calendar_time",
        time_freq=config.time_freq,
        y="y",
        treated_unit=config.treated_unit,
        treatment_start=treatment_start,
        donor_units=donor_names,
        covariate_cols=("exposure", "macro_index", "seasonality_index"),
        observed_col="observed",
        allow_missing_outcome=True,
    )
    return panel.model_copy(update={"df": df})
