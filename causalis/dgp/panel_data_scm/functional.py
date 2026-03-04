from __future__ import annotations

from typing import Any, Hashable, Literal, Optional, Union

import numpy as np
import pandas as pd

from causalis.data_contracts.panel_data_scm import PanelDataSCM
from .base import PanelSCMGenerator, PanelSCMGeneratorConfig

PanelOutput = Union[pd.DataFrame, PanelDataSCM]

_SCM26_ORACLE_COLS = (
    "is_treated_unit",
    "y_cf",
    "tau_realized_true",
    "mu_cf",
    "mu_treated",
    "tau_mean_true",
)
_SCM26_COVARIATE_COLS = ("exposure", "macro_index", "seasonality_index")
_DEFAULT_SCM26_PRE_PERIODS = 36
_DEFAULT_SCM26_POST_PERIODS = 12


def _build_panel_from_output_df(df: pd.DataFrame) -> PanelDataSCM:
    return PanelDataSCM(
        df=df,
        y="y",
        unit_col="unit_id",
        time_col="calendar_time",
        treated_time="treated_time",
    )


def _rebuild_panel_with_df(panel: PanelDataSCM, df: pd.DataFrame) -> PanelDataSCM:
    return PanelDataSCM(
        df=df,
        y=panel.y,
        unit_col=panel.unit_col,
        time_col=panel.time_col,
        treated_time=panel.treated_time,
    )


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
        "treated_time",
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


def _finalize_output(out: PanelOutput) -> PanelOutput:
    """Finalize outputs uniformly across DataFrame and PanelDataSCM return types."""
    if isinstance(out, pd.DataFrame):
        return _finalize_output_df(out)
    return _rebuild_panel_with_df(out, _finalize_output_df(out.df))


def _panel_from_dataframe(
    *,
    df: pd.DataFrame,
    config: PanelSCMGeneratorConfig,
) -> PanelDataSCM:
    out_df = df.copy()
    if "treated_time" not in out_df.columns:
        treatment_start = (
            pd.Period(config.calendar_start, freq=config.time_freq)
            + (int(config.time_start) - 1 + int(config.n_pre_periods))
        )
        out_df["treated_time"] = (
            (out_df["unit_id"] == config.treated_unit) & (out_df["calendar_time"] >= treatment_start)
        ).astype(int)
    return _build_panel_from_output_df(out_df)


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
) -> PanelOutput:
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
    return _finalize_output(out)


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
) -> PanelOutput:
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
    return _finalize_output(out)


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
) -> PanelOutput:
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

    return _panel_from_dataframe(
        df=df,
        config=config,
    )


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


def _expand_scm26_periods_with_anchor(
    *,
    n_pre_periods: int,
    n_post_periods: int,
) -> tuple[int, int]:
    # Scenario wrappers include one explicit intervention-anchor period.
    return int(n_pre_periods) + 1, int(n_post_periods)


def _apply_scm26_anchor_period_windows(
    out: PanelOutput,
    *,
    n_pre_periods: int,
    n_post_periods: int,
) -> PanelOutput:
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
    return out


def _derive_scm26_treatment_start(
    out: PanelOutput,
    *,
    n_pre_periods: int,
) -> Any:
    out_df = out if isinstance(out, pd.DataFrame) else out.df

    if "calendar_time" not in out_df.columns:
        raise RuntimeError("Expected 'calendar_time' column to derive treatment_start.")

    time_values = sorted(pd.Index(out_df["calendar_time"].unique()).tolist())
    post_start_idx = int(n_pre_periods) + 1
    if post_start_idx >= len(time_values):
        raise RuntimeError(
            "Internal period mapping error: cannot derive treatment_start from calendar_time."
        )
    return time_values[post_start_idx]


def _with_scm26_treated_time(
    out: PanelOutput,
    *,
    n_pre_periods: int,
) -> PanelOutput:
    treatment_start = _derive_scm26_treatment_start(out, n_pre_periods=n_pre_periods)

    if isinstance(out, pd.DataFrame):
        if "calendar_time" not in out.columns:
            raise RuntimeError("Expected 'calendar_time' column to derive treated_time.")
        if "is_treated_unit" in out.columns:
            treated_rows = out[out["is_treated_unit"].astype(int) == 1]
            treated_units = pd.Index(treated_rows["unit_id"].unique()).tolist()
            if len(treated_units) != 1:
                raise RuntimeError(
                    "Expected exactly one treated unit in 'is_treated_unit' when deriving treated_time."
                )
            treated_unit = treated_units[0]
            treated_unit_mask = out["unit_id"] == treated_unit
        else:
            raise RuntimeError("Expected 'is_treated_unit' column to derive treated_time.")

        out_df = out.copy()
        out_df["treated_time"] = (
            treated_unit_mask & (out_df["calendar_time"] >= treatment_start)
        ).astype(int)
        return out_df

    df = out.df.copy()
    df["treated_time"] = (
        (df[out.unit_col] == out.treated_unit) & (df[out.time_col] >= treatment_start)
    ).astype(int)
    return _rebuild_panel_with_df(out, df)


def _apply_scm26_include_oracles(
    out: PanelOutput,
    *,
    include_oracles: bool,
) -> PanelOutput:
    drop_cols = list(_SCM26_COVARIATE_COLS)
    if not include_oracles:
        drop_cols.extend(_SCM26_ORACLE_COLS)
    if isinstance(out, pd.DataFrame):
        return out.drop(columns=drop_cols, errors="ignore")
    df = out.df.drop(columns=drop_cols, errors="ignore")
    return _rebuild_panel_with_df(out, df)


def _format_scm26_output_columns(out: PanelOutput) -> PanelOutput:
    ordered = [
        "unit_id",
        "calendar_time",
        "treated_time",
        "observed",
        "y",
        "y_cf",
        "tau_realized_true",
        "mu_cf",
        "mu_treated",
        "tau_mean_true",
    ]
    if isinstance(out, pd.DataFrame):
        df = out.drop(columns=["is_treated_unit"], errors="ignore")
        columns = [col for col in ordered if col in df.columns]
        columns.extend(col for col in df.columns if col not in columns)
        return df.loc[:, columns]

    df = out.df.drop(columns=["is_treated_unit"], errors="ignore")
    columns = [col for col in ordered if col in df.columns]
    columns.extend(col for col in df.columns if col not in columns)
    return _rebuild_panel_with_df(out, df.loc[:, columns])


def _postprocess_scm26_output(
    out: PanelOutput,
    *,
    n_pre_periods: int,
    n_post_periods: int,
    include_oracles: bool,
) -> PanelOutput:
    out = _apply_scm26_anchor_period_windows(
        out,
        n_pre_periods=n_pre_periods,
        n_post_periods=n_post_periods,
    )
    out = _with_scm26_treated_time(out, n_pre_periods=n_pre_periods)
    out = _apply_scm26_include_oracles(out, include_oracles=include_oracles)
    return _format_scm26_output_columns(out)


def generate_scm_gamma_26_data(
    *,
    seed: int,
    return_panel_data: bool,
    include_oracles: bool,
    n_donors: int,
    n_pre_periods: Optional[int],
    n_post_periods: Optional[int],
    treatment_effect_rate: float,
    treatment_effect_slope: float,
    missing_outcome_frac: float,
    advanced_params: dict[str, Any],
) -> PanelOutput:
    n_pre_periods_resolved, n_post_periods_resolved = _resolve_scm26_periods(
        n_pre_periods=n_pre_periods,
        n_post_periods=n_post_periods,
    )
    n_pre_effective, n_post_effective = _expand_scm26_periods_with_anchor(
        n_pre_periods=n_pre_periods_resolved,
        n_post_periods=n_post_periods_resolved,
    )
    n_total_target = int((n_donors + 1) * (n_pre_effective + n_post_effective))
    out = generate_scm_gamma_data(
        n=n_total_target,
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
    return _postprocess_scm26_output(
        out,
        n_pre_periods=n_pre_periods_resolved,
        n_post_periods=n_post_periods_resolved,
        include_oracles=include_oracles,
    )


def generate_scm_poisson_26_data(
    *,
    seed: int,
    return_panel_data: bool,
    include_oracles: bool,
    n_donors: int,
    n_pre_periods: Optional[int],
    n_post_periods: Optional[int],
    treatment_effect_rate: float,
    treatment_effect_slope: float,
    donor_missing_block_frac: float,
    advanced_params: dict[str, Any],
) -> PanelOutput:
    n_pre_periods_resolved, n_post_periods_resolved = _resolve_scm26_periods(
        n_pre_periods=n_pre_periods,
        n_post_periods=n_post_periods,
    )
    n_pre_effective, n_post_effective = _expand_scm26_periods_with_anchor(
        n_pre_periods=n_pre_periods_resolved,
        n_post_periods=n_post_periods_resolved,
    )
    n_total_target = int((n_donors + 1) * (n_pre_effective + n_post_effective))
    out = generate_scm_poisson_data(
        n=n_total_target,
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
    return _postprocess_scm26_output(
        out,
        n_pre_periods=n_pre_periods_resolved,
        n_post_periods=n_post_periods_resolved,
        include_oracles=include_oracles,
    )
