from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pandas as pd

from causalis.data_contracts.panel_data_did import PanelDataDID


PanelOutput = Union[pd.DataFrame, PanelDataDID]

_DEFAULT_PRE_PERIODS = 24
_DEFAULT_POST_PERIODS = 8
_DEFAULT_N_COHORTS = 4
_GAMMA_SHAPE = 25.0
_COVARIATE_COLS = (
    "market_traffic",
    "avg_order_value",
    "market_competition",
    "macro_index",
    "seasonality_index",
)
_DID_ADJUSTMENT_COVARIATE_COLS = (
    "market_traffic",
    "avg_order_value",
    "market_competition",
)
_PANEL_DATA_DID_COLS = (
    "unit_id",
    "calendar_time",
    "treated_time",
    "y",
    "region",
    *_COVARIATE_COLS,
)
_ORACLE_COLS = (
    "y_cf",
    "mu_cf",
    "mu_treated",
    "tau_mean_true",
    "tau_realized_true",
    "tau_rate_true",
)
_REGIONS = ("north", "south", "east", "west", "central")
_SEGMENTS = ("core", "growth", "enterprise")


@dataclass(frozen=True)
class _UnitPanel:
    unit_ids: list[str]
    is_ever_treated: np.ndarray
    first_treatment_idx: np.ndarray
    first_treatment_period: list[Optional[pd.Period]]
    regions: list[str]
    segments: list[str]
    log_size: np.ndarray
    growth: np.ndarray
    quality: np.ndarray
    effect_multiplier: np.ndarray


def _validate_positive_int(value: int, name: str) -> int:
    out = int(value)
    if out <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return out


def _validate_nonnegative_float(value: float, name: str) -> float:
    out = float(value)
    if not np.isfinite(out) or out < 0.0:
        raise ValueError(f"{name} must be non-negative and finite.")
    return out


def _validate_float(value: float, name: str) -> float:
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite.")
    return out


def _period_axis(
    *,
    calendar_start: str,
    time_freq: str,
    n_periods: int,
) -> list[pd.Period]:
    try:
        start = pd.Period(calendar_start, freq=time_freq)
        return list(pd.period_range(start=start, periods=n_periods, freq=time_freq))
    except Exception as exc:
        raise ValueError("calendar_start/time_freq must define a valid pandas period axis.") from exc


def _ar1(
    rng: np.random.Generator,
    *,
    n_periods: int,
    rho: float,
    innovation_std: float,
) -> np.ndarray:
    if n_periods <= 0:
        return np.empty(0, dtype=float)
    if innovation_std <= 0.0:
        return np.zeros(n_periods, dtype=float)

    out = np.empty(n_periods, dtype=float)
    out[0] = rng.normal(0.0, innovation_std / np.sqrt(max(1e-12, 1.0 - rho * rho)))
    for idx in range(1, n_periods):
        out[idx] = rho * out[idx - 1] + rng.normal(0.0, innovation_std)
    return out


def _seasonality(t_rel: np.ndarray) -> np.ndarray:
    t = np.asarray(t_rel, dtype=float)
    return np.sin(2.0 * np.pi * t / 12.0) + 0.5 * np.cos(2.0 * np.pi * t / 6.0)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


def _unit_ids(prefix: str, n_units: int) -> list[str]:
    width = max(2, len(str(n_units)))
    return [f"{prefix}{idx + 1:0{width}d}" for idx in range(n_units)]


def _cohort_offsets(*, n_post_periods: int, n_cohorts: int) -> np.ndarray:
    n = min(n_cohorts, n_post_periods)
    if n == 1:
        return np.asarray([0], dtype=int)
    return np.unique(np.rint(np.linspace(0, n_post_periods - 1, n)).astype(int))


def _sample_units(
    rng: np.random.Generator,
    *,
    periods: list[pd.Period],
    n_pre_periods: int,
    n_treated_units: int,
    n_control_units: int,
    n_cohorts: int,
    treated_prefix: str,
    control_prefix: str,
    treatment_effect_heterogeneity_std: float,
    parallel_trend_violation: float,
) -> _UnitPanel:
    treated_ids = _unit_ids(treated_prefix, n_treated_units)
    control_ids = _unit_ids(control_prefix, n_control_units)
    unit_ids = treated_ids + control_ids
    n_units = len(unit_ids)

    is_ever_treated = np.zeros(n_units, dtype=int)
    is_ever_treated[:n_treated_units] = 1

    offsets = _cohort_offsets(n_post_periods=len(periods) - n_pre_periods, n_cohorts=n_cohorts)
    cohort_indices = n_pre_periods + offsets
    treated_order = rng.permutation(n_treated_units)
    treated_splits = np.array_split(treated_order, len(cohort_indices))

    first_treatment_idx = np.full(n_units, -1, dtype=int)
    for cohort_idx, split in zip(cohort_indices, treated_splits):
        first_treatment_idx[split] = int(cohort_idx)
    first_treatment_period = [
        None if idx < 0 else periods[int(idx)] for idx in first_treatment_idx
    ]

    regions = rng.choice(_REGIONS, size=n_units, replace=True).tolist()
    segments = rng.choice(_SEGMENTS, size=n_units, replace=True, p=[0.52, 0.34, 0.14]).tolist()

    segment_size = {"core": 0.00, "growth": -0.16, "enterprise": 0.34}
    segment_quality = {"core": 0.00, "growth": 0.08, "enterprise": 0.18}
    region_quality = {"north": 0.04, "south": -0.03, "east": 0.02, "west": 0.06, "central": -0.01}

    log_size = (
        7.10
        + 0.10 * is_ever_treated
        + np.asarray([segment_size[s] for s in segments], dtype=float)
        + rng.normal(0.0, 0.45, size=n_units)
    )
    growth = rng.normal(0.0035, 0.0020, size=n_units)
    growth += parallel_trend_violation * is_ever_treated
    quality = (
        np.asarray([segment_quality[s] for s in segments], dtype=float)
        + np.asarray([region_quality[r] for r in regions], dtype=float)
        + rng.normal(0.0, 0.14, size=n_units)
    )

    effect_multiplier = np.ones(n_units, dtype=float)
    if treatment_effect_heterogeneity_std > 0.0:
        sigma = treatment_effect_heterogeneity_std
        effect_multiplier[:n_treated_units] = rng.lognormal(
            mean=-0.5 * sigma * sigma,
            sigma=sigma,
            size=n_treated_units,
        )

    return _UnitPanel(
        unit_ids=unit_ids,
        is_ever_treated=is_ever_treated,
        first_treatment_idx=first_treatment_idx,
        first_treatment_period=first_treatment_period,
        regions=regions,
        segments=segments,
        log_size=log_size,
        growth=growth,
        quality=quality,
        effect_multiplier=effect_multiplier,
    )


def _build_panel_contract(df: pd.DataFrame) -> PanelDataDID:
    panel = PanelDataDID(
        df=df,
        y="y",
        unit_col="unit_id",
        time_col="calendar_time",
        treated_time="treated_time",
        covariates=_DID_ADJUSTMENT_COVARIATE_COLS,
        cluster_col="region",
    )
    full_df = df.copy()
    object.__setattr__(panel, "df", full_df)
    object.__setattr__(panel, "_df_validated", full_df.copy(deep=True))
    return panel


def _finalize_output(
    df: pd.DataFrame,
    *,
    return_panel_data: bool,
    include_oracles: bool,
) -> PanelOutput:
    columns = list(_PANEL_DATA_DID_COLS)
    if include_oracles:
        columns.extend(_ORACLE_COLS)
    out = df.loc[:, columns].copy()
    if return_panel_data:
        return _build_panel_contract(out)
    return out


def generate_did_gamma_26(
    *,
    seed: int = 42,
    return_panel_data: bool = True,
    include_oracles: bool = True,
    n_treated_units: int = 200,
    n_control_units: int = 600,
    n_pre_periods: Optional[int] = None,
    n_post_periods: Optional[int] = None,
    n_cohorts: int = _DEFAULT_N_COHORTS,
    treatment_effect_rate: float = 0.08,
    treatment_effect_slope: float = 0.0,
    treatment_effect_heterogeneity_std: float = 0.025,
    parallel_trend_violation: float = 0.0,
    calendar_start: str = "2021-01",
    time_freq: str = "M",
    treated_prefix: str = "treated_market_",
    control_prefix: str = "control_market_",
    **advanced_params,
) -> PanelOutput:
    r"""Generate a staggered-adoption Gamma DID panel for CSA estimators.

    The returned panel is long-format with absorbing treatment, multiple
    first-treatment cohorts, never-treated controls, baseline-compatible
    covariates, contextual time indices, cluster labels, and optional oracle
    counterfactual/effect columns. It is intended to be consumed directly by
    :class:`causalis.scenarios.did.model.CallawaySantAnnaDID`.

    Parameters
    ----------
    seed : int, default 42
        Random seed for reproducibility.
    return_panel_data : bool, default True
        If True, returns a :class:`PanelDataDID` contract; if False, returns a raw
        :class:`pd.DataFrame`.
    include_oracles : bool, default True
        Whether to include ground-truth columns (e.g., ``y_cf``, ``tau_mean_true``).
    n_treated_units : int, default 20
        Number of units that will eventually receive treatment.
    n_control_units : int, default 60
        Number of never-treated units.
    n_pre_periods : int, optional
        Number of periods before any treatment starts. Defaults to 24.
    n_post_periods : int, optional
        Number of periods after the first treatment cohort starts. Defaults to 8.
    n_cohorts : int, default 4
        Number of distinct treatment-start times (cohorts) among treated units.
    treatment_effect_rate : float, default 0.08
        Base treatment effect as a fraction of the counterfactual outcome.
    treatment_effect_slope : float, default 0.0
        The rate at which the treatment effect grows or decays per period after start.
    treatment_effect_heterogeneity_std : float, default 0.025
        Standard deviation of unit-level treatment effect multipliers.
    parallel_trend_violation : float, default 0.0
        Strength of a differential trend between treated and control units (0 = parallel).
    calendar_start : str, default "2021-01"
        The starting period string for the pandas index.
    time_freq : str, default "M"
        The pandas frequency alias (e.g., "M", "W", "D").
    treated_prefix : str, default "treated_market_"
        Prefix for treated unit IDs.
    control_prefix : str, default "control_market_"
        Prefix for control unit IDs.
    **advanced_params : dict
        Reserved for future parameters.

    Returns
    -------
    PanelDataDID or pd.DataFrame
        The generated panel dataset.

    Examples
    --------
    >>> from causalis.scenarios.did.dgp import generate_did_gamma_26
    >>> # Generate default panel data
    >>> data = generate_did_gamma_26(n_treated_units=10, n_control_units=30, seed=123)
    >>> type(data)
    <class 'causalis.data_contracts.panel_data_did.PanelDataDID'>
    >>> # Access the underlying dataframe
    >>> df = data.df
    >>> df[['unit_id', 'calendar_time', 'y', 'treated_time']].head()
    ... # doctest: +SKIP

    Notes
    -----
    The DGP simulates a complex business environment where the outcome :math:`Y_{it}`
    (e.g., revenue) follows a Gamma distribution:

    .. math::

        Y_{it} \sim \text{Gamma}\left( \text{shape}=\kappa, \text{scale}=\frac{\mu_{it}}{\kappa} \right)

    The mean :math:`\mu_{it}` is decomposed into a counterfactual :math:`\mu_{it}(0)` and a
    treatment effect :math:`\tau_{it}`:

    .. math::

        \mu_{it}(1) = \mu_{it}(0) \cdot (1 + \tau_{it})

    The counterfactual mean :math:`\mu_{it}(0)` is modeled as a product of market traffic,
    conversion rate, and average order value (AOV), each subject to macro-economic
    cycles, seasonality, and unit-level trends:

    .. math::

        \ln \mu_{it}(0) = \ln(\text{Exposure}_{it}) + \ln(\text{ConvRate}_{it}) + \ln(\text{AOV}_{it})

    where each component follows an AR(1) process with latent confounding.
    The generated dataframe keeps ``macro_index`` and ``seasonality_index`` as
    contextual time indices, but the returned :class:`PanelDataDID` contract uses
    only unit-varying covariates for DID adjustment.
    The treatment effect :math:`\tau_{it}` for a unit :math:`i` treated at time :math:`G_i` is:

    .. math::

        \tau_{it} = (\theta_{rate} + \theta_{slope} \cdot (t - G_i)) \cdot \text{Ramp}(t - G_i) \cdot \eta_i

    where :math:`\text{Ramp}(\cdot)` is an exponential adoption curve and :math:`\eta_i` is unit-level
    heterogeneity. Parallel trend violations are introduced by adding a differential
    linear trend to treated units' counterfactuals.
    """

    if advanced_params:
        unknown = ", ".join(sorted(advanced_params))
        raise ValueError(f"Unknown advanced_params for generate_did_gamma_26: {unknown}")

    n_treated = _validate_positive_int(n_treated_units, "n_treated_units")
    n_control = _validate_positive_int(n_control_units, "n_control_units")
    pre = _DEFAULT_PRE_PERIODS if n_pre_periods is None else _validate_positive_int(n_pre_periods, "n_pre_periods")
    post = _DEFAULT_POST_PERIODS if n_post_periods is None else _validate_positive_int(n_post_periods, "n_post_periods")
    cohorts = min(_validate_positive_int(n_cohorts, "n_cohorts"), n_treated, post)
    effect_rate = _validate_float(treatment_effect_rate, "treatment_effect_rate")
    effect_slope = _validate_float(treatment_effect_slope, "treatment_effect_slope")
    heterogeneity = _validate_nonnegative_float(
        treatment_effect_heterogeneity_std,
        "treatment_effect_heterogeneity_std",
    )
    parallel_violation = _validate_float(parallel_trend_violation, "parallel_trend_violation")
    if treated_prefix == control_prefix:
        raise ValueError("treated_prefix and control_prefix must be distinct.")
    if not str(time_freq).strip():
        raise ValueError("time_freq must be a non-empty pandas period frequency alias.")

    periods = _period_axis(calendar_start=calendar_start, time_freq=time_freq, n_periods=pre + post)
    rng = np.random.default_rng(seed)
    units = _sample_units(
        rng,
        periods=periods,
        n_pre_periods=pre,
        n_treated_units=n_treated,
        n_control_units=n_control,
        n_cohorts=cohorts,
        treated_prefix=treated_prefix,
        control_prefix=control_prefix,
        treatment_effect_heterogeneity_std=heterogeneity,
        parallel_trend_violation=parallel_violation,
    )

    n_periods = len(periods)
    n_units = len(units.unit_ids)
    t_rel = np.arange(n_periods, dtype=float)
    centered_t = t_rel - t_rel.mean()
    season = _seasonality(t_rel)
    macro_log = _ar1(rng, n_periods=n_periods, rho=0.45, innovation_std=0.05)
    macro_index = np.exp(macro_log)
    seasonality_index = 1.0 + 0.16 * season

    exposure = np.empty((n_periods, n_units), dtype=float)
    avg_order_value = np.empty((n_periods, n_units), dtype=float)
    market_competition = np.empty((n_periods, n_units), dtype=float)
    mu_cf = np.empty((n_periods, n_units), dtype=float)

    segment_aov_shift = {"core": 0.00, "growth": -4.0, "enterprise": 18.0}
    region_competition_shift = {"north": 0.10, "south": -0.05, "east": 0.03, "west": 0.08, "central": -0.02}

    for unit_idx in range(n_units):
        exposure_noise = _ar1(rng, n_periods=n_periods, rho=0.35, innovation_std=0.07)
        log_exposure = (
            units.log_size[unit_idx]
            + units.growth[unit_idx] * centered_t
            + 0.35 * macro_log
            + 0.08 * season
            + exposure_noise
        )
        exposure[:, unit_idx] = np.exp(np.clip(log_exposure, np.log(80.0), np.log(120_000.0)))

        aov_base = 44.0 + segment_aov_shift[units.segments[unit_idx]] + rng.normal(0.0, 5.0)
        aov_noise = _ar1(rng, n_periods=n_periods, rho=0.35, innovation_std=0.025)
        avg_order_value[:, unit_idx] = np.clip(
            aov_base * np.exp(0.03 * season + 0.04 * macro_log + aov_noise),
            8.0,
            180.0,
        )

        comp_noise = _ar1(rng, n_periods=n_periods, rho=0.35, innovation_std=0.10)
        comp_score = (
            -0.25
            + region_competition_shift[units.regions[unit_idx]]
            - 0.15 * units.quality[unit_idx]
            + 0.05 * season
            - 0.20 * macro_log
            + comp_noise
        )
        market_competition[:, unit_idx] = _sigmoid(comp_score)

        outcome_noise = _ar1(rng, n_periods=n_periods, rho=0.35, innovation_std=0.06)
        conversion_log_rate = (
            np.log(0.035)
            + units.quality[unit_idx]
            + 0.0015 * centered_t
            + 0.22 * macro_log
            + 0.16 * season
            - 0.55 * market_competition[:, unit_idx]
            + outcome_noise
        )
        order_mean = exposure[:, unit_idx] * np.exp(np.clip(conversion_log_rate, -8.0, -1.2))
        mu_cf[:, unit_idx] = np.clip(order_mean * avg_order_value[:, unit_idx], 1e-6, None)

    tau_rate_true = np.zeros((n_periods, n_units), dtype=float)
    for unit_idx, first_idx in enumerate(units.first_treatment_idx):
        if first_idx < 0:
            continue
        for period_idx in range(int(first_idx), n_periods):
            event_time = period_idx - int(first_idx)
            ramp = 1.0 - np.exp(-(event_time + 1.0) / 2.5)
            rate = (effect_rate + effect_slope * event_time) * ramp * units.effect_multiplier[unit_idx]
            if 1.0 + rate <= 0.0:
                raise ValueError(
                    "treatment_effect_rate/slope imply non-positive treated potential outcomes."
                )
            tau_rate_true[period_idx, unit_idx] = rate

    mu_treated_full = mu_cf * (1.0 + tau_rate_true)
    tau_mean_full = mu_treated_full - mu_cf
    y_cf = rng.gamma(shape=_GAMMA_SHAPE, scale=mu_cf / _GAMMA_SHAPE)
    y_treated = y_cf * np.divide(
        mu_treated_full,
        mu_cf,
        out=np.ones_like(mu_cf),
        where=mu_cf > 0.0,
    )

    rows = []
    for period_idx, period in enumerate(periods):
        for unit_idx, unit_id in enumerate(units.unit_ids):
            first_idx = int(units.first_treatment_idx[unit_idx])
            treated_time = int(first_idx >= 0 and period_idx >= first_idx)
            observed_y = y_treated[period_idx, unit_idx] if treated_time else y_cf[period_idx, unit_idx]
            realized_tau = y_treated[period_idx, unit_idx] - y_cf[period_idx, unit_idx] if treated_time else 0.0
            cohort = units.first_treatment_period[unit_idx]
            rows.append(
                {
                    "unit_id": unit_id,
                    "region": units.regions[unit_idx],
                    "segment": units.segments[unit_idx],
                    "is_treated_unit": int(units.is_ever_treated[unit_idx]),
                    "cohort": cohort,
                    "calendar_time": period,
                    "event_time": period_idx - first_idx if first_idx >= 0 else pd.NA,
                    "observed": 1,
                    "treated_time": treated_time,
                    "y": float(observed_y),
                    "y_cf": float(y_cf[period_idx, unit_idx]),
                    "tau_realized_true": float(realized_tau),
                    "mu_cf": float(mu_cf[period_idx, unit_idx]),
                    "mu_treated": float(mu_treated_full[period_idx, unit_idx] if treated_time else mu_cf[period_idx, unit_idx]),
                    "tau_mean_true": float(tau_mean_full[period_idx, unit_idx] if treated_time else 0.0),
                    "tau_rate_true": float(tau_rate_true[period_idx, unit_idx] if treated_time else 0.0),
                    "market_traffic": float(exposure[period_idx, unit_idx] / 1000.0),
                    "avg_order_value": float(avg_order_value[period_idx, unit_idx]),
                    "market_competition": float(market_competition[period_idx, unit_idx]),
                    "macro_index": float(macro_index[period_idx]),
                    "seasonality_index": float(seasonality_index[period_idx]),
                }
            )

    df = pd.DataFrame(rows).sort_values(["unit_id", "calendar_time"]).reset_index(drop=True)
    return _finalize_output(df, return_panel_data=bool(return_panel_data), include_oracles=bool(include_oracles))


generate_staggered_did_gamma_26 = generate_did_gamma_26


__all__ = ["generate_did_gamma_26", "generate_staggered_did_gamma_26"]
