from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd

from causalis.data_contracts.panel_data_did import PanelDataDID


def _draw_ar1_series(
    *,
    rng: np.random.Generator,
    n_periods: int,
    rho: float,
    innovation_std: float,
) -> np.ndarray:
    """Draw an AR(1) series with approximately stationary marginal scale."""
    if n_periods <= 0:
        return np.empty(0, dtype=float)
    if innovation_std <= 0.0:
        return np.zeros(n_periods, dtype=float)

    out = np.empty(n_periods, dtype=float)
    init_std = innovation_std / np.sqrt(max(1e-12, 1.0 - rho * rho))
    out[0] = float(rng.normal(0.0, init_std))
    for t in range(1, n_periods):
        out[t] = float(rho * out[t - 1] + rng.normal(0.0, innovation_std))
    return out


def _monthly_seasonality_signal(t_rel: np.ndarray) -> np.ndarray:
    """Calendar-like seasonality for monthly panels."""
    t = np.asarray(t_rel, dtype=float)
    return np.sin(2.0 * np.pi * t / 12.0) + 0.5 * np.cos(2.0 * np.pi * t / 6.0)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def _sample_coupled_poisson_pair(
    *,
    rng: np.random.Generator,
    mu_cf: np.ndarray,
    mu_treated: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample coupled Poisson potential outcomes with exact Poisson marginals."""
    if mu_cf.shape != mu_treated.shape:
        raise ValueError("mu_cf and mu_treated must share the same shape.")

    mu0 = np.clip(np.asarray(mu_cf, dtype=float), 1e-12, None)
    mu1 = np.clip(np.asarray(mu_treated, dtype=float), 1e-12, None)
    y0 = rng.poisson(mu0).astype(np.int64)
    y1 = np.empty_like(y0)

    up_mask = mu1 >= mu0
    if bool(np.any(up_mask)):
        delta = rng.poisson(mu1[up_mask] - mu0[up_mask]).astype(np.int64)
        y1[up_mask] = y0[up_mask] + delta
    if bool(np.any(~up_mask)):
        down = ~up_mask
        retain_prob = np.divide(
            mu1[down],
            mu0[down],
            out=np.zeros_like(mu1[down], dtype=float),
            where=mu0[down] > 0.0,
        )
        y1[down] = rng.binomial(y0[down], np.clip(retain_prob, 0.0, 1.0)).astype(np.int64)
    return y0.astype(float), y1.astype(float)


@dataclass(frozen=True)
class PanelDIDGeneratorConfig:
    # Panel shape / IDs
    n_treated_units: int = 20
    n_control_units: int = 60
    n_pre_periods: int = 24
    n_post_periods: int = 12
    time_start: int = 1
    time_freq: str = "M"
    calendar_start: str = "2021-01"
    treated_prefix: str = "treated_market_"
    control_prefix: str = "control_market_"
    random_state: Optional[int] = 42
    return_panel_data: bool = True

    # Outcome family
    outcome_distribution: Literal["gaussian", "gamma", "poisson"] = "gaussian"
    gamma_shape: float = 9.0

    # Realism knobs
    exposure_log_mean: float = 7.10
    exposure_log_std: float = 0.45
    treated_selection_shift: float = 0.12
    common_factor_std_log: float = 0.05
    unit_noise_std_log: float = 0.07
    outcome_noise_std_log: float = 0.06
    gaussian_noise_std: float = 90.0
    rho_common: float = 0.45
    rho_unit: float = 0.35
    seasonality_strength: float = 0.16
    parallel_trend_violation: float = 0.0

    # Relative treatment effect path on the natural outcome scale.
    treatment_effect_rate: float = 0.08
    treatment_effect_slope: float = 0.0
    treatment_effect_heterogeneity_std: float = 0.025


@dataclass(frozen=True)
class _UnitComponents:
    unit_ids: list[str]
    is_treated_unit: np.ndarray
    regions: list[str]
    segments: list[str]
    unit_log_size: np.ndarray
    unit_growth: np.ndarray
    unit_quality: np.ndarray
    unit_effect_multiplier: np.ndarray


@dataclass(frozen=True)
class _PanelComponents:
    calendar_times: list[pd.Period]
    treatment_start: pd.Period
    unit_components: _UnitComponents
    exposure: np.ndarray
    avg_order_value: np.ndarray
    market_competition: np.ndarray
    macro_index: np.ndarray
    seasonality_index: np.ndarray
    mu_cf: np.ndarray
    mu_treated: np.ndarray
    tau_mean_true: np.ndarray
    tau_rate_true: np.ndarray


class PanelDIDGenerator:
    """Low-level generator for realistic simultaneous-adoption DID panels."""

    covariate_cols: tuple[str, ...] = (
        "exposure",
        "avg_order_value",
        "market_competition",
        "macro_index",
        "seasonality_index",
    )
    cluster_col: str = "region"

    _regions: tuple[str, ...] = ("north", "south", "east", "west", "central")
    _segments: tuple[str, ...] = ("core", "growth", "enterprise")

    def __init__(self, config: PanelDIDGeneratorConfig):
        self.config = config
        self._validate_config()

    def _validate_config(self) -> None:
        c = self.config
        if c.n_treated_units < 1:
            raise ValueError("n_treated_units must be >= 1.")
        if c.n_control_units < 1:
            raise ValueError("n_control_units must be >= 1.")
        if c.n_pre_periods < 1:
            raise ValueError("n_pre_periods must be >= 1.")
        if c.n_post_periods < 1:
            raise ValueError("n_post_periods must be >= 1.")
        if c.time_start < 1:
            raise ValueError("time_start must be >= 1.")
        if not str(c.time_freq).strip():
            raise ValueError("time_freq must be a non-empty pandas period frequency alias.")
        try:
            start_period = pd.Period(c.calendar_start, freq=c.time_freq)
            _ = pd.period_range(start=start_period, periods=1, freq=c.time_freq)
        except Exception as exc:
            raise ValueError("calendar_start/time_freq must define a valid pandas period axis.") from exc
        if c.treated_prefix == c.control_prefix:
            raise ValueError("treated_prefix and control_prefix must be distinct.")
        if c.outcome_distribution not in {"gaussian", "gamma", "poisson"}:
            raise ValueError("outcome_distribution must be 'gaussian', 'gamma', or 'poisson'.")
        if c.gamma_shape <= 0.0:
            raise ValueError("gamma_shape must be > 0.")
        for name in (
            "exposure_log_std",
            "common_factor_std_log",
            "unit_noise_std_log",
            "outcome_noise_std_log",
            "gaussian_noise_std",
            "seasonality_strength",
            "treatment_effect_heterogeneity_std",
        ):
            if getattr(c, name) < 0.0:
                raise ValueError(f"{name} must be >= 0.")
        for rho_name in ("rho_common", "rho_unit"):
            rho = getattr(c, rho_name)
            if not (-1.0 < rho < 1.0):
                raise ValueError(f"{rho_name} must be in (-1, 1).")

    def _n_total_periods(self) -> int:
        c = self.config
        return int(c.n_pre_periods + c.n_post_periods)

    def _calendar_axis(self) -> tuple[list[pd.Period], pd.Period]:
        c = self.config
        base = pd.Period(c.calendar_start, freq=c.time_freq)
        start = base + (int(c.time_start) - 1)
        calendar_times = list(
            pd.period_range(start=start, periods=self._n_total_periods(), freq=c.time_freq)
        )
        treatment_start = calendar_times[int(c.n_pre_periods)]
        return calendar_times, treatment_start

    def _unit_ids(self) -> tuple[list[str], list[str]]:
        c = self.config
        treated_width = max(2, len(str(c.n_treated_units)))
        control_width = max(2, len(str(c.n_control_units)))
        treated = [f"{c.treated_prefix}{i + 1:0{treated_width}d}" for i in range(c.n_treated_units)]
        controls = [f"{c.control_prefix}{i + 1:0{control_width}d}" for i in range(c.n_control_units)]
        return treated, controls

    def _sample_unit_components(self, *, rng: np.random.Generator) -> _UnitComponents:
        c = self.config
        treated_ids, control_ids = self._unit_ids()
        unit_ids = treated_ids + control_ids
        n_units = len(unit_ids)
        is_treated_unit = np.zeros(n_units, dtype=int)
        is_treated_unit[: c.n_treated_units] = 1

        regions = rng.choice(self._regions, size=n_units, replace=True).tolist()
        segments = rng.choice(self._segments, size=n_units, replace=True, p=[0.52, 0.34, 0.14]).tolist()

        segment_size = {"core": 0.00, "growth": -0.18, "enterprise": 0.35}
        segment_quality = {"core": 0.00, "growth": 0.08, "enterprise": 0.18}
        region_quality = {"north": 0.04, "south": -0.03, "east": 0.02, "west": 0.06, "central": -0.01}

        segment_size_shift = np.array([segment_size[s] for s in segments], dtype=float)
        segment_quality_shift = np.array([segment_quality[s] for s in segments], dtype=float)
        region_quality_shift = np.array([region_quality[r] for r in regions], dtype=float)

        unit_log_size = (
            c.exposure_log_mean
            + c.treated_selection_shift * is_treated_unit
            + segment_size_shift
            + rng.normal(0.0, c.exposure_log_std, size=n_units)
        )
        unit_growth = rng.normal(0.0035, 0.0020, size=n_units)
        unit_growth += c.parallel_trend_violation * is_treated_unit
        unit_quality = (
            segment_quality_shift
            + region_quality_shift
            + rng.normal(0.0, 0.14, size=n_units)
        )

        hetero = np.ones(n_units, dtype=float)
        if c.treatment_effect_heterogeneity_std > 0.0:
            sigma = c.treatment_effect_heterogeneity_std
            hetero[: c.n_treated_units] = rng.lognormal(
                mean=-0.5 * sigma * sigma,
                sigma=sigma,
                size=c.n_treated_units,
            )

        return _UnitComponents(
            unit_ids=unit_ids,
            is_treated_unit=is_treated_unit,
            regions=regions,
            segments=segments,
            unit_log_size=unit_log_size,
            unit_growth=unit_growth,
            unit_quality=unit_quality,
            unit_effect_multiplier=hetero,
        )

    def _post_effect_rate(self, *, unit_components: _UnitComponents) -> np.ndarray:
        c = self.config
        n_total = self._n_total_periods()
        n_units = len(unit_components.unit_ids)
        effect_rate = np.zeros((n_total, n_units), dtype=float)

        post_steps = np.arange(c.n_post_periods, dtype=float)
        ramp = 1.0 - np.exp(-(post_steps + 1.0) / 2.5)
        post_rate = (c.treatment_effect_rate + c.treatment_effect_slope * post_steps) * ramp
        treated_units = unit_components.is_treated_unit.astype(bool)
        effect_rate[c.n_pre_periods :, treated_units] = (
            post_rate[:, None] * unit_components.unit_effect_multiplier[treated_units][None, :]
        )
        if np.any(1.0 + effect_rate <= 0.0):
            raise ValueError(
                "treatment_effect_rate/slope imply non-positive post multipliers; "
                "ensure treatment_effect_rate + slope*k > -1 for all post periods."
            )
        return effect_rate

    def _build_components(self, *, rng: np.random.Generator) -> _PanelComponents:
        c = self.config
        n_total = self._n_total_periods()
        t_rel = np.arange(n_total, dtype=float)
        centered_t = t_rel - t_rel.mean()
        calendar_times, treatment_start = self._calendar_axis()
        units = self._sample_unit_components(rng=rng)
        n_units = len(units.unit_ids)

        season = _monthly_seasonality_signal(t_rel)
        macro_log = _draw_ar1_series(
            rng=rng,
            n_periods=n_total,
            rho=c.rho_common,
            innovation_std=c.common_factor_std_log,
        )
        macro_index = np.exp(macro_log)
        seasonality_index = 1.0 + c.seasonality_strength * season

        exposure = np.empty((n_total, n_units), dtype=float)
        avg_order_value = np.empty((n_total, n_units), dtype=float)
        market_competition = np.empty((n_total, n_units), dtype=float)
        mu_cf = np.empty((n_total, n_units), dtype=float)

        segment_aov_shift = {"core": 0.00, "growth": -4.0, "enterprise": 18.0}
        region_competition_shift = {
            "north": 0.10,
            "south": -0.05,
            "east": 0.03,
            "west": 0.08,
            "central": -0.02,
        }

        for j in range(n_units):
            exposure_noise = _draw_ar1_series(
                rng=rng,
                n_periods=n_total,
                rho=c.rho_unit,
                innovation_std=c.unit_noise_std_log,
            )
            log_exposure = (
                units.unit_log_size[j]
                + units.unit_growth[j] * centered_t
                + 0.35 * macro_log
                + 0.08 * season
                + exposure_noise
            )
            exposure[:, j] = np.exp(np.clip(log_exposure, np.log(80.0), np.log(120_000.0)))

            aov_base = 44.0 + segment_aov_shift[units.segments[j]] + rng.normal(0.0, 5.0)
            aov_noise = _draw_ar1_series(
                rng=rng,
                n_periods=n_total,
                rho=c.rho_unit,
                innovation_std=0.025,
            )
            avg_order_value[:, j] = np.clip(
                aov_base * np.exp(0.03 * season + 0.04 * macro_log + aov_noise),
                8.0,
                180.0,
            )

            comp_noise = _draw_ar1_series(
                rng=rng,
                n_periods=n_total,
                rho=c.rho_unit,
                innovation_std=0.10,
            )
            comp_score = (
                -0.25
                + region_competition_shift[units.regions[j]]
                - 0.15 * units.unit_quality[j]
                + 0.05 * season
                - 0.20 * macro_log
                + comp_noise
            )
            market_competition[:, j] = _sigmoid(comp_score)

            outcome_noise = _draw_ar1_series(
                rng=rng,
                n_periods=n_total,
                rho=c.rho_unit,
                innovation_std=c.outcome_noise_std_log,
            )
            conversion_log_rate = (
                np.log(0.035)
                + units.unit_quality[j]
                + 0.0015 * centered_t
                + 0.22 * macro_log
                + c.seasonality_strength * season
                - 0.55 * market_competition[:, j]
                + outcome_noise
            )
            order_mean = exposure[:, j] * np.exp(np.clip(conversion_log_rate, -8.0, -1.2))
            if c.outcome_distribution == "poisson":
                mu_cf[:, j] = np.clip(order_mean, 1e-6, None)
            else:
                mu_cf[:, j] = np.clip(order_mean * avg_order_value[:, j], 1e-6, None)

        tau_rate_true = self._post_effect_rate(unit_components=units)
        mu_treated = mu_cf * (1.0 + tau_rate_true)
        tau_mean_true = mu_treated - mu_cf

        return _PanelComponents(
            calendar_times=calendar_times,
            treatment_start=treatment_start,
            unit_components=units,
            exposure=exposure,
            avg_order_value=avg_order_value,
            market_competition=market_competition,
            macro_index=macro_index,
            seasonality_index=seasonality_index,
            mu_cf=mu_cf,
            mu_treated=mu_treated,
            tau_mean_true=tau_mean_true,
            tau_rate_true=tau_rate_true,
        )

    def _sample_potential_outcomes(
        self,
        *,
        rng: np.random.Generator,
        components: _PanelComponents,
    ) -> tuple[np.ndarray, np.ndarray]:
        c = self.config
        mu_cf = components.mu_cf
        mu_treated = components.mu_treated
        treated_mask = components.tau_rate_true != 0.0

        if c.outcome_distribution == "gaussian":
            sigma = np.maximum(c.gaussian_noise_std, 0.04 * np.sqrt(np.clip(mu_cf, 1.0, None)))
            y_cf = rng.normal(mu_cf, sigma)
            y_treated = y_cf + components.tau_mean_true
            return np.clip(y_cf, 0.0, None), np.clip(y_treated, 0.0, None)

        if c.outcome_distribution == "gamma":
            y_cf = rng.gamma(shape=c.gamma_shape, scale=np.clip(mu_cf, 1e-6, None) / c.gamma_shape)
            y_treated = y_cf.copy()
            ratio = np.divide(
                mu_treated,
                mu_cf,
                out=np.ones_like(mu_treated, dtype=float),
                where=mu_cf > 0.0,
            )
            y_treated[treated_mask] = y_cf[treated_mask] * ratio[treated_mask]
            return y_cf, y_treated

        y_cf = rng.poisson(np.clip(mu_cf, 1e-6, None)).astype(float)
        y_treated = y_cf.copy()
        if bool(np.any(treated_mask)):
            y0, y1 = _sample_coupled_poisson_pair(
                rng=rng,
                mu_cf=np.clip(mu_cf[treated_mask], 1e-6, None),
                mu_treated=np.clip(mu_treated[treated_mask], 1e-6, None),
            )
            y_cf[treated_mask] = y0
            y_treated[treated_mask] = y1
        return y_cf, y_treated

    def _assemble_rows(
        self,
        *,
        components: _PanelComponents,
        y_cf: np.ndarray,
        y_treated: np.ndarray,
    ) -> pd.DataFrame:
        c = self.config
        rows = []
        units = components.unit_components
        for t_idx, period in enumerate(components.calendar_times):
            is_post = period >= components.treatment_start
            for unit_idx, unit_id in enumerate(units.unit_ids):
                is_treated_unit = bool(units.is_treated_unit[unit_idx])
                treated_time = int(is_treated_unit and is_post)
                observed_y = y_treated[t_idx, unit_idx] if treated_time else y_cf[t_idx, unit_idx]
                realized_tau = y_treated[t_idx, unit_idx] - y_cf[t_idx, unit_idx] if treated_time else 0.0
                rows.append(
                    {
                        "unit_id": unit_id,
                        "calendar_time": period,
                        "treated_time": treated_time,
                        "y": float(observed_y),
                        "y_cf": float(y_cf[t_idx, unit_idx]),
                        "tau_realized_true": float(realized_tau),
                        "mu_cf": float(components.mu_cf[t_idx, unit_idx]),
                        "mu_treated": float(
                            components.mu_treated[t_idx, unit_idx]
                            if treated_time
                            else components.mu_cf[t_idx, unit_idx]
                        ),
                        "tau_mean_true": float(
                            components.tau_mean_true[t_idx, unit_idx] if treated_time else 0.0
                        ),
                        "tau_rate_true": float(
                            components.tau_rate_true[t_idx, unit_idx] if treated_time else 0.0
                        ),
                        "is_treated_unit": int(is_treated_unit),
                        "observed": 1,
                        "exposure": float(components.exposure[t_idx, unit_idx]),
                        "avg_order_value": float(components.avg_order_value[t_idx, unit_idx]),
                        "market_competition": float(components.market_competition[t_idx, unit_idx]),
                        "macro_index": float(components.macro_index[t_idx]),
                        "seasonality_index": float(components.seasonality_index[t_idx]),
                        "region": units.regions[unit_idx],
                        "segment": units.segments[unit_idx],
                    }
                )
        return pd.DataFrame(rows)

    @classmethod
    def _build_panel_contract(cls, df: pd.DataFrame) -> PanelDataDID:
        covariates = tuple(col for col in cls.covariate_cols if col in df.columns)
        cluster_col = cls.cluster_col if cls.cluster_col in df.columns else None
        panel = PanelDataDID(
            df=df,
            y="y",
            unit_col="unit_id",
            time_col="calendar_time",
            treated_time="treated_time",
            covariates=covariates,
            cluster_col=cluster_col,
        )

        # DGP outputs are benchmark fixtures, so keep generated diagnostics and
        # oracle columns visible while still validating the DID contract above.
        full_df = df.copy()
        object.__setattr__(panel, "df", full_df)
        object.__setattr__(panel, "_df_validated", full_df.copy(deep=True))
        return panel

    def generate(
        self,
        *,
        return_panel_data: Optional[bool] = None,
    ) -> Union[pd.DataFrame, PanelDataDID]:
        c = self.config
        return_panel_data_flag = c.return_panel_data if return_panel_data is None else bool(return_panel_data)
        rng = np.random.default_rng(c.random_state)
        components = self._build_components(rng=rng)
        y_cf, y_treated = self._sample_potential_outcomes(rng=rng, components=components)
        df = self._assemble_rows(components=components, y_cf=y_cf, y_treated=y_treated)
        df = df.sort_values(["unit_id", "calendar_time"]).reset_index(drop=True)

        if not return_panel_data_flag:
            return df
        return self._build_panel_contract(df)
