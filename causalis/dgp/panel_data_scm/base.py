from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Literal, Optional, Union

import numpy as np
import pandas as pd

from causalis.data_contracts.panel_data_scm import PanelDataSCM


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


def _drop_block_missing_cells(
    df: pd.DataFrame,
    *,
    rng: np.random.Generator,
    missing_block_frac: float,
    block_min_len: int,
    block_max_len: Optional[int],
    protected_index: Optional[set[int]] = None,
) -> pd.DataFrame:
    if missing_block_frac <= 0.0:
        return df

    n_target = int(round(missing_block_frac * len(df)))
    if n_target <= 0:
        return df

    protected_set = set() if protected_index is None else set(protected_index)

    by_unit = {
        unit: grp.sort_values("calendar_time").index.to_numpy(dtype=int)
        for unit, grp in df.groupby("unit_id", sort=False)
    }
    units = list(by_unit.keys())
    if not units:
        return df

    missing_set: set[int] = set()
    n_tries = max(100, 25 * n_target)
    for _ in range(n_tries):
        if len(missing_set) >= n_target:
            break

        unit = units[int(rng.integers(0, len(units)))]
        unit_idx = by_unit[unit]
        n_unit = int(unit_idx.size)
        if n_unit <= 1:
            continue

        min_len = int(min(max(1, block_min_len), n_unit))
        max_len_candidate = n_unit if block_max_len is None else int(block_max_len)
        max_len = int(min(max_len_candidate, n_unit))
        if max_len < min_len:
            continue

        block_len = int(rng.integers(min_len, max_len + 1))
        start = int(rng.integers(0, n_unit - block_len + 1))
        candidate = unit_idx[start : start + block_len]
        for idx in candidate:
            idx_int = int(idx)
            if idx_int in protected_set:
                continue
            missing_set.add(idx_int)
            if len(missing_set) >= n_target:
                break

    if not missing_set:
        return df
    out = df.copy()
    out.loc[list(missing_set), "y"] = np.nan
    return out


@dataclass(frozen=True)
class PanelSCMGeneratorConfig:
    # Shared panel shape / IDs
    n_donors: int = 5
    n_pre_periods: int = 20
    n_post_periods: int = 10
    time_start: int = 1
    time_freq: str = "M"
    calendar_start: str = "2000-01"
    treated_unit: Hashable = "treated"
    donor_prefix: str = "donor_"
    random_state: Optional[int] = 42
    return_panel_data: bool = True

    # Shared realism knobs
    dirichlet_alpha: float = 1.0
    rho_common: float = 0.0
    rho_donor: float = 0.0
    n_latent_factors: int = 0
    latent_loading_std: float = 0.35
    rho_latent: float = 0.0
    rho_prefit_mismatch: float = 0.0

    # Shared missingness knobs
    missing_outcome_frac: float = 0.0
    missing_cell_frac: float = 0.0
    missing_block_frac: float = 0.0
    missing_block_min_len: int = 2
    missing_block_max_len: Optional[int] = None
    protect_treated_pre: bool = False
    protect_treated_post: bool = False

    # Mode selector
    outcome_distribution: Literal["gaussian", "gamma", "poisson"] = "gaussian"

    # Gaussian-mode parameters
    treatment_effect: float = 2.0
    treatment_effect_slope: float = 0.0
    donor_noise_std: float = 0.20
    treated_noise_std: float = 0.10
    common_factor_std: float = 0.15
    latent_factor_std: float = 0.20
    prefit_mismatch_std: float = 0.0
    treatment_effect_mode: Literal["additive", "multiplicative"] = "additive"

    # Gamma/Poisson effect parameter (long-run scale; first post period ramps in).
    treatment_effect_rate: float = 0.12
    gamma_shape: float = 6.0
    donor_noise_std_log: float = 0.15
    common_factor_std_log: float = 0.10
    latent_factor_std_log: float = 0.10
    prefit_mismatch_std_log: float = 0.08


class PanelSCMGenerator:
    """Low-level panel SCM generator supporting Gaussian, Gamma, and Poisson outcomes."""

    def __init__(self, config: PanelSCMGeneratorConfig):
        self.config = config
        self._validate_config()

    def _validate_config(self) -> None:
        c = self.config
        if c.n_donors < 1:
            raise ValueError("n_donors must be >= 1.")
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
        if c.dirichlet_alpha <= 0.0:
            raise ValueError("dirichlet_alpha must be > 0.")
        if c.n_latent_factors < 0:
            raise ValueError("n_latent_factors must be >= 0.")
        if c.latent_loading_std < 0.0:
            raise ValueError("latent_loading_std must be >= 0.")
        for rho_name, rho in (
            ("rho_common", c.rho_common),
            ("rho_donor", c.rho_donor),
            ("rho_latent", c.rho_latent),
            ("rho_prefit_mismatch", c.rho_prefit_mismatch),
        ):
            if not (-1.0 < rho < 1.0):
                raise ValueError(f"{rho_name} must be in (-1, 1).")
        if not (0.0 <= c.missing_outcome_frac < 1.0):
            raise ValueError("missing_outcome_frac must be in [0, 1).")
        if not (0.0 <= c.missing_cell_frac < 1.0):
            raise ValueError("missing_cell_frac must be in [0, 1).")
        if not (0.0 <= c.missing_block_frac < 1.0):
            raise ValueError("missing_block_frac must be in [0, 1).")
        if c.missing_block_min_len < 1:
            raise ValueError("missing_block_min_len must be >= 1.")
        if c.missing_block_max_len is not None and c.missing_block_max_len < 1:
            raise ValueError("missing_block_max_len must be >= 1 when provided.")
        if (
            c.missing_block_max_len is not None
            and c.missing_block_max_len < c.missing_block_min_len
        ):
            raise ValueError("missing_block_max_len must be >= missing_block_min_len.")
        if not isinstance(c.protect_treated_pre, bool):
            raise ValueError("protect_treated_pre must be a boolean.")
        if not isinstance(c.protect_treated_post, bool):
            raise ValueError("protect_treated_post must be a boolean.")
        if c.outcome_distribution not in {"gaussian", "gamma", "poisson"}:
            raise ValueError("outcome_distribution must be 'gaussian', 'gamma', or 'poisson'.")

        if c.outcome_distribution == "gaussian":
            if c.donor_noise_std < 0.0:
                raise ValueError("donor_noise_std must be >= 0.")
            if c.treated_noise_std < 0.0:
                raise ValueError("treated_noise_std must be >= 0.")
            if c.common_factor_std < 0.0:
                raise ValueError("common_factor_std must be >= 0.")
            if c.latent_factor_std < 0.0:
                raise ValueError("latent_factor_std must be >= 0.")
            if c.prefit_mismatch_std < 0.0:
                raise ValueError("prefit_mismatch_std must be >= 0.")
            if c.treatment_effect_mode not in {"additive", "multiplicative"}:
                raise ValueError("treatment_effect_mode must be 'additive' or 'multiplicative'.")
        elif c.outcome_distribution == "gamma":
            if c.gamma_shape <= 0.0:
                raise ValueError("gamma_shape must be > 0.")
            if c.donor_noise_std_log < 0.0:
                raise ValueError("donor_noise_std_log must be >= 0.")
            if c.common_factor_std_log < 0.0:
                raise ValueError("common_factor_std_log must be >= 0.")
            if c.latent_factor_std_log < 0.0:
                raise ValueError("latent_factor_std_log must be >= 0.")
            if c.prefit_mismatch_std_log < 0.0:
                raise ValueError("prefit_mismatch_std_log must be >= 0.")
        else:
            if c.donor_noise_std_log < 0.0:
                raise ValueError("donor_noise_std_log must be >= 0.")
            if c.common_factor_std_log < 0.0:
                raise ValueError("common_factor_std_log must be >= 0.")
            if c.latent_factor_std_log < 0.0:
                raise ValueError("latent_factor_std_log must be >= 0.")
            if c.prefit_mismatch_std_log < 0.0:
                raise ValueError("prefit_mismatch_std_log must be >= 0.")

    def _calendar_axis(self, *, n_total: int) -> tuple[list[pd.Period], pd.Period]:
        c = self.config
        base = pd.Period(c.calendar_start, freq=c.time_freq)
        start = base + (int(c.time_start) - 1)
        calendar_times = list(pd.period_range(start=start, periods=int(n_total), freq=c.time_freq))
        treatment_start = calendar_times[int(c.n_pre_periods)]
        return calendar_times, treatment_start

    def _sample_latent_factors(
        self,
        *,
        rng: np.random.Generator,
        n_total: int,
        innovation_std: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        c = self.config
        if c.n_latent_factors <= 0:
            return (
                np.empty((n_total, 0), dtype=float),
                np.empty((c.n_donors, 0), dtype=float),
            )
        latent_factors = np.column_stack(
            [
                _draw_ar1_series(
                    rng=rng,
                    n_periods=n_total,
                    rho=c.rho_latent,
                    innovation_std=innovation_std,
                )
                for _ in range(c.n_latent_factors)
            ]
        )
        loadings = rng.normal(0.0, c.latent_loading_std, size=(c.n_donors, c.n_latent_factors))
        return latent_factors, loadings

    def _generate_gaussian_panel(
        self,
        *,
        rng: np.random.Generator,
    ) -> tuple[pd.DataFrame, list[Hashable], tuple[str, ...]]:
        c = self.config
        n_total = int(c.n_pre_periods + c.n_post_periods)
        t_rel = np.arange(n_total, dtype=float)
        calendar_times, _ = self._calendar_axis(n_total=n_total)

        common_shock = _draw_ar1_series(
            rng=rng,
            n_periods=n_total,
            rho=c.rho_common,
            innovation_std=c.common_factor_std,
        )
        season = np.sin(2.0 * np.pi * t_rel / max(4, n_total))

        donor_names = [f"{c.donor_prefix}{j + 1}" for j in range(c.n_donors)]
        donor_matrix = np.empty((n_total, c.n_donors), dtype=float)

        latent_factors, donor_factor_loadings = self._sample_latent_factors(
            rng=rng,
            n_total=n_total,
            innovation_std=c.latent_factor_std,
        )

        for j in range(c.n_donors):
            intercept = rng.normal(10.0, 1.5)
            slope = rng.normal(0.08, 0.03)
            seasonal_loading = rng.normal(0.35, 0.10)
            latent_term = (
                latent_factors @ donor_factor_loadings[j]
                if c.n_latent_factors > 0
                else np.zeros(n_total, dtype=float)
            )
            donor_noise = _draw_ar1_series(
                rng=rng,
                n_periods=n_total,
                rho=c.rho_donor,
                innovation_std=c.donor_noise_std,
            )
            donor_matrix[:, j] = (
                intercept
                + slope * t_rel
                + seasonal_loading * season
                + common_shock
                + latent_term
                + donor_noise
            )

        true_weights = rng.dirichlet(np.full(c.n_donors, fill_value=c.dirichlet_alpha, dtype=float))
        treated_noise = _draw_ar1_series(
            rng=rng,
            n_periods=n_total,
            rho=c.rho_donor,
            innovation_std=c.treated_noise_std,
        )
        prefit_mismatch = _draw_ar1_series(
            rng=rng,
            n_periods=n_total,
            rho=c.rho_prefit_mismatch,
            innovation_std=c.prefit_mismatch_std,
        )
        treated_counterfactual = donor_matrix @ true_weights + treated_noise + prefit_mismatch

        effect = np.zeros(n_total, dtype=float)
        post_idx = np.arange(c.n_pre_periods, n_total)
        post_effect_path = c.treatment_effect + c.treatment_effect_slope * np.arange(c.n_post_periods, dtype=float)
        if c.treatment_effect_mode == "additive":
            effect[post_idx] = post_effect_path
            treated_observed = treated_counterfactual + effect
        else:
            post_multipliers = 1.0 + post_effect_path
            if np.any(post_multipliers <= 0.0):
                raise ValueError(
                    "multiplicative treatment path implies non-positive multipliers; "
                    "ensure treatment_effect + slope * k > -1."
                )
            treated_observed = treated_counterfactual.copy()
            treated_observed[post_idx] = treated_counterfactual[post_idx] * post_multipliers
            effect[post_idx] = treated_observed[post_idx] - treated_counterfactual[post_idx]

        rows = []
        for i, t_cal in enumerate(calendar_times):
            rows.append(
                {
                    "unit_id": c.treated_unit,
                    "calendar_time": t_cal,
                    "y": float(treated_observed[i]),
                    "y_cf": float(treated_counterfactual[i]),
                    "tau_realized_true": float(effect[i]),
                    "is_treated_unit": 1,
                }
            )
            for j, unit in enumerate(donor_names):
                rows.append(
                    {
                        "unit_id": unit,
                        "calendar_time": t_cal,
                        "y": float(donor_matrix[i, j]),
                        "y_cf": float(donor_matrix[i, j]),
                        "tau_realized_true": 0.0,
                        "is_treated_unit": 0,
                    }
                )

        return pd.DataFrame(rows), donor_names, ()

    def _generate_gamma_panel(
        self,
        *,
        rng: np.random.Generator,
    ) -> tuple[pd.DataFrame, list[Hashable], tuple[str, ...]]:
        c = self.config
        n_total = int(c.n_pre_periods + c.n_post_periods)
        t_rel = np.arange(n_total, dtype=float)
        calendar_times, _ = self._calendar_axis(n_total=n_total)
        post_idx = np.arange(c.n_pre_periods, n_total, dtype=int)

        season = np.sin(2.0 * np.pi * t_rel / 12.0) + 0.5 * np.cos(2.0 * np.pi * t_rel / 6.0)
        macro_log = _draw_ar1_series(
            rng=rng,
            n_periods=n_total,
            rho=c.rho_common,
            innovation_std=c.common_factor_std_log,
        )

        latent_factors, donor_factor_loadings = self._sample_latent_factors(
            rng=rng,
            n_total=n_total,
            innovation_std=c.latent_factor_std_log,
        )

        donor_names = [f"{c.donor_prefix}{j + 1}" for j in range(c.n_donors)]
        donor_exposure = np.empty((n_total, c.n_donors), dtype=float)
        donor_eta = np.empty((n_total, c.n_donors), dtype=float)
        donor_mu = np.empty((n_total, c.n_donors), dtype=float)

        centered_t = t_rel - t_rel.mean()
        exposure_time_log = _draw_ar1_series(
            rng=rng,
            n_periods=n_total,
            rho=c.rho_common,
            innovation_std=0.08,
        )
        for j in range(c.n_donors):
            exposure_level = float(rng.normal(np.log(800.0), 0.30))
            exposure_growth = float(rng.normal(0.004, 0.002))
            exposure_noise = _draw_ar1_series(
                rng=rng,
                n_periods=n_total,
                rho=c.rho_donor,
                innovation_std=0.10,
            )
            log_exposure = exposure_level + exposure_growth * centered_t + exposure_time_log + exposure_noise
            donor_exposure[:, j] = np.exp(np.clip(log_exposure, np.log(20.0), np.log(50_000.0)))

            eta_intercept = float(rng.normal(np.log(0.025), 0.30))
            eta_growth = float(rng.normal(0.005, 0.003))
            season_loading = float(rng.normal(0.14, 0.05))
            donor_noise = _draw_ar1_series(
                rng=rng,
                n_periods=n_total,
                rho=c.rho_donor,
                innovation_std=c.donor_noise_std_log,
            )
            latent_term = (
                latent_factors @ donor_factor_loadings[j]
                if c.n_latent_factors > 0
                else np.zeros(n_total, dtype=float)
            )
            eta = (
                eta_intercept
                + eta_growth * centered_t
                + season_loading * season
                + macro_log
                + latent_term
                + donor_noise
            )
            donor_eta[:, j] = eta
            log_mu = np.log(donor_exposure[:, j]) + eta
            donor_mu[:, j] = np.exp(np.clip(log_mu, -6.0, 10.0))

        true_weights = rng.dirichlet(np.full(c.n_donors, fill_value=c.dirichlet_alpha, dtype=float))
        prefit_mismatch_eta = _draw_ar1_series(
            rng=rng,
            n_periods=n_total,
            rho=c.rho_prefit_mismatch,
            innovation_std=c.prefit_mismatch_std_log,
        )
        prefit_mismatch_exposure = _draw_ar1_series(
            rng=rng,
            n_periods=n_total,
            rho=c.rho_prefit_mismatch,
            innovation_std=0.5 * c.prefit_mismatch_std_log,
        )
        treated_exposure_cf = (donor_exposure @ true_weights) * np.exp(prefit_mismatch_exposure)
        treated_exposure_cf = np.clip(treated_exposure_cf, 1.0, None)
        treated_eta_cf = (donor_eta @ true_weights) + prefit_mismatch_eta
        treated_log_mu_cf = np.log(treated_exposure_cf) + treated_eta_cf
        treated_mu_cf = np.exp(np.clip(treated_log_mu_cf, -6.0, 10.0))
        treated_mu_cf = np.clip(treated_mu_cf, 1e-6, None)

        donor_y = rng.gamma(shape=c.gamma_shape, scale=np.clip(donor_mu, 1e-6, None) / c.gamma_shape)
        treated_y_cf = rng.gamma(
            shape=c.gamma_shape,
            scale=np.clip(treated_mu_cf, 1e-6, None) / c.gamma_shape,
        )

        effect_rate = np.zeros(n_total, dtype=float)
        post_steps = np.arange(c.n_post_periods, dtype=float)
        ramp = 1.0 - np.exp(-(post_steps + 1.0) / 2.5)
        post_rate = (c.treatment_effect_rate + c.treatment_effect_slope * post_steps) * ramp
        if np.any(1.0 + post_rate <= 0.0):
            raise ValueError(
                "treatment_effect_rate/slope imply non-positive post multipliers; "
                "ensure treatment_effect_rate + slope*k > -1 for all post periods."
            )
        effect_rate[post_idx] = post_rate

        treated_mu = treated_mu_cf.copy()
        treated_mu[post_idx] = treated_mu_cf[post_idx] * (1.0 + effect_rate[post_idx])
        tau_mean_true = treated_mu - treated_mu_cf

        treated_y = treated_y_cf.copy()
        treated_y[post_idx] = treated_y_cf[post_idx] * (1.0 + effect_rate[post_idx])
        tau_realized_true = treated_y - treated_y_cf

        macro_index = np.exp(macro_log)
        seasonality_index = 1.0 + 0.15 * season

        rows = []
        for i, t_cal in enumerate(calendar_times):
            rows.append(
                {
                    "unit_id": c.treated_unit,
                    "calendar_time": t_cal,
                    "y": float(treated_y[i]),
                    "y_cf": float(treated_y_cf[i]),
                    "tau_realized_true": float(tau_realized_true[i]),
                    "mu_cf": float(treated_mu_cf[i]),
                    "mu_treated": float(treated_mu[i]),
                    "tau_mean_true": float(tau_mean_true[i]),
                    "tau_rate_true": float(effect_rate[i]),
                    "is_treated_unit": 1,
                    "exposure": float(treated_exposure_cf[i]),
                    "macro_index": float(macro_index[i]),
                    "seasonality_index": float(seasonality_index[i]),
                }
            )
            for j, unit in enumerate(donor_names):
                rows.append(
                    {
                        "unit_id": unit,
                        "calendar_time": t_cal,
                        "y": float(donor_y[i, j]),
                        "y_cf": float(donor_y[i, j]),
                        "tau_realized_true": 0.0,
                        "mu_cf": float(donor_mu[i, j]),
                        "mu_treated": float(donor_mu[i, j]),
                        "tau_mean_true": 0.0,
                        "tau_rate_true": 0.0,
                        "is_treated_unit": 0,
                        "exposure": float(donor_exposure[i, j]),
                        "macro_index": float(macro_index[i]),
                        "seasonality_index": float(seasonality_index[i]),
                    }
                )

        return (
            pd.DataFrame(rows),
            donor_names,
            ("exposure", "macro_index", "seasonality_index"),
        )

    def _generate_poisson_panel(
        self,
        *,
        rng: np.random.Generator,
    ) -> tuple[pd.DataFrame, list[Hashable], tuple[str, ...]]:
        c = self.config
        n_total = int(c.n_pre_periods + c.n_post_periods)
        t_rel = np.arange(n_total, dtype=float)
        calendar_times, _ = self._calendar_axis(n_total=n_total)
        post_idx = np.arange(c.n_pre_periods, n_total, dtype=int)

        season = np.sin(2.0 * np.pi * t_rel / 12.0) + 0.5 * np.cos(2.0 * np.pi * t_rel / 6.0)
        macro_log = _draw_ar1_series(
            rng=rng,
            n_periods=n_total,
            rho=c.rho_common,
            innovation_std=c.common_factor_std_log,
        )

        latent_factors, donor_factor_loadings = self._sample_latent_factors(
            rng=rng,
            n_total=n_total,
            innovation_std=c.latent_factor_std_log,
        )

        donor_names = [f"{c.donor_prefix}{j + 1}" for j in range(c.n_donors)]
        donor_exposure = np.empty((n_total, c.n_donors), dtype=float)
        donor_eta = np.empty((n_total, c.n_donors), dtype=float)
        donor_mu = np.empty((n_total, c.n_donors), dtype=float)

        centered_t = t_rel - t_rel.mean()
        exposure_time_log = _draw_ar1_series(
            rng=rng,
            n_periods=n_total,
            rho=c.rho_common,
            innovation_std=0.08,
        )
        for j in range(c.n_donors):
            exposure_level = float(rng.normal(np.log(700.0), 0.30))
            exposure_growth = float(rng.normal(0.004, 0.002))
            exposure_noise = _draw_ar1_series(
                rng=rng,
                n_periods=n_total,
                rho=c.rho_donor,
                innovation_std=0.10,
            )
            log_exposure = exposure_level + exposure_growth * centered_t + exposure_time_log + exposure_noise
            donor_exposure[:, j] = np.exp(np.clip(log_exposure, np.log(15.0), np.log(45_000.0)))

            eta_intercept = float(rng.normal(np.log(0.018), 0.25))
            eta_growth = float(rng.normal(0.005, 0.003))
            season_loading = float(rng.normal(0.16, 0.06))
            donor_noise = _draw_ar1_series(
                rng=rng,
                n_periods=n_total,
                rho=c.rho_donor,
                innovation_std=c.donor_noise_std_log,
            )
            latent_term = (
                latent_factors @ donor_factor_loadings[j]
                if c.n_latent_factors > 0
                else np.zeros(n_total, dtype=float)
            )
            eta = (
                eta_intercept
                + eta_growth * centered_t
                + season_loading * season
                + macro_log
                + latent_term
                + donor_noise
            )
            donor_eta[:, j] = eta
            log_mu = np.log(donor_exposure[:, j]) + eta
            donor_mu[:, j] = np.exp(np.clip(log_mu, -6.0, 10.0))

        true_weights = rng.dirichlet(np.full(c.n_donors, fill_value=c.dirichlet_alpha, dtype=float))
        prefit_mismatch_eta = _draw_ar1_series(
            rng=rng,
            n_periods=n_total,
            rho=c.rho_prefit_mismatch,
            innovation_std=c.prefit_mismatch_std_log,
        )
        prefit_mismatch_exposure = _draw_ar1_series(
            rng=rng,
            n_periods=n_total,
            rho=c.rho_prefit_mismatch,
            innovation_std=0.5 * c.prefit_mismatch_std_log,
        )
        treated_exposure_cf = (donor_exposure @ true_weights) * np.exp(prefit_mismatch_exposure)
        treated_exposure_cf = np.clip(treated_exposure_cf, 1.0, None)
        treated_eta_cf = (donor_eta @ true_weights) + prefit_mismatch_eta
        treated_log_mu_cf = np.log(treated_exposure_cf) + treated_eta_cf
        treated_mu_cf = np.exp(np.clip(treated_log_mu_cf, -6.0, 10.0))
        treated_mu_cf = np.clip(treated_mu_cf, 1e-6, None)

        effect_rate = np.zeros(n_total, dtype=float)
        post_steps = np.arange(c.n_post_periods, dtype=float)
        ramp = 1.0 - np.exp(-(post_steps + 1.0) / 2.5)
        post_rate = (c.treatment_effect_rate + c.treatment_effect_slope * post_steps) * ramp
        if np.any(1.0 + post_rate <= 0.0):
            raise ValueError(
                "treatment_effect_rate/slope imply non-positive post multipliers; "
                "ensure treatment_effect_rate + slope*k > -1 for all post periods."
            )
        effect_rate[post_idx] = post_rate

        treated_mu = treated_mu_cf.copy()
        treated_mu[post_idx] = treated_mu_cf[post_idx] * (1.0 + effect_rate[post_idx])
        tau_mean_true = treated_mu - treated_mu_cf

        donor_y = rng.poisson(np.clip(donor_mu, 1e-6, None)).astype(float)
        treated_y_cf, treated_y = _sample_coupled_poisson_pair(
            rng=rng,
            mu_cf=np.clip(treated_mu_cf, 1e-6, None),
            mu_treated=np.clip(treated_mu, 1e-6, None),
        )
        tau_realized_true = treated_y - treated_y_cf

        macro_index = np.exp(macro_log)
        seasonality_index = 1.0 + 0.15 * season

        rows = []
        for i, t_cal in enumerate(calendar_times):
            rows.append(
                {
                    "unit_id": c.treated_unit,
                    "calendar_time": t_cal,
                    "y": float(treated_y[i]),
                    "y_cf": float(treated_y_cf[i]),
                    "tau_realized_true": float(tau_realized_true[i]),
                    "mu_cf": float(treated_mu_cf[i]),
                    "mu_treated": float(treated_mu[i]),
                    "tau_mean_true": float(tau_mean_true[i]),
                    "tau_rate_true": float(effect_rate[i]),
                    "is_treated_unit": 1,
                    "exposure": float(treated_exposure_cf[i]),
                    "macro_index": float(macro_index[i]),
                    "seasonality_index": float(seasonality_index[i]),
                }
            )
            for j, unit in enumerate(donor_names):
                rows.append(
                    {
                        "unit_id": unit,
                        "calendar_time": t_cal,
                        "y": float(donor_y[i, j]),
                        "y_cf": float(donor_y[i, j]),
                        "tau_realized_true": 0.0,
                        "mu_cf": float(donor_mu[i, j]),
                        "mu_treated": float(donor_mu[i, j]),
                        "tau_mean_true": 0.0,
                        "tau_rate_true": 0.0,
                        "is_treated_unit": 0,
                        "exposure": float(donor_exposure[i, j]),
                        "macro_index": float(macro_index[i]),
                        "seasonality_index": float(seasonality_index[i]),
                    }
                )

        return (
            pd.DataFrame(rows),
            donor_names,
            ("exposure", "macro_index", "seasonality_index"),
        )

    def _apply_missingness(self, *, df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
        c = self.config
        out = df.copy()
        _, treatment_start = self._calendar_axis(n_total=int(c.n_pre_periods + c.n_post_periods))

        treated_mask = out["unit_id"] == c.treated_unit
        protected_treated_idx: set[int] = set()
        if c.protect_treated_pre:
            protected_treated_idx.update(
                out.index[(treated_mask) & (out["calendar_time"] < treatment_start)].to_numpy(dtype=int).tolist()
            )
        if c.protect_treated_post:
            protected_treated_idx.update(
                out.index[(treated_mask) & (out["calendar_time"] >= treatment_start)].to_numpy(dtype=int).tolist()
            )

        protected_structured_idx: set[int] = set(
            out.groupby("unit_id", sort=False, as_index=False)
            .head(1)
            .index
            .to_numpy(dtype=int)
            .tolist()
        )
        protected_structured_idx.update(protected_treated_idx)

        if c.missing_outcome_frac > 0.0:
            n_missing = int(round(c.missing_outcome_frac * len(out)))
            if n_missing > 0:
                eligible = out.index.difference(pd.Index(sorted(protected_treated_idx))).to_numpy(dtype=int)
                if eligible.size > 0:
                    n_missing_eff = int(min(n_missing, eligible.size))
                    idx_missing = rng.choice(eligible, size=n_missing_eff, replace=False)
                    out.loc[idx_missing, "y"] = np.nan

        if c.missing_cell_frac > 0.0:
            n_missing = int(round(c.missing_cell_frac * len(out)))
            if n_missing > 0:
                eligible = out.index.difference(pd.Index(sorted(protected_structured_idx))).to_numpy(dtype=int)
                if eligible.size > 0:
                    n_missing_eff = int(min(n_missing, eligible.size))
                    idx_missing = rng.choice(eligible, size=n_missing_eff, replace=False)
                    out.loc[idx_missing, "y"] = np.nan

        if c.missing_block_frac > 0.0:
            out = _drop_block_missing_cells(
                out,
                rng=rng,
                missing_block_frac=c.missing_block_frac,
                block_min_len=c.missing_block_min_len,
                block_max_len=c.missing_block_max_len,
                protected_index=protected_structured_idx,
            )

        return out

    def generate(
        self,
        *,
        return_panel_data: Optional[bool] = None,
    ) -> Union[pd.DataFrame, PanelDataSCM]:
        c = self.config
        return_panel_data_flag = c.return_panel_data if return_panel_data is None else bool(return_panel_data)
        rng = np.random.default_rng(c.random_state)

        if c.outcome_distribution == "gaussian":
            df, donor_names, covariate_cols = self._generate_gaussian_panel(rng=rng)
        elif c.outcome_distribution == "gamma":
            df, donor_names, covariate_cols = self._generate_gamma_panel(rng=rng)
        else:
            df, donor_names, covariate_cols = self._generate_poisson_panel(rng=rng)

        _, treatment_start = self._calendar_axis(n_total=int(c.n_pre_periods + c.n_post_periods))
        df = self._apply_missingness(df=df, rng=rng)
        df = df.sort_values(["unit_id", "calendar_time"]).reset_index(drop=True)
        df["observed"] = (~df["y"].isna()).astype(int)

        if not return_panel_data_flag:
            return df

        allow_missing = bool(
            c.missing_outcome_frac > 0.0
            or c.missing_cell_frac > 0.0
            or c.missing_block_frac > 0.0
        )
        panel = PanelDataSCM(
            df=df,
            unit_col="unit_id",
            time_col="calendar_time",
            time_freq=c.time_freq,
            y="y",
            treated_unit=c.treated_unit,
            treatment_start=treatment_start,
            donor_units=donor_names,
            covariate_cols=covariate_cols,
            observed_col="observed",
            allow_missing_outcome=allow_missing,
        )
        # Restore generator-facing columns (oracle and helper columns) after validation.
        return panel.model_copy(update={"df": df})
