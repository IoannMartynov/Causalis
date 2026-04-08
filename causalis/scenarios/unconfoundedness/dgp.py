"""Synthetic observational datasets for unconfoundedness experiments."""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Union
from causalis.dgp.causaldata import CausalData
from causalis.dgp.causaldata.functional import obs_linear_effect
from causalis.dgp.causaldata.base import CausalDatasetGenerator
from causalis.dgp.base import _gaussian_copula, _sigmoid

def obs_linear_26_dataset(n: int = 10000, seed: int = 42, include_oracle: bool = True, return_causal_data: bool = True):
    """
    A pre-configured observational linear dataset with 5 standard confounders.
    Based on the scenario in docs/cases/dml_ate.ipynb.

    Parameters
    ----------
    n : int, default=10000
        Number of samples.
    seed : int, default=42
        Random seed.
    include_oracle : bool, default=True
        Whether to include oracle ground-truth columns like 'cate', 'propensity', etc.
    return_causal_data : bool, default=True
        If True, returns a CausalData object. If False, returns a pandas DataFrame.

    Returns
    -------
    pandas.DataFrame or CausalData
        Generated observational sample, optionally wrapped as ``CausalData``.
    """
    confounder_specs = [
        {"name": "tenure_months",     "dist": "normal",   "mu": 24, "sd": 12, "clip_min": 0, "clip_max": 120},
        {"name": "avg_sessions_week", "dist": "normal",   "mu": 5,  "sd": 2,  "clip_min": 0, "clip_max": 40},
        {"name": "spend_last_month",  "dist": "lognormal","mu": np.log(60), "sigma": 0.9, "clip_max": 500},
        {"name": "premium_user",      "dist": "bernoulli","p": 0.25},
        {"name": "urban_resident",    "dist": "bernoulli","p": 0.60},
    ]
    # Coefficients from dml_ate.ipynb
    beta_y = [0.05, 0.40, 0.02, 2.00, 1.00]
    beta_d = [0.015, 0.10, 0.002, 0.75, 0.30]

    gen = CausalDatasetGenerator(
        theta=1.8,
        sigma_y=3.5,
        target_d_rate=0.05,
        confounder_specs=confounder_specs,
        beta_y=beta_y,
        beta_d=beta_d,
        seed=seed,
        include_oracle=include_oracle,
        use_copula=True,
        copula_corr=np.array([
            [1.00, 0.30, 0.20, 0.15, 0.00],
            [0.30, 1.00, 0.40, 0.35, 0.00],
            [0.20, 0.40, 1.00, 0.30, 0.00],
            [0.15, 0.35, 0.30, 1.00, 0.00],
            [0.00, 0.00, 0.00, 0.00, 1.00],
        ])
    )
    df = gen.generate(n)
    
    if not return_causal_data:
        return df

    confounder_names = [s["name"] for s in confounder_specs]
    return CausalData(df=df, treatment='d', outcome='y', confounders=confounder_names)

def generate_obs_hte_26(
    n: int = 10000,
    seed: int = 42,
    include_oracle: bool = True,
    return_causal_data: bool = True
) -> Union[pd.DataFrame, CausalData]:
    """
    Observational dataset with nonlinear outcome model, nonlinear treatment assignment,
    and a heterogeneous (nonlinear) treatment effect tau(X).
    Based on the scenario in notebooks/cases/dml_atte.ipynb.

    Parameters
    ----------
    n : int, default=10000
        Number of samples.
    seed : int, default=42
        Random seed.
    include_oracle : bool, default=True
        Whether to include oracle ground-truth columns like 'cate', 'propensity', etc.
    return_causal_data : bool, default=True
        If True, returns a CausalData object. If False, returns a pandas DataFrame.

    Returns
    -------
    pandas.DataFrame or CausalData
        Generated heterogeneous-treatment-effect sample, optionally wrapped as
        ``CausalData``.
    """
    # `x_sampler` builds correlated/derived features; specs define output names/order.
    confounder_specs = [
        {"name": "tenure_months",     "dist": "lognormal", "mu": np.log(24), "sigma": 0.6, "clip_max": 120},
        {"name": "avg_sessions_week", "dist": "negbin",    "mu": 5, "alpha": 0.5, "clip_max": 40},
        {"name": "spend_last_month",  "dist": "lognormal", "mu": np.log(60), "sigma": 0.9, "clip_max": 500},
        {"name": "premium_user",      "dist": "bernoulli", "p": 0.25},
        {"name": "urban_resident",    "dist": "bernoulli", "p": 0.60},
    ]

    # Reference points used to center log-transformed features around typical usage.
    LOG_SESS_REF = np.log1p(5.0)
    LOG_SPEND_REF = np.log1p(60.0)

    def _split_features(X: np.ndarray) -> dict[str, np.ndarray]:
        """Build reusable transformed feature views for the scenario equations."""
        # Shared feature engineering for g_y / g_d / tau to avoid repeated slicing.
        tenure, sessions, spend, premium, urban = X.T
        log_sessions = np.log1p(sessions)
        log_spend = np.log1p(spend)
        return {
            "tenure": tenure,
            "premium": premium,
            "urban": urban,
            "log_sessions": log_sessions,
            "log_spend": log_spend,
            "sessions_ctr": log_sessions - LOG_SESS_REF,
            "spend_ctr": log_spend - LOG_SPEND_REF,
        }

    # Nonlinear baseline for outcome f_y(X) = X @ beta_y + g_y(X)
    beta_y = np.array([
        0.01,   # tenure_months
        0.00,   # avg_sessions_week (moved to g_y)
        0.00,   # spend_last_month (moved to g_y)
        1.20,   # premium_user
        0.60,   # urban_resident
    ], dtype=float)

    def g_y(X: np.ndarray) -> np.ndarray:
        """Evaluate the nonlinear baseline outcome component."""
        f = _split_features(X)

        return (
            1.5 * np.tanh(f["tenure"] / 24.0)                    # saturating tenure curve
            + 0.5 * f["sessions_ctr"] ** 2                       # convex effect of sessions
            + 0.2 * f["spend_ctr"] * f["sessions_ctr"]           # log-log interaction
            + 0.5 * f["premium"] * f["sessions_ctr"]             # premium × sessions interaction
            + 0.8 * f["urban"] * np.tanh(f["spend_ctr"] / 2.0)   # urban-specific nonlinear spend effect
        )

    # Nonlinear treatment score f_t(X) = X @ beta_t + g_t(X)
    beta_d = np.array([
        0.005,  # tenure_months
        0.00,   # avg_sessions_week (moved to g_d)
        0.00,   # spend_last_month (moved to g_d)
        0.80,   # premium_user
        0.25,   # urban_resident
    ], dtype=float)

    def g_d(X: np.ndarray) -> np.ndarray:
        """Evaluate the nonlinear treatment-assignment component."""
        f = _split_features(X)

        # Smoothly increasing selection with log_spend; interactions make selection non-separable
        soft_spend = np.tanh(f["spend_ctr"])
        return (
            0.8 * soft_spend
            + 0.2 * f["sessions_ctr"] * np.tanh(f["tenure"] / 24.0 - 1.0)
            + 0.4 * f["premium"] * (f["urban"] - 0.5)
        )

    # Heterogeneous, nonlinear treatment effect tau(X) on the natural scale.
    def tau_fn(X: np.ndarray) -> np.ndarray:
        """Evaluate the heterogeneous treatment effect on the natural scale."""
        f = _split_features(X)

        # Base effect + stronger effect for higher sessions and premium users,
        # diminishes with tenure, mild modulation by spend and urban
        tau = (
            1.2
            + 0.6 * np.tanh(f["sessions_ctr"])                  # saturating in sessions
            + 0.4 * f["premium"]
            - 0.5 * np.tanh(f["tenure"] / 48.0)                 # taper with long tenure
            + 0.2 * f["urban"] * np.tanh(f["spend_ctr"])
        )
        return np.clip(tau, 0.1, 3.0)

    def hte_26_x_sampler(n: int, _k: int, seed: int) -> np.ndarray:
        """Sample correlated covariates used by the HTE benchmark scenario."""
        rng = np.random.default_rng(seed)
        # Sample a correlated latent base first, then derive premium from behavior.
        base_specs = [
            {"name": "tenure_months",     "dist": "lognormal", "mu": np.log(24), "sigma": 0.6, "clip_max": 120},
            {"name": "avg_sessions_week", "dist": "negbin",    "mu": 5, "alpha": 0.5, "clip_max": 40},
            {"name": "spend_last_month",  "dist": "lognormal", "mu": np.log(60), "sigma": 0.9, "clip_max": 500},
            {"name": "urban_resident",    "dist": "bernoulli", "p": 0.60},
        ]
        corr = np.array([
            [1.0, 0.3, 0.2, 0.0],
            [0.3, 1.0, 0.4, 0.0],
            [0.2, 0.4, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        X_base, _ = _gaussian_copula(rng, n, base_specs, corr)

        tenure, sessions, spend, urban = X_base.T
        # Premium is endogenous to user behavior, creating richer confounding.
        logits_p = -5.0 + 0.7 * np.log1p(sessions) + 0.5 * np.log1p(spend) + 0.01 * tenure
        premium = rng.binomial(1, _sigmoid(logits_p)).astype(float)

        # Order: tenure, sessions, spend, premium, urban
        return np.column_stack([tenure, sessions, spend, premium, urban])

    gen = CausalDatasetGenerator(
        outcome_type="continuous",
        sigma_y=3.5,
        target_d_rate=0.35,  # enforce ~35% treated via intercept calibration
        seed=seed,
        # confounders
        confounder_specs=confounder_specs,
        x_sampler=hte_26_x_sampler,
        score_bounding=2.0,
        # Outcome/treatment structure
        beta_y=beta_y,
        beta_d=beta_d,
        g_y=g_y,
        g_d=g_d,
        # Heterogeneous effect
        tau=tau_fn,
        include_oracle=include_oracle,
    )

    df = gen.generate(n)
    df.insert(0, "user_id", np.arange(1, n + 1))

    if not return_causal_data:
        return df

    confounder_names = [s["name"] for s in confounder_specs]
    return CausalData(
        df=df,
        treatment="d",
        outcome="y",
        confounders=confounder_names,
    )

def generate_obs_hte_26_rich(
    n: int = 100000,
    seed: int = 42,
    include_oracle: bool = True,
    return_causal_data: bool = True
) -> Union[pd.DataFrame, CausalData]:
    """
    Observational dataset with richer confounding, nonlinear outcome model,
    nonlinear treatment assignment, and heterogeneous treatment effects.
    Adds additional realistic covariates and dependencies to mimic real data.

    Parameters
    ----------
    n : int, default=100000
        Number of samples.
    seed : int, default=42
        Random seed.
    include_oracle : bool, default=True
        Whether to include oracle ground-truth columns like 'cate', 'propensity', etc.
    return_causal_data : bool, default=True
        If True, returns a CausalData object. If False, returns a pandas DataFrame.

    Returns
    -------
    pandas.DataFrame or CausalData
        Generated rich observational sample, optionally wrapped as
        ``CausalData``.
    """
    # `x_sampler` builds correlated/derived features; specs define output names/order.
    confounder_specs = [
        {"name": "tenure_months",        "dist": "lognormal", "mu": np.log(24), "sigma": 0.6, "clip_max": 120},
        {"name": "avg_sessions_week",    "dist": "negbin",    "mu": 5, "alpha": 0.5, "clip_max": 40},
        {"name": "spend_last_month",     "dist": "lognormal", "mu": np.log(60), "sigma": 0.9, "clip_max": 500},
        {"name": "age_years",            "dist": "normal",    "mu": 36, "sd": 12, "clip_min": 18, "clip_max": 80},
        {"name": "income_monthly",       "dist": "lognormal", "mu": np.log(4000), "sigma": 0.5, "clip_max": 20000},
        {"name": "prior_purchases_12m",  "dist": "poisson",   "lam": 3, "clip_max": 50},
        {"name": "support_tickets_90d",  "dist": "poisson",   "lam": 1.5, "clip_max": 20},
        {"name": "premium_user",         "dist": "bernoulli", "p": 0.25},
        {"name": "mobile_user",          "dist": "bernoulli", "p": 0.65},
        {"name": "urban_resident",       "dist": "bernoulli", "p": 0.60},
        {"name": "referred_user",        "dist": "bernoulli", "p": 0.20},
    ]

    # Reference points for centered/log features used across multiple equations.
    LOG_SESS_REF = np.log1p(5.0)
    LOG_SESS_REF_GD = np.log1p(3.0)
    LOG_SESS_REF_GZI = np.log1p(4.0)
    LOG_SPEND_REF = np.log1p(60.0)
    LOG_SPEND_REF_GZI = np.log1p(50.0)
    LOG_INCOME_REF = np.log1p(4000.0)
    LOG_TICKETS_REF = np.log1p(1.5)
    LOG_TICKETS_REF_GZI = np.log1p(1.0)

    def _split_rich_features(X: np.ndarray) -> dict[str, np.ndarray]:
        """Build transformed feature views reused across rich-model equations."""
        # Centralized rich feature map reused in g_y, g_d, tau, g_zi, tau_zi.
        tenure, sessions, spend, age, income, purchases, tickets, premium, mobile, urban, referred = X.T
        log_sessions = np.log1p(sessions)
        log_spend = np.log1p(spend)
        return {
            "tenure": tenure,
            "age": age,
            "premium": premium,
            "mobile": mobile,
            "urban": urban,
            "referred": referred,
            "log_sessions": log_sessions,
            "log_spend": log_spend,
            "log_income": np.log1p(income),
            "log_purchases": np.log1p(purchases),
            "log_tickets": np.log1p(tickets),
            "sessions_ctr": log_sessions - LOG_SESS_REF,
            "sessions_ctr_gd": log_sessions - LOG_SESS_REF_GD,
            "spend_ctr": log_spend - LOG_SPEND_REF,
        }

    beta_y = np.array([
        0.005,   # tenure_months
        0.00,    # avg_sessions_week (moved to g_y)
        0.00,    # spend_last_month (moved to g_y)
        0.02,    # age_years
        0.00005, # income_monthly
        0.08,    # prior_purchases_12m
        -0.55,   # support_tickets_90d
        1.10,    # premium_user
        0.40,    # mobile_user
        0.60,    # urban_resident
        0.30,    # referred_user
    ], dtype=float)

    def g_y(X: np.ndarray) -> np.ndarray:
        """Evaluate the nonlinear baseline outcome component."""
        f = _split_rich_features(X)

        return (
            1.4 * np.tanh(f["tenure"] / 24.0)
            + 0.6 * f["sessions_ctr"] ** 2
            + 0.25 * f["spend_ctr"] * f["sessions_ctr"]
            + 0.35 * np.tanh((f["log_income"] - LOG_INCOME_REF) / 1.5)
            - 0.45 * f["log_tickets"]
            + 0.20 * f["log_purchases"] * np.tanh(f["tenure"] / 18.0)
            + 0.30 * f["mobile"] * f["sessions_ctr"]
            + 0.25 * f["premium"] * (f["urban"] - 0.5)
            + 0.15 * f["referred"] * np.tanh(f["tenure"] / 12.0 - 1.0)
            - 0.20 * np.tanh((f["age"] - 40.0) / 15.0)
        )

    beta_d = np.array([
        -0.004,  # tenure_months
        0.00,    # avg_sessions_week (moved to g_d)
        0.00,    # spend_last_month (moved to g_d)
        -0.012,  # age_years
        -0.00005,# income_monthly
        -0.04,   # prior_purchases_12m
        0.22,    # support_tickets_90d
        -0.45,   # premium_user
        -0.08,   # mobile_user
        -0.12,   # urban_resident
        0.10,    # referred_user
    ], dtype=float)

    def g_d(X: np.ndarray) -> np.ndarray:
        """Evaluate the nonlinear treatment-assignment component."""
        f = _split_rich_features(X)

        # Selection is tilted toward lower-baseline users so observed treated means can be lower.
        soft_spend = -0.55 * np.tanh(f["spend_ctr"])
        return (
            soft_spend
            - 0.20 * f["sessions_ctr"] * np.tanh(f["tenure"] / 24.0 - 1.0)
            - 0.25 * f["premium"] * (f["urban"] - 0.5)
            - 0.20 * f["mobile"] * np.tanh(f["sessions_ctr_gd"])
            + 0.30 * f["referred"] * (1.0 - np.tanh(f["tenure"] / 36.0))
            + 0.35 * np.tanh(f["log_tickets"] - LOG_TICKETS_REF)
            - 0.25 * np.tanh((f["log_income"] - LOG_INCOME_REF) / 1.3)
            - 0.12 * np.tanh(f["tenure"] / 24.0)
            - 0.10 * np.tanh((f["age"] - 45.0) / 12.0)
        )

    def tau_fn(X: np.ndarray) -> np.ndarray:
        """Evaluate the treatment effect shift for the positive outcome branch."""
        f = _split_rich_features(X)

        # tau is a log-mean shift for the positive-part branch (not natural-scale CATE).
        # Keep this moderate so the resulting ATTE on the natural scale is not extreme.
        tau = (
            0.08
            + 0.08 * np.tanh(f["sessions_ctr"])
            + 0.11 * f["premium"]
            + 0.03 * f["mobile"]
            + 0.03 * f["referred"]
            - 0.07 * np.tanh(f["tenure"] / 48.0)
            - 0.05 * np.tanh((f["age"] - 40.0) / 15.0)
            + 0.03 * f["urban"] * np.tanh(f["spend_ctr"])
            - 0.04 * np.tanh(f["log_tickets"] - LOG_TICKETS_REF)
        )
        return np.clip(tau, 0.005, 0.35)

    def hte_26_rich_x_sampler(n: int, _k: int, seed: int) -> np.ndarray:
        """Sample correlated rich covariates and derived behavior indicators."""
        rng = np.random.default_rng(seed)
        # Start from correlated core demographics/usage, then generate derived binaries/counts.
        base_specs = [
            {"name": "tenure_months",     "dist": "lognormal", "mu": np.log(24), "sigma": 0.6, "clip_max": 120},
            {"name": "avg_sessions_week", "dist": "negbin",    "mu": 5, "alpha": 0.5, "clip_max": 40},
            {"name": "spend_last_month",  "dist": "lognormal", "mu": np.log(60), "sigma": 0.9, "clip_max": 500},
            {"name": "age_years",         "dist": "normal",    "mu": 36, "sd": 12, "clip_min": 18, "clip_max": 80},
            {"name": "income_monthly",    "dist": "lognormal", "mu": np.log(4000), "sigma": 0.5, "clip_max": 20000},
            {"name": "urban_resident",    "dist": "bernoulli", "p": 0.60},
        ]
        corr = np.array([
            [1.00, 0.20, 0.20, 0.30, 0.20, 0.00],
            [0.20, 1.00, 0.45, -0.20, 0.30, 0.10],
            [0.20, 0.45, 1.00, -0.10, 0.40, 0.10],
            [0.30, -0.20, -0.10, 1.00, 0.20, -0.10],
            [0.20, 0.30, 0.40, 0.20, 1.00, 0.10],
            [0.00, 0.10, 0.10, -0.10, 0.10, 1.00],
        ])
        X_base, _ = _gaussian_copula(rng, n, base_specs, corr)

        tenure, sessions, spend, age, income, urban = X_base.T

        log_sessions = np.log1p(sessions)
        log_spend = np.log1p(spend)
        log_income = np.log1p(income)

        # Premium/mobile/referred are behavior-linked, not independent Bernoullis.
        logits_p = -5.0 + 0.7 * log_sessions + 0.45 * log_spend + 0.35 * log_income + 0.015 * tenure
        premium = rng.binomial(1, _sigmoid(logits_p)).astype(float)

        logits_m = 1.5 - 0.03 * (age - 35.0) + 0.30 * urban + 0.25 * log_sessions
        mobile = rng.binomial(1, _sigmoid(logits_m)).astype(float)

        logits_r = -1.2 - 0.02 * (tenure - 12.0) + 0.45 * mobile + 0.20 * urban
        referred = rng.binomial(1, _sigmoid(logits_r)).astype(float)

        # Event-intensity counts are generated from clipped Poisson means for stability.
        mean_purchases = 1.0 + 0.55 * log_sessions + 0.45 * log_spend + 0.25 * premium
        purchases = rng.poisson(lam=np.clip(mean_purchases, 0.1, 30.0)).astype(float)

        mean_tickets = 0.6 + 0.25 * log_sessions + 0.30 * (1.0 - premium) + 0.15 * np.tanh((age - 45.0) / 12.0)
        tickets = rng.poisson(lam=np.clip(mean_tickets, 0.05, 10.0)).astype(float)

        purchases = np.clip(purchases, 0, 50)
        tickets = np.clip(tickets, 0, 20)

        return np.column_stack([
            tenure,
            sessions,
            spend,
            age,
            income,
            purchases,
            tickets,
            premium,
            mobile,
            urban,
            referred,
        ])

    def g_zi(X: np.ndarray) -> np.ndarray:
        """Evaluate the zero-inflation baseline logit component."""
        f = _split_rich_features(X)

        # Zero-inflation baseline: logit P(y>0 | X, D=0).
        return (
            0.45 * np.tanh(f["log_sessions"] - LOG_SESS_REF_GZI)
            + 0.25 * np.tanh(f["log_spend"] - LOG_SPEND_REF_GZI)
            + 0.20 * f["log_purchases"]
            - 0.30 * np.tanh(f["log_tickets"] - LOG_TICKETS_REF_GZI)
            + 0.15 * f["premium"]
            + 0.10 * f["mobile"]
        )

    def tau_zi_fn(X: np.ndarray) -> np.ndarray:
        """Evaluate the treatment effect on the zero-inflation logit branch."""
        f = _split_rich_features(X)

        # Effect on the nonzero-probability branch: logit shift when treated.
        return (
            0.03
            + 0.02 * f["premium"]
            + 0.015 * f["referred"]
            - 0.02 * np.tanh(f["tenure"] / 48.0)
        )

    gen = CausalDatasetGenerator(
        outcome_type="tweedie",
        sigma_y=3.8,
        target_d_rate=0.05,
        seed=seed,
        confounder_specs=confounder_specs,
        x_sampler=hte_26_rich_x_sampler,
        score_bounding=2.0,
        # Positive-part mean model (log link).
        beta_y=beta_y,
        beta_d=beta_d,
        g_y=g_y,
        g_d=g_d,
        tau=tau_fn,
        # Zero-inflation model (logit link).
        alpha_zi=-0.8,
        beta_zi=np.array([
            0.00,   # tenure_months
            0.02,   # avg_sessions_week
            0.00,   # spend_last_month
            -0.01,  # age_years
            0.00002,# income_monthly
            0.10,   # prior_purchases_12m
            -0.15,  # support_tickets_90d
            0.30,   # premium_user
            0.20,   # mobile_user
            0.05,   # urban_resident
            0.15,   # referred_user
        ], dtype=float),
        g_zi=g_zi,
        tau_zi=tau_zi_fn,
        pos_dist="gamma",
        gamma_shape=2.2,
        include_oracle=include_oracle,
    )

    df = gen.generate(n)
    df.insert(0, "user_id", np.arange(1, n + 1))
    if include_oracle and {"g0", "g1"}.issubset(df.columns):
        # Keep oracle CATE definition explicit on natural scale for this scenario.
        df["cate"] = df["g1"] - df["g0"]

    if not return_causal_data:
        return df

    confounder_names = [s["name"] for s in confounder_specs]
    return CausalData(
        df=df,
        treatment="d",
        outcome="y",
        confounders=confounder_names,
    )


def generate_obs_hte_binary_26(
    n: int = 100000,
    seed: int = 42,
    include_oracle: bool = True,
    return_causal_data: bool = True
) -> Union[pd.DataFrame, CausalData]:
    """
    Observational binary-outcome dataset with nonlinear confounding and
    heterogeneous treatment effects.

    This scenario follows the structure of `generate_obs_hte_26_rich`, but uses
    a binary outcome model and a modified confounder set.

    Parameters
    ----------
    n : int, default=100000
        Number of samples.
    seed : int, default=42
        Random seed.
    include_oracle : bool, default=True
        Whether to include oracle columns like 'cate', 'propensity', etc.
    return_causal_data : bool, default=True
        If True, returns a CausalData object. If False, returns a pandas DataFrame.

    Returns
    -------
    pandas.DataFrame or CausalData
        Generated binary-outcome sample, optionally wrapped as ``CausalData``.
    """
    # Modified confounder set vs `generate_obs_hte_26_rich`:
    # - removed: income_monthly, urban_resident
    # - added: weekend_user, email_opt_in
    confounder_specs = [
        {"name": "tenure_months",        "dist": "lognormal", "mu": np.log(24), "sigma": 0.6, "clip_max": 120},
        {"name": "avg_sessions_week",    "dist": "negbin",    "mu": 5, "alpha": 0.5, "clip_max": 40},
        {"name": "spend_last_month",     "dist": "lognormal", "mu": np.log(60), "sigma": 0.9, "clip_max": 500},
        {"name": "age_years",            "dist": "normal",    "mu": 36, "sd": 12, "clip_min": 18, "clip_max": 80},
        {"name": "prior_purchases_12m",  "dist": "poisson",   "lam": 3, "clip_max": 50},
        {"name": "support_tickets_90d",  "dist": "poisson",   "lam": 1.5, "clip_max": 20},
        {"name": "premium_user",         "dist": "bernoulli", "p": 0.25},
        {"name": "mobile_user",          "dist": "bernoulli", "p": 0.65},
        {"name": "weekend_user",         "dist": "bernoulli", "p": 0.55},
        {"name": "email_opt_in",         "dist": "bernoulli", "p": 0.60},
        {"name": "referred_user",        "dist": "bernoulli", "p": 0.20},
    ]

    # Reference values used to center log-transformed features.
    LOG_SESS_REF = np.log1p(5.0)
    LOG_SESS_REF_GD = np.log1p(3.0)
    LOG_SPEND_REF = np.log1p(60.0)
    LOG_TICKETS_REF = np.log1p(1.5)

    def _split_binary_features(X: np.ndarray) -> dict[str, np.ndarray]:
        """Build transformed feature views reused across binary-model equations."""
        # Shared feature engineering across g_y / g_d / tau.
        tenure, sessions, spend, age, purchases, tickets, premium, mobile, weekend, email, referred = X.T
        log_sessions = np.log1p(sessions)
        log_spend = np.log1p(spend)
        return {
            "tenure": tenure,
            "age": age,
            "premium": premium,
            "mobile": mobile,
            "weekend": weekend,
            "email": email,
            "referred": referred,
            "log_sessions": log_sessions,
            "log_spend": log_spend,
            "log_purchases": np.log1p(purchases),
            "log_tickets": np.log1p(tickets),
            "sessions_ctr": log_sessions - LOG_SESS_REF,
            "sessions_ctr_gd": log_sessions - LOG_SESS_REF_GD,
            "spend_ctr": log_spend - LOG_SPEND_REF,
        }

    # Baseline log-odds model components for Y.
    beta_y = np.array([
        0.004,   # tenure_months
        0.00,    # avg_sessions_week (moved to g_y)
        0.00,    # spend_last_month (moved to g_y)
        -0.012,  # age_years
        0.09,    # prior_purchases_12m
        -0.50,   # support_tickets_90d
        0.80,    # premium_user
        0.30,    # mobile_user
        0.20,    # weekend_user
        0.28,    # email_opt_in
        0.45,    # referred_user
    ], dtype=float)

    def g_y(X: np.ndarray) -> np.ndarray:
        """Evaluate the baseline log-odds component for the binary outcome."""
        f = _split_binary_features(X)
        return (
            1.1 * np.tanh(f["tenure"] / 24.0)
            + 0.50 * f["sessions_ctr"] ** 2
            + 0.22 * f["spend_ctr"] * f["sessions_ctr"]
            - 0.35 * f["log_tickets"]
            + 0.18 * f["log_purchases"] * np.tanh(f["tenure"] / 18.0)
            + 0.22 * f["mobile"] * f["sessions_ctr"]
            + 0.20 * f["premium"] * (f["weekend"] - 0.5)
            + 0.18 * f["email"] * np.tanh(f["spend_ctr"] / 2.0)
            + 0.14 * f["referred"] * np.tanh(f["tenure"] / 12.0 - 1.0)
            - 0.20 * np.tanh((f["age"] - 40.0) / 15.0)
        )

    # Treatment assignment score components.
    beta_d = np.array([
        0.002,   # tenure_months
        0.00,    # avg_sessions_week (moved to g_d)
        0.00,    # spend_last_month (moved to g_d)
        -0.008,  # age_years
        0.04,    # prior_purchases_12m
        0.08,    # support_tickets_90d
        0.65,    # premium_user
        0.24,    # mobile_user
        0.34,    # weekend_user
        0.18,    # email_opt_in
        0.40,    # referred_user
    ], dtype=float)

    def g_d(X: np.ndarray) -> np.ndarray:
        """Evaluate the nonlinear treatment-assignment component."""
        f = _split_binary_features(X)
        soft_spend = 0.85 * np.tanh(f["spend_ctr"])
        return (
            soft_spend
            + 0.24 * f["sessions_ctr"] * np.tanh(f["tenure"] / 24.0 - 1.0)
            + 0.30 * f["premium"] * (f["weekend"] - 0.5)
            + 0.22 * f["email"] * np.tanh(f["sessions_ctr_gd"])
            + 0.25 * f["referred"] * (1.0 - np.tanh(f["tenure"] / 36.0))
            + 0.14 * np.tanh(f["log_tickets"] - LOG_TICKETS_REF)
            - 0.10 * np.tanh((f["age"] - 45.0) / 12.0)
        )

    def tau_fn(X: np.ndarray) -> np.ndarray:
        """Evaluate the heterogeneous treatment effect on the log-odds scale."""
        f = _split_binary_features(X)
        # For binary outcomes, tau is on log-odds scale; oracle cate remains
        # risk difference via g1 - g0 in CausalDatasetGenerator.
        tau = (
            0.30
            + 0.40 * np.tanh(f["sessions_ctr"])
            + 0.45 * f["premium"]
            + 0.12 * f["mobile"]
            + 0.10 * f["email"]
            + 0.12 * f["referred"]
            - 0.32 * np.tanh(f["tenure"] / 48.0)
            - 0.18 * np.tanh((f["age"] - 40.0) / 15.0)
            + 0.10 * f["weekend"] * np.tanh(f["spend_ctr"])
            - 0.12 * np.tanh(f["log_tickets"] - LOG_TICKETS_REF)
        )
        return np.clip(tau, -0.35, 1.4)

    def hte_binary_26_x_sampler(n: int, _k: int, seed: int) -> np.ndarray:
        """Sample correlated covariates and derived binary-behavior features."""
        rng = np.random.default_rng(seed)
        # Correlated latent base; binary/count covariates are derived from behavior.
        base_specs = [
            {"name": "tenure_months",     "dist": "lognormal", "mu": np.log(24), "sigma": 0.6, "clip_max": 120},
            {"name": "avg_sessions_week", "dist": "negbin",    "mu": 5, "alpha": 0.5, "clip_max": 40},
            {"name": "spend_last_month",  "dist": "lognormal", "mu": np.log(60), "sigma": 0.9, "clip_max": 500},
            {"name": "age_years",         "dist": "normal",    "mu": 36, "sd": 12, "clip_min": 18, "clip_max": 80},
            {"name": "weekend_user",      "dist": "bernoulli", "p": 0.55},
        ]
        corr = np.array([
            [1.00, 0.25, 0.20, 0.25, 0.05],
            [0.25, 1.00, 0.45, -0.20, 0.30],
            [0.20, 0.45, 1.00, -0.10, 0.20],
            [0.25, -0.20, -0.10, 1.00, -0.15],
            [0.05, 0.30, 0.20, -0.15, 1.00],
        ])
        X_base, _ = _gaussian_copula(rng, n, base_specs, corr)

        tenure, sessions, spend, age, weekend = X_base.T
        log_sessions = np.log1p(sessions)
        log_spend = np.log1p(spend)

        logits_p = -4.6 + 0.70 * log_sessions + 0.45 * log_spend + 0.012 * tenure + 0.20 * weekend
        premium = rng.binomial(1, _sigmoid(logits_p)).astype(float)

        logits_m = 1.4 - 0.03 * (age - 35.0) + 0.25 * weekend + 0.22 * log_sessions
        mobile = rng.binomial(1, _sigmoid(logits_m)).astype(float)

        logits_r = -1.3 - 0.02 * (tenure - 12.0) + 0.42 * mobile + 0.25 * weekend
        referred = rng.binomial(1, _sigmoid(logits_r)).astype(float)

        logits_e = -0.6 + 0.50 * mobile + 0.35 * premium + 0.20 * weekend + 0.12 * log_sessions
        email = rng.binomial(1, _sigmoid(logits_e)).astype(float)

        mean_purchases = 0.9 + 0.52 * log_sessions + 0.40 * log_spend + 0.24 * premium + 0.12 * email
        purchases = rng.poisson(lam=np.clip(mean_purchases, 0.05, 25.0)).astype(float)

        mean_tickets = (
            0.55
            + 0.22 * log_sessions
            + 0.32 * (1.0 - premium)
            + 0.18 * (1.0 - email)
            + 0.12 * np.tanh((age - 45.0) / 12.0)
        )
        tickets = rng.poisson(lam=np.clip(mean_tickets, 0.05, 10.0)).astype(float)

        purchases = np.clip(purchases, 0, 50)
        tickets = np.clip(tickets, 0, 20)

        return np.column_stack([
            tenure,
            sessions,
            spend,
            age,
            purchases,
            tickets,
            premium,
            mobile,
            weekend,
            email,
            referred,
        ])

    gen = CausalDatasetGenerator(
        outcome_type="binary",
        alpha_y=-1.7,
        target_d_rate=0.15,
        seed=seed,
        confounder_specs=confounder_specs,
        x_sampler=hte_binary_26_x_sampler,
        score_bounding=2.0,
        beta_y=beta_y,
        beta_d=beta_d,
        g_y=g_y,
        g_d=g_d,
        tau=tau_fn,
        include_oracle=include_oracle,
    )

    df = gen.generate(n)

    if not return_causal_data:
        return df

    confounder_names = [s["name"] for s in confounder_specs]
    return CausalData(
        df=df,
        treatment="d",
        outcome="y",
        confounders=confounder_names,
    )
