"""High-level instrumental-variable DGPs with realistic business covariates."""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd

from causalis.data_contracts.iv_causal_data import IVCausalData
from causalis.dgp.base import _deterministic_ids, _sigmoid
from causalis.dgp.causaldata_instrumental import InstrumentalGenerator

_OUTCOME = "net_revenue_90d"
_TREATMENT = "accepted_offer"
_INSTRUMENT = "offer_eligible"

_CONFOUNDER_SPECS = [
    {"name": "age", "dist": "normal"},
    {"name": "tenure_months", "dist": "normal"},
    {"name": "annual_income", "dist": "lognormal"},
    {"name": "credit_score", "dist": "normal"},
    {"name": "app_sessions_30d", "dist": "poisson"},
    {"name": "prior_spend_30d", "dist": "lognormal"},
    {"name": "savings_balance", "dist": "lognormal"},
    {"name": "premium_user", "dist": "bernoulli"},
    {"name": "autopay_enabled", "dist": "bernoulli"},
    {"name": "region_north", "dist": "bernoulli"},
    {"name": "region_west", "dist": "bernoulli"},
    {"name": "acquisition_paid", "dist": "bernoulli"},
]

_COL = {spec["name"]: i for i, spec in enumerate(_CONFOUNDER_SPECS)}
_ORACLE_COLS = {
    "m",
    "r_obs",
    "r_z0",
    "r_z1",
    "g_z0",
    "g_z1",
    "iv_first_stage",
    "iv_reduced_form",
    "late_x",
    "late",
    "tau_link",
    "g_d0",
    "g_d1",
    "cate",
}


def _customer_features(x: np.ndarray) -> dict[str, np.ndarray]:
    """Return scaled feature views used by the scenario equations."""
    return {
        "age_ctr": (x[:, _COL["age"]] - 38.0) / 12.0,
        "tenure_ctr": np.log1p(x[:, _COL["tenure_months"]]) - np.log1p(24.0),
        "income_ctr": np.log1p(x[:, _COL["annual_income"]]) - np.log1p(80_000.0),
        "credit_ctr": (x[:, _COL["credit_score"]] - 680.0) / 70.0,
        "sessions_ctr": np.log1p(x[:, _COL["app_sessions_30d"]]) - np.log1p(8.0),
        "spend_ctr": np.log1p(x[:, _COL["prior_spend_30d"]]) - np.log1p(120.0),
        "balance_ctr": np.log1p(x[:, _COL["savings_balance"]]) - np.log1p(4_000.0),
        "premium": x[:, _COL["premium_user"]],
        "autopay": x[:, _COL["autopay_enabled"]],
        "north": x[:, _COL["region_north"]],
        "west": x[:, _COL["region_west"]],
        "paid": x[:, _COL["acquisition_paid"]],
    }


def _sample_offer_iv_customers(n: int, _k: int, seed: int) -> np.ndarray:
    """
    Sample realistic fintech/subscription customer covariates.

    The sampler uses latent engagement, affluence, and risk factors to create
    observed correlations among income, credit, spend, app activity, and prior
    product usage.
    """
    rng = np.random.default_rng(seed)
    engagement = rng.normal(size=n)
    affluence = rng.normal(size=n)
    risk = rng.normal(size=n)

    age = np.clip(rng.normal(38.0 + 3.0 * affluence, 11.0, n), 18.0, 74.0).round()
    tenure_months = np.clip(
        rng.gamma(shape=2.2, scale=13.0, size=n) + 4.0 * np.maximum(engagement, 0.0),
        1.0,
        144.0,
    )
    annual_income = np.clip(
        rng.lognormal(mean=np.log(78_000.0) + 0.35 * affluence, sigma=0.45, size=n),
        18_000.0,
        280_000.0,
    )
    credit_score = np.clip(
        rng.normal(675.0 + 38.0 * affluence - 32.0 * risk, 42.0, n),
        420.0,
        850.0,
    ).round()

    premium_p = _sigmoid(-1.15 + 0.75 * affluence + 0.40 * engagement - 0.20 * risk)
    premium_user = rng.binomial(1, premium_p, n).astype(float)

    sessions_lam = np.exp(
        np.clip(1.85 + 0.38 * engagement + 0.20 * premium_user - 0.08 * risk, -1.0, 3.5)
    )
    app_sessions_30d = np.clip(rng.poisson(sessions_lam, size=n), 0, 60).astype(float)

    prior_spend_30d = np.clip(
        rng.lognormal(
            mean=np.log(115.0) + 0.25 * engagement + 0.22 * affluence + 0.18 * premium_user,
            sigma=0.70,
            size=n,
        ),
        0.0,
        1_500.0,
    )
    savings_balance = np.clip(
        rng.lognormal(mean=np.log(4_000.0) + 0.55 * affluence - 0.20 * risk, sigma=1.0, size=n),
        50.0,
        150_000.0,
    )
    autopay_p = _sigmoid(
        -0.55
        + 0.55 * premium_user
        + 0.35 * ((credit_score - 680.0) / 70.0)
        + 0.18 * engagement
    )
    autopay_enabled = rng.binomial(1, autopay_p, n).astype(float)

    regions = rng.choice(["south", "east", "north", "west"], size=n, p=[0.30, 0.28, 0.22, 0.20])
    acquisition_paid_p = _sigmoid(-0.35 + 0.35 * engagement - 0.25 * affluence + 0.15 * risk)
    acquisition_paid = rng.binomial(1, acquisition_paid_p, n).astype(float)

    return np.column_stack(
        [
            age.astype(float),
            tenure_months.astype(float),
            annual_income.astype(float),
            credit_score.astype(float),
            app_sessions_30d.astype(float),
            prior_spend_30d.astype(float),
            savings_balance.astype(float),
            premium_user.astype(float),
            autopay_enabled.astype(float),
            (regions == "north").astype(float),
            (regions == "west").astype(float),
            acquisition_paid.astype(float),
        ]
    )


def generate_offer_iv_26(
    n: int = 20_000,
    seed: int = 42,
    include_oracle: bool = True,
    return_causal_data: bool = True,
    *,
    deterministic_ids: bool = True,
) -> Union[pd.DataFrame, IVCausalData]:
    """
    Generate a realistic IV dataset with a positive business effect.

    This scenario mimics an offer-eligibility experiment in a customer product:

    - ``offer_eligible`` is the binary instrument (Z). It affects whether customers
      can accept an offer, but it has no direct outcome effect in the DGP.
    - ``accepted_offer`` is the binary endogenous treatment (D).
    - ``net_revenue_90d`` is a continuous outcome (Y) with a positive heterogeneous
      treatment effect among customers induced into treatment by eligibility.

    The DGP follows a structural model where unobserved confounders :math:`U`
    affect both treatment :math:`D` and outcome :math:`Y`, but not the
    instrument :math:`Z`:

    .. math::

        Z &= f_Z(X, \epsilon_Z) \\
        D &= f_D(Z, X, U, \epsilon_D) \\
        Y &= f_Y(D, X, U, \epsilon_Y)

    Parameters
    ----------
    n : int, default 20_000
        Number of observations to generate.
    seed : int, default 42
        Random seed for reproducibility.
    include_oracle : bool, default True
        Whether to include latent variables (ITE, LATE, etc.) in the output DataFrame.
    return_causal_data : bool, default True
        If True, returns an :class:`~causalis.data_contracts.iv_causal_data.IVCausalData` object.
        If False, returns a :class:`pandas.DataFrame`.
    deterministic_ids : bool, default True
        Whether to generate stable, deterministic user IDs.

    Returns
    -------
    Union[pd.DataFrame, IVCausalData]
        The generated dataset.

    Examples
    --------
    >>> from causalis.scenarios.iv.dgp import generate_offer_iv_26
    >>> # Generate data as a pandas DataFrame
    >>> data = generate_offer_iv_26(n=5000, return_causal_data=False)
    >>> data[['offer_eligible', 'accepted_offer', 'net_revenue_90d']].head()
    """

    def g_y(x: np.ndarray) -> np.ndarray:
        f = _customer_features(x)
        return (
            42.0
            + 8.0 * f["income_ctr"]
            + 9.0 * f["spend_ctr"]
            + 5.0 * f["sessions_ctr"]
            + 7.0 * f["premium"]
            + 4.0 * f["autopay"]
            + 3.5 * f["balance_ctr"]
            - 2.5 * f["paid"]
            + 2.0 * f["west"]
            + 1.5 * f["north"]
            + 2.0 * f["premium"] * f["sessions_ctr"]
            + 2.5 * np.tanh(f["tenure_ctr"])
        )

    def g_d(x: np.ndarray) -> np.ndarray:
        f = _customer_features(x)
        return (
            0.45 * f["sessions_ctr"]
            + 0.38 * f["spend_ctr"]
            + 0.40 * f["premium"]
            + 0.28 * f["autopay"]
            + 0.18 * f["income_ctr"]
            - 0.20 * f["paid"]
            + 0.14 * f["west"]
            - 0.10 * f["age_ctr"]
        )

    def g_z(x: np.ndarray) -> np.ndarray:
        f = _customer_features(x)
        return (
            0.52 * f["credit_ctr"]
            + 0.36 * f["tenure_ctr"]
            + 0.24 * f["income_ctr"]
            + 0.18 * f["autopay"]
            + 0.16 * f["north"]
            - 0.18 * f["paid"]
        )

    def tau(x: np.ndarray) -> np.ndarray:
        f = _customer_features(x)
        effect = (
            16.0
            + 5.0 * np.tanh(f["sessions_ctr"])
            + 4.0 * f["premium"]
            + 3.0 * np.tanh(f["spend_ctr"])
            + 2.0 * f["autopay"]
            - 2.0 * np.maximum(f["age_ctr"], 0.0)
        )
        return np.clip(effect, 8.0, 32.0)

    gen = InstrumentalGenerator(
        outcome_type="continuous",
        sigma_y=18.0,
        alpha_y=0.0,
        tau=tau,
        first_stage=1.55,
        alpha_d=-0.35,
        target_d_rate=0.34,
        target_z_rate=0.46,
        confounder_specs=_CONFOUNDER_SPECS,
        x_sampler=_sample_offer_iv_customers,
        g_y=g_y,
        g_d=g_d,
        g_z=g_z,
        u_strength_d=0.75,
        u_strength_y=9.0,
        propensity_sharpness=1.0,
        instrument_sharpness=1.0,
        seed=seed,
        include_oracle=include_oracle,
    )
    df = gen.generate(n)
    df = df.rename(columns={"y": _OUTCOME, "d": _TREATMENT, "z": _INSTRUMENT})

    rng = np.random.default_rng(seed)
    user_ids = (
        _deterministic_ids(rng, len(df))
        if deterministic_ids
        else [f"cust_{rng.integers(0, 16**10):010x}" for _ in range(len(df))]
    )
    df.insert(0, "user_id", user_ids)

    ordered = _order_offer_iv_columns(df)
    if not return_causal_data:
        return ordered

    confounders = [spec["name"] for spec in _CONFOUNDER_SPECS]
    return IVCausalData.from_df(
        ordered,
        treatment=_TREATMENT,
        outcome=_OUTCOME,
        instruments=_INSTRUMENT,
        confounders=confounders,
        user_id="user_id",
    )


def _order_offer_iv_columns(df: pd.DataFrame) -> pd.DataFrame:
    core = ["user_id", _OUTCOME, _TREATMENT, _INSTRUMENT]
    confounders = [spec["name"] for spec in _CONFOUNDER_SPECS if spec["name"] in df.columns]
    oracle = [c for c in df.columns if c in _ORACLE_COLS]
    return df[core + confounders + oracle]


__all__ = ["generate_offer_iv_26"]
