from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Callable, List, Optional, Union

from causalis.data_contracts.multicausaldata import MultiCausalData
from causalis.dgp.multicausaldata.functional import generate_multitreatment

# Shared defaults across the 26 multi-treatment scenarios.
_D_NAMES = ["d_0", "d_1", "d_2"]
_TARGET_D_RATE = [0.5, 0.25, 0.25]
_COPULA_RHO = 0.30
_CX_TREATMENT_NAMES = ["control", "neg_contact_flg", "error_flg", "neg_contact_flg_error_flg"]
_CX_CONFOUNDER_SPECS = [
    {"name": "age", "dist": "normal"},
    {"name": "risk_latent", "dist": "normal"},
    {"name": "income", "dist": "normal"},
    {"name": "sessions_30d", "dist": "poisson"},
    {"name": "clicks_7d", "dist": "poisson"},
    {"name": "n_products", "dist": "poisson"},
    {"name": "has_debt", "dist": "bernoulli"},
    {"name": "csat_prev", "dist": "normal"},
    {"name": "prev_contact", "dist": "bernoulli"},
    {"name": "prev_repeat", "dist": "bernoulli"},
    {"name": "prev_apps", "dist": "bernoulli"},
    {"name": "prev_util", "dist": "bernoulli"},
    {"name": "product_emb_1", "dist": "normal"},
    {"name": "product_emb_2", "dist": "uniform"},
    {"name": "channel_callcenter", "dist": "bernoulli"},
    {"name": "channel_partner", "dist": "bernoulli"},
    {"name": "channel_web", "dist": "bernoulli"},
    {"name": "region_B", "dist": "bernoulli"},
    {"name": "region_C", "dist": "bernoulli"},
    {"name": "region_D", "dist": "bernoulli"},
]
_CX_COL = {spec["name"]: idx for idx, spec in enumerate(_CX_CONFOUNDER_SPECS)}
_CX_P_PREV_CONTACT = 0.25
_CX_P_PREV_REPEAT = 0.18
_CX_P_LONG_T2D = 0.22
_CX_P_PREV_APPS = 0.30
_CX_P_PREV_UTIL = 0.28


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return 1.0 / (1.0 + np.exp(-values))


def _toeplitz_copula_corr(n_features: int, rho: float = _COPULA_RHO) -> np.ndarray:
    # Positive-definite Toeplitz structure: nearby features are more correlated.
    idx = np.arange(n_features)
    return rho ** np.abs(idx[:, None] - idx[None, :])


def _run_multitreatment_26(
    *,
    n: int,
    seed: int,
    include_oracle: bool,
    return_causal_data: bool,
    outcome_type: str,
    confounder_specs: List[dict],
    beta_y: np.ndarray,
    beta_d: np.ndarray,
    theta: List[float],
    tau: List[Optional[Callable[[np.ndarray], np.ndarray]]],
    alpha_y: float = 0.0,
    gamma_shape: float = 2.0,
) -> Union[pd.DataFrame, MultiCausalData]:
    # Centralized wrapper so binary/gamma variants share calibration + naming conventions.
    return generate_multitreatment(
        n=n,
        n_treatments=3,
        outcome_type=outcome_type,
        alpha_y=alpha_y,
        gamma_shape=gamma_shape,
        tau=tau,
        target_d_rate=_TARGET_D_RATE,
        confounder_specs=confounder_specs,
        use_copula=True,
        copula_corr=_toeplitz_copula_corr(len(confounder_specs)),
        beta_y=beta_y,
        beta_d=beta_d,
        theta=theta,
        random_state=seed,
        include_oracle=include_oracle,
        return_causal_data=return_causal_data,
        d_names=_D_NAMES,
    )


def _sample_multi_dml_cx_26_x(n: int, _k: int, seed: int) -> np.ndarray:
    """
    Sample the notebook-style CX confounders used by ``generate_multi_dml_cx_26``.

    The source notebook builds contact/repeat propensities from a latent risk
    factor plus channel/region effects, then one-hot encodes the categorical
    features. We reproduce that observed covariate space here so the scenario
    can be generated through the shared multi-treatment DGP engine.
    """
    rng = np.random.default_rng(seed)

    age = np.clip(rng.normal(37.0, 11.0, n), 18.0, 70.0).round()
    age_centered = (age - 37.0) / 10.0

    channels = rng.choice(["web", "app", "partner", "callcenter"], size=n, p=[0.40, 0.35, 0.15, 0.10])
    regions = rng.choice(["A", "B", "C", "D"], size=n, p=[0.35, 0.30, 0.20, 0.15])

    risk = rng.normal(0.0, 1.0, n)
    prev_contact = rng.binomial(1, _CX_P_PREV_CONTACT, n).astype(float)
    prev_repeat = rng.binomial(1, _CX_P_PREV_REPEAT, n).astype(float)

    prev_apps_logit = np.log(_CX_P_PREV_APPS / (1.0 - _CX_P_PREV_APPS)) + 0.7 * risk + 0.15 * age_centered
    prev_apps = rng.binomial(1, _sigmoid(prev_apps_logit), n).astype(float)

    prev_util_logit = np.log(_CX_P_PREV_UTIL / (1.0 - _CX_P_PREV_UTIL)) + 1.1 * prev_apps + 0.6 * risk
    prev_util = rng.binomial(1, _sigmoid(prev_util_logit), n).astype(float)

    sessions_30d = rng.poisson(lam=8.0 + 2.0 * _sigmoid(risk) + 1.2 * prev_apps, size=n).astype(float)
    clicks_7d = rng.poisson(lam=12.0 + 3.0 * _sigmoid(risk) + 0.8 * prev_apps, size=n).astype(float)
    income = np.clip(rng.normal(85_000.0, 30_000.0, n) + 8_000.0 * risk, 25_000.0, 250_000.0)

    p_long = _sigmoid(np.log(_CX_P_LONG_T2D / (1.0 - _CX_P_LONG_T2D)) + 0.10 * (sessions_30d - 8.0) / 5.0)

    n_products = np.clip(rng.poisson(1.6 + 0.3 * _sigmoid(risk) + 0.4 * prev_util, size=n), 0, 8).astype(float)
    has_debt = rng.binomial(1, _sigmoid(-1.0 + 0.6 * risk + 0.15 * age_centered + 0.25 * prev_apps), n).astype(float)

    contact_logit = _cx_contact_logit_from_latents(
        age=age,
        risk=risk,
        sessions_30d=sessions_30d,
        prev_contact=prev_contact,
        prev_apps=prev_apps,
        prev_util=prev_util,
        channels=channels,
    )
    csat_prev = np.clip(rng.normal(4.2 - 0.4 * _sigmoid(contact_logit) - 0.3 * p_long, 0.6, n), 1.0, 5.0)

    product_emb_1 = rng.normal(0.0, 1.0, n)
    product_emb_2 = rng.integers(0, 100, n).astype(float)

    return np.column_stack([
        age.astype(float),
        risk.astype(float),
        income.astype(float),
        sessions_30d.astype(float),
        clicks_7d.astype(float),
        n_products.astype(float),
        has_debt.astype(float),
        csat_prev.astype(float),
        prev_contact.astype(float),
        prev_repeat.astype(float),
        prev_apps.astype(float),
        prev_util.astype(float),
        product_emb_1.astype(float),
        product_emb_2.astype(float),
        (channels == "callcenter").astype(float),
        (channels == "partner").astype(float),
        (channels == "web").astype(float),
        (regions == "B").astype(float),
        (regions == "C").astype(float),
        (regions == "D").astype(float),
    ])


def _cx_contact_logit_from_latents(
    *,
    age: np.ndarray,
    risk: np.ndarray,
    sessions_30d: np.ndarray,
    prev_contact: np.ndarray,
    prev_apps: np.ndarray,
    prev_util: np.ndarray,
    channels: np.ndarray,
) -> np.ndarray:
    channel_shift = pd.Series(channels).map(
        {"web": -0.15, "app": -0.10, "partner": 0.10, "callcenter": 0.35}
    ).to_numpy(dtype=float)
    return (
        -1.05
        + 2.20 * prev_contact
        + 0.35 * risk
        + 0.20 * prev_apps
        + 0.10 * prev_util
        + 0.05 * (sessions_30d - 8.0)
        + channel_shift
    )


def _cx_contact_logit(x: np.ndarray) -> np.ndarray:
    risk = x[:, _CX_COL["risk_latent"]]
    sessions = x[:, _CX_COL["sessions_30d"]]
    prev_contact = x[:, _CX_COL["prev_contact"]]
    prev_apps = x[:, _CX_COL["prev_apps"]]
    prev_util = x[:, _CX_COL["prev_util"]]
    channel_callcenter = x[:, _CX_COL["channel_callcenter"]]
    channel_partner = x[:, _CX_COL["channel_partner"]]
    channel_web = x[:, _CX_COL["channel_web"]]

    # Baseline channel is "app" because the notebook uses pd.get_dummies(..., drop_first=True).
    return (
        -1.15
        + 2.20 * prev_contact
        + 0.35 * risk
        + 0.20 * prev_apps
        + 0.10 * prev_util
        + 0.05 * (sessions - 8.0)
        + 0.45 * channel_callcenter
        + 0.20 * channel_partner
        - 0.05 * channel_web
    )


def _cx_repeat_logit(x: np.ndarray) -> np.ndarray:
    age = x[:, _CX_COL["age"]]
    risk = x[:, _CX_COL["risk_latent"]]
    prev_repeat = x[:, _CX_COL["prev_repeat"]]
    prev_apps = x[:, _CX_COL["prev_apps"]]
    prev_util = x[:, _CX_COL["prev_util"]]
    channel_callcenter = x[:, _CX_COL["channel_callcenter"]]
    channel_partner = x[:, _CX_COL["channel_partner"]]
    channel_web = x[:, _CX_COL["channel_web"]]

    return (
        -1.20
        + 1.30 * prev_repeat
        + 0.45 * ((age - 37.0) / 10.0)
        + 0.20 * risk
        + 0.35 * prev_apps
        + 0.15 * prev_util
        - 0.15 * channel_callcenter
        + 0.10 * channel_partner
        - 0.30 * channel_web
    )


def _cx_baseline_logit(x: np.ndarray) -> np.ndarray:
    age = x[:, _CX_COL["age"]]
    risk = x[:, _CX_COL["risk_latent"]]
    income = x[:, _CX_COL["income"]]
    clicks = x[:, _CX_COL["clicks_7d"]]
    prev_contact = x[:, _CX_COL["prev_contact"]]
    prev_repeat = x[:, _CX_COL["prev_repeat"]]
    prev_apps = x[:, _CX_COL["prev_apps"]]
    prev_util = x[:, _CX_COL["prev_util"]]
    channel_callcenter = x[:, _CX_COL["channel_callcenter"]]
    channel_partner = x[:, _CX_COL["channel_partner"]]
    channel_web = x[:, _CX_COL["channel_web"]]
    region_b = x[:, _CX_COL["region_B"]]
    region_c = x[:, _CX_COL["region_C"]]
    region_d = x[:, _CX_COL["region_D"]]
    sessions = x[:, _CX_COL["sessions_30d"]]

    p_long = _sigmoid(np.log(_CX_P_LONG_T2D / (1.0 - _CX_P_LONG_T2D)) + 0.10 * (sessions - 8.0) / 5.0)

    return (
        -0.37
        + 0.30 * risk
        + 0.20 * prev_contact
        + 0.15 * prev_repeat
        + 0.18 * ((age - 37.0) / 10.0)
        + 0.03 * (np.log1p(income) - np.log1p(85_000.0))
        + 0.02 * (clicks - 12.0) / 5.0
        + 1.10 * prev_apps
        + 1.35 * prev_util
        - 0.16 * channel_callcenter
        - 0.13 * channel_partner
        - 0.03 * channel_web
        - 0.10 * region_b
        - 0.20 * region_c
        - 0.30 * region_d
        - 0.55 * p_long
    )


def generate_multitreatment_gamma_26(
    n: int = 100_000,
    seed: int = 42,
    include_oracle: bool = False,
    return_causal_data: bool = True,
) -> Union[pd.DataFrame, MultiCausalData]:
    r"""
    Pre-configured multi-treatment dataset with Gamma-distributed outcome.

    - 3 treatment classes: ``d_0`` (control), ``d_1``, ``d_2``
    - 8 confounders with realistic marginals sampled through a Gaussian copula
    - Gamma outcome with log-link confounding and heterogeneous arm effects

    Examples
    --------
    >>> df = generate_multitreatment_gamma_26(n=256, seed=7, return_causal_data=False)
    >>> bool(df[["d_0", "d_1", "d_2"]].sum(axis=1).eq(1).all())
    True
    >>> {"tenure_months", "credit_utilization", "y"}.issubset(df.columns)
    True

    Notes
    -----
    Let
    :math:`X = (\text{tenure}, \text{sessions}, \text{spend}, \text{premium}, \text{urban}, \text{tickets}, \text{discount}, \text{credit})`
    denote the 8 observed confounders. The treatment assignment mechanism is a
    multinomial logit with calibrated marginal arm rates near
    :math:`(0.50, 0.25, 0.25)`:

    .. math::

        s_k(X) = \alpha_{d,k} + \beta_{d,k}^{\top} X, \qquad
        \Pr(D = k \mid X) = \frac{\exp(s_k(X))}{\sum_{j=0}^{2} \exp(s_j(X))}.

    The confounders are jointly sampled through a Toeplitz copula with
    :math:`\mathrm{Corr}(X_i, X_j) = 0.3^{|i-j|}`.

    The outcome uses a log link. For arm :math:`k`,

    .. math::

        \log \mu_k(X) = \alpha_y + \beta_y^{\top} X + \theta_k + \tau_k(X),
        \qquad
        Y(k) \mid X \sim \Gamma(\text{shape}=2, \text{scale}=\mu_k(X)/2).

    This scenario fixes :math:`\theta = (0, -0.05, 0.10)` and uses the
    heterogeneous shifts

    .. math::

        \tau_1(X) =
        \min \left\{
        -0.22
        - 0.0010 \, \text{tenure}
        - 0.006 \, \text{sessions}
        - 0.05 \, \text{premium}
        - 0.04 \, \text{discount}
        - 0.10 \, (\text{credit} - 0.45),
        -0.02
        \right\},

    .. math::

        \tau_2(X) =
        \max \left\{
        0.16
        + 0.014 \, \text{sessions}
        + 0.030 \, \log(1 + \text{spend})
        + 0.06 \, \text{urban}
        - 0.006 \, \text{tickets}
        + 0.12 \, (\text{credit} - 0.45),
        0.02
        \right\}.

    So ``d_1`` is always weakly worse than control on the log-mean scale, while
    ``d_2`` is always weakly better than control.
    """
    confounder_specs = [
        {"name": "tenure_months",     "dist": "normal",   "mu": 24, "sd": 12, "clip_min": 0, "clip_max": 120},
        {"name": "avg_sessions_week", "dist": "normal",   "mu": 5,  "sd": 2,  "clip_min": 0, "clip_max": 40},
        {"name": "spend_last_month",  "dist": "lognormal","mu": np.log(60), "sigma": 0.9, "clip_max": 500},
        {"name": "premium_user",      "dist": "bernoulli","p": 0.25},
        {"name": "urban_resident",    "dist": "bernoulli","p": 0.60},
        {"name": "support_tickets_q", "dist": "poisson",  "lam": 1.5, "clip_max": 15},
        {"name": "discount_eligible", "dist": "bernoulli","p": 0.35},
        {"name": "credit_utilization","dist": "beta",     "mean": 0.45, "kappa": 20.0},
    ]

    beta_y = np.array([0.01, 0.08, 0.0015, 0.35, 0.12, 0.06, 0.20, 0.50], dtype=float)

    beta_d = np.array([
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [0.01, 0.10, 0.0015, 0.50, 0.20, 0.05, 0.35, 0.40],
        # Align d_2 selection with positive-effect drivers so oracle ATTE exceeds ATE.
        [-0.008, 0.13, 0.0020, 0.10, 0.30, -0.08, 0.00, 0.55],
    ], dtype=float)

    theta = [0.0, -0.05, 0.10]

    def tau_d1(x: np.ndarray) -> np.ndarray:
        # Columns: 0=tenure, 1=sessions, 3=premium, 6=discount, 7=credit_utilization.
        tenure = np.clip(x[:, 0], 0.0, 120.0)
        sessions = np.clip(x[:, 1], 0.0, 40.0)
        premium = x[:, 3]
        discount = x[:, 6]
        credit = np.clip(x[:, 7], 0.0, 1.0)
        raw = -0.22 - 0.0010 * tenure - 0.006 * sessions - 0.05 * premium - 0.04 * discount - 0.10 * (credit - 0.45)
        # Enforce d_1 < d_0 on the link scale for all rows.
        return np.minimum(raw, -0.02)

    def tau_d2(x: np.ndarray) -> np.ndarray:
        # Columns: 1=sessions, 2=spend_last_month, 4=urban, 5=tickets, 7=credit_utilization.
        sessions = np.clip(x[:, 1], 0.0, 40.0)
        spend = np.log1p(np.clip(x[:, 2], 0.0, 500.0))
        urban = x[:, 4]
        tickets = np.clip(x[:, 5], 0.0, 15.0)
        credit = np.clip(x[:, 7], 0.0, 1.0)
        raw = 0.16 + 0.014 * sessions + 0.030 * spend + 0.06 * urban - 0.006 * tickets + 0.12 * (credit - 0.45)
        # Enforce d_2 > d_0 on the link scale for all rows.
        return np.maximum(raw, 0.02)

    tau = [None, tau_d1, tau_d2]

    return _run_multitreatment_26(
        n=n,
        outcome_type="gamma",
        confounder_specs=confounder_specs,
        beta_y=beta_y,
        beta_d=beta_d,
        theta=theta,
        tau=tau,
        alpha_y=0.0,
        gamma_shape=2.0,
        seed=seed,
        include_oracle=include_oracle,
        return_causal_data=return_causal_data,
    )


def generate_multitreatment_binary_26(
    n: int = 100_000,
    seed: int = 42,
    include_oracle: bool = False,
    return_causal_data: bool = True,
) -> Union[pd.DataFrame, MultiCausalData]:
    r"""
    Pre-configured multi-treatment dataset with Binary outcome.

    - 3 treatment classes: ``d_0`` (control), ``d_1``, ``d_2``
    - 8 confounders with realistic marginals sampled through a Gaussian copula
    - Binary outcome with a logistic baseline and heterogeneous arm effects

    Examples
    --------
    >>> df = generate_multitreatment_binary_26(n=256, seed=7, return_causal_data=False)
    >>> bool(df[["d_0", "d_1", "d_2"]].sum(axis=1).eq(1).all())
    True
    >>> {"weekly_active_days", "engagement_score", "y"}.issubset(df.columns)
    True

    Notes
    -----
    Let
    :math:`X = (\text{tenure}, \text{active days}, \text{income}, \text{premium}, \text{family}, \text{complaints}, \text{discount}, \text{engagement})`
    denote the 8 confounders. Treatment assignment again follows a calibrated
    multinomial logit with target arm rates near :math:`(0.50, 0.25, 0.25)`:

    .. math::

        s_k(X) = \alpha_{d,k} + \beta_{d,k}^{\top} X, \qquad
        \Pr(D = k \mid X) = \frac{\exp(s_k(X))}{\sum_{j=0}^{2} \exp(s_j(X))}.

    The outcome uses a logistic link with ``alpha_y = -1.1``:

    .. math::

        \operatorname{logit}\Pr(Y(k)=1 \mid X)
        = -1.1 + \beta_y^{\top} X + \theta_k + \tau_k(X).

    This scenario fixes :math:`\theta = (0, -0.18, 0.26)` and uses

    .. math::

        \tau_1(X) =
        \min \left\{
        -0.16
        - 0.0008 \, \text{tenure}
        - 0.020 \, \text{active days}
        - 0.08 \, \text{premium}
        - 0.03 \, \text{complaints}
        - 0.10 \, (\text{engagement} - 0.60),
        -0.02
        \right\},

    .. math::

        \tau_2(X) =
        \max \left\{
        0.14
        + 0.020 \, \text{active days}
        + 0.028 \, \log(1 + \text{income})
        + 0.05 \, \text{family}
        - 0.010 \, \text{complaints}
        + 0.12 \, (\text{engagement} - 0.60),
        0.02
        \right\}.

    The clipping keeps ``d_1`` uniformly below control and ``d_2`` uniformly
    above control on the log-odds scale, while the Gaussian copula with
    :math:`\mathrm{Corr}(X_i, X_j) = 0.3^{|i-j|}` induces cross-feature
    dependence.
    """
    confounder_specs = [
        {"name": "tenure_months",      "dist": "normal",   "mu": 24, "sd": 12, "clip_min": 0, "clip_max": 120},
        {"name": "weekly_active_days", "dist": "normal",   "mu": 4.0, "sd": 1.5, "clip_min": 0, "clip_max": 7},
        {"name": "annual_income_k",    "dist": "gamma",    "shape": 4.0, "scale": 18.0, "clip_max": 300},
        {"name": "premium_user",       "dist": "bernoulli","p": 0.22},
        {"name": "family_plan",        "dist": "bernoulli","p": 0.38},
        {"name": "recent_complaints",  "dist": "poisson",  "lam": 0.8, "clip_max": 10},
        {"name": "discount_eligible",  "dist": "bernoulli","p": 0.30},
        {"name": "engagement_score",   "dist": "beta",     "mean": 0.60, "kappa": 16.0},
    ]

    beta_y = np.array([0.003, 0.11, 0.004, 0.40, -0.25, -0.12, 0.20, 0.90], dtype=float)

    beta_d = np.array([
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [0.01, 0.09, 0.0018, 0.45, 0.20, 0.08, 0.30, 0.28],
        [-0.004, 0.07, 0.0012, 0.30, 0.12, 0.10, 0.18, 0.22],
    ], dtype=float)

    theta = [0.0, -0.18, 0.26]

    def tau_d1(x: np.ndarray) -> np.ndarray:
        # Columns: 0=tenure, 1=weekly_active_days, 3=premium, 5=complaints, 7=engagement.
        tenure = np.clip(x[:, 0], 0.0, 120.0)
        active_days = np.clip(x[:, 1], 0.0, 7.0)
        premium = x[:, 3]
        complaints = np.clip(x[:, 5], 0.0, 10.0)
        engagement = np.clip(x[:, 7], 0.0, 1.0)
        raw = -0.16 - 0.0008 * tenure - 0.020 * active_days - 0.08 * premium - 0.03 * complaints - 0.10 * (engagement - 0.60)
        # Enforce d_1 < d_0 on the link scale for all rows.
        return np.minimum(raw, -0.02)

    def tau_d2(x: np.ndarray) -> np.ndarray:
        # Columns: 1=weekly_active_days, 2=annual_income_k, 4=family_plan, 5=complaints, 7=engagement.
        active_days = np.clip(x[:, 1], 0.0, 7.0)
        income = np.log1p(np.clip(x[:, 2], 0.0, 300.0))
        family_plan = x[:, 4]
        complaints = np.clip(x[:, 5], 0.0, 10.0)
        engagement = np.clip(x[:, 7], 0.0, 1.0)
        raw = 0.14 + 0.020 * active_days + 0.028 * income + 0.05 * family_plan - 0.010 * complaints + 0.12 * (engagement - 0.60)
        # Enforce d_2 > d_0 on the link scale for all rows.
        return np.maximum(raw, 0.02)

    tau = [None, tau_d1, tau_d2]

    return _run_multitreatment_26(
        n=n,
        outcome_type="binary",
        confounder_specs=confounder_specs,
        beta_y=beta_y,
        beta_d=beta_d,
        theta=theta,
        tau=tau,
        # Keep baseline probability away from 0/1 saturation before treatment shifts.
        alpha_y=-1.1,
        seed=seed,
        include_oracle=include_oracle,
        return_causal_data=return_causal_data,
    )


def generate_multitreatment_irm_26(
    n: int = 100_000,
    seed: int = 42,
    include_oracle: bool = False,
    return_causal_data: bool = True,
) -> Union[pd.DataFrame, MultiCausalData]:
    # Backward-compatible alias.
    return generate_multitreatment_gamma_26(
        n=n,
        seed=seed,
        include_oracle=include_oracle,
        return_causal_data=return_causal_data,
    )


def generate_multi_dml_cx_26(
    n: int = 100_000,
    seed: int = 42,
    include_oracle: bool = False,
    return_causal_data: bool = True,
) -> Union[pd.DataFrame, MultiCausalData]:
    r"""
    The notebook simulates overlapping ``contact`` and ``repeat`` actions. This
    packaged DGP resolves them into a mutually exclusive one-hot treatment:

    - ``control``
    - ``neg_contact_flg``
    - ``error_flg``
    - ``neg_contact_flg_error_flg``

    Treatment assignment matches the notebook's independent Bernoulli contact
    and repeat mechanisms exactly after overlap-resolution, but is exposed
    through the shared multi-treatment generator so it integrates with
    ``MultiCausalData`` and the scenario tooling.

    Examples
    --------
    >>> df = generate_multi_dml_cx_26(n=256, seed=7, return_causal_data=False)
    >>> treatment_cols = ["control", "neg_contact_flg", "error_flg", "neg_contact_flg_error_flg"]
    >>> bool(df[treatment_cols].sum(axis=1).eq(1).all())
    True
    >>> {"age", "prev_apps", "csat_prev", "y"}.issubset(df.columns)
    True

    Notes
    -----
    Write :math:`a(X)` for the contact logit and :math:`b(X)` for the repeat
    logit. The notebook first draws two conditionally independent Bernoulli
    actions,

    .. math::

        C \mid X \sim \operatorname{Bernoulli}(\sigma(a(X))), \qquad
        R \mid X \sim \operatorname{Bernoulli}(\sigma(b(X))),

    where :math:`\sigma(z) = 1 / (1 + e^{-z})`. In this packaged benchmark the
    pair :math:`(C, R)` is re-encoded as a one-hot treatment:

    .. math::

        D =
        \begin{cases}
        \text{control} & (C, R) = (0, 0), \\
        \text{neg\_contact\_flg} & (C, R) = (1, 0), \\
        \text{error\_flg} & (C, R) = (0, 1), \\
        \text{neg\_contact\_flg\_error\_flg} & (C, R) = (1, 1).
        \end{cases}

    Let :math:`p_c = \sigma(a(X))` and :math:`p_r = \sigma(b(X))`. Then the arm
    probabilities are

    .. math::

        \Pr(D=\text{control}\mid X) = (1-p_c)(1-p_r),

    .. math::

        \Pr(D=\text{neg\_contact\_flg}\mid X) = p_c (1-p_r),

    .. math::

        \Pr(D=\text{error\_flg}\mid X) = (1-p_c) p_r,

    .. math::

        \Pr(D=\text{neg\_contact\_flg\_error\_flg}\mid X) = p_c p_r.

    Equivalently, this is exactly the softmax model with class scores
    :math:`(0, a(X), b(X), a(X)+b(X))`, which is why the implementation passes
    ``g_d=[None, _cx_contact_logit, _cx_repeat_logit, lambda x: _cx_contact_logit(x) + _cx_repeat_logit(x)]``.

    The observed outcome uses a binary logit baseline :math:`g_y(X)` plus a
    class effect

    .. math::

        \operatorname{logit}\Pr(Y=1 \mid X, D)
        = g_y(X) + \theta(D),

    with :math:`\theta(\text{control}) = \theta(\text{neg\_contact\_flg}) = 0`
    and :math:`\theta(\text{error\_flg}) = \theta(\text{neg\_contact\_flg\_error\_flg}) = -0.65`.

    Worked overlap example: if :math:`a(X)=0.8` and :math:`b(X)=-0.2`, then
    :math:`p_c \approx 0.690` and :math:`p_r \approx 0.450`, giving arm
    probabilities approximately ``(0.170, 0.379, 0.140, 0.311)`` for
    ``(control, neg_contact_flg, error_flg, neg_contact_flg_error_flg)``.
    """
    return generate_multitreatment(
        n=n,
        n_treatments=4,
        outcome_type="binary",
        alpha_y=0.0,
        theta=[0.0, 0.0, -0.65, -0.65],
        tau=[None, None, None, None],
        confounder_specs=_CX_CONFOUNDER_SPECS,
        x_sampler=_sample_multi_dml_cx_26_x,
        g_y=_cx_baseline_logit,
        g_d=[
            None,
            _cx_contact_logit,
            _cx_repeat_logit,
            lambda x: _cx_contact_logit(x) + _cx_repeat_logit(x),
        ],
        random_state=seed,
        include_oracle=include_oracle,
        return_causal_data=return_causal_data,
        d_names=_CX_TREATMENT_NAMES,
    )


multi_dml_cx_26 = generate_multi_dml_cx_26
