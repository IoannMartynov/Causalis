"""
High-level helpers for single-treatment synthetic causal datasets.

This module provides convenient wrappers around
:class:`causalis.dgp.causaldata.base.CausalDatasetGenerator` for common
benchmarking setups such as classic A/B tests, observational confounding, and
CUPED-oriented examples with pre-period covariates.

Notes
-----
Use this module when you want a ready-made synthetic dataset without specifying
every structural component manually. For finer control over the DGP, instantiate
``CausalDatasetGenerator`` directly instead.

Examples
--------
>>> from causalis.dgp.causaldata.functional import generate_rct, obs_linear_effect
>>> rct = generate_rct(n=500, outcome_type="normal", return_causal_data=True)
>>> rct.outcome, rct.treatment
('y', 'd')
>>> obs = obs_linear_effect(n=500, theta=1.0, target_d_rate=0.3)
>>> {"y", "d"}.issubset(obs.columns)
True
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Optional, Union, List, Callable, Any

from .base import CausalDatasetGenerator
from causalis.dgp.base import _sigmoid, _logit, _add_ancillary_info
from .preperiod import PreCorrSpec, calibrate_sigma_for_target_corr, add_preperiod_covariate
from causalis.dgp.causaldata import CausalData

class _PrognosticScore:
    """
    Deterministic (given seed) prognostic signal s(X) used in outcome baseline.

    Works with any X dimension (incl. one-hot expansions) because coef is initialized on first call.

    Parameters
    ----------
    rng : numpy.random.Generator
        The random number generator to use.
    scale : float, default=1.0
        Scale of the generated coefficients.
    """
    def __init__(self, rng: np.random.Generator, scale: float = 1.0):
        self.rng = rng
        self.scale = float(scale)
        self.coef_: Optional[np.ndarray] = None
        self.mean_: Optional[float] = None  # for centering in-sample

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the prognostic score s(X).

        Parameters
        ----------
        X : numpy.ndarray
            Confounder matrix.

        Returns
        -------
        numpy.ndarray
            Calculated prognostic score.
        """
        X = np.asarray(X, dtype=float)
        if X.shape[1] == 0:
            return np.zeros(X.shape[0], dtype=float)

        if self.coef_ is None:
            # Scale ~ 1/sqrt(k) to keep variance stable with dimension
            k = X.shape[1]
            self.coef_ = self.rng.normal(size=k) * (self.scale / np.sqrt(max(1, k)))

        s = X @ self.coef_
        # Center in-sample so alpha_y continues to mean something for continuous outcomes
        if self.mean_ is None:
            self.mean_ = float(s.mean())
        return s - self.mean_

    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Alias for __call__.

        Parameters
        ----------
        X : numpy.ndarray
            Confounder matrix.

        Returns
        -------
        numpy.ndarray
            Calculated prognostic score.
        """
        # Same as __call__ but safe to reuse after generator ran
        return self.__call__(X)


def generate_rct(
    n: int = 20_000,
    split: float = 0.5,
    random_state: Optional[int] = 42,
    outcome_type: str = "binary",              # {"binary","normal","poisson","gamma"}; "nonnormal" -> "poisson"
    outcome_params: Optional[Dict] = None,
    confounder_specs: Optional[List[Dict[str, Any]]] = None,
    k: int = 0,
    x_sampler: Optional[Callable[[int, int, int], np.ndarray]] = None,
    add_ancillary: bool = True,
    deterministic_ids: bool = False,
    # --- CUPED knobs ---
    add_pre: bool = True,
    pre_name: str = "y_pre",
    pre_corr: float = 0.7,           # target Corr(y_pre, y|D=0) for continuous-ish settings
    prognostic_scale: float = 1.0,   # strength of s(X) in outcome baseline
    # --- NEW: baseline outcome signal knobs (X -> Y) ---
    beta_y: Optional[Union[List[float], np.ndarray]] = None,
    g_y: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    use_prognostic: Optional[bool] = None,  # if None: defaults to add_pre (backward compatible)
    include_oracle: bool = True,
    return_causal_data: bool = False,
) -> Union[pd.DataFrame, CausalData]:
    """
    Generate an RCT dataset with randomized treatment assignment.

    Uses `CausalDatasetGenerator` internally, ensuring treatment is independent of X.
    Specifically designed for benchmarking variance reduction techniques like CUPED.

    **Notes on effect scale**

    How `outcome_params` maps into the structural effect:
      - outcome_type="normal": treatment shifts the mean by (mean["B"] - mean["A"]) on the outcome scale.
      - outcome_type="binary": treatment shifts the log-odds by (logit(p_B) - logit(p_A)).
      - outcome_type="poisson" or "gamma": treatment shifts the log-mean by log(lam_B / lam_A).

    Ancillary columns (if add_ancillary=True) are generated from baseline confounders X only,
    avoiding outcome leakage and post-treatment adjustment issues.

    Parameters
    ----------
    n : int, default=20_000
        Number of samples to generate.
    split : float, default=0.5
        Proportion of samples assigned to the treatment group.
    random_state : int, optional
        Random seed for reproducibility.
    outcome_type : {"binary", "normal", "poisson", "gamma"}, default="binary"
        Distribution family of the outcome.
    outcome_params : dict, optional
        Parameters defining baseline rates/means and treatment effects.
        e.g., {"p": {"A": 0.1, "B": 0.12}} for binary, or
        {"shape": 2.0, "scale": {"A": 1.0, "B": 1.1}} for poisson/gamma.
    confounder_specs : list of dict, optional
        Schema for confounder distributions.
    k : int, default=0
        Number of confounders if specs not provided.
    x_sampler : callable, optional
        Custom sampler for confounders.
    add_ancillary : bool, default=True
        Whether to add descriptive columns like 'age', 'platform', etc.
    deterministic_ids : bool, default=False
        Whether to generate deterministic user IDs.
    add_pre : bool, default=True
        Whether to generate a pre-period covariate (`y_pre`).
    pre_name : str, default="y_pre"
        Name of the pre-period covariate column.
    pre_corr : float, default=0.7
        Target correlation between `y_pre` and the outcome Y in the control group.
    prognostic_scale : float, default=1.0
        Scale of the prognostic signal derived from confounders.
    include_oracle : bool, default=True
        Whether to include oracle ground-truth columns like 'cate', 'm', etc.
    return_causal_data : bool, default=False
        Whether to return a `CausalData` object instead of a `pandas.DataFrame`.

    Returns
    -------
    pandas.DataFrame or CausalData
        Synthetic RCT dataset.

    Examples
    --------
    >>> from causalis.dgp.causaldata.functional import generate_rct
    >>> data = generate_rct(
    ...     n=1000,
    ...     outcome_type="binary",
    ...     outcome_params={"p": {"A": 0.10, "B": 0.12}},
    ...     add_pre=True,
    ...     return_causal_data=True,
    ... )
    >>> data.treatment, data.outcome
    ('d', 'y')
    >>> "y_pre" in data.df.columns
    True
    >>> {"g0", "g1", "cate"}.issubset(data.df.columns)
    True
    """
    # RNG for ancillary generation
    rng = np.random.default_rng(random_state)

    # Backward-compatible default: previously, X affected Y only when add_pre=True via prognostic
    if use_prognostic is None:
        use_prognostic = add_pre

    # Ensure y_pre is informative: if you ask for y_pre, force some baseline signal in Y
    if add_pre:
        use_prognostic = True

    # Normalize beta_y
    beta_y_arr: Optional[np.ndarray]
    if beta_y is None:
        beta_y_arr = None
    else:
        beta_y_arr = np.asarray(beta_y, dtype=float).reshape(-1)

    # Decide which nonlinear baseline g_y to use
    # Priority:
    #   1) user-provided g_y
    #   2) PrognosticScore if requested (use_prognostic)
    #   3) None
    prognostic = None
    if g_y is not None:
        g_y_used = g_y
    elif use_prognostic:
        prognostic = _PrognosticScore(rng, scale=prognostic_scale)
        g_y_used = prognostic
    else:
        g_y_used = None

    # Validate split
    split_f = float(split)
    if not (0.0 < split_f < 1.0):
        raise ValueError("split must be in (0,1).")

    # If you want CUPED to work, you must have some baseline X to carry signal.
    if add_pre and (confounder_specs is None) and (x_sampler is None) and int(k) == 0:
        k = 2

    # Normalize outcome_type
    ttype = outcome_type.lower()
    if ttype == "nonnormal":
        ttype = "poisson"
    if ttype not in {"binary", "normal", "poisson", "gamma"}:
        raise ValueError("outcome_type must be 'binary', 'normal', 'poisson', or 'gamma' (or alias 'nonnormal').")

    # Default outcome_params
    if outcome_params is None:
        if ttype == "binary":
            outcome_params = {"p": {"A": 0.10, "B": 0.12}}
        elif ttype == "normal":
            outcome_params = {"mean": {"A": 0.00, "B": 0.20}, "std": 1.0}
        elif ttype == "gamma":
            outcome_params = {"shape": 2.0, "scale": {"A": 1.0, "B": 1.1}}
        else:
            outcome_params = {"shape": 2.0, "scale": {"A": 1.0, "B": 1.1}}

    # Map to natural-scale means
    # Keep a single natural-scale representation first, then convert once to
    # CausalDatasetGenerator link-scale parameters below.
    if ttype == "binary":
        pA = float(outcome_params["p"]["A"]); pB = float(outcome_params["p"]["B"])
        if not (0.0 < pA < 1.0 and 0.0 < pB < 1.0):
            raise ValueError("For binary outcomes, probabilities must be in (0,1).")
        mu0_nat, mu1_nat = pA, pB
    elif ttype == "normal":
        muA = float(outcome_params["mean"]["A"]); muB = float(outcome_params["mean"]["B"])
        sd  = float(outcome_params.get("std", 1.0))
        if not (sd > 0):
            raise ValueError("For normal outcomes, std must be > 0.")
        mu0_nat, mu1_nat = muA, muB
    elif ttype in {"poisson", "gamma"}:
        shape = float(outcome_params.get("shape", 2.0))
        scaleA = float(outcome_params["scale"]["A"])
        scaleB = float(outcome_params["scale"]["B"])
        if ttype == "gamma" and not (shape > 0):
            raise ValueError("For gamma outcomes, shape must be > 0.")
        if not (scaleA > 0 and scaleB > 0):
            raise ValueError("For Poisson/Gamma outcomes, scales must be > 0.")
        mu0_nat = shape * scaleA
        mu1_nat = shape * scaleB
        if not (mu0_nat > 0 and mu1_nat > 0):
            raise ValueError("For Poisson/Gamma outcomes, implied means must be > 0.")

    # Convert to class parameters
    # CausalDatasetGenerator expects:
    # - alpha_y/theta on outcome link scale
    # - outcome_type="continuous" for normal outcomes.
    if ttype == "binary":
        alpha_y = _logit(mu0_nat)
        theta = _logit(mu1_nat) - _logit(mu0_nat)
        outcome_type_cls = "binary"
        sigma_y = 1.0
    elif ttype == "normal":
        alpha_y = mu0_nat
        theta = mu1_nat - mu0_nat
        outcome_type_cls = "continuous"
        sigma_y = sd
    elif ttype == "gamma":
        alpha_y = float(np.log(mu0_nat))
        theta = float(np.log(mu1_nat / mu0_nat))
        outcome_type_cls = "gamma"
        sigma_y = 1.0
    else:  # poisson
        alpha_y = float(np.log(mu0_nat))
        theta = float(np.log(mu1_nat / mu0_nat))
        outcome_type_cls = "poisson"
        sigma_y = 1.0

    # Instantiate the unified generator with randomized treatment
    # RCT property is encoded by beta_d=None, g_d=None, u_strength_d=0 and fixed alpha_d=logit(split).
    gen_kwargs = dict(
        theta=theta,
        tau=None,
        beta_y=beta_y_arr,
        beta_d=None,
        g_y=g_y_used,
        g_d=None,
        alpha_y=alpha_y,
        alpha_d=_logit(split_f),
        sigma_y=sigma_y,
        outcome_type=outcome_type_cls,
        confounder_specs=confounder_specs,
        k=int(k),
        x_sampler=x_sampler,
        use_copula=False,
        target_d_rate=None,
        u_strength_d=0.0,
        u_strength_y=0.0,
        include_oracle=include_oracle,
        seed=random_state,
    )
    if outcome_type_cls == "gamma":
        gen_kwargs["gamma_shape"] = float(outcome_params.get("shape", 2.0))
    gen = CausalDatasetGenerator(**gen_kwargs)

    df = gen.generate(n)

    if add_pre:
        # Build a baseline signal from exactly the same outcome-baseline components
        # used by the generator (beta_y + g_y), excluding treatment effects.
        exclude = {"y","d","m","m_obs","tau_link","g0","g1","cate"}
        x_cols = [c for c in df.columns if c not in exclude]
        
        def base_builder(df_in: pd.DataFrame) -> np.ndarray:
            X = df_in[x_cols].to_numpy(dtype=float)
            s_base = np.zeros(len(df_in), dtype=float)
            if beta_y_arr is not None:
                s_base += X @ beta_y_arr
            if g_y_used is not None:
                s_base += np.asarray(g_y_used(X), dtype=float).reshape(-1)
            # Center for stability
            return s_base - float(s_base.mean())

        spec = PreCorrSpec(
            target_corr=pre_corr,
            # For skewed/nonnegative outcomes, calibrate correlation in transformed space.
            transform="none" if outcome_type_cls == "continuous" else "log1p"
        )
        
        df = add_preperiod_covariate(
            df=df,
            y_col="y",
            d_col="d",
            pre_name=pre_name,
            base_builder=base_builder,
            spec=spec,
            rng=rng
        )

    if add_ancillary:
        exclude = {"y", "d", "m", "m_obs", "tau_link", "g0", "g1", "cate"}
        if pre_name in df.columns:
            exclude.add(pre_name)
        x_cols = [c for c in df.columns if c not in exclude]
        df = _add_ancillary_info(df, n, rng, deterministic_ids, x_cols)

    # Reorder columns: user_id, y, d, confounders, y_pre, oracle
    all_cols = list(df.columns)
    oracle_cols = [
        "m", "m_obs", "tau_link", "g0", "g1", "cate"
    ]
    core_cols = ["user_id", "y", "d"]

    actual_core = [c for c in core_cols if c in all_cols]
    actual_oracle = [c for c in oracle_cols if c in all_cols]
    actual_pre = [pre_name] if pre_name in all_cols else []

    # Confounders = everything else
    excluded_from_conf = set(actual_core + actual_oracle + actual_pre)
    actual_conf = [c for c in all_cols if c not in excluded_from_conf]

    df = df[actual_core + actual_conf + actual_pre + actual_oracle]

    if return_causal_data:
        # Determine confounders: numeric columns except known non-confounders
        exclude = {"y", "d", "m", "m_obs", "tau_link", "g0", "g1", "cate", "user_id"}
        confounder_cols = [
            c for c in df.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
        ]
        return CausalData(
            df=df,
            treatment="d",
            outcome="y",
            confounders=confounder_cols,
            user_id="user_id" if "user_id" in df.columns else None
        )

    return df


def generate_classic_rct(
    n: int = 10_000,
    split: float = 0.5,
    random_state: Optional[int] = 42,
    outcome_params: Optional[Dict] = None,
    add_pre: bool = False,
    beta_y: Optional[Union[List[float], np.ndarray]] = None,
    outcome_depends_on_x: bool = True,
    prognostic_scale: float = 1.0,
    pre_corr: float = 0.7,
    return_causal_data: bool = False,
    add_ancillary: bool = False,
    deterministic_ids: bool = False,
    include_oracle: bool = True,
    **kwargs
) -> Union[pd.DataFrame, CausalData]:
    """
    Generate a classic RCT dataset with three binary confounders:
    platform_ios, country_usa, and source_paid.

    Parameters
    ----------
    n : int, default=10_000
        Number of samples to generate.
    split : float, default=0.5
        Proportion of samples assigned to the treatment group.
    random_state : int, optional
        Random seed for reproducibility.
    outcome_params : dict, optional
        Parameters defining baseline rates/means and treatment effects.
        e.g., {"p": {"A": 0.1, "B": 0.15}} for binary.
    add_pre : bool, default=False
        Whether to generate a pre-period covariate (`y_pre`).
    beta_y : array-like, optional
        Linear coefficients for confounders in the outcome model.
    outcome_depends_on_x : bool, default=True
        Whether to add default effects for confounders if beta_y is None.
    prognostic_scale : float, default=1.0
        Scale of nonlinear prognostic signal (passed to generate_rct).
    pre_corr : float, default=0.7
        Target correlation for y_pre (passed to generate_rct).
    return_causal_data : bool, default=False
        Whether to return a `CausalData` object instead of a `pandas.DataFrame`.
    add_ancillary : bool, default=False
        Whether to add standard ancillary columns (age, platform, etc.).
    deterministic_ids : bool, default=False
        Whether to generate deterministic user IDs.
    include_oracle : bool, default=True
        Whether to include oracle ground-truth columns like 'cate', 'propensity', etc.
    **kwargs :
        Additional arguments passed to `generate_rct`.

    Returns
    -------
    pandas.DataFrame or CausalData
        Synthetic classic RCT dataset.
    """
    confounder_specs = [
        {"name": "platform_ios", "dist": "bernoulli", "p": 0.5},
        {"name": "country_usa",  "dist": "bernoulli", "p": 0.6},
        {"name": "source_paid",  "dist": "bernoulli", "p": 0.3},
    ]

    if outcome_depends_on_x and beta_y is None:
        # Default effects for [platform_ios, country_usa, source_paid]
        # Interpretation depends on outcome_type:
        #   - binary: log-odds shifts
        #   - normal: mean shifts
        #   - poisson/gamma: log-mean shifts
        beta_y = [0.6, 0.4, 0.8]

    return generate_rct(
        n=n,
        split=split,
        random_state=random_state,
        outcome_params=outcome_params,
        confounder_specs=confounder_specs,
        add_ancillary=add_ancillary,
        deterministic_ids=deterministic_ids,
        add_pre=add_pre,
        beta_y=beta_y,
        pre_corr=pre_corr,
        prognostic_scale=prognostic_scale,
        include_oracle=include_oracle,
        return_causal_data=return_causal_data,
        **kwargs
    )


def classic_rct_gamma(
    n: int = 10_000,
    split: float = 0.5,
    random_state: Optional[int] = 42,
    outcome_params: Optional[Dict] = None,
    add_pre: bool = False,
    beta_y: Optional[Union[List[float], np.ndarray]] = None,
    outcome_depends_on_x: bool = True,
    prognostic_scale: float = 1.0,
    pre_corr: float = 0.7,
    add_ancillary: bool = True,
    deterministic_ids: bool = False,
    include_oracle: bool = True,
    return_causal_data: bool = False,
    **kwargs
) -> Union[pd.DataFrame, CausalData]:
    """
    Generate a classic RCT dataset with three binary confounders and a gamma outcome.

    The gamma outcome uses a log-mean link, so treatment effects are multiplicative
    on the mean scale. The default parameters are chosen to resemble a skewed
    real-world metric (e.g., spend or revenue).

    Parameters
    ----------
    n : int, default=10_000
        Number of samples to generate.
    split : float, default=0.5
        Proportion of samples assigned to the treatment group.
    random_state : int, optional
        Random seed for reproducibility.
    outcome_params : dict, optional
        Gamma parameters, e.g. {"shape": 2.0, "scale": {"A": 15.0, "B": 16.5}}.
        Mean = shape * scale.
    add_pre : bool, default=False
        Whether to generate a pre-period covariate (`y_pre`).
    beta_y : array-like, optional
        Linear coefficients for confounders in the log-mean outcome model.
    outcome_depends_on_x : bool, default=True
        Whether to add default effects for confounders if beta_y is None.
    prognostic_scale : float, default=1.0
        Scale of nonlinear prognostic signal.
    pre_corr : float, default=0.7
        Target correlation for y_pre with post-outcome in control group.
    add_ancillary : bool, default=True
        Whether to add standard ancillary columns (age, platform, etc.).
    deterministic_ids : bool, default=False
        Whether to generate deterministic user IDs.
    include_oracle : bool, default=True
        Whether to include oracle ground-truth columns like 'cate', 'propensity', etc.
    return_causal_data : bool, default=False
        Whether to return a `CausalData` object instead of a `pandas.DataFrame`.
    **kwargs :
        Additional arguments passed to `generate_rct` (e.g., pre_name, g_y, use_prognostic).

    Returns
    -------
    pandas.DataFrame or CausalData
        Synthetic classic RCT dataset with gamma outcome.
    """
    confounder_specs = [
        {"name": "platform_ios", "dist": "bernoulli", "p": 0.5},
        {"name": "country_usa",  "dist": "bernoulli", "p": 0.6},
        {"name": "source_paid",  "dist": "bernoulli", "p": 0.3},
    ]

    if outcome_depends_on_x and beta_y is None:
        # Default log-mean shifts for [platform_ios, country_usa, source_paid]
        beta_y = [0.25, 0.20, 0.45]

    if outcome_params is None:
        outcome_params = {"shape": 2.0, "scale": {"A": 15.0, "B": 16.5}}

    return generate_rct(
        n=n,
        split=split,
        random_state=random_state,
        outcome_type="gamma",
        outcome_params=outcome_params,
        confounder_specs=confounder_specs,
        add_ancillary=add_ancillary,
        deterministic_ids=deterministic_ids,
        add_pre=add_pre,
        beta_y=beta_y,
        pre_corr=pre_corr,
        prognostic_scale=prognostic_scale,
        include_oracle=include_oracle,
        return_causal_data=return_causal_data,
        **kwargs
    )



def obs_linear_effect(
    n: int = 10_000,
    theta: float = 1.0,
    outcome_type: str = "continuous",
    sigma_y: float = 1.0,
    target_d_rate: Optional[float] = None,
    confounder_specs: Optional[List[Dict[str, Any]]] = None,
    beta_y: Optional[np.ndarray] = None,
    beta_d: Optional[np.ndarray] = None,
    random_state: Optional[int] = 42,
    k: int = 0,
    x_sampler: Optional[Callable[[int, int, int], np.ndarray]] = None,
    include_oracle: bool = True,
    add_ancillary: bool = False,
    deterministic_ids: bool = False,
) -> pd.DataFrame:
    """
    Generate an observational dataset with linear effects of confounders and a constant treatment effect.

    Parameters
    ----------
    n : int, default=10_000
        Number of samples to generate.
    theta : float, default=1.0
        Constant treatment effect.
    outcome_type : {"continuous", "binary", "poisson", "gamma"}, default="continuous"
        Family of the outcome distribution.
    sigma_y : float, default=1.0
        Noise level for continuous outcomes.
    target_d_rate : float, optional
        Target treatment prevalence (propensity mean).
    confounder_specs : list of dict, optional
        Schema for confounder distributions.
    beta_y : array-like, optional
        Linear coefficients for confounders in the outcome model.
    beta_d : array-like, optional
        Linear coefficients for confounders in the treatment model.
    random_state : int, optional
        Random seed for reproducibility.
    k : int, default=0
        Number of confounders if specs not provided.
    x_sampler : callable, optional
        Custom sampler for confounders.
    include_oracle : bool, default=True
        Whether to include oracle ground-truth columns like 'cate', 'm', etc.
    add_ancillary : bool, default=False
        If True, adds standard ancillary columns (age, platform, etc.).
    deterministic_ids : bool, default=False
        If True, generates deterministic user IDs.

    Returns
    -------
    pandas.DataFrame
        Synthetic observational dataset.

    Notes
    -----
    This helper is a lightweight observational benchmark:

    - treatment is not randomized unless ``beta_d`` is zero and
      ``target_d_rate`` forces a near-constant propensity;
    - oracle columns such as ``m`` and ``cate`` are available when
      ``include_oracle=True``;
    - the treatment effect is constant on the structural link scale, so
      heterogeneity only enters through the outcome family transformation.

    Examples
    --------
    >>> from causalis.dgp.causaldata.functional import obs_linear_effect
    >>> df = obs_linear_effect(
    ...     n=1000,
    ...     theta=1.0,
    ...     target_d_rate=0.35,
    ...     k=3,
    ...     random_state=3141,
    ... )
    >>> sorted(col for col in ["y", "d", "m", "cate"] if col in df.columns)
    ['cate', 'd', 'm', 'y']
    """
    gen = CausalDatasetGenerator(
        theta=theta,
        outcome_type=outcome_type,
        sigma_y=sigma_y,
        target_d_rate=target_d_rate,
        seed=random_state,
        confounder_specs=confounder_specs,
        beta_y=beta_y,
        beta_d=beta_d,
        k=int(k),
        x_sampler=x_sampler,
        include_oracle=include_oracle,
    )
    df = gen.generate(n)

    if add_ancillary:
        rng = np.random.default_rng(random_state)
        exclude = {"y", "d", "m", "m_obs", "tau_link", "g0", "g1", "cate"}
        x_cols = [c for c in df.columns if c not in exclude]
        df = _add_ancillary_info(df, n, rng, deterministic_ids, x_cols)

    return df


def _add_tweedie_pre(
    df: pd.DataFrame,
    n: int,
    gen: CausalDatasetGenerator,
    names: List[str],
    target_corr: float = 0.6,
    pre_name: str = "y_pre",
    A: Optional[np.ndarray] = None,
    pre_spec: Optional[PreCorrSpec] = None
) -> pd.DataFrame:
    """
    Helper to add a two-part pre-period covariate (y_pre) to a Tweedie-like dataset.
    Calibrates correlation with the post-period outcome using a shared latent driver
    and additive noise.
    """
    rng = gen.rng
    if A is None:
        A = rng.normal(size=n)

    if pre_spec is None:
        pre_spec = PreCorrSpec(target_corr=target_corr)

    X = df[names].to_numpy()
    ctrl = (df["d"].to_numpy() == 0)
    y_post = df["y"].to_numpy()

    # Reuse generator structural components so pre-period signal mirrors post-period DGP.
    alpha_zi = gen.alpha_zi
    beta_zi = gen.beta_zi
    g_zi = gen.g_zi
    alpha_y = gen.alpha_y
    beta_y = gen.beta_y
    g_y = gen.g_y
    k = gen.gamma_shape

    def sample_pre_base(w_shared: float) -> np.ndarray:
        # Shared latent driver A enters both zero-inflation and positive-part location.
        base_zi = np.full(n, float(alpha_zi))
        if beta_zi is not None:
            base_zi += X @ beta_zi
        if g_zi is not None:
            base_zi += g_zi(X)
        base_zi += w_shared * A

        loc = np.full(n, float(alpha_y))
        if beta_y is not None:
            loc += X @ beta_y
        if g_y is not None:
            loc += g_y(X)
        loc += w_shared * A

        p = _sigmoid(base_zi)
        is_pos = rng.binomial(1, p, size=n)

        mu = np.exp(np.clip(loc, -20, 20))
        y_pos = rng.gamma(shape=k, scale=mu / k, size=n)

        return (is_pos * y_pos).astype(float)

    # Use grid search over w_shared to find a base signal that can hit target_corr.
    # We prefer w_shared closer to 1.0 if possible.
    best_w = 1.0
    best_sigma = 0.0
    best_achieved = -1.0
    best_pre_base: Optional[np.ndarray] = None
    best_noise: Optional[np.ndarray] = None
    
    # Candidates for w_shared
    w_candidates = [1.0, 0.5, 2.0, 0.0, 4.0]
    
    for w in w_candidates:
        y_pre_w = sample_pre_base(w)
        # Calibrate and materialize using the same noise realization to avoid
        # target-correlation drift from re-sampling later.
        noise_w = rng.normal(size=n)
        sigma, achieved = calibrate_sigma_for_target_corr(
            y_pre_base=y_pre_w[ctrl],
            y_post=y_post[ctrl],
            rng=rng,
            spec=pre_spec,
            noise=noise_w[ctrl]
        )
        
        # If we hit the target (within tolerance), we prioritize smaller sigma 
        # (meaning we rely more on the latent driver)
        if abs(achieved - pre_spec.target_corr) <= pre_spec.sigma_tol:
            # We found a w that can hit the target. 
            # For simplicity, we take the first one that hits it from our prioritized list.
            best_w = w
            best_sigma = sigma
            best_achieved = achieved
            best_pre_base = y_pre_w
            best_noise = noise_w
            break
            
        # Otherwise, keep track of the one that gets closest to the target
        if abs(achieved - pre_spec.target_corr) < abs(best_achieved - pre_spec.target_corr):
            best_w = w
            best_sigma = sigma
            best_achieved = achieved
            best_pre_base = y_pre_w
            best_noise = noise_w

    # Materialize from the exact calibrated base/noise pair to avoid drift in achieved corr.
    if best_pre_base is None or best_noise is None:
        best_pre_base = sample_pre_base(best_w)
        best_noise = rng.normal(size=n)
    df[pre_name] = best_pre_base + best_sigma * best_noise
    return df


def make_cuped_tweedie(
    n: int = 10000,
    seed: int = 42,
    add_pre: bool = True,
    pre_name: str = "y_pre",
    pre_target_corr: float = 0.6,
    pre_spec: Optional[PreCorrSpec] = None,
    include_oracle: bool = False,
    return_causal_data: bool = True,
    theta_log: float = 0.2
) -> Union[pd.DataFrame, CausalData]:
    """
    Tweedie-like DGP with mixed marginals and structured HTE.
    Features many zeros and a heavy right tail. Suitable for CUPED benchmarking.

    Parameters
    ----------
    n : int, default=10000
        Number of samples to generate.
    seed : int, default=42
        Random seed.
    add_pre : bool, default=True
        Whether to add a pre-period covariate 'y_pre'.
    pre_name : str, default="y_pre"
        Name of the pre-period covariate column.
    pre_target_corr : float, default=0.6
        Target correlation between y_pre and post-outcome y in control group.
    pre_spec : PreCorrSpec, optional
        Detailed specification for pre-period calibration (transform, method, etc.).
        If provided, `pre_target_corr` is ignored in favor of `pre_spec.target_corr`.
    include_oracle : bool, default=False
        Whether to include oracle ground-truth columns like 'cate', 'propensity', etc.
    return_causal_data : bool, default=True
        Whether to return a CausalData object.
    theta_log : float, default=0.2
        The log-uplift theta parameter for the treatment effect.

    Returns
    -------
    pd.DataFrame or CausalData
    """
    confounder_specs = [
        {"name": "tenure_months",     "dist": "lognormal", "mu": 2.5, "sigma": 0.5},
        {"name": "avg_sessions_week", "dist": "negbin",    "mu": 5,   "alpha": 0.5},
        {"name": "spend_last_month",  "dist": "lognormal", "mu": 4.0, "sigma": 0.8},
        {"name": "discount_rate",     "dist": "beta",      "mean": 0.1, "kappa": 20},
        {"name": "platform",          "dist": "categorical", 
         "categories": ["android", "ios", "web"],
         "probs": [0.65, 0.30, 0.05]},
    ]

    # Use a hand-built correlation matrix for copula
    d_specs = len(confounder_specs)
    corr_matrix = np.eye(d_specs)
    corr_matrix[0, 1] = corr_matrix[1, 0] = 0.4  # tenure and sessions correlated
    corr_matrix[1, 2] = corr_matrix[2, 1] = 0.5  # sessions and spend correlated

    # Robust column mapping derived from specs
    tmp_gen = CausalDatasetGenerator(
        confounder_specs=confounder_specs,
        use_copula=True,
        copula_corr=corr_matrix,
        seed=seed
    )
    _, names = tmp_gen._sample_X(1)
    col_map = {name: i for i, name in enumerate(names)}

    def tau_fn(X: np.ndarray) -> np.ndarray:
        # Structured HTE: monotone + saturation + segment modifiers
        t = X[:, col_map["tenure_months"]]
        s = X[:, col_map["avg_sessions_week"]]

        # monotone in tenure (saturating)
        g_t = 1.0 / (1.0 + np.exp(-0.25 * (t - 12.0)))

        # diminishing returns in engagement
        g_s = 1.0 / (1.0 + np.exp(-0.35 * (s - 4.0)))

        ios = X[:, col_map.get("platform_ios", 0)] # fallback if not expanded
        seg = (1.0 + 0.30 * ios)

        tau_raw = theta_log * g_t * g_s * seg
        return np.clip(tau_raw, 0.0, 2.5 * theta_log)

    gen = CausalDatasetGenerator(
        outcome_type="tweedie",
        alpha_y=2.0,
        alpha_zi=0.0,  # about 50% non-zero baseline
        pos_dist="gamma",
        gamma_shape=2.0,
        tau=tau_fn,
        confounder_specs=confounder_specs,
        use_copula=True,
        copula_corr=corr_matrix,
        u_strength_y=1.0 if add_pre else 0.0,
        u_strength_zi=1.0 if add_pre else 0.0,
        include_oracle=include_oracle,
        seed=seed
    )

    if add_pre:
        # Shared latent driver
        rng = np.random.default_rng(seed)
        A = rng.normal(size=n)
        df = gen.generate(n, U=A)
        df = _add_tweedie_pre(
            df, n, gen, names, 
            target_corr=pre_target_corr, 
            pre_name=pre_name, 
            A=A,
            pre_spec=pre_spec
        )
        if include_oracle:
            # Expose the shared latent used to construct both y and y_pre for downstream diagnostics.
            df["_latent_A"] = A
    else:
        df = gen.generate(n)

    if not return_causal_data:
        return df

    # Re-infer confounders including y_pre
    exclude = {"y", "d", "m", "m_obs", "tau_link", "g0", "g1", "cate", "user_id", "_latent_A"}
    confounder_cols = [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]
    return CausalData(
        df=df,
        treatment="d",
        outcome="y",
        confounders=confounder_cols
    )


def generate_cuped_binary(
    n: int = 10000,
    seed: int = 42,
    add_pre: bool = True,
    pre_name: str = "y_pre",
    pre_target_corr: float = 0.65,
    pre_spec: Optional[PreCorrSpec] = None,
    include_oracle: bool = True,
    return_causal_data: bool = True,
    theta_logit: float = 0.38
) -> Union[pd.DataFrame, CausalData]:
    """
    Binary CUPED-oriented DGP with richer confounders and structured HTE.

    Designed for CUPED benchmarking with randomized treatment and a calibrated
    pre-period covariate while preserving exact oracle cate under include_oracle.

    Parameters
    ----------
    n : int, default=10000
        Number of samples to generate.
    seed : int, default=42
        Random seed.
    add_pre : bool, default=True
        Whether to add a pre-period covariate.
    pre_name : str, default="y_pre"
        Name of the pre-period covariate column.
    pre_target_corr : float, default=0.65
        Target correlation between y_pre and post-outcome y in the control group.
    pre_spec : PreCorrSpec, optional
        Detailed specification for pre-period calibration.
        If provided, `pre_target_corr` is ignored in favor of `pre_spec.target_corr`.
    include_oracle : bool, default=True
        Whether to include oracle columns like m, g0, g1, cate.
    return_causal_data : bool, default=True
        Whether to return a CausalData object.
    theta_logit : float, default=0.38
        Baseline log-odds uplift scale for heterogeneous treatment effects.

    Returns
    -------
    pd.DataFrame or CausalData
    """
    confounder_specs = [
        {"name": "tenure_months",     "dist": "lognormal", "mu": 2.5, "sigma": 0.5},
        {"name": "spend_last_month",  "dist": "lognormal", "mu": 4.0, "sigma": 0.8},
        {"name": "discount_rate",     "dist": "beta",      "mean": 0.1, "kappa": 20},
        {"name": "support_tickets",   "dist": "poisson",   "lam": 1.8},
        {"name": "email_open_rate",   "dist": "beta",      "mean": 0.45, "kappa": 14},
        {"name": "referral_count",    "dist": "poisson",   "lam": 0.8},
        {"name": "plan_tier",         "dist": "categorical",
         "categories": ["free", "plus", "pro"],
         "probs": [0.55, 0.30, 0.15]},
        {"name": "region",            "dist": "categorical",
         "categories": ["na", "eu"],
         "probs": [0.80, 0.20]},
    ]

    # Get expanded column names once so tau/g_y can reliably index one-hot columns.
    tmp_gen = CausalDatasetGenerator(
        confounder_specs=confounder_specs,
        seed=seed
    )
    _, names = tmp_gen._sample_X(1)
    col_map = {name: i for i, name in enumerate(names)}

    def _col(X: np.ndarray, name: str) -> np.ndarray:
        idx = col_map.get(name)
        if idx is None:
            return np.zeros(X.shape[0], dtype=float)
        return X[:, idx]

    def baseline_logit_fn(X: np.ndarray) -> np.ndarray:
        # Keep baseline effects moderate to preserve overlap in binary outcomes.
        tenure_log = np.log1p(_col(X, "tenure_months"))
        spend_log = np.log1p(_col(X, "spend_last_month"))
        discount = _col(X, "discount_rate")
        tickets_log = np.log1p(_col(X, "support_tickets"))
        open_rate = _col(X, "email_open_rate")
        referral_log = np.log1p(_col(X, "referral_count"))
        plan_plus = _col(X, "plan_tier_plus")
        plan_pro = _col(X, "plan_tier_pro")
        region_eu = _col(X, "region_eu")

        return (
            0.18 * (tenure_log - 2.4)
            + 0.14 * (spend_log - 4.0)
            - 0.95 * discount
            - 0.10 * tickets_log
            + 0.35 * open_rate
            + 0.08 * referral_log
            + 0.15 * plan_plus
            + 0.28 * plan_pro
            + 0.05 * region_eu
        )

    def tau_fn(X: np.ndarray) -> np.ndarray:
        tenure = _col(X, "tenure_months")
        spend_log = np.log1p(_col(X, "spend_last_month"))
        open_rate = _col(X, "email_open_rate")
        tickets = _col(X, "support_tickets")
        referrals = _col(X, "referral_count")
        plan_plus = _col(X, "plan_tier_plus")
        plan_pro = _col(X, "plan_tier_pro")
        region_eu = _col(X, "region_eu")

        g_t = 1.0 / (1.0 + np.exp(-0.20 * (tenure - 10.0)))
        g_sp = 1.0 / (1.0 + np.exp(-0.55 * (spend_log - 4.0)))
        g_o = 0.85 + 0.35 * open_rate
        # Heterogeneous modifiers: high support friction dampens effect,
        # while organic referral behavior amplifies it.
        g_ticket = 1.0 / (1.0 + np.exp(0.45 * (tickets - 2.5)))
        g_ref = 0.90 + 0.18 * np.tanh(0.8 * referrals)
        seg = 1.0 + 0.12 * plan_plus + 0.26 * plan_pro + 0.06 * region_eu

        tau_raw = theta_logit * g_t * g_sp * g_o * g_ticket * g_ref * seg
        return np.clip(tau_raw, 0.0, 2.0 * theta_logit)

    alpha_y = -1.4
    # Shared latent prognostic signal makes y_pre materially informative for CUPED
    # while preserving randomization-based identification (u_strength_d stays 0).
    shared_strength = 1.2 if add_pre else 0.0

    gen = CausalDatasetGenerator(
        outcome_type="binary",
        alpha_y=alpha_y,
        alpha_d=_logit(0.5),
        g_y=baseline_logit_fn,
        tau=tau_fn,
        confounder_specs=confounder_specs,
        u_strength_d=0.0,
        u_strength_y=shared_strength,
        include_oracle=include_oracle,
        seed=seed
    )

    if add_pre:
        rng = np.random.default_rng(seed)
        A = rng.normal(size=n)
        df = gen.generate(n, U=A)

        def base_builder(df_in: pd.DataFrame) -> np.ndarray:
            X = df_in[names].to_numpy(dtype=float)
            # Probability-scale baseline aligned with binary outcome is more reliable
            # for variance reduction than raw logit score.
            lin0 = (
                float(alpha_y)
                + np.asarray(baseline_logit_fn(X), dtype=float).reshape(-1)
                + float(shared_strength) * A
            )
            p0 = np.asarray(_sigmoid(np.clip(lin0, -20.0, 20.0)), dtype=float).reshape(-1)
            return p0 - float(p0.mean())

        spec = pre_spec or PreCorrSpec(
            target_corr=pre_target_corr,
            transform="none"
        )
        df = add_preperiod_covariate(
            df=df,
            y_col="y",
            d_col="d",
            pre_name=pre_name,
            base_builder=base_builder,
            spec=spec,
            rng=rng
        )
    else:
        df = gen.generate(n)

    if not return_causal_data:
        return df

    exclude = {"y", "d", "m", "m_obs", "tau_link", "g0", "g1", "cate", "user_id"}
    confounder_cols = [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]
    return CausalData(
        df=df,
        treatment="d",
        outcome="y",
        confounders=confounder_cols
    )


def make_gold_linear(n: int = 10000, seed: int = 42) -> CausalData:
    """
    A standard linear benchmark with moderate confounding.
    Based on the benchmark scenario in docs/research/dgp_benchmarking.ipynb.
    """
    confounder_specs = [
        {"name": "tenure_months", "dist": "normal", "mu": 24, "sd": 12},
        {"name": "avg_sessions_week", "dist": "normal", "mu": 5, "sd": 2},
        {"name": "spend_last_month", "dist": "uniform", "a": 0, "b": 200},
        {"name": "premium_user", "dist": "bernoulli", "p": 0.25},
        {"name": "urban_resident", "dist": "bernoulli", "p": 0.60},
    ]

    # Coefficients aligned to the specs above
    beta_y = [0.05, 0.60, 0.005, 0.80, 0.20]
    beta_d = [0.08, 0.12, 0.004, 0.25, 0.10]

    gen = CausalDatasetGenerator(
        theta=0.80,
        beta_y=beta_y,
        beta_d=beta_d,
        alpha_y=0.0,
        alpha_d=0.0,
        sigma_y=1.0,
        outcome_type="continuous",
        confounder_specs=confounder_specs,
        target_d_rate=0.20,
        seed=seed
    )
    return gen.to_causal_data(n)
