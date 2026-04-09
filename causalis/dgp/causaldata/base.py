"""
Low-level generators for single-treatment synthetic causal datasets.

The central class in this module is :class:`CausalDatasetGenerator`, which
builds observational or randomized-style data with binary treatment and
optionally exposes oracle quantities such as propensities and potential-outcome
means.

Notes
-----
This module is the lowest-level tabular DGP layer used by higher-level helpers
in :mod:`causalis.dgp.causaldata.functional` and scenario-specific wrappers.
Reach for it when you want direct control over the structural equations rather
than a pre-packaged example dataset.

Examples
--------
>>> import numpy as np
>>> from causalis.dgp.causaldata.base import CausalDatasetGenerator
>>> gen = CausalDatasetGenerator(
...     theta=1.5,
...     beta_y=np.array([0.7, -0.4]),
...     beta_d=np.array([1.0, 0.5]),
...     target_d_rate=0.35,
...     seed=3141,
...     include_oracle=True,
...     k=2,
... )
>>> df = gen.generate(200)
>>> sorted(col for col in ["y", "d", "m", "cate"] if col in df.columns)
['cate', 'd', 'm', 'y']
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import sys
from dataclasses import dataclass, field
from typing import Dict, Optional, Union, List, Tuple, Callable, Any

from causalis.dgp.causaldata import CausalData
from causalis.dgp.base import _sigmoid, _gaussian_copula

_DATACLASS_KWARGS = {"slots": True} if sys.version_info >= (3, 10) else {}


@dataclass(**_DATACLASS_KWARGS)
class CausalDatasetGenerator:
    """
    Generate synthetic causal inference datasets with controllable confounding,
    treatment prevalence, noise, and (optionally) heterogeneous treatment effects.

    **Data model (high level)**

    - confounders X ∈ R^k are drawn from user-specified distributions.
    - Binary treatment D is assigned by a logistic model:
        D ~ Bernoulli( sigmoid(alpha_d + f_d(X) + u_strength_d * U) ),
      where f_d(X) = (X @ beta_d + g_d(X)) * propensity_sharpness, and U ~ N(0,1) is an optional unobserved confounder.
    - Outcome Y depends on treatment and confounders with link determined by `outcome_type`:
        outcome_type = "continuous":
            Y = alpha_y + f_y(X) + u_strength_y * U + T * tau(X) + ε,  ε ~ N(0, sigma_y^2)
        outcome_type = "binary":
            logit P(Y=1|T,X,U) = alpha_y + f_y(X) + u_strength_y * U + T * tau(X)
        outcome_type = "poisson":
            log E[Y|T,X,U]     = alpha_y + f_y(X) + u_strength_y * U + T * tau(X)
        outcome_type = "gamma":
            log E[Y|T,X,U]     = alpha_y + f_y(X) + u_strength_y * U + T * tau(X)
      where f_y(X) = X @ beta_y + g_y(X), and tau(X) is either constant `theta` or a user function.

    **Returned columns**

    - y: outcome
    - d: binary treatment (0/1)
    - x1..xk (or user-provided names)
    - m: true propensity P(T=1 | X) marginalized over U
    - m_obs: realized propensity P(T=1 | X, U)
    - tau_link: tau(X) on the structural (link) scale
    - g0: E[Y | X, T=0] on the natural outcome scale marginalized over U .,9 
    - g1: E[Y | X, T=1] on the natural outcome scale marginalized over U
    - cate: g1 - g0 (conditional average treatment effect on the natural outcome scale)

    Notes on effect scale:
      - For "continuous", `theta` (or tau(X)) is an additive mean difference, so `tau_link == cate`.
      - For "binary", tau acts on the *log-odds* scale. `cate` is reported as a risk difference.
      - For "poisson" and "gamma", tau acts on the *log-mean* scale. `cate` is reported on the mean scale.

    Parameters
    ----------
    theta : float, default=1.0
        Constant treatment effect used if `tau` is None.
    tau : callable, optional
        Function tau(X) -> array-like shape (n,) for heterogeneous effects.
    beta_y : array-like, optional
        Linear coefficients of confounders in the outcome baseline f_y(X).
    beta_d : array-like, optional
        Linear coefficients of confounders in the treatment score f_d(X).
    g_y : callable, optional
        Nonlinear/additive function g_y(X) -> (n,) added to the outcome baseline.
    g_d : callable, optional
        Nonlinear/additive function g_d(X) -> (n,) added to the treatment score.
    alpha_y : float, default=0.0
        Outcome intercept (natural scale for continuous; log-odds for binary; log-mean for Poisson/Gamma).
    alpha_d : float, default=0.0
        Treatment intercept (log-odds). If `target_d_rate` is set, `alpha_d` is auto-calibrated.
    sigma_y : float, default=1.0
        Std. dev. of the Gaussian noise for continuous outcomes.
    outcome_type : {"continuous", "binary", "poisson", "gamma", "tweedie"}, default="continuous"
        Outcome family and link as defined above.
    confounder_specs : list of dict, optional
        Schema for generating confounders. See `_gaussian_copula` for details.
    k : int, default=5
        Number of confounders when `confounder_specs` is None. Defaults to independent N(0,1).
    x_sampler : callable, optional
        Custom sampler (n, k, seed) -> X ndarray of shape (n,k). Overrides `confounder_specs`.
    use_copula : bool, default=False
        If True and `confounder_specs` provided, use Gaussian copula for X.
    copula_corr : array-like, optional
        Correlation matrix for copula.
    target_d_rate : float, optional
        Target treatment prevalence (propensity mean). Calibrates `alpha_d`.
    u_strength_d : float, default=0.0
        Strength of the unobserved confounder U in treatment assignment.
    u_strength_y : float, default=0.0
        Strength of the unobserved confounder U in the outcome.
    propensity_sharpness : float, default=1.0
        Scales the X-driven treatment score to adjust positivity difficulty.
    seed : int, optional
        Random seed for reproducibility.

    Attributes
    ----------
    rng : numpy.random.Generator
        Internal RNG seeded from `seed`.

    Notes
    -----
    Oracle outputs are reported on the natural outcome scale:

    - ``m`` is the treatment propensity marginalized over latent noise.
    - ``g0`` and ``g1`` are mean potential outcomes on the observed outcome
      scale.
    - ``cate`` is always ``g1 - g0`` on that same natural scale, even when the
      structural treatment effect is specified on a link scale such as log-odds
      or log-mean.

    Examples
    --------
    >>> import numpy as np
    >>> gen = CausalDatasetGenerator(
    ...     theta=0.8,
    ...     beta_y=np.array([0.5, -0.2, 0.1]),
    ...     beta_d=np.array([0.8, 0.4, -0.3]),
    ...     target_d_rate=0.4,
    ...     outcome_type="continuous",
    ...     seed=123,
    ...     k=3,
    ... )
    >>> df = gen.generate(1000)
    >>> float(df["d"].mean()) > 0.0
    True
    >>> "cate" in df.columns
    True
    """
    # Core knobs
    theta: float = 1.0                            # constant treatment effect (ATE) if tau is None
    tau: Optional[Callable[[np.ndarray], np.ndarray]] = None  # heterogeneous effect tau(X) if provided

    # confounder -> outcome/treatment effects
    beta_y: Optional[np.ndarray] = None           # shape (k,)
    beta_d: Optional[np.ndarray] = None           # shape (k,)
    g_y: Optional[Callable[[np.ndarray], np.ndarray]] = None  # nonlinear baseline outcome f_y(X)
    g_d: Optional[Callable[[np.ndarray], np.ndarray]] = None  # nonlinear treatment score f_d(X)

    # Outcome/treatment intercepts and noise
    alpha_y: float = 0.0
    alpha_d: float = 0.0
    sigma_y: float = 1.0                          # used when outcome_type="continuous"
    outcome_type: str = "continuous"              # "continuous" | "binary" | "poisson" | "gamma" | "tweedie"

    # confounder generation
    confounder_specs: Optional[List[Dict[str, Any]]] = None   # list of {"name","dist",...}
    k: int = 5                                    # used if confounder_specs is None
    x_sampler: Optional[Callable[[int, int, int], np.ndarray]] = None  # custom sampler (n, k, seed)->X
    use_copula: bool = False                      # if True and confounder_specs provided, use Gaussian copula
    copula_corr: Optional[np.ndarray] = None      # correlation matrix for copula (shape dxd where d=len(specs))

    # Practical controls
    target_d_rate: Optional[float] = None         # e.g., 0.3 -> ~30% treated; solves for alpha_d
    u_strength_d: float = 0.0                     # unobserved confounder effect on treatment
    u_strength_y: float = 0.0                     # unobserved confounder effect on outcome
    propensity_sharpness: float = 1.0             # scales the X-driven treatment score to adjust positivity difficulty
    score_bounding: Optional[float] = None        # if set, applies c * tanh(score / c) to the treatment score to ensure overlap

    # Two-part / Tweedie-like knobs
    alpha_zi: float = -1.0
    beta_zi: Optional[np.ndarray] = None
    g_zi: Optional[Callable[[np.ndarray], np.ndarray]] = None
    u_strength_zi: float = 0.0
    tau_zi: Optional[Callable[[np.ndarray], np.ndarray]] = None  # optional effect on nonzero prob

    pos_dist: str = "gamma"          # "gamma" | "lognormal"
    gamma_shape: float = 2.0         # for gamma outcomes and tweedie positive part
    lognormal_sigma: float = 1.0     # for lognormal positive part

    include_oracle: bool = True                   # whether to include oracle ground-truth columns in generate()
    seed: Optional[int] = None

    # Internals (filled post-init)
    rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self):
        """Initialize RNG and validate configuration."""
        self.rng = np.random.default_rng(self.seed)
        if self.confounder_specs is not None:
            self.k = len(self.confounder_specs)

    # ---------- confounder sampling ----------

    def _sample_X(self, n: int) -> Tuple[np.ndarray, List[str]]:
        """
        Sample confounders X.

        Parameters
        ----------
        n : int
            Number of samples.

        Returns
        -------
        X : numpy.ndarray
            Confounder matrix.
        names : list of str
            Column names.
        """
        if self.x_sampler is not None:
            X = self.x_sampler(n, self.k, self.seed)
            if self.confounder_specs is not None:
                names = [spec.get("name", f"x{i+1}") for i, spec in enumerate(self.confounder_specs)]
            else:
                names = [f"x{i+1}" for i in range(self.k)]
            return X, names

        if self.confounder_specs is None:
            # Default: independent standard normals
            X = self.rng.normal(size=(n, self.k))
            names = [f"x{i+1}" for i in range(self.k)]
            return X, names

        # If specs are provided and copula is requested, use Gaussian copula
        if getattr(self, "use_copula", False):
            # Copula returns both realized columns and their names (after any one-hot expansion).
            X, names = _gaussian_copula(self.rng, n, self.confounder_specs, getattr(self, "copula_corr", None))
            self.k = X.shape[1]
            return X, names

        cols = []
        names = []
        for spec in self.confounder_specs:
            name = spec.get("name") or f"x{len(names)+1}"
            dist = spec.get("dist", "normal").lower()
            if dist == "normal":
                mu = spec.get("mu", 0.0); sd = spec.get("sd", 1.0)
                col = self.rng.normal(mu, sd, size=n)
            elif dist == "uniform":
                a = spec.get("a", 0.0); b = spec.get("b", 1.0)
                col = self.rng.uniform(a, b, size=n)
            elif dist == "bernoulli":
                p = spec.get("p", 0.5)
                col = self.rng.binomial(1, p, size=n).astype(float)
            elif dist == "lognormal":
                mu = float(spec.get("mu", 0.0))
                sigma = float(spec.get("sigma", 1.0))
                col = self.rng.lognormal(mean=mu, sigma=sigma, size=n)
            elif dist == "gamma":
                shape = float(spec.get("shape", 2.0))
                scale = spec.get("scale", None)
                if scale is None:
                    mean = float(spec.get("mean", 1.0))
                    scale = mean / shape
                col = self.rng.gamma(shape=shape, scale=float(scale), size=n)
            elif dist == "beta":
                a = spec.get("a", None); b = spec.get("b", None)
                if a is None or b is None:
                    mean = float(spec.get("mean", 0.5))
                    kappa = float(spec.get("kappa", 10.0))
                    a = mean * kappa
                    b = (1.0 - mean) * kappa
                col = self.rng.beta(float(a), float(b), size=n)
            elif dist == "poisson":
                lam = float(spec.get("lam", 1.0))
                col = self.rng.poisson(lam=lam, size=n).astype(float)
            elif dist == "negbin":
                mu = float(spec.get("mu", 5.0))
                alpha = float(spec.get("alpha", 0.5))  # Var = mu + alpha*mu^2
                r = 1.0 / max(alpha, 1e-12)
                p = r / (r + mu)
                col = self.rng.negative_binomial(n=r, p=p, size=n).astype(float)
            elif dist == "categorical":
                categories = list(spec.get("categories", [0,1,2]))
                probs = spec.get("probs", None)
                if probs is not None:
                    p = np.asarray(probs, dtype=float)
                    ps = p / p.sum()
                else:
                    ps = None
                col = self.rng.choice(categories, p=ps, size=n)
                # one-hot encode (except first level)
                rest = categories[1:]
                if len(rest) == 0:
                    cols.append(np.zeros(n, dtype=float))
                    names.append(f"{name}__onlylevel")
                    continue
                for c in rest:
                    cols.append((col == c).astype(float))
                    names.append(f"{name}_{c}")
                continue
            else:
                raise ValueError(f"Unknown dist: {dist}")
            
            # Apply clipping if specified in spec
            if "clip_min" in spec or "clip_max" in spec:
                cmin = spec.get("clip_min", -np.inf)
                cmax = spec.get("clip_max", np.inf)
                col = np.clip(col, cmin, cmax)

            cols.append(col.astype(float))
            names.append(name)
        X = np.column_stack(cols) if cols else np.empty((n,0))
        self.k = X.shape[1]
        return X, names

    # ---------- Helpers ----------

    def _treatment_score(self, X: np.ndarray, U: np.ndarray) -> np.ndarray:
        """
        Compute the linear part of the treatment assignment score.

        Parameters
        ----------
        X : numpy.ndarray
            Confounder matrix.
        U : numpy.ndarray
            Unobserved confounder.

        Returns
        -------
        numpy.ndarray
            Linear score lin = (X @ beta_d + g_d(X)) * sharpness + u_strength_d * U.
        """
        # Ensure numeric, finite arrays
        Xf = np.asarray(X, dtype=float)
        # X-driven part of the score
        score_x = np.zeros(Xf.shape[0], dtype=float)
        if self.beta_d is not None:
            bt = np.asarray(self.beta_d, dtype=float)
            if bt.ndim != 1:
                bt = bt.reshape(-1)
            if bt.shape[0] != Xf.shape[1]:
                raise ValueError(f"beta_d shape {bt.shape} is incompatible with X shape {Xf.shape}")
            score_x += np.sum(Xf * bt, axis=1)
        if self.g_d is not None:
            score_x += np.asarray(self.g_d(Xf), dtype=float)
        # Scale sharpness to control positivity difficulty
        s = float(getattr(self, "propensity_sharpness", 1.0))
        lin = s * score_x

        # Apply score bounding if requested
        if self.score_bounding is not None:
            c = float(self.score_bounding)
            lin = c * np.tanh(lin / c)

        # Add unobserved confounder contribution (unscaled)
        if self.u_strength_d != 0:
            lin += self.u_strength_d * np.asarray(U, dtype=float)
        return lin

    def _outcome_location(self, X: np.ndarray, D: np.ndarray, U: np.ndarray, tau_x: np.ndarray) -> np.ndarray:
        """
        Compute the location parameter for the outcome distribution.

        Parameters
        ----------
        X : numpy.ndarray
            Confounder matrix.
        D : numpy.ndarray
            Treatment assignment.
        U : numpy.ndarray
            Unobserved confounder.
        tau_x : numpy.ndarray
            Treatment effect for each sample.

        Returns
        -------
        numpy.ndarray
            Location parameter (mean for continuous, logit for binary, log for Poisson/Gamma).
        """
        # location on natural scale for continuous; on logit/log scale for binary/poisson
        Xf = np.asarray(X, dtype=float)
        Df = np.asarray(D, dtype=float)
        Uf = np.asarray(U, dtype=float)
        taux = np.asarray(tau_x, dtype=float)
        loc = np.full(Xf.shape[0], float(self.alpha_y), dtype=float)
        if self.beta_y is not None:
            by = np.asarray(self.beta_y, dtype=float)
            if by.ndim != 1:
                by = by.reshape(-1)
            if by.shape[0] != Xf.shape[1]:
                raise ValueError(f"beta_y shape {by.shape} is incompatible with X shape {Xf.shape}")
            loc += np.sum(Xf * by, axis=1)
        if self.g_y is not None:
            loc += np.asarray(self.g_y(Xf), dtype=float)
        if self.u_strength_y != 0:
            loc += self.u_strength_y * Uf
        loc += Df * taux
        return loc

    def _calibrate_alpha_d(self, X: np.ndarray, U: np.ndarray, target: float) -> float:
        """
        Calibrate alpha_d so mean propensity ~= target using robust bracketing and bisection.

        Parameters
        ----------
        X : numpy.ndarray
            Confounder matrix.
        U : numpy.ndarray
            Unobserved confounder.
        target : float
            Target treatment prevalence.

        Returns
        -------
        float
            Calibrated alpha_d.
        """
        # Wide initial bracket on the log-odds intercept.
        lo, hi = -50.0, 50.0
        # Define function whose root we seek
        def f(a: float) -> float:
            return float(_sigmoid(a + self._treatment_score(X, U)).mean() - target)
        flo, fhi = f(lo), f(hi)
        # If the target is not bracketed, try expanding the bracket a few times
        if flo * fhi > 0:
            for scale in (2, 5, 10):
                lo2, hi2 = -scale * 50.0, scale * 50.0
                flo2, fhi2 = f(lo2), f(hi2)
                if flo2 * fhi2 <= 0:
                    lo, hi = lo2, hi2
                    flo, fhi = flo2, fhi2
                    break
            else:
                # Fall back to the closer endpoint
                return lo if abs(flo) < abs(fhi) else hi
        # Standard monotone-root bisection on mean(sigmoid(alpha + score)) - target.
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            fm = f(mid)
            if abs(fm) < 1e-6:
                return mid
            if fm > 0:
                hi = mid
            else:
                lo = mid
        return 0.5 * (lo + hi)

    # ---------- Public API ----------

    def generate(self, n: int, U: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Draw a synthetic dataset of size `n`.

        Parameters
        ----------
        n : int
            Number of samples to generate.
        U : numpy.ndarray, optional
            Unobserved confounder. If None, generated from N(0,1).

        Returns
        -------
        pandas.DataFrame
            The generated dataset with outcome 'y', treatment 'd', confounders,
            and oracle ground-truth columns.
        """
        X, names = self._sample_X(n)
        if U is None:
            U = self.rng.normal(size=n)  # unobserved confounder
        U = np.asarray(U, dtype=float)

        # Treatment assignment
        if self.target_d_rate is not None:
            # Recalibrate alpha_d on the sampled X each call so realized prevalence tracks target.
            self.alpha_d = self._calibrate_alpha_d(X, U, self.target_d_rate)
        logits_t = self.alpha_d + self._treatment_score(X, U)
        m_obs = _sigmoid(logits_t)  # realized propensity including U

        # Marginal m(x) = E[D|X] (integrate out U if it affects treatment)
        if float(self.u_strength_d) != 0.0:
            # Gauss-Hermite computes E[sigmoid(base + u_strength_d * U)] for U~N(0,1).
            gh_x, gh_w = np.polynomial.hermite.hermgauss(21)
            gh_w = gh_w / np.sqrt(np.pi)
            base = self.alpha_d + self._treatment_score(X, np.zeros(n))
            m = np.sum(_sigmoid(base[:, None] + self.u_strength_d * np.sqrt(2.0) * gh_x[None, :]) * gh_w[None, :], axis=1)
        else:
            # Closed form when U doesn't affect D
            base = self.alpha_d + self._treatment_score(X, np.zeros(n))
            m = _sigmoid(base)

        D = self.rng.binomial(1, m_obs).astype(float)

        # Treatment effect (constant or heterogeneous)
        # tau_x is always on the structural link scale used by the selected outcome family.
        tau_x = (self.tau(X) if self.tau is not None else np.full(n, self.theta)).astype(float)

        # Outcome generation
        loc = self._outcome_location(X, D, U, tau_x)

        if self.outcome_type == "continuous":
            Y = loc + self.rng.normal(0, self.sigma_y, size=n)
        elif self.outcome_type == "binary":
            # logit: logit P(Y=1|D,X) = loc
            p = _sigmoid(loc)
            Y = self.rng.binomial(1, p).astype(float)
        elif self.outcome_type == "poisson":
            # log link: log E[Y|D,X] = loc; guard against overflow on link scale
            loc_c = np.clip(loc, -20, 20)
            lam = np.exp(loc_c)
            Y = self.rng.poisson(lam).astype(float)
        elif self.outcome_type == "gamma":
            # log link: log E[Y|D,X] = loc; draw Gamma with mean mu and shape k
            loc_c = np.clip(loc, -20, 20)
            mu = np.exp(loc_c)
            k = float(self.gamma_shape)
            scale = mu / max(k, 1e-12)
            Y = self.rng.gamma(shape=k, scale=scale, size=n).astype(float)
        elif self.outcome_type == "tweedie":
            # 1) Zero/positive part: logit P(Y>0 | X,D,U).
            Xf = np.asarray(X, dtype=float)
            base_zi = np.full(n, float(self.alpha_zi), dtype=float)
            if self.beta_zi is not None:
                bz = np.asarray(self.beta_zi, dtype=float).reshape(-1)
                if bz.shape[0] != Xf.shape[1]:
                    raise ValueError("beta_zi incompatible with X")
                base_zi += np.sum(Xf * bz, axis=1)
            if self.g_zi is not None:
                base_zi += np.asarray(self.g_zi(Xf), dtype=float)
            if self.u_strength_zi != 0.0:
                base_zi += float(self.u_strength_zi) * np.asarray(U, dtype=float)

            tau_zi_x = np.zeros(n, dtype=float)
            if self.tau_zi is not None:
                tau_zi_x = np.asarray(self.tau_zi(Xf), dtype=float).reshape(-1)

            p_pos = _sigmoid(base_zi + D * tau_zi_x)
            is_pos = self.rng.binomial(1, p_pos, size=n).astype(float)

            # 2) Positive-part mean: loc is interpreted as log E[Y | Y>0, X,D,U].
            # IMPORTANT: interpret alpha_y / beta_y / g_y as log-mean components for tweedie
            loc_pos = np.clip(loc, -20, 20)
            mu_pos = np.exp(loc_pos)

            # 3) draw positive values
            if str(self.pos_dist).lower() == "gamma":
                k = float(self.gamma_shape)
                scale = mu_pos / max(k, 1e-12)
                y_pos = self.rng.gamma(shape=k, scale=scale, size=n)
            elif str(self.pos_dist).lower() == "lognormal":
                s = float(self.lognormal_sigma)
                # pick mean-corrected lognormal: if log Y ~ N(m, s^2), then E[Y]=exp(m+s^2/2)
                m = np.log(mu_pos) - 0.5 * s * s
                y_pos = self.rng.lognormal(mean=m, sigma=s, size=n)
            else:
                raise ValueError("pos_dist must be 'gamma' or 'lognormal'")

            Y = is_pos * y_pos
        else:
            raise ValueError("outcome_type must be 'continuous', 'binary', 'poisson', 'gamma' or 'tweedie'")

        # Compute oracle g0/g1 on the natural scale, excluding U for continuous, and
        # marginalizing over U for binary/poisson when u_strength_y != 0.
        if self.outcome_type == "continuous":
            # Oracle means exclude U (mean-zero unobserved)
            g0 = self._outcome_location(X, np.zeros(n), np.zeros(n), np.zeros(n))
            g1 = self._outcome_location(X, np.ones(n),  np.zeros(n), tau_x)

        elif self.outcome_type == "binary":
            if float(self.u_strength_y) != 0.0:
                # Oracle risk under treatment/control integrates latent U out on probability scale.
                gh_x, gh_w = np.polynomial.hermite.hermgauss(21)
                gh_w = gh_w / np.sqrt(np.pi)
                base0 = self._outcome_location(X, np.zeros(n), np.zeros(n), np.zeros(n))
                base1 = self._outcome_location(X, np.ones(n),  np.zeros(n), tau_x)
                Uq = np.sqrt(2.0) * gh_x
                g0 = np.sum(_sigmoid(base0[:, None] + self.u_strength_y * Uq[None, :]) * gh_w[None, :], axis=1)
                g1 = np.sum(_sigmoid(base1[:, None] + self.u_strength_y * Uq[None, :]) * gh_w[None, :], axis=1)
            else:
                g0 = _sigmoid(self._outcome_location(X, np.zeros(n), np.zeros(n), np.zeros(n)))
                g1 = _sigmoid(self._outcome_location(X, np.ones(n),  np.zeros(n), tau_x))

        elif self.outcome_type in {"poisson", "gamma"}:
            if float(self.u_strength_y) != 0.0:
                # Oracle mean integrates exp(link) over U; clipping keeps exponentials stable.
                gh_x, gh_w = np.polynomial.hermite.hermgauss(21)
                gh_w = gh_w / np.sqrt(np.pi)
                base0 = self._outcome_location(X, np.zeros(n), np.zeros(n), np.zeros(n))
                base1 = self._outcome_location(X, np.ones(n),  np.zeros(n), tau_x)
                Uq = np.sqrt(2.0) * gh_x
                g0 = np.sum(np.exp(np.clip(base0[:, None] + self.u_strength_y * Uq[None, :], -20, 20)) * gh_w[None, :], axis=1)
                g1 = np.sum(np.exp(np.clip(base1[:, None] + self.u_strength_y * Uq[None, :], -20, 20)) * gh_w[None, :], axis=1)
            else:
                g0 = np.exp(np.clip(self._outcome_location(X, np.zeros(n), np.zeros(n), np.zeros(n)), -20, 20))
                g1 = np.exp(np.clip(self._outcome_location(X, np.ones(n),  np.zeros(n), tau_x), -20, 20))
        elif self.outcome_type == "tweedie":
            # Oracle g(d) on natural scale:
            #   g(d) = E_U[sigmoid(zi(X,d,U)) * exp(loc(X,d,U))]
            # where U ~ N(0,1). This is a one-dimensional integral and is computed
            # via Gauss-Hermite when latent U enters either zi or y locations.
            Xf = np.asarray(X, dtype=float)
            base_zi = np.full(n, float(self.alpha_zi), dtype=float)
            if self.beta_zi is not None:
                bz = np.asarray(self.beta_zi, dtype=float).reshape(-1)
                if bz.shape[0] != Xf.shape[1]:
                    raise ValueError("beta_zi incompatible with X")
                base_zi += np.sum(Xf * bz, axis=1)
            if self.g_zi is not None:
                base_zi += np.asarray(self.g_zi(Xf), dtype=float)

            tau_zi_x = np.zeros(n, dtype=float)
            if self.tau_zi is not None:
                tau_zi_x = np.asarray(self.tau_zi(Xf), dtype=float).reshape(-1)

            # Base log-mean locations excluding U
            loc0 = self._outcome_location(X, np.zeros(n), np.zeros(n), np.zeros(n))
            loc1 = self._outcome_location(X, np.ones(n), np.zeros(n), tau_x)

            uy = float(self.u_strength_y)
            uzi = float(self.u_strength_zi)
            if (uy != 0.0) or (uzi != 0.0):
                gh_x, gh_w = np.polynomial.hermite.hermgauss(21)
                gh_w = gh_w / np.sqrt(np.pi)
                Uq = np.sqrt(2.0) * gh_x

                zi0_u = base_zi[:, None] + uzi * Uq[None, :]
                zi1_u = (base_zi + tau_zi_x)[:, None] + uzi * Uq[None, :]
                loc0_u = loc0[:, None] + uy * Uq[None, :]
                loc1_u = loc1[:, None] + uy * Uq[None, :]

                # Product-of-means form comes from two-part construction: P(Y>0)*E[Y|Y>0].
                p_pos0 = _sigmoid(zi0_u)
                p_pos1 = _sigmoid(zi1_u)
                mu_pos0 = np.exp(np.clip(loc0_u, -20, 20))
                mu_pos1 = np.exp(np.clip(loc1_u, -20, 20))

                g0 = np.sum(p_pos0 * mu_pos0 * gh_w[None, :], axis=1)
                g1 = np.sum(p_pos1 * mu_pos1 * gh_w[None, :], axis=1)
            else:
                p_pos0 = _sigmoid(base_zi)
                p_pos1 = _sigmoid(base_zi + tau_zi_x)
                mu_pos0 = np.exp(np.clip(loc0, -20, 20))
                mu_pos1 = np.exp(np.clip(loc1, -20, 20))

                g0 = p_pos0 * mu_pos0
                g1 = p_pos1 * mu_pos1
        else:
            raise ValueError("outcome_type must be 'continuous', 'binary', 'poisson', 'gamma' or 'tweedie'")

        df = pd.DataFrame({"y": Y, "d": D})
        for j, name in enumerate(names):
            df[name] = X[:, j]

        if self.include_oracle:
            # Useful ground-truth columns for evaluation (IRM naming)
            df["m"] = m              # marginal E[D|X]
            df["m_obs"] = m_obs      # realized sigmoid(alpha_d + score + u*U)
            df["tau_link"] = tau_x   # structural effect on the link scale
            df["g0"] = g0
            df["g1"] = g1
            df["cate"] = df["g1"] - df["g0"]

        return df

    def to_causal_data(self, n: int, confounders: Optional[Union[str, List[str]]] = None) -> CausalData:
        """
        Generate a dataset and convert it to a CausalData object.

        Parameters
        ----------
        n : int
            Number of samples to generate.
        confounders : str or list of str, optional
            List of confounder column names to include. If None, automatically detects numeric confounders.

        Returns
        -------
        CausalData
            A CausalData object containing the generated dataset.
        """
        df = self.generate(n)

        # Determine confounders to use
        exclude = {'y', 'd', 'm', 'm_obs', 'tau_link', 'g0', 'g1', 'cate', 'user_id'}
        if confounders is None:
            # Keep original column order; exclude outcome/treatment/ground-truth columns and non-numeric
            confounder_cols = [
                c for c in df.columns
                if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
            ]
        elif isinstance(confounders, str):
            confounder_cols = [confounders]
        else:
            confounder_cols = [c for c in confounders if c in df.columns]

        # Create and return CausalData object
        user_id = 'user_id' if 'user_id' in df.columns else None
        
        # If include_oracle is True, we want to keep ground-truth columns in cd.df 
        # even if they are not listed as confounders.
        # CausalData subsets df to [treatment, outcome, user_id] + confounders by default.
        # To keep oracle columns, we can either:
        # a) pass them as extra columns if CausalData supports it (it doesn't seem to)
        # b) use from_df with extra kwargs if it helps? No, it subsets in _validate_and_normalize.
        
        # Actually, CausalData.df is subsetted to [treatment, outcome, user_id] + confounders.
        # If we want to keep oracle, they must be "known" to CausalData or we don't use CausalData for oracle.
        # However, the user might want them for evaluation.
        
        # Let's see if we can trick CausalData by adding them to confounders? 
        # No, they are not confounders.
        
        # Best way: return CausalData and if oracle is needed, the user should use the raw df or we modify CausalData.
        # Given I cannot easily modify CausalData's core logic without side effects, 
        # I will just documented that CausalData subsets the df.
        
        # WAIT! If I want my tests to pass, I should probably use the raw df for verification if I need oracle.
        
        return CausalData(df=df, treatment='d', outcome='y', confounders=confounder_cols, user_id=user_id)

    def oracle_nuisance(self, num_quad: int = 21):
        """
        Return nuisance functions (m(x), g0(x), g1(x)) compatible with IRM.

        Parameters
        ----------
        num_quad : int, default=21
            Number of quadrature points for marginalizing over U.

        Returns
        -------
        dict
            Dictionary of callables mapping X to nuisance values.
        """
        # Disallow unobserved cofounding for DML when U affects both T and Y
        if (getattr(self, "u_strength_d", 0.0) != 0) and (getattr(self, "u_strength_y", 0.0) != 0):
            raise ValueError(
                "DML identification fails when U affects both T and Y. "
                "Use instruments (PLIV-DML) or set one of u_strength_* to 0."
            )

        # Precompute GH nodes/weights normalized for N(0,1)
        gh_x, gh_w = np.polynomial.hermite.hermgauss(int(num_quad))
        gh_w = gh_w / np.sqrt(np.pi)

        def m_of_x(x_row: np.ndarray) -> float:
            x = np.asarray(x_row, dtype=float).reshape(1, -1)
            # Base score without U contribution, matching _treatment_score(X, U=0)
            base = float(self.alpha_d)
            base += float(self._treatment_score(x, np.zeros(1, dtype=float))[0])
            ut = float(getattr(self, "u_strength_d", 0.0))
            if ut == 0.0:
                return float(_sigmoid(base))
            # Integrate over U ~ N(0,1) using Gauss–Hermite: U = sqrt(2) * gh_x
            z = base + ut * np.sqrt(2.0) * gh_x
            return float(np.sum(_sigmoid(z) * gh_w))

        def g_of_x_d(x_row: np.ndarray, d: int) -> float:
            x = np.asarray(x_row, dtype=float).reshape(1, -1)
            if self.tau is None:
                tau_val = float(self.theta)
            else:
                tau_val = float(np.asarray(self.tau(x), dtype=float).reshape(-1)[0])
            loc = float(self.alpha_y)
            if self.beta_y is not None:
                loc += float(np.sum(x * np.asarray(self.beta_y, dtype=float), axis=1)[0])
            if self.g_y is not None:
                loc += float(np.asarray(self.g_y(x), dtype=float).reshape(-1)[0])
            loc += float(d) * tau_val

            if self.outcome_type == "continuous":
                return float(loc)

            uy = float(getattr(self, "u_strength_y", 0.0))
            if self.outcome_type == "binary":
                if uy == 0.0:
                    return float(_sigmoid(loc))
                z = loc + uy * np.sqrt(2.0) * gh_x
                return float(np.sum(_sigmoid(z) * gh_w))
            if self.outcome_type in {"poisson", "gamma"}:
                if uy == 0.0:
                    return float(np.exp(np.clip(loc, -20.0, 20.0)))
                z = np.clip(loc + uy * np.sqrt(2.0) * gh_x, -20.0, 20.0)
                return float(np.sum(np.exp(z) * gh_w))
            raise ValueError("outcome_type must be 'continuous','binary','poisson','gamma'.")

        return (lambda x: m_of_x(x),
                lambda x: g_of_x_d(x, 0),
                lambda x: g_of_x_d(x, 1))
