r"""
IRM estimator consuming CausalData.

Implements cross-fitted nuisance estimation for g0, g1 and m, and supports ATE/ATTE/GATE scores.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import warnings
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, List

from joblib import Parallel, delayed
from sklearn.base import clone, BaseEstimator
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.validation import check_is_fitted
from scipy.stats import norm
try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

from causalis.dgp.causaldata import CausalData
from causalis.data_contracts.causal_estimate import CausalEstimate
from causalis.data_contracts.causal_diagnostic_data import UnconfoundednessDiagnosticData
from causalis.data_contracts.gate_estimate import GateEstimate
from causalis.scenarios.gate.model import estimate_gate_from_irm
from causalis.scenarios.unconfoundedness._utils import (
    _clip_propensity,
    _is_binary,
    _predict_prob_or_value,
    _safe_is_classifier,
)


# IRMResults removed, functionality replaced by CausalEstimate


class IRM(BaseEstimator):
    """Interactive Regression Model (IRM) with cross-fitting using CausalData.

    Parameters
    ----------
    data : CausalData
        Data container with outcome, binary treatment (0/1), and confounders.
    ml_g : estimator
        Learner for E[Y|X,D]. If classifier and Y is binary, predict_proba is used; otherwise predict().
    ml_m : classifier
        Learner for E[D|X] (propensity). Must support predict_proba() or predict() in (0,1).
    n_folds : int, default 5
        Number of cross-fitting folds.
    n_rep : int, default 1
        Number of repetitions of sample splitting. Currently only 1 is supported.
    normalize_ipw : bool, default False
        Whether to normalize IPW terms within the score. Applied to ATE only.
        For ATTE, normalization is ignored to preserve the canonical ATTE EIF.
    trimming_rule : {"truncate"}, default "truncate"
        Trimming approach for propensity scores.
    trimming_threshold : float, default 1e-2
        Threshold for trimming if rule is "truncate".
    weights : Optional[np.ndarray or Dict], default None
        Optional weights.
        - If array of shape (n,), used as ATE weights (w). Assumed E[w|X] = w.
        - If dict, can contain 'weights' (w) and 'weights_bar' (E[w|X]).
        - For ATTE, computed internally (w=D/P(D=1), w_bar=m(X)/P(D=1)).
        Note: If weights depend on treatment or outcome, E[w|X] must be provided for correct sensitivity analysis.
    relative_baseline_min : float, default 1e-8
        Minimum absolute baseline value used for relative effects. If |mu_c| is below this
        threshold, relative estimates are set to NaN with a warning.
    random_state : Optional[int], default None
        Random seed for fold creation.
    n_jobs : int, default 1
        Number of parallel jobs for fold-level cross-fitting.
        Use `-1` to use all available CPUs.
        Practical guidance:
        - Start with `n_jobs=1` for stable, low-contention defaults.
        - Increase to `n_jobs=2/4/-1` when cross-fitting is the bottleneck.
        - If nuisance learners are already multithreaded (e.g. CatBoost with
          `thread_count=-1`), keep `n_jobs=1` or set learner threads to `1`
          to avoid CPU oversubscription.
        - On shared machines, prefer a bounded value (for example `2` or `4`)
          instead of `-1`.
    """

    def __init__(
        self,
        data: Optional[CausalData] = None,
        ml_g: Any = None,
        ml_m: Any = None,
        *,
        n_folds: int = 5,
        n_rep: int = 1,
        normalize_ipw: bool = False,
        trimming_rule: str = "truncate",
        trimming_threshold: float = 1e-2,
        weights: Optional[np.ndarray | Dict[str, Any]] = None,
        relative_baseline_min: float = 1e-8,
        random_state: Optional[int] = None,
        n_jobs: int = 1,
    ) -> None:
        self.data = data
        self.ml_g = ml_g
        self.ml_m = ml_m
        self.n_folds = int(n_folds)
        self.n_rep = int(n_rep)
        self.score = "ATE"
        self.normalize_ipw = bool(normalize_ipw)
        self.trimming_rule = str(trimming_rule)
        self.trimming_threshold = float(trimming_threshold)
        self.weights = weights
        self.relative_baseline_min = float(relative_baseline_min)
        self.random_state = random_state
        self.n_jobs = int(n_jobs)
        self.normalize_ipw_effective_ = bool(normalize_ipw)

        # Initialize default learners if not provided
        if HAS_CATBOOST:
            if self.ml_m is None:
                self.ml_m = CatBoostClassifier(
                    thread_count=-1,
                    verbose=False,
                    allow_writing_files=False,
                    random_seed=self.random_state,
                )
            if self.ml_g is None and self.data is not None:
                y_is_binary = False
                try:
                    df_tmp = self.data.get_df()
                    y_tmp = df_tmp[self.data.outcome.name].to_numpy()
                    y_is_binary = _is_binary(y_tmp)
                except (AttributeError, KeyError, ValueError):
                    pass

                if y_is_binary:
                    self.ml_g = CatBoostClassifier(
                        thread_count=-1,
                        verbose=False,
                        allow_writing_files=False,
                        random_seed=self.random_state,
                    )
                else:
                    self.ml_g = CatBoostRegressor(
                        thread_count=-1,
                        verbose=False,
                        allow_writing_files=False,
                        random_seed=self.random_state,
                    )
        
        # If ml_g is still None and HAS_CATBOOST is True, it means data was not provided.
        # It will be initialized in fit().
        if not np.isfinite(self.trimming_threshold) or not (0.0 <= self.trimming_threshold < 0.5):
            raise ValueError("trimming_threshold must be finite and in [0, 0.5).")
        if self.relative_baseline_min < 0.0:
            raise ValueError("relative_baseline_min must be non-negative.")
        if self.n_jobs == 0 or self.n_jobs < -1:
            raise ValueError("n_jobs must be -1 or a positive integer.")

    # --------- Helpers ---------
    def _check_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        """Check and prepare data for IRM.

        Ensures treatment is binary, confounders are present, and returns relevant arrays.

        Returns
        -------
        X : np.ndarray
            Confounders matrix.
        y : np.ndarray
            Outcome array.
        d : np.ndarray
            Treatment array.
        y_is_binary : bool
            Whether the outcome is binary.
        """
        df = self.data.get_df()
        y = df[self.data.outcome.name].to_numpy(dtype=float, copy=False)
        d = df[self.data.treatment.name].to_numpy(copy=False)
        # Ensure binary 0/1
        if df[self.data.treatment.name].dtype == bool:
            d = d.astype(int)
        if not _is_binary(d):
            raise ValueError("Treatment must be binary 0/1 or boolean.")
        d = d.astype(int)

        x_cols = list(self.data.confounders)
        if len(x_cols) == 0:
            raise ValueError("CausalData must include non-empty confounders.")
        X = df[x_cols].to_numpy(dtype=float, copy=False)

        y_is_binary = _is_binary(y)
        return X, y, d, y_is_binary

    def _initialize_default_learners_for_fit(self, y_is_binary: bool) -> None:
        """Initialize default learners if missing and CatBoost is available."""
        if not HAS_CATBOOST:
            return
        if self.ml_m is None:
            self.ml_m = CatBoostClassifier(
                thread_count=-1,
                verbose=False,
                allow_writing_files=False,
                random_seed=self.random_state,
            )
        if self.ml_g is None:
            if y_is_binary:
                self.ml_g = CatBoostClassifier(
                    thread_count=-1,
                    verbose=False,
                    allow_writing_files=False,
                    random_seed=self.random_state,
                )
            else:
                self.ml_g = CatBoostRegressor(
                    thread_count=-1,
                    verbose=False,
                    allow_writing_files=False,
                    random_seed=self.random_state,
                )

    def _ensure_learners_available(self) -> None:
        """Ensure nuisance learners are configured."""
        if self.ml_g is None or self.ml_m is None:
            raise ValueError("ml_g and ml_m must be provided (either as defaults or in __init__).")

    def _validate_fit_config(self, y_is_binary: bool) -> None:
        """Validate IRM fit-time configuration."""
        # Enforce valid propensity model: must expose predict_proba when classifier
        if _safe_is_classifier(self.ml_m) and not hasattr(self.ml_m, "predict_proba"):
            raise ValueError("ml_m must support predict_proba() to produce valid propensity probabilities.")
        # For binary outcomes, require probabilistic outcome models when using classifiers
        if y_is_binary and _safe_is_classifier(self.ml_g) and not hasattr(self.ml_g, "predict_proba"):
            raise ValueError("Binary outcome: ml_g is a classifier but does not expose predict_proba(). Use a probabilistic classifier or calibrate it.")

        if self.n_rep != 1:
            raise NotImplementedError("IRM currently supports n_rep=1 only.")
        if self.n_folds < 2:
            raise ValueError("n_folds must be at least 2")
        if self.trimming_rule not in {"truncate"}:
            raise ValueError("Only trimming_rule='truncate' is supported")
        if not np.isfinite(self.trimming_threshold) or not (0.0 <= self.trimming_threshold < 0.5):
            raise ValueError("trimming_threshold must be finite and in [0, 0.5).")

    def _fit_outcome_nuisance_for_treatment(
        self,
        *,
        treatment_value: int,
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        d_tr: np.ndarray,
        X_te: np.ndarray,
        y_is_binary: bool,
        empty_group_error: str,
    ) -> np.ndarray:
        """Fit one outcome nuisance model (g0 or g1) and predict on test fold."""
        model_g = clone(self.ml_g)
        mask = (d_tr == treatment_value)
        if not np.any(mask):
            raise RuntimeError(empty_group_error)
        X_g, y_g = X_tr[mask], y_tr[mask]
        if y_is_binary:
            uniq_y = np.unique(y_g)
            if uniq_y.size == 1:
                # Single-class folds can skip fitting while preserving cross-fit independence.
                return np.full(X_te.shape[0], float(uniq_y[0]), dtype=float)
        model_g.fit(X_g, y_g)
        pred = _predict_prob_or_value(model_g, X_te, is_propensity=False)
        if y_is_binary:
            pred = np.clip(pred, 1e-12, 1 - 1e-12)
        return pred

    def _fit_nuisances_for_fold(
        self,
        *,
        fold_id: int,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        d: np.ndarray,
        y_is_binary: bool,
    ) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Fit nuisance models for one fold and return held-out predictions."""
        X_tr, y_tr, d_tr = X[train_idx], y[train_idx], d[train_idx]
        X_te = X[test_idx]

        g0_te = self._fit_outcome_nuisance_for_treatment(
            treatment_value=0,
            X_tr=X_tr,
            y_tr=y_tr,
            d_tr=d_tr,
            X_te=X_te,
            y_is_binary=y_is_binary,
            empty_group_error=(
                "IRM: A CV fold has no controls in the training split. "
                "This violates the IRM nuisance definition. "
                "Consider reducing n_folds or increasing sample size."
            ),
        )

        g1_te = self._fit_outcome_nuisance_for_treatment(
            treatment_value=1,
            X_tr=X_tr,
            y_tr=y_tr,
            d_tr=d_tr,
            X_te=X_te,
            y_is_binary=y_is_binary,
            empty_group_error=(
                "IRM: A CV fold has no treated units in the training split. "
                "This violates the IRM nuisance definition. "
                "Consider reducing n_folds or increasing sample size."
            ),
        )

        model_m = clone(self.ml_m)
        model_m.fit(X_tr, d_tr)
        m_te = _predict_prob_or_value(model_m, X_te, is_propensity=True)

        return fold_id, test_idx, g0_te, g1_te, m_te

    def _cross_fit_nuisances(
        self,
        X: np.ndarray,
        y: np.ndarray,
        d: np.ndarray,
        y_is_binary: bool,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run cross-fitting and return nuisance predictions and fold ids."""
        n = X.shape[0]
        g0_hat = np.full(n, np.nan, dtype=float)
        g1_hat = np.full(n, np.nan, dtype=float)
        m_hat = np.full(n, np.nan, dtype=float)
        folds = np.full(n, -1, dtype=int)

        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        splits = list(skf.split(X, d))

        if self.n_jobs == 1:
            fold_results = [
                self._fit_nuisances_for_fold(
                    fold_id=i,
                    train_idx=train_idx,
                    test_idx=test_idx,
                    X=X,
                    y=y,
                    d=d,
                    y_is_binary=y_is_binary,
                )
                for i, (train_idx, test_idx) in enumerate(splits)
            ]
        else:
            fold_results = Parallel(n_jobs=self.n_jobs, prefer="threads")(
                delayed(self._fit_nuisances_for_fold)(
                    fold_id=i,
                    train_idx=train_idx,
                    test_idx=test_idx,
                    X=X,
                    y=y,
                    d=d,
                    y_is_binary=y_is_binary,
                )
                for i, (train_idx, test_idx) in enumerate(splits)
            )

        for fold_id, test_idx, g0_te, g1_te, m_te in fold_results:
            folds[test_idx] = fold_id
            g0_hat[test_idx] = g0_te
            g1_hat[test_idx] = g1_te
            m_hat[test_idx] = m_te

        return g0_hat, g1_hat, m_hat, folds

    def _store_cross_fitted_predictions(
        self,
        g0_hat: np.ndarray,
        g1_hat: np.ndarray,
        m_hat: np.ndarray,
        folds: np.ndarray,
    ) -> None:
        """Validate and store cross-fitted nuisance predictions."""
        if np.any(np.isnan(m_hat)) or np.any(np.isnan(g0_hat)) or np.any(np.isnan(g1_hat)):
            raise RuntimeError("Cross-fitted predictions contain NaN values.")
        self.folds_ = folds
        self.g0_hat_ = g0_hat
        self.g1_hat_ = g1_hat
        self.m_hat_raw_ = np.asarray(m_hat, dtype=float).copy()
        self.m_hat_ = _clip_propensity(m_hat, self.trimming_threshold)

    def _get_weights(self, n: int, m_hat_adj: Optional[np.ndarray], d: np.ndarray, score: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Compute weights for the IRM score.

        Parameters
        ----------
        n : int
            Number of observations.
        m_hat_adj : Optional[np.ndarray]
            Adjusted propensity scores.
        d : np.ndarray
            Treatment indicators.
        score : Optional[str], default None
            Target estimand. If None, uses self.score.

        Returns
        -------
        w : np.ndarray
            Weights for the outcome terms.
        w_bar : np.ndarray
            Weights for the representer terms.
        """
        if score is None:
            score = self.score
        score = score.upper()

        if self.weights is not None and score != "ATE":
            raise ValueError(f"weights are only supported for score='ATE', but got score='{score}'")

        def _to_1d(arr: Any, *, name: str) -> np.ndarray:
            vec = np.asarray(arr, dtype=float)
            if vec.ndim == 1:
                pass
            elif vec.ndim == 2 and 1 in vec.shape:
                vec = vec.reshape(-1)
            else:
                raise ValueError(f"{name} must be 1D with shape (n,), got shape {vec.shape}.")
            if vec.shape[0] != n:
                raise ValueError(f"{name} must have shape (n,) with n={n}, got shape {vec.shape}.")
            if not np.all(np.isfinite(vec)):
                raise ValueError(f"{name} must contain only finite values.")
            return vec

        # Standard ATE
        if score == "ATE":
            if self.weights is None:
                w = np.ones(n, dtype=float)
            elif isinstance(self.weights, np.ndarray):
                w = _to_1d(self.weights, name="weights")
            elif isinstance(self.weights, dict):
                if "weights" not in self.weights:
                    raise ValueError("weights dict must contain key 'weights'.")
                w = _to_1d(self.weights["weights"], name="weights['weights']")
            else:
                raise TypeError("weights must be None, np.ndarray, or dict")
            # By default, use w as its own conditional expectation proxy in EIF terms.
            w_bar = w
            if isinstance(self.weights, dict) and "weights_bar" in self.weights:
                w_bar_arr = np.asarray(self.weights["weights_bar"], dtype=float)
                if w_bar_arr.ndim == 2:
                    if w_bar_arr.shape[0] == n and w_bar_arr.shape[1] >= 1:
                        if w_bar_arr.shape[1] > 1:
                            warnings.warn(
                                "weights['weights_bar'] has multiple columns; using the first column.",
                                RuntimeWarning,
                            )
                        w_bar = w_bar_arr[:, 0]
                    elif w_bar_arr.shape == (1, n):
                        w_bar = w_bar_arr.reshape(-1)
                    else:
                        raise ValueError(
                            "weights['weights_bar'] must be shape (n,), (n,1), (1,n), or (n,r) for r>=1."
                        )
                else:
                    w_bar = _to_1d(w_bar_arr, name="weights['weights_bar']")
                if not np.all(np.isfinite(w_bar)):
                    raise ValueError("weights['weights_bar'] must contain only finite values.")
        # ATTE requires m_hat
        elif score == "ATTE":
            if m_hat_adj is None:
                raise ValueError("m_hat required for ATTE weights computation")
            w = d.astype(float)
            w_bar = m_hat_adj.astype(float)
        else:
            raise ValueError("score must be 'ATE' or 'ATTE'")

        # Normalize by E_n[w] so effect, orthogonal score, and sensitivity pieces
        # all use the same weight scale.
        mean_w = float(np.mean(w))
        if not np.isfinite(mean_w) or mean_w <= 1e-12:
            raise ValueError("weights must have a strictly positive finite mean.")
        w = w / mean_w
        w_bar = w_bar / mean_w
        return w, w_bar

    def _use_normalized_ipw(self, score: Optional[str] = None, *, warn: bool = False) -> bool:
        """Return whether Hájek normalization is active for a given score."""
        score_u = (self.score if score is None else str(score)).upper()
        if self.normalize_ipw and score_u == "ATTE":
            if warn:
                warnings.warn(
                    "normalize_ipw=True is ignored for ATTE to preserve the canonical ATTE EIF.",
                    RuntimeWarning,
                )
            return False
        return bool(self.normalize_ipw)

    def _normalize_ipw_terms(
        self,
        d: np.ndarray,
        m_hat: np.ndarray,
        score: Optional[str] = None,
        *,
        warn: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute and optionally normalize IPW terms.

        Parameters
        ----------
        d : np.ndarray
            Treatment indicators.
        m_hat : np.ndarray
            Propensity scores.

        Returns
        -------
        h1 : np.ndarray
            IPW term for treated units (d / m_hat).
        h0 : np.ndarray
            IPW term for control units ((1 - d) / (1 - m_hat)).
        """
        # Horvitz-Thompson terms (unnormalized).
        h1 = d / m_hat
        h0 = (1 - d) / (1 - m_hat)
        if self._use_normalized_ipw(score=score, warn=warn):
            # Hájek normalization: force empirical means of treated/control IPW terms to 1.
            h1_mean = np.mean(h1)
            h0_mean = np.mean(h0)
            # Avoid division by zero
            h1 = h1 / (h1_mean if h1_mean != 0 else 1.0)
            h0 = h0 / (h0_mean if h0_mean != 0 else 1.0)
        return h1, h0

    # --------- API ---------
    def fit(self, data: Optional[CausalData] = None) -> "IRM":
        """Fit nuisance models via cross-fitting.

        Parameters
        ----------
        data : Optional[CausalData], default None
            CausalData container. If None, uses self.data.

        Returns
        -------
        self : IRM
            Fitted estimator.
        """
        if data is not None:
            self.data = data
        if self.data is None:
            raise ValueError("Model must be provided with CausalData either in __init__ or in .fit(data_contracts).")
        X, y, d, y_is_binary = self._check_data()

        # Initialize default learners if not provided and data is now available
        self._initialize_default_learners_for_fit(y_is_binary=y_is_binary)
        self._ensure_learners_available()

        # Cache for sensitivity analysis and effect calculation
        self._X = X.copy()
        self._y = y.copy()
        self._d = d.copy()
        self._validate_fit_config(y_is_binary=y_is_binary)

        g0_hat, g1_hat, m_hat, folds = self._cross_fit_nuisances(X=X, y=y, d=d, y_is_binary=y_is_binary)
        self._store_cross_fitted_predictions(g0_hat=g0_hat, g1_hat=g1_hat, m_hat=m_hat, folds=folds)

        return self

    def _validate_estimate_request(self, score: str, alpha: float) -> str:
        """Validate estimate() arguments and return normalized score."""
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be in (0,1)")
        score_u = str(score).upper()
        if score_u == "CATE":
            raise NotImplementedError("score='CATE' is not supported.")
        if score_u not in {"ATE", "ATTE", "GATE"}:
            raise ValueError("score must be 'ATE', 'ATTE', or 'GATE'")
        return score_u

    def _estimate_inference_approx_flags(self, score: str, normalize_ipw_effective: bool) -> Dict[str, bool]:
        """Return flags for inference paths that use ratio-style approximations."""
        return {
            "se_approx_hajek": bool(score == "ATE" and normalize_ipw_effective),
            "se_approx_weight_norm": bool(score == "ATE" and self.weights is not None),
        }

    def _warn_if_inference_is_approximate(self, approx_flags: Dict[str, bool]) -> None:
        """Emit warnings when inference relies on ratio-normalization approximations."""
        if approx_flags.get("se_approx_hajek", False):
            warnings.warn(
                "normalize_ipw=True (Hájek) uses approximate SE/IF; "
                "denominator variability is treated as fixed.",
                RuntimeWarning,
            )
        if approx_flags.get("se_approx_weight_norm", False):
            warnings.warn(
                "ATE custom weights are normalized by sample mean; "
                "SE/IF treats this normalization as fixed (approximate).",
                RuntimeWarning,
            )

    def _compute_psi_terms(
        self,
        *,
        y: np.ndarray,
        d: np.ndarray,
        g0_hat: np.ndarray,
        g1_hat: np.ndarray,
        m_hat: np.ndarray,
        score: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute score ingredients (weights and psi_a/psi_b terms)."""
        n = len(y)
        u0 = y - g0_hat
        u1 = y - g1_hat
        h1, h0 = self._normalize_ipw_terms(d, m_hat, score=score)
        w, w_bar = self._get_weights(n, m_hat, d, score=score)
        # Orthogonal score part for the target moment E[psi_a * theta + psi_b] = 0.
        psi_b = w * (g1_hat - g0_hat) + w_bar * (u1 * h1 - u0 * h0)

        if score == "ATE":
            # ATE moment derivative w.r.t. theta is constant -1.
            psi_a = -np.ones(n)
        elif score == "ATTE":
            # ATT(E) derivative is weighted by the target-treatment indicator weight.
            psi_a = -w
        else:
            raise ValueError("score must be 'ATE' or 'ATTE'")

        return w, w_bar, psi_a, psi_b

    def _solve_moment_equation(
        self,
        *,
        psi_a: np.ndarray,
        psi_b: np.ndarray,
        alpha: float,
    ) -> Tuple[float, np.ndarray, float, float, float, float, float, float]:
        """Solve the moment equation and compute inference statistics."""
        n = len(psi_a)
        # Jacobian (Neyman score derivative) for the one-dimensional moment.
        J = float(np.mean(psi_a))

        if abs(J) < 1e-16:
            theta_hat = np.nan
            IF = np.zeros(n)
            se = np.nan
        else:
            # Closed-form root of E_n[psi_a * theta + psi_b] = 0.
            theta_hat = -float(np.mean(psi_b) / J)
            psi_res = psi_b + psi_a * theta_hat
            # Estimated influence function and its plug-in sandwich variance.
            IF = -psi_res / J
            var = float(np.var(IF, ddof=1)) / n
            se = float(np.sqrt(max(var, 0.0)))

        t_stat = theta_hat / se if se > 0 else np.nan
        pval = 2 * (1 - norm.cdf(abs(t_stat))) if np.isfinite(t_stat) else np.nan
        z = norm.ppf(1 - alpha / 2.0)
        ci_low = theta_hat - z * se
        ci_high = theta_hat + z * se

        return theta_hat, IF, se, t_stat, pval, ci_low, ci_high, z

    def _cache_estimate_core(
        self,
        *,
        theta_hat: float,
        se: float,
        IF: np.ndarray,
        psi_a: np.ndarray,
        psi_b: np.ndarray,
    ) -> None:
        """Cache core estimate quantities used by diagnostics/sensitivity."""
        self.coef_ = np.array([theta_hat])
        self.se_ = np.array([se])
        self.psi_ = IF
        self.psi_a_ = psi_a
        self.psi_b_ = psi_b

    def _compute_relative_effect_stats(
        self,
        *,
        theta_hat: float,
        IF: np.ndarray,
        y: np.ndarray,
        d: np.ndarray,
        w: np.ndarray,
        w_bar: np.ndarray,
        g0_hat: np.ndarray,
        m_hat: np.ndarray,
        score: str,
        z: float,
    ) -> Tuple[float, float, float, float, float]:
        """Compute relative effect and delta-method interval."""
        n = len(w)
        u0 = y - g0_hat
        _, h0 = self._normalize_ipw_terms(d, m_hat, score=score, warn=False)
        # Orthogonal signal for baseline E[w * Y(0)].
        psi_mu_c = w * g0_hat + w_bar * (u0 * h0)

        mu_c = float(np.mean(psi_mu_c))
        mu_c_var = float(np.var(psi_mu_c, ddof=1)) / n if n > 1 else 0.0
        mu_c_se = float(np.sqrt(max(mu_c_var, 0.0)))
        tau_rel = np.nan
        ci_low_rel = np.nan
        ci_high_rel = np.nan
        se_rel = np.nan

        baseline_too_small = abs(mu_c) < self.relative_baseline_min
        baseline_low_signal = np.isfinite(mu_c_se) and mu_c_se > 0.0 and abs(mu_c) < z * mu_c_se

        if np.isfinite(mu_c) and not (baseline_too_small or baseline_low_signal):
            tau_rel = 100.0 * theta_hat / mu_c
            IF_mu = psi_mu_c - mu_c
            with np.errstate(divide="ignore", invalid="ignore"):
                # Delta-method IF for tau_rel = 100 * theta / mu_c.
                IF_rel = 100.0 * (IF / mu_c - (theta_hat * IF_mu) / (mu_c ** 2))
            var_rel = float(np.var(IF_rel, ddof=1)) / n
            se_rel = float(np.sqrt(max(var_rel, 0.0)))
            ci_low_rel = tau_rel - z * se_rel
            ci_high_rel = tau_rel + z * se_rel
            if ci_low_rel > ci_high_rel:
                ci_low_rel, ci_high_rel = ci_high_rel, ci_low_rel
        else:
            reasons = []
            if not np.isfinite(mu_c):
                reasons.append("is not finite")
            if baseline_too_small:
                reasons.append(f"is below relative_baseline_min={self.relative_baseline_min:.3e}")
            if baseline_low_signal:
                reasons.append(f"is within {z:.2f} SE of 0 (SE={mu_c_se:.3e})")
            reason_str = "; ".join(reasons) if reasons else "is too small"
            warnings.warn(
                f"Relative effect baseline |mu_c|={abs(mu_c):.3e} {reason_str}. "
                "Relative estimates are set to NaN.",
                RuntimeWarning,
            )

        return mu_c, tau_rel, ci_low_rel, ci_high_rel, se_rel

    def _build_estimate_diagnostic_data(
        self,
        *,
        diagnostic_data: bool,
        y: np.ndarray,
        d: np.ndarray,
        g0_hat: np.ndarray,
        g1_hat: np.ndarray,
        m_hat: np.ndarray,
        w: np.ndarray,
        w_bar: np.ndarray,
        IF: np.ndarray,
        psi_b: np.ndarray,
        score: str,
        normalize_ipw_effective: bool,
    ) -> Optional[UnconfoundednessDiagnosticData]:
        """Build optional diagnostics payload for CausalEstimate."""
        if not diagnostic_data:
            return None

        sens_elements = self._sensitivity_element_est(
            y=y, d=d, g0=g0_hat, g1=g1_hat, m_hat=m_hat, w=w, w_bar=w_bar, psi=IF
        )

        score_plot_cache = self._build_score_plot_cache(
            d=d,
            m_hat=m_hat,
            psi=IF,
            score=score,
            normalize_ipw_effective=normalize_ipw_effective,
        )
        residual_plot_cache = self._build_residual_plot_cache(
            y=y,
            d=d,
            g0_hat=g0_hat,
            g1_hat=g1_hat,
            m_hat=m_hat,
        )

        diag = UnconfoundednessDiagnosticData(
            m_hat=m_hat,
            m_hat_raw=getattr(self, "m_hat_raw_", None),
            d=d,
            y=y,
            x=np.asarray(
                getattr(
                    self,
                    "_X",
                    self.data.get_df()[list(self.data.confounders)].to_numpy(dtype=float, copy=False),
                ),
                dtype=float,
            ),
            g0_hat=g0_hat,
            g1_hat=g1_hat,
            w=w,
            w_bar=w_bar,
            psi_b=psi_b,
            folds=self.folds_,
            trimming_threshold=self.trimming_threshold,
            normalize_ipw=normalize_ipw_effective,
            score=score,
            score_plot_cache=score_plot_cache,
            residual_plot_cache=residual_plot_cache,
            **sens_elements
        )
        diag._model = self
        return diag

    def _build_score_plot_cache(
        self,
        *,
        d: np.ndarray,
        m_hat: np.ndarray,
        psi: np.ndarray,
        score: str,
        normalize_ipw_effective: bool,
    ) -> Dict[str, Any]:
        """Build cached arrays used by score influence plots."""
        m_clipped = np.clip(m_hat, self.trimming_threshold, 1.0 - self.trimming_threshold)
        if score == "ATE":
            ipw_t, ipw_c = self._normalize_ipw_terms(d, m_clipped, score=score, warn=False)
            ipw_t_label = r"$D/m$"
            ipw_c_label = r"$(1-D)/(1-m)$"
        else:
            p_treated = float(np.mean(d))
            ipw_t = d / (p_treated + 1e-12)
            ipw_c = ((1.0 - d) * (m_clipped / (1.0 - m_clipped))) / (p_treated + 1e-12)
            ipw_t_label = r"$D/\mathbb{E}[D]$"
            ipw_c_label = r"$(1-D)\cdot m/(1-m)/\mathbb{E}[D]$"

        return {
            "score": str(score),
            "trimming_threshold": float(self.trimming_threshold),
            "normalize_ipw": bool(normalize_ipw_effective),
            "d": np.asarray(d, dtype=float).ravel(),
            "m_clipped": np.asarray(m_clipped, dtype=float).ravel(),
            "psi": np.asarray(psi, dtype=float).ravel(),
            "ipw_t": np.asarray(ipw_t, dtype=float).ravel(),
            "ipw_c": np.asarray(ipw_c, dtype=float).ravel(),
            "ipw_t_label": ipw_t_label,
            "ipw_c_label": ipw_c_label,
        }

    def _build_residual_plot_cache(
        self,
        *,
        y: np.ndarray,
        d: np.ndarray,
        g0_hat: np.ndarray,
        g1_hat: np.ndarray,
        m_hat: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Build cached arrays used by residual diagnostic plots."""
        return {
            "y": np.asarray(y, dtype=float).ravel(),
            "d": np.asarray(d, dtype=float).ravel(),
            "g0": np.asarray(g0_hat, dtype=float).ravel(),
            "g1": np.asarray(g1_hat, dtype=float).ravel(),
            "m": np.asarray(m_hat, dtype=float).ravel(),
        }

    def _build_causal_estimate(
        self,
        *,
        score: str,
        alpha: float,
        theta_hat: float,
        se: float,
        t_stat: float,
        pval: float,
        ci_low: float,
        ci_high: float,
        tau_rel: float,
        ci_low_rel: float,
        ci_high_rel: float,
        y: np.ndarray,
        d: np.ndarray,
        normalize_ipw_effective: bool,
        approx_flags: Dict[str, bool],
        diag: Optional[UnconfoundednessDiagnosticData],
    ) -> CausalEstimate:
        """Build the CausalEstimate object."""
        treatment_mean = float(np.mean(y[d == 1])) if np.any(d == 1) else np.nan
        control_mean = float(np.mean(y[d == 0])) if np.any(d == 0) else np.nan

        return CausalEstimate(
            estimand=score,
            model="IRM",
            model_options={
                "n_folds": self.n_folds,
                "n_rep": self.n_rep,
                "normalize_ipw": normalize_ipw_effective,
                "trimming_rule": self.trimming_rule,
                "trimming_threshold": self.trimming_threshold,
                "random_state": self.random_state,
                "n_jobs": self.n_jobs,
                "std_error": se,
                "t_stat": t_stat,
                "se_approx_hajek": bool(approx_flags.get("se_approx_hajek", False)),
                "se_approx_weight_norm": bool(approx_flags.get("se_approx_weight_norm", False)),
            },
            value=theta_hat,
            ci_upper_absolute=ci_high,
            ci_lower_absolute=ci_low,
            value_relative=tau_rel,
            ci_upper_relative=ci_high_rel,
            ci_lower_relative=ci_low_rel,
            alpha=alpha,
            p_value=pval,
            is_significant=bool(pval < alpha) if np.isfinite(pval) else False,
            n_treated=int(np.sum(d == 1)),
            n_control=int(np.sum(d == 0)),
            treatment_mean=treatment_mean,
            control_mean=control_mean,
            outcome=self.data.outcome.name,
            treatment=self.data.treatment.name,
            confounders=list(self.data.confounders),
            time=datetime.now().strftime("%Y-%m-%d"),
            diagnostic_data=diag,
        )

    def _update_estimate_state(
        self,
        *,
        theta_hat: float,
        se: float,
        t_stat: float,
        pval: float,
        ci_low: float,
        ci_high: float,
        results: CausalEstimate,
    ) -> None:
        """Finalize and cache estimate state used by public accessors."""
        self.coef_ = np.array([theta_hat])
        self.se_ = np.array([se])
        self.t_stat_ = np.array([t_stat])
        self.pval_ = np.array([pval])
        self.confint_ = np.array([[ci_low, ci_high]])
        self.summary_ = results.summary()

    def estimate(
        self,
        score: str = "ATE",
        alpha: float = 0.05,
        diagnostic_data: bool = True,
        groups: Optional[pd.DataFrame | pd.Series] = None,
        cov_type: str = "HC3",
        cov_kwds: Optional[Dict[str, Any]] = None,
    ) -> CausalEstimate | GateEstimate:
        """Compute treatment effects using stored nuisance predictions.

        Parameters
        ----------
        score : {"ATE", "ATTE", "GATE"}, default "ATE"
            Target estimand.
        alpha : float, default 0.05
            Significance level for intervals.
        diagnostic_data : bool, default True
            Whether to include diagnostic data_contracts in the result.
        groups : Optional[pd.DataFrame | pd.Series], default None
            Group labels/indicators for ``score="GATE"``.
            If None, fallback to ``self.data.gate_groups`` when present.
        cov_type : {"HC0", "HC1", "HC2", "HC3"}, default "HC3"
            Robust covariance type for ``score="GATE"`` inference.
        cov_kwds : Optional[Dict[str, Any]], default None
            Additional covariance keyword arguments passed to statsmodels for ``score="GATE"``.

        Returns
        -------
        CausalEstimate or GateEstimate
            Result container for the estimated effect.
        """
        check_is_fitted(self, attributes=["g0_hat_", "g1_hat_", "m_hat_"])
        score = self._validate_estimate_request(score=score, alpha=alpha)
        self.score = score

        if score == "GATE":
            return estimate_gate_from_irm(
                irm_model=self,
                groups=groups,
                alpha=alpha,
                cov_type=cov_type,
                cov_kwds=cov_kwds,
                diagnostic_data=diagnostic_data,
            )

        # For ATTE we intentionally disable Hájek even if normalize_ipw=True.
        normalize_ipw_effective = self._use_normalized_ipw(score=score, warn=False)
        self.normalize_ipw_effective_ = normalize_ipw_effective
        # Track known finite-sample approximation paths in inference metadata.
        approx_flags = self._estimate_inference_approx_flags(score=score, normalize_ipw_effective=normalize_ipw_effective)
        self._warn_if_inference_is_approximate(approx_flags)

        y, d = self._y, self._d
        g0_hat, g1_hat, m_hat = self.g0_hat_, self.g1_hat_, self.m_hat_

        w, w_bar, psi_a, psi_b = self._compute_psi_terms(
            y=y, d=d, g0_hat=g0_hat, g1_hat=g1_hat, m_hat=m_hat, score=score
        )
        theta_hat, IF, se, t_stat, pval, ci_low, ci_high, z = self._solve_moment_equation(
            psi_a=psi_a, psi_b=psi_b, alpha=alpha
        )
        self._cache_estimate_core(theta_hat=theta_hat, se=se, IF=IF, psi_a=psi_a, psi_b=psi_b)

        mu_c, tau_rel, ci_low_rel, ci_high_rel, se_rel = self._compute_relative_effect_stats(
            theta_hat=theta_hat, IF=IF, y=y, d=d, w=w, w_bar=w_bar, g0_hat=g0_hat, m_hat=m_hat, score=score, z=z
        )
        self.mu_c_ = mu_c
        self.se_relative_ = np.array([se_rel])
        self.confint_relative_ = np.array([[ci_low_rel, ci_high_rel]])

        diag = self._build_estimate_diagnostic_data(
            diagnostic_data=diagnostic_data,
            y=y,
            d=d,
            g0_hat=g0_hat,
            g1_hat=g1_hat,
            m_hat=m_hat,
            w=w,
            w_bar=w_bar,
            IF=IF,
            psi_b=psi_b,
            score=score,
            normalize_ipw_effective=normalize_ipw_effective,
        )

        results = self._build_causal_estimate(
            score=score,
            alpha=alpha,
            theta_hat=theta_hat,
            se=se,
            t_stat=t_stat,
            pval=pval,
            ci_low=ci_low,
            ci_high=ci_high,
            tau_rel=tau_rel,
            ci_low_rel=ci_low_rel,
            ci_high_rel=ci_high_rel,
            y=y,
            d=d,
            normalize_ipw_effective=normalize_ipw_effective,
            approx_flags=approx_flags,
            diag=diag,
        )
        self._update_estimate_state(
            theta_hat=theta_hat,
            se=se,
            t_stat=t_stat,
            pval=pval,
            ci_low=ci_low,
            ci_high=ci_high,
            results=results,
        )

        return results

    @property
    def diagnostics_(self) -> Dict[str, Any]:
        """Return diagnostic data.

        Returns
        -------
        dict
            Dictionary containing 'm_hat', 'g0_hat', 'g1_hat', and 'folds'.
        """
        check_is_fitted(self, attributes=["m_hat_", "g0_hat_", "g1_hat_"])
        return {
            "m_hat": self.m_hat_,
            "m_hat_raw": getattr(self, "m_hat_raw_", None),
            "g0_hat": self.g0_hat_,
            "g1_hat": self.g1_hat_,
            "folds": self.folds_,
        }

    # Convenience properties
    @property
    def coef(self) -> np.ndarray:
        """Return the estimated coefficient.

        Returns
        -------
        np.ndarray
            The estimated coefficient.
        """
        check_is_fitted(self, attributes=["coef_"])
        return self.coef_

    @property
    def se(self) -> np.ndarray:
        """Return the standard error of the estimate.

        Returns
        -------
        np.ndarray
            The standard error.
        """
        check_is_fitted(self, attributes=["se_"])
        return self.se_

    @property
    def pvalues(self) -> np.ndarray:
        """Return the p-values for the estimate.

        Returns
        -------
        np.ndarray
            The p-values.
        """
        check_is_fitted(self, attributes=["pval_"])
        return self.pval_

    @property
    def summary(self) -> pd.DataFrame:
        """Return a summary DataFrame of the results.

        Returns
        -------
        pd.DataFrame
            The results summary.
        """
        check_is_fitted(self, attributes=["summary_"])
        return self.summary_

    @property
    def orth_signal(self) -> np.ndarray:
        """Return the cross-fitted orthogonal signal (psi_b).

        Returns
        -------
        np.ndarray
            The orthogonal signal.
        """
        check_is_fitted(self, attributes=["psi_b_"])
        return self.psi_b_

    def gate(
        self,
        groups: pd.DataFrame | pd.Series,
        alpha: float = 0.05,
        cov_type: str = "HC3",
        cov_kwds: Optional[Dict[str, Any]] = None,
        diagnostic_data: bool = True,
    ) -> GateEstimate:
        """Convenience wrapper for ``estimate(score=\"GATE\", ...)``."""
        return self.estimate(
            score="GATE",
            groups=groups,
            alpha=alpha,
            cov_type=cov_type,
            cov_kwds=cov_kwds,
            diagnostic_data=diagnostic_data,
        )

    # --------- Sensitivity ---------
    def _sensitivity_element_est(
        self,
        y: Optional[np.ndarray] = None,
        d: Optional[np.ndarray] = None,
        g0: Optional[np.ndarray] = None,
        g1: Optional[np.ndarray] = None,
        m_hat: Optional[np.ndarray] = None,
        w: Optional[np.ndarray] = None,
        w_bar: Optional[np.ndarray] = None,
        psi: Optional[np.ndarray] = None,
    ) -> dict:
        """Compute elements needed for sensitivity bias bounds.

        Mirrors a standard IRM sensitivity element computation using fitted nuisances.
        Requires fit() to have been called.

        Parameters
        ----------
        y : Optional[np.ndarray], default None
            Outcomes. If None, uses cached outcomes.
        d : Optional[np.ndarray], default None
            Treatment indicators. If None, uses cached indicators.
        g0 : Optional[np.ndarray], default None
            Outcome predictions under control. If None, uses fitted g0_hat_.
        g1 : Optional[np.ndarray], default None
            Outcome predictions under treatment. If None, uses fitted g1_hat_.
        m_hat : Optional[np.ndarray], default None
            Propensity scores. If None, uses fitted m_hat_.
        w : Optional[np.ndarray], default None
            Outcome weights. If None, computed internally.
        w_bar : Optional[np.ndarray], default None
            Representer weights. If None, computed internally.
        psi : Optional[np.ndarray], default None
            Score values. If None, uses fitted psi_.

        Returns
        -------
        dict
            Sensitivity elements including 'sigma2', 'nu2', 'psi_sigma2', 'psi_nu2',
            'riesz_rep', 'm_alpha', and 'psi'.
        """
        if any(getattr(self, attr) is None for attr in ["g0_hat_", "g1_hat_", "m_hat_"]):
            raise RuntimeError("IRM model must be fitted before sensitivity analysis.")

        if y is None:
            y = self._y
        if d is None:
            d = self._d

        if y is None or d is None:
            # fallback to current data_contracts
            df = self.data.get_df()
            y = df[self.data.outcome.name].to_numpy(dtype=float)
            d = df[self.data.treatment.name].to_numpy(dtype=int)

        if m_hat is None:
            m_hat = np.asarray(self.m_hat_, dtype=float)
        if g0 is None:
            g0 = np.asarray(self.g0_hat_, dtype=float)
        if g1 is None:
            g1 = np.asarray(self.g1_hat_, dtype=float)

        if w is None or w_bar is None:
            w, w_bar = self._get_weights(n=len(y), m_hat_adj=m_hat, d=d)

        if psi is None:
            psi = self.psi_

        # Residual variance component used in sensitivity bounds.
        sigma2_score_element = np.square(y - d * g1 - (1.0 - d) * g0)
        sigma2 = float(np.mean(sigma2_score_element))
        psi_sigma2 = sigma2_score_element - sigma2

        # Riesz representer building blocks from propensity terms.
        with np.errstate(divide='ignore', invalid='ignore'):
            inv_m = 1.0 / m_hat
            inv_1m = 1.0 / (1.0 - m_hat)
        # If IPW terms are normalized in the score, mirror that normalization here.
        if self._use_normalized_ipw(score=getattr(self, "score", "ATE"), warn=False):
            h1_raw = d * inv_m
            h0_raw = (1.0 - d) * inv_1m
            c1 = float(np.mean(h1_raw))
            c0 = float(np.mean(h0_raw))
            if not (np.isfinite(c1) and abs(c1) > 1e-12):
                c1 = 1.0
            if not (np.isfinite(c0) and abs(c0) > 1e-12):
                c0 = 1.0
            inv_m = inv_m / c1
            inv_1m = inv_1m / c0
        # rr is the Riesz representer; m_alpha enters the nu^2 sensitivity term.
        m_alpha = (w_bar ** 2) * (inv_m + inv_1m)
        rr = w_bar * (d * inv_m - (1.0 - d) * inv_1m)

        # nu^2 component from the score decomposition in the sensitivity bound.
        nu2_score_element = 2.0 * m_alpha - np.square(rr)
        nu2 = float(np.mean(nu2_score_element))
        psi_nu2 = nu2_score_element - nu2

        return {
            "sigma2": sigma2,
            "nu2": nu2,
            "psi_sigma2": psi_sigma2,
            "psi_nu2": psi_nu2,
            "riesz_rep": rr,
            "m_alpha": m_alpha,
            "psi": psi,
        }

    def sensitivity_analysis(self, r2_y: float, r2_d: float, rho: float = 1.0, H0: float = 0.0, alpha: float = 0.05) -> "IRM":
        """Compute a sensitivity analysis following Chernozhukov et al. (2022).
        
        Parameters
        ----------
        r2_y : float
            Sensitivity parameter for outcome equation (R^2 form, R_Y^2; converted to odds form internally).
        r2_d : float
            Sensitivity parameter for treatment equation (R^2 form, R_D^2).
        rho : float, default 1.0
            Correlation between unobserved components.
        H0 : float, default 0.0
            Null hypothesis for robustness values.
        alpha : float, default 0.05
            Significance level for CI bounds.
        """
        from causalis.scenarios.unconfoundedness.refutation.unconfoundedness.sensitivity import (
            sensitivity_analysis as sa_fn,
            get_sensitivity_summary
        )
        check_is_fitted(self, attributes=["coef_", "se_", "psi_"])

        # Execute sensitivity analysis using the centralized module logic
        res = sa_fn(self, r2_y=r2_y, r2_d=r2_d, rho=rho, H0=H0, alpha=alpha)

        # Cache the summary string for display
        self.sensitivity_summary = get_sensitivity_summary({"model": self, "bias_aware": res})

        return self

    def confint(self, alpha: float = 0.05) -> pd.DataFrame:
        """Compute confidence intervals for the estimated coefficient.

        Parameters
        ----------
        alpha : float, default 0.05
            Significance level.

        Returns
        -------
        pd.DataFrame
            DataFrame with confidence intervals.
        """
        check_is_fitted(self, attributes=["coef_", "se_"])
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be in (0,1)")
        z = norm.ppf(1 - alpha / 2.0)
        ci_low = self.coef_[0] - z * self.se_[0]
        ci_high = self.coef_[0] + z * self.se_[0]
        return pd.DataFrame(
            {f"{alpha/2*100:.1f} %": [ci_low], f"{(1-alpha/2)*100:.1f} %": [ci_high]},
            index=[self.data.treatment.name],
        )
