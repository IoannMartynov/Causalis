r"""
IRM estimator consuming CausalData.

Implements cross-fitted nuisance estimation for g0, g1 and m, and supports ATE/ATTE/GATE/GATET scores.
https://github.com/DoubleML/doubleml-for-py/blob/main/doubleml/irm/irm.py
"""

from __future__ import annotations

import hashlib
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
from causalis.data_contracts.causal_diagnostic_data import (
    UnconfoundednessDiagnosticData,
)
from causalis.data_contracts.causal_estimate import CausalEstimate
from causalis.data_contracts.gate_estimate import GateEstimate
from causalis.scenarios.gate.model import (
    estimate_gate_from_irm,
    estimate_gatet_from_irm,
)
from causalis.scenarios.unconfoundedness._diagnostic_utils import (
    _build_irm_estimate_diagnostic_data,
)
from causalis.scenarios.unconfoundedness._score_utils import (
    _compute_ipw_components as _compute_irm_ipw_components,
    _normalize_ipw_terms as _normalize_irm_ipw_terms,
    _resolve_irm_weights,
    _use_normalized_ipw as _use_normalized_irm_ipw,
)
from causalis.scenarios.unconfoundedness._utils import (
    _apply_overlap_policy,
    _is_binary,
    _predict_prob_or_value,
    _safe_is_classifier,
    _validate_overlap_config,
)


# IRMResults removed, functionality replaced by CausalEstimate


class IRM(BaseEstimator):
    r"""Interactive Regression Model (IRM) with cross-fitting using CausalData.

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
    overlap_policy : {"clip", "drop"}, default "clip"
        How to handle propensity scores near 0 or 1. ``"clip"`` bounds
        propensity scores to ``[overlap_threshold, 1 - overlap_threshold]``.
        ``"drop"`` removes observations outside that interval after
        cross-fitted propensities are estimated.
    overlap_threshold : float, default 1e-2
        Boundary used by the overlap policy. Must be finite and in ``[0, 0.5)``.
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
    store_diagnostics : bool, default True
        Whether to retain raw fit-time arrays and diagnostic-only artifacts on the
        fitted model. Set to ``False`` for a lighter-weight estimator that still
        supports effect estimation, while only retaining immutable outcome and
        treatment snapshots. In lightweight mode the estimator no longer keeps
        the confounder matrix, raw propensities, fold assignments, or compact
        native feature-importance diagnostics in memory after ``fit()``.
        When enabled, supported native feature-importance sources are learner
        ``feature_importances_``, ``coef_``, and CatBoost
        ``get_feature_importance()``.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    >>> from causalis.dgp import obs_linear_26_dataset
    >>> from causalis.scenarios.unconfoundedness.model import IRM
    >>> data = obs_linear_26_dataset(
    ...     n=1000,
    ...     seed=3141,
    ...     include_oracle=False,
    ...     return_causal_data=True,
    ... )
    >>> ml_g = RandomForestRegressor(
    ...     n_estimators=200,
    ...     max_depth=6,
    ...     min_samples_leaf=5,
    ...     random_state=3141,
    ... )
    >>> ml_m = RandomForestClassifier(
    ...     n_estimators=200,
    ...     max_depth=6,
    ...     min_samples_leaf=5,
    ...     random_state=3141,
    ... )
    >>> irm = IRM(data=data, ml_g=ml_g, ml_m=ml_m, n_folds=3, random_state=3141)
    >>> ate = irm.fit().estimate(score="ATE")
    >>> ate.summary()  # doctest: +SKIP
    >>> atte = irm.estimate(score="ATTE")
    >>> atte.value  # doctest: +SKIP

    Notes
    -----
    The IRM model targets binary-treatment causal effects under unconfoundedness.
    Let :math:`W = (Y, D, X)` with :math:`D \in \{0, 1\}` and define

    .. math::

        g_0(d, x) = \mathbb{E}[Y \mid D=d, X=x], \qquad
        m_0(x) = \mathbb{P}(D=1 \mid X=x).

    Under conditional ignorability and overlap,

    .. math::

        (Y(0), Y(1)) \perp D \mid X, \qquad 0 < m_0(X) < 1 \ \text{a.s.},

    the target functionals are identified as

    .. math::

        \theta_0^{ATE} = \mathbb{E}[g_0(1, X) - g_0(0, X)]

    and

    .. math::

        \theta_0^{ATTE} = \mathbb{E}[g_0(1, X) - g_0(0, X) \mid D=1].

    This implementation cross-fits three nuisance objects:
    :math:`\hat g_1(x) \approx \mathbb{E}[Y \mid D=1, X=x]`,
    :math:`\hat g_0(x) \approx \mathbb{E}[Y \mid D=0, X=x]`, and
    :math:`\hat m(x) \approx \mathbb{P}(D=1 \mid X=x)`.
    By default, propensities are clipped via

    .. math::

        \tilde m(x) = \min\{1-\varepsilon, \max(\hat m(x), \varepsilon)\},

    where :math:`\varepsilon =` ``overlap_threshold``. With
    ``overlap_policy="drop"``, rows with raw cross-fitted propensity outside
    :math:`(\varepsilon, 1-\varepsilon)` are removed from the estimation and
    diagnostic sample instead. The resulting estimand is therefore defined on
    the retained overlap sample.

    Estimation solves the sample moment equation

    .. math::

        \mathbb{E}_n[\psi_a(W_i; \hat\eta)\theta + \psi_b(W_i; \hat\eta)] = 0,

    giving the closed-form estimator

    .. math::

        \hat\theta = -\frac{\mathbb{E}_n[\psi_b(W_i; \hat\eta)]}
        {\mathbb{E}_n[\psi_a(W_i; \hat\eta)]}.

    For both ATE and ATTE, the orthogonal score component used here is

    .. math::

        \psi_b =
        w \, (\hat g_1(X) - \hat g_0(X))
        + \bar w
        \left[
        (Y - \hat g_1(X)) \frac{D}{\tilde m(X)}
        -
        (Y - \hat g_0(X)) \frac{1-D}{1-\tilde m(X)}
        \right].

    The score derivative differs by estimand:

    .. math::

        \psi_a = -1 \quad \text{for ATE}, \qquad
        \psi_a = -w \quad \text{for ATTE}.

    The corresponding weights are

    .. math::

        w = \bar w = 1 \quad \text{for unweighted ATE},

    while for ATTE` this implementation uses normalized treated weights

    .. math::

        w_i = \frac{D_i}{\mathbb{E}_n[D]}, \qquad
        \bar w_i = \frac{\tilde m(X_i)}{\mathbb{E}_n[D]}.

    If ``normalize_ipw=True``, the inverse-probability factors
    :math:`D / \tilde m(X)` and :math:`(1-D) / (1-\tilde m(X))` are additionally
    stabilized by their sample means (a Hajek-style normalization). This option
    is applied to ATE only; for ATTE it is intentionally ignored to preserve the
    canonical ATTE efficient influence function used by the estimator.
    """

    def __init__(
        self,
        data: Optional[CausalData] = None,
        ml_g: Any = None,
        ml_m: Any = None,
        *,
        n_folds: int = 4,
        n_rep: int = 1,
        normalize_ipw: bool = False,
        overlap_policy: str = "clip",
        overlap_threshold: float = 1e-2,
        weights: Optional[np.ndarray | Dict[str, Any]] = None,
        relative_baseline_min: float = 1e-8,
        random_state: Optional[int] = None,
        n_jobs: int = 1,
        store_diagnostics: bool = True,
    ) -> None:
        """Initialize the estimator and validate configuration options."""
        self.data = data
        self.ml_g = ml_g
        self.ml_m = ml_m
        self._ml_g_is_default = False
        self._ml_m_is_default = False
        self.n_folds = int(n_folds)
        self.n_rep = int(n_rep)
        self.score = "ATE"
        self.normalize_ipw = bool(normalize_ipw)
        self.overlap_policy, self.overlap_threshold = _validate_overlap_config(
            overlap_policy,
            overlap_threshold,
        )
        self.weights = weights
        self.relative_baseline_min = float(relative_baseline_min)
        self.random_state = random_state
        self.n_jobs = int(n_jobs)
        self.store_diagnostics = bool(store_diagnostics)
        self.normalize_ipw_effective_ = bool(normalize_ipw)
        self._X = None
        self._y = None
        self._d = None
        self._fit_index_ = None
        self._fit_row_index_ = None
        self._fit_store_diagnostics_ = bool(store_diagnostics)
        self._fit_sample_fingerprint_ = None
        self.folds_ = None
        self._full_sample_folds_ = None
        self._fixed_fold_assignments_ = None
        self.m_hat_raw_ = None
        self.overlap_mask_ = None
        self.overlap_n_dropped_ = 0
        self.feature_importance_ = None

        # Initialize default learners if not provided
        if HAS_CATBOOST:
            if self.ml_m is None:
                self.ml_m = CatBoostClassifier(
                    thread_count=-1,
                    logging_level='Silent',
                    allow_writing_files=False,
                    random_seed=self.random_state,
                )
                self._ml_m_is_default = True
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
                        logging_level='Silent',
                        allow_writing_files=False,
                        random_seed=self.random_state,
                    )
                else:
                    self.ml_g = CatBoostRegressor(
                        thread_count=-1,
                        logging_level='Silent',
                        allow_writing_files=False,
                        random_seed=self.random_state,
                    )
                self._ml_g_is_default = True

        # If ml_g is still None and HAS_CATBOOST is True, it means data was not provided.
        # It will be initialized in fit().
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
                logging_level='Silent',
                allow_writing_files=False,
                random_seed=self.random_state,
            )
            self._ml_m_is_default = True
        if self.ml_g is None:
            if y_is_binary:
                self.ml_g = CatBoostClassifier(
                    thread_count=-1,
                    logging_level='Silent',
                    allow_writing_files=False,
                    random_seed=self.random_state,
                )
            else:
                self.ml_g = CatBoostRegressor(
                    thread_count=-1,
                    logging_level='Silent',
                    allow_writing_files=False,
                    random_seed=self.random_state,
                )
            self._ml_g_is_default = True

    def _configure_default_learner_parallelism(self) -> None:
        """Avoid oversubscribing CPUs when fold-level parallelism is enabled."""
        if self.n_jobs == 1 or not HAS_CATBOOST:
            return

        for estimator, is_default in (
            (self.ml_g, self._ml_g_is_default),
            (self.ml_m, self._ml_m_is_default),
        ):
            if not is_default or estimator is None:
                continue
            if not hasattr(estimator, "get_params") or not hasattr(
                estimator, "set_params"
            ):
                continue
            params = estimator.get_params(deep=False)
            if params.get("thread_count", None) == -1:
                estimator.set_params(thread_count=1)

    def _ensure_learners_available(self) -> None:
        """Ensure nuisance learners are configured."""
        if self.ml_g is None or self.ml_m is None:
            raise ValueError(
                "ml_g and ml_m must be provided (either as defaults or in __init__)."
            )

    def _validate_fit_config(self, y_is_binary: bool) -> None:
        """Validate IRM fit-time configuration."""
        # Enforce valid propensity model: must expose predict_proba when classifier
        if _safe_is_classifier(self.ml_m) and not hasattr(self.ml_m, "predict_proba"):
            raise ValueError(
                "ml_m must support predict_proba() to produce valid propensity probabilities."
            )
        # For binary outcomes, require probabilistic outcome models when using classifiers
        if (
            y_is_binary
            and _safe_is_classifier(self.ml_g)
            and not hasattr(self.ml_g, "predict_proba")
        ):
            raise ValueError(
                "Binary outcome: ml_g is a classifier but does not expose predict_proba(). Use a probabilistic classifier or calibrate it."
            )

        if self.n_rep != 1:
            raise NotImplementedError("IRM currently supports n_rep=1 only.")
        if self.n_folds < 2:
            raise ValueError("n_folds must be at least 2")
        self.overlap_policy, self.overlap_threshold = _validate_overlap_config(
            self.overlap_policy,
            self.overlap_threshold,
        )

    def _should_collect_feature_importance(self) -> bool:
        """Return whether native feature importances should be collected."""
        return bool(self.store_diagnostics)

    @staticmethod
    def _coerce_native_feature_importance(
        values: Any,
        *,
        n_features: int,
    ) -> Optional[np.ndarray]:
        """Convert a native importance payload into a normalized vector."""
        try:
            arr = np.asarray(values, dtype=float)
        except (TypeError, ValueError):
            return None

        if arr.ndim == 0:
            return None

        if arr.ndim == 1:
            vector = arr
        elif arr.shape[-1] == n_features:
            vector = np.mean(np.abs(arr), axis=tuple(range(arr.ndim - 1)))
        elif arr.shape[0] == n_features:
            vector = np.mean(np.abs(arr), axis=tuple(range(1, arr.ndim)))
        else:
            return None

        vector = np.asarray(vector, dtype=float).ravel()
        if vector.size != n_features:
            return None

        vector = np.where(np.isfinite(vector), np.abs(vector), 0.0)
        total = float(np.sum(vector))
        if total > 0.0:
            vector = vector / total
        return vector.astype(float, copy=False)

    def _extract_native_feature_importance(
        self,
        estimator: Any,
        *,
        n_features: int,
    ) -> Optional[np.ndarray]:
        """Extract normalized native feature importance from a fitted estimator."""
        candidates: List[Any] = []

        if hasattr(estimator, "get_feature_importance"):
            try:
                candidates.append(estimator.get_feature_importance())
            except Exception:
                pass

        if hasattr(estimator, "feature_importances_"):
            candidates.append(getattr(estimator, "feature_importances_", None))

        if hasattr(estimator, "coef_"):
            candidates.append(getattr(estimator, "coef_", None))

        for candidate in candidates:
            importance = self._coerce_native_feature_importance(
                candidate,
                n_features=n_features,
            )
            if importance is not None:
                return importance
        return None

    def _summarize_fold_feature_importances(
        self,
        fold_importances: List[Optional[Dict[str, Optional[np.ndarray]]]],
        *,
        n_features: int,
    ) -> Dict[str, Any]:
        """Aggregate fold-level native importances into compact diagnostics."""
        feature_names = [str(name) for name in list(self.data.confounders)]
        if len(feature_names) != n_features:
            feature_names = [f"x{j + 1}" for j in range(n_features)]

        nuisances: Dict[str, Dict[str, Any]] = {}
        for key in ("m", "g0", "g1"):
            values = [
                np.asarray(payload[key], dtype=float).ravel()
                for payload in fold_importances
                if payload is not None
                and payload.get(key) is not None
                and np.asarray(payload[key]).size == n_features
            ]
            if values:
                stacked = np.vstack(values)
                nuisances[key] = {
                    "available": True,
                    "mean": np.mean(stacked, axis=0),
                    "std": (
                        np.std(stacked, axis=0, ddof=1)
                        if stacked.shape[0] > 1
                        else np.zeros(n_features)
                    ),
                    "n_folds": int(stacked.shape[0]),
                }
            else:
                nuisances[key] = {
                    "available": False,
                    "mean": None,
                    "std": None,
                    "n_folds": 0,
                }

        return {
            "method": "native",
            "feature_names": feature_names,
            "n_features": int(n_features),
            "n_folds": int(len(fold_importances)),
            "nuisances": nuisances,
        }

    def _validate_treatment_support(self, d: np.ndarray) -> None:
        """Ensure both treatment arms have enough rows for stratified cross-fitting."""
        class_counts = np.bincount(d, minlength=2)
        if np.any(class_counts == 0):
            missing = np.where(class_counts == 0)[0].tolist()
            raise RuntimeError(
                f"Missing treatment classes in data: {missing}. Need support for both treatment arms."
            )
        min_class_count = int(class_counts.min())
        if self.n_folds > min_class_count:
            raise ValueError(
                f"n_folds={self.n_folds} exceeds minimum treatment class count={min_class_count}. "
                "Reduce n_folds or collect more data."
            )

    @staticmethod
    def _hash_array(arr: np.ndarray, *, dtype: Optional[np.dtype] = None) -> str:
        """Return an order-sensitive digest for a numeric array."""
        arr_np = np.asarray(arr, dtype=dtype)
        arr_c = np.ascontiguousarray(arr_np)
        return hashlib.blake2b(arr_c.view(np.uint8), digest_size=16).hexdigest()

    @staticmethod
    def _hash_index(index: pd.Index) -> str:
        """Return an order-sensitive digest for a pandas index."""
        hashed = pd.util.hash_pandas_object(index, index=False).to_numpy(
            dtype=np.uint64, copy=False
        )
        return hashlib.blake2b(
            np.ascontiguousarray(hashed).view(np.uint8), digest_size=16
        ).hexdigest()

    def _compute_sample_fingerprint(
        self,
        *,
        X: np.ndarray,
        y: np.ndarray,
        d: np.ndarray,
    ) -> Dict[str, Any]:
        """Build a compact fingerprint for the fitted sample order and contents."""
        if self.data is None or not hasattr(self.data, "df"):
            raise RuntimeError(
                "CausalData must be available to fingerprint the fitted sample."
            )

        return {
            "n_obs": int(len(y)),
            "index_hash": self._hash_index(self.data.df.index),
            "x_hash": self._hash_array(X, dtype=float),
            "y_hash": self._hash_array(y, dtype=float),
            "d_hash": self._hash_array(d, dtype=int),
        }

    def _validate_current_data_matches_fit(
        self,
        *,
        X: np.ndarray,
        y: np.ndarray,
        d: np.ndarray,
    ) -> None:
        """Reject fallback reloads when the underlying CausalData changed after fit."""
        expected_n = int(self.g0_hat_.shape[0])
        if len(y) != expected_n:
            raise RuntimeError(
                "Current data does not match the fitted nuisance predictions. "
                "Refit the model with matching data."
            )

        fingerprint = getattr(self, "_fit_sample_fingerprint_", None)
        if fingerprint is None:
            return

        current = self._compute_sample_fingerprint(X=X, y=y, d=d)
        if current != fingerprint:
            raise RuntimeError(
                "Current data does not match the fitted nuisance predictions. "
                "The underlying CausalData changed after fit(); refit the model."
            )

    def _compute_ipw_components(
        self,
        *,
        d: np.ndarray,
        m_hat: np.ndarray,
        score: Optional[str] = None,
        warn: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute IPW terms plus Riesz-compatible inverse propensity factors."""
        return _compute_irm_ipw_components(
            d=d,
            m_hat=m_hat,
            normalize_ipw=self.normalize_ipw,
            score=score,
            warn=warn,
        )

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
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Fit one outcome nuisance model (g0 or g1) and predict on test fold."""
        model_g = clone(self.ml_g)
        mask = d_tr == treatment_value
        if not np.any(mask):
            raise RuntimeError(empty_group_error)
        X_g, y_g = X_tr[mask], y_tr[mask]
        if y_is_binary:
            uniq_y = np.unique(y_g)
            if uniq_y.size == 1:
                # Single-class folds can skip fitting while preserving cross-fit independence.
                return np.full(X_te.shape[0], float(uniq_y[0]), dtype=float), None
        model_g.fit(X_g, y_g)
        pred = _predict_prob_or_value(model_g, X_te, is_propensity=False)
        if y_is_binary:
            pred = np.clip(pred, 1e-12, 1 - 1e-12)
        importance = None
        if self._should_collect_feature_importance():
            importance = self._extract_native_feature_importance(
                model_g,
                n_features=X_tr.shape[1],
            )
        return pred, importance

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
    ) -> Tuple[
        int,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        Optional[Dict[str, Optional[np.ndarray]]],
    ]:
        """Fit nuisance models for one fold and return held-out predictions."""
        X_tr, y_tr, d_tr = X[train_idx], y[train_idx], d[train_idx]
        X_te = X[test_idx]

        g0_te, g0_importance = self._fit_outcome_nuisance_for_treatment(
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

        g1_te, g1_importance = self._fit_outcome_nuisance_for_treatment(
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
        m_importance = None
        if self._should_collect_feature_importance():
            m_importance = self._extract_native_feature_importance(
                model_m,
                n_features=X_tr.shape[1],
            )

        fold_importance = None
        if self._should_collect_feature_importance():
            fold_importance = {
                "m": m_importance,
                "g0": g0_importance,
                "g1": g1_importance,
            }

        return fold_id, test_idx, g0_te, g1_te, m_te, fold_importance

    def _cross_fit_nuisances(
        self,
        X: np.ndarray,
        y: np.ndarray,
        d: np.ndarray,
        y_is_binary: bool,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        Optional[Dict[str, Any]],
    ]:
        """Run cross-fitting and return nuisance predictions and fold ids."""
        n = X.shape[0]
        g0_hat = np.full(n, np.nan, dtype=float)
        g1_hat = np.full(n, np.nan, dtype=float)
        m_hat = np.full(n, np.nan, dtype=float)
        folds = np.full(n, -1, dtype=int)
        fold_importances: List[Optional[Dict[str, Optional[np.ndarray]]]] = []

        fixed_folds = getattr(self, "_fixed_fold_assignments_", None)
        if fixed_folds is None:
            skf = StratifiedKFold(
                n_splits=self.n_folds, shuffle=True, random_state=self.random_state
            )
            splits = list(skf.split(X, d))
        else:
            fixed_folds = np.asarray(fixed_folds, dtype=int).ravel()
            if fixed_folds.size != n:
                raise ValueError(
                    "Fixed fold assignments must match the fitted sample size. "
                    f"Got {fixed_folds.size} and {n}."
                )
            expected_folds = np.arange(self.n_folds, dtype=int)
            if not np.array_equal(np.unique(fixed_folds), expected_folds):
                raise ValueError(
                    "Fixed fold assignments must contain every fold id from 0 to "
                    f"{self.n_folds - 1}."
                )
            all_idx = np.arange(n, dtype=int)
            splits = [
                (all_idx[fixed_folds != fold_id], all_idx[fixed_folds == fold_id])
                for fold_id in expected_folds
            ]

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

        for fold_id, test_idx, g0_te, g1_te, m_te, fold_importance in fold_results:
            folds[test_idx] = fold_id
            g0_hat[test_idx] = g0_te
            g1_hat[test_idx] = g1_te
            m_hat[test_idx] = m_te
            if self._should_collect_feature_importance():
                fold_importances.append(fold_importance)

        feature_importance = None
        if self._should_collect_feature_importance():
            feature_importance = self._summarize_fold_feature_importances(
                fold_importances,
                n_features=X.shape[1],
            )

        return g0_hat, g1_hat, m_hat, folds, feature_importance

    def _store_cross_fitted_predictions(
        self,
        g0_hat: np.ndarray,
        g1_hat: np.ndarray,
        m_hat: np.ndarray,
        folds: np.ndarray,
        feature_importance: Optional[Dict[str, Any]],
    ) -> None:
        """Validate and store cross-fitted nuisance predictions."""
        if (
            np.any(np.isnan(m_hat))
            or np.any(np.isnan(g0_hat))
            or np.any(np.isnan(g1_hat))
        ):
            raise RuntimeError("Cross-fitted predictions contain NaN values.")

        full_sample_folds = np.asarray(folds, dtype=int).ravel()
        if full_sample_folds.size != np.asarray(m_hat).size:
            raise RuntimeError("Cross-fitting fold assignments have inconsistent length.")
        # Sensitivity benchmark refits need the original fold assignment even
        # when overlap_policy='drop' later removes rows from the estimation
        # sample. This private cache is intentionally independent of the
        # diagnostics storage setting; ``folds_`` keeps its existing semantics.
        self._full_sample_folds_ = full_sample_folds.copy()

        m_policy, overlap_mask = _apply_overlap_policy(
            m_hat,
            policy=self.overlap_policy,
            threshold=self.overlap_threshold,
        )
        overlap_mask = np.asarray(overlap_mask, dtype=bool).ravel()
        if overlap_mask.size != np.asarray(m_hat).size:
            raise RuntimeError("Overlap mask has inconsistent length.")

        self.overlap_mask_ = overlap_mask.copy()
        self.overlap_n_dropped_ = int(overlap_mask.size - int(np.sum(overlap_mask)))

        raw_m_hat = np.asarray(m_hat, dtype=float).ravel()
        if self.overlap_policy == "drop":
            if not np.any(overlap_mask):
                raise ValueError(
                    "overlap_policy='drop' removed all observations. "
                    "Lower overlap_threshold or use overlap_policy='clip'."
                )
            if self._y is not None:
                self._y = self._y[overlap_mask]
            if self._d is not None:
                self._d = self._d[overlap_mask]
                n_treated = int(np.sum(self._d == 1))
                n_control = int(np.sum(self._d == 0))
                if n_treated == 0 or n_control == 0:
                    raise ValueError(
                        "overlap_policy='drop' must retain at least one treated "
                        "and one control observation."
                    )
            if self._X is not None:
                self._X = self._X[overlap_mask]
            if self._fit_index_ is not None:
                self._fit_index_ = pd.Index(self._fit_index_[overlap_mask], name=self._fit_index_.name)
            if self._fit_row_index_ is not None:
                self._fit_row_index_ = pd.Index(
                    self._fit_row_index_[overlap_mask],
                    name=self._fit_row_index_.name,
                )
            g0_hat = np.asarray(g0_hat, dtype=float).ravel()[overlap_mask]
            g1_hat = np.asarray(g1_hat, dtype=float).ravel()[overlap_mask]
            folds = full_sample_folds[overlap_mask]
            raw_m_hat = raw_m_hat[overlap_mask]
        else:
            folds = full_sample_folds

        self.g0_hat_ = g0_hat
        self.g1_hat_ = g1_hat
        self.m_hat_ = m_policy
        if self.store_diagnostics:
            self.folds_ = folds
            self.m_hat_raw_ = raw_m_hat.copy()
            self.feature_importance_ = feature_importance
        else:
            self.folds_ = None
            self.m_hat_raw_ = None
            self.feature_importance_ = None

    def _store_fit_sample(self, X: np.ndarray, y: np.ndarray, d: np.ndarray) -> None:
        """Persist immutable fit-time targets and optional diagnostic covariates."""
        self._fit_sample_fingerprint_ = self._compute_sample_fingerprint(X=X, y=y, d=d)
        if getattr(self.data, "user_id_name", None):
            self._fit_index_ = pd.Index(
                self.data.user_id.copy(), name=self.data.user_id_name
            )
            self._fit_row_index_ = self.data.df.index.copy()
        else:
            self._fit_index_ = self.data.df.index.copy()
            self._fit_row_index_ = None
        self._y = np.asarray(y, dtype=float).copy()
        self._d = np.asarray(d, dtype=int).copy()
        if self.store_diagnostics:
            self._X = np.asarray(X, dtype=float).copy()
        else:
            self._X = None

    def _resolve_estimation_targets(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return fit-time outcomes and treatments for ATE/ATTE/GATE estimation."""
        if self._y is not None and self._d is not None:
            return self._y, self._d

        # Backward-compatibility fallback for older fitted objects.
        X, y, d, _ = self._check_data()
        self._validate_current_data_matches_fit(X=X, y=y, d=d)
        return y, d

    def _resolve_diagnostic_features(self) -> np.ndarray:
        """Return fit-time confounders for diagnostic-only payloads."""
        if self._X is not None:
            return self._X
        if not getattr(self, "_fit_store_diagnostics_", self.store_diagnostics):
            raise RuntimeError(
                "Confounder diagnostics are unavailable because the model was fitted with "
                "store_diagnostics=False."
            )

        # Backward-compatibility fallback for older fitted objects.
        X, y, d, _ = self._check_data()
        self._validate_current_data_matches_fit(X=X, y=y, d=d)
        return X

    def _resolve_estimation_sample(
        self,
    ) -> Tuple[Optional[np.ndarray], np.ndarray, np.ndarray]:
        """Backward-compatible wrapper exposing cached estimation inputs."""
        y, d = self._resolve_estimation_targets()
        has_fit_features = getattr(
            self, "_fit_store_diagnostics_", self.store_diagnostics
        )
        X = self._resolve_diagnostic_features() if has_fit_features else None
        return X, y, d

    def _get_weights(
        self,
        n: int,
        m_hat_adj: Optional[np.ndarray],
        d: np.ndarray,
        score: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
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
        return _resolve_irm_weights(
            n=n,
            m_hat_adj=m_hat_adj,
            d=d,
            score=self.score if score is None else score,
            weights=self.weights,
        )

    def _use_normalized_ipw(
        self, score: Optional[str] = None, *, warn: bool = False
    ) -> bool:
        """Return whether Hájek normalization is active for a given score."""
        return _use_normalized_irm_ipw(
            normalize_ipw=self.normalize_ipw,
            score=self.score if score is None else score,
            warn=warn,
        )

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
        return _normalize_irm_ipw_terms(
            d,
            m_hat,
            normalize_ipw=self.normalize_ipw,
            score=self.score if score is None else score,
            warn=warn,
        )

    def _compute_estimate_components(
        self,
        *,
        y: np.ndarray,
        d: np.ndarray,
        g0_hat: np.ndarray,
        g1_hat: np.ndarray,
        m_hat: np.ndarray,
        score: str,
    ) -> Dict[str, np.ndarray]:
        """Compute reusable per-estimate arrays once."""
        n = len(y)
        u0 = y - g0_hat
        u1 = y - g1_hat
        h1, h0, inv_m, inv_1m = self._compute_ipw_components(
            d=d, m_hat=m_hat, score=score
        )
        w, w_bar = self._get_weights(n, m_hat, d, score=score)
        psi_b = w * (g1_hat - g0_hat) + w_bar * (u1 * h1 - u0 * h0)

        if score == "ATE":
            psi_a = -np.ones(n)
        elif score == "ATTE":
            psi_a = -w
        else:
            raise ValueError("score must be 'ATE' or 'ATTE'")

        return {
            "u0": u0,
            "u1": u1,
            "h1": h1,
            "h0": h0,
            "inv_m": inv_m,
            "inv_1m": inv_1m,
            "w": w,
            "w_bar": w_bar,
            "psi_a": psi_a,
            "psi_b": psi_b,
        }

    # --------- API ---------
    def fit(
        self,
        data: Optional[CausalData] = None,
        *,
        store_diagnostics: Optional[bool] = None,
    ) -> "IRM":
        """Fit nuisance models via cross-fitting.

        Parameters
        ----------
        data : Optional[CausalData], default None
            CausalData container. If None, uses self.data.
        store_diagnostics : Optional[bool], default None
            Optional override for whether the fitted model should retain
            diagnostics-oriented arrays and expose diagnostic payloads from
            subsequent ``estimate()`` calls. Outcome and treatment snapshots are
            always retained to keep post-fit estimation deterministic.

        Returns
        -------
        self : IRM
            Fitted estimator.
        """
        if data is not None:
            self.data = data
        if store_diagnostics is not None:
            self.store_diagnostics = bool(store_diagnostics)
        self._fit_store_diagnostics_ = bool(self.store_diagnostics)
        self.feature_importance_ = None
        if self.data is None:
            raise ValueError(
                "Model must be provided with CausalData either in __init__ or in .fit(data_contracts)."
            )
        X, y, d, y_is_binary = self._check_data()

        # Initialize default learners if not provided and data is now available
        self._initialize_default_learners_for_fit(y_is_binary=y_is_binary)
        self._ensure_learners_available()
        self._configure_default_learner_parallelism()

        # Cache for sensitivity analysis and effect calculation
        self._validate_fit_config(y_is_binary=y_is_binary)
        self._validate_treatment_support(d)
        self._store_fit_sample(X=X, y=y, d=d)

        g0_hat, g1_hat, m_hat, folds, feature_importance = self._cross_fit_nuisances(
            X=X, y=y, d=d, y_is_binary=y_is_binary
        )
        self._store_cross_fitted_predictions(
            g0_hat=g0_hat,
            g1_hat=g1_hat,
            m_hat=m_hat,
            folds=folds,
            feature_importance=feature_importance,
        )

        return self

    def _validate_estimate_request(self, score: str, alpha: float) -> str:
        """Validate estimate() arguments and return normalized score."""
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be in (0,1)")
        score_u = str(score).upper()
        if score_u == "CATE":
            raise NotImplementedError(
                "score='CATE' is a prediction task, not a scalar estimand. "
                "Use model.predict_cate(X) to score new clients."
            )
        if score_u not in {"ATE", "ATTE", "GATE", "GATET"}:
            raise ValueError("score must be 'ATE', 'ATTE', 'GATE', or 'GATET'")
        return score_u

    def _estimate_inference_approx_flags(
        self, score: str, normalize_ipw_effective: bool
    ) -> Dict[str, bool]:
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
        w: np.ndarray,
        w_bar: np.ndarray,
        g0_hat: np.ndarray,
        u0: np.ndarray,
        h0: np.ndarray,
        z: float,
    ) -> Tuple[float, float, float, float, float]:
        """Compute relative effect and delta-method interval."""
        n = len(w)
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
        baseline_low_signal = (
            np.isfinite(mu_c_se) and mu_c_se > 0.0 and abs(mu_c) < z * mu_c_se
        )

        if np.isfinite(mu_c) and not (baseline_too_small or baseline_low_signal):
            tau_rel = 100.0 * theta_hat / mu_c
            IF_mu = psi_mu_c - mu_c
            with np.errstate(divide="ignore", invalid="ignore"):
                # Delta-method IF for tau_rel = 100 * theta / mu_c.
                IF_rel = 100.0 * (IF / mu_c - (theta_hat * IF_mu) / (mu_c**2))
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
                reasons.append(
                    f"is below relative_baseline_min={self.relative_baseline_min:.3e}"
                )
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
        x: Optional[np.ndarray],
        inv_m: np.ndarray,
        inv_1m: np.ndarray,
    ) -> Optional[Any]:
        """Build optional diagnostics payload for CausalEstimate."""
        return _build_irm_estimate_diagnostic_data(
            model=self,
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
            x=x,
            inv_m=inv_m,
            inv_1m=inv_1m,
        )

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
                "overlap_policy": self.overlap_policy,
                "overlap_threshold": self.overlap_threshold,
                "overlap_n_dropped": int(getattr(self, "overlap_n_dropped_", 0)),
                "random_state": self.random_state,
                "n_jobs": self.n_jobs,
                "std_error": se,
                "t_stat": t_stat,
                "se_approx_hajek": bool(approx_flags.get("se_approx_hajek", False)),
                "se_approx_weight_norm": bool(
                    approx_flags.get("se_approx_weight_norm", False)
                ),
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
        groups: Optional[pd.DataFrame | pd.Series] = None,
        cov_type: str = "HC3",
        cov_kwds: Optional[Dict[str, Any]] = None,
    ) -> CausalEstimate | GateEstimate:
        """Compute treatment effects using stored nuisance predictions.

        Parameters
        ----------
        score : {"ATE", "ATTE", "GATE", "GATET"}, default "ATE"
            Target estimand.
        alpha : float, default 0.05
            Significance level for intervals.
            Diagnostic payloads are included only when the model was fitted with
            ``store_diagnostics=True``.
        groups : Optional[pd.DataFrame | pd.Series], default None
            Group labels/indicators for ``score="GATE"`` or ``score="GATET"``.
            If None, fallback to ``self.data.gate_groups`` when present.
            GATE/GATET requires ``CausalData.user_id`` and aligns groups to those
            fit-time observation ids. Row-indexed groups are also accepted only
            when the fit-time row-to-``user_id`` mapping is still unchanged.
        cov_type : {"HC0", "HC1", "HC2", "HC3"}, default "HC3"
            Robust covariance type for ``score="GATE"`` / ``score="GATET"`` inference.
        cov_kwds : Optional[Dict[str, Any]], default None
            Additional covariance keyword arguments requested for subgroup inference.
            These are currently ignored because GATE/GATET use closed-form HCx
            covariance formulas rather than delegating to statsmodels.

        Returns
        -------
        CausalEstimate or GateEstimate
            Result container for the estimated effect. For subgroup scores,
            the returned ``GateEstimate`` supports ``summary()`` for
            subgroup-vs-zero inference, ``contrast(...)`` for formal
            group-vs-group tests, and ``pairwise_summary(...)`` for a
            broader comparison table.
        """
        check_is_fitted(self, attributes=["g0_hat_", "g1_hat_", "m_hat_"])
        score = self._validate_estimate_request(score=score, alpha=alpha)
        self.score = score

        if score in {"GATE", "GATET"}:
            subgroup_estimator = (
                estimate_gate_from_irm if score == "GATE" else estimate_gatet_from_irm
            )
            return subgroup_estimator(
                irm_model=self,
                groups=groups,
                alpha=alpha,
                cov_type=cov_type,
                cov_kwds=cov_kwds,
            )

        # For ATTE we intentionally disable Hájek even if normalize_ipw=True.
        normalize_ipw_effective = self._use_normalized_ipw(score=score, warn=False)
        self.normalize_ipw_effective_ = normalize_ipw_effective
        # Track known finite-sample approximation paths in inference metadata.
        approx_flags = self._estimate_inference_approx_flags(
            score=score, normalize_ipw_effective=normalize_ipw_effective
        )
        self._warn_if_inference_is_approximate(approx_flags)

        y, d = self._resolve_estimation_targets()
        g0_hat, g1_hat, m_hat = self.g0_hat_, self.g1_hat_, self.m_hat_

        components = self._compute_estimate_components(
            y=y, d=d, g0_hat=g0_hat, g1_hat=g1_hat, m_hat=m_hat, score=score
        )
        w = components["w"]
        w_bar = components["w_bar"]
        psi_a = components["psi_a"]
        psi_b = components["psi_b"]
        theta_hat, IF, se, t_stat, pval, ci_low, ci_high, z = (
            self._solve_moment_equation(psi_a=psi_a, psi_b=psi_b, alpha=alpha)
        )
        self._cache_estimate_core(
            theta_hat=theta_hat, se=se, IF=IF, psi_a=psi_a, psi_b=psi_b
        )

        mu_c, tau_rel, ci_low_rel, ci_high_rel, se_rel = (
            self._compute_relative_effect_stats(
                theta_hat=theta_hat,
                IF=IF,
                w=w,
                w_bar=w_bar,
                g0_hat=g0_hat,
                u0=components["u0"],
                h0=components["h0"],
                z=z,
            )
        )
        self.mu_c_ = mu_c
        self.se_relative_ = np.array([se_rel])
        self.confint_relative_ = np.array([[ci_low_rel, ci_high_rel]])

        x = self._resolve_diagnostic_features() if self.store_diagnostics else None
        diag = self._build_estimate_diagnostic_data(
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
            x=x,
            inv_m=components["inv_m"],
            inv_1m=components["inv_1m"],
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
            "overlap_policy": self.overlap_policy,
            "overlap_threshold": self.overlap_threshold,
            "overlap_mask": getattr(self, "overlap_mask_", None),
            "g0_hat": self.g0_hat_,
            "g1_hat": self.g1_hat_,
            "folds": getattr(self, "folds_", None),
            "feature_importance": getattr(self, "feature_importance_", None),
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
    ) -> GateEstimate:
        """Convenience wrapper for ``estimate(score="GATE", ...)``.

        Parameters
        ----------
        groups : pd.DataFrame or pd.Series
            Subgroup labels or a strict dummy basis. GATE requires
            ``CausalData.user_id`` and aligns groups to those fit-time
            observation ids.
        alpha : float, default 0.05
            Significance level for confidence intervals.
        cov_type : {"HC0", "HC1", "HC2", "HC3"}, default "HC3"
            Robust covariance type for subgroup inference.
        cov_kwds : Optional[Dict[str, Any]], default None
            Additional covariance keyword arguments requested by the caller.
            These are currently ignored by the closed-form GATE implementation.

        Returns
        -------
        GateEstimate
            Estimated subgroup effects and diagnostics. The returned result
            also supports ``contrast(...)`` and ``pairwise_summary(...)``
            for formal post-estimation group comparisons.
        """
        return self.estimate(
            score="GATE",
            groups=groups,
            alpha=alpha,
            cov_type=cov_type,
            cov_kwds=cov_kwds,
        )

    def gatet(
        self,
        groups: pd.DataFrame | pd.Series,
        alpha: float = 0.05,
        cov_type: str = "HC3",
        cov_kwds: Optional[Dict[str, Any]] = None,
    ) -> GateEstimate:
        """Convenience wrapper for ``estimate(score="GATET", ...)``."""
        return self.estimate(
            score="GATET",
            groups=groups,
            alpha=alpha,
            cov_type=cov_type,
            cov_kwds=cov_kwds,
        )

    def predict_cate(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict CATE/uplift for new rows using lazy full-sample scoring models."""
        from causalis.scenarios.uplift.model import predict_cate

        return predict_cate(self, X)

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
        inv_m: Optional[np.ndarray] = None,
        inv_1m: Optional[np.ndarray] = None,
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
        if any(
            getattr(self, attr) is None for attr in ["g0_hat_", "g1_hat_", "m_hat_"]
        ):
            raise RuntimeError("IRM model must be fitted before sensitivity analysis.")

        if y is None:
            y = self._y
        if d is None:
            d = self._d

        if y is None or d is None:
            # Backward-compatibility fallback for older fitted objects.
            X_cur, y_cur, d_cur, _ = self._check_data()
            self._validate_current_data_matches_fit(X=X_cur, y=y_cur, d=d_cur)
            if y is None:
                y = y_cur
            if d is None:
                d = d_cur

        if m_hat is None:
            m_hat = np.asarray(self.m_hat_, dtype=float)
        if g0 is None:
            g0 = np.asarray(self.g0_hat_, dtype=float)
        if g1 is None:
            g1 = np.asarray(self.g1_hat_, dtype=float)

        from causalis.scenarios.unconfoundedness.refutation.unconfoundedness.sensitivity import (
            compute_irm_sensitivity_elements,
        )

        return compute_irm_sensitivity_elements(
            model=self,
            y=np.asarray(y, dtype=float),
            d=np.asarray(d, dtype=int),
            g0=np.asarray(g0, dtype=float),
            g1=np.asarray(g1, dtype=float),
            m_hat=np.asarray(m_hat, dtype=float),
            w=w,
            w_bar=w_bar,
            psi=psi,
            inv_m=inv_m,
            inv_1m=inv_1m,
            score=getattr(self, "score", "ATE"),
        )

    def sensitivity_analysis(
        self,
        r2_y: float,
        r2_d: float,
        rho: float = 1.0,
        H0: float = 0.0,
        alpha: float = 0.05,
    ) -> "IRM":
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
            get_sensitivity_summary,
        )

        check_is_fitted(self, attributes=["coef_", "se_", "psi_"])

        # Execute sensitivity analysis using the centralized module logic
        res = sa_fn(self, r2_y=r2_y, r2_d=r2_d, rho=rho, H0=H0, alpha=alpha)

        self.sensitivity_result = res
        # Cache the summary string for display
        self.sensitivity_summary = get_sensitivity_summary(
            {"model": self, "bias_aware": res}
        )

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

    def __repr__(self) -> str:
        """Concise representation of IRM to avoid verbose learner output."""
        status = "fitted" if hasattr(self, "g0_hat_") else "unfitted"
        return f"IRM(status='{status}', n_folds={self.n_folds}, random_state={self.random_state})"

    _repr_html_ = None
    _repr_mimebundle_ = None
