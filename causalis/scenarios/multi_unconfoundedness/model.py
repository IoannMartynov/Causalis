from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
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

from causalis.data_contracts.multicausaldata import MultiCausalData
from causalis.data_contracts.multicausal_estimate import MultiCausalEstimate
from causalis.data_contracts.causal_diagnostic_data import MultiUnconfoundednessDiagnosticData
from causalis.scenarios.multi_unconfoundedness._utils import (
    _is_binary,
    _normalize_multiclass_ipw_terms,
    _predict_propensity_matrix,
    _safe_is_classifier,
    _trim_multiclass_propensity,
)


class MultiTreatmentIRM(BaseEstimator):
    """Interactive Regression Model for multi-treatment unconfoundedness.

    DoubleML-style cross-fitting estimator consuming ``MultiCausalData`` and
    producing pairwise ATE contrasts against baseline treatment (column 0).
       Model supports >= 2 treatments.

        Parameters
        ----------
        data : MultiCausalData
            Data container with outcome, one-hot multi-treatment indicators, and confounders.
        ml_g : estimator
            Learner for E[Y|X,D]. If classifier and Y is binary, predict_proba is used; otherwise predict().
        ml_m : classifier
            Learner for E[D|X] (generelized propensity score). Must support predict_proba() or predict() in (0,1).
        n_folds : int, default 5
            Number of cross-fitting folds.
        n_rep : int, default 1
            Number of repetitions of sample splitting. Currently only 1 is supported.
        normalize_ipw : bool, default False
            Whether to normalize IPW terms within the score.
            This is a Hajek-style stabilization: it can reduce variance but may add
            small finite-sample bias.
        trimming_rule : {"truncate"}, default "truncate"
            Trimming approach for propensity scores.
        trimming_threshold : float, default 1e-2
            Threshold for trimming if rule is "truncate".
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
        store_diagnostics : bool, default True
            Whether to retain raw fit-time arrays and diagnostic-only artifacts on
            the fitted model. Set to ``False`` for a lighter-weight estimator that
            still supports ATE estimation, while omitting heavy diagnostic caches.
        """
    def __init__(
        self,
        data: Optional[MultiCausalData] = None,
        ml_g: Any = None,
        ml_m: Any = None,
        *,
        n_folds: int = 5,
        n_rep: int = 1,
        normalize_ipw: bool = False,
        trimming_rule: str = "truncate",
        trimming_threshold: float = 1e-2,
        random_state: Optional[int] = None,
        n_jobs: int = 1,
        store_diagnostics: bool = True,
    ) -> None:
        self.data = data
        self.ml_g = ml_g
        self.ml_m = ml_m
        self._ml_g_is_default = False
        self._ml_m_is_default = False
        self.n_folds = int(n_folds)
        self.n_rep = int(n_rep)
        self.score = "ATE"
        self.normalize_ipw = bool(normalize_ipw)
        self.trimming_rule = str(trimming_rule)
        self.trimming_threshold = float(trimming_threshold)
        self.random_state = random_state
        self.n_jobs = int(n_jobs)
        self.store_diagnostics = bool(store_diagnostics)
        self.normalize_ipw_effective_ = bool(normalize_ipw)
        self._X = None
        self._y = None
        self._d = None
        self._fit_store_diagnostics_ = bool(store_diagnostics)
        self._fit_sample_fingerprint_ = None
        self.folds_ = None
        self.m_hat_raw_ = None
        if not np.isfinite(self.trimming_threshold) or not (0.0 <= self.trimming_threshold < 0.5):
            raise ValueError("trimming_threshold must be finite and in [0, 0.5).")
        if self.n_jobs == 0 or self.n_jobs < -1:
            raise ValueError("n_jobs must be -1 or a positive integer.")

        # Initialize default learners if not provided
        if HAS_CATBOOST:
            if self.ml_m is None:
                self.ml_m = CatBoostClassifier(
                    loss_function="MultiClass",
                    thread_count=-1,
                    verbose=False,
                    allow_writing_files=False,
                    random_seed=self.random_state,
                )
                self._ml_m_is_default = True
            if self.ml_g is None and self.data is not None:
                y_is_binary = False
                try:
                    df_tmp = self.data.get_df()
                    y_tmp = df_tmp[self.data.outcome].to_numpy()
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
                    self._ml_g_is_default = True
                else:
                    self.ml_g = CatBoostRegressor(
                        thread_count=-1,
                        verbose=False,
                        allow_writing_files=False,
                        random_seed=self.random_state,
                    )
                    self._ml_g_is_default = True

        # If ml_g is still None and HAS_CATBOOST is True, it means data was not provided.
        # It will be initialized in fit().

    def _check_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool, int]:
        """Validate input contract and return model-ready arrays."""
        if not isinstance(self.data, MultiCausalData):
            raise TypeError(
                f"data must be MultiCausalData, got {type(self.data).__name__}."
            )
        df = self.data.get_df()
        y = df[self.data.outcome].to_numpy(dtype=float, copy=False)
        d = df[self.data.treatments.columns].to_numpy(dtype=int, copy=False)

        n_treatments_ = len(self.data.treatments.columns)
        if n_treatments_ < 2:
            raise ValueError("Need at least 2 treatment columns (baseline + at least one active treatment).")
        if d.ndim != 2 or d.shape[1] != n_treatments_:
            raise ValueError("Treatment matrix must be 2D with shape (n, K).")
        if not np.all((d == 0) | (d == 1)):
            raise ValueError("Treatment matrix must be one-hot encoded with 0/1 entries.")
        if not np.all(d.sum(axis=1) == 1):
            raise ValueError("Treatment matrix must be one-hot encoded with exactly one active treatment per row.")

        x_cols = list(self.data.confounders)
        if len(x_cols) == 0:
            raise ValueError("MultiCausalData must include non-empty confounders.")
        X = df[x_cols].to_numpy(dtype=float, copy=False)
        y_is_binary = _is_binary(y)

        return X, y, d, y_is_binary, n_treatments_

    def _use_normalized_ipw(self, score: Optional[str] = None) -> bool:
        """Return whether Hajek-style IPW normalization is active for a score."""
        score_u = str(self.score if score is None else score).upper()
        return bool(self.normalize_ipw) and score_u == "ATE"

    def _normalize_ipw_terms(
        self,
        d: np.ndarray,
        m_hat: np.ndarray,
        score: Optional[str] = None,
    ) -> np.ndarray:
        """Return d_k / m_k with score-aware Hajek column normalization."""
        return _normalize_multiclass_ipw_terms(
            d=d,
            m_hat=m_hat,
            normalize_ipw=self._use_normalized_ipw(score=score),
        )

    def _initialize_default_learners_for_fit(self, y_is_binary: bool) -> None:
        """Initialize default CatBoost nuisances when users do not provide learners."""
        if not HAS_CATBOOST:
            return
        if self.ml_m is None:
            self.ml_m = CatBoostClassifier(
                thread_count=-1,
                verbose=False,
                allow_writing_files=False,
                random_seed=self.random_state,
            )
            self._ml_m_is_default = True
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
            self._ml_g_is_default = True

    def _configure_default_learner_parallelism(self) -> None:
        """Avoid oversubscribing CPUs when fold-level parallelism is enabled."""
        if self.n_jobs == 1 or not HAS_CATBOOST:
            return

        for estimator, is_default in ((self.ml_g, self._ml_g_is_default), (self.ml_m, self._ml_m_is_default)):
            if not is_default or estimator is None:
                continue
            if not hasattr(estimator, "get_params") or not hasattr(estimator, "set_params"):
                continue
            params = estimator.get_params(deep=False)
            if params.get("thread_count", None) == -1:
                estimator.set_params(thread_count=1)

    def _ensure_learners_available(self) -> None:
        if self.ml_g is None or self.ml_m is None:
            raise ValueError("ml_g and ml_m must be provided (either as defaults or in __init__).")

    def _validate_fit_config(self, *, y_is_binary: bool) -> None:
        """Validate fit-time config before running expensive cross-fitting."""
        if (not _safe_is_classifier(self.ml_m)) or (not hasattr(self.ml_m, "predict_proba")):
            raise ValueError("ml_m must be a classifier with predict_proba() for valid multiclass propensity scores.")
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
        if self.trimming_threshold >= (1.0 / self.n_treatments):
            raise ValueError(
                f"trimming_threshold must be < 1/K for multiclass renormalized trimming; "
                f"got threshold={self.trimming_threshold} with K={self.n_treatments}."
            )

    def _validate_class_support(self, d_strat: np.ndarray) -> None:
        """Ensure every treatment class has enough rows for stratified cross-fitting."""
        class_counts = np.bincount(d_strat, minlength=self.n_treatments)
        if np.any(class_counts == 0):
            missing = np.where(class_counts == 0)[0].tolist()
            raise RuntimeError(
                f"Missing treatment classes in data: {missing}. Need support for all treatment columns."
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
        hashed = pd.util.hash_pandas_object(index, index=False).to_numpy(dtype=np.uint64, copy=False)
        return hashlib.blake2b(np.ascontiguousarray(hashed).view(np.uint8), digest_size=16).hexdigest()

    def _compute_sample_fingerprint(
        self,
        *,
        X: np.ndarray,
        y: np.ndarray,
        d: np.ndarray,
    ) -> Dict[str, Any]:
        """Build a compact fingerprint for the fitted sample order and contents."""
        if self.data is None or not hasattr(self.data, "df"):
            raise RuntimeError("MultiCausalData must be available to fingerprint the fitted sample.")

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
        """Reject fallback reloads when the underlying MultiCausalData changed after fit."""
        expected_n = int(self.g_hat_.shape[0])
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
                "The underlying MultiCausalData changed after fit(); refit the model."
            )

    def _store_fit_sample(self, X: np.ndarray, y: np.ndarray, d: np.ndarray) -> None:
        """Persist immutable fit-time targets and optional diagnostic covariates."""
        self._fit_sample_fingerprint_ = self._compute_sample_fingerprint(X=X, y=y, d=d)
        self._y = np.asarray(y, dtype=float).copy()
        self._d = np.asarray(d, dtype=int).copy()
        if self.store_diagnostics:
            self._X = np.asarray(X, dtype=float).copy()
        else:
            self._X = None

    def _store_cross_fitted_predictions(
        self,
        *,
        g_hat: np.ndarray,
        m_hat: np.ndarray,
        folds: np.ndarray,
    ) -> None:
        """Validate and store cross-fitted nuisance predictions."""
        if np.any(np.isnan(m_hat)) or np.any(np.isnan(g_hat)):
            raise RuntimeError("Cross-fitted predictions contain NaN values.")
        self.g_hat_ = g_hat
        self.m_hat_ = _trim_multiclass_propensity(m_hat, self.trimming_threshold)
        if self.store_diagnostics:
            self.folds_ = folds
            self.m_hat_raw_ = np.asarray(m_hat, dtype=float).copy()
        else:
            self.folds_ = None
            self.m_hat_raw_ = None

    def _resolve_estimation_targets(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return fit-time outcomes and treatments for ATE/ATTE estimation."""
        if self._y is not None and self._d is not None:
            return self._y, self._d

        X, y, d, _, _ = self._check_data()
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

        X, y, d, _, _ = self._check_data()
        self._validate_current_data_matches_fit(X=X, y=y, d=d)
        return X

    def _predict_binary_outcome_probability(self, model_g, X: np.ndarray) -> np.ndarray:
        """Predict P(Y=1|X,D=k) robustly across binary classifier APIs."""
        pred_g = np.asarray(model_g.predict_proba(X), dtype=float)
        if pred_g.ndim == 2:
            g_classes = np.asarray(getattr(model_g, "classes_", np.array([0, 1])))
            if 1 in g_classes:
                one_col = int(np.where(g_classes == 1)[0][0])
                pred_g = pred_g[:, one_col]
            elif g_classes.size == 1 and g_classes[0] == 0:
                pred_g = np.zeros(pred_g.shape[0], dtype=float)
            elif g_classes.size == 1 and g_classes[0] == 1:
                pred_g = np.ones(pred_g.shape[0], dtype=float)
            elif pred_g.shape[1] == 2:
                pred_g = pred_g[:, 1]
            else:
                raise ValueError("Binary outcome model must return probability for class 1.")
        else:
            pred_g = pred_g.ravel()
        return np.asarray(pred_g, dtype=float).ravel()

    def _fit_one_outcome_nuisance(
        self,
        *,
        k: int,
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        d_tr_onehot: np.ndarray,
        X_te: np.ndarray,
        y_is_binary: bool,
    ) -> np.ndarray:
        """Fit arm-specific outcome nuisance g_k(x) and predict on held-out fold."""
        treatment_mask_train = (d_tr_onehot[:, k] == 1)
        if not np.any(treatment_mask_train):
            treatment_name = self.data.treatments.columns[k]
            raise RuntimeError(
                f"IRM: A CV fold has no observations for treatment '{treatment_name}' "
                "in the training split. This violates nuisance estimation requirements. "
                "Consider reducing n_folds or increasing sample size."
            )
        X_g, y_g = X_tr[treatment_mask_train], y_tr[treatment_mask_train]

        if y_is_binary:
            uniq_y = np.unique(y_g)
            if uniq_y.size == 1:
                # Rare-event folds can be single-class within an arm; deterministic predictions
                # avoid classifier crashes while preserving cross-fitting independence.
                return np.full(X_te.shape[0], float(uniq_y[0]), dtype=float)

        model_g = clone(self.ml_g)
        model_g.fit(X_g, y_g)

        if y_is_binary and _safe_is_classifier(model_g) and hasattr(model_g, "predict_proba"):
            pred_g = self._predict_binary_outcome_probability(model_g, X_te)
        else:
            pred_g = np.asarray(model_g.predict(X_te), dtype=float).ravel()
        if y_is_binary:
            pred_g = np.clip(pred_g, 1e-12, 1 - 1e-12)
        return pred_g

    def _fit_nuisances_for_fold(
        self,
        *,
        fold_id: int,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        d: np.ndarray,
        d_strat: np.ndarray,
        y_is_binary: bool,
    ) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
        """Fit nuisance models for one fold and return held-out predictions."""
        X_tr, y_tr = X[train_idx], y[train_idx]
        d_tr_labels = d_strat[train_idx]
        d_tr_onehot = d[train_idx]
        X_te = X[test_idx]

        # Propensity nuisance m_k(x): multiclass probabilities over treatment arms.
        train_classes = np.unique(d_tr_labels)
        if train_classes.size != self.n_treatments:
            missing = [k for k in range(self.n_treatments) if k not in set(train_classes.tolist())]
            raise RuntimeError(
                "A CV fold has no observations for treatment classes "
                f"{missing}. Reduce n_folds or increase sample size."
            )
        model_m = clone(self.ml_m)
        model_m.fit(X_tr, d_tr_labels)
        m_te = _predict_propensity_matrix(model_m, X_te, n_treatments=self.n_treatments)

        # Outcome nuisances g_k(x): one model per treatment arm.
        g_te = np.empty((X_te.shape[0], self.n_treatments), dtype=float)
        for k in range(self.n_treatments):
            g_te[:, k] = self._fit_one_outcome_nuisance(
                k=k,
                X_tr=X_tr,
                y_tr=y_tr,
                d_tr_onehot=d_tr_onehot,
                X_te=X_te,
                y_is_binary=y_is_binary,
            )

        return fold_id, test_idx, g_te, m_te

    def _cross_fit_nuisances(
        self,
        *,
        X: np.ndarray,
        y: np.ndarray,
        d: np.ndarray,
        y_is_binary: bool,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run cross-fitting for propensity and arm-specific outcome nuisances."""
        n = X.shape[0]
        g_hat = np.full((n, self.n_treatments), np.nan, dtype=float)
        m_hat = np.full((n, self.n_treatments), np.nan, dtype=float)
        folds = np.full(n, -1, dtype=int)

        d_strat = d.argmax(axis=1)
        self._validate_class_support(d_strat)

        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        splits = list(skf.split(X, d_strat))

        if self.n_jobs == 1:
            fold_results = [
                self._fit_nuisances_for_fold(
                    fold_id=i,
                    train_idx=train_idx,
                    test_idx=test_idx,
                    X=X,
                    y=y,
                    d=d,
                    d_strat=d_strat,
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
                    d_strat=d_strat,
                    y_is_binary=y_is_binary,
                )
                for i, (train_idx, test_idx) in enumerate(splits)
            )

        for fold_id, test_idx, g_te, m_te in fold_results:
            folds[test_idx] = fold_id
            g_hat[test_idx] = g_te
            m_hat[test_idx] = m_te

        if np.any(np.isnan(m_hat)) or np.any(np.isnan(g_hat)):
            raise RuntimeError("Cross-fitted predictions contain NaN values.")
        return g_hat, m_hat, folds

    def fit(
        self,
        data: Optional[MultiCausalData] = None,
        *,
        store_diagnostics: Optional[bool] = None,
    ) -> "MultiTreatmentIRM":
        if data is not None:
            self.data = data
        if store_diagnostics is not None:
            self.store_diagnostics = bool(store_diagnostics)
        self._fit_store_diagnostics_ = bool(self.store_diagnostics)
        if self.data is None:
            raise ValueError("Model must be provided with MultiCausalData either in __init__ or in .fit(data_contracts).")
        X, y, d, y_is_binary, n_treatments = self._check_data()
        self.n_treatments = n_treatments

        self._initialize_default_learners_for_fit(y_is_binary=y_is_binary)
        self._ensure_learners_available()
        self._configure_default_learner_parallelism()
        self._validate_fit_config(y_is_binary=y_is_binary)

        self._store_fit_sample(X=X, y=y, d=d)

        g_hat, m_hat_raw, folds = self._cross_fit_nuisances(X=X, y=y, d=d, y_is_binary=y_is_binary)
        self._store_cross_fitted_predictions(g_hat=g_hat, m_hat=m_hat_raw, folds=folds)
        return self

    def _validate_estimate_request(self, score: str, alpha: float) -> str:
        if not (0.0 < float(alpha) < 1.0):
            raise ValueError("alpha must be in (0, 1).")
        score_u = str(score).upper()
        if score_u not in {"ATE", "ATTE"}:
            raise RuntimeError("Only ATE and ATTE are supported")
        return score_u

    def _compute_score_terms(
        self, *, y: np.ndarray, d: np.ndarray, g_hat: np.ndarray, m_hat: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute orthogonal score building blocks for multi-arm ATE or ATTE contrasts."""
        y_col = y.reshape(-1, 1)
        u = y_col - g_hat
        score_u = str(self.score).upper()
        h = self._normalize_ipw_terms(d, m_hat, score=score_u)
        if score_u == "ATE":
            # For each active arm k>0, compare k vs baseline arm 0 in one vectorized expression.
            psi_b = (
                (g_hat[:, 1:] - g_hat[:, [0]])
                + (u[:, 1:] * h[:, 1:])
                - (u[:, [0]] * h[:, [0]])
            )
        else:
            g0_hat = g_hat[:, [0]]
            residual0 = y_col - g0_hat
            d0 = d[:, [0]].astype(float)
            dk = d[:, 1:].astype(float)
            pk = dk.mean(axis=0)
            if np.any(pk <= 0.0):
                raise RuntimeError("ATTE requested but some active treatment arms have zero sample share.")
            ratio = m_hat[:, 1:] / m_hat[:, [0]]
            psi_b = (dk / pk[None, :]) * residual0 - (d0 / pk[None, :]) * ratio * residual0
        psi_a = -np.ones(y.shape[0], dtype=float)
        return y_col, u, h, psi_a, psi_b

    def _solve_moment_and_inference(
        self,
        *,
        psi_a: np.ndarray,
        psi_b: np.ndarray,
        alpha: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """Solve E_n[psi_a * theta + psi_b] = 0 and compute Wald inference."""
        n = psi_a.shape[0]
        J = float(np.mean(psi_a))
        if abs(J) < 1e-16:
            theta_hat = np.full(self.n_treatments - 1, np.nan, dtype=float)
            influence = np.zeros((n, self.n_treatments - 1), dtype=float)
            se = np.full(self.n_treatments - 1, np.nan, dtype=float)
        else:
            theta_hat = -np.mean(psi_b, axis=0) / J
            psi_res = psi_b + psi_a[:, None] * theta_hat[None, :]
            influence = -psi_res / J
            var = (
                np.var(influence, axis=0, ddof=1) / n
                if n > 1
                else np.full(influence.shape[1], np.nan, dtype=float)
            )
            se = np.sqrt(np.maximum(var, 0.0))

        t_stat = np.where(se > 0, theta_hat / se, np.nan)
        pval = np.full_like(t_stat, np.nan, dtype=float)
        finite = np.isfinite(t_stat)
        pval[finite] = 2 * (1 - norm.cdf(np.abs(t_stat[finite])))
        z = float(norm.ppf(1 - alpha / 2.0))
        ci_low = theta_hat - z * se
        ci_high = theta_hat + z * se
        return theta_hat, influence, se, t_stat, pval, ci_low, ci_high, z

    def _compute_relative_effect_inference(
        self,
        *,
        score: str,
        theta_hat: np.ndarray,
        se: np.ndarray,
        psi_b: np.ndarray,
        influence: np.ndarray,
        y_col: np.ndarray,
        d: np.ndarray,
        g_hat: np.ndarray,
        u: np.ndarray,
        h: np.ndarray,
        m_hat: np.ndarray,
        z: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Compute relative effect (% vs baseline) and delta-method interval."""
        score_u = str(score).upper()
        n = u.shape[0]
        if score_u == "ATE":
            psi_mu_c = (g_hat[:, [0]] + u[:, [0]] * h[:, [0]]).ravel()
            mu_c = float(psi_mu_c.mean())

            if mu_c == 0:
                nan_vec = np.full_like(theta_hat, np.nan, dtype=float)
                return nan_vec, nan_vec, nan_vec, None

            se_mu_c = psi_mu_c.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
            psi_mu_c_centered = psi_mu_c - mu_c
            psi_b_centered = psi_b - theta_hat
            if n > 1:
                cov_theta_mu_c = (psi_b_centered * psi_mu_c_centered[:, None]).mean(axis=0) / n
            else:
                cov_theta_mu_c = np.full(theta_hat.shape, np.nan, dtype=float)

            tau_rel = 100.0 * theta_hat / mu_c
            d_theta = 100.0 / mu_c
            d_mu = -100.0 * theta_hat / (mu_c ** 2)
            var_rel = (
                (d_theta ** 2) * (se ** 2)
                + (d_mu ** 2) * (se_mu_c ** 2)
                + 2.0 * d_theta * d_mu * cov_theta_mu_c
            )
            se_rel = np.sqrt(np.maximum(var_rel, 0.0))
            ci_low_rel = tau_rel - z * se_rel
            ci_high_rel = tau_rel + z * se_rel
            return tau_rel, ci_low_rel, ci_high_rel, None

        g0_hat = g_hat[:, [0]]
        residual0 = y_col - g0_hat
        d0 = d[:, [0]].astype(float)
        dk = d[:, 1:].astype(float)
        pk = dk.mean(axis=0)
        ratio = m_hat[:, 1:] / m_hat[:, [0]]
        psi_mu_c = (dk / pk[None, :]) * g0_hat + (ratio / pk[None, :]) * d0 * residual0
        mu_c = np.mean(psi_mu_c, axis=0)
        psi_mu_c_centered = psi_mu_c - mu_c[None, :]

        tau_rel = np.full_like(theta_hat, np.nan, dtype=float)
        ci_low_rel = np.full_like(theta_hat, np.nan, dtype=float)
        ci_high_rel = np.full_like(theta_hat, np.nan, dtype=float)
        valid = np.isfinite(mu_c) & (np.abs(mu_c) > 0.0)
        if np.any(valid):
            tau_rel[valid] = 100.0 * theta_hat[valid] / mu_c[valid]
            with np.errstate(divide="ignore", invalid="ignore"):
                if_rel = 100.0 * (
                    influence[:, valid] / mu_c[None, valid]
                    - (theta_hat[None, valid] * psi_mu_c_centered[:, valid]) / (mu_c[None, valid] ** 2)
                )
            if n > 1:
                var_rel = np.var(if_rel, axis=0, ddof=1) / n
            else:
                var_rel = np.full(np.sum(valid), np.nan, dtype=float)
            se_rel = np.sqrt(np.maximum(var_rel, 0.0))
            ci_low_rel[valid] = tau_rel[valid] - z * se_rel
            ci_high_rel[valid] = tau_rel[valid] + z * se_rel
        return tau_rel, ci_low_rel, ci_high_rel, mu_c

    def _build_estimate_diagnostic_data(
        self,
        *,
        diagnostic_data: bool,
        y_col: np.ndarray,
        d: np.ndarray,
        g_hat: np.ndarray,
        m_hat: np.ndarray,
        psi_b: np.ndarray,
        influence: np.ndarray,
        score: str,
        normalize_ipw_effective: bool,
        x: Optional[np.ndarray],
    ) -> Optional[MultiUnconfoundednessDiagnosticData]:
        if (not diagnostic_data) or x is None:
            return None
        sens_elements = {}
        if score == "ATE":
            sens_elements = self._sensitivity_element_est(
                y=y_col, d=d, g_hat=g_hat, m_hat=m_hat, psi=influence
            )
        diag = MultiUnconfoundednessDiagnosticData(
            m_hat=m_hat,
            m_hat_raw=getattr(self, "m_hat_raw_", None),
            d=d,
            y=y_col,
            x=np.asarray(x, dtype=float),
            g_hat=g_hat,
            psi_b=psi_b,
            folds=self.folds_,
            trimming_threshold=self.trimming_threshold,
            normalize_ipw=normalize_ipw_effective,
            score=score,
            **sens_elements,
        )
        diag._model = self
        return diag

    def estimate(
        self, score: str = "ATE", alpha: float = 0.05, diagnostic_data: bool = True
    ) -> MultiCausalEstimate:
        check_is_fitted(self, attributes=["g_hat_", "m_hat_"])
        score_u = self._validate_estimate_request(score=score, alpha=alpha)
        self.score = score_u
        normalize_ipw_effective = self._use_normalized_ipw(score=score_u)
        self.normalize_ipw_effective_ = normalize_ipw_effective

        y, d = self._resolve_estimation_targets()
        g_hat, m_hat = self.g_hat_, self.m_hat_
        y_col, u, h, psi_a, psi_b = self._compute_score_terms(y=y, d=d, g_hat=g_hat, m_hat=m_hat)
        theta_hat, influence, se, t_stat, pval, ci_low, ci_high, z = self._solve_moment_and_inference(
            psi_a=psi_a, psi_b=psi_b, alpha=alpha
        )
        tau_rel, ci_low_rel, ci_high_rel, control_mean_by_arm = self._compute_relative_effect_inference(
            score=score_u,
            theta_hat=theta_hat,
            se=se,
            psi_b=psi_b,
            influence=influence,
            y_col=y_col,
            d=d,
            g_hat=g_hat,
            u=u,
            h=h,
            m_hat=m_hat,
            z=z,
        )
        has_fit_features = getattr(self, "_fit_store_diagnostics_", self.store_diagnostics)
        x = self._resolve_diagnostic_features() if (diagnostic_data and has_fit_features) else None
        diag = self._build_estimate_diagnostic_data(
            diagnostic_data=diagnostic_data,
            y_col=y_col,
            d=d,
            g_hat=g_hat,
            m_hat=m_hat,
            psi_b=psi_b,
            influence=influence,
            score=score_u,
            normalize_ipw_effective=normalize_ipw_effective,
            x=x,
        )
        treatment_cols = list(self.data.treatments.columns)
        baseline_treatment = treatment_cols[0]
        active_treatments = treatment_cols[1:]
        contrast_labels = [f"{t} vs {baseline_treatment}" for t in active_treatments]

        n_treated_by_arm = d[:, 1:].sum(axis=0).astype(int)
        treatment_mean = np.array(
            [float(np.mean(y[d[:, k] == 1])) if np.any(d[:, k] == 1) else np.nan for k in range(1, self.n_treatments)],
            dtype=float,
        )
        control_mean = float(np.mean(y[d[:, 0] == 1])) if np.any(d[:, 0] == 1) else np.nan

        results = MultiCausalEstimate(
            estimand=score_u,
            model="MultiTreatmentIRM",
            model_options={
                "n_folds": self.n_folds,
                "n_rep": self.n_rep,
                "normalize_ipw": normalize_ipw_effective,
                "normalize_ipw_requested": self.normalize_ipw,
                "normalize_ipw_effective": normalize_ipw_effective,
                "trimming_rule": self.trimming_rule,
                "trimming_threshold": self.trimming_threshold,
                "random_state": self.random_state,
                "n_jobs": self.n_jobs,
                "std_error": se,
                "t_stat": t_stat,
            },
            value=theta_hat,
            ci_upper_absolute=ci_high,
            ci_lower_absolute=ci_low,
            value_relative=tau_rel,
            ci_upper_relative=ci_high_rel,
            ci_lower_relative=ci_low_rel,
            alpha=alpha,
            p_value=pval,
            is_significant=np.where(
                np.isfinite(pval),
                pval < alpha / max(self.n_treatments - 1, 1),
                False
            ),
            n_treated=int(np.sum(d[:, 1:] == 1)),
            n_control=int(np.sum(d[:, 0] == 1)),
            outcome=self.data.outcome,
            treatment=treatment_cols,
            n_treated_by_arm=n_treated_by_arm,
            treatment_mean=treatment_mean,
            control_mean=control_mean,
            control_mean_by_arm=control_mean_by_arm,
            contrast_labels=contrast_labels,
            confounders=list(self.data.confounders),
            time=datetime.now().strftime("%Y-%m-%d"),
            diagnostic_data=diag,
        )

        # Cache inference artifacts for downstream diagnostics/sensitivity APIs.
        self.psi_a_ = psi_a
        self.psi_b_ = psi_b
        self.psi_ = influence
        self.coef_ = np.array(theta_hat)
        self.se_ = np.array(se)
        self.t_stat_ = np.array(t_stat)
        self.pval_ = np.array(pval)
        self.confint_ = np.array([ci_low, ci_high])
        self.summary_ = results.summary()
        return results

    @property
    def diagnostics_(self) -> Dict[str, Any]:
        """Return diagnostic data.

                Returns
                -------
                dict
                    Dictionary containing 'm_hat', 'g_hat' and 'folds'.
                """
        check_is_fitted(self, attributes=["m_hat_", "g_hat_"])
        return {
            "m_hat": self.m_hat_,
            "m_hat_raw": getattr(self, "m_hat_raw_", None),
            "g_hat": self.g_hat_,
            "folds": self.folds_,
        }

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

    def _sensitivity_element_est(
            self,
            y: Optional[np.ndarray] = None,
            d: Optional[np.ndarray] = None,
            g_hat: Optional[np.ndarray] = None,
            m_hat: Optional[np.ndarray] = None,
            psi: Optional[np.ndarray] = None,
    ) -> dict:
        if any(getattr(self, attr, None) is None for attr in ["g_hat_", "m_hat_"]):
            raise RuntimeError("Model must be fitted before sensitivity analysis.")
        if str(getattr(self, "score", "ATE")).upper() != "ATE":
            raise RuntimeError("Sensitivity elements are implemented only for score='ATE' in MultiTreatmentIRM.")

        if y is None or d is None:
            y_cached, d_cached = self._resolve_estimation_targets()
            if y is None:
                y = y_cached
            if d is None:
                d = d_cached

        # --- get fitted nuisances ---
        if m_hat is None:
            m_hat = np.asarray(self.m_hat_, dtype=float)
        else:
            m_hat = np.asarray(m_hat, dtype=float)

        if g_hat is None:
            g_hat = np.asarray(self.g_hat_, dtype=float)
        else:
            g_hat = np.asarray(g_hat, dtype=float)

        if psi is None:
            psi = getattr(self, "psi_", None)
        psi = None if psi is None else np.asarray(psi, dtype=float)

        y = np.asarray(y, dtype=float).reshape(-1)
        d = np.asarray(d)

        # --- shape checks / normalization ---
        n = y.shape[0]

        # Expect one-hot encoded treatment matrix (n, K).
        if d.ndim != 2:
            raise ValueError(
                "Expected d as one-hot matrix of shape (n, K) for multi-treatment. "
                "If you store d as labels (n,), tell me and I'll add conversion."
            )
        if d.shape[0] != n:
            raise ValueError("y and d must have same number of rows.")
        if m_hat.shape[0] != n or g_hat.shape[0] != n:
            raise ValueError("m_hat and g_hat must have same number of rows as y.")
        if m_hat.shape[1] != d.shape[1] or g_hat.shape[1] != d.shape[1]:
            raise ValueError("d, m_hat, g_hat must have same number of treatments (columns).")

        K = d.shape[1]
        if K < 2:
            raise ValueError("Need at least two treatments for pairwise contrasts 0 vs k.")

        g_obs = np.sum(d * g_hat, axis=1)
        sigma2_score_element = (y - g_obs) ** 2
        sigma2 = float(np.mean(sigma2_score_element))
        psi_sigma2 = sigma2_score_element - sigma2

        # Use the same representer normalization as the score when normalize_ipw=True.
        with np.errstate(divide="ignore", invalid="ignore"):
            inv_p = 1.0 / m_hat  # (n, K)
        if bool(getattr(self, "normalize_ipw", False)):
            h_raw = d * inv_p
            h_mean = np.mean(h_raw, axis=0)
            h_mean = np.where(
                np.isfinite(h_mean) & (np.abs(h_mean) > 1e-12),
                h_mean,
                1.0,
            )
            inv_p = inv_p / h_mean[None, :]

        w_bar = np.ones(n, dtype=float)

        d0 = d[:, [0]]       # baseline indicator (n, 1)
        inv_p0 = inv_p[:, [0]]
        dk = d[:, 1:]        # active treatment indicators (n, K-1)
        inv_pk = inv_p[:, 1:]

        # Riesz representer for each contrast (0 vs k), k=1..K-1.
        rr = w_bar[:, None] * (dk * inv_pk - d0 * inv_p0)  # (n, K-1)
        m_alpha = (w_bar[:, None] ** 2) * (inv_pk + inv_p0)

        # DoubleML-style nu2 nuisance components for sensitivity analysis.
        nu2_score_element = 2.0 * m_alpha - rr ** 2
        nu2 = np.mean(nu2_score_element, axis=0)
        psi_nu2 = nu2_score_element - nu2[None, :]

        if psi is not None:
            if psi.shape[0] != n:
                raise ValueError("psi must have same number of rows as y.")
            if psi.ndim == 1 and (K - 1) == 1:
                psi = psi.reshape(-1, 1)
            if psi.ndim != 2 or psi.shape[1] != (K - 1):
                raise ValueError(f"psi must have shape (n, K-1) = (n, {K-1}).")

        return {
            "sigma2": sigma2,
            "psi_sigma2": psi_sigma2,
            "nu2": nu2,
            "psi_nu2": psi_nu2,
            "riesz_rep": rr,
            "m_alpha": m_alpha,
            "psi": psi,
        }

    def sensitivity_analysis(
            self,
            cf_y: Optional[float] = None,
            r2_d: Any = 0.0,
            rho: Any = 1.0,
            H0: float = 0.0,
            alpha: float = 0.05,
            *,
            r2_y: Optional[float] = None,
    ) -> "MultiTreatmentIRM":
        if str(getattr(self, "score", "ATE")).upper() != "ATE":
            raise RuntimeError("Sensitivity analysis is currently supported only for score='ATE' in MultiTreatmentIRM.")
        from causalis.scenarios.multi_unconfoundedness.refutation.unconfoundedness.sensitivity import (
            sensitivity_analysis as sa_fn,
            get_sensitivity_summary
        )

        res = sa_fn(self, cf_y=cf_y, r2_y=r2_y, r2_d=r2_d, rho=rho, H0=H0, alpha=alpha)

        self.sensitivity_result = res
        self.sensitivity_summary = get_sensitivity_summary({"model": self, "bias_aware": res})

        return self

    def confint(self) -> pd.DataFrame:
        check_is_fitted(self, attributes=["confint_"])
        treatment_cols = list(self.data.treatments.columns)
        contrast_labels = [f"{treatment_cols[0]}_vs_{t}" for t in treatment_cols[1:]]
        return pd.DataFrame(
            {
                "lower": self.confint_[0],
                "upper": self.confint_[1],
            },
            index=contrast_labels,
        )
