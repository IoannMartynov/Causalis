"""Instrumental Interactive Regression Model for binary IV LATE estimation."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import norm
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.validation import check_is_fitted

try:
    from catboost import CatBoostClassifier, CatBoostRegressor

    HAS_CATBOOST = True
except ImportError:  # pragma: no cover - exercised only without optional runtime dep
    HAS_CATBOOST = False

from causalis.data_contracts.causal_diagnostic_data import IVDiagnosticData
from causalis.data_contracts.iv_causal_estimate import IVCausalEstimate
from causalis.data_contracts.iv_causal_data import IVCausalData
from causalis.scenarios.unconfoundedness._utils import (
    _is_binary,
    _predict_prob_or_value,
    _safe_is_classifier,
)


def _clip_iv_propensity(p: np.ndarray, threshold: float) -> np.ndarray:
    """Clip IV propensity scores to the configured trimming interval."""
    threshold_f = float(threshold)
    if not np.isfinite(threshold_f) or not (0.0 <= threshold_f < 0.5):
        raise ValueError("trimming_threshold must be finite and in [0, 0.5).")
    return np.clip(p, threshold_f, 1.0 - threshold_f)


class IIVM(BaseEstimator):
    r"""DoubleML-style IIVM estimator for LATE with binary treatment and IV.

    The model consumes :class:`~causalis.data_contracts.IVCausalData`, which
    stores exactly one binary instrument. It cross-fits nuisance functions:

    .. math::

        g_0(z, X) = \mathbb{E}[Y \mid Z=z, X],
        \quad
        r_0(z, X) = \mathbb{E}[D \mid Z=z, X],
        \quad
        m_0(X) = \mathbb{P}(Z=1 \mid X).

    ``estimate(score="LATE")`` then solves the linear orthogonal score

    .. math::

        \psi(W; \theta, \eta) = \phi_Y(W; \eta) - \theta \phi_D(W; \eta),

    returning

    .. math::

        \hat\theta = \mathbb{E}_n[\phi_Y] / \mathbb{E}_n[\phi_D].

    where the orthogonal signals are:

    .. math::

        \phi_Y(W; \eta) &= g(1, X) - g(0, X) + \frac{Z(Y - g(1, X))}{m(X)} - \frac{(1-Z)(Y - g(0, X))}{1 - m(X)} \\
        \phi_D(W; \eta) &= r(1, X) - r(0, X) + \frac{Z(D - r(1, X))}{m(X)} - \frac{(1-Z)(D - r(0, X))}{1 - m(X)}

    Notes
    -----
    The Local Average Treatment Effect (LATE) is the effect of the treatment among "compliers"
    — those whose treatment status is changed by the instrument.

    Examples
    --------
    >>> from causalis.scenarios.iv.dgp import generate_offer_iv_26
    >>> from causalis.data_contracts.iv_causal_data import IVCausalData
    >>> from causalis.scenarios.iv.model import IIVM
    >>> data = generate_offer_iv_26(n=5000, return_causal_data=False)
    >>> causal_data = IVCausalData.from_df(
    ...     df=data,
    ...     treatment='accepted_offer',
    ...     outcome='net_revenue_90d',
    ...     instruments='offer_eligible',
    ...     confounders=['age', 'tenure_months', 'annual_income']
    ... )
    >>> model = IIVM()
    >>> model.fit(causal_data)
    >>> result = model.estimate(score="LATE")
    >>> result.summary()
    """

    def __init__(
        self,
        data: Optional[IVCausalData] = None,
        ml_g: Any = None,
        ml_m: Any = None,
        ml_r: Any = None,
        *,
        n_folds: int = 5,
        n_rep: int = 1,
        normalize_ipw: bool = False,
        trimming_rule: str = "truncate",
        trimming_threshold: float = 1e-2,
        weak_iv_threshold: float = 1e-2,
        random_state: Optional[int] = None,
        n_jobs: int = 1,
    ) -> None:
        self.data = data
        self.ml_g = ml_g
        self.ml_m = ml_m
        self.ml_r = ml_r
        self._ml_g_is_default = False
        self._ml_m_is_default = False
        self._ml_r_is_default = False
        self.n_folds = int(n_folds)
        self.n_rep = int(n_rep)
        self.normalize_ipw = bool(normalize_ipw)
        self.trimming_rule = str(trimming_rule)
        self.trimming_threshold = float(trimming_threshold)
        self.weak_iv_threshold = float(weak_iv_threshold)
        self.random_state = random_state
        self.n_jobs = int(n_jobs)

        if HAS_CATBOOST:
            if self.ml_m is None:
                self.ml_m = CatBoostClassifier(
                    thread_count=-1,
                    logging_level='Silent',
                    allow_writing_files=False,
                    random_seed=self.random_state,
                )
                self._ml_m_is_default = True
            if self.ml_r is None:
                self.ml_r = CatBoostClassifier(
                    thread_count=-1,
                    logging_level='Silent',
                    allow_writing_files=False,
                    random_seed=self.random_state,
                )
                self._ml_r_is_default = True

            if self.ml_g is None and self.data is not None:
                y_is_binary = False
                try:
                    y = self.data.get_df()[self.data.outcome_name].to_numpy(
                        dtype=float, copy=False
                    )
                    y_is_binary = _is_binary(y)
                except (AttributeError, KeyError, ValueError):
                    pass
                self.ml_g = self._make_default_g_learner(y_is_binary=y_is_binary)
                self._ml_g_is_default = True

        self._validate_init_config()

    def _validate_init_config(self) -> None:
        """Validate static estimator configuration."""
        if self.n_jobs == 0 or self.n_jobs < -1:
            raise ValueError("n_jobs must be -1 or a positive integer.")
        if self.n_rep != 1:
            raise NotImplementedError("IIVM currently supports n_rep=1 only.")
        if self.n_folds < 2:
            raise ValueError("n_folds must be at least 2.")
        if self.trimming_rule not in {"truncate"}:
            raise ValueError("Only trimming_rule='truncate' is supported.")
        if not np.isfinite(self.trimming_threshold) or not (
            0.0 <= self.trimming_threshold < 0.5
        ):
            raise ValueError("trimming_threshold must be finite and in [0, 0.5).")
        if not np.isfinite(self.weak_iv_threshold) or self.weak_iv_threshold < 0.0:
            raise ValueError("weak_iv_threshold must be finite and non-negative.")

    def _make_default_g_learner(self, *, y_is_binary: bool) -> Any:
        """Build a CatBoost default learner for the outcome nuisance."""
        if not HAS_CATBOOST:
            return None
        if y_is_binary:
            return CatBoostClassifier(
                thread_count=-1,
                logging_level='Silent',
                allow_writing_files=False,
                random_seed=self.random_state,
            )
        return CatBoostRegressor(
            thread_count=-1,
            logging_level='Silent',
            allow_writing_files=False,
            random_seed=self.random_state,
        )

    def _initialize_default_learners_for_fit(self, *, y_is_binary: bool) -> None:
        """Initialize missing default learners once the outcome type is known."""
        if not HAS_CATBOOST:
            return
        if self.ml_g is None:
            self.ml_g = self._make_default_g_learner(y_is_binary=y_is_binary)
            self._ml_g_is_default = True
        if self.ml_m is None:
            self.ml_m = CatBoostClassifier(
                thread_count=-1,
                logging_level='Silent',
                allow_writing_files=False,
                random_seed=self.random_state,
            )
            self._ml_m_is_default = True
        if self.ml_r is None:
            self.ml_r = CatBoostClassifier(
                thread_count=-1,
                logging_level='Silent',
                allow_writing_files=False,
                random_seed=self.random_state,
            )
            self._ml_r_is_default = True

    def _configure_default_learner_parallelism(self) -> None:
        """Avoid CPU oversubscription for default CatBoost learners."""
        if self.n_jobs == 1 or not HAS_CATBOOST:
            return
        for estimator, is_default in (
            (self.ml_g, self._ml_g_is_default),
            (self.ml_m, self._ml_m_is_default),
            (self.ml_r, self._ml_r_is_default),
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
        """Ensure all nuisance learners are configured."""
        if self.ml_g is None or self.ml_m is None or self.ml_r is None:
            raise ValueError(
                "ml_g, ml_m, and ml_r must be provided when CatBoost defaults "
                "are unavailable."
            )

    def _validate_fit_learners(self, *, y_is_binary: bool) -> None:
        """Validate learner interfaces needed by the score."""
        if _safe_is_classifier(self.ml_m) and not hasattr(self.ml_m, "predict_proba"):
            raise ValueError("ml_m must expose predict_proba().")
        if _safe_is_classifier(self.ml_r) and not hasattr(self.ml_r, "predict_proba"):
            raise ValueError("ml_r must expose predict_proba().")
        if (
            y_is_binary
            and _safe_is_classifier(self.ml_g)
            and not hasattr(self.ml_g, "predict_proba")
        ):
            raise ValueError(
                "Binary outcome: ml_g is a classifier but does not expose "
                "predict_proba()."
            )

    def _check_data(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
        """Validate and prepare IVCausalData arrays."""
        if self.data is None:
            raise ValueError(
                "Model must be provided with IVCausalData either in __init__ "
                "or in .fit(data)."
            )
        if not isinstance(self.data, IVCausalData):
            raise TypeError("IIVM.fit() requires an IVCausalData instance.")
        if len(self.data.instruments_names) != 1:
            raise ValueError(
                "IIVM supports exactly one binary instrument. For multiple "
                "instruments, use another IV scenario."
            )

        y_col = self.data.outcome_name
        d_col = self.data.treatment_name
        z_col = self.data.instruments_names[0]
        x_cols = list(self.data.confounders_names)
        df = self.data.get_df(
            include_outcome=True,
            include_treatment=True,
            include_instruments=True,
            include_confounders=True,
            include_user_id=False,
        )

        y = df[y_col].to_numpy(dtype=float, copy=False)
        d = df[d_col].to_numpy(copy=False)
        z = df[z_col].to_numpy(copy=False)
        if df[d_col].dtype == bool:
            d = d.astype(int)
        if df[z_col].dtype == bool:
            z = z.astype(int)
        d = np.asarray(d, dtype=int).ravel()
        z = np.asarray(z, dtype=int).ravel()

        if not _is_binary(d):
            raise ValueError("IIVM requires a binary treatment D encoded as 0/1.")
        if not _is_binary(z):
            raise ValueError("IIVM requires a binary instrument Z encoded as 0/1.")

        if x_cols:
            X = df[x_cols].to_numpy(dtype=float, copy=False)
        else:
            X = np.empty((len(df), 0), dtype=float)

        if not np.all(np.isfinite(y)):
            raise ValueError("Outcome contains non-finite values.")
        if not np.all(np.isfinite(X)):
            raise ValueError("Confounders contain non-finite values.")

        y_is_binary = _is_binary(y)
        return X, y, d, z, y_is_binary

    def _validate_support(self, *, d: np.ndarray, z: np.ndarray) -> None:
        """Check treatment and instrument support for cross-fitting."""
        z_counts = np.bincount(z, minlength=2)
        d_counts = np.bincount(d, minlength=2)
        if np.any(z_counts == 0):
            missing = np.where(z_counts == 0)[0].tolist()
            raise RuntimeError(f"Missing instrument classes in data: {missing}.")
        if np.any(d_counts == 0):
            missing = np.where(d_counts == 0)[0].tolist()
            raise RuntimeError(f"Missing treatment classes in data: {missing}.")
        min_z = int(z_counts.min())
        min_d = int(d_counts.min())
        if self.n_folds > min_z:
            raise ValueError(
                f"n_folds={self.n_folds} exceeds minimum instrument class "
                f"count={min_z}. Reduce n_folds or collect more data."
            )
        if self.n_folds > min_d:
            raise ValueError(
                f"n_folds={self.n_folds} exceeds minimum treatment class "
                f"count={min_d}. Reduce n_folds or collect more data."
            )

    @staticmethod
    def _augment_z_x(z: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Create the (Z, X) feature matrix used by g and r nuisances."""
        z_col = np.asarray(z, dtype=float).reshape(-1, 1)
        if X.shape[1] == 0:
            return z_col
        return np.column_stack([z_col, X])

    @staticmethod
    def _constant_prediction(value: float, n: int) -> np.ndarray:
        """Return a constant prediction vector."""
        return np.full(n, float(value), dtype=float)

    def _fit_m_for_fold(
        self,
        *,
        X_tr: np.ndarray,
        z_tr: np.ndarray,
        X_te: np.ndarray,
    ) -> Tuple[Optional[Any], np.ndarray]:
        """Fit m(X)=P(Z=1|X), with an intercept-only path for empty X."""
        if X_tr.shape[1] == 0:
            return None, self._constant_prediction(np.mean(z_tr), X_te.shape[0])

        model_m = clone(self.ml_m)
        model_m.fit(X_tr, z_tr)
        m_te = _predict_prob_or_value(model_m, X_te, is_propensity=True)
        return model_m, m_te

    def _fit_g_for_fold(
        self,
        *,
        ZX_tr: np.ndarray,
        y_tr: np.ndarray,
        ZX_te_z0: np.ndarray,
        ZX_te_z1: np.ndarray,
        y_is_binary: bool,
    ) -> Tuple[Optional[Any], np.ndarray, np.ndarray]:
        """Fit g(Z,X)=E[Y|Z,X] and predict both instrument arms."""
        if y_is_binary and _safe_is_classifier(self.ml_g):
            uniq_y = np.unique(y_tr)
            if uniq_y.size == 1:
                pred = self._constant_prediction(float(uniq_y[0]), ZX_te_z0.shape[0])
                return None, pred, pred.copy()

        model_g = clone(self.ml_g)
        model_g.fit(ZX_tr, y_tr)
        g0_te = _predict_prob_or_value(model_g, ZX_te_z0, is_propensity=False)
        g1_te = _predict_prob_or_value(model_g, ZX_te_z1, is_propensity=False)
        if y_is_binary:
            g0_te = np.clip(g0_te, 1e-12, 1.0 - 1e-12)
            g1_te = np.clip(g1_te, 1e-12, 1.0 - 1e-12)
        return model_g, g0_te, g1_te

    def _fit_r_for_fold(
        self,
        *,
        ZX_tr: np.ndarray,
        d_tr: np.ndarray,
        ZX_te_z0: np.ndarray,
        ZX_te_z1: np.ndarray,
    ) -> Tuple[Optional[Any], np.ndarray, np.ndarray]:
        """Fit r(Z,X)=P(D=1|Z,X) and predict both instrument arms."""
        uniq_d = np.unique(d_tr)
        if uniq_d.size == 1:
            pred = self._constant_prediction(float(uniq_d[0]), ZX_te_z0.shape[0])
            return None, pred, pred.copy()

        model_r = clone(self.ml_r)
        model_r.fit(ZX_tr, d_tr)
        r0_te = _predict_prob_or_value(model_r, ZX_te_z0, is_propensity=True)
        r1_te = _predict_prob_or_value(model_r, ZX_te_z1, is_propensity=True)
        return model_r, r0_te, r1_te

    def _fit_nuisances_for_fold(
        self,
        *,
        fold_id: int,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        d: np.ndarray,
        z: np.ndarray,
        y_is_binary: bool,
    ) -> Tuple[
        int,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        Optional[Any],
        Optional[Any],
        Optional[Any],
    ]:
        """Fit all nuisance models for one held-out fold."""
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, d_tr, z_tr = y[train_idx], d[train_idx], z[train_idx]

        if np.unique(z_tr).size < 2:
            raise RuntimeError(
                "IIVM: a training split has no support for both instrument "
                "classes. Reduce n_folds or collect more data."
            )
        if np.unique(d_tr).size < 2:
            raise RuntimeError(
                "IIVM: a training split has no support for both treatment "
                "classes. Reduce n_folds or collect more data."
            )

        ZX_tr = self._augment_z_x(z_tr, X_tr)
        ZX_te_z0 = self._augment_z_x(np.zeros(test_idx.shape[0], dtype=int), X_te)
        ZX_te_z1 = self._augment_z_x(np.ones(test_idx.shape[0], dtype=int), X_te)

        model_m, m_te = self._fit_m_for_fold(X_tr=X_tr, z_tr=z_tr, X_te=X_te)
        model_g, g0_te, g1_te = self._fit_g_for_fold(
            ZX_tr=ZX_tr,
            y_tr=y_tr,
            ZX_te_z0=ZX_te_z0,
            ZX_te_z1=ZX_te_z1,
            y_is_binary=y_is_binary,
        )
        model_r, r0_te, r1_te = self._fit_r_for_fold(
            ZX_tr=ZX_tr,
            d_tr=d_tr,
            ZX_te_z0=ZX_te_z0,
            ZX_te_z1=ZX_te_z1,
        )

        return (
            fold_id,
            test_idx,
            g0_te,
            g1_te,
            m_te,
            r0_te,
            r1_te,
            model_g,
            model_m,
            model_r,
        )

    def _make_cross_fit_splits(
        self, *, X: np.ndarray, d: np.ndarray, z: np.ndarray
    ) -> list[Tuple[np.ndarray, np.ndarray]]:
        """Create stratified splits, preferring joint (Z,D) stratification."""
        joint = z.astype(str) + "_" + d.astype(str)
        labels = joint
        joint_counts = pd.Series(joint).value_counts()
        if int(joint_counts.min()) < self.n_folds:
            labels = z

        skf = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True, random_state=self.random_state
        )
        splits = list(skf.split(X, labels))
        for train_idx, _ in splits:
            if np.unique(z[train_idx]).size < 2 or np.unique(d[train_idx]).size < 2:
                raise RuntimeError(
                    "IIVM could not create cross-fitting splits with both "
                    "instrument and treatment classes in every training fold. "
                    "Reduce n_folds or collect more data."
                )
        return splits

    def _cross_fit_nuisances(
        self,
        *,
        X: np.ndarray,
        y: np.ndarray,
        d: np.ndarray,
        z: np.ndarray,
        y_is_binary: bool,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, list[Any]], np.ndarray]:
        """Run cross-fitting and return nuisance predictions, models, and folds."""
        n = len(y)
        g_hat0 = np.full(n, np.nan, dtype=float)
        g_hat1 = np.full(n, np.nan, dtype=float)
        r_hat0 = np.full(n, np.nan, dtype=float)
        r_hat1 = np.full(n, np.nan, dtype=float)
        m_hat_raw = np.full(n, np.nan, dtype=float)
        folds = np.full(n, -1, dtype=int)
        fitted_models: Dict[str, list[Any]] = {"g": [], "m": [], "r": []}

        splits = self._make_cross_fit_splits(X=X, d=d, z=z)
        if self.n_jobs == 1:
            fold_results = [
                self._fit_nuisances_for_fold(
                    fold_id=i,
                    train_idx=train_idx,
                    test_idx=test_idx,
                    X=X,
                    y=y,
                    d=d,
                    z=z,
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
                    z=z,
                    y_is_binary=y_is_binary,
                )
                for i, (train_idx, test_idx) in enumerate(splits)
            )

        for (
            fold_id,
            test_idx,
            g0_te,
            g1_te,
            m_te,
            r0_te,
            r1_te,
            model_g,
            model_m,
            model_r,
        ) in fold_results:
            folds[test_idx] = fold_id
            g_hat0[test_idx] = g0_te
            g_hat1[test_idx] = g1_te
            m_hat_raw[test_idx] = m_te
            r_hat0[test_idx] = r0_te
            r_hat1[test_idx] = r1_te
            fitted_models["g"].append(model_g)
            fitted_models["m"].append(model_m)
            fitted_models["r"].append(model_r)

        predictions = {
            "g_hat0": g_hat0,
            "g_hat1": g_hat1,
            "m_hat_raw": m_hat_raw,
            "m_hat": _clip_iv_propensity(m_hat_raw, self.trimming_threshold),
            "r_hat0": r_hat0,
            "r_hat1": r_hat1,
        }
        if any(np.any(np.isnan(arr)) for arr in predictions.values()):
            raise RuntimeError("Cross-fitted predictions contain NaN values.")
        return predictions, fitted_models, folds

    def fit(self, data: Optional[IVCausalData] = None) -> "IIVM":
        """Fit cross-fitted nuisance functions for IIVM."""
        if data is not None:
            self.data = data

        X, y, d, z, y_is_binary = self._check_data()
        self._initialize_default_learners_for_fit(y_is_binary=y_is_binary)
        self._ensure_learners_available()
        self._validate_fit_learners(y_is_binary=y_is_binary)
        self._configure_default_learner_parallelism()
        self._validate_support(d=d, z=z)

        predictions, fitted_models, folds = self._cross_fit_nuisances(
            X=X, y=y, d=d, z=z, y_is_binary=y_is_binary
        )

        self.y_ = np.asarray(y, dtype=float).copy()
        self.d_ = np.asarray(d, dtype=int).copy()
        self.z_ = np.asarray(z, dtype=int).copy()
        self.X_ = np.asarray(X, dtype=float).copy()
        self.predictions_ = predictions
        self.models_ = fitted_models
        self.folds_ = folds
        self.causaldata_ = self.data
        self.g_hat0_ = predictions["g_hat0"]
        self.g_hat1_ = predictions["g_hat1"]
        self.m_hat_ = predictions["m_hat"]
        self.m_hat_raw_ = predictions["m_hat_raw"]
        self.r_hat0_ = predictions["r_hat0"]
        self.r_hat1_ = predictions["r_hat1"]
        return self

    def _compute_ipw_terms(
        self, *, z: np.ndarray, m: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute instrument-arm inverse-probability weights."""
        w_z1 = z / m
        w_z0 = (1.0 - z) / (1.0 - m)
        if self.normalize_ipw:
            mean_w1 = float(np.mean(w_z1))
            mean_w0 = float(np.mean(w_z0))
            if abs(mean_w1) < 1e-16 or abs(mean_w0) < 1e-16:
                raise RuntimeError("Cannot normalize IPW terms with zero mean weight.")
            w_z1 = w_z1 / mean_w1
            w_z0 = w_z0 / mean_w0
        return w_z1, w_z0

    def estimate(self, score: str = "LATE", alpha: float = 0.05) -> IVCausalEstimate:
        """Estimate LATE from cross-fitted IIVM nuisance predictions."""
        check_is_fitted(
            self,
            attributes=[
                "g_hat0_",
                "g_hat1_",
                "m_hat_",
                "r_hat0_",
                "r_hat1_",
                "y_",
                "d_",
                "z_",
            ],
        )
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be in (0,1).")
        score_u = str(score).upper()
        if score_u != "LATE":
            raise ValueError("IIVM currently supports only score='LATE'.")

        y = self.y_
        d = self.d_
        z = self.z_
        g0 = self.g_hat0_
        g1 = self.g_hat1_
        m = self.m_hat_
        r0 = self.r_hat0_
        r1 = self.r_hat1_
        n = len(y)

        w_z1, w_z0 = self._compute_ipw_terms(z=z, m=m)
        phi_y = g1 - g0 + w_z1 * (y - g1) - w_z0 * (y - g0)
        phi_d = r1 - r0 + w_z1 * (d - r1) - w_z0 * (d - r0)

        numerator = float(np.mean(phi_y))
        denominator = float(np.mean(phi_d))
        if abs(denominator) < 1e-8:
            raise ValueError(
                "Weak or zero first stage: mean(phi_d) is too close to zero. "
                "LATE is not numerically stable."
            )
        if abs(denominator) < self.weak_iv_threshold:
            warnings.warn(
                "The estimated first stage is weak; LATE may be unstable.",
                RuntimeWarning,
            )

        theta_hat = numerator / denominator
        psi_a = -phi_d
        psi_b = phi_y
        psi = psi_a * theta_hat + psi_b
        J = float(np.mean(psi_a))
        se = float(np.sqrt(np.mean(psi**2) / (n * J**2)))
        t_stat = theta_hat / se if se > 0 else np.nan
        p_value = 2.0 * (1.0 - norm.cdf(abs(t_stat))) if np.isfinite(t_stat) else np.nan
        z_crit = norm.ppf(1.0 - alpha / 2.0)
        ci_low = float(theta_hat - z_crit * se)
        ci_high = float(theta_hat + z_crit * se)

        clipped_propensity = bool(
            np.any(self.m_hat_raw_ <= self.trimming_threshold)
            or np.any(self.m_hat_raw_ >= 1.0 - self.trimming_threshold)
        )
        if clipped_propensity:
            warnings.warn(
                "Some instrument propensities are close to 0 or 1 and were "
                "trimmed for overlap stability.",
                RuntimeWarning,
            )

        diagnostics = {
            "n_obs": n,
            "instrument_rate": float(np.mean(z)),
            "treatment_rate": float(np.mean(d)),
            "first_stage_naive": float(np.mean(d[z == 1]) - np.mean(d[z == 0])),
            "orthogonal_first_stage": denominator,
            "late_numerator": numerator,
            "late_denominator": denominator,
            "mean_phi_y": numerator,
            "mean_phi_d": denominator,
            "mean_m_hat": float(np.mean(m)),
            "min_m_hat": float(np.min(m)),
            "max_m_hat": float(np.max(m)),
            "m_hat_p01": float(np.quantile(m, 0.01)),
            "m_hat_p99": float(np.quantile(m, 0.99)),
            "weak_iv_warning": bool(abs(denominator) < self.weak_iv_threshold),
            "overlap_warning": clipped_propensity,
            "score_mean": float(np.mean(psi)),
        }

        diagnostic_data = IVDiagnosticData(
            y=y,
            d=d,
            z=z,
            x=getattr(self, "X_", None),
            x_names=list(getattr(self.causaldata_, "confounders_names", [])),
            g0_hat=g0,
            g1_hat=g1,
            m_hat=m,
            m_hat_raw=self.m_hat_raw_,
            r0_hat=r0,
            r1_hat=r1,
            folds=getattr(self, "folds_", None),
            psi=psi,
            psi_a=psi_a,
            psi_b=psi_b,
            phi_y=phi_y,
            phi_d=phi_d,
            score=score_u,
            trimming_threshold=self.trimming_threshold,
            normalize_ipw=self.normalize_ipw,
            diagnostics=diagnostics,
        )
        from causalis.scenarios.iv.refutation.diagnostics import (
            compute_first_stage_diagnostics,
            compute_instrument_overlap_diagnostics,
            compute_reduced_form_diagnostics,
        )

        diagnostic_data.instrument_overlap = compute_instrument_overlap_diagnostics(
            diagnostic_data
        )
        diagnostic_data.first_stage = compute_first_stage_diagnostics(
            diagnostic_data,
            weak_iv_threshold=self.weak_iv_threshold,
        )
        diagnostic_data.reduced_form = compute_reduced_form_diagnostics(
            diagnostic_data,
            late_value=float(theta_hat),
        )
        diagnostic_data.diagnostics.update(
            {
                "instrument_overlap": diagnostic_data.instrument_overlap,
                "first_stage": diagnostic_data.first_stage,
                "reduced_form": diagnostic_data.reduced_form,
            }
        )

        result = IVCausalEstimate(
            estimand=score_u,
            model="IIVM",
            value=float(theta_hat),
            std_error=se,
            t_stat=float(t_stat),
            p_value=float(p_value),
            ci_lower_absolute=ci_low,
            ci_upper_absolute=ci_high,
            alpha=alpha,
            is_significant=bool(p_value < alpha) if np.isfinite(p_value) else False,
            outcome=self.causaldata_.outcome_name,
            treatment=self.causaldata_.treatment_name,
            instrument=self.causaldata_.instruments_names[0],
            confounders=list(self.causaldata_.confounders_names),
            diagnostic_data=diagnostic_data,
            model_options={
                "n_folds": self.n_folds,
                "n_rep": self.n_rep,
                "normalize_ipw": self.normalize_ipw,
                "trimming_rule": self.trimming_rule,
                "trimming_threshold": self.trimming_threshold,
                "weak_iv_threshold": self.weak_iv_threshold,
                "random_state": self.random_state,
                "n_jobs": self.n_jobs,
            },
        )

        self.result_ = result
        self.coef_ = np.array([theta_hat], dtype=float)
        self.se_ = np.array([se], dtype=float)
        self.t_stat_ = np.array([t_stat], dtype=float)
        self.pval_ = np.array([p_value], dtype=float)
        self.confint_ = np.array([[ci_low, ci_high]], dtype=float)
        self.psi_ = psi
        self.psi_a_ = psi_a
        self.psi_b_ = psi_b
        self.phi_y_ = phi_y
        self.phi_d_ = phi_d
        self.summary_ = result.summary()
        return result

    @property
    def diagnostics_(self) -> Dict[str, Any]:
        """Return fit-time diagnostic arrays."""
        check_is_fitted(self, attributes=["predictions_", "folds_"])
        return {
            "g_hat0": self.g_hat0_,
            "g_hat1": self.g_hat1_,
            "m_hat": self.m_hat_,
            "m_hat_raw": self.m_hat_raw_,
            "r_hat0": self.r_hat0_,
            "r_hat1": self.r_hat1_,
            "folds": self.folds_,
        }

    @property
    def coef(self) -> np.ndarray:
        """Return the estimated coefficient."""
        check_is_fitted(self, attributes=["coef_"])
        return self.coef_

    @property
    def se(self) -> np.ndarray:
        """Return the standard error."""
        check_is_fitted(self, attributes=["se_"])
        return self.se_

    @property
    def pvalues(self) -> np.ndarray:
        """Return p-values."""
        check_is_fitted(self, attributes=["pval_"])
        return self.pval_

    @property
    def summary(self) -> pd.DataFrame:
        """Return the latest estimate summary table."""
        check_is_fitted(self, attributes=["summary_"])
        return self.summary_

    def confint(self) -> pd.DataFrame:
        """Return the latest confidence interval as a DataFrame."""
        check_is_fitted(self, attributes=["confint_"])
        return pd.DataFrame(self.confint_, columns=["2.5 %", "97.5 %"], index=["LATE"])

    def __repr__(self) -> str:
        """Concise representation of IIVM to avoid verbose learner output."""
        status = "fitted" if hasattr(self, "predictions_") else "unfitted"
        return f"IIVM(status='{status}', n_folds={self.n_folds}, random_state={self.random_state})"

    _repr_html_ = None
    _repr_mimebundle_ = None


__all__ = ["IIVM", "IVCausalEstimate"]
