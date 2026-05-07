"""
Lazy Conditional Average Treatment Effect (CATE) and uplift scoring helpers.

This module provides functionality to predict heterogeneous treatment effects (CATE)
for new observations using models fitted under the unconfoundedness assumption.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.utils.validation import check_is_fitted

from causalis.dgp.causaldata import CausalData
from causalis.scenarios.unconfoundedness._utils import _predict_prob_or_value


@dataclass
class _ConstantOutcomeModel:
    """
    Minimal prediction model for single-class outcome arms.

    Used when a treatment arm in the training data has only one unique outcome value.

    Attributes
    ----------
    value : float
        The constant value to predict.
    """

    value: float

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the constant value for all inputs."""
        return np.full(np.asarray(X).shape[0], self.value, dtype=float)


def _validate_fitted_irm(irm_model: Any) -> None:
    """Validate that the object can provide lazy CATE predictions."""
    check_is_fitted(irm_model, attributes=["g0_hat_", "g1_hat_", "m_hat_"])
    if not hasattr(irm_model, "_check_data"):
        raise TypeError("predict_cate currently supports fitted IRM models only.")
    if not isinstance(getattr(irm_model, "data", None), CausalData):
        raise TypeError("predict_cate requires a fitted IRM model with CausalData.")


def _training_arrays(irm_model: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Resolve fit-time training arrays and reject data changed after fit."""
    X, y, d, y_is_binary = irm_model._check_data()
    irm_model._validate_current_data_matches_fit(X=X, y=y, d=d)
    return X, y, d, bool(y_is_binary)


def _fit_arm_model(
    irm_model: Any,
    X: np.ndarray,
    y: np.ndarray,
    d: np.ndarray,
    treatment_value: int,
    y_is_binary: bool,
):
    """Fit a full-sample outcome model for one treatment arm."""
    mask = d == treatment_value
    if not np.any(mask):
        raise RuntimeError(
            f"Cannot fit uplift scorer: no rows with treatment={treatment_value}."
        )

    y_arm = y[mask]
    if y_is_binary:
        unique_y = np.unique(y_arm)
        if unique_y.size == 1:
            return _ConstantOutcomeModel(float(unique_y[0]))

    model = clone(irm_model.ml_g)
    model.fit(X[mask], y_arm)
    return model


def _ensure_uplift_models(irm_model: Any) -> None:
    """Train and cache final full-sample scoring models on first use."""
    if hasattr(irm_model, "_uplift_g0_model_") and hasattr(
        irm_model, "_uplift_g1_model_"
    ):
        return

    X_train, y_train, d_train, y_is_binary = _training_arrays(irm_model)
    irm_model._uplift_g0_model_ = _fit_arm_model(
        irm_model,
        X_train,
        y_train,
        d_train,
        treatment_value=0,
        y_is_binary=y_is_binary,
    )
    irm_model._uplift_g1_model_ = _fit_arm_model(
        irm_model,
        X_train,
        y_train,
        d_train,
        treatment_value=1,
        y_is_binary=y_is_binary,
    )
    irm_model._uplift_feature_names_ = tuple(irm_model.data.confounders)
    irm_model._uplift_y_is_binary_ = bool(y_is_binary)


def _validate_scoring_features(
    irm_model: Any, X: pd.DataFrame | np.ndarray
) -> np.ndarray:
    """Validate and coerce new-client features to the fitted confounder order."""
    feature_names = list(irm_model.data.confounders)
    if len(feature_names) == 0:
        raise ValueError(
            "predict_cate requires the fitted IRM model to have confounders."
        )

    if isinstance(X, pd.DataFrame):
        missing = [col for col in feature_names if col not in X.columns]
        if missing:
            raise ValueError(
                f"Scoring data is missing required feature column(s): {missing}."
            )
        X_ordered = X.loc[:, feature_names]
        if X_ordered.isna().any().any():
            raise ValueError(
                "Scoring data contains NaN values in required feature columns."
            )
        try:
            X_np = X_ordered.to_numpy(dtype=float, copy=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("Scoring feature columns must be numeric.") from exc
    else:
        try:
            X_np = np.asarray(X, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("Scoring features must be numeric.") from exc
        if X_np.ndim == 1:
            X_np = X_np.reshape(1, -1)
        if X_np.ndim != 2:
            raise ValueError("Scoring features must be a 2D array or DataFrame.")
        if X_np.shape[1] != len(feature_names):
            raise ValueError(
                f"Scoring features must have {len(feature_names)} columns matching the fitted confounders; "
                f"got {X_np.shape[1]}."
            )

    if not np.all(np.isfinite(X_np)):
        raise ValueError("Scoring features must contain only finite values.")
    return X_np


def predict_cate(irm_model: Any, X: pd.DataFrame | np.ndarray) -> np.ndarray:
    r"""
    Predict Conditional Average Treatment Effect (CATE), or uplift, for new observations.

    This function implements a "lazy" T-learner approach. On its first call for a given
    fitted :class:`~causalis.scenarios.unconfoundedness.model.IRM` model, it trains two
    separate full-sample outcome models—one for the control group ($D=0$) and one for the
    treatment group ($D=1$). These models are then cached on the ``irm_model`` object for
    subsequent calls.

    Mathematical Note
    -----------------
    The CATE at a point $x$ is defined as the difference in potential outcomes:

    .. math::

        \tau(x) = \mathbb{E}[Y(1) - Y(0) \mid X = x]

    Under the assumption of unconfoundedness (no unmeasured confounders), this can be
    estimated as:

    .. math::

        \tau(x) = \mathbb{E}[Y \mid X = x, D = 1] - \mathbb{E}[Y \mid X = x, D = 0]

    The estimator $\hat{\tau}(x)$ used here is:

    .. math::

        \hat{\tau}(x) = \hat{g}_1(x) - \hat{g}_0(x)

    where $\hat{g}_1$ is a machine learning model (cloned from ``irm_model.ml_g``) trained
    on the subset of the original training data where $D_i = 1$, and $\hat{g}_0$ is
    similarly trained on the subset where $D_i = 0$.

    Parameters
    ----------
    irm_model : IRM
        A fitted :class:`~causalis.scenarios.unconfoundedness.model.IRM` object.
        The model must have been fitted using :meth:`~causalis.scenarios.unconfoundedness.model.IRM.fit`.
    X : pd.DataFrame or np.ndarray
        New observations for which to predict the CATE.
        If a DataFrame, it must contain the same confounder columns as used during fit.
        If an ndarray, it must have the same number of columns as the fitted confounders.

    Returns
    -------
    np.ndarray
        A 1D array of predicted CATE/uplift values for each row in ``X``.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from causalis.scenarios.unconfoundedness.model import IRM
    >>> from causalis.scenarios.unconfoundedness.dgp import generate_obs_hte_26
    >>> from causalis.scenarios.uplift.model import predict_cate
    >>> # Generate data
    >>> data = generate_obs_hte_26(n=1000, seed=42)
    >>> # Fit IRM model
    >>> irm = IRM(data=data).fit()
    >>> # Predict uplift for new data
    >>> X_new = data.df.loc[:5, data.confounders]
    >>> uplift = predict_cate(irm, X_new)
    >>> print(uplift.shape)
    (6,)

    Notes
    -----
    The models trained for prediction use the full sample (split by treatment arm)
    available at fit time, whereas the :meth:`~causalis.scenarios.unconfoundedness.model.IRM.fit`
    method uses cross-fitting for nuisance estimation.
    """
    _validate_fitted_irm(irm_model)
    X_np = _validate_scoring_features(irm_model, X)
    _ensure_uplift_models(irm_model)

    y0 = _predict_prob_or_value(irm_model._uplift_g0_model_, X_np, is_propensity=False)
    y1 = _predict_prob_or_value(irm_model._uplift_g1_model_, X_np, is_propensity=False)
    return np.asarray(y1 - y0, dtype=float).ravel()


__all__ = ["predict_cate"]
