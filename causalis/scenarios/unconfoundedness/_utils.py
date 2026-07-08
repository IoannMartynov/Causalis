"""Numeric helpers for binary-treatment unconfoundedness estimators."""
from __future__ import annotations

from typing import Any, Optional
import warnings

import numpy as np
from sklearn.base import is_classifier


def _is_binary(values: np.ndarray) -> bool:
    """Check if an array contains only binary values (0 and 1)."""
    uniq = np.unique(values)
    return np.array_equal(np.sort(uniq), np.array([0, 1])) or np.array_equal(np.sort(uniq), np.array([0.0, 1.0]))


def _safe_is_classifier(estimator) -> bool:
    """Safely check if an estimator is a classifier."""
    try:
        return is_classifier(estimator)
    except (AttributeError, TypeError):
        return getattr(estimator, "_estimator_type", None) == "classifier"


def _binary_label_is_one(label: Any) -> Optional[bool]:
    """Map a binary-like class label to {False, True}, if possible."""
    if isinstance(label, (bool, np.bool_)):
        return bool(label)
    try:
        val = float(label)
    except (TypeError, ValueError):
        return None
    if np.isclose(val, 1.0):
        return True
    if np.isclose(val, 0.0):
        return False
    return None


def _predict_prob_or_value(model, X: np.ndarray, is_propensity: bool = False) -> np.ndarray:
    """Predict probabilities or values using a model."""
    if _safe_is_classifier(model) and hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(X))
        if proba.ndim == 1:
            # Assume this is already P(class=1).
            res = proba.ravel()
        elif proba.shape[1] == 1:
            # Can happen if the training fold has a single class.
            # Resolve P(class=1) from classes_ when available.
            classes = np.asarray(getattr(model, "classes_", [])).ravel()
            if classes.size == 1:
                class_is_one = _binary_label_is_one(classes[0])
                if class_is_one is True:
                    res = proba[:, 0]
                elif class_is_one is False:
                    res = np.zeros(proba.shape[0], dtype=float)
                else:
                    # Unknown class label semantics; fall back to available column.
                    res = proba[:, 0]
            else:
                # No reliable class metadata; infer from hard labels when possible.
                if hasattr(model, "predict"):
                    pred = np.asarray(model.predict(X)).ravel()
                    try:
                        pred_f = pred.astype(float)
                        res = np.where(np.isclose(pred_f, 1.0), 1.0, 0.0)
                    except (TypeError, ValueError):
                        res = proba[:, 0]
                else:
                    res = proba[:, 0]
        else:
            classes = np.asarray(getattr(model, "classes_", [])).ravel()
            pos_idx = None
            if classes.size == proba.shape[1]:
                for i, cls in enumerate(classes):
                    if _binary_label_is_one(cls) is True:
                        pos_idx = i
                        break
            if pos_idx is None:
                # Fallback to the second column when binary classes metadata is missing.
                pos_idx = 1
            res = proba[:, pos_idx]
    else:
        res = model.predict(X)

    res = np.asarray(res, dtype=float).ravel()
    if is_propensity:
        if np.any((res < -1e-12) | (res > 1.0 + 1e-12)):
            warnings.warn(
                "Propensity model produced values outside [0, 1]. "
                "Consider using a classifier or a model with a logistic link.",
                RuntimeWarning,
            )
        res = np.clip(res, 0.0, 1.0)
    return res


def _validate_overlap_config(policy: str, threshold: float) -> tuple[str, float]:
    """Validate and normalize the overlap policy configuration."""
    policy_norm = str(policy).lower()
    if policy_norm not in {"clip", "drop"}:
        raise ValueError("overlap_policy must be either 'clip' or 'drop'.")

    threshold_f = float(threshold)
    if not np.isfinite(threshold_f) or not (0.0 <= threshold_f < 0.5):
        raise ValueError("overlap_threshold must be finite and in [0, 0.5).")
    return policy_norm, threshold_f


def _overlap_retained_mask(p: np.ndarray, threshold: float) -> np.ndarray:
    """Return rows whose propensity scores satisfy strict overlap."""
    threshold_f = float(threshold)
    p_arr = np.asarray(p, dtype=float).ravel()
    return (p_arr > threshold_f) & (p_arr < 1.0 - threshold_f)


def _apply_overlap_policy(
    p: np.ndarray,
    *,
    policy: str,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the configured overlap policy to a propensity vector.

    ``clip`` returns a clipped vector with an all-true mask. ``drop`` returns
    only retained propensity scores and the full-sample boolean retention mask.
    """
    policy_norm, threshold_f = _validate_overlap_config(policy, threshold)
    p_arr = np.asarray(p, dtype=float).ravel()

    if policy_norm == "clip":
        mask = np.ones(p_arr.shape[0], dtype=bool)
        return np.clip(p_arr, threshold_f, 1.0 - threshold_f), mask

    mask = _overlap_retained_mask(p_arr, threshold_f)
    return p_arr[mask], mask
