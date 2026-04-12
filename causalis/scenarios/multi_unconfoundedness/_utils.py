"""Shared numeric helpers for multi-treatment IRM models."""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.base import is_classifier


def _is_binary(values: np.ndarray) -> bool:
    """Check if an array contains only binary values (0 and 1)."""
    uniq = np.unique(values)
    if uniq.size == 0:
        return False
    return np.all(np.isin(uniq, np.array([0, 1], dtype=float)))


def _safe_is_classifier(estimator) -> bool:
    """Safely check if an estimator is a classifier."""
    try:
        return is_classifier(estimator)
    except (AttributeError, TypeError):
        return getattr(estimator, "_estimator_type", None) == "classifier"


def _predict_propensity_matrix(model, X: np.ndarray, n_treatments: int) -> np.ndarray:
    """Predict propensity matrix P(D=k|X) with aligned treatment columns."""
    if not _safe_is_classifier(model) or not hasattr(model, "predict_proba"):
        raise ValueError("ml_m must be a probabilistic classifier exposing predict_proba().")

    proba = np.asarray(model.predict_proba(X), dtype=float)
    if proba.ndim != 2:
        raise ValueError(
            f"ml_m.predict_proba() must return 2D array (n, K). Got shape {proba.shape}."
        )

    n = X.shape[0]
    classes = getattr(model, "classes_", None)
    if classes is None:
        if proba.shape[1] != n_treatments:
            raise ValueError(
                f"ml_m returned {proba.shape[1]} probability columns, expected {n_treatments}."
            )
        out = proba
    else:
        classes = np.asarray(classes)
        if classes.ndim != 1:
            raise ValueError("ml_m.classes_ must be a 1D array.")
        out = np.zeros((n, n_treatments), dtype=float)
        seen = set()
        for j, cls in enumerate(classes):
            if not np.isfinite(cls):
                raise ValueError("ml_m.classes_ contains non-finite labels.")
            cls_int = int(cls)
            if cls_int != cls:
                raise ValueError("ml_m.classes_ must contain integer treatment labels 0..K-1.")
            if cls_int < 0 or cls_int >= n_treatments:
                raise ValueError(
                    f"ml_m.classes_ contains out-of-range label {cls_int}; expected 0..{n_treatments - 1}."
                )
            out[:, cls_int] = proba[:, j]
            seen.add(cls_int)
        missing = [k for k in range(n_treatments) if k not in seen]
        if missing:
            raise RuntimeError(
                "A cross-fitting training fold is missing treatment classes "
                f"{missing}. Reduce n_folds or increase sample size."
            )

    if np.any((out < -1e-12) | (out > 1.0 + 1e-12)):
        warnings.warn(
            "Propensity model produced values outside [0, 1]. "
            "Values are clipped and renormalized row-wise.",
            RuntimeWarning,
        )
    out = np.clip(out, 0.0, 1.0)
    row_sums = out.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 1e-12):
        raise RuntimeError("Propensity predictions contain rows with zero total probability.")
    if np.any(np.abs(row_sums - 1.0) > 1e-6):
        warnings.warn(
            "Propensity probabilities do not sum to 1. Values are renormalized row-wise.",
            RuntimeWarning,
        )
    out = out / row_sums
    return out


def _normalize_rows_to_simplex(p: np.ndarray) -> np.ndarray:
    """Normalize row-wise probabilities onto the simplex."""
    p = np.asarray(p, dtype=float)
    p = np.maximum(p, 1e-12)
    row_sums = p.sum(axis=1, keepdims=True)
    if np.any(np.isfinite(row_sums) & (row_sums <= 1e-12)):
        raise RuntimeError("Propensity matrix contains rows with zero total probability.")
    safe_sums = np.where(np.isfinite(row_sums) & (row_sums > 1e-12), row_sums, 1.0)
    return p / safe_sums


def _trim_multiclass_propensity(p: np.ndarray, thr: float) -> np.ndarray:
    """Lower-trim multiclass propensity and renormalize rows to sum to 1."""
    p = np.asarray(p, dtype=float)
    if p.ndim != 2:
        raise ValueError(f"Propensity matrix must be 2D (n, K). Got shape {p.shape}.")
    n_treatments = p.shape[1]
    if n_treatments < 2:
        raise ValueError("Need at least 2 treatment columns for multiclass propensity.")

    thr = float(thr)
    if not np.isfinite(thr) or not (0.0 <= thr < (1.0 / n_treatments)):
        raise ValueError(
            f"trimming_threshold must be finite and in [0, 1/K) for K={n_treatments}; got {thr}."
        )

    p_simplex = _normalize_rows_to_simplex(p)
    if thr <= 0.0:
        return p_simplex

    p_trim = np.maximum(p_simplex, thr)
    return _normalize_rows_to_simplex(p_trim)


def _normalize_multiclass_ipw_terms(
    d: np.ndarray,
    m_hat: np.ndarray,
    *,
    normalize_ipw: bool,
) -> np.ndarray:
    """Return d_k / m_k with optional Hajek column normalization."""
    d = np.asarray(d, dtype=float)
    m_hat = np.asarray(m_hat, dtype=float)
    h = d / m_hat
    if normalize_ipw:
        h_mean = h.mean(axis=0, keepdims=True)
        h = h / np.where(h_mean != 0.0, h_mean, 1.0)
    return h
