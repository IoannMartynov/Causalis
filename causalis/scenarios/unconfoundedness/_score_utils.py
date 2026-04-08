"""Shared internal score helpers for binary-treatment IRM models."""

from __future__ import annotations

from typing import Any, Optional, Tuple
import warnings

import numpy as np


def _normalize_ate_atte_score(score: Any) -> str:
    """Normalize supported score aliases to ``ATE`` or ``ATTE``."""
    score_u = str(score or "ATE").upper()
    if "ATT" in score_u:
        return "ATTE"
    if score_u == "ATE":
        return "ATE"
    raise ValueError(f"score must be 'ATE' or 'ATTE'. Got {score!r}.")


def _use_normalized_ipw(
    *,
    normalize_ipw: bool,
    score: Optional[str] = None,
    warn: bool = False,
) -> bool:
    """Return whether Hajek normalization is active for a given score."""
    score_u = _normalize_ate_atte_score(score)
    if normalize_ipw and score_u == "ATTE":
        if warn:
            warnings.warn(
                "normalize_ipw=True is ignored for ATTE to preserve the canonical ATTE EIF.",
                RuntimeWarning,
            )
        return False
    return bool(normalize_ipw)


def _compute_ipw_components(
    *,
    d: np.ndarray,
    m_hat: np.ndarray,
    normalize_ipw: bool,
    score: Optional[str] = None,
    warn: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute IPW terms plus inverse propensity factors used in IRM scores."""
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_m = 1.0 / m_hat
        inv_1m = 1.0 / (1.0 - m_hat)

    h1 = d * inv_m
    h0 = (1.0 - d) * inv_1m

    if _use_normalized_ipw(normalize_ipw=normalize_ipw, score=score, warn=warn):
        h1_mean = np.mean(h1)
        h0_mean = np.mean(h0)
        c1 = h1_mean if h1_mean != 0 else 1.0
        c0 = h0_mean if h0_mean != 0 else 1.0
        h1 = h1 / c1
        h0 = h0 / c0
        inv_m = inv_m / c1
        inv_1m = inv_1m / c0

    return h1, h0, inv_m, inv_1m


def _normalize_ipw_terms(
    d: np.ndarray,
    m_hat: np.ndarray,
    *,
    normalize_ipw: bool,
    score: Optional[str] = None,
    warn: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build treated/control IPW terms with optional mean normalization."""
    h1, h0, _, _ = _compute_ipw_components(
        d=d,
        m_hat=m_hat,
        normalize_ipw=normalize_ipw,
        score=score,
        warn=warn,
    )
    return h1, h0


def _resolve_ate_weights(
    n: int,
    w_raw: Optional[np.ndarray],
    w_bar_raw: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Resolve ATE weight vectors from diagnostic payloads."""
    if w_raw is None:
        w = np.ones(n, dtype=float)
    else:
        w = np.asarray(w_raw, dtype=float).ravel()
        if w.size != n:
            raise ValueError(f"diagnostic_data.w must have length n={n}, got {w.size}.")
        if not np.all(np.isfinite(w)):
            raise ValueError("diagnostic_data.w must contain finite values.")

    if w_bar_raw is None:
        w_bar = w
    else:
        w_bar = np.asarray(w_bar_raw, dtype=float).ravel()
        if w_bar.size != n:
            raise ValueError(f"diagnostic_data.w_bar must have length n={n}, got {w_bar.size}.")
        if not np.all(np.isfinite(w_bar)):
            raise ValueError("diagnostic_data.w_bar must contain finite values.")

    return w, w_bar


def _resolve_irm_weights(
    *,
    n: int,
    m_hat_adj: Optional[np.ndarray],
    d: np.ndarray,
    score: Optional[str] = None,
    weights: Optional[np.ndarray | dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute score and representer weights for binary-treatment IRM."""
    score_u = _normalize_ate_atte_score(score)

    if weights is not None and score_u != "ATE":
        raise ValueError(f"weights are only supported for score='ATE', but got score='{score_u}'")

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

    if score_u == "ATE":
        if weights is None:
            w = np.ones(n, dtype=float)
        elif isinstance(weights, np.ndarray):
            w = _to_1d(weights, name="weights")
        elif isinstance(weights, dict):
            if "weights" not in weights:
                raise ValueError("weights dict must contain key 'weights'.")
            w = _to_1d(weights["weights"], name="weights['weights']")
        else:
            raise TypeError("weights must be None, np.ndarray, or dict")

        w_bar = w
        if isinstance(weights, dict) and "weights_bar" in weights:
            w_bar_arr = np.asarray(weights["weights_bar"], dtype=float)
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
    elif score_u == "ATTE":
        if m_hat_adj is None:
            raise ValueError("m_hat required for ATTE weights computation")
        w = d.astype(float)
        w_bar = m_hat_adj.astype(float)
    else:
        raise ValueError("score must be 'ATE' or 'ATTE'")

    mean_w = float(np.mean(w))
    if not np.isfinite(mean_w) or mean_w <= 1e-12:
        raise ValueError("weights must have a strictly positive finite mean.")
    w = w / mean_w
    w_bar = w_bar / mean_w
    return w, w_bar
