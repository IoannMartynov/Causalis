"""Private builders for IRM diagnostic payloads and plot caches."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from causalis.data_contracts.causal_diagnostic_data import UnconfoundednessDiagnosticData


def _build_score_plot_cache(
    *,
    d: np.ndarray,
    m_hat: np.ndarray,
    psi: np.ndarray,
    score: str,
    trimming_threshold: float,
    normalize_ipw_effective: bool,
) -> dict[str, Any]:
    """Build cached arrays used by lightweight score influence plots."""
    _ = normalize_ipw_effective
    m_clipped = np.clip(m_hat, trimming_threshold, 1.0 - trimming_threshold)

    return {
        "score": str(score),
        "trimming_threshold": float(trimming_threshold),
        "d": np.asarray(d, dtype=float).ravel(),
        "m_clipped": np.asarray(m_clipped, dtype=float).ravel(),
        "psi": np.asarray(psi, dtype=float).ravel(),
        "row_index": np.arange(np.asarray(d).size, dtype=int),
    }


def _build_residual_plot_cache(
    *,
    y: np.ndarray,
    d: np.ndarray,
    g0_hat: np.ndarray,
    g1_hat: np.ndarray,
    m_hat: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build cached arrays used by residual diagnostic plots."""
    return {
        "y": np.asarray(y, dtype=float).ravel(),
        "d": np.asarray(d, dtype=float).ravel(),
        "g0": np.asarray(g0_hat, dtype=float).ravel(),
        "g1": np.asarray(g1_hat, dtype=float).ravel(),
        "m": np.asarray(m_hat, dtype=float).ravel(),
    }


def _build_irm_estimate_diagnostic_data(
    *,
    model: Any,
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
) -> Optional[UnconfoundednessDiagnosticData]:
    """Build the diagnostic payload attached to ``CausalEstimate`` results."""
    if not model.store_diagnostics:
        return None
    if x is None:
        raise RuntimeError("Diagnostic payloads require cached confounders. Refit with store_diagnostics=True.")

    from causalis.scenarios.unconfoundedness.refutation.unconfoundedness.sensitivity import (
        compute_irm_sensitivity_elements,
    )

    sens_elements = compute_irm_sensitivity_elements(
        model=model,
        y=y,
        d=d,
        g0=g0_hat,
        g1=g1_hat,
        m_hat=m_hat,
        w=w,
        w_bar=w_bar,
        psi=IF,
        inv_m=inv_m,
        inv_1m=inv_1m,
        score=score,
    )

    diag = UnconfoundednessDiagnosticData(
        m_hat=m_hat,
        m_hat_raw=getattr(model, "m_hat_raw_", None),
        d=d,
        y=y,
        x=np.asarray(x, dtype=float),
        g0_hat=g0_hat,
        g1_hat=g1_hat,
        w=w,
        w_bar=w_bar,
        psi_b=psi_b,
        folds=getattr(model, "folds_", None),
        trimming_threshold=model.trimming_threshold,
        normalize_ipw=normalize_ipw_effective,
        score=score,
        score_plot_cache=_build_score_plot_cache(
            d=d,
            m_hat=m_hat,
            psi=IF,
            score=score,
            trimming_threshold=model.trimming_threshold,
            normalize_ipw_effective=normalize_ipw_effective,
        ),
        residual_plot_cache=_build_residual_plot_cache(
            y=y,
            d=d,
            g0_hat=g0_hat,
            g1_hat=g1_hat,
            m_hat=m_hat,
        ),
        feature_importance=getattr(model, "feature_importance_", None),
        **sens_elements,
    )
    diag._model = model
    return diag
