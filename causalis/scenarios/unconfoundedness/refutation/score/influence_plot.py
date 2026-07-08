"""Lightweight plots for the most influential score contributions."""

from __future__ import annotations

from typing import Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from causalis.data_contracts.causal_estimate import CausalEstimate
from causalis.dgp.causaldata import CausalData

from .score_validation import (
    _aipw_score_ate,
    _aipw_score_atte,
    _normalize_score,
    _resolve_ate_weights,
    _resolve_normalize_ipw,
    _resolve_overlap_threshold,
    _validate_estimate_matches_data,
)


def _resolve_data_for_labels(
    data: Optional[CausalData],
    estimate: CausalEstimate,
) -> Optional[CausalData]:
    """Resolve a CausalData source for labeling plotted observations."""
    if data is not None:
        return data

    diagnostic_data = estimate.diagnostic_data
    model_ref = getattr(diagnostic_data, "_model", None) if diagnostic_data is not None else None
    model_data = getattr(model_ref, "data", None)
    return model_data if isinstance(model_data, CausalData) else None


def _labels_from_user_id(
    label_data: Optional[CausalData],
    row_index: np.ndarray,
) -> np.ndarray:
    """Prefer user_id labels when available; otherwise fall back to row indices."""
    idx = np.asarray(row_index, dtype=int).ravel()
    if label_data is not None and getattr(label_data, "user_id_name", None):
        user_ids = label_data.user_id
        if len(user_ids) > 0 and idx.size > 0 and int(np.max(idx)) < len(user_ids):
            return user_ids.iloc[idx].astype(str).to_numpy(dtype=object)
    return idx.astype(str)


def _top_k_indices(abs_psi: np.ndarray, top_k: int) -> np.ndarray:
    """Return indices of the largest ``top_k`` absolute influences in descending order."""
    values = np.asarray(abs_psi, dtype=float).ravel()
    n = int(values.size)
    if n == 0:
        return np.zeros(0, dtype=int)

    k = min(max(int(top_k), 1), n)
    if k == n:
        return np.argsort(-values)

    idx = np.argpartition(-values, k - 1)[:k]
    return idx[np.argsort(-values[idx])]


def _resolve_inputs(
    estimate: CausalEstimate,
    data: Optional[CausalData],
    overlap_threshold: Optional[float],
    use_estimator_psi: bool,
) -> dict[str, object]:
    """Resolve arrays needed for lightweight top-influence plotting."""
    if data is not None:
        _validate_estimate_matches_data(data=data, estimate=estimate)

    diagnostic_data = estimate.diagnostic_data
    if diagnostic_data is None:
        raise ValueError(
            "Missing estimate.diagnostic_data. "
            "Fit IRM with store_diagnostics=True and call estimate() first."
        )

    cache = getattr(diagnostic_data, "score_plot_cache", None)
    if isinstance(cache, dict) and use_estimator_psi and overlap_threshold is None:
        required = {"score", "overlap_threshold", "d", "m_clipped", "psi"}
        if required.issubset(set(cache.keys())):
            d_cached = np.asarray(cache.get("d"), dtype=float).ravel()
            m_cached = np.asarray(cache.get("m_clipped"), dtype=float).ravel()
            psi_cached = np.asarray(cache.get("psi"), dtype=float).ravel()
            row_index_cached = np.asarray(
                cache.get("row_index", np.arange(d_cached.size, dtype=int)),
                dtype=int,
            ).ravel()
            n_cached = d_cached.size
            if (
                n_cached > 0
                and m_cached.size == n_cached
                and psi_cached.size == n_cached
                and row_index_cached.size == n_cached
                and np.all(np.isfinite(d_cached))
                and np.all(np.isfinite(m_cached))
                and np.all(np.isfinite(psi_cached))
            ):
                return {
                    "score": str(cache.get("score", "ATE")),
                    "overlap_threshold": float(cache.get("overlap_threshold", 0.0)),
                    "d": d_cached,
                    "m_clipped": m_cached,
                    "psi": psi_cached,
                    "row_index": row_index_cached,
                    "observation_label": _labels_from_user_id(
                        _resolve_data_for_labels(data=data, estimate=estimate),
                        row_index_cached,
                    ),
                }

    m_raw = getattr(diagnostic_data, "m_hat", None)
    g0_raw = getattr(diagnostic_data, "g0_hat", None)
    if m_raw is None or g0_raw is None:
        raise ValueError("estimate.diagnostic_data must include `m_hat` and `g0_hat`.")

    score = _normalize_score(getattr(diagnostic_data, "score", estimate.estimand))
    overlap_thr = _resolve_overlap_threshold(overlap_threshold, diagnostic_data, estimate)
    normalize_ipw = _resolve_normalize_ipw(score, diagnostic_data, estimate)

    y_raw = getattr(diagnostic_data, "y", None)
    if y_raw is None:
        if data is None:
            raise ValueError("diagnostic_data must include `y`, or pass `data` for fallback.")
        y_raw = data.get_df()[str(data.outcome_name)].to_numpy(dtype=float)

    d_raw = getattr(diagnostic_data, "d", None)
    if d_raw is None:
        if data is None:
            raise ValueError("diagnostic_data must include `d`, or pass `data` for fallback.")
        d_raw = data.get_df()[str(data.treatment_name)].to_numpy(dtype=float)

    g1_raw = getattr(diagnostic_data, "g1_hat", None)
    if g1_raw is None:
        g1_raw = np.asarray(g0_raw, dtype=float)

    psi_raw = getattr(diagnostic_data, "psi", None) if use_estimator_psi else None
    diag_w_raw = getattr(diagnostic_data, "w", None)
    diag_w_bar_raw = getattr(diagnostic_data, "w_bar", None)

    y = np.asarray(y_raw, dtype=float).ravel()
    d = (np.asarray(d_raw, dtype=float).ravel() > 0.5).astype(float)
    g0 = np.asarray(g0_raw, dtype=float).ravel()
    g1 = np.asarray(g1_raw, dtype=float).ravel()
    m = np.asarray(m_raw, dtype=float).ravel()

    n = int(y.size)
    if any(arr.size != n for arr in (d, g0, g1, m)):
        raise ValueError("All diagnostic arrays must have matching sample size n.")

    if score == "ATE" and (diag_w_raw is None or diag_w_bar_raw is None):
        model_ref = getattr(diagnostic_data, "_model", None)
        if model_ref is not None and hasattr(model_ref, "_get_weights"):
            try:
                w_model, w_bar_model = model_ref._get_weights(
                    n=n,
                    m_hat_adj=np.clip(m, overlap_thr, 1.0 - overlap_thr),
                    d=d.astype(int),
                    score="ATE",
                )
                if diag_w_raw is None:
                    diag_w_raw = w_model
                if diag_w_bar_raw is None:
                    diag_w_bar_raw = w_bar_model
            except Exception:
                pass

    if score == "ATE":
        w, w_bar = _resolve_ate_weights(n=n, w_raw=diag_w_raw, w_bar_raw=diag_w_bar_raw)
    else:
        w = np.ones(n, dtype=float)
        w_bar = np.ones(n, dtype=float)

    psi = None
    if psi_raw is not None:
        psi_tmp = np.asarray(psi_raw, dtype=float).ravel()
        if psi_tmp.size == n:
            psi = psi_tmp

    finite_rows = (
        np.isfinite(y)
        & np.isfinite(d)
        & np.isfinite(g0)
        & np.isfinite(g1)
        & np.isfinite(m)
        & np.isfinite(w)
        & np.isfinite(w_bar)
    )
    if psi is not None:
        finite_rows = finite_rows & np.isfinite(psi)

    row_index = np.flatnonzero(finite_rows).astype(int)
    y = y[finite_rows]
    d = d[finite_rows]
    g0 = g0[finite_rows]
    g1 = g1[finite_rows]
    m = m[finite_rows]
    w = w[finite_rows]
    w_bar = w_bar[finite_rows]
    psi = psi[finite_rows] if psi is not None else None

    if y.size == 0:
        raise ValueError("No finite observations available for influence plotting.")

    if score != "ATE":
        p_treated = float(np.mean(d))
        w = d / (p_treated + 1e-12)
        w_bar = np.clip(m, overlap_thr, 1.0 - overlap_thr) / (p_treated + 1e-12)

    theta = float(estimate.value)
    if psi is None:
        if score == "ATE":
            psi = _aipw_score_ate(
                y=y,
                d=d,
                g0=g0,
                g1=g1,
                m=m,
                theta=theta,
                overlap_threshold=overlap_thr,
                normalize_ipw=normalize_ipw,
                w=w,
                w_bar=w_bar,
            )
        else:
            psi = _aipw_score_atte(
                y=y,
                d=d,
                g0=g0,
                m=m,
                theta=theta,
                overlap_threshold=overlap_thr,
            )

    resolved = {
        "score": score,
        "overlap_threshold": float(overlap_thr),
        "d": d,
        "m_clipped": np.clip(m, overlap_thr, 1.0 - overlap_thr),
        "psi": np.asarray(psi, dtype=float).ravel(),
        "row_index": row_index,
        "observation_label": _labels_from_user_id(
            _resolve_data_for_labels(data=data, estimate=estimate),
            row_index,
        ),
    }
    try:
        diagnostic_data.score_plot_cache = dict(resolved)
    except Exception:
        pass
    return resolved


def plot_influence_instability(
    estimate: CausalEstimate,
    data: Optional[CausalData] = None,
    *,
    overlap_threshold: Optional[float] = None,
    use_estimator_psi: bool = True,
    top_k: int = 20,
    annotate: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 220,
    font_scale: float = 1.10,
    save: Optional[str] = None,
    save_dpi: Optional[int] = None,
    transparent: bool = False,
) -> plt.Figure:
    r"""
    Plot only the most influential score contributions.

    Panels
    ------
    1. Ranked bar chart of the top ``k`` observations by ``|psi_i|``.
    2. Scatter of those top ``k`` observations versus clipped propensity ``m_i``.

    This replacement intentionally avoids plotting every observation. Use
    ``run_score_diagnostics(... )["summary"]`` for global tail metrics and this
    plot for a lightweight drill-down into the largest contributors.

    Notes
    -----
    The figure ranks observations by :math:`|\hat\psi_i|`, where
    :math:`\hat\psi_i` is the fitted score contribution. Large values indicate
    observations with unusually strong leverage on the final estimate. If many
    top points sit near propensity clipping boundaries, that is usually a sign
    to inspect overlap and nuisance fit quality together.

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
    >>> irm = IRM(
    ...     data=data,
    ...     ml_g=RandomForestRegressor(
    ...         n_estimators=200,
    ...         max_depth=6,
    ...         min_samples_leaf=5,
    ...         random_state=3141,
    ...     ),
    ...     ml_m=RandomForestClassifier(
    ...         n_estimators=200,
    ...         max_depth=6,
    ...         min_samples_leaf=5,
    ...         random_state=3141,
    ...     ),
    ...     n_folds=3,
    ...     random_state=3141,
    ... )
    >>> estimate = irm.fit().estimate(score="ATE")
    >>> fig = plot_influence_instability(estimate, data=data, top_k=15)  # doctest: +SKIP
    """

    resolved = _resolve_inputs(
        estimate=estimate,
        data=data,
        overlap_threshold=overlap_threshold,
        use_estimator_psi=use_estimator_psi,
    )
    score = str(resolved["score"])
    overlap = float(resolved["overlap_threshold"])
    d = np.asarray(resolved["d"], dtype=float)
    m_clipped = np.asarray(resolved["m_clipped"], dtype=float)
    psi = np.asarray(resolved["psi"], dtype=float)
    row_index = np.asarray(resolved["row_index"], dtype=int)
    observation_label = np.asarray(resolved["observation_label"], dtype=object)

    abs_psi = np.abs(psi)
    top_idx = _top_k_indices(abs_psi, top_k=top_k)
    top_n = int(top_idx.size)
    if top_n == 0:
        raise ValueError("No observations available for top-influence plotting.")

    top_abs = np.clip(abs_psi[top_idx], 1e-12, None)
    top_signed = psi[top_idx]
    top_m = m_clipped[top_idx]
    top_d = d[top_idx]
    top_rows = row_index[top_idx]
    top_labels = observation_label[top_idx]
    ranks = np.arange(1, top_n + 1)
    point_colors = np.where(top_d > 0.5, "C0", "C1")
    label_rows = [f"#{rank} user_id={label}" for rank, label in zip(ranks, top_labels)]

    fig_size = figsize or (12.0, max(4.8, 2.4 + 0.32 * top_n))
    rc = {
        "font.size": 11 * font_scale,
        "axes.titlesize": 13 * font_scale,
        "axes.labelsize": 11 * font_scale,
        "legend.fontsize": 10 * font_scale,
        "xtick.labelsize": 10 * font_scale,
        "ytick.labelsize": 10 * font_scale,
    }

    with mpl.rc_context(rc):
        fig, (ax_rank, ax_scatter) = plt.subplots(
            1,
            2,
            figsize=fig_size,
            dpi=dpi,
            gridspec_kw={"width_ratios": [1.25, 1.0]},
        )

        y_pos = np.arange(top_n)
        ax_rank.barh(y_pos, top_abs, color=point_colors, alpha=0.88)
        ax_rank.set_yticks(y_pos, label_rows)
        ax_rank.invert_yaxis()
        ax_rank.set_xscale("log")
        ax_rank.set_xlabel(r"$|\psi_i|$")
        ax_rank.set_ylabel("Ranked observations")
        ax_rank.set_title(f"Top-{top_n} |psi| Contributions")
        ax_rank.grid(True, axis="x", linewidth=0.5, alpha=0.45)

        for y_tick, signed_val, abs_val in zip(y_pos, top_signed, top_abs):
            sign = "+" if signed_val >= 0.0 else "-"
            ax_rank.text(
                abs_val * 1.04,
                y_tick,
                sign,
                va="center",
                ha="left",
            )

        treated_mask = top_d > 0.5
        control_mask = ~treated_mask
        if np.any(treated_mask):
            ax_scatter.scatter(
                top_m[treated_mask],
                top_abs[treated_mask],
                s=58,
                color="C0",
                alpha=0.85,
                label="Treated",
            )
        if np.any(control_mask):
            ax_scatter.scatter(
                top_m[control_mask],
                top_abs[control_mask],
                s=58,
                color="C1",
                alpha=0.85,
                label="Control",
            )

        if annotate:
            for rank, x_val, y_val in zip(ranks, top_m, top_abs):
                ax_scatter.annotate(
                    f"#{rank}",
                    xy=(float(x_val), float(y_val)),
                    xytext=(5, 4),
                    textcoords="offset points",
                )

        ax_scatter.axvline(overlap, color="0.35", linestyle=":", linewidth=1.1)
        ax_scatter.axvline(1.0 - overlap, color="0.35", linestyle=":", linewidth=1.1)
        ax_scatter.set_xlim(0.0, 1.0)
        ax_scatter.set_yscale("log")
        ax_scatter.set_xlabel(r"Clipped propensity $m_i$")
        ax_scatter.set_ylabel(r"$|\psi_i|$")
        ax_scatter.set_title("Top Influential Points by Propensity")
        ax_scatter.grid(True, linewidth=0.5, alpha=0.45)
        ax_scatter.legend(frameon=False)
        ax_scatter.text(
            0.02,
            0.98,
            f"score={score}\noverlap={overlap:.4f}\ntop_k={top_n}",
            transform=ax_scatter.transAxes,
            ha="left",
            va="top",
            bbox={
                "facecolor": "white",
                "edgecolor": "0.80",
                "alpha": 0.90,
                "boxstyle": "round,pad=0.35",
            },
        )

        for axis in fig.axes:
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)

        fig.tight_layout()

        if save is not None:
            ext = str(save).lower().split(".")[-1]
            _dpi = save_dpi or (300 if ext in {"png", "jpg", "jpeg", "tif", "tiff"} else dpi)
            fig.savefig(
                save,
                dpi=_dpi,
                bbox_inches="tight",
                pad_inches=0.1,
                transparent=transparent,
                facecolor="none" if transparent else "white",
            )

    plt.close(fig)
    return fig


__all__ = ["plot_influence_instability"]
