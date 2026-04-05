"""Influence/instability plots for score-based diagnostics."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from causalis.data_contracts.causal_estimate import CausalEstimate
from causalis.dgp.causaldata import CausalData

from .score_validation import (
    _aipw_score_ate,
    _aipw_score_atte,
    _normalize_ipw_terms,
    _normalize_score,
    _resolve_ate_weights,
    _resolve_normalize_ipw,
    _resolve_trimming_threshold,
    _validate_estimate_matches_data,
)


def _ess(weights: np.ndarray) -> float:
    """Compute effective sample size for a vector of weights."""
    w = np.asarray(weights, dtype=float).ravel()
    if w.size == 0:
        return float("nan")
    s = float(np.sum(w))
    q = float(np.sum(w ** 2))
    if q <= 0.0:
        return float("nan")
    return float((s * s) / q)


def _resolve_inputs(
    estimate: CausalEstimate,
    data: Optional[CausalData],
    trimming_threshold: Optional[float],
    use_estimator_psi: bool,
) -> dict[str, Any]:
    """Resolve plotting arrays for influence and instability diagnostics."""
    if data is not None:
        _validate_estimate_matches_data(data=data, estimate=estimate)

    diagnostic_data = estimate.diagnostic_data
    if diagnostic_data is None:
        raise ValueError(
            "Missing estimate.diagnostic_data. "
            "Fit IRM with store_diagnostics=True and call estimate() first."
        )

    # Fast path: reuse cached score-plot inputs stored in diagnostic_data.
    cache = getattr(diagnostic_data, "score_plot_cache", None)
    if (
        isinstance(cache, dict)
        and use_estimator_psi
        and trimming_threshold is None
    ):
        required = {"score", "trimming_threshold", "normalize_ipw", "d", "m_clipped", "psi", "ipw_t", "ipw_c"}
        if required.issubset(set(cache.keys())):
            d_cached = np.asarray(cache.get("d"), dtype=float).ravel()
            m_cached = np.asarray(cache.get("m_clipped"), dtype=float).ravel()
            psi_cached = np.asarray(cache.get("psi"), dtype=float).ravel()
            ipw_t_cached = np.asarray(cache.get("ipw_t"), dtype=float).ravel()
            ipw_c_cached = np.asarray(cache.get("ipw_c"), dtype=float).ravel()
            n_cached = d_cached.size
            if (
                n_cached > 0
                and m_cached.size == n_cached
                and psi_cached.size == n_cached
                and ipw_t_cached.size == n_cached
                and ipw_c_cached.size == n_cached
                and np.all(np.isfinite(d_cached))
                and np.all(np.isfinite(m_cached))
                and np.all(np.isfinite(psi_cached))
                and np.all(np.isfinite(ipw_t_cached))
                and np.all(np.isfinite(ipw_c_cached))
            ):
                return {
                    "score": str(cache.get("score", "ATE")),
                    "trimming_threshold": float(cache.get("trimming_threshold", 0.0)),
                    "normalize_ipw": bool(cache.get("normalize_ipw", False)),
                    "d": d_cached,
                    "m_clipped": m_cached,
                    "psi": psi_cached,
                    "ipw_t": ipw_t_cached,
                    "ipw_c": ipw_c_cached,
                    "ipw_t_label": str(cache.get("ipw_t_label", r"$D/m$")),
                    "ipw_c_label": str(cache.get("ipw_c_label", r"$(1-D)/(1-m)$")),
                }

    m_raw = getattr(diagnostic_data, "m_hat", None)
    g0_raw = getattr(diagnostic_data, "g0_hat", None)
    if m_raw is None or g0_raw is None:
        raise ValueError("estimate.diagnostic_data must include `m_hat` and `g0_hat`.")

    score = _normalize_score(getattr(diagnostic_data, "score", estimate.estimand))
    trimming_thr = _resolve_trimming_threshold(trimming_threshold, diagnostic_data, estimate)
    normalize_ipw = _resolve_normalize_ipw(score, diagnostic_data, estimate)

    y_raw = getattr(diagnostic_data, "y", None)
    if y_raw is None:
        if data is None:
            raise ValueError(
                "diagnostic_data must include `y`, or pass `data` for fallback."
            )
        y_raw = data.get_df()[str(data.outcome_name)].to_numpy(dtype=float)

    d_raw = getattr(diagnostic_data, "d", None)
    if d_raw is None:
        if data is None:
            raise ValueError(
                "diagnostic_data must include `d`, or pass `data` for fallback."
            )
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
                    m_hat_adj=np.clip(m, trimming_thr, 1.0 - trimming_thr),
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
        # For non-ATE scores we derive sample-moment weights after filtering.
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
        w_bar = np.clip(m, trimming_thr, 1.0 - trimming_thr) / (p_treated + 1e-12)

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
                trimming_threshold=trimming_thr,
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
                trimming_threshold=trimming_thr,
            )

    m_clipped = np.clip(m, trimming_thr, 1.0 - trimming_thr)
    if score == "ATE":
        ipw_t, ipw_c = _normalize_ipw_terms(d, m_clipped, normalize_ipw=normalize_ipw)
        ipw_t_label = r"$D/m$"
        ipw_c_label = r"$(1-D)/(1-m)$"
    else:
        p_treated = float(np.mean(d))
        ipw_t = d / (p_treated + 1e-12)
        ipw_c = ((1.0 - d) * (m_clipped / (1.0 - m_clipped))) / (p_treated + 1e-12)
        ipw_t_label = r"$D/\mathbb{E}[D]$"
        ipw_c_label = r"$(1-D)\cdot m/(1-m)/\mathbb{E}[D]$"

    resolved = {
        "score": score,
        "trimming_threshold": float(trimming_thr),
        "normalize_ipw": bool(normalize_ipw),
        "d": d,
        "m_clipped": m_clipped,
        "psi": np.asarray(psi, dtype=float).ravel(),
        "ipw_t": np.asarray(ipw_t, dtype=float).ravel(),
        "ipw_c": np.asarray(ipw_c, dtype=float).ravel(),
        "ipw_t_label": ipw_t_label,
        "ipw_c_label": ipw_c_label,
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
    trimming_threshold: Optional[float] = None,
    use_estimator_psi: bool = True,
    include_ipw: bool = True,
    bins: Any = "fd",
    log_hist: bool = False,
    scatter_log_y: bool = True,
    top_k: int = 10,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 220,
    font_scale: float = 1.10,
    save: Optional[str] = None,
    save_dpi: Optional[int] = None,
    transparent: bool = False,
) -> plt.Figure:
    """
    Plot instability diagnostics for per-unit score (EIF moment).

    Panels
    ------
    1. Histogram of ``|psi_i|`` (optional log-x scale).
    2. Scatter of ``|psi_i|`` versus clipped propensity ``m_i``.
    3. (optional) Histogram of IPW terms.
    4. (optional) ESS ratio bars for treated/control weights.

    Parameters
    ----------
    estimate : CausalEstimate
        Estimate with diagnostic data (`m_hat`, `g0_hat`; optionally `y`, `d`, `g1_hat`, `psi`).
    data : CausalData, optional
        Optional fallback source for `y` and `d` if not stored in diagnostic data.
    trimming_threshold : float, optional
        Propensity clipping threshold. If omitted, uses diagnostic/model defaults.
    use_estimator_psi : bool, default True
        Use estimator-provided `diagnostic_data.psi` when available; otherwise reconstruct score.
    include_ipw : bool, default True
        Add IPW-term histogram and ESS ratio bar panels.
    bins : Any, default "fd"
        Histogram bins for non-log histograms.
    log_hist : bool, default False
        Use log-scaled x-axis bins for ``|psi_i|`` histogram when possible.
    scatter_log_y : bool, default True
        Plot ``|psi_i|`` on log scale in the scatter panel.
    top_k : int, default 10
        Highlight top-k largest ``|psi_i|`` in the scatter panel.
    figsize : tuple, optional
        Figure size. Defaults to `(12, 8)` with IPW panels, `(11, 4.6)` otherwise.
    dpi : int, default 220
        Dots per inch.
    font_scale : float, default 1.10
        Font scaling factor.
    save : str, optional
        Path to save the figure.
    save_dpi : int, optional
        DPI for saving.
    transparent : bool, default False
        Whether to save with transparency.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """

    resolved = _resolve_inputs(
        estimate=estimate,
        data=data,
        trimming_threshold=trimming_threshold,
        use_estimator_psi=use_estimator_psi,
    )
    score = str(resolved["score"])
    trim = float(resolved["trimming_threshold"])
    d = np.asarray(resolved["d"], dtype=float)
    m_clipped = np.asarray(resolved["m_clipped"], dtype=float)
    psi = np.asarray(resolved["psi"], dtype=float)
    ipw_t = np.asarray(resolved["ipw_t"], dtype=float)
    ipw_c = np.asarray(resolved["ipw_c"], dtype=float)
    ipw_t_label = str(resolved["ipw_t_label"])
    ipw_c_label = str(resolved["ipw_c_label"])

    abs_psi = np.abs(psi)
    abs_for_scatter = np.clip(abs_psi, 1e-12, None) if scatter_log_y else abs_psi

    p99 = float(np.quantile(abs_psi, 0.99)) if abs_psi.size else float("nan")
    med = float(np.median(abs_psi)) if abs_psi.size else float("nan")
    ratio = float(p99 / (med + 1e-12)) if np.isfinite(p99) and np.isfinite(med) else float("nan")
    var = float(np.var(psi, ddof=1)) if psi.size > 1 else float("nan")
    kurt = float(np.mean((psi - np.mean(psi)) ** 4) / (var ** 2 + 1e-12)) if np.isfinite(var) and var > 0.0 else float(
        "nan"
    )

    if include_ipw:
        fig_size = figsize or (12.0, 8.0)
    else:
        fig_size = figsize or (11.0, 4.6)

    rc = {
        "font.size": 11 * font_scale,
        "axes.titlesize": 13 * font_scale,
        "axes.labelsize": 11 * font_scale,
        "legend.fontsize": 10 * font_scale,
        "xtick.labelsize": 10 * font_scale,
        "ytick.labelsize": 10 * font_scale,
    }

    with mpl.rc_context(rc):
        if include_ipw:
            fig, axs = plt.subplots(2, 2, figsize=fig_size, dpi=dpi)
            ax_hist, ax_scatter, ax_ipw, ax_ess = axs.ravel()
        else:
            fig, axs = plt.subplots(1, 2, figsize=fig_size, dpi=dpi)
            ax_hist, ax_scatter = axs.ravel()
            ax_ipw = None
            ax_ess = None

        # 1) Histogram of |psi|
        positive_abs = abs_psi[abs_psi > 0.0]
        if log_hist and positive_abs.size >= 2:
            lo = float(max(np.min(positive_abs), 1e-12))
            hi = float(np.max(positive_abs))
            if hi <= lo:
                hi = lo * 1.01
            log_bins = np.geomspace(lo, hi, 40)
            ax_hist.hist(positive_abs, bins=log_bins, alpha=0.80, color="C0", edgecolor="white")
            ax_hist.set_xscale("log")
            if positive_abs.size < abs_psi.size:
                ax_hist.text(
                    0.98,
                    0.98,
                    f"zeros={int(abs_psi.size - positive_abs.size)}",
                    ha="right",
                    va="top",
                    transform=ax_hist.transAxes,
                )
        else:
            ax_hist.hist(abs_psi, bins=bins, alpha=0.80, color="C0", edgecolor="white")

        if np.isfinite(p99):
            ax_hist.axvline(p99, color="C3", linestyle="--", linewidth=1.6, label="p99")
        if np.isfinite(med):
            ax_hist.axvline(med, color="C2", linestyle=":", linewidth=1.6, label="median")
        ax_hist.set_xlabel(r"$|\psi_i|$")
        ax_hist.set_ylabel("Count")
        ax_hist.set_title(r"Influence Magnitude: Histogram of $|\psi_i|$")
        ax_hist.grid(True, linewidth=0.5, alpha=0.45)
        ax_hist.legend(frameon=False)

        # 2) Scatter |psi| vs propensity
        treated_mask = d > 0.5
        control_mask = ~treated_mask
        ax_scatter.scatter(
            m_clipped[treated_mask],
            abs_for_scatter[treated_mask],
            s=14,
            alpha=0.45,
            color="C0",
            label="Treated",
            linewidths=0.0,
        )
        ax_scatter.scatter(
            m_clipped[control_mask],
            abs_for_scatter[control_mask],
            s=14,
            alpha=0.45,
            color="C1",
            label="Control",
            linewidths=0.0,
        )

        if int(top_k) > 0 and abs_psi.size > 0:
            idx = np.argsort(-abs_psi)[: min(int(top_k), abs_psi.size)]
            ax_scatter.scatter(
                m_clipped[idx],
                abs_for_scatter[idx],
                s=42,
                facecolors="none",
                edgecolors="crimson",
                linewidths=1.0,
                label=f"Top-{min(int(top_k), abs_psi.size)} |psi|",
            )

        edges = np.linspace(0.0, 1.0, 21)
        mids = 0.5 * (edges[:-1] + edges[1:])
        means = np.full(mids.shape, np.nan, dtype=float)
        for j in range(mids.size):
            in_bin = (m_clipped >= edges[j]) & (m_clipped < edges[j + 1])
            if np.any(in_bin):
                means[j] = float(np.mean(abs_psi[in_bin]))
        valid = np.isfinite(means)
        if np.any(valid):
            line_vals = np.clip(means[valid], 1e-12, None) if scatter_log_y else means[valid]
            ax_scatter.plot(mids[valid], line_vals, color="black", linewidth=1.5, label="Binned mean |psi|")

        ax_scatter.axvline(trim, color="0.35", linestyle=":", linewidth=1.1)
        ax_scatter.axvline(1.0 - trim, color="0.35", linestyle=":", linewidth=1.1)
        if scatter_log_y:
            ax_scatter.set_yscale("log")
        ax_scatter.set_xlim(0.0, 1.0)
        ax_scatter.set_xlabel(r"Clipped propensity $m_i$")
        ax_scatter.set_ylabel(r"$|\psi_i|$")
        ax_scatter.set_title(r"Instability: $|\psi_i|$ vs $m_i$")
        ax_scatter.grid(True, linewidth=0.5, alpha=0.45)
        ax_scatter.legend(frameon=False)

        note_ratio = "nan" if not np.isfinite(ratio) else f"{ratio:.2f}"
        note_kurt = "nan" if not np.isfinite(kurt) else f"{kurt:.2f}"
        ax_scatter.text(
            0.02,
            0.98,
            f"score={score}\ntrim={trim:.4f}\np99/med={note_ratio}\nkurt={note_kurt}",
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

        # 3) IPW terms and 4) ESS bars
        if include_ipw and ax_ipw is not None and ax_ess is not None:
            ipw_t_nz = ipw_t[treated_mask]
            ipw_c_nz = ipw_c[control_mask]
            if ipw_t_nz.size > 0:
                ax_ipw.hist(
                    ipw_t_nz,
                    bins=bins,
                    alpha=0.55,
                    density=True,
                    color="C0",
                    label=f"{ipw_t_label} (treated)",
                    edgecolor="white",
                )
            if ipw_c_nz.size > 0:
                ax_ipw.hist(
                    ipw_c_nz,
                    bins=bins,
                    alpha=0.55,
                    density=True,
                    color="C1",
                    label=f"{ipw_c_label} (control)",
                    edgecolor="white",
                )
            ax_ipw.set_xlabel("Weight term value")
            ax_ipw.set_ylabel("Density")
            ax_ipw.set_title("IPW-Term Distributions (post-trim/normalization)")
            ax_ipw.grid(True, linewidth=0.5, alpha=0.45)
            ax_ipw.legend(frameon=False)

            ess_t = _ess(ipw_t_nz)
            ess_c = _ess(ipw_c_nz)
            n_t = int(ipw_t_nz.size)
            n_c = int(ipw_c_nz.size)
            ratio_t = float(ess_t / n_t) if n_t > 0 and np.isfinite(ess_t) else float("nan")
            ratio_c = float(ess_c / n_c) if n_c > 0 and np.isfinite(ess_c) else float("nan")

            ratios = [ratio_t, ratio_c]
            bar_vals = [r if np.isfinite(r) else 0.0 for r in ratios]
            bars = ax_ess.bar(["treated", "control"], bar_vals, color=["C0", "C1"], alpha=0.85)
            ylim_top = max(1.0, max(bar_vals) * 1.20 + 0.05)
            ax_ess.set_ylim(0.0, ylim_top)
            ax_ess.set_ylabel("ESS / group size (IPW terms)")
            ax_ess.set_title("Effective Sample Size of IPW Terms")
            ax_ess.grid(True, axis="y", linewidth=0.5, alpha=0.45)

            texts = [
                f"ESS={ess_t:.1f}\nratio={ratio_t:.2f}" if np.isfinite(ess_t) else "ESS=nan",
                f"ESS={ess_c:.1f}\nratio={ratio_c:.2f}" if np.isfinite(ess_c) else "ESS=nan",
            ]
            for bar, text in zip(bars, texts):
                ax_ess.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + 0.02 * ylim_top,
                    text,
                    ha="center",
                    va="bottom",
                )

        for axis in fig.axes:
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)

        fig.tight_layout()

        if save is not None:
            ext = str(save).lower().split(".")[-1]
            _dpi = save_dpi or (
                300 if ext in {"png", "jpg", "jpeg", "tif", "tiff"} else dpi
            )
            fig.savefig(
                save,
                dpi=_dpi,
                bbox_inches="tight",
                pad_inches=0.1,
                transparent=transparent,
                facecolor="none" if transparent else "white",
            )

    # Close notebook-managed figure to avoid duplicate Jupyter rendering
    # (inline auto-display + returned Figure rich repr).
    plt.close(fig)
    return fig


__all__ = ["plot_influence_instability"]
