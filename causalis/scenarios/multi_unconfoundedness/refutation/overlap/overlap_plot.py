from typing import Tuple, Optional, Any, List, Union
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from causalis.data_contracts.multicausal_estimate import MultiCausalEstimate
from causalis.data_contracts.multicausaldata import MultiCausalData
from causalis.data_contracts.causal_diagnostic_data import MultiUnconfoundednessDiagnosticData


def _resolve_overlap_diag(
    diag: Union[MultiUnconfoundednessDiagnosticData, MultiCausalEstimate, dict, Any]
) -> MultiUnconfoundednessDiagnosticData:
    if isinstance(diag, MultiUnconfoundednessDiagnosticData):
        resolved = diag
    elif isinstance(diag, MultiCausalEstimate):
        resolved = diag.diagnostic_data
    elif isinstance(diag, dict):
        resolved = diag.get("diagnostic_data", None)
    else:
        resolved = getattr(diag, "diagnostic_data", None)

    if resolved is None:
        raise ValueError(
            "plot_m_overlap expects MultiUnconfoundednessDiagnosticData or "
            "MultiCausalEstimate with diagnostic_data. "
            "Call estimate(..., diagnostic_data=True)."
        )
    if not hasattr(resolved, "m_hat") or not hasattr(resolved, "d"):
        raise ValueError("diagnostic_data must include both `m_hat` and `d`.")
    return resolved


def plot_m_overlap(
    diag: Union[MultiUnconfoundednessDiagnosticData, MultiCausalEstimate, dict, Any],
    clip: Tuple[float, float] = (0.01, 0.99),
    bins: Any = "fd",
    kde: bool = True,
    shade_overlap: bool = True,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (9, 5.5),
    dpi: int = 220,
    font_scale: float = 1.15,
    save: Optional[str] = None,
    save_dpi: Optional[int] = None,
    transparent: bool = False,
    color_t: Optional[Any] = None,
    color_c: Optional[Any] = None,
    *,
    treatment_idx: Optional[Union[int, List[int]]] = None,
    baseline_idx: int = 0,
    treatment_names: Optional[List[str]] = None,
) -> plt.Figure:
    """
    Multi-treatment overlap plot for pairwise conditional propensity scores.

    For each comparison baseline (default 0) vs k, this plots
    ``P(D=k | X, D in {baseline, k}) = m_k(X) / (m_baseline(X) + m_k(X))``
    on the observed pair sample ``D in {baseline, k}``, comparing:
      - units with D=k (treated for the pair),
      - units with D=baseline (control for the pair).

    Parameters:
      - diag.d: (n, K) one-hot
      - diag.m_hat / diag.m_hat_raw: (n, K) propensity
      - treatment_idx:
          * None -> plot all k != baseline_idx (multi-panel)
          * int -> plot one comparison
          * list[int] -> plot selected comparisons
      - ax: supported only for a single comparison (exactly one k)

    Returns matplotlib.figure.Figure.
    """

    # ------- Helpers --------------------------------------------------------
    def _silverman_bandwidth(x: np.ndarray) -> float:
        x = np.asarray(x, float)
        n = x.size
        if n < 2:
            return 0.04
        sd = np.std(x, ddof=1)
        iqr = np.subtract(*np.percentile(x, [75, 25]))
        s = sd if iqr <= 0 else min(sd, iqr / 1.34)
        h = 0.9 * s * n ** (-1 / 5)
        return float(max(h, 0.02))

    def _kde_reflect(x: np.ndarray, xs: np.ndarray, h: float) -> np.ndarray:
        x = np.asarray(x, float)
        xs = np.asarray(xs, float)
        if x.size == 0:
            return np.zeros_like(xs)
        if x.size < 2 or np.std(x) < 1e-8:
            mu = float(np.mean(x)) if x.size else 0.5
            h0 = max(float(h), 0.02)
            z = (xs - mu) / h0
            return np.exp(-0.5 * z**2) / (np.sqrt(2 * np.pi) * h0)

        grid_size = int(min(2048, max(512, xs.size)))
        edges = np.linspace(0.0, 1.0, grid_size + 1, dtype=float)
        centers = 0.5 * (edges[:-1] + edges[1:])
        dx = float(edges[1] - edges[0])

        counts, _ = np.histogram(np.clip(x, 0.0, 1.0), bins=edges)
        density = counts.astype(float) / (x.size * dx)

        radius = max(1, int(np.ceil(4.0 * h / dx)))
        offsets = np.arange(-radius, radius + 1, dtype=float)
        kernel = np.exp(-0.5 * ((offsets * dx) / h) ** 2)
        kernel /= np.sqrt(2 * np.pi) * h
        kernel *= dx

        padded = np.pad(density, pad_width=radius, mode="reflect")
        smooth = np.convolve(padded, kernel, mode="same")[radius:-radius]
        return np.interp(xs, centers, smooth, left=smooth[0], right=smooth[-1])

    def _patch_color(patches, fallback):
        for p in patches:
            fc = p.get_facecolor()
            if fc is not None:
                return fc
        return fallback

    def _pairwise_scores(
        m_arr: np.ndarray,
        d_arr: np.ndarray,
        *,
        tr_idx: int,
        base_idx: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        mask_t = d_arr[:, tr_idx] > 0.5
        mask_c = d_arr[:, base_idx] > 0.5
        mask_pair = mask_t | mask_c
        if not np.any(mask_pair):
            return np.array([], dtype=float), np.array([], dtype=float)

        m_t = np.clip(m_arr[mask_pair, tr_idx], 0.0, 1.0)
        m_c = np.clip(m_arr[mask_pair, base_idx], 0.0, 1.0)
        denom = np.clip(m_t + m_c, 1e-12, None)
        s_pair = m_t / denom
        return s_pair[mask_t[mask_pair]], s_pair[mask_c[mask_pair]]

    def _plot_one(
        ax1: plt.Axes,
        mt: np.ndarray,
        mc: np.ndarray,
        *,
        label_t: str,
        label_c: str,
        xlabel: str,
        title: str,
    ):
        # Clamp to [0,1]
        mtp = np.clip(mt, 0.0, 1.0)
        mcp = np.clip(mc, 0.0, 1.0)

        # Histograms
        ht = ax1.hist(
            mtp, bins=bins, range=(0.0, 1.0), density=True,
            alpha=0.45, label=label_t, edgecolor="white", linewidth=0.6,
            color=color_t
        )
        hc = ax1.hist(
            mcp, bins=bins, range=(0.0, 1.0), density=True,
            alpha=0.45, label=label_c, edgecolor="white", linewidth=0.6,
            color=color_c
        )

        cycle = mpl.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1"])
        used_t = color_t or _patch_color(ht[2], cycle[0])
        used_c = color_c or _patch_color(hc[2], cycle[1])

        # KDE
        if kde:
            if clip:
                allm = np.clip(np.r_[mtp, mcp], 0, 1)
                lo, hi = np.quantile(allm, [clip[0], clip[1]])
                lo, hi = float(max(0.0, lo)), float(min(1.0, hi))
                if not (hi > lo):
                    lo, hi = 0.0, 1.0
            else:
                lo, hi = 0.0, 1.0

            xs = np.linspace(lo, hi, 800)
            h_t = _silverman_bandwidth(mtp)
            h_c = _silverman_bandwidth(mcp)
            yt = _kde_reflect(mtp, xs, h_t)
            yc = _kde_reflect(mcp, xs, h_c)

            ax1.plot(xs, yt, linewidth=2.2, label=f"{label_t} (KDE)", color=used_t, antialiased=True)
            ax1.plot(xs, yc, linewidth=2.2, linestyle="--", label=f"{label_c} (KDE)", color=used_c, antialiased=True)

            if shade_overlap:
                ax1.fill_between(xs, np.minimum(yt, yc), 0, alpha=0.12, color="grey", rasterized=False)

        # Means
        ax1.axvline(float(np.mean(mtp)), linestyle=":", linewidth=1.8, color=used_t, alpha=0.95)
        ax1.axvline(float(np.mean(mcp)), linestyle=":", linewidth=1.8, color=used_c, alpha=0.95)

        # Cosmetics
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel("Density")
        ax1.set_title(title)
        ax1.set_xlim(0.0, 1.0)
        ax1.grid(True, linewidth=0.5, alpha=0.45)
        for spine in ("top", "right"):
            ax1.spines[spine].set_visible(False)
        ax1.legend(frameon=False)

    # ------- Data -----------------------------------------------------------
    diag_resolved = _resolve_overlap_diag(diag)
    d = np.asarray(getattr(diag_resolved, "d"), dtype=float)
    m_post = np.asarray(getattr(diag_resolved, "m_hat"), dtype=float)
    m_raw = getattr(diag_resolved, "m_hat_raw", None)
    if m_raw is not None:
        m_raw_arr = np.asarray(m_raw, dtype=float)
        m = m_raw_arr if m_raw_arr.shape == m_post.shape else m_post
    else:
        m = m_post

    if d.ndim != 2 or m.ndim != 2:
        raise ValueError("Expected multi-treatment diag: d and m_hat must be 2D arrays (n, K).")
    if d.shape != m.shape:
        raise ValueError(f"d and m_hat must have same shape (n, K). Got d={d.shape}, m_hat={m.shape}.")

    n, K = d.shape
    if not (0 <= baseline_idx < K):
        raise ValueError(f"baseline_idx must be in [0, {K-1}]")

    # treatment names
    if treatment_names is None:
        treatment_names = getattr(diag_resolved, "treatment_names", None) or getattr(diag_resolved, "d_names", None)
    if not treatment_names or len(treatment_names) != K:
        treatment_names = [str(k) for k in range(K)]

    # which k to plot
    if treatment_idx is None:
        ks = [k for k in range(K) if k != baseline_idx]
    elif isinstance(treatment_idx, int):
        ks = [treatment_idx]
    else:
        ks = list(treatment_idx)

    ks = [int(k) for k in ks]
    for k in ks:
        if not (0 <= k < K):
            raise ValueError(f"treatment_idx contains invalid k={k} for K={K}")
        if k == baseline_idx:
            raise ValueError("treatment_idx cannot include baseline_idx (comparison would be baseline vs baseline).")

    # clean finite rows (по соответствующему столбцу m_k и one-hot d)
    # (чистим по всему d и m, чтобы не было NaN/inf)
    mask = np.isfinite(m).all(axis=1) & np.isfinite(d).all(axis=1)
    d = d[mask]
    m = m[mask]

    # ------- Figure/axes with high DPI & scaled fonts ----------------------
    rc = {
        "font.size": 11 * font_scale,
        "axes.titlesize": 13 * font_scale,
        "axes.labelsize": 12 * font_scale,
        "legend.fontsize": 10 * font_scale,
        "xtick.labelsize": 10 * font_scale,
        "ytick.labelsize": 10 * font_scale,
    }

    with mpl.rc_context(rc):
        # Single plot case (allow ax)
        if len(ks) == 1:
            k = ks[0]
            ax_provided = ax is not None
            if not ax_provided:
                fig, ax1 = plt.subplots(figsize=figsize, dpi=dpi)
            else:
                fig = ax.figure
                ax1 = ax
                try:
                    fig.set_dpi(dpi)
                except Exception:
                    pass

            mt, mc = _pairwise_scores(m, d, tr_idx=k, base_idx=baseline_idx)

            if mt.size == 0 or mc.size == 0:
                raise ValueError(
                    f"Both groups must have at least one observation for baseline={baseline_idx} vs k={k}."
                )

            _plot_one(
                ax1, mt, mc,
                label_t=f"T={treatment_names[k]} (n={mt.size})",
                label_c=f"T={treatment_names[baseline_idx]} (n={mc.size})",
                xlabel=(
                    f"Pairwise score P(D={treatment_names[k]} | X, "
                    f"D in {{{treatment_names[baseline_idx]}, {treatment_names[k]}}})"
                ),
                title=f"Pairwise overlap score: {treatment_names[baseline_idx]} vs {treatment_names[k]}",
            )

            fig.tight_layout()

            if save is not None:
                ext = str(save).lower().split(".")[-1]
                _dpi = save_dpi or (300 if ext in {"png", "jpg", "jpeg", "tif", "tiff"} else dpi)
                fig.savefig(
                    save, dpi=_dpi, bbox_inches="tight", pad_inches=0.1,
                    transparent=transparent,
                    facecolor="none" if transparent else "white",
                )
            if not ax_provided:
                plt.close(fig)
            return fig

        # Multi-panel case (ax not supported)
        if ax is not None:
            raise ValueError("`ax` can be used only when plotting a single treatment comparison. "
                             "Set treatment_idx=int for single plot, or pass ax=None.")

        # layout
        n_plots = len(ks)
        ncols = 2 if n_plots > 1 else 1
        nrows = int(np.ceil(n_plots / ncols))

        # scale figsize a bit with number of panels
        fig_w = max(figsize[0], figsize[0] * (ncols / 2))
        fig_h = max(figsize[1], figsize[1] * (nrows / 1.6))
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h), dpi=dpi)
        axs = np.atleast_1d(axs).ravel()

        for i, k in enumerate(ks):
            ax1 = axs[i]
            mt, mc = _pairwise_scores(m, d, tr_idx=k, base_idx=baseline_idx)

            if mt.size == 0 or mc.size == 0:
                ax1.set_title(f"{treatment_names[baseline_idx]} vs {treatment_names[k]} (insufficient data)")
                ax1.axis("off")
                continue

            _plot_one(
                ax1, mt, mc,
                label_t=f"T={treatment_names[k]} (n={mt.size})",
                label_c=f"T={treatment_names[baseline_idx]} (n={mc.size})",
                xlabel=(
                    f"Pairwise score P(D={treatment_names[k]} | X, "
                    f"D in {{{treatment_names[baseline_idx]}, {treatment_names[k]}}})"
                ),
                title=f"Pairwise overlap: {treatment_names[baseline_idx]} vs {treatment_names[k]}",
            )

        # turn off unused axes
        for j in range(n_plots, len(axs)):
            axs[j].axis("off")

        fig.tight_layout()

        if save is not None:
            ext = str(save).lower().split(".")[-1]
            _dpi = save_dpi or (300 if ext in {"png", "jpg", "jpeg", "tif", "tiff"} else dpi)
            fig.savefig(
                save, dpi=_dpi, bbox_inches="tight", pad_inches=0.1,
                transparent=transparent,
                facecolor="none" if transparent else "white",
            )
        plt.close(fig)
        return fig


def overlap_plot(
    data: MultiCausalData,
    estimate: MultiCausalEstimate,
    **kwargs: Any,
) -> plt.Figure:
    """Convenience wrapper to match `overlap_plot(data, estimate)` API style."""
    if not isinstance(data, MultiCausalData):
        raise TypeError(f"data must be MultiCausalData, got {type(data).__name__}.")
    if not isinstance(estimate, MultiCausalEstimate):
        raise TypeError(f"estimate must be MultiCausalEstimate, got {type(estimate).__name__}.")
    return plot_m_overlap(estimate, treatment_names=list(data.treatment_names), **kwargs)


__all__ = ["plot_m_overlap", "overlap_plot"]
