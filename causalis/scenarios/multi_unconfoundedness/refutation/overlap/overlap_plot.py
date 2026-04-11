from typing import Tuple, Optional, Any, List, Union
import textwrap

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

    def _resolve_hist_bins(values: np.ndarray) -> Any:
        """Use shared histogram bins per comparison to keep plots aligned and fast."""
        if not isinstance(bins, str):
            return bins

        values = np.asarray(values, dtype=float)
        if values.size <= 1:
            return np.linspace(0.0, 1.0, 11, dtype=float)

        edges = np.histogram_bin_edges(np.clip(values, 0.0, 1.0), bins=bins, range=(0.0, 1.0))
        n_edges = int(edges.size)
        if n_edges <= 2:
            return edges

        max_bins = 512
        if n_edges - 1 > max_bins:
            return np.linspace(0.0, 1.0, max_bins + 1, dtype=float)

        return edges

    def _pairwise_scores(
        m_arr: np.ndarray,
        base_rows: np.ndarray,
        tr_rows: np.ndarray,
        *,
        tr_idx: int,
        base_idx: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_base = int(base_rows.size)
        n_tr = int(tr_rows.size)
        if n_base == 0 and n_tr == 0:
            empty = np.array([], dtype=float)
            return empty, empty, empty

        pair_rows = np.concatenate((base_rows, tr_rows))
        m_t = np.clip(m_arr[pair_rows, tr_idx], 0.0, 1.0)
        m_c = np.clip(m_arr[pair_rows, base_idx], 0.0, 1.0)
        denom = np.clip(m_t + m_c, 1e-12, None)
        pair_scores = m_t / denom
        return pair_scores[n_base:], pair_scores[:n_base], pair_scores

    def _wrap_piece(text: str, width: int) -> str:
        if width <= 0 or len(text) <= width:
            return text
        return "\n".join(
            textwrap.wrap(
                text,
                width=width,
                break_long_words=True,
                break_on_hyphens=False,
            )
        )

    def _wrap_treatment_name(name: str, width: int = 22) -> str:
        tokens = str(name).replace("-", "_").split("_")
        lines: List[str] = []
        current = ""

        for token in tokens:
            if not token:
                continue
            candidate = token if not current else f"{current}_{token}"
            if len(candidate) <= width:
                current = candidate
                continue
            if current:
                lines.append(current)
            current = token

        if current:
            lines.append(current)

        if not lines:
            return str(name)

        return "\n".join(_wrap_piece(line, width) for line in lines)

    def _wrap_label(text: str, width: int) -> str:
        lines = text.splitlines() or [text]
        return "\n".join(_wrap_piece(line, width) for line in lines)

    def _format_panel_text(
        base_name: str,
        treat_name: str,
        n_t: int,
        n_c: int,
        *,
        compact: bool,
    ) -> Tuple[str, str, str, str]:
        name_width = 18 if compact else 24
        title_width = 34 if compact else 48
        xlabel_width = 42 if compact else 62

        treat_wrapped = _wrap_treatment_name(treat_name, width=name_width)
        base_wrapped = _wrap_treatment_name(base_name, width=name_width)

        label_t = _wrap_label(f"T={treat_name} (n={n_t})", width=title_width)
        label_c = _wrap_label(f"T={base_name} (n={n_c})", width=title_width)

        title = _wrap_label(
            f"Pairwise overlap: {base_name} vs {treat_name}",
            width=title_width,
        )
        xlabel = _wrap_label(
            f"Pairwise score P(D={treat_wrapped} | X, D in {{{base_wrapped}, {treat_wrapped}}})",
            width=xlabel_width,
        )
        return label_t, label_c, title, xlabel

    def _plot_one(
        ax1: plt.Axes,
        mt: np.ndarray,
        mc: np.ndarray,
        pair_scores: np.ndarray,
        *,
        label_t: str,
        label_c: str,
        xlabel: str,
        title: str,
    ):
        # Clamp to [0,1]
        mtp = np.clip(mt, 0.0, 1.0)
        mcp = np.clip(mc, 0.0, 1.0)
        pair_scores = np.clip(pair_scores, 0.0, 1.0)
        hist_bins = _resolve_hist_bins(pair_scores)

        # Histograms
        ht = ax1.hist(
            mtp, bins=hist_bins, range=(0.0, 1.0), density=True,
            alpha=0.45, label=label_t, edgecolor="white", linewidth=0.6,
            color=color_t
        )
        hc = ax1.hist(
            mcp, bins=hist_bins, range=(0.0, 1.0), density=True,
            alpha=0.45, label=label_c, edgecolor="white", linewidth=0.6,
            color=color_c
        )

        cycle = mpl.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1"])
        used_t = color_t or _patch_color(ht[2], cycle[0])
        used_c = color_c or _patch_color(hc[2], cycle[1])

        # KDE
        if kde:
            if clip:
                lo, hi = np.quantile(pair_scores, [clip[0], clip[1]])
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
        ax1.legend(
            frameon=False,
            loc="best",
            handlelength=2.4,
            labelspacing=0.35,
            borderaxespad=0.2,
        )

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

    _, K = d.shape
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
    group_rows = [np.flatnonzero(d[:, k] > 0.5) for k in range(K)]
    baseline_rows = group_rows[baseline_idx]

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
            compact = False
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

            mt, mc, pair_scores = _pairwise_scores(
                m,
                baseline_rows,
                group_rows[k],
                tr_idx=k,
                base_idx=baseline_idx,
            )

            if mt.size == 0 or mc.size == 0:
                raise ValueError(
                    f"Both groups must have at least one observation for baseline={baseline_idx} vs k={k}."
                )

            label_t, label_c, title, xlabel = _format_panel_text(
                treatment_names[baseline_idx],
                treatment_names[k],
                mt.size,
                mc.size,
                compact=compact,
            )
            _plot_one(
                ax1, mt, mc, pair_scores,
                label_t=label_t,
                label_c=label_c,
                xlabel=xlabel,
                title=title,
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
        compact = ncols > 1

        # scale figsize a bit with number of panels
        base_w = max(figsize[0], 7.2)
        base_h = max(figsize[1], 5.8)
        panel_w = 7.2 if compact else base_w
        panel_h = 5.8 if compact else base_h
        fig_w = max(base_w, panel_w * ncols)
        fig_h = max(base_h, panel_h * nrows)
        fig, axs = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(fig_w, fig_h),
            dpi=dpi,
            constrained_layout=True,
        )
        axs = np.atleast_1d(axs).ravel()

        for i, k in enumerate(ks):
            ax1 = axs[i]
            mt, mc, pair_scores = _pairwise_scores(
                m,
                baseline_rows,
                group_rows[k],
                tr_idx=k,
                base_idx=baseline_idx,
            )

            if mt.size == 0 or mc.size == 0:
                ax1.set_title(
                    _wrap_label(
                        f"{treatment_names[baseline_idx]} vs {treatment_names[k]} (insufficient data)",
                        width=34,
                    )
                )
                ax1.axis("off")
                continue

            label_t, label_c, title, xlabel = _format_panel_text(
                treatment_names[baseline_idx],
                treatment_names[k],
                mt.size,
                mc.size,
                compact=compact,
            )
            _plot_one(
                ax1, mt, mc, pair_scores,
                label_t=label_t,
                label_c=label_c,
                xlabel=xlabel,
                title=title,
            )

        # turn off unused axes
        for j in range(n_plots, len(axs)):
            axs[j].axis("off")

        if not getattr(fig, "get_constrained_layout", lambda: False)():
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
