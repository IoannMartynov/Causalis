from typing import Tuple, Optional, Any, Union
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from causalis.data_contracts.causal_estimate import CausalEstimate
from causalis.data_contracts.causal_diagnostic_data import UnconfoundednessDiagnosticData


def _resolve_overlap_diag(
    diag: Union[UnconfoundednessDiagnosticData, CausalEstimate, dict]
) -> UnconfoundednessDiagnosticData:
    if isinstance(diag, UnconfoundednessDiagnosticData):
        resolved = diag
    elif isinstance(diag, CausalEstimate):
        resolved = diag.diagnostic_data
    elif isinstance(diag, dict):
        resolved = diag.get("diagnostic_data", None)
    else:
        resolved = getattr(diag, "diagnostic_data", None)

    if resolved is None:
        raise ValueError(
            "plot_m_overlap expects UnconfoundednessDiagnosticData or CausalEstimate "
            "with diagnostic_data. Call estimate(..., diagnostic_data=True)."
        )

    if not hasattr(resolved, "m_hat") or not hasattr(resolved, "d"):
        raise ValueError("diagnostic_data must include both `m_hat` and `d`.")

    return resolved


def plot_m_overlap(
    diag: Union[UnconfoundednessDiagnosticData, CausalEstimate, dict],
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
) -> plt.Figure:
    """
    Overlap plot for m(x)=P(D=1|X) with high-res rendering.
    - x in [0,1]
    - Stable NumPy KDE w/ boundary reflection (no SciPy warnings)
    - Uses Matplotlib default colors unless color_t/color_c are provided

    Parameters
    ----------
    diag : UnconfoundednessDiagnosticData or CausalEstimate
        Diagnostic data directly, or an estimate containing diagnostic_data with m_hat and d.
    clip : tuple, default (0.01, 0.99)
        Quantiles to clip for KDE range.
    bins : str or int, default "fd"
        Histogram bins.
    kde : bool, default True
        Whether to show KDE.
    shade_overlap : bool, default True
        Whether to shade the overlap area.
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on.
    figsize : tuple, default (9, 5.5)
        Figure size.
    dpi : int, default 220
        Dots per inch.
    font_scale : float, default 1.15
        Font scaling factor.
    save : str, optional
        Path to save the figure.
    save_dpi : int, optional
        DPI for saving.
    transparent : bool, default False
        Whether to save with transparency.
    color_t : color, optional
        Color for treated group.
    color_c : color, optional
        Color for control group.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """

    # ------- Helpers --------------------------------------------------------
    def _silverman_bandwidth(x):
        x = np.asarray(x, float)
        n = x.size
        if n < 2:
            return 0.04
        sd = np.std(x, ddof=1)
        iqr = np.subtract(*np.percentile(x, [75, 25]))
        s = sd if iqr <= 0 else min(sd, iqr / 1.34)
        h = 0.9 * s * n ** (-1 / 5)
        return float(max(h, 0.02))

    def _kde_reflect(x, xs, h):
        x = np.asarray(x, float)
        if x.size == 0:
            return np.zeros_like(xs)
        if x.size < 2 or np.std(x) < 1e-8:
            mu = float(np.mean(x)) if x.size else 0.5
            h0 = max(h, 0.02)
            z = (xs - mu) / h0
            return np.exp(-0.5 * z ** 2) / (np.sqrt(2 * np.pi) * h0)
        xr = np.concatenate([x, -x, 2 - x])  # reflect at 0 and 1
        density = np.zeros_like(xs, dtype=float)
        norm = xr.size * np.sqrt(2 * np.pi) * h
        chunk_size = max(256, int(np.ceil(2_000_000 / max(xs.size, 1))))
        for start in range(0, xr.size, chunk_size):
            stop = min(start + chunk_size, xr.size)
            diff = (xs[None, :] - xr[start:stop, None]) / h
            np.square(diff, out=diff)
            diff *= -0.5
            np.exp(diff, out=diff)
            density += diff.sum(axis=0)
        return density / norm

    def _patch_color(patches, fallback):
        # Grab facecolor from the first bar; fallback to cycle color if needed
        for p in patches:
            fc = p.get_facecolor()
            if fc is not None:
                return fc  # RGBA
        return fallback

    # ------- Data -----------------------------------------------------------
    diag_resolved = _resolve_overlap_diag(diag)
    d = np.asarray(diag_resolved.d).astype(int)
    m = np.asarray(diag_resolved.m_hat, dtype=float)
    mask = np.isfinite(m) & np.isfinite(d)
    d, m = d[mask], m[mask]
    mt = m[d == 1]
    mc = m[d == 0]
    if mt.size == 0 or mc.size == 0:
        raise ValueError("Both treated and control must have at least one observation after cleaning.")

    # Clamp to [0,1] to keep plot stable and inside bounds
    mtp = np.clip(mt, 0.0, 1.0)
    mcp = np.clip(mc, 0.0, 1.0)

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
        ax_provided = ax is not None
        if not ax_provided:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        else:
            fig = ax.figure
            try:
                fig.set_dpi(dpi)
            except Exception:
                pass

        # ------- Histograms (ALWAYS [0,1]) ---------------------------------
        ht = ax.hist(mtp, bins=bins, range=(0.0, 1.0), density=True,
                     alpha=0.45, label=f"Treated (n={mt.size})",
                     edgecolor="white", linewidth=0.6,
                     color=color_t)  # None -> default color
        hc = ax.hist(mcp, bins=bins, range=(0.0, 1.0), density=True,
                     alpha=0.45, label=f"Control (n={mc.size})",
                     edgecolor="white", linewidth=0.6,
                     color=color_c)  # None -> default color

        # Determine the actual colors used (so KDE/means match the bars)
        cycle = mpl.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1"])
        used_t = color_t or _patch_color(ht[2], cycle[0])
        used_c = color_c or _patch_color(hc[2], cycle[1])

        # ------- KDE (stable NumPy implementation) -------------------------
        if kde:
            if clip:
                lo, hi = np.quantile(np.clip(m, 0, 1), [clip[0], clip[1]])
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

            ax.plot(xs, yt, linewidth=2.2, label="Treated (KDE)", color=used_t, antialiased=True)
            ax.plot(xs, yc, linewidth=2.2, linestyle="--", label="Control (KDE)", color=used_c, antialiased=True)

            if shade_overlap:
                y_min = np.minimum(yt, yc)
                ax.fill_between(xs, y_min, 0, alpha=0.12, color="grey", rasterized=False)

        # ------- Means ------------------------------------------------------
        ax.axvline(float(mtp.mean()), linestyle=":", linewidth=1.8, color=used_t, alpha=0.95)
        ax.axvline(float(mcp.mean()), linestyle=":", linewidth=1.8, color=used_c, alpha=0.95)

        # ------- Cosmetics --------------------------------------------------
        ax.set_xlabel(r"$m(x) = \mathbb{P}(D=1 \mid X)$")
        ax.set_ylabel("Density")
        ax.set_title("Propensity Overlap by Treatment Group")
        ax.set_xlim(0.0, 1.0)
        ax.grid(True, linewidth=0.5, alpha=0.45)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.legend(frameon=False)
        fig.tight_layout()

        # ------- Optional save ---------------------------------------------
        if save is not None:
            ext = str(save).lower().split(".")[-1]
            _dpi = save_dpi or (300 if ext in {"png", "jpg", "jpeg", "tif", "tiff"} else dpi)
            fig.savefig(
                save, dpi=_dpi, bbox_inches="tight", pad_inches=0.1,
                transparent=transparent,
                facecolor="none" if transparent else "white"
            )

        if not ax_provided:
            plt.close(fig)

    return fig
