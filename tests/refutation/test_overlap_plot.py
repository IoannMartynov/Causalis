import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from causalis.data_contracts.causal_diagnostic_data import (
    MultiUnconfoundednessDiagnosticData,
    UnconfoundednessDiagnosticData,
)
from causalis.scenarios.multi_unconfoundedness.refutation.overlap import (
    plot_m_overlap as plot_multi_overlap,
)
from causalis.scenarios.unconfoundedness.refutation.overlap import (
    plot_m_overlap as plot_overlap,
)


def test_plot_m_overlap_handles_large_sample_kde_smoke():
    rng = np.random.default_rng(123)
    n = 120_000
    x = rng.normal(size=n)
    m = 1.0 / (1.0 + np.exp(-(0.9 * x)))
    d = rng.binomial(1, m)

    diag = UnconfoundednessDiagnosticData(m_hat=m, d=d)

    fig = plot_overlap(diag, kde=True, bins="fd")

    assert fig is not None
    assert len(fig.axes) == 1
    ax = fig.axes[0]
    assert ax.get_xlim() == (0.0, 1.0)
    labels = [line.get_label() for line in ax.lines]
    assert "Treated (KDE)" in labels
    assert "Control (KDE)" in labels

    plt.close(fig)


def test_multi_plot_m_overlap_handles_large_pairwise_sample_kde_smoke():
    rng = np.random.default_rng(321)
    n = 80_000
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    logits = np.column_stack(
        [
            0.3 + 0.5 * x1,
            -0.2 + 0.6 * x2,
            0.1 - 0.4 * x1 - 0.3 * x2,
        ]
    )
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)

    draws = rng.random(size=n)
    labels = (draws[:, None] > np.cumsum(probs, axis=1)[:, :-1]).sum(axis=1)
    d = np.eye(3, dtype=int)[labels]

    diag = MultiUnconfoundednessDiagnosticData(m_hat=probs, d=d)

    fig = plot_multi_overlap(
        diag,
        kde=True,
        bins="fd",
        treatment_idx=1,
        treatment_names=["d_0", "d_1", "d_2"],
    )

    assert fig is not None
    assert len(fig.axes) == 1
    ax = fig.axes[0]
    assert ax.get_xlim() == (0.0, 1.0)
    labels = [line.get_label() for line in ax.lines]
    assert any("KDE" in label for label in labels)

    plt.close(fig)


def test_multi_plot_m_overlap_wraps_long_panel_text():
    rng = np.random.default_rng(777)
    n = 5_000
    logits = rng.normal(size=(n, 3))
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)

    draws = rng.random(size=n)
    labels = (draws[:, None] > np.cumsum(probs, axis=1)[:, :-1]).sum(axis=1)
    d = np.eye(3, dtype=int)[labels]
    diag = MultiUnconfoundednessDiagnosticData(m_hat=probs, d=d)

    fig = plot_multi_overlap(
        diag,
        kde=True,
        bins="fd",
        treatment_names=["control", "neg_contact_flg", "neg_contact_flg_error_flg"],
    )

    assert fig is not None
    wrapped_titles = [ax.get_title() for ax in fig.axes if ax.get_title()]
    wrapped_xlabels = [ax.get_xlabel() for ax in fig.axes if ax.get_xlabel()]
    assert any("\n" in title for title in wrapped_titles)
    assert any("\n" in xlabel for xlabel in wrapped_xlabels)

    plt.close(fig)
