from .diagnostics import (
    did_base_design_table,
    did_covariate_balance_table,
    did_support_table,
    raw_did_event_study_table,
    run_did_diagnostics,
)
from .post_inference import (
    did_cluster_influence_table,
    did_influence_table,
    did_post_inference_cell_table,
    run_did_inference_diagnostics,
    run_did_post_inference_diagnostics,
)


def plot_did_support(*args, **kwargs):
    from .diagnostic_plots import plot_did_support as _plot_did_support

    return _plot_did_support(*args, **kwargs)


def plot_raw_did_event_study(*args, **kwargs):
    from .diagnostic_plots import (
        plot_raw_did_event_study as _plot_raw_did_event_study,
    )

    return _plot_raw_did_event_study(*args, **kwargs)


def plot_did_post_inference_event_study(*args, **kwargs):
    from .post_inference_plots import (
        plot_did_post_inference_event_study as _plot_did_post_inference_event_study,
    )

    return _plot_did_post_inference_event_study(*args, **kwargs)


def plot_did_influence_concentration(*args, **kwargs):
    from .post_inference_plots import (
        plot_did_influence_concentration as _plot_did_influence_concentration,
    )

    return _plot_did_influence_concentration(*args, **kwargs)


__all__ = [
    "did_support_table",
    "raw_did_event_study_table",
    "did_covariate_balance_table",
    "did_base_design_table",
    "run_did_diagnostics",
    "plot_did_support",
    "plot_raw_did_event_study",
    "did_post_inference_cell_table",
    "did_influence_table",
    "did_cluster_influence_table",
    "run_did_post_inference_diagnostics",
    "run_did_inference_diagnostics",
    "plot_did_post_inference_event_study",
    "plot_did_influence_concentration",
]
