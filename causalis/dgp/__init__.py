"""
Synthetic data-generating processes (DGPs) for causal benchmarking and examples.

This package collects the public dataset builders used across ``causalis`` for
single-treatment tabular data, multi-treatment data, instrumental-variable
placeholders, and synthetic-control style panel data.

The most commonly used entry points are:

- ``generate_rct`` for randomized experiments with optional oracle columns and
  optional pre-period covariates.
- ``obs_linear_effect`` and ``obs_linear_26_dataset`` for observational data
  with confounding and ground-truth nuisance objects.
- ``generate_scm_data`` for synthetic panel data with one treated unit and a
  donor pool.
- ``generate_did_data`` for simultaneous-adoption panel data with treated and
  never-treated groups.
- ``generate_multitreatment`` for one-hot multi-arm treatment assignment.

Notes
-----
Across the package, generators follow a few conventions:

- Outcomes are typically stored in ``y``.
- Binary treatment indicators are typically stored in ``d``.
- Confounders are returned as ``x1``, ``x2``, ... unless explicit names are
  provided in the generator configuration.
- When ``include_oracle=True``, many generators also expose ground-truth
  columns such as propensities or potential-outcome means. These oracle columns
  are intended for benchmarking, diagnostics, and unit tests rather than as
  inputs to downstream estimators.

Examples
--------
>>> from causalis.dgp import (
...     generate_rct,
...     generate_scm_data,
...     generate_did_data,
...     obs_linear_26_dataset,
... )
>>> rct = generate_rct(
...     n=1000,
...     outcome_type="binary",
...     outcome_params={"p": {"A": 0.10, "B": 0.12}},
...     return_causal_data=True,
... )
>>> rct.treatment, rct.outcome
('d', 'y')
>>> obs = obs_linear_26_dataset(n=1000, seed=3141, return_causal_data=True)
>>> sorted(col for col in ["m", "g0", "g1", "cate"] if col in obs.df.columns)
['cate', 'g0', 'g1', 'm']
>>> panel = generate_scm_data(n_donors=5, n_pre_periods=24, n_post_periods=6)
>>> panel.df.shape[0] > 0  # doctest: +SKIP
True
>>> did_panel = generate_did_data(n_treated_units=5, n_control_units=10)
>>> did_panel.design_type
'simultaneous_adoption'
"""

from .base import _sigmoid, _logit
from .causaldata import (
    CausalDatasetGenerator,
    generate_rct,
    generate_classic_rct,
    classic_rct_gamma,
    obs_linear_effect,
    make_gold_linear, obs_linear_26_dataset,
    generate_classic_rct_26,
    classic_rct_gamma_26,
    make_cuped_tweedie, generate_cuped_tweedie_26,
    generate_cuped_binary, make_cuped_binary_26
)
from .causaldata_instrumental import generate_iv_data
from .panel_data_scm import generate_scm_data
from .panel_data_did import (
    generate_did_data,
    generate_did_gamma,
    generate_did_gamma_data,
    generate_did_poisson_data,
)

__all__ = [
    "CausalDatasetGenerator",
    "generate_rct",
    "generate_classic_rct",
    "classic_rct_gamma",
    "obs_linear_effect",
    "generate_iv_data",
    "make_gold_linear",
    "obs_linear_26_dataset",
    "generate_classic_rct_26",
    "classic_rct_gamma_26",
    "make_cuped_tweedie",
    "generate_cuped_tweedie_26",
    "generate_cuped_binary",
    "make_cuped_binary_26",
    "generate_scm_data",
    "generate_did_data",
    "generate_did_gamma",
    "generate_did_gamma_data",
    "generate_did_poisson_data",
]
