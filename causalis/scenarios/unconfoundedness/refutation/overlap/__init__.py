"""Overlap diagnostics and plotting utilities for propensity scores.

These tools help answer three basic overlap questions after fitting IRM:

1. Do treated and control units share the same propensity region?
2. Are inverse-probability weights becoming unstable?
3. Is the propensity model calibrated to observed treatment rates?

Notes
-----
Overlap, or positivity, means the propensity score stays away from the edges:

.. math::

    0 < m(X) = \mathbb{P}(D=1 \mid X) < 1.

When many units have :math:`m(X)` very close to `0` or `1`, inverse-probability
weights like :math:`1 / m(X)` or :math:`1 / (1-m(X))` can explode, making the
causal estimate unstable even if the identifying assumptions are otherwise
plausible.

Examples
--------
>>> from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
>>> from causalis.dgp import obs_linear_26_dataset
>>> from causalis.scenarios.unconfoundedness.model import IRM
>>> from causalis.scenarios.unconfoundedness.refutation.overlap import (
...     run_overlap_diagnostics,
...     plot_m_overlap,
...     plot_propensity_reliability,
... )
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
>>> report = run_overlap_diagnostics(data, estimate)
>>> report["summary"]  # doctest: +SKIP
>>> fig1 = plot_m_overlap(estimate)  # doctest: +SKIP
>>> fig2 = plot_propensity_reliability(estimate, data=data)  # doctest: +SKIP
"""

from __future__ import annotations

from .overlap_validation import run_overlap_diagnostics
from .overlap_plot import plot_m_overlap
from .reliability_plot import plot_propensity_reliability

__all__ = [
    "run_overlap_diagnostics",
    "plot_m_overlap",
    "plot_propensity_reliability",
]
