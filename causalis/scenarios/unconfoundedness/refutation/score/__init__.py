r"""Score-based diagnostics and plots for unconfoundedness.

These tools check whether the orthogonal score used by IRM behaves the way it
should after fitting:

1. The score moments should be close to zero.
2. Small perturbations of nuisance functions should not move the score much.
3. A few observations should not dominate the estimate.

Notes
-----
For the ATE case, the estimating equation is based on an orthogonal score
:math:`\psi(W; \theta, \eta)` with target condition

.. math::

    \mathbb{E}[\psi(W; \theta_0, \eta_0)] = 0.

In practice, score diagnostics inspect the empirical score
:math:`\hat\psi_i`, finite-basis derivatives with respect to nuisance parts,
and the tail behavior of :math:`|\hat\psi_i|`.

Examples
--------
>>> from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
>>> from causalis.dgp import obs_linear_26_dataset
>>> from causalis.scenarios.unconfoundedness.model import IRM
>>> from causalis.scenarios.unconfoundedness.refutation.score import (
...     run_score_diagnostics,
...     plot_influence_instability,
...     plot_residual_diagnostics,
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
>>> report = run_score_diagnostics(data, estimate)
>>> report["summary"]  # doctest: +SKIP
>>> fig1 = plot_influence_instability(estimate, data=data)  # doctest: +SKIP
>>> fig2 = plot_residual_diagnostics(estimate, data=data)  # doctest: +SKIP
"""

from .score_validation import (
    run_score_diagnostics,
)
from .influence_plot import (
    plot_influence_instability,
)
from .residual_plots import (
    plot_residual_diagnostics,
)

__all__ = [
    "run_score_diagnostics",
    "plot_influence_instability",
    "plot_residual_diagnostics",
]
