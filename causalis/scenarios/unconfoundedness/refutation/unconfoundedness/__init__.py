"""Balance and sensitivity diagnostics for unconfoundedness.

These utilities help answer three practical questions after fitting an
unconfoundedness model such as :class:`~causalis.scenarios.unconfoundedness.model.IRM`:

1. Are the observed confounders balanced after weighting?
2. Which variables still have the worst imbalance?
3. How large could hidden-confounding bias be under user-chosen assumptions?

Notes
-----
For balance diagnostics, the main quantity is the standardized mean
difference (SMD) for each confounder :math:`X_j`:

.. math::

    \mathrm{SMD}_j =
    \frac{|\mu_{1j}^{(w)} - \mu_{0j}^{(w)}|}
    {\sqrt{(s_{1j}^{2,(w)} + s_{0j}^{2,(w)}) / 2}}.

Small weighted SMDs mean the treated and control pseudo-populations look
similar on observed covariates. This is evidence in favor of the design, but
it does not prove ignorability for unobserved confounders.

Examples
--------
>>> from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
>>> from causalis.dgp import obs_linear_26_dataset
>>> from causalis.scenarios.unconfoundedness.model import IRM
>>> from causalis.scenarios.unconfoundedness.refutation.unconfoundedness import (
...     run_unconfoundedness_diagnostics,
...     love_plot,
...     sensitivity_analysis,
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
>>> report = run_unconfoundedness_diagnostics(data, estimate)
>>> report["balance"]["worst_features"].head()  # doctest: +SKIP
>>> fig = love_plot(data, estimate)  # doctest: +SKIP
>>> sensitivity_analysis(estimate, r2_y=0.02, r2_d=0.02, rho=1.0)  # doctest: +SKIP
"""

from .love_plot import love_plot
from .unconfoundedness_validation import (
    run_unconfoundedness_diagnostics,
)
from .sensitivity import (
    sensitivity_analysis,
    get_sensitivity_summary,
    sensitivity_benchmark,
    compute_bias_aware_ci,
)

__all__ = [
    "love_plot",
    "run_unconfoundedness_diagnostics",
    "sensitivity_analysis",
    "get_sensitivity_summary",
    "sensitivity_benchmark",
    "compute_bias_aware_ci",
]
