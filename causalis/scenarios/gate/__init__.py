r"""Group-level treatment effect estimation utilities for fitted IRM models.

This submodule exposes:

- ``estimate_gate_from_irm`` for
  :math:`\theta_g^{\mathrm{GATE}} = \mathbb{E}[Y(1)-Y(0)\mid G=g]`
- ``estimate_gatet_from_irm`` for
  :math:`\theta_g^{\mathrm{GATET}} = \mathbb{E}[Y(1)-Y(0)\mid G=g, D=1]`

The first estimand is a subgroup ATE. The second is a subgroup ATT among the
treated inside subgroup :math:`g`, not a treated-only average of the ordinary
GATE score. Both estimands assume subgroup membership is a pre-treatment
partition.
"""

__all__ = ["estimate_gate_from_irm", "estimate_gatet_from_irm", "gate_plot", "plot_gate_estimate"]

from causalis.scenarios.gate.model import estimate_gate_from_irm
from causalis.scenarios.gate.model import estimate_gatet_from_irm
from causalis.scenarios.gate.gate_plot import gate_plot, plot_gate_estimate


def __getattr__(name: str):
    if name == "estimate_gate_from_irm":
        from causalis.scenarios.gate.model import estimate_gate_from_irm as value
        globals()[name] = value
        return value
    if name == "estimate_gatet_from_irm":
        from causalis.scenarios.gate.model import estimate_gatet_from_irm as value
        globals()[name] = value
        return value
    if name == "gate_plot":
        from causalis.scenarios.gate.gate_plot import gate_plot as value
        globals()[name] = value
        return value
    if name == "plot_gate_estimate":
        from causalis.scenarios.gate.gate_plot import plot_gate_estimate as value
        globals()[name] = value
        return value
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
