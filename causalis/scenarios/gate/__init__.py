"""Group Average Treatment Effect (GATE) estimation utilities."""

__all__ = ["estimate_gate_from_irm", "gate_plot", "plot_gate_estimate"]

from causalis.scenarios.gate.model import estimate_gate_from_irm
from causalis.scenarios.gate.gate_plot import gate_plot, plot_gate_estimate


def __getattr__(name: str):
    if name == "estimate_gate_from_irm":
        from causalis.scenarios.gate.model import estimate_gate_from_irm as value
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
