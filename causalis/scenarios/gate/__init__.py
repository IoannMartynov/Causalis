"""Group Average Treatment Effect (GATE) estimation utilities."""

__all__ = ["estimate_gate_from_irm"]

from causalis.scenarios.gate.model import estimate_gate_from_irm


def __getattr__(name: str):
    if name == "estimate_gate_from_irm":
        from causalis.scenarios.gate.model import estimate_gate_from_irm as value
        globals()[name] = value
        return value
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
