from __future__ import annotations

from .causaldata import CausalData
from .multicausaldata import MultiCausalData
from .causaldata_instrumental import CausalDataInstrumental
from .panel_data_scm import PanelDataSCM
from .panel_estimate import PanelEstimate
from .causal_estimate import CausalEstimate
from .gate_estimate import GateEstimate
from .gate_contrast_estimate import GateContrastEstimate
from .causal_diagnostic_data import DiagnosticData, UnconfoundednessDiagnosticData, RegressionChecks

_DGP_EXPORTS = {
    "generate_rct",
    "generate_classic_rct",
    "classic_rct_gamma",
    "obs_linear_effect",
    "make_gold_linear",
    "obs_linear_26_dataset",
    "generate_classic_rct_26",
    "classic_rct_gamma_26",
    "generate_cuped_binary",
    "make_cuped_binary_26",
    "CausalDatasetGenerator",
}

__all__ = [
    "CausalData",
    "MultiCausalData",
    "CausalDataInstrumental",
    "PanelDataSCM",
    "PanelEstimate",
    "CausalEstimate",
    "GateEstimate",
    "GateContrastEstimate",
    "DiagnosticData",
    "UnconfoundednessDiagnosticData",
    "RegressionChecks",
    "generate_rct",
    "generate_classic_rct",
    "classic_rct_gamma",
    "obs_linear_effect",
    "make_gold_linear",
    "obs_linear_26_dataset",
    "generate_classic_rct_26",
    "classic_rct_gamma_26",
    "generate_cuped_binary",
    "make_cuped_binary_26",
    "CausalDatasetGenerator",
]


def __getattr__(name: str):
    if name in _DGP_EXPORTS:
        from causalis import dgp

        value = getattr(dgp, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
