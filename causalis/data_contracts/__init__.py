from __future__ import annotations

from .causaldata import CausalData
from .iv_causal_data import IVCausalData
from .multicausaldata import MultiCausalData
from .panel_data_did import PanelDataDID
from .panel_data_scm import PanelDataSCM
from .panel_did_estimate import CallawaySantAnnaDIDEstimate, PanelDIDDiagnosticData, PanelDIDEstimate
from .panel_estimate import PanelEstimate
from .causal_estimate import CausalEstimate
from .iv_causal_estimate import IVCausalEstimate
from .gate_estimate import GateEstimate
from .gate_contrast_estimate import GateContrastEstimate
from .causal_diagnostic_data import (
    DiagnosticData,
    IVDiagnosticData,
    RegressionChecks,
    UnconfoundednessDiagnosticData,
)
from .sensitivity_analysis_result import SensitivityAnalysisResult

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
    "generate_iv_data",
    "InstrumentalGenerator",
    "IVCausalDatasetGenerator",
}

__all__ = [
    "CausalData",
    "IVCausalData",
    "MultiCausalData",
    "PanelDataDID",
    "PanelDataSCM",
    "PanelDIDDiagnosticData",
    "PanelDIDEstimate",
    "CallawaySantAnnaDIDEstimate",
    "PanelEstimate",
    "CausalEstimate",
    "IVCausalEstimate",
    "GateEstimate",
    "GateContrastEstimate",
    "DiagnosticData",
    "IVDiagnosticData",
    "UnconfoundednessDiagnosticData",
    "RegressionChecks",
    "SensitivityAnalysisResult",
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
    "generate_iv_data",
    "InstrumentalGenerator",
    "IVCausalDatasetGenerator",
]


def __getattr__(name: str):
    if name in _DGP_EXPORTS:
        from causalis import dgp

        value = getattr(dgp, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
