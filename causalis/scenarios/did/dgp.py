from __future__ import annotations

from typing import Optional, Union

import pandas as pd

from causalis.data_contracts.panel_data_did import PanelDataDID
from causalis.dgp.panel_data_did.functional import generate_did_gamma

PanelOutput = Union[pd.DataFrame, PanelDataDID]


def generate_did_gamma_26(
    *,
    seed: int = 42,
    return_panel_data: bool = True,
    include_oracles: bool = True,
    n_treated_units: int = 20,
    n_control_units: int = 60,
    n_pre_periods: Optional[int] = None,
    n_post_periods: Optional[int] = None,
    treatment_effect_rate: float = 0.08,
    treatment_effect_slope: float = 0.0,
    **advanced_params,
) -> PanelOutput:
    """Generate scenario-style Gamma DID panel data with Causalis 26 defaults."""
    return generate_did_gamma(
        seed=seed,
        return_panel_data=return_panel_data,
        include_oracles=include_oracles,
        n_treated_units=n_treated_units,
        n_control_units=n_control_units,
        n_pre_periods=n_pre_periods,
        n_post_periods=n_post_periods,
        treatment_effect_rate=treatment_effect_rate,
        treatment_effect_slope=treatment_effect_slope,
        advanced_params=advanced_params,
    )
