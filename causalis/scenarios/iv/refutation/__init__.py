"""
Refutation and diagnostic utilities for instrumental-variable scenarios.

This module includes checks for IV assumptions:

- **First-stage strength**: Is the instrument strongly correlated with the treatment?
- **Instrument overlap**: Is the instrument assignment well-spread across covariates?
- **Confounder balance**: Are covariates balanced across instrument groups?
- **Reduced-form sanity**: Does the instrument have an effect on the outcome?
"""

from .diagnostics import (
    first_stage,
    instrument_overlap,
    instrument_overlap_plot,
    reduced_form,
)
from .iv_confounders_balance import iv_confounders_balance

__all__ = [
    "first_stage",
    "instrument_overlap",
    "instrument_overlap_plot",
    "iv_confounders_balance",
    "reduced_form",
]
