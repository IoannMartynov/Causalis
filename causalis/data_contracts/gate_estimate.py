from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator


class GateEstimate(BaseModel):
    """Result contract for Group Average Treatment Effects (GATE)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    estimand: str = "GATE"
    model: str = "IRM"

    group_names: List[str]

    values: np.ndarray
    std_errors: np.ndarray
    test_stats: np.ndarray
    p_values: np.ndarray
    ci_lower: np.ndarray
    ci_upper: np.ndarray
    alpha: float

    covariance: pd.DataFrame
    summary_table: pd.DataFrame
    model_options: Dict[str, Any] = Field(default_factory=dict)

    n_group: np.ndarray
    n_treated: np.ndarray
    n_control: np.ndarray
    share_treated: np.ndarray
    mean_phi: np.ndarray
    std_phi: np.ndarray
    mean_propensity: np.ndarray
    min_propensity: np.ndarray
    max_propensity: np.ndarray

    time: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    diagnostic_data: Optional[Dict[str, Any]] = None

    @model_validator(mode="after")
    def _validate_shapes(self) -> "GateEstimate":
        k = len(self.group_names)
        if k == 0:
            raise ValueError("group_names must not be empty.")

        one_d_fields = (
            "values",
            "std_errors",
            "test_stats",
            "p_values",
            "ci_lower",
            "ci_upper",
            "n_group",
            "n_treated",
            "n_control",
            "share_treated",
            "mean_phi",
            "std_phi",
            "mean_propensity",
            "min_propensity",
            "max_propensity",
        )
        for field_name in one_d_fields:
            arr = np.asarray(getattr(self, field_name)).reshape(-1)
            if arr.size != k:
                raise ValueError(f"{field_name} must have length {k}, got {arr.size}.")

        if not (0.0 < float(self.alpha) < 1.0):
            raise ValueError("alpha must be in (0, 1).")

        if self.covariance.shape != (k, k):
            raise ValueError(f"covariance must have shape ({k}, {k}).")
        if self.summary_table.shape[0] != k:
            raise ValueError(f"summary_table must have {k} rows.")

        return self

    def summary(self) -> pd.DataFrame:
        """Return per-group GATE summary table."""
        return self.summary_table.copy()
