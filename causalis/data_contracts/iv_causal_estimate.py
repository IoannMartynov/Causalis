from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from causalis.data_contracts.causal_diagnostic_data import IVDiagnosticData


class IVCausalEstimate(BaseModel):
    """Result container for instrumental-variable causal effect estimates."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    estimand: str
    model: str
    model_options: Dict[str, Any] = Field(default_factory=dict)
    value: float
    std_error: float
    t_stat: float
    p_value: Optional[float] = None
    ci_lower_absolute: float
    ci_upper_absolute: float
    value_relative: Optional[float] = None
    ci_lower_relative: Optional[float] = None
    ci_upper_relative: Optional[float] = None
    alpha: float
    is_significant: bool
    outcome: str
    treatment: str
    instrument: str
    confounders: List[str] = Field(default_factory=list)
    time: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    diagnostic_data: Optional[IVDiagnosticData] = None

    def summary(self) -> pd.DataFrame:
        """Return a vertical summary DataFrame of the results."""
        def _fmt_float(val: Optional[float]) -> Optional[str]:
            if val is None:
                return None
            return f"{val:.4f}"

        value_abs = (
            f"{_fmt_float(self.value)} "
            f"(ci_abs: {_fmt_float(self.ci_lower_absolute)}, "
            f"{_fmt_float(self.ci_upper_absolute)})"
        )
        value_rel = None
        if self.value_relative is not None:
            value_rel = (
                f"{_fmt_float(self.value_relative)} "
                f"(ci_rel: {_fmt_float(self.ci_lower_relative)}, "
                f"{_fmt_float(self.ci_upper_relative)})"
            )

        summary = {
            "outcome": self.outcome,
            "estimand": self.estimand,
            "model": self.model,
            "value": value_abs,
            "value_relative": value_rel,
            "std_error": _fmt_float(self.std_error),
            "t_stat": _fmt_float(self.t_stat),
            "alpha": _fmt_float(self.alpha),
            "p_value": _fmt_float(self.p_value),
            "is_significant": self.is_significant,
            "treatment": self.treatment,
            "instrument": self.instrument,
            "time": self.time,
        }
        return pd.DataFrame(
            {"field": list(summary.keys()), "value": list(summary.values())}
        ).set_index("field")
