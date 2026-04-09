from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator


_SUPPORTED_ALTERNATIVES = {"two-sided", "greater", "less"}


class GateContrastEstimate(BaseModel):
    """Result contract for a post-estimation subgroup-effect contrast."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    estimand: str = "GATE_CONTRAST"
    model: str = "IRM"
    model_options: Dict[str, Any] = Field(default_factory=dict)

    left_group: str
    right_group: str
    contrast_label: str

    value: float
    std_error: float
    test_stat: float
    p_value: float
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    alpha: float
    alternative: str = "two-sided"
    is_significant: bool

    left_value: float
    right_value: float
    n_left: int
    n_right: int

    time: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    diagnostic_data: Optional[Dict[str, Any]] = None

    @model_validator(mode="after")
    def _validate_fields(self) -> "GateContrastEstimate":
        if not (0.0 < float(self.alpha) < 1.0):
            raise ValueError("alpha must be in (0, 1).")

        alternative = str(self.alternative).lower()
        if alternative not in _SUPPORTED_ALTERNATIVES:
            supported = ", ".join(sorted(_SUPPORTED_ALTERNATIVES))
            raise ValueError(f"alternative must be one of {{{supported}}}. Got {self.alternative!r}.")
        self.alternative = alternative

        if self.left_group == self.right_group:
            raise ValueError("left_group and right_group must be different.")

        if alternative != "two-sided" and (self.ci_lower is not None or self.ci_upper is not None):
            contrast_family = str(self.estimand).replace("_CONTRAST", "")
            raise ValueError(f"One-sided {contrast_family} contrasts must not report ci_lower/ci_upper.")

        return self

    def summary(self) -> pd.DataFrame:
        """Return a CausalEstimate-style summary for the requested contrast."""

        def _fmt_float(val: Optional[float]) -> Optional[str]:
            if val is None:
                return None
            return f"{val:.4f}"

        if self.ci_lower is not None and self.ci_upper is not None:
            value_repr = (
                f"{_fmt_float(self.value)} "
                f"(ci_abs: {_fmt_float(self.ci_lower)}, {_fmt_float(self.ci_upper)})"
            )
        else:
            value_repr = _fmt_float(self.value)

        summary = {
            "estimand": self.estimand,
            "model": self.model,
            "contrast": self.contrast_label,
            "left_group": self.left_group,
            "right_group": self.right_group,
            "value": value_repr,
            "std_error": _fmt_float(self.std_error),
            "test_stat": _fmt_float(self.test_stat),
            "alpha": _fmt_float(self.alpha),
            "alternative": self.alternative,
            "p_value": _fmt_float(self.p_value),
            "is_significant": self.is_significant,
            "left_value": _fmt_float(self.left_value),
            "right_value": _fmt_float(self.right_value),
            "n_left": self.n_left,
            "n_right": self.n_right,
            "time": self.time,
        }
        return pd.DataFrame({"field": list(summary.keys()), "value": list(summary.values())}).set_index("field")
