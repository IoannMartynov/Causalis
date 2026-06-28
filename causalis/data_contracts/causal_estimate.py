from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from causalis.data_contracts.causal_diagnostic_data import DiagnosticData


class CausalEstimate(BaseModel):
    """
    Result container for causal effect estimates.

    Parameters
    ----------
    estimand : str
        The estimand being estimated (e.g., 'ATE', 'ATTE', 'CATE').
    model : str
        The name of the model used for estimation.
    model_options : dict
        Options passed to the model.
    value : float
        The estimated absolute effect.
    ci_upper_absolute : float
        Upper bound of the absolute confidence interval.
    ci_lower_absolute : float
        Lower bound of the absolute confidence interval.
    value_relative : float, optional
        The estimated relative effect.
    ci_upper_relative : float, optional
        Upper bound of the relative confidence interval.
    ci_lower_relative : float, optional
        Lower bound of the relative confidence interval.
    alpha : float
        The significance level (e.g., 0.05).
    p_value : float, optional
        The p-value from the test.
    is_significant : bool
        Whether the result is statistically significant at alpha.
    n_treated : int
        Number of units in the treatment group.
    n_control : int
        Number of units in the control group.
    treatment_mean : float
        Mean outcome in the treatment group.
    control_mean : float
        Mean outcome in the control group.
    outcome : str
        The name of the outcome variable.
    treatment : str
        The name of the treatment variable.
    confounders : list of str, optional
        The names of the confounders used in the model.
    time : str
        The date when the estimate was created (YYYY-MM-DD).
    diagnostic_data : DiagnosticData, optional
        Additional diagnostic data_contracts.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    estimand: str
    model: str
    model_options: Dict[str, Any] = Field(default_factory=dict)
    value: float
    ci_upper_absolute: float
    ci_lower_absolute: float
    value_relative: Optional[float] = None
    ci_upper_relative: Optional[float] = None
    ci_lower_relative: Optional[float] = None
    alpha: float
    p_value: Optional[float] = None
    is_significant: bool
    n_treated: int
    n_control: int
    treatment_mean: float
    control_mean: float
    outcome: str
    treatment: str
    confounders: List[str] = Field(default_factory=list)
    time: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    diagnostic_data: Optional[DiagnosticData] = None

    def summary(self) -> pd.DataFrame:
        """
        Return a summary DataFrame of the results.

        Returns
        -------
        pd.DataFrame
            Summary DataFrame.
        """
        def _fmt_float(val: Optional[float]) -> Optional[str]:
            if val is None:
                return None
            return f"{val:.4f}"

        value_abs = (
            f"{_fmt_float(self.value)} "
            f"(ci_abs: {_fmt_float(self.ci_lower_absolute)}, {_fmt_float(self.ci_upper_absolute)})"
        )
        value_rel = None
        if self.value_relative is not None:
            value_rel = (
                f"{_fmt_float(self.value_relative)} "
                f"(ci_rel: {_fmt_float(self.ci_lower_relative)}, {_fmt_float(self.ci_upper_relative)})"
            )

        summary = {
            "outcome": self.outcome,
            "estimand": self.estimand,
            "model": self.model,
            "value": value_abs,
            "value_relative": value_rel,
            "alpha": _fmt_float(self.alpha),
            "p_value": _fmt_float(self.p_value),
            "is_significant": self.is_significant,
            "n_treated": self.n_treated,
            "n_control": self.n_control,
            "treatment_mean": _fmt_float(self.treatment_mean),
            "control_mean": _fmt_float(self.control_mean),
            "time": self.time,
        }
        return pd.DataFrame({"field": list(summary.keys()), "value": list(summary.values())}).set_index("field")
