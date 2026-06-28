from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from causalis.data_contracts.causal_diagnostic_data import DiagnosticData


class MultiCausalEstimate(BaseModel):
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
    value: np.ndarray
    ci_upper_absolute: np.ndarray
    ci_lower_absolute: np.ndarray
    value_relative: Optional[np.ndarray] = None
    ci_upper_relative: Optional[np.ndarray] = None
    ci_lower_relative: Optional[np.ndarray] = None
    alpha: float
    p_value: Optional[np.ndarray] = None
    is_significant: List[bool]
    n_treated: int
    n_control: int
    outcome: str
    treatment: List[str]
    n_treated_by_arm: Optional[np.ndarray] = None
    treatment_mean: Optional[np.ndarray] = None
    control_mean: Optional[float] = None
    control_mean_by_arm: Optional[np.ndarray] = None
    contrast_labels: List[str] = Field(default_factory=list)
    confounders: List[str] = Field(default_factory=list)
    time: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    diagnostic_data: Optional[DiagnosticData] = None

    @staticmethod
    def _as_1d_array(x: Any, n_expected: int, *, name: str) -> np.ndarray:
        arr = np.asarray(x)
        if arr.ndim == 0:
            arr = np.repeat(arr.reshape(1), n_expected)
        else:
            arr = arr.reshape(-1)
            if arr.size == 1 and n_expected > 1:
                arr = np.repeat(arr, n_expected)
        if arr.size != n_expected:
            raise ValueError(
                f"'{name}' must have length {n_expected}, got {arr.size}."
            )
        return arr

    def _contrast_names(self, n_effects: int) -> List[str]:
        if self.contrast_labels and len(self.contrast_labels) == n_effects:
            return list(self.contrast_labels)

        treatment = list(self.treatment)
        if len(treatment) >= n_effects + 1:
            baseline = treatment[0]
            return [f"{t} vs {baseline}" for t in treatment[1:n_effects + 1]]

        return [f"contrast_{i + 1}" for i in range(n_effects)]

    def summary(self) -> pd.DataFrame:
        """
        Return a CausalEstimate-like summary for all baseline contrasts.

        Returns
        -------
        pd.DataFrame
            Summary indexed by `field` and with one column per contrast (d_k vs d_0).
        """
        def _fmt_float(val: Optional[float]) -> Optional[str]:
            if val is None:
                return None
            return f"{val:.4f}"

        value = np.asarray(self.value, dtype=float).reshape(-1)
        n_effects = value.size
        if n_effects == 0:
            return pd.DataFrame(columns=["value"]).rename_axis("field")

        ci_low = self._as_1d_array(self.ci_lower_absolute, n_effects, name="ci_lower_absolute").astype(float)
        ci_high = self._as_1d_array(self.ci_upper_absolute, n_effects, name="ci_upper_absolute").astype(float)

        p_value = None
        if self.p_value is not None:
            p_value = self._as_1d_array(self.p_value, n_effects, name="p_value").astype(float)

        value_relative = None
        if self.value_relative is not None:
            value_relative = self._as_1d_array(self.value_relative, n_effects, name="value_relative").astype(float)

        ci_low_rel = None
        if self.ci_lower_relative is not None:
            ci_low_rel = self._as_1d_array(self.ci_lower_relative, n_effects, name="ci_lower_relative").astype(float)

        ci_high_rel = None
        if self.ci_upper_relative is not None:
            ci_high_rel = self._as_1d_array(self.ci_upper_relative, n_effects, name="ci_upper_relative").astype(float)

        is_significant = self._as_1d_array(self.is_significant, n_effects, name="is_significant").astype(bool)

        n_treated_by_arm = None
        if self.n_treated_by_arm is not None:
            n_treated_by_arm = self._as_1d_array(self.n_treated_by_arm, n_effects, name="n_treated_by_arm").astype(int)

        treatment_mean = None
        if self.treatment_mean is not None:
            treatment_mean = self._as_1d_array(self.treatment_mean, n_effects, name="treatment_mean").astype(float)

        control_mean = float(self.control_mean) if self.control_mean is not None else None
        control_mean_by_arm = None
        if self.control_mean_by_arm is not None:
            control_mean_by_arm = self._as_1d_array(
                self.control_mean_by_arm,
                n_effects,
                name="control_mean_by_arm",
            ).astype(float)

        summary_columns: Dict[str, Dict[str, Any]] = {}
        for i, contrast in enumerate(self._contrast_names(n_effects)):
            value_abs = (
                f"{_fmt_float(float(value[i]))} "
                f"(ci_abs: {_fmt_float(float(ci_low[i]))}, {_fmt_float(float(ci_high[i]))})"
            )
            value_rel_str = None
            if value_relative is not None and ci_low_rel is not None and ci_high_rel is not None:
                value_rel_str = (
                    f"{_fmt_float(float(value_relative[i]))} "
                    f"(ci_rel: {_fmt_float(float(ci_low_rel[i]))}, {_fmt_float(float(ci_high_rel[i]))})"
                )

            n_treated_i = int(n_treated_by_arm[i]) if n_treated_by_arm is not None else int(self.n_treated)
            treatment_mean_i = _fmt_float(float(treatment_mean[i])) if treatment_mean is not None else None
            control_mean_i = control_mean
            if control_mean_by_arm is not None:
                control_mean_i = float(control_mean_by_arm[i])

            summary_columns[contrast] = {
                "outcome": self.outcome,
                "estimand": self.estimand,
                "model": self.model,
                "value": value_abs,
                "value_relative": value_rel_str,
                "alpha": _fmt_float(float(self.alpha)),
                "p_value": _fmt_float(float(p_value[i])) if p_value is not None else None,
                "is_significant": bool(is_significant[i]),
                "n_treated": n_treated_i,
                "n_control": int(self.n_control),
                "treatment_mean": treatment_mean_i,
                "control_mean": _fmt_float(control_mean_i),
                "time": self.time,
            }

        return pd.DataFrame(summary_columns).rename_axis("field")
