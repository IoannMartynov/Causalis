from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Hashable, List, Literal, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from causalis.data_contracts.panel_data_did import TimeLike


class PanelDIDDiagnosticData(BaseModel):
    """Diagnostic payload for scalar panel DID estimators."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    unit_ids: List[Hashable]
    d: np.ndarray
    delta_y: np.ndarray
    x: Optional[np.ndarray] = None
    covariate_names: List[str] = Field(default_factory=list)

    propensity_score: np.ndarray
    control_outcome_evolution: np.ndarray
    treated_weights: np.ndarray
    control_weights: np.ndarray
    influence_scores: np.ndarray

    gamma_hat: np.ndarray
    beta_hat: np.ndarray

    overlap: Dict[str, Any] = Field(default_factory=dict)
    balance: pd.DataFrame = Field(default_factory=pd.DataFrame)
    cluster_scores: Optional[pd.Series] = None

    @model_validator(mode="after")
    def _validate_dimensions(self) -> "PanelDIDDiagnosticData":
        n = len(self.unit_ids)
        for field_name in (
            "d",
            "delta_y",
            "propensity_score",
            "control_outcome_evolution",
            "treated_weights",
            "control_weights",
            "influence_scores",
        ):
            values = np.asarray(getattr(self, field_name), dtype=float)
            if values.ndim != 1 or values.size != n:
                raise ValueError(f"{field_name} must be a one-dimensional array of length n_units.")
            if not np.isfinite(values).all():
                raise ValueError(f"{field_name} must contain only finite values.")

        pscore = np.asarray(self.propensity_score, dtype=float)
        if ((pscore <= 0.0) | (pscore >= 1.0)).any():
            raise ValueError("propensity_score values must be strictly inside (0, 1).")

        if self.x is not None:
            x = np.asarray(self.x, dtype=float)
            if x.ndim != 2 or x.shape[0] != n:
                raise ValueError("x must be a two-dimensional array with n_units rows.")
            if x.shape[1] != len(self.covariate_names):
                raise ValueError("x column count must match covariate_names length.")
            if not np.isfinite(x).all():
                raise ValueError("x must contain only finite values.")
        elif self.covariate_names:
            raise ValueError("covariate_names must be empty when x is None.")

        for field_name in ("gamma_hat", "beta_hat"):
            values = np.asarray(getattr(self, field_name), dtype=float)
            if values.ndim != 1 or values.size < 1:
                raise ValueError(f"{field_name} must be a non-empty one-dimensional array.")
            if not np.isfinite(values).all():
                raise ValueError(f"{field_name} must contain only finite values.")

        return self


class PanelDIDEstimate(BaseModel):
    """Result contract for scalar panel difference-in-differences estimates."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        validate_assignment=True,
    )

    estimand: Literal["ATT"] = "ATT"
    model: str

    treatment_start: TimeLike
    pre_time: TimeLike
    post_time: TimeLike

    att: float
    se: float
    ci_lower: float
    ci_upper: float
    p_value: float
    is_significant: bool
    alpha: float

    n_units: int
    n_treated: int
    n_control: int
    treatment_mean_delta: float
    control_mean_delta: float

    outcome: str
    treatment: str
    covariates: List[str] = Field(default_factory=list)
    cluster_col: Optional[str] = None
    inference: Literal["influence", "clustered_influence"] = "influence"

    diagnostic_data: Optional[PanelDIDDiagnosticData] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def value(self) -> float:
        """CausalEstimate-style alias for the ATT."""

        return self.att

    @property
    def std_error(self) -> float:
        """Readable alias for the influence-function standard error."""

        return self.se

    @model_validator(mode="after")
    def _validate_fields(self) -> "PanelDIDEstimate":
        if self.created_at.tzinfo is None:
            raise ValueError("created_at must be timezone-aware.")

        for field_name in (
            "att",
            "se",
            "ci_lower",
            "ci_upper",
            "p_value",
            "alpha",
            "treatment_mean_delta",
            "control_mean_delta",
        ):
            value = float(getattr(self, field_name))
            if not np.isfinite(value):
                raise ValueError(f"{field_name} must be finite.")

        if self.se < 0.0:
            raise ValueError("se must be non-negative.")
        if self.ci_lower > self.ci_upper:
            raise ValueError("ci_lower must be <= ci_upper.")
        if not (0.0 < self.alpha < 1.0):
            raise ValueError("alpha must be in (0, 1).")
        if not (0.0 <= self.p_value <= 1.0):
            raise ValueError("p_value must be in [0, 1].")

        if self.n_units <= 0:
            raise ValueError("n_units must be positive.")
        if self.n_treated <= 0:
            raise ValueError("n_treated must be positive.")
        if self.n_control <= 0:
            raise ValueError("n_control must be positive.")
        if self.n_units != self.n_treated + self.n_control:
            raise ValueError("n_units must equal n_treated + n_control.")

        if not self.outcome:
            raise ValueError("outcome must be non-empty.")
        if not self.treatment:
            raise ValueError("treatment must be non-empty.")
        if any(not isinstance(c, str) or not c for c in self.covariates):
            raise ValueError("covariates must contain only non-empty strings.")

        try:
            if self.pre_time >= self.post_time:
                raise ValueError("pre_time must be before post_time.")
        except TypeError as exc:
            raise ValueError("pre_time and post_time must be comparable.") from exc

        return self

    @staticmethod
    def _fmt_float(value: Any) -> str | None:
        try:
            out = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(out):
            return None
        return f"{out:.4f}"

    def summary(self) -> pd.DataFrame:
        """Return a compact scalar panel-DID summary table."""

        value = (
            f"{self._fmt_float(self.att)} "
            f"(ci_abs: {self._fmt_float(self.ci_lower)}, {self._fmt_float(self.ci_upper)})"
        )
        summary = {
            "estimand": self.estimand,
            "model": self.model,
            "value": value,
            "std_error": self._fmt_float(self.se),
            "alpha": self._fmt_float(self.alpha),
            "p_value": self._fmt_float(self.p_value),
            "is_significant": self.is_significant,
            "n_units": self.n_units,
            "n_treated": self.n_treated,
            "n_control": self.n_control,
            "pre_time": str(self.pre_time),
            "post_time": str(self.post_time),
            "treatment_start": str(self.treatment_start),
            "treatment_mean_delta": self._fmt_float(self.treatment_mean_delta),
            "control_mean_delta": self._fmt_float(self.control_mean_delta),
            "inference": self.inference,
            "time": self.created_at.strftime("%Y-%m-%d"),
        }
        return pd.DataFrame({"field": list(summary.keys()), "value": list(summary.values())}).set_index("field")
