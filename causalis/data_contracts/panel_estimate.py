from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Hashable, List, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from causalis.data_contracts.panel_data_scm import TimeLike


class PanelEstimate(BaseModel):
    """Result contract for panel estimators such as Synthetic Control.

    Parameters
    ----------
    model : str
        Name of the fitted estimator or pipeline.
    treated_unit : Hashable
        Identifier of the treated unit.
    treatment_start : TimeLike
        Treatment boundary used to split pre/post periods.
    pre_times : list[TimeLike]
        Sorted, strictly pre-treatment periods.
    post_times : list[TimeLike]
        Sorted, strictly post-treatment periods.

    Notes
    -----
    The contract stores aggregate ATTE metrics, full time paths, donor weights,
    and basic diagnostics needed for reporting or downstream checks.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        validate_assignment=True,
    )

    estimand: str = "ATTE"
    model: str

    treated_unit: Hashable
    treatment_start: TimeLike
    pre_times: List[TimeLike]
    post_times: List[TimeLike]

    att: float
    att_sc: float
    ci_upper_absolute: Optional[float] = None
    ci_lower_absolute: Optional[float] = None
    value_relative: Optional[float] = None
    ci_upper_relative: Optional[float] = None
    ci_lower_relative: Optional[float] = None
    alpha: Optional[float] = None
    p_value: Optional[float] = None
    is_significant: Optional[bool] = None

    att_by_time: pd.Series
    att_by_time_sc: pd.Series
    observed_outcome: pd.Series
    synthetic_outcome: pd.Series
    synthetic_outcome_sc: pd.Series

    donor_weights_augmented: Dict[Hashable, float]
    donor_weights_sc: Dict[Hashable, float]

    diagnostics: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def _validate_dimensions(self) -> "PanelEstimate":
        """Validate shape, index alignment, and numeric sanity constraints.

        Returns
        -------
        PanelEstimate
            Validated instance.
        """
        n_pre = len(self.pre_times)
        n_post = len(self.post_times)
        n_all = n_pre + n_post

        if n_pre < 1:
            raise ValueError("pre_times must contain at least one period.")
        if n_post < 1:
            raise ValueError("post_times must contain at least one period.")

        # Build canonical index objects once and reuse them for all alignment checks.
        pre_idx = pd.Index(list(self.pre_times))
        post_idx = pd.Index(list(self.post_times))
        all_idx = pre_idx.append(post_idx)

        if self.created_at.tzinfo is None:
            raise ValueError("created_at must be timezone-aware.")

        try:
            has_overlap = bool(set(self.pre_times).intersection(self.post_times))
        except TypeError as exc:
            raise ValueError("pre_times/post_times contain unhashable values.") from exc
        if has_overlap:
            raise ValueError("pre_times and post_times must be disjoint.")
        try:
            if max(self.pre_times) >= min(self.post_times):
                raise ValueError("Expected all pre_times < all post_times.")
            pre_sorted = sorted(pre_idx.tolist())
            post_sorted = sorted(post_idx.tolist())
        except TypeError as exc:
            raise ValueError("pre_times/post_times contain incomparable values.") from exc
        if list(pre_idx) != pre_sorted or list(post_idx) != post_sorted:
            raise ValueError("pre_times and post_times must be sorted ascending.")

        # Every post-only series must align exactly to post_times.
        if len(self.att_by_time) != n_post:
            raise ValueError("att_by_time length must equal len(post_times).")
        if not self.att_by_time.index.equals(post_idx):
            raise ValueError("att_by_time index must exactly equal post_times (same order).")
        if len(self.att_by_time_sc) != n_post:
            raise ValueError("att_by_time_sc length must equal len(post_times).")
        if not self.att_by_time_sc.index.equals(post_idx):
            raise ValueError("att_by_time_sc index must exactly equal post_times (same order).")

        # Full-path series must align exactly to pre_times followed by post_times.
        if len(self.observed_outcome) != n_all:
            raise ValueError("observed_outcome length must equal len(pre_times)+len(post_times).")
        if not self.observed_outcome.index.equals(all_idx):
            raise ValueError(
                "observed_outcome index must exactly equal pre_times+post_times (same order)."
            )
        if len(self.synthetic_outcome) != n_all:
            raise ValueError("synthetic_outcome length must equal len(pre_times)+len(post_times).")
        if not self.synthetic_outcome.index.equals(all_idx):
            raise ValueError(
                "synthetic_outcome index must exactly equal pre_times+post_times (same order)."
            )
        if len(self.synthetic_outcome_sc) != n_all:
            raise ValueError("synthetic_outcome_sc length must equal len(pre_times)+len(post_times).")
        if not self.synthetic_outcome_sc.index.equals(all_idx):
            raise ValueError(
                "synthetic_outcome_sc index must exactly equal pre_times+post_times (same order)."
            )

        if set(self.donor_weights_augmented.keys()) != set(self.donor_weights_sc.keys()):
            raise ValueError("donor_weights_augmented and donor_weights_sc must use the same donor ids.")
        if len(self.donor_weights_sc) < 1:
            raise ValueError("At least one donor weight is required.")

        # Scalar and vector-valued estimates must be numeric and finite.
        if not (np.isfinite(self.att) and np.isfinite(self.att_sc)):
            raise ValueError("att and att_sc must be finite floats.")
        for series_name in (
            "att_by_time",
            "att_by_time_sc",
            "observed_outcome",
            "synthetic_outcome",
            "synthetic_outcome_sc",
        ):
            numeric = pd.to_numeric(getattr(self, series_name), errors="coerce")
            if numeric.isna().any():
                raise ValueError(
                    f"{series_name} must contain only numeric values (no NaN/non-numeric)."
                )
            if not np.isfinite(numeric.to_numpy()).all():
                raise ValueError(f"{series_name} must contain only finite values.")

        for lower_name, upper_name in (
            ("ci_lower_absolute", "ci_upper_absolute"),
            ("ci_lower_relative", "ci_upper_relative"),
        ):
            lower = getattr(self, lower_name)
            upper = getattr(self, upper_name)
            if (lower is None) ^ (upper is None):
                raise ValueError(f"{lower_name} and {upper_name} must be provided together.")
            if lower is not None:
                if not (np.isfinite(lower) and np.isfinite(upper)):
                    raise ValueError(f"{lower_name} and {upper_name} must be finite when provided.")
                if float(lower) > float(upper):
                    raise ValueError(f"{lower_name} must be <= {upper_name}.")

        if self.value_relative is not None and not np.isfinite(self.value_relative):
            raise ValueError("value_relative must be finite when provided.")
        if self.value_relative is None and self.ci_lower_relative is not None:
            raise ValueError(
                "value_relative must be provided when relative confidence interval bounds are set."
            )
        if self.alpha is not None:
            if not np.isfinite(self.alpha) or not (0.0 < float(self.alpha) < 1.0):
                raise ValueError("alpha must be in (0, 1) when provided.")
        if self.p_value is not None:
            if not np.isfinite(self.p_value) or not (0.0 <= float(self.p_value) <= 1.0):
                raise ValueError("p_value must be in [0, 1] when provided.")
        if self.is_significant is not None and not isinstance(self.is_significant, bool):
            raise ValueError("is_significant must be a bool when provided.")

        # Standard SC weights are a simplex: nonnegative and summing to one.
        w_sc = np.asarray(list(self.donor_weights_sc.values()), dtype=float)
        if not np.isfinite(w_sc).all():
            raise ValueError("donor_weights_sc must be finite.")
        if (w_sc < -1e-12).any():
            raise ValueError("donor_weights_sc must be nonnegative.")
        if abs(float(w_sc.sum()) - 1.0) > 1e-6:
            raise ValueError("donor_weights_sc must sum to 1 (within tolerance).")

        # Augmented weights may be unconstrained in some estimators, so the
        # sum-to-one check is optional and controlled by diagnostics.
        w_aug = np.asarray(list(self.donor_weights_augmented.values()), dtype=float)
        if not np.isfinite(w_aug).all():
            raise ValueError("donor_weights_augmented must be finite.")
        enforce_sum_to_one_augmented = self.diagnostics.get("enforce_sum_to_one_augmented")
        if enforce_sum_to_one_augmented is True and abs(float(w_aug.sum()) - 1.0) > 1e-6:
            raise ValueError("donor_weights_augmented must sum to 1 (within tolerance).")

        return self

    def summary(self) -> pd.DataFrame:
        """Return a compact tabular summary of key estimate metadata.

        Returns
        -------
        pd.DataFrame
            Two-column dataframe indexed by field name.
        """
        def _fmt_float(val: Optional[float]) -> Optional[str]:
            if val is None:
                return None
            return f"{val:.4f}"

        value_abs = _fmt_float(self.att)
        if self.ci_lower_absolute is not None and self.ci_upper_absolute is not None:
            value_abs = (
                f"{_fmt_float(self.att)} "
                f"(ci_abs: {_fmt_float(self.ci_lower_absolute)}, {_fmt_float(self.ci_upper_absolute)})"
            )

        value_rel = None
        if self.value_relative is not None:
            if self.ci_lower_relative is not None and self.ci_upper_relative is not None:
                value_rel = (
                    f"{_fmt_float(self.value_relative)} "
                    f"(ci_rel: {_fmt_float(self.ci_lower_relative)}, {_fmt_float(self.ci_upper_relative)})"
                )
            else:
                value_rel = _fmt_float(self.value_relative)

        summary = {
            "estimand": self.estimand,
            "model": self.model,
            "value": value_abs,
            "value_relative": value_rel,
            "alpha": _fmt_float(self.alpha),
            "p_value": _fmt_float(self.p_value),
            "is_significant": self.is_significant,
        }
        return pd.DataFrame({"field": list(summary.keys()), "value": list(summary.values())}).set_index("field")
