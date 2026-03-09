from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Hashable, List, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from causalis.data_contracts.panel_data_scm import TimeLike


class PanelEstimate(BaseModel):
    """Result contract for dynamic synthetic-control effect-path estimates."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        validate_assignment=True,
    )

    estimand: Literal["dynamic_effect_path"] = "dynamic_effect_path"
    model: str

    treated_unit: Hashable
    treatment_start: TimeLike
    pre_times: List[TimeLike]
    post_times: List[TimeLike]

    effect_by_time: pd.Series
    ci_lower_by_time: pd.Series
    ci_upper_by_time: pd.Series
    p_value_by_time: pd.Series
    is_significant_by_time: pd.Series
    confidence_set_by_time: Dict[TimeLike, list[tuple[float, float]]]

    alpha: float

    observed_outcome: pd.Series
    synthetic_outcome: pd.Series
    donor_weights_augmented: Dict[Hashable, float]

    diagnostics: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def _validate_dimensions(self) -> "PanelEstimate":
        n_pre = len(self.pre_times)
        n_post = len(self.post_times)
        n_all = n_pre + n_post

        if n_pre < 1:
            raise ValueError("pre_times must contain at least one period.")
        if n_post < 1:
            raise ValueError("post_times must contain at least one period.")

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

        for series_name in (
            "effect_by_time",
            "ci_lower_by_time",
            "ci_upper_by_time",
            "p_value_by_time",
            "is_significant_by_time",
        ):
            series = getattr(self, series_name)
            if len(series) != n_post:
                raise ValueError(f"{series_name} length must equal len(post_times).")
            if not series.index.equals(post_idx):
                raise ValueError(f"{series_name} index must exactly equal post_times (same order).")

        if set(self.confidence_set_by_time.keys()) != set(post_idx.tolist()):
            raise ValueError("confidence_set_by_time keys must exactly match post_times.")
        for time_key in post_idx.tolist():
            segments = self.confidence_set_by_time[time_key]
            if not isinstance(segments, list):
                raise ValueError("confidence_set_by_time values must be lists of (lower, upper).")
            for seg in segments:
                if not isinstance(seg, tuple) or len(seg) != 2:
                    raise ValueError("Each confidence set segment must be a 2-tuple (lower, upper).")
                low, high = seg
                try:
                    low_f = float(low)
                    high_f = float(high)
                except (TypeError, ValueError) as exc:
                    raise ValueError("Confidence set segment bounds must be numeric.") from exc
                if not (np.isfinite(low_f) and np.isfinite(high_f)):
                    raise ValueError("Confidence set segment bounds must be finite.")
                if low_f > high_f:
                    raise ValueError("Confidence set segment lower bound must be <= upper bound.")

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

        if len(self.donor_weights_augmented) < 1:
            raise ValueError("At least one donor weight is required.")

        effect_numeric = pd.to_numeric(self.effect_by_time, errors="coerce")
        if effect_numeric.isna().any() or not np.isfinite(effect_numeric.to_numpy()).all():
            raise ValueError("effect_by_time must contain only finite numeric values.")

        p_numeric = pd.to_numeric(self.p_value_by_time, errors="coerce")
        if p_numeric.isna().any() or not np.isfinite(p_numeric.to_numpy()).all():
            raise ValueError("p_value_by_time must contain only finite numeric values.")
        if ((p_numeric < 0.0) | (p_numeric > 1.0)).any():
            raise ValueError("p_value_by_time values must be in [0, 1].")

        for val in self.is_significant_by_time.to_list():
            if not isinstance(val, (bool, np.bool_)):
                raise ValueError("is_significant_by_time must contain only boolean values.")

        for name in ("observed_outcome", "synthetic_outcome"):
            numeric = pd.to_numeric(getattr(self, name), errors="coerce")
            if numeric.isna().any() or not np.isfinite(numeric.to_numpy()).all():
                raise ValueError(f"{name} must contain only finite numeric values.")

        lower_vals = self.ci_lower_by_time.to_list()
        upper_vals = self.ci_upper_by_time.to_list()
        for idx, time_key in enumerate(post_idx.tolist()):
            low = lower_vals[idx]
            high = upper_vals[idx]
            low_missing = low is None or (isinstance(low, (float, np.floating)) and np.isnan(low))
            high_missing = high is None or (isinstance(high, (float, np.floating)) and np.isnan(high))
            if low_missing != high_missing:
                raise ValueError("ci_lower_by_time and ci_upper_by_time must be paired per period.")
            segments = self.confidence_set_by_time[time_key]
            if len(segments) == 1:
                seg_low, seg_high = segments[0]
                if low_missing or high_missing:
                    raise ValueError(
                        "ci_lower_by_time/ci_upper_by_time must be finite when confidence set is a single interval."
                    )
                low_f = float(low)
                high_f = float(high)
                if not (np.isfinite(low_f) and np.isfinite(high_f)):
                    raise ValueError("CI bounds must be finite when provided.")
                if low_f > high_f:
                    raise ValueError(
                        f"ci_lower_by_time must be <= ci_upper_by_time at post index {idx}."
                    )
                if abs(low_f - float(seg_low)) > 1e-9 or abs(high_f - float(seg_high)) > 1e-9:
                    raise ValueError(
                        "Single-interval CI bounds must equal the confidence_set_by_time segment."
                    )
            else:
                if not (low_missing and high_missing):
                    raise ValueError(
                        "ci_lower_by_time/ci_upper_by_time must be missing when confidence set is empty or disconnected."
                    )

        if not np.isfinite(self.alpha) or not (0.0 < float(self.alpha) < 1.0):
            raise ValueError("alpha must be finite and in (0, 1).")

        w_aug = np.asarray(list(self.donor_weights_augmented.values()), dtype=float)
        if not np.isfinite(w_aug).all():
            raise ValueError("donor_weights_augmented must be finite.")
        enforce_sum_to_one_augmented = self.diagnostics.get("enforce_sum_to_one_augmented")
        if enforce_sum_to_one_augmented is True and abs(float(w_aug.sum()) - 1.0) > 1e-6:
            raise ValueError("donor_weights_augmented must sum to 1 (within tolerance).")

        return self

    @staticmethod
    def _fmt_float(value: Any) -> str | None:
        if value is None:
            return None
        try:
            out = float(value)
        except (TypeError, ValueError):
            return None
        if np.isnan(out):
            return None
        return f"{out:.4f}"

    @staticmethod
    def _float_or_none(value: Any) -> float | None:
        if value is None:
            return None
        try:
            out = float(value)
        except (TypeError, ValueError):
            return None
        if np.isnan(out):
            return None
        return out

    def summary(self) -> pd.DataFrame:
        """Return a compact CausalEstimate-style summary table."""

        post_idx = pd.Index(list(self.post_times))
        n_post = int(len(post_idx))
        n_sig = int(np.sum(self.is_significant_by_time.astype(bool).to_numpy()))

        avg_effect = float(np.mean(pd.to_numeric(self.effect_by_time, errors="coerce").to_numpy(dtype=float)))
        cumulative_effect = float(np.sum(pd.to_numeric(self.effect_by_time, errors="coerce").to_numpy(dtype=float)))
        observed_post_mean = float(np.mean(pd.to_numeric(self.observed_outcome.loc[post_idx], errors="coerce")))
        synthetic_post_mean = float(np.mean(pd.to_numeric(self.synthetic_outcome.loc[post_idx], errors="coerce")))

        att_available = bool(self.diagnostics.get("average_att_ttest_available", False))
        att_estimate = self._float_or_none(self.diagnostics.get("average_att_estimate"))
        att_ci_low = self._float_or_none(self.diagnostics.get("average_att_ci_lower"))
        att_ci_high = self._float_or_none(self.diagnostics.get("average_att_ci_upper"))
        att_p_value = self._float_or_none(self.diagnostics.get("average_att_p_value"))

        estimand = self.estimand
        value = f"{self._fmt_float(avg_effect)} (post_period_average)"
        effect_for_relative = avg_effect
        ci_low_for_relative = None
        ci_high_for_relative = None
        p_value = None
        is_significant: bool | None = n_sig > 0
        if att_available and att_estimate is not None:
            estimand = "average_post_effect"
            effect_for_relative = att_estimate
            ci_low_for_relative = att_ci_low
            ci_high_for_relative = att_ci_high
            if att_ci_low is not None and att_ci_high is not None:
                value = (
                    f"{self._fmt_float(att_estimate)} "
                    f"(ci_abs: {self._fmt_float(att_ci_low)}, {self._fmt_float(att_ci_high)})"
                )
            else:
                value = self._fmt_float(att_estimate)
            p_value = self._fmt_float(att_p_value)
            is_significant = bool(att_p_value < float(self.alpha)) if att_p_value is not None else None

        value_relative = None
        control_eps = 1e-12 * max(1.0, abs(synthetic_post_mean))
        if np.isfinite(synthetic_post_mean) and abs(synthetic_post_mean) >= control_eps:
            rel = 100.0 * effect_for_relative / synthetic_post_mean
            if ci_low_for_relative is not None and ci_high_for_relative is not None:
                rel_low = 100.0 * ci_low_for_relative / synthetic_post_mean
                rel_high = 100.0 * ci_high_for_relative / synthetic_post_mean
                if rel_low > rel_high:
                    rel_low, rel_high = rel_high, rel_low
                value_relative = (
                    f"{self._fmt_float(rel)} "
                    f"(ci_rel: {self._fmt_float(rel_low)}, {self._fmt_float(rel_high)})"
                )
            else:
                value_relative = self._fmt_float(rel)

        summary = {
            "estimand": estimand,
            "model": self.model,
            "inference": self.diagnostics.get(
                "pointwise_ci_method",
                "cwz_overlapping_moving_block",
            ),
            "value": value,
            "value_relative": value_relative,
            "alpha": self._fmt_float(self.alpha),
            "p_value": p_value,
            "is_significant": is_significant,
            "post_outcome_d_mean": self._fmt_float(observed_post_mean),
            "pointwise_post_period_average": self._fmt_float(avg_effect),
            "effect_by_time": [
                {
                    "period": time_key,
                    "estimate": float(self.effect_by_time.loc[time_key]),
                }
                for time_key in post_idx.tolist()
            ],
            "cumulative_effect": self._fmt_float(cumulative_effect),
            "n_significant_periods": n_sig,
            "n_donors": int(len(self.donor_weights_augmented)),
            "n_pre_periods": int(len(self.pre_times)),
            "n_post_periods": n_post,
            "time": self.created_at.strftime("%Y-%m-%d")
        }
        return pd.DataFrame({"field": list(summary.keys()), "value": list(summary.values())}).set_index("field")

    def summary_poinwise(self) -> pd.DataFrame:
        """Return pointwise post-period estimates as a flat DataFrame."""

        post_times = list(self.post_times)
        rows: list[dict[str, Any]] = []
        for time_key in post_times:
            rows.append(
                {
                    "time": time_key,
                    "effect": float(self.effect_by_time.loc[time_key]),
                    "ci_lower": self._float_or_none(self.ci_lower_by_time.loc[time_key]),
                    "ci_upper": self._float_or_none(self.ci_upper_by_time.loc[time_key]),
                    "p_value": float(self.p_value_by_time.loc[time_key]),
                    "is_significant": bool(self.is_significant_by_time.loc[time_key]),
                }
            )
        return pd.DataFrame(rows)
