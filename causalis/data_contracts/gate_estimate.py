from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.stats import norm

from causalis.data_contracts.gate_contrast_estimate import GateContrastEstimate


_SUPPORTED_ALTERNATIVES = {"two-sided", "greater", "less"}
_SUPPORTED_P_ADJUST = {"none", "holm", "bonferroni"}


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
        if not self.summary_table.index.is_unique:
            raise ValueError("summary_table index must be unique.")

        covariance_index = [str(x) for x in self.covariance.index]
        covariance_columns = [str(x) for x in self.covariance.columns]
        if set(covariance_index) != set(self.group_names) or set(covariance_columns) != set(self.group_names):
            raise ValueError("covariance index/columns must match group_names.")
        self.covariance = self.covariance.loc[self.group_names, self.group_names].copy()

        summary_index = [str(x) for x in self.summary_table.index]
        if set(summary_index) != set(self.group_names):
            raise ValueError("summary_table index must match group_names.")
        self.summary_table = self.summary_table.loc[self.group_names].copy()
        self.summary_table["is_significant"] = np.asarray(self.p_values, dtype=float) < float(self.alpha)

        return self

    def summary(self) -> pd.DataFrame:
        """Return per-group GATE summary table."""
        return self.summary_table.copy()

    def contrast(
        self,
        left_group: str,
        right_group: str,
        *,
        alpha: Optional[float] = None,
        alternative: str = "two-sided",
    ) -> GateContrastEstimate:
        """Construct a formal post-estimation contrast between two GATE groups."""
        left_idx = self._resolve_group_position(group_name=left_group)
        right_idx = self._resolve_group_position(group_name=right_group, argument_name="right_group")
        if left_idx == right_idx:
            raise ValueError("left_group and right_group must be different.")

        alpha_resolved = self._resolve_alpha(alpha)
        alternative_resolved = self._resolve_alternative(alternative)
        diff, variance, std_error, test_stat = self._compute_pairwise_stats(left_idx=left_idx, right_idx=right_idx)
        p_value, ci_lower, ci_upper = self._inference_from_test_stat(
            value=diff,
            std_error=std_error,
            test_stat=test_stat,
            alpha=alpha_resolved,
            alternative=alternative_resolved,
        )

        return GateContrastEstimate(
            estimand="GATE_CONTRAST",
            model=self.model,
            model_options=dict(self.model_options),
            left_group=self.group_names[left_idx],
            right_group=self.group_names[right_idx],
            contrast_label=self._contrast_label(self.group_names[left_idx], self.group_names[right_idx]),
            value=float(diff),
            std_error=float(std_error),
            test_stat=float(test_stat),
            p_value=float(p_value),
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            alpha=float(alpha_resolved),
            alternative=alternative_resolved,
            is_significant=bool(np.isfinite(p_value) and p_value < alpha_resolved),
            left_value=float(np.asarray(self.values, dtype=float)[left_idx]),
            right_value=float(np.asarray(self.values, dtype=float)[right_idx]),
            n_left=int(np.asarray(self.n_group, dtype=int)[left_idx]),
            n_right=int(np.asarray(self.n_group, dtype=int)[right_idx]),
        )

    def pairwise_summary(
        self,
        *,
        reference: Optional[str] = None,
        alpha: Optional[float] = None,
        p_adjust: str = "none",
    ) -> pd.DataFrame:
        """Return a long-form table of formal pairwise GATE contrasts."""
        if len(self.group_names) < 2:
            raise ValueError("pairwise_summary requires at least two estimable GATE groups.")

        alpha_resolved = self._resolve_alpha(alpha)
        p_adjust_resolved = self._resolve_p_adjust(p_adjust)

        if reference is None:
            pairs = [
                (self.group_names[i], self.group_names[j])
                for i in range(len(self.group_names))
                for j in range(i + 1, len(self.group_names))
            ]
        else:
            reference_idx = self._resolve_group_position(group_name=reference, argument_name="reference")
            reference_name = self.group_names[reference_idx]
            pairs = [
                (group_name, reference_name)
                for idx, group_name in enumerate(self.group_names)
                if idx != reference_idx
            ]

        rows: list[Dict[str, Any]] = []
        for left_group, right_group in pairs:
            contrast = self.contrast(
                left_group=left_group,
                right_group=right_group,
                alpha=alpha_resolved,
                alternative="two-sided",
            )
            rows.append(
                {
                    "left_group": contrast.left_group,
                    "right_group": contrast.right_group,
                    "contrast_label": contrast.contrast_label,
                    "left_value": contrast.left_value,
                    "right_value": contrast.right_value,
                    "estimate_diff": contrast.value,
                    "std_error": contrast.std_error,
                    "test_stat": contrast.test_stat,
                    "p_value": contrast.p_value,
                    "ci_lower": contrast.ci_lower,
                    "ci_upper": contrast.ci_upper,
                    "n_left": contrast.n_left,
                    "n_right": contrast.n_right,
                }
            )

        pairwise = pd.DataFrame(rows)
        pairwise["p_value_adj"] = self._adjust_pvalues(
            pairwise["p_value"].to_numpy(dtype=float),
            method=p_adjust_resolved,
        )
        pairwise["is_significant"] = pairwise["p_value"] < alpha_resolved
        pairwise["is_significant_adj"] = pairwise["p_value_adj"] < alpha_resolved
        pairwise["alpha"] = float(alpha_resolved)
        pairwise["p_adjust"] = p_adjust_resolved

        return pairwise[
            [
                "left_group",
                "right_group",
                "contrast_label",
                "left_value",
                "right_value",
                "estimate_diff",
                "std_error",
                "test_stat",
                "p_value",
                "p_value_adj",
                "ci_lower",
                "ci_upper",
                "is_significant",
                "is_significant_adj",
                "n_left",
                "n_right",
                "alpha",
                "p_adjust",
            ]
        ].copy()

    def _resolve_group_position(self, group_name: str, argument_name: str = "left_group") -> int:
        if group_name not in self.group_names:
            valid = ", ".join(repr(name) for name in self.group_names)
            raise ValueError(f"Unknown {argument_name} {group_name!r}. Valid groups are: {valid}.")
        return self.group_names.index(group_name)

    def _resolve_alpha(self, alpha: Optional[float]) -> float:
        alpha_resolved = float(self.alpha if alpha is None else alpha)
        if not (0.0 < alpha_resolved < 1.0):
            raise ValueError("alpha must be in (0, 1).")
        return alpha_resolved

    def _resolve_alternative(self, alternative: str) -> str:
        alternative_resolved = str(alternative).lower()
        if alternative_resolved not in _SUPPORTED_ALTERNATIVES:
            supported = ", ".join(sorted(_SUPPORTED_ALTERNATIVES))
            raise ValueError(f"alternative must be one of {{{supported}}}. Got {alternative!r}.")
        return alternative_resolved

    def _resolve_p_adjust(self, p_adjust: str) -> str:
        p_adjust_resolved = str(p_adjust).lower()
        if p_adjust_resolved not in _SUPPORTED_P_ADJUST:
            supported = ", ".join(sorted(_SUPPORTED_P_ADJUST))
            raise ValueError(f"p_adjust must be one of {{{supported}}}. Got {p_adjust!r}.")
        return p_adjust_resolved

    def _compute_pairwise_stats(self, *, left_idx: int, right_idx: int) -> tuple[float, float, float, float]:
        values = np.asarray(self.values, dtype=float).reshape(-1)

        diff = float(values[left_idx] - values[right_idx])
        if self.model_options.get("covariance_structure") == "diagonal":
            variances = np.square(np.asarray(self.std_errors, dtype=float).reshape(-1))
            left_var = variances[left_idx]
            right_var = variances[right_idx]
            variance = float(left_var + right_var) if np.isfinite(left_var) and np.isfinite(right_var) else np.nan
        else:
            covariance = self.covariance.loc[self.group_names, self.group_names].to_numpy(dtype=float)
            contrast_vector = np.zeros(len(self.group_names), dtype=float)
            contrast_vector[left_idx] = 1.0
            contrast_vector[right_idx] = -1.0
            variance = float(contrast_vector @ covariance @ contrast_vector)
        if np.isfinite(variance) and variance < 0.0 and abs(variance) < 1e-12:
            variance = 0.0
        if np.isfinite(variance) and variance < 0.0:
            raise RuntimeError("Stored GATE covariance produced a negative contrast variance.")

        std_error = float(np.sqrt(variance)) if np.isfinite(variance) else np.nan
        with np.errstate(divide="ignore", invalid="ignore"):
            test_stat = float(np.divide(diff, std_error))

        return diff, variance, std_error, test_stat

    def _inference_from_test_stat(
        self,
        *,
        value: float,
        std_error: float,
        test_stat: float,
        alpha: float,
        alternative: str,
    ) -> tuple[float, Optional[float], Optional[float]]:
        if alternative == "two-sided":
            p_value = float(2.0 * norm.sf(np.abs(test_stat)))
            z_crit = float(norm.ppf(1.0 - (alpha / 2.0)))
            ci_lower = float(value - z_crit * std_error) if np.isfinite(std_error) else np.nan
            ci_upper = float(value + z_crit * std_error) if np.isfinite(std_error) else np.nan
            return p_value, ci_lower, ci_upper
        if alternative == "greater":
            return float(norm.sf(test_stat)), None, None
        return float(norm.cdf(test_stat)), None, None

    def _adjust_pvalues(self, p_values: np.ndarray, *, method: str) -> np.ndarray:
        p_values = np.asarray(p_values, dtype=float).reshape(-1)
        adjusted = np.full(p_values.shape, np.nan, dtype=float)
        valid_mask = np.isfinite(p_values)
        valid_p = p_values[valid_mask]

        if valid_p.size == 0:
            return adjusted
        if method == "none":
            adjusted[valid_mask] = valid_p
            return adjusted
        if method == "bonferroni":
            adjusted[valid_mask] = np.minimum(valid_p * valid_p.size, 1.0)
            return adjusted

        order = np.argsort(valid_p)
        sorted_p = valid_p[order]
        m = sorted_p.size
        holm_sorted = np.maximum.accumulate((m - np.arange(m)) * sorted_p)
        holm_sorted = np.minimum(holm_sorted, 1.0)
        holm_adjusted = np.empty_like(sorted_p)
        holm_adjusted[order] = holm_sorted
        adjusted[valid_mask] = holm_adjusted
        return adjusted

    @staticmethod
    def _contrast_label(left_group: str, right_group: str) -> str:
        return f"{left_group} - {right_group}"
