from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import pandas as pd

from causalis.scenarios.cuped.refutation.regression_checks import (
    FLAG_GREEN,
    FLAG_RED,
    FLAG_YELLOW,
)


@dataclass(frozen=True)
class CUPEDRefutationConfig:
    """Configuration for CUPED regression refutation checks and actions."""

    run_regression_checks: bool = True
    check_action: Literal["ignore", "raise"] = "ignore"
    raise_on_yellow: bool = False
    condition_number_warn_threshold: float = 1e8
    corr_near_one_tol: float = 1e-10
    vif_warn_threshold: float = 20.0
    winsor_q: Optional[float] = 0.01
    tiny_one_minus_h_tol: float = 1e-8

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_regression_checks", bool(self.run_regression_checks))
        object.__setattr__(self, "raise_on_yellow", bool(self.raise_on_yellow))
        object.__setattr__(
            self,
            "condition_number_warn_threshold",
            float(self.condition_number_warn_threshold),
        )
        object.__setattr__(self, "corr_near_one_tol", float(self.corr_near_one_tol))
        object.__setattr__(self, "vif_warn_threshold", float(self.vif_warn_threshold))
        object.__setattr__(self, "tiny_one_minus_h_tol", float(self.tiny_one_minus_h_tol))

        if self.check_action not in {"ignore", "raise"}:
            raise ValueError("check_action must be one of {'ignore', 'raise'}.")
        if self.condition_number_warn_threshold <= 0.0:
            raise ValueError("condition_number_warn_threshold must be positive.")
        if self.corr_near_one_tol < 0.0:
            raise ValueError("corr_near_one_tol must be non-negative.")
        if self.vif_warn_threshold <= 0.0:
            raise ValueError("vif_warn_threshold must be positive.")
        if self.tiny_one_minus_h_tol <= 0.0:
            raise ValueError("tiny_one_minus_h_tol must be positive.")

        if self.winsor_q is not None:
            winsor_q = float(self.winsor_q)
            if not (0.0 < winsor_q < 0.5):
                raise ValueError("winsor_q must be in (0, 0.5) when provided.")
            object.__setattr__(self, "winsor_q", winsor_q)

    def signal_assumption_flags(
        self,
        table: pd.DataFrame,
        skip_test_ids: Optional[set[str]] = None,
    ) -> None:
        """Raise according to GREEN/YELLOW/RED assumption table flags."""
        if self.check_action == "ignore":
            return

        skip = set(skip_test_ids or set())
        for _, row in table.iterrows():
            test_id = str(row.get("test_id", ""))
            if test_id in skip:
                continue

            flag = str(row.get("flag", FLAG_GREEN)).upper()
            if flag == FLAG_GREEN:
                continue

            test_name = str(row.get("test", "assumption"))
            msg = str(row.get("message", "diagnostic check failed"))
            text = f"{test_name}: {msg}"

            should_raise = flag == FLAG_RED or (self.raise_on_yellow and flag == FLAG_YELLOW)
            if self.check_action == "raise" and should_raise:
                raise ValueError(text)

    def signal_message(self, msg: str) -> None:
        """Raise a configured refutation exception for a plain message."""
        if self.check_action == "ignore":
            return
        raise ValueError(msg)

    def to_model_options(self) -> dict[str, object]:
        """Return JSON-friendly fields for CausalEstimate.model_options."""
        return {
            "run_regression_checks": self.run_regression_checks,
            "check_action": self.check_action,
            "raise_on_yellow": self.raise_on_yellow,
            "condition_number_warn_threshold": self.condition_number_warn_threshold,
            "corr_near_one_tol": self.corr_near_one_tol,
            "vif_warn_threshold": self.vif_warn_threshold,
            "winsor_q": self.winsor_q,
            "tiny_one_minus_h_tol": self.tiny_one_minus_h_tol,
        }
