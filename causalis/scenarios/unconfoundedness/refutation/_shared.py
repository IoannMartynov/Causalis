"""Private helpers shared across unconfoundedness refutation modules."""

from __future__ import annotations

from typing import Any

from causalis.data_contracts.causal_estimate import CausalEstimate
from causalis.dgp.causaldata import CausalData
from causalis.scenarios.unconfoundedness._score_utils import _normalize_ate_atte_score


def _normalize_score(score: Any) -> str:
    """Normalize supported score aliases to ``ATE`` or ``ATTE``."""
    return _normalize_ate_atte_score(score)


def _validate_estimate_matches_data(
    data: CausalData,
    estimate: CausalEstimate,
    *,
    require_confounders: bool = False,
) -> None:
    """Ensure an estimate is aligned with the supplied causal dataset."""
    if str(estimate.treatment) != str(data.treatment_name):
        raise ValueError(
            "estimate.treatment must match data.treatment_name "
            f"({estimate.treatment!r} != {data.treatment_name!r})."
        )

    if str(estimate.outcome) != str(data.outcome_name):
        raise ValueError(
            "estimate.outcome must match data.outcome_name "
            f"({estimate.outcome!r} != {data.outcome_name!r})."
        )

    if not require_confounders:
        return

    df = data.get_df()
    missing_confounders = [name for name in estimate.confounders if name not in df.columns]
    if missing_confounders:
        raise ValueError(
            "estimate.confounders are missing in data.get_df(): "
            + ", ".join(sorted(map(str, missing_confounders)))
        )
