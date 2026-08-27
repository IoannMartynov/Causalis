import numpy as np
import pytest

from causalis.scenarios.unconfoundedness.refutation.overlap.overlap_validation import (
    _common_support_from_sorted_propensity,
    _grade_support_rate,
    run_overlap_diagnostics,
)
from tests.refutation._overlap_test_utils import make_overlap_data_and_estimate


def _expit(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return 1.0 / (1.0 + np.exp(-values))


def _make_report_from_logits(
    treated_logits: np.ndarray,
    control_logits: np.ndarray,
    *,
    score: str,
    return_summary: bool = True,
):
    treated_logits = np.asarray(treated_logits, dtype=float)
    control_logits = np.asarray(control_logits, dtype=float)
    propensity = np.concatenate([_expit(treated_logits), _expit(control_logits)])
    treatment = np.concatenate(
        [
            np.ones(treated_logits.size, dtype=int),
            np.zeros(control_logits.size, dtype=int),
        ]
    )
    data, estimate = make_overlap_data_and_estimate(
        m_hat=propensity,
        d=treatment,
    )
    diagnostic_data = estimate.diagnostic_data.model_copy(update={"score": score})
    estimate = estimate.model_copy(
        update={"estimand": score, "diagnostic_data": diagnostic_data}
    )
    return run_overlap_diagnostics(
        data,
        estimate,
        return_summary=return_summary,
    )


def test_support_rate_matches_direct_caliper_definition_for_ate_and_atte():
    treated = np.array([-1.0, 0.0, 1.0, 2.0])
    control = np.array([-0.95, 0.05, 1.05, 1.95, 2.05, 4.0])

    pooled_sd = np.sqrt(
        (
            (treated.size - 1) * np.var(treated, ddof=1)
            + (control.size - 1) * np.var(control, ddof=1)
        )
        / (treated.size + control.size - 2)
    )
    expected_caliper = 0.2 * pooled_sd
    expected_treated = float(
        np.mean(np.min(np.abs(treated[:, None] - control), axis=1) <= expected_caliper)
    )
    expected_control = float(
        np.mean(np.min(np.abs(control[:, None] - treated), axis=1) <= expected_caliper)
    )

    ate = _make_report_from_logits(treated, control, score="ATE")
    atte = _make_report_from_logits(treated, control, score="ATTE")

    assert expected_treated == 1.0
    assert expected_control == pytest.approx(5.0 / 6.0)
    assert ate["support_caliper"] == pytest.approx(expected_caliper)
    assert ate["support_rate_treated"] == expected_treated
    assert ate["support_rate_control"] == pytest.approx(expected_control)
    assert ate["support_rate"] == pytest.approx(expected_control)
    assert ate["flags"]["support_rate"] == "RED"
    assert atte["support_rate_treated"] == expected_treated
    assert atte["support_rate_control"] == pytest.approx(expected_control)
    assert atte["support_rate"] == expected_treated
    assert atte["flags"]["support_rate"] == "GREEN"

    summary_row = ate["summary"].loc[ate["summary"]["metric"] == "support_rate"]
    assert summary_row.shape[0] == 1
    assert float(summary_row["value"].iloc[0]) == pytest.approx(expected_control)
    assert summary_row["flag"].iloc[0] == "RED"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1.0, "GREEN"),
        (0.95, "GREEN"),
        (0.949999, "YELLOW"),
        (0.90, "YELLOW"),
        (0.899999, "RED"),
        (0.0, "RED"),
    ],
)
def test_support_rate_status_boundaries(value: float, expected: str):
    assert (
        _grade_support_rate(
            value,
            warn_threshold=0.95,
            strong_threshold=0.90,
        )
        == expected
    )


def test_support_rate_clips_extreme_propensities_with_fixed_epsilon():
    propensity = np.array([0.0, 0.5, 1.0, 0.0, 0.5, 1.0])
    treatment = np.array([1, 1, 1, 0, 0, 0])
    data, estimate = make_overlap_data_and_estimate(
        m_hat=propensity,
        d=treatment,
    )

    report = run_overlap_diagnostics(data, estimate)

    clipped = np.clip(np.array([0.0, 0.5, 1.0]), 1e-8, 1.0 - 1e-8)
    logits = np.log(clipped / (1.0 - clipped))
    expected_caliper = 0.2 * np.std(logits, ddof=1)
    assert report["support_caliper"] == pytest.approx(expected_caliper)
    assert report["support_rate_treated"] == 1.0
    assert report["support_rate_control"] == 1.0
    assert report["meta"]["support_epsilon"] == 1e-8


def test_support_rate_zero_pooled_sd_accepts_exact_matches():
    propensity = np.full(8, 0.5)
    treatment = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    data, estimate = make_overlap_data_and_estimate(
        m_hat=propensity,
        d=treatment,
    )

    report = run_overlap_diagnostics(data, estimate)

    assert report["support_caliper"] == 0.0
    assert report["support_rate_treated"] == 1.0
    assert report["support_rate_control"] == 1.0
    assert report["support_rate"] == 1.0
    assert report["flags"]["support_rate"] == "GREEN"


def test_support_rate_prefers_raw_propensity_and_falls_back_to_estimate_estimand():
    treated = np.array([-1.0, 0.0, 1.0, 2.0])
    control = np.array([-0.95, 0.05, 1.05, 1.95, 2.05, 4.0])
    raw_propensity = np.concatenate([_expit(treated), _expit(control)])
    treatment = np.concatenate(
        [np.ones(treated.size, dtype=int), np.zeros(control.size, dtype=int)]
    )
    data, estimate = make_overlap_data_and_estimate(
        m_hat=np.full(treatment.size, 0.5),
        d=treatment,
    )
    diagnostic_data = estimate.diagnostic_data.model_copy(
        update={"m_hat_raw": raw_propensity, "score": None}
    )
    estimate = estimate.model_copy(
        update={"estimand": "ATTE", "diagnostic_data": diagnostic_data}
    )

    report = run_overlap_diagnostics(data, estimate, return_summary=False)

    assert report["meta"]["propensity_source"] == "m_hat_raw"
    assert report["meta"]["score"] == "ATTE"
    assert report["support_rate_treated"] == 1.0
    assert report["support_rate_control"] == pytest.approx(5.0 / 6.0)
    assert report["support_rate"] == 1.0
    assert "summary" not in report


def test_support_rate_prefers_diagnostic_score_over_estimate_estimand():
    treated = np.array([-1.0, 0.0, 1.0, 2.0])
    control = np.array([-0.95, 0.05, 1.05, 1.95, 2.05, 4.0])
    propensity = np.concatenate([_expit(treated), _expit(control)])
    treatment = np.concatenate(
        [np.ones(treated.size, dtype=int), np.zeros(control.size, dtype=int)]
    )
    data, estimate = make_overlap_data_and_estimate(
        m_hat=propensity,
        d=treatment,
    )
    diagnostic_data = estimate.diagnostic_data.model_copy(update={"score": "ATE"})
    estimate = estimate.model_copy(
        update={"estimand": "ATTE", "diagnostic_data": diagnostic_data}
    )

    report = run_overlap_diagnostics(data, estimate)

    assert report["meta"]["score"] == "ATE"
    assert report["support_rate"] == pytest.approx(5.0 / 6.0)


def test_support_rate_rejects_groups_too_small_for_pooled_sd():
    propensity = np.array([0.4, 0.5, 0.6, 0.7])
    treatment = np.array([1, 0, 0, 0])
    data, estimate = make_overlap_data_and_estimate(
        m_hat=propensity,
        d=treatment,
    )

    with pytest.raises(ValueError, match="at least two treated and two control"):
        run_overlap_diagnostics(data, estimate)


def test_support_rate_large_sample_smoke_has_no_pairwise_distance_matrix():
    n_per_group = 25_000
    treated = np.linspace(0.05, 0.95, n_per_group, dtype=float)
    control = np.linspace(0.05001, 0.95001, n_per_group, dtype=float)

    result = _common_support_from_sorted_propensity(
        treated,
        control,
        score="ATE",
    )

    assert all(np.isfinite(value) for value in result)
    assert 0.0 <= result[0] <= 1.0
    assert 0.0 <= result[1] <= 1.0
    assert 0.0 <= result[2] <= 1.0
    assert result[3] >= 0.0
