import numpy as np
import pandas as pd
import pytest

from causalis.data_contracts import CallawaySantAnnaDIDEstimate, PanelDataDID
from causalis.scenarios.did import CallawaySantAnnaDID, generate_did_gamma_26


def _staggered_panel(*, include_never: bool = True, cluster_col: bool = False) -> PanelDataDID:
    periods = pd.period_range("2020-01", periods=4, freq="M")
    units = {
        "A1": ("2020-02", 2.0, 10.0, "A"),
        "A2": ("2020-02", 2.0, 12.0, "A"),
        "B1": ("2020-03", 3.0, 20.0, "B"),
        "B2": ("2020-03", 3.0, 22.0, "B"),
    }
    if include_never:
        units.update(
            {
                "C1": (None, 0.0, 30.0, "C"),
                "C2": (None, 0.0, 32.0, "C"),
            }
        )

    rows = []
    for unit, (cohort, tau, base, cluster) in units.items():
        cohort_period = None if cohort is None else pd.Period(cohort, freq="M")
        for idx, period in enumerate(periods):
            treated = cohort_period is not None and period >= cohort_period
            row = {
                "unit": unit,
                "time": period,
                "y": base + idx + (tau if treated else 0.0),
                "d": int(treated),
            }
            if cluster_col:
                row["cluster"] = cluster
            rows.append(row)

    kwargs = {
        "df": pd.DataFrame(rows),
        "y": "y",
        "unit_col": "unit",
        "time_col": "time",
        "treated_time": "d",
    }
    if cluster_col:
        kwargs["cluster_col"] = "cluster"
    return PanelDataDID(**kwargs)


def test_dr_att_gt_and_aggregates_match_manual_staggered_did():
    panel = _staggered_panel()

    estimate = CallawaySantAnnaDID(control_group="never_treated").fit(panel).estimate()

    assert isinstance(estimate, CallawaySantAnnaDIDEstimate)
    assert estimate.model == "CallawaySantAnnaStaggeredDID"
    assert estimate.att_gt["att"].tolist() == pytest.approx([2.0, 2.0, 2.0, 3.0, 3.0])
    assert estimate.att == pytest.approx(2.4)
    assert estimate.value == pytest.approx(2.4)

    event = estimate.event_study()
    assert event["event_time"].tolist() == [0, 1, 2]
    assert event["estimate"].tolist() == pytest.approx([2.5, 2.5, 2.0])

    cohort = estimate.aggregate("cohort")
    assert cohort["estimate"].tolist() == pytest.approx([2.0, 3.0])
    assert estimate.summary().loc["model", "value"] == "CallawaySantAnnaStaggeredDID"


def test_not_yet_treated_controls_work_without_never_treated_units():
    panel = _staggered_panel(include_never=False)

    estimate = CallawaySantAnnaDID(control_group="not_yet_treated").fit(panel).estimate()

    assert estimate.att_gt[["cohort", "time"]].astype(str).to_numpy().tolist() == [
        ["2020-02", "2020-02"],
    ]
    assert estimate.att == pytest.approx(2.0)


def test_never_treated_control_group_requires_never_treated_support():
    panel = _staggered_panel(include_never=False)

    with pytest.raises(ValueError, match="No estimable Callaway-Sant'Anna ATT"):
        CallawaySantAnnaDID(control_group="never_treated").fit(panel)


def test_ipw_and_diagnostics_flag_are_supported_and_or_is_disabled():
    panel = _staggered_panel()

    estimate_ipw = CallawaySantAnnaDID(estimator="ipw", control_group="never_treated").fit(panel).estimate()

    assert estimate_ipw.att == pytest.approx(2.4)
    assert {
        "estimand",
        "ci_alpha",
        "diagnostic_data_requested",
        "estimator",
        "control_group",
        "anticipation",
        "base_period",
        "include_pre_periods",
        "inference",
        "bootstrap_replications",
        "cluster_col",
        "n_units",
        "n_att_gt_cells",
        "n_post_treatment_att_gt_cells",
        "n_pre_period_att_gt_cells",
        "n_skipped_cells",
        "min_treated_per_cell",
        "min_control_per_cell",
        "min_control_ess",
        "max_propensity_clip_share",
        "max_condition_number",
        "support",
        "skipped_cells",
        "weights",
        "unit_level",
        "overlap",
        "balance",
        "influence_scores",
        "cell_diagnostics",
    } == set(estimate_ipw.diagnostics)
    assert estimate_ipw.diagnostic_data is estimate_ipw.diagnostics
    with pytest.raises(ValueError, match="OR point estimator is disabled"):
        CallawaySantAnnaDID(estimator="or")


def test_clustered_inference_and_estimate_overrides_work():
    panel = _staggered_panel(cluster_col=True)
    model = CallawaySantAnnaDID(control_group="never_treated", alpha=0.2, diagnostic_data=True).fit(panel)

    estimate = model.estimate(alpha=0.1, diagnostic_data=False)

    assert model.is_fitted is True
    assert estimate.alpha == pytest.approx(0.1)
    assert estimate.inference == "clustered_influence"
    assert np.isfinite(estimate.se)
    assert estimate.diagnostic_data is None
    assert "unit_level" not in estimate.diagnostics
    assert {"support", "skipped_cells", "cell_diagnostics", "weights"}.issubset(estimate.diagnostics)
    assert model.support_["is_supported"].all()
    assert model.skipped_cells_.empty
    assert set(model.cell_diagnostics_["diagnostic_status"]).issubset({"green", "yellow", "red"})


def test_pre_period_support_and_event_study_are_explicit():
    panel = _staggered_panel()

    estimate = CallawaySantAnnaDID(
        control_group="never_treated",
        include_pre_periods=True,
        base_period="varying",
    ).fit(panel).estimate(diagnostic_data=False)

    assert estimate.include_pre_periods is True
    assert estimate.base_period == "varying"
    assert estimate.att_gt["event_time"].min() < 0
    assert estimate.event_study()["event_time"].min() < 0
    assert estimate.aggregate("simple")["estimate"].iloc[0] == pytest.approx(2.4)


def test_generated_staggered_gamma_panel_fits_csa_model():
    panel = generate_did_gamma_26(
        seed=19,
        n_treated_units=8,
        n_control_units=12,
        n_pre_periods=6,
        n_post_periods=4,
        n_cohorts=4,
    )

    estimate = CallawaySantAnnaDID().fit(panel).estimate(diagnostic_data=False)

    assert panel.design_type == "staggered_adoption"
    assert len(panel.cohorts) == 4
    assert not estimate.att_gt.empty
    assert np.isfinite(estimate.att)
    assert estimate.event_study()["event_time"].min() == 0


def test_invalid_inputs_and_estimate_before_fit_are_rejected():
    with pytest.raises(ValueError, match="PanelDataDID"):
        CallawaySantAnnaDID().fit(pd.DataFrame())

    with pytest.raises(ValueError, match="estimator"):
        CallawaySantAnnaDID(estimator="bad")

    with pytest.raises(ValueError, match="control_group"):
        CallawaySantAnnaDID(control_group="bad")

    with pytest.raises(RuntimeError, match="fit"):
        CallawaySantAnnaDID().estimate()
