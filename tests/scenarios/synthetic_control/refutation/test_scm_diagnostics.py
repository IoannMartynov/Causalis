import warnings

import pandas as pd

import causalis.scenarios.synthetic_control.model as sc_model
from causalis.data_contracts import PanelDataSCM, PanelEstimate
from causalis.scenarios.synthetic_control import ASCM, run_scm_diagnostics


def _make_panel_with_effect(effect: float = 2.5) -> pd.DataFrame:
    rows = []
    for idx, t in enumerate(
        ["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01", "2020-05-01", "2020-06-01"],
        start=1,
    ):
        y_c1 = 10.0 + 0.5 * idx
        y_c2 = 12.0 + 0.2 * idx
        y_treat = 0.65 * y_c1 + 0.35 * y_c2
        if idx >= 4:
            y_treat += effect

        rows.extend(
            [
                {"unit_id": "T", "time_id": t, "y": y_treat, "treated_time": 1 if idx >= 4 else 0},
                {"unit_id": "C1", "time_id": t, "y": y_c1, "treated_time": 0},
                {"unit_id": "C2", "time_id": t, "y": y_c2, "treated_time": 0},
            ]
        )
    return pd.DataFrame(rows)


def _panel(df: pd.DataFrame) -> PanelDataSCM:
    return PanelDataSCM(
        unit_col="unit_id",
        time_col="time_id",
        y="y",
        treated_time="treated_time",
        df=df,
    )


def _make_panel_with_flat_pre(effect: float = 2.0) -> pd.DataFrame:
    rows = []
    for idx, t in enumerate(
        ["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01", "2020-05-01", "2020-06-01"],
        start=1,
    ):
        if idx <= 3:
            y_c1 = 100.0
            y_c2 = 100.0
            y_treat = 100.0
        else:
            y_c1 = 100.0 + 0.3 * (idx - 3)
            y_c2 = 100.0 - 0.2 * (idx - 3)
            y_treat = 100.0 + effect

        rows.extend(
            [
                {"unit_id": "T", "time_id": t, "y": y_treat, "treated_time": 1 if idx >= 4 else 0},
                {"unit_id": "C1", "time_id": t, "y": y_c1, "treated_time": 0},
                {"unit_id": "C2", "time_id": t, "y": y_c2, "treated_time": 0},
            ]
        )
    return pd.DataFrame(rows)


def test_scm_fit_warnings_are_suppressed_and_exported_in_run_diagnostics(monkeypatch):
    class _FailedResult:
        success = False
        message = "Positive directional derivative for linesearch"
        x = None

    def _always_fail(*args, **kwargs):
        return _FailedResult()

    def _force_extreme_weight_metrics(self, *, x0_pre, w_aug):
        _ = x0_pre, w_aug
        return 1.0, 10.0, 3.0

    monkeypatch.setattr(sc_model, "minimize", _always_fail)
    monkeypatch.setattr(
        sc_model.AugmentedSyntheticControl,
        "_compute_augmented_weight_metrics",
        _force_extreme_weight_metrics,
    )

    data = _panel(_make_panel_with_effect(effect=3.0))
    model = ASCM(lambda_aug=0.5)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", category=RuntimeWarning)
        estimate = model.fit(data).estimate()

    runtime_messages = [str(item.message) for item in caught if issubclass(item.category, RuntimeWarning)]
    assert not any("SLSQP simplex optimization did not converge" in msg for msg in runtime_messages)
    assert not any("Augmented donor weights are extreme" in msg for msg in runtime_messages)

    report = run_scm_diagnostics(estimate, data)
    assert list(report.columns) == ["test", "flag", "value", "threshold", "message"]
    assert report.shape[0] == 8

    fallback_row = report.loc[report["test"] == "slsqp_fallback_count"].iloc[0]
    assert str(fallback_row["flag"]) == "RED"
    assert float(fallback_row["value"]) >= 1.0
    assert "Positive directional derivative for linesearch" in str(fallback_row["message"])

    warning_row = report.loc[report["test"] == "suppressed_fit_warning_count"].iloc[0]
    assert str(warning_row["flag"]) == "RED"
    warning_message = str(warning_row["message"])
    assert "SLSQP simplex optimization did not converge" in warning_message
    assert "Augmented donor weights are extreme" in warning_message


def test_scm_diagnostics_skips_scale_based_pre_checks_when_pre_is_flat():
    data = _panel(_make_panel_with_flat_pre(effect=2.0))
    model = ASCM(lambda_aug=0.5)
    estimate = model.fit(data).estimate()

    report = run_scm_diagnostics(estimate, data)
    for test_name in (
        "pre_rmse_augmented",
        "max_abs_pre_gap_augmented",
        "mean_gap_last_k_pre_augmented",
    ):
        row = report.loc[report["test"] == test_name].iloc[0]
        assert str(row["flag"]) == "YELLOW"
        assert str(row["threshold"]) == "n/a"
        assert "Skipped: pre-period variability is near zero" in str(row["message"])


