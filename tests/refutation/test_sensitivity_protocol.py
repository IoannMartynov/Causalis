import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression

from causalis.dgp.causaldata import CausalData
from causalis.scenarios.unconfoundedness.model import IRM
from causalis.scenarios.unconfoundedness.refutation import run_sensitivity_protocol
from causalis.scenarios.unconfoundedness.refutation.unconfoundedness import sensitivity as sensitivity_module


class _DummyIRM:
    def __init__(self, *, theta: float, se: float) -> None:
        self.coef_ = np.array([theta], dtype=float)
        self.se_ = np.array([se], dtype=float)
        self.n_rep = 1

    def _sensitivity_element_est(self) -> dict:
        return {"sigma2": 1.0, "nu2": 1.0}


def _install_benchmarks(monkeypatch, payload: dict[tuple[str, ...], dict]) -> None:
    def _fake_benchmark(effect_estimation, data, benchmarking_group, fit_args=None):
        del effect_estimation, data, fit_args
        group = tuple(benchmarking_group)
        values = payload[group]
        return pd.DataFrame(
            [
                {
                    "benchmark_group": group,
                    "r2_y": values["r2_y"],
                    "r2_d": values["r2_d"],
                    "rho": values["rho"],
                    "theta_long": values.get("theta_long", 0.3),
                    "theta_short": values.get("theta_short", 0.31),
                    "delta": values.get("delta", -0.01),
                    "rho_raw": values.get("rho_raw", values["rho"]),
                    "rho_clipped": values.get("rho_clipped", False),
                    "cf_y_clipped": values.get("cf_y_clipped", False),
                    "cf_d_clipped": values.get("cf_d_clipped", False),
                    "rho_fallback": values.get("rho_fallback", False),
                    "boundary_calibration": values.get("boundary_calibration", False),
                    "strengths_valid": values.get("strengths_valid", True),
                    "calibration_valid": values.get("calibration_valid", True),
                    "calibration_issue": values.get("calibration_issue"),
                    "calibration_warning": values.get("calibration_warning"),
                }
            ]
        )

    monkeypatch.setattr(sensitivity_module, "sensitivity_benchmark_group", _fake_benchmark)


def test_protocol_passes_primary_and_reports_limited_stress_margin(monkeypatch):
    _install_benchmarks(
        monkeypatch,
        {
            ("x1", "x2"): {
                "r2_y": 0.1,
                "r2_d": 0.1,
                "rho": 0.2,
            }
        },
    )
    model = _DummyIRM(theta=0.3, se=0.1)

    report = run_sensitivity_protocol(
        model,
        object(),  # type: ignore[arg-type]
        benchmark_groups={"engagement": ["x1", "x2"]},
        decision_threshold=0.0,
        direction="positive",
    )

    assert report["status"] == "PASS"
    assert report["passed"] is True
    assert bool(report["primary"].loc[0, "passed"]) is True
    assert bool(report["adversarial"].loc[0, "passed"]) is False
    assert report["primary"].loc[0, "rho"] == pytest.approx(0.2)
    assert report["stress"].loc[0, "rho"] == pytest.approx(0.2)
    assert report["adversarial"].loc[0, "rho"] == pytest.approx(1.0)
    assert report["limited_margin"] is True
    assert "limited" in report["summary"]
    assert any("n_rep=1" in warning for warning in report["warnings"])


def test_protocol_fails_if_any_primary_benchmark_crosses_threshold(monkeypatch):
    _install_benchmarks(
        monkeypatch,
        {
            ("x1", "x2"): {"r2_y": 0.01, "r2_d": 0.01, "rho": 0.1},
            ("x3", "x4"): {"r2_y": 0.1, "r2_d": 0.1, "rho": 1.0},
        },
    )
    model = _DummyIRM(theta=0.3, se=0.1)

    report = run_sensitivity_protocol(
        model,
        object(),  # type: ignore[arg-type]
        benchmark_groups={"weak": ["x1", "x2"], "strong": ["x3", "x4"]},
        decision_threshold=0.0,
        direction="positive",
    )

    assert report["status"] == "FAIL"
    assert report["primary"]["passed"].tolist() == [True, False]
    assert "strong" in report["summary"]


def test_protocol_scales_two_times_scenario_on_odds_scale(monkeypatch):
    _install_benchmarks(
        monkeypatch,
        {("x1", "x2"): {"r2_y": 0.2, "r2_d": 0.4, "rho": 0.5}},
    )
    model = _DummyIRM(theta=5.0, se=0.1)

    report = run_sensitivity_protocol(
        model,
        object(),  # type: ignore[arg-type]
        benchmark_groups={"drivers": ["x1", "x2"]},
        decision_threshold=0.0,
        direction="positive",
    )

    stress = report["stress"].iloc[0]
    assert stress["r2_y"] == pytest.approx(1.0 / 3.0)
    assert stress["r2_d"] == pytest.approx(4.0 / 7.0)
    assert stress["multiplier"] == 2.0


def test_protocol_fails_closed_for_uncalibratable_primary(monkeypatch):
    _install_benchmarks(
        monkeypatch,
        {
            ("x1",): {
                "r2_y": np.nan,
                "r2_d": np.nan,
                "rho": np.nan,
                "strengths_valid": False,
                "calibration_valid": False,
                "calibration_issue": "nu2 gain is non-positive",
            }
        },
    )

    report = run_sensitivity_protocol(
        _DummyIRM(theta=0.3, se=0.1),
        object(),  # type: ignore[arg-type]
        benchmark_groups={"invalid": ["x1"]},
        decision_threshold=0.0,
        direction="positive",
    )

    assert report["status"] == "FAIL"
    assert bool(report["primary"].loc[0, "scenario_valid"]) is False
    assert bool(report["primary"].loc[0, "passed"]) is False
    assert bool(report["stress"].loc[0, "scenario_valid"]) is False
    assert bool(report["adversarial"].loc[0, "scenario_valid"]) is False
    assert report["primary"].loc[0, "scenario_issue"] == "nu2 gain is non-positive"
    assert "could not be calibrated" in report["summary"]
    assert any("not calibratable" in warning for warning in report["warnings"])


def test_protocol_keeps_adversarial_when_only_rho_is_invalid(monkeypatch):
    _install_benchmarks(
        monkeypatch,
        {
            ("x1",): {
                "r2_y": 0.02,
                "r2_d": 0.03,
                "rho": np.nan,
                "strengths_valid": True,
                "calibration_valid": False,
                "calibration_issue": "rho denominator is numerically zero",
            }
        },
    )

    report = run_sensitivity_protocol(
        _DummyIRM(theta=0.3, se=0.1),
        object(),  # type: ignore[arg-type]
        benchmark_groups={"rho_invalid": ["x1"]},
        decision_threshold=0.0,
        direction="positive",
    )

    assert report["status"] == "FAIL"
    assert bool(report["primary"].loc[0, "scenario_valid"]) is False
    assert bool(report["stress"].loc[0, "scenario_valid"]) is False
    assert bool(report["adversarial"].loc[0, "scenario_valid"]) is True
    assert report["adversarial"].loc[0, "rho"] == pytest.approx(1.0)


def test_protocol_evaluates_doubleml_boundary_benchmark(monkeypatch):
    _install_benchmarks(
        monkeypatch,
        {
            ("x1",): {
                "r2_y": 0.05,
                "r2_d": 0.0,
                "rho": 1.0,
                "rho_raw": np.nan,
                "cf_d_clipped": True,
                "rho_fallback": True,
                "boundary_calibration": True,
                "calibration_warning": (
                    "cf_d_raw=-0.02 was clipped to 0; "
                    "rho used the DoubleML boundary fallback"
                ),
            }
        },
    )

    report = run_sensitivity_protocol(
        _DummyIRM(theta=0.3, se=0.1),
        object(),  # type: ignore[arg-type]
        benchmark_groups={"boundary": ["x1"]},
        decision_threshold=0.0,
        direction="positive",
    )

    assert report["status"] == "PASS"
    assert bool(report["primary"].loc[0, "scenario_valid"]) is True
    assert bool(report["primary"].loc[0, "passed"]) is True
    assert report["stress"].loc[0, "r2_d"] == 0.0
    assert report["adversarial"].loc[0, "rho"] == 1.0
    assert report["boundary_benchmarks"] == ["boundary"]
    assert "boundary calibration" in report["summary"]
    assert any("clipped to 0" in warning for warning in report["warnings"])


def test_protocol_warns_when_calibrated_rho_is_clipped(monkeypatch):
    _install_benchmarks(
        monkeypatch,
        {
            ("x1",): {
                "r2_y": 0.01,
                "r2_d": 0.01,
                "rho": -1.0,
                "rho_raw": -1.4,
                "rho_clipped": True,
            }
        },
    )

    report = run_sensitivity_protocol(
        _DummyIRM(theta=3.0, se=0.1),
        object(),  # type: ignore[arg-type]
        benchmark_groups={"clipped": ["x1"]},
        decision_threshold=0.0,
        direction="positive",
    )

    assert report["primary"].loc[0, "rho"] == -1.0
    assert report["adversarial"].loc[0, "rho"] == 1.0
    assert any("clipped to +/-1" in warning for warning in report["warnings"])


def test_protocol_supports_negative_claims(monkeypatch):
    _install_benchmarks(
        monkeypatch,
        {("x1", "x2"): {"r2_y": 0.01, "r2_d": 0.01, "rho": 0.2}},
    )
    model = _DummyIRM(theta=-1.0, se=0.1)

    report = run_sensitivity_protocol(
        model,
        object(),  # type: ignore[arg-type]
        benchmark_groups={"cost": ["x1", "x2"]},
        decision_threshold=-0.5,
        direction="negative",
    )

    assert report["status"] == "PASS"
    assert report["primary"].loc[0, "ci_upper"] < -0.5


def test_protocol_empty_benchmarks_and_failed_preconditions_return_fail(monkeypatch):
    model = _DummyIRM(theta=5.0, se=0.1)
    empty = run_sensitivity_protocol(
        model,
        object(),  # type: ignore[arg-type]
        benchmark_groups={},
        decision_threshold=0.0,
        direction="positive",
    )
    assert empty["status"] == "FAIL"
    assert empty["primary"].empty

    _install_benchmarks(
        monkeypatch,
        {("x1", "x2"): {"r2_y": 0.01, "r2_d": 0.01, "rho": 0.1}},
    )
    failed = run_sensitivity_protocol(
        model,
        object(),  # type: ignore[arg-type]
        benchmark_groups={"drivers": ["x1", "x2"]},
        decision_threshold=0.0,
        direction="positive",
        preconditions_passed=False,
    )
    assert failed["status"] == "FAIL"
    assert bool(failed["primary"].loc[0, "passed"]) is True
    assert "preconditions" in failed["summary"]


@pytest.mark.parametrize("overlap_policy", ["clip", "drop"])
def test_protocol_runs_end_to_end_with_irm(overlap_policy):
    rng = np.random.default_rng(42)
    n = 240
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x3 = rng.normal(size=n)
    propensity = 1.0 / (1.0 + np.exp(-(0.6 * x1 + 0.3 * x2)))
    d = rng.binomial(1, propensity)
    y = 1.5 * d + 0.8 * x1 + 0.2 * x2 + rng.normal(scale=0.7, size=n)
    data = CausalData(
        df=pd.DataFrame({"y": y, "d": d, "x1": x1, "x2": x2, "x3": x3}),
        treatment="d",
        outcome="y",
        confounders=["x1", "x2", "x3"],
    )
    model = IRM(
        data=data,
        ml_g=RandomForestRegressor(n_estimators=20, random_state=1),
        ml_m=LogisticRegression(max_iter=1000),
        n_folds=3,
        random_state=1,
        overlap_policy=overlap_policy,
        overlap_threshold=0.15,
    ).fit()
    model.estimate(score="ATE")

    report = run_sensitivity_protocol(
        model,
        data,
        benchmark_groups={"joint": ["x1", "x2"]},
        decision_threshold=0.0,
        direction="positive",
    )

    assert report["status"] in {"PASS", "FAIL"}
    assert report["benchmarks"]["benchmark_group"].tolist() == [("x1", "x2")]
    assert report["scenarios"]["scenario"].tolist() == ["primary", "2x", "adversarial"]
    valid = report["scenarios"]["scenario_valid"].astype(bool)
    assert np.isfinite(
        report["scenarios"].loc[valid, ["ci_lower", "ci_upper"]]
    ).all().all()
    assert report["scenarios"].loc[~valid, "scenario_issue"].notna().all()
    assert len(model._full_sample_folds_) == len(data.df)
    if overlap_policy == "drop":
        assert model.overlap_n_dropped_ > 0
        assert len(model.folds_) < len(model._full_sample_folds_)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"direction": "up"}, ValueError),
        ({"stress_multiplier": 1.0}, ValueError),
        ({"alpha": 1.0}, ValueError),
        ({"preconditions_passed": "yes"}, TypeError),
    ],
)
def test_protocol_validates_policy_inputs(kwargs, error):
    call = {
        "benchmark_groups": {},
        "decision_threshold": 0.0,
        "direction": "positive",
    }
    call.update(kwargs)
    with pytest.raises(error):
        run_sensitivity_protocol(
            _DummyIRM(theta=1.0, se=0.1),
            object(),  # type: ignore[arg-type]
            **call,
        )
