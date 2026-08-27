import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm
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


@pytest.mark.parametrize("theta", [0.3, -0.3])
def test_protocol_fails_if_any_primary_benchmark_crosses_threshold(monkeypatch, theta):
    _install_benchmarks(
        monkeypatch,
        {
            ("x1", "x2"): {"r2_y": 0.01, "r2_d": 0.01, "rho": 0.1},
            ("x3", "x4"): {"r2_y": 0.1, "r2_d": 0.1, "rho": 1.0},
        },
    )
    model = _DummyIRM(theta=theta, se=0.1)

    report = run_sensitivity_protocol(
        model,
        object(),  # type: ignore[arg-type]
        benchmark_groups={"weak": ["x1", "x2"], "strong": ["x3", "x4"]},
        decision_threshold=0.0,
        direction="auto",
    )

    assert report["status"] == "FAIL"
    assert report["primary"]["passed"].tolist() == [True, False]
    assert "strong" in report["summary"]


@pytest.mark.parametrize("direction_kwargs", [{}, {"direction": "auto"}], ids=["default", "auto"])
@pytest.mark.parametrize("theta", [0.3, -0.3])
def test_protocol_auto_direction_matches_explicit_direction(monkeypatch, theta, direction_kwargs):
    _install_benchmarks(
        monkeypatch,
        {
            ("x1", "x2"): {
                "r2_y": 0.1,
                "r2_d": 0.1,
                "rho": 0.2,
                "theta_long": theta,
                "theta_short": -theta,
                "delta": 2 * theta,
            }
        },
    )
    model = _DummyIRM(theta=theta, se=0.1)
    expected_direction = "positive" if theta > 0 else "negative"
    policy = {"benchmark_groups": {"drivers": ["x1", "x2"]}, "decision_threshold": 0.0}

    report = run_sensitivity_protocol(model, object(), **policy, **direction_kwargs)
    explicit = run_sensitivity_protocol(model, object(), **policy, direction=expected_direction)

    assert report["direction"] == expected_direction
    assert report["status"] == explicit["status"] == "PASS"
    assert report["limited_margin"] is True
    pd.testing.assert_frame_equal(report["scenarios"], explicit["scenarios"])
    assert any(f"inferred '{expected_direction}'" in warning for warning in report["warnings"])
    assert not any("inferred" in warning for warning in explicit["warnings"])
    rule = "ci_lower > 0" if theta > 0 else "ci_upper < 0"
    assert rule in report["summary"]


@pytest.mark.parametrize("ci_upper", [-0.1, 0.0, 0.1], ids=["below", "touching", "crossing"])
def test_protocol_negative_auto_requires_ci_strictly_below_zero(monkeypatch, ci_upper):
    _install_benchmarks(
        monkeypatch,
        {("x1",): {"r2_y": 0.0, "r2_d": 0.0, "rho": 0.2}},
    )
    # Zero confounding makes the upper endpoint exactly zero in the touching case.
    theta = -norm.ppf(0.975) * 0.1 + ci_upper
    report = run_sensitivity_protocol(
        _DummyIRM(theta=theta, se=0.1),
        object(),
        benchmark_groups={"drivers": ["x1"]},
        decision_threshold=0.0,
    )

    assert report["direction"] == "negative"
    assert report["passed"] is (ci_upper < 0)
    assert report["scenarios"]["ci_upper"].to_numpy() == pytest.approx(ci_upper)
    assert report["scenarios"]["passed"].tolist() == [ci_upper < 0] * 3
    assert "ci_upper < 0" in report["summary"]


@pytest.mark.parametrize(
    ("theta", "threshold", "expected_direction", "passed"),
    [
        (1.0, 0.5, "positive", True),
        (-1.0, -0.5, "negative", True),
        (0.5, 1.0, "negative", True),
        (-0.5, -1.0, "positive", True),
        (0.0, 0.0, "positive", False),
        (0.5, 0.5, "positive", False),
        (-0.5, -0.5, "positive", False),
    ],
)
def test_protocol_auto_direction_is_relative_to_threshold(
    monkeypatch, theta, threshold, expected_direction, passed
):
    _install_benchmarks(
        monkeypatch,
        {("x1",): {"r2_y": 0.01, "r2_d": 0.01, "rho": 0.2}},
    )
    report = run_sensitivity_protocol(
        _DummyIRM(theta=theta, se=0.1),
        object(),
        benchmark_groups={"drivers": ["x1"]},
        decision_threshold=threshold,
    )

    assert report["direction"] == expected_direction
    assert report["decision_threshold"] == threshold
    assert report["passed"] is passed
    assert report["scenarios"]["passed"].tolist() == [passed] * 3
    for raw in report["details"]["drivers"].values():
        assert raw["H0"] == threshold


@pytest.mark.parametrize(
    ("theta", "direction", "rule"),
    [(-1.0, "positive", "ci_lower > 0"), (1.0, "negative", "ci_upper < 0")],
)
def test_protocol_does_not_override_explicit_direction(monkeypatch, theta, direction, rule):
    _install_benchmarks(
        monkeypatch,
        {("x1",): {"r2_y": 0.01, "r2_d": 0.01, "rho": 0.2}},
    )
    report = run_sensitivity_protocol(
        _DummyIRM(theta=theta, se=0.1),
        object(),
        benchmark_groups={"drivers": ["x1"]},
        decision_threshold=0.0,
        direction=direction,
    )

    assert report["direction"] == direction
    assert report["status"] == "FAIL"
    assert not report["scenarios"]["passed"].any()
    assert rule in report["summary"]
    assert "crosses" not in report["summary"]
    assert not any("inferred" in warning for warning in report["warnings"])


def test_protocol_auto_direction_uses_dict_coefficient(monkeypatch):
    _install_benchmarks(
        monkeypatch,
        {("x1",): {"r2_y": 0.01, "r2_d": 0.01, "rho": 0.2}},
    )
    report = run_sensitivity_protocol(
        {"model": _DummyIRM(theta=1.0, se=0.1), "coefficient": -1.0},
        object(),
        benchmark_groups={"drivers": ["x1"]},
        decision_threshold=0.0,
    )

    assert report["direction"] == "negative"
    assert report["status"] == "PASS"
    assert (report["scenarios"]["ci_upper"] < 0).all()


@pytest.mark.parametrize("theta", [np.nan, np.inf, -np.inf])
def test_protocol_auto_direction_rejects_nonfinite_theta(theta):
    with pytest.raises(ValueError, match="requires a finite effect estimate"):
        run_sensitivity_protocol(
            _DummyIRM(theta=theta, se=0.1),
            object(),
            benchmark_groups={"drivers": ["x1"]},
            decision_threshold=0.0,
        )


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


@pytest.mark.parametrize(("theta", "direction"), [(5.0, "positive"), (-5.0, "auto")])
def test_protocol_empty_benchmarks_and_failed_preconditions_return_fail(monkeypatch, theta, direction):
    model = _DummyIRM(theta=theta, se=0.1)
    empty = run_sensitivity_protocol(
        model,
        object(),  # type: ignore[arg-type]
        benchmark_groups={},
        decision_threshold=0.0,
        direction=direction,
    )
    assert empty["status"] == "FAIL"
    assert empty["primary"].empty
    assert empty["direction"] == ("positive" if theta > 0 else "negative")
    assert any("benchmark group is required" in warning for warning in empty["warnings"])
    if direction == "auto":
        assert any("inferred 'negative'" in warning for warning in empty["warnings"])

    _install_benchmarks(
        monkeypatch,
        {("x1", "x2"): {"r2_y": 0.01, "r2_d": 0.01, "rho": 0.1}},
    )
    failed = run_sensitivity_protocol(
        model,
        object(),  # type: ignore[arg-type]
        benchmark_groups={"drivers": ["x1", "x2"]},
        decision_threshold=0.0,
        direction=direction,
        preconditions_passed=False,
    )
    assert failed["status"] == "FAIL"
    assert bool(failed["primary"].loc[0, "passed"]) is True
    assert "preconditions" in failed["summary"]


@pytest.mark.parametrize("overlap_policy", ["clip", "drop"])
@pytest.mark.parametrize("effect_sign", [1, -1], ids=["positive", "negative"])
def test_protocol_runs_end_to_end_with_irm(overlap_policy, effect_sign):
    rng = np.random.default_rng(42)
    n = 240
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x3 = rng.normal(size=n)
    propensity = 1.0 / (1.0 + np.exp(-(0.6 * x1 + 0.3 * x2)))
    d = rng.binomial(1, propensity)
    y = effect_sign * (1.5 * d + 0.8 * x1 + 0.2 * x2 + rng.normal(scale=0.7, size=n))
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
    estimate = model.estimate(score="ATE")

    reports = [
        run_sensitivity_protocol(
            effect_estimation,
            data,
            benchmark_groups={"joint": ["x1", "x2"]},
            decision_threshold=0.0,
        )
        for effect_estimation in (model, {"model": model}, estimate)
    ]
    report = reports[0]
    assert report["direction"] == ("positive" if effect_sign > 0 else "negative")
    for other in reports[1:]:
        assert other["direction"] == report["direction"]
        assert other["status"] == report["status"]
        pd.testing.assert_frame_equal(other["scenarios"], report["scenarios"])

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
