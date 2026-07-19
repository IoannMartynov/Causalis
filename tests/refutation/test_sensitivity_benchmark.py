import numpy as np
import pandas as pd
import pytest

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression

from causalis.dgp.causaldata import CausalData
from causalis.scenarios.unconfoundedness.model import IRM
from causalis.scenarios.unconfoundedness.refutation.unconfoundedness.sensitivity import (
    _calibrate_benchmark_gain_statistics,
    sensitivity_benchmark,
    sensitivity_benchmark_group,
)


BENCHMARK_AUDIT_COLUMNS = [
    "cf_y",
    "cf_d",
    "cf_y_raw",
    "cf_d_raw",
    "sigma2_long",
    "sigma2_short",
    "nu2_long",
    "nu2_short",
    "rho_raw",
    "rho_clipped",
    "cf_y_clipped",
    "cf_d_clipped",
    "rho_fallback",
    "boundary_calibration",
    "strengths_valid",
    "calibration_valid",
    "calibration_issue",
    "calibration_warning",
]


def make_synthetic(n: int = 400, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    logits = 1.0 * x1 + 0.2 * x2
    p = 1.0 / (1.0 + np.exp(-logits))
    d = rng.binomial(1, p)
    y = 1.0 * d + 0.8 * x1 + 0.3 * x2 + rng.normal(scale=0.5, size=n)
    x3 = rng.normal(size=n)
    return pd.DataFrame({"y": y, "d": d, "x1": x1, "x2": x2, "x3": x3})


def make_data(df: pd.DataFrame, *, confounders: list[str]) -> CausalData:
    return CausalData(df=df, treatment="d", outcome="y", confounders=confounders)


def fit_irm(
    data: CausalData,
    *,
    score: str = "ATE",
    overlap_policy: str = "clip",
    overlap_threshold: float = 0.01,
    random_state: int | None = 1,
) -> IRM:
    ml_g = RandomForestRegressor(n_estimators=50, random_state=1)
    ml_m = LogisticRegression(max_iter=1000)
    irm = IRM(
        data=data,
        ml_g=ml_g,
        ml_m=ml_m,
        n_folds=3,
        random_state=random_state,
        overlap_policy=overlap_policy,
        overlap_threshold=overlap_threshold,
    )
    irm.fit().estimate(score=score)
    return irm


def test_sensitivity_benchmark_single_confounder_returns_long_row():
    df = make_synthetic()
    data = make_data(df, confounders=["x1", "x2"])
    irm = fit_irm(data)

    res = sensitivity_benchmark({"model": irm}, data, ["x1"])

    assert isinstance(res, pd.DataFrame)
    assert list(res.columns[:7]) == [
        "benchmark_confounder",
        "r2_y",
        "r2_d",
        "rho",
        "theta_long",
        "theta_short",
        "delta",
    ]
    assert list(res.columns[7:]) == BENCHMARK_AUDIT_COLUMNS
    assert isinstance(res.index, pd.RangeIndex)
    assert res.shape == (1, 25)
    assert res["benchmark_confounder"].tolist() == ["x1"]
    assert np.isfinite(res["r2_y"].to_numpy(dtype=float)).all()
    assert np.isfinite(res["r2_d"].to_numpy(dtype=float)).all()
    assert np.isfinite(res["rho"].to_numpy(dtype=float)).all()
    assert pd.notna(res["theta_short"]).all()
    assert pd.notna(res["delta"]).all()
    assert bool(res.loc[0, "calibration_valid"]) is True
    expected_rho = (
        float(res.loc[0, "theta_short"] - res.loc[0, "theta_long"])
        / np.sqrt(
            float(res.loc[0, "sigma2_short"] - res.loc[0, "sigma2_long"])
            * float(res.loc[0, "nu2_long"] - res.loc[0, "nu2_short"])
        )
    )
    assert float(res.loc[0, "rho"]) == pytest.approx(expected_rho)
    assert abs(float(res.loc[0, "rho"])) < 1.0
    assert abs(float(res["delta"].iloc[0])) > 0.0


def test_sensitivity_benchmark_multiple_confounders_returns_one_row_each():
    df = make_synthetic(seed=7)
    data = make_data(df, confounders=["x1", "x2"])
    irm = fit_irm(data)

    res = sensitivity_benchmark({"model": irm}, data, ["x2", "x1", "x2"])

    assert res.shape == (2, 25)
    assert res["benchmark_confounder"].tolist() == ["x2", "x1"]
    assert np.isfinite(res["theta_short"].to_numpy(dtype=float)).all()
    assert np.isfinite(res["delta"].to_numpy(dtype=float)).all()
    assert np.any(np.abs(res["delta"].to_numpy(dtype=float)) > 0.0)


@pytest.mark.parametrize("overlap_policy", ["clip", "drop"])
def test_sensitivity_benchmark_group_excludes_features_together(overlap_policy):
    df = make_synthetic(seed=17)
    data = make_data(df, confounders=["x1", "x2", "x3"])
    irm = fit_irm(
        data,
        overlap_policy=overlap_policy,
        overlap_threshold=0.15,
    )

    res = sensitivity_benchmark_group(
        {"model": irm},
        data,
        ["x2", "x1", "x2"],
    )

    assert list(res.columns[:7]) == [
        "benchmark_group",
        "r2_y",
        "r2_d",
        "rho",
        "theta_long",
        "theta_short",
        "delta",
    ]
    assert list(res.columns[7:]) == BENCHMARK_AUDIT_COLUMNS
    assert res.shape == (1, 25)
    assert res["benchmark_group"].tolist() == [("x2", "x1")]
    assert np.isfinite(res[["theta_long", "theta_short", "delta"]].to_numpy(dtype=float)).all()
    if bool(res.loc[0, "calibration_valid"]):
        assert np.isfinite(res[["r2_y", "r2_d", "rho"]].to_numpy(dtype=float)).all()
    else:
        assert pd.notna(res.loc[0, "calibration_issue"])


def test_sensitivity_benchmark_group_validates_group_input():
    df = make_synthetic(seed=18)
    data = make_data(df, confounders=["x1", "x2", "x3"])
    irm = fit_irm(data)
    effect = {"model": irm}

    with pytest.raises(TypeError, match="benchmarking_group must be a list"):
        sensitivity_benchmark_group(effect, data, "x1")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must not be empty"):
        sensitivity_benchmark_group(effect, data, [])
    with pytest.raises(TypeError, match="contain only strings"):
        sensitivity_benchmark_group(effect, data, [1])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="must be a subset"):
        sensitivity_benchmark_group(effect, data, ["not_in_data"])
    with pytest.raises(ValueError, match="would leave no confounders"):
        sensitivity_benchmark_group(effect, data, ["x1", "x2", "x3"])


def test_sensitivity_benchmark_all_uses_data_confounder_order():
    df = make_synthetic(seed=99)
    model_data = make_data(df, confounders=["x1", "x2"])
    irm = fit_irm(model_data)
    benchmark_data = make_data(df, confounders=["x2", "x1"])

    res = sensitivity_benchmark({"model": irm}, benchmark_data, "all")

    assert res.shape == (2, 25)
    assert res["benchmark_confounder"].tolist() == ["x2", "x1"]
    assert (res["theta_long"] == res["theta_long"].iloc[0]).all()


def test_sensitivity_benchmark_atte_returns_real_short_refit_outputs():
    df = make_synthetic(seed=77)
    data = make_data(df, confounders=["x1", "x2"])
    irm = fit_irm(data, score="ATTE")

    res = sensitivity_benchmark({"model": irm}, data, "all")

    assert res["benchmark_confounder"].tolist() == ["x1", "x2"]
    assert pd.notna(res["theta_short"]).all()
    assert pd.notna(res["delta"]).all()
    valid = res["calibration_valid"].astype(bool)
    assert np.isfinite(res.loc[valid, ["r2_y", "r2_d", "rho"]].to_numpy(dtype=float)).all()
    assert res.loc[~valid, "calibration_issue"].notna().all()


def test_sensitivity_benchmark_reuses_long_cross_fitting_folds(monkeypatch):
    df = make_synthetic(seed=78)
    data = make_data(df, confounders=["x1", "x2"])
    irm = fit_irm(data)
    expected_folds = np.asarray(irm.folds_, dtype=int).copy()
    observed_fixed_folds = []
    original = IRM._cross_fit_nuisances

    def _spy_cross_fit(self, X, y, d, y_is_binary):
        observed_fixed_folds.append(
            None
            if self._fixed_fold_assignments_ is None
            else np.asarray(self._fixed_fold_assignments_, dtype=int).copy()
        )
        return original(self, X, y, d, y_is_binary)

    monkeypatch.setattr(IRM, "_cross_fit_nuisances", _spy_cross_fit)
    sensitivity_benchmark({"model": irm}, data, ["x1"])

    assert len(observed_fixed_folds) == 1
    assert np.array_equal(observed_fixed_folds[0], expected_folds)


def test_sensitivity_benchmark_drop_reuses_full_sample_folds(monkeypatch):
    df = make_synthetic(seed=79)
    data = make_data(df, confounders=["x1", "x2"])
    irm = fit_irm(
        data,
        overlap_policy="drop",
        overlap_threshold=0.15,
    )
    expected_folds = np.asarray(irm._full_sample_folds_, dtype=int).copy()
    observed_fixed_folds = []
    original = IRM._cross_fit_nuisances

    assert expected_folds.size == len(df)
    assert len(irm.folds_) < len(df)

    def _spy_cross_fit(self, X, y, d, y_is_binary):
        observed_fixed_folds.append(
            None
            if self._fixed_fold_assignments_ is None
            else np.asarray(self._fixed_fold_assignments_, dtype=int).copy()
        )
        return original(self, X, y, d, y_is_binary)

    monkeypatch.setattr(IRM, "_cross_fit_nuisances", _spy_cross_fit)
    result = sensitivity_benchmark({"model": irm}, data, ["x1"])

    assert result.shape == (1, 25)
    assert len(observed_fixed_folds) == 1
    assert np.array_equal(observed_fixed_folds[0], expected_folds)


def test_sensitivity_benchmark_drop_legacy_seeded_model_recreates_folds():
    df = make_synthetic(seed=80)
    data = make_data(df, confounders=["x1", "x2"])
    irm = fit_irm(
        data,
        overlap_policy="drop",
        overlap_threshold=0.15,
        random_state=3,
    )
    irm._full_sample_folds_ = None

    result = sensitivity_benchmark({"model": irm}, data, ["x1"])

    assert result.shape == (1, 25)
    assert "cross-fitting splits cannot be aligned" not in str(
        result.loc[0, "calibration_issue"]
    )


def test_sensitivity_benchmark_drop_legacy_unseeded_model_fails_closed():
    df = make_synthetic(seed=81)
    data = make_data(df, confounders=["x1", "x2"])
    irm = fit_irm(
        data,
        overlap_policy="drop",
        overlap_threshold=0.15,
        random_state=None,
    )
    irm._full_sample_folds_ = None

    result = sensitivity_benchmark({"model": irm}, data, ["x1"])

    assert bool(result.loc[0, "calibration_valid"]) is False
    assert "cross-fitting splits cannot be aligned" in str(
        result.loc[0, "calibration_issue"]
    )


def test_sensitivity_benchmark_validates_data_and_benchmark_inputs():
    df = make_synthetic(seed=123)
    data = make_data(df, confounders=["x1", "x2"])
    irm = fit_irm(data)
    effect = {"model": irm}

    with pytest.raises(TypeError):
        sensitivity_benchmark(effect, None, ["x1"])  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        sensitivity_benchmark(effect, df, ["x1"])  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        sensitivity_benchmark(effect, data, "x1")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        sensitivity_benchmark(effect, data, [1])  # type: ignore[list-item]
    with pytest.raises(ValueError):
        sensitivity_benchmark(effect, data, [])
    with pytest.raises(ValueError):
        sensitivity_benchmark(effect, data, ["not_in_data"])


def test_sensitivity_benchmark_rejects_data_model_confounder_mismatch():
    df = make_synthetic(seed=222)
    model_data = make_data(df, confounders=["x1", "x2"])
    irm = fit_irm(model_data)
    mismatched_data = make_data(df, confounders=["x1"])

    with pytest.raises(ValueError, match="data.confounders must match"):
        sensitivity_benchmark({"model": irm}, mismatched_data, ["x1"])


def test_sensitivity_benchmark_rejects_sample_order_mismatch():
    df = make_synthetic(seed=333)
    data = make_data(df, confounders=["x1", "x2"])
    irm = fit_irm(data)
    shuffled_df = df.sample(frac=1.0, random_state=5).reset_index(drop=True)
    shuffled_data = make_data(shuffled_df, confounders=["x1", "x2"])

    with pytest.raises(ValueError, match="sample and row order"):
        sensitivity_benchmark({"model": irm}, shuffled_data, ["x1"])


def test_sensitivity_benchmark_rejects_single_confounder_long_model():
    df = make_synthetic(seed=444).drop(columns=["x2"])
    data = make_data(df, confounders=["x1"])
    irm = fit_irm(data)

    with pytest.raises(ValueError, match="at least two confounders"):
        sensitivity_benchmark({"model": irm}, data, "all")


def test_benchmark_rejects_gate_score_override():
    df = make_synthetic(seed=202)
    data = make_data(df, confounders=["x1", "x2"])
    irm = fit_irm(data)

    with pytest.raises(ValueError, match="supports only score='ATE' or score='ATTE'"):
        sensitivity_benchmark({"model": irm}, data, ["x1"], fit_args={"score": "GATE"})


def test_gain_calibration_uses_long_short_elements_and_preserves_delta_sign():
    calibrated = _calibrate_benchmark_gain_statistics(
        theta_long=2.0,
        theta_short=2.3,
        sigma2_long=4.0,
        sigma2_short=5.0,
        nu2_long=9.0,
        nu2_short=8.0,
    )

    assert calibrated["cf_y_raw"] == pytest.approx(0.25)
    assert calibrated["cf_y"] == pytest.approx(0.25)
    assert calibrated["r2_y"] == pytest.approx(0.2)
    assert calibrated["cf_d_raw"] == pytest.approx(0.125)
    assert calibrated["r2_d"] == pytest.approx(0.125)
    assert calibrated["rho_raw"] == pytest.approx(0.3)
    assert calibrated["rho"] == pytest.approx(0.3)
    assert calibrated["rho_clipped"] is False
    assert calibrated["strengths_valid"] is True
    assert calibrated["calibration_valid"] is True


def test_gain_calibration_clips_cf_y_and_rho_but_remains_valid():
    calibrated = _calibrate_benchmark_gain_statistics(
        theta_long=0.0,
        theta_short=10.0,
        sigma2_long=1.0,
        sigma2_short=3.0,
        nu2_long=2.0,
        nu2_short=1.5,
    )

    assert calibrated["cf_y_raw"] == pytest.approx(2.0)
    assert calibrated["cf_y"] == pytest.approx(1.0)
    assert calibrated["r2_y"] == pytest.approx(0.5)
    assert calibrated["rho_raw"] > 1.0
    assert calibrated["rho"] == 1.0
    assert calibrated["rho_clipped"] is True
    assert calibrated["cf_y_clipped"] is True
    assert calibrated["boundary_calibration"] is True
    assert calibrated["calibration_valid"] is True


@pytest.mark.parametrize(
    ("kwargs", "expected_cf_y", "expected_cf_d", "expected_rho"),
    [
        (
            {"sigma2_long": 1.0, "sigma2_short": 1.0, "nu2_long": 2.0, "nu2_short": 1.5},
            0.0,
            1.0 / 3.0,
            1.0,
        ),
        (
            {"sigma2_long": 1.0, "sigma2_short": 2.0, "nu2_long": 2.0, "nu2_short": 2.0},
            1.0,
            0.0,
            1.0,
        ),
        (
            {"sigma2_long": 1.0, "sigma2_short": 2.0, "nu2_long": 2.0, "nu2_short": 3.0},
            1.0,
            0.0,
            1.0,
        ),
    ],
)
def test_gain_calibration_uses_doubleml_boundary_fallback(
    kwargs,
    expected_cf_y,
    expected_cf_d,
    expected_rho,
):
    calibrated = _calibrate_benchmark_gain_statistics(
        theta_long=1.0,
        theta_short=1.1,
        **kwargs,
    )

    assert calibrated["cf_y"] == pytest.approx(expected_cf_y)
    assert calibrated["cf_d"] == pytest.approx(expected_cf_d)
    assert calibrated["rho"] == pytest.approx(expected_rho)
    assert calibrated["rho_fallback"] is True
    assert calibrated["boundary_calibration"] is True
    assert calibrated["strengths_valid"] is True
    assert calibrated["calibration_valid"] is True
    assert calibrated["calibration_issue"] is None


def test_gain_calibration_preserves_negative_raw_gain_before_doubleml_clipping():
    calibrated = _calibrate_benchmark_gain_statistics(
        theta_long=1.0,
        theta_short=0.9,
        sigma2_long=1.0,
        sigma2_short=2.0,
        nu2_long=2.0,
        nu2_short=3.0,
    )

    assert calibrated["cf_d_raw"] == pytest.approx(-1.0 / 3.0)
    assert calibrated["cf_d"] == 0.0
    assert calibrated["cf_d_clipped"] is True
    assert calibrated["rho_raw"] != calibrated["rho_raw"]
    assert calibrated["rho"] == -1.0
    assert calibrated["rho_fallback"] is True
    assert "clipped to 0" in calibrated["calibration_warning"]


def test_gain_calibration_keeps_doubleml_cf_d_upper_boundary_but_fails_safely():
    calibrated = _calibrate_benchmark_gain_statistics(
        theta_long=1.0,
        theta_short=1.1,
        sigma2_long=1.0,
        sigma2_short=2.0,
        nu2_long=3.0,
        nu2_short=1.0,
    )

    assert calibrated["cf_d_raw"] == pytest.approx(2.0)
    assert calibrated["cf_d"] == 1.0
    assert calibrated["cf_d_clipped"] is True
    assert calibrated["strengths_valid"] is False
    assert calibrated["calibration_valid"] is False
    assert "cf_d=1" in calibrated["calibration_issue"]


def test_gain_calibration_still_fails_for_unavailable_elements():
    calibrated = _calibrate_benchmark_gain_statistics(
        theta_long=1.0,
        theta_short=1.1,
        sigma2_long=np.nan,
        sigma2_short=2.0,
        nu2_long=2.0,
        nu2_short=1.0,
        element_issue="sensitivity elements are unavailable",
    )

    assert calibrated["strengths_valid"] is False
    assert calibrated["calibration_valid"] is False
    assert np.isnan(calibrated["rho"])
    assert "sensitivity elements are unavailable" in calibrated["calibration_issue"]
