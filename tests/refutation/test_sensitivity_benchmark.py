import numpy as np
import pandas as pd
import pytest

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression

from causalis.dgp.causaldata import CausalData
from causalis.scenarios.unconfoundedness.model import IRM
from causalis.scenarios.unconfoundedness.refutation.unconfoundedness.sensitivity import sensitivity_benchmark


def make_synthetic(n: int = 400, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    logits = 1.0 * x1 + 0.2 * x2
    p = 1.0 / (1.0 + np.exp(-logits))
    d = rng.binomial(1, p)
    y = 1.0 * d + 0.8 * x1 + 0.3 * x2 + rng.normal(scale=0.5, size=n)
    return pd.DataFrame({"y": y, "d": d, "x1": x1, "x2": x2})


def make_data(df: pd.DataFrame, *, confounders: list[str]) -> CausalData:
    return CausalData(df=df, treatment="d", outcome="y", confounders=confounders)


def fit_irm(data: CausalData, *, score: str = "ATE") -> IRM:
    ml_g = RandomForestRegressor(n_estimators=50, random_state=1)
    ml_m = LogisticRegression(max_iter=1000)
    irm = IRM(data=data, ml_g=ml_g, ml_m=ml_m, n_folds=3, random_state=1)
    irm.fit().estimate(score=score)
    return irm


def test_sensitivity_benchmark_single_confounder_returns_long_row():
    df = make_synthetic()
    data = make_data(df, confounders=["x1", "x2"])
    irm = fit_irm(data)

    res = sensitivity_benchmark({"model": irm}, data, ["x1"])

    assert isinstance(res, pd.DataFrame)
    assert list(res.columns) == [
        "benchmark_confounder",
        "r2_y",
        "r2_d",
        "rho",
        "theta_long",
        "theta_short",
        "delta",
    ]
    assert isinstance(res.index, pd.RangeIndex)
    assert res.shape == (1, 7)
    assert res["benchmark_confounder"].tolist() == ["x1"]
    assert np.isfinite(res["r2_y"].to_numpy(dtype=float)).all()
    assert np.isfinite(res["r2_d"].to_numpy(dtype=float)).all()
    assert np.isfinite(res["rho"].to_numpy(dtype=float)).all()
    assert pd.notna(res["theta_short"]).all()
    assert pd.notna(res["delta"]).all()
    assert abs(float(res["delta"].iloc[0])) > 0.0


def test_sensitivity_benchmark_multiple_confounders_returns_one_row_each():
    df = make_synthetic(seed=7)
    data = make_data(df, confounders=["x1", "x2"])
    irm = fit_irm(data)

    res = sensitivity_benchmark({"model": irm}, data, ["x2", "x1", "x2"])

    assert res.shape == (2, 7)
    assert res["benchmark_confounder"].tolist() == ["x2", "x1"]
    assert np.isfinite(res["theta_short"].to_numpy(dtype=float)).all()
    assert np.isfinite(res["delta"].to_numpy(dtype=float)).all()
    assert np.any(np.abs(res["delta"].to_numpy(dtype=float)) > 0.0)


def test_sensitivity_benchmark_all_uses_data_confounder_order():
    df = make_synthetic(seed=99)
    model_data = make_data(df, confounders=["x1", "x2"])
    irm = fit_irm(model_data)
    benchmark_data = make_data(df, confounders=["x2", "x1"])

    res = sensitivity_benchmark({"model": irm}, benchmark_data, "all")

    assert res.shape == (2, 7)
    assert res["benchmark_confounder"].tolist() == ["x2", "x1"]
    assert (res["theta_long"] == res["theta_long"].iloc[0]).all()


def test_sensitivity_benchmark_atte_returns_real_short_refit_outputs():
    df = make_synthetic(seed=77)
    data = make_data(df, confounders=["x1", "x2"])
    irm = fit_irm(data, score="ATTE")

    res = sensitivity_benchmark({"model": irm}, data, "all")

    assert res["benchmark_confounder"].tolist() == ["x1", "x2"]
    assert np.isfinite(res["r2_y"].to_numpy(dtype=float)).all()
    assert np.isfinite(res["r2_d"].to_numpy(dtype=float)).all()
    assert np.isfinite(res["rho"].to_numpy(dtype=float)).all()
    assert pd.notna(res["theta_short"]).all()
    assert pd.notna(res["delta"]).all()


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
