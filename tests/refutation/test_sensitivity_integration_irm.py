import numpy as np
import pandas as pd
import pytest

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression

from causalis.dgp.causaldata import CausalData
from causalis.dgp import generate_rct
from causalis.scenarios.unconfoundedness.model import IRM
from causalis.scenarios.unconfoundedness.refutation.unconfoundedness.sensitivity import sensitivity_analysis, get_sensitivity_summary


def _make_cd(n=600, random_state=3, outcome_type="normal"):
    df = generate_rct(n=n, split=0.5, random_state=random_state, outcome_type=outcome_type, k=3, add_ancillary=False)
    y = "y"; d = "d"
    xcols = [c for c in df.columns if c not in {y, d, "m", "m_obs", "tau_link", "g0", "g1", "cate"}]
    return CausalData(df=df[[y, d] + xcols], treatment=d, outcome=y, confounders=xcols)


def _make_overlap_cd(n: int = 300, random_state: int = 23) -> CausalData:
    rng = np.random.default_rng(random_state)
    x1 = np.linspace(-4.0, 4.0, n)
    x2 = rng.normal(size=n)
    propensity = 1.0 / (1.0 + np.exp(-(1.8 * x1 + 0.2 * x2)))
    d = rng.binomial(1, propensity)
    y = 1.2 * d + 0.7 * x1 + 0.2 * x2 + rng.normal(scale=0.5, size=n)
    return CausalData(
        df=pd.DataFrame({"y": y, "d": d, "x1": x1, "x2": x2}),
        treatment="d",
        outcome="y",
        confounders=["x1", "x2"],
    )


@pytest.mark.parametrize("overlap_policy", ["clip", "drop"])
@pytest.mark.parametrize("score", ["ATE", "ATTE"])
def test_sensitivity_respects_overlap_policy_for_model_and_estimate(overlap_policy, score):
    data = _make_overlap_cd()
    model = IRM(
        data,
        ml_g=LinearRegression(),
        ml_m=LogisticRegression(max_iter=1000),
        n_folds=3,
        random_state=17,
        overlap_policy=overlap_policy,
        overlap_threshold=0.10,
    ).fit()
    estimate = model.estimate(score=score)

    model_result = sensitivity_analysis(
        model, r2_y=0.02, r2_d=0.03, rho=0.8
    )
    estimate_result = sensitivity_analysis(
        estimate, r2_y=0.02, r2_d=0.03, rho=0.8
    )
    elements = model._sensitivity_element_est()
    diagnostic = estimate.diagnostic_data
    effective_n = len(model.m_hat_)

    assert diagnostic is not None
    assert len(model._full_sample_folds_) == len(data.df)
    assert len(model.folds_) == effective_n
    for name in ("psi", "psi_sigma2", "psi_nu2", "riesz_rep", "m_alpha"):
        assert len(elements[name]) == effective_n
        np.testing.assert_allclose(getattr(diagnostic, name), elements[name])
    for name in ("theta", "se", "sigma2", "nu2", "bound_width"):
        assert estimate_result[name] == pytest.approx(model_result[name])
        assert np.isfinite(model_result[name])
    np.testing.assert_allclose(
        estimate_result["bias_aware_ci"], model_result["bias_aware_ci"]
    )

    if overlap_policy == "clip":
        assert effective_n == len(data.df)
        assert model.overlap_n_dropped_ == 0
        assert np.all(model.overlap_mask_)
        assert np.all((model.m_hat_ >= 0.10) & (model.m_hat_ <= 0.90))
    else:
        assert effective_n < len(data.df)
        assert model.overlap_n_dropped_ == len(data.df) - effective_n
        assert np.all((model.m_hat_ > 0.10) & (model.m_hat_ < 0.90))


def test_sensitivity_with_dml_ate_runs_and_returns_dict():
    cd = _make_cd(n=400, random_state=11, outcome_type="normal")
    ml_g = RandomForestRegressor(n_estimators=30, random_state=1)
    ml_m = RandomForestClassifier(n_estimators=30, random_state=1)

    res = IRM(cd, ml_g=ml_g, ml_m=ml_m, n_folds=3).fit()
    res.estimate(score="ATE")
    out = sensitivity_analysis(res, r2_y=0.02, r2_d=0.03, rho=1.0)

    assert isinstance(out, dict)
    summary_df = out.summary()
    assert isinstance(summary_df, pd.DataFrame)
    assert list(summary_df.columns) == ["statistics", "value"]
    assert summary_df["statistics"].tolist() == [
        "bias_aware_ci",
        "theta",
        "sampling_ci",
        "rv",
        "rva",
        "se",
        "max_bias",
        "max_bias_base",
        "bound_width",
        "sigma2",
        "nu2",
    ]
    assert summary_df.iloc[0]["value"] == [round(out["bias_aware_ci"][0], 4), round(out["bias_aware_ci"][1], 4)]
    assert summary_df.iloc[1]["value"] == [round(out["theta_bounds_cofounding"][0], 4), round(out["theta"], 4), round(out["theta_bounds_cofounding"][1], 4)]
    assert "Bias-aware Interval" in str(out)
    assert "Bias-aware Interval" in repr(out)
    assert isinstance(get_sensitivity_summary(out), str)
    # Integration: summary should be retrievable via the getter
    summ = get_sensitivity_summary(res)
    assert isinstance(summ, str)
    assert any(kw in summ for kw in ("Bias-aware Interval", "Intervals"))


def test_sensitivity_with_dml_att_runs_and_returns_dict():
    cd = _make_cd(n=400, random_state=7, outcome_type="normal")
    ml_g = RandomForestRegressor(n_estimators=25, random_state=0)
    ml_m = RandomForestClassifier(n_estimators=25, random_state=0)

    res = IRM(cd, ml_g=ml_g, ml_m=ml_m, n_folds=3).fit()
    res.estimate(score="ATTE")
    out = sensitivity_analysis(res, r2_y=0.01, r2_d=0.04, rho=0.8)

    assert isinstance(out, dict)
    assert isinstance(out.summary(), pd.DataFrame)
    summ = get_sensitivity_summary(res)
    assert isinstance(summ, str)
    assert "Bias-aware Interval" in summ
