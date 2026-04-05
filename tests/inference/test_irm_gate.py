import numpy as np
import pandas as pd
import pytest
import warnings
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression, LogisticRegression

from causalis.data_contracts.gate_estimate import GateEstimate
from causalis.dgp.causaldata import CausalData
from causalis.scenarios.gate.model import _coerce_groups_to_basis, _compute_gate_signal_from_irm
from causalis.scenarios.unconfoundedness.model import IRM


def _make_synthetic_data(n: int = 500, seed: int = 7) -> tuple[CausalData, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    p = 1.0 / (1.0 + np.exp(-(0.6 * x1 - 0.4 * x2)))
    d = rng.binomial(1, p)
    tau = 1.2 + 0.5 * (x1 > 0.0).astype(float)
    y = 1.0 + 0.4 * x1 - 0.2 * x2 + tau * d + rng.normal(scale=1.0, size=n)

    df = pd.DataFrame({"y": y, "d": d, "x1": x1, "x2": x2})
    cd = CausalData(df=df, treatment="d", outcome="y", confounders=["x1", "x2"])
    return cd, df


def _fit_irm(
    cd: CausalData,
    *,
    normalize_ipw: bool = False,
    random_state: int = 17,
    store_diagnostics: bool = True,
) -> IRM:
    irm = IRM(
        data=cd,
        ml_g=LinearRegression(),
        ml_m=LogisticRegression(max_iter=2000, solver="lbfgs"),
        n_folds=4,
        normalize_ipw=normalize_ipw,
        trimming_threshold=1e-3,
        random_state=random_state,
    )
    irm.fit(store_diagnostics=store_diagnostics)
    return irm


def _groups_from_df(df: pd.DataFrame) -> pd.Series:
    return pd.Series(np.where(df["x1"] >= 0.0, "high_x1", "low_x1"), name="segment")


def test_irm_estimate_gate_dispatches_to_new_module(monkeypatch):
    cd, df = _make_synthetic_data(n=180, seed=1)
    irm = _fit_irm(cd, store_diagnostics=False)
    groups = _groups_from_df(df)
    marker = object()
    captured = {}

    def _fake_gate_estimator(**kwargs):
        captured.update(kwargs)
        return marker

    monkeypatch.setattr("causalis.scenarios.unconfoundedness.model.estimate_gate_from_irm", _fake_gate_estimator)

    res = irm.estimate(
        score="GATE",
        groups=groups,
        alpha=0.1,
        cov_type="HC2",
        cov_kwds={"use_correction": True},
    )
    assert res is marker
    assert captured["irm_model"] is irm
    assert captured["groups"] is groups
    assert captured["alpha"] == 0.1
    assert captured["cov_type"] == "HC2"
    assert captured["cov_kwds"] == {"use_correction": True}
    assert captured["irm_model"].store_diagnostics is False


def test_gate_basis_handling_inputs():
    cd, df = _make_synthetic_data(n=260, seed=2)
    irm = _fit_irm(cd)
    groups_series = _groups_from_df(df)
    groups_df_1col = groups_series.to_frame()
    groups_dummy = pd.DataFrame(
        {
            "g_high": (groups_series == "high_x1").astype(int),
            "g_low": (groups_series == "low_x1").astype(int),
        }
    )

    res_series = irm.estimate(score="GATE", groups=groups_series)
    res_df = irm.estimate(score="GATE", groups=groups_df_1col)
    res_dummy = irm.estimate(score="GATE", groups=groups_dummy)

    assert isinstance(res_series, GateEstimate)
    assert isinstance(res_df, GateEstimate)
    assert isinstance(res_dummy, GateEstimate)
    assert sorted(res_series.group_names) == ["segment=high_x1", "segment=low_x1"]
    np.testing.assert_allclose(np.sort(res_series.values), np.sort(res_df.values), atol=1e-10)

    overlap = groups_dummy.copy()
    overlap.iloc[:20, :] = 1
    with pytest.raises(ValueError, match="mutually exclusive and exhaustive"):
        irm.estimate(score="GATE", groups=overlap)

    non_exhaustive = groups_dummy.copy()
    non_exhaustive.iloc[:20, :] = 0
    with pytest.raises(ValueError, match="mutually exclusive and exhaustive"):
        irm.estimate(score="GATE", groups=non_exhaustive)


def test_gate_matches_manual_group_means_of_phi():
    cd, df = _make_synthetic_data(n=320, seed=3)
    irm = _fit_irm(cd, normalize_ipw=True)
    groups = _groups_from_df(df)

    with pytest.warns(RuntimeWarning, match="ignored for GATE"):
        res = irm.estimate(score="GATE", groups=groups, cov_type="HC3")

    assert res.model_options["normalize_ipw_requested"] is True
    assert res.model_options["normalize_ipw_effective"] is False
    assert res.model_options["se_approx_hajek"] is False

    phi, _, _ = _compute_gate_signal_from_irm(irm)
    basis = _coerce_groups_to_basis(groups, n_obs=phi.shape[0])
    basis_np = basis.to_numpy(dtype=float)
    manual = (basis_np.T @ phi) / basis_np.sum(axis=0)

    np.testing.assert_allclose(res.values, manual, atol=1e-10)


def test_gate_inference_covariance_options_and_se_validity():
    cd, df = _make_synthetic_data(n=340, seed=4)
    irm = _fit_irm(cd)
    groups = _groups_from_df(df)

    res_default = irm.estimate(score="GATE", groups=groups)
    assert res_default.model_options["cov_type"] == "HC3"
    assert np.all(np.isfinite(res_default.std_errors))
    assert np.all(res_default.std_errors >= 0.0)

    res_hc1 = irm.estimate(score="GATE", groups=groups, cov_type="HC1")
    assert res_hc1.model_options["cov_type"] == "HC1"
    assert np.all(np.isfinite(res_hc1.std_errors))
    assert np.all(res_hc1.std_errors >= 0.0)


def test_gate_payload_follows_fit_store_diagnostics_setting():
    cd, df = _make_synthetic_data(n=220, seed=14)
    groups = _groups_from_df(df)

    irm_diag = _fit_irm(cd, store_diagnostics=True)
    irm_light = _fit_irm(cd, store_diagnostics=False)

    assert irm_diag.estimate(score="GATE", groups=groups).diagnostic_data is not None
    assert irm_light.estimate(score="GATE", groups=groups).diagnostic_data is None


def test_gate_lightweight_mode_uses_cached_targets_without_reloading_dataframe(monkeypatch):
    cd, df = _make_synthetic_data(n=200, seed=12)
    groups = _groups_from_df(df)
    irm = _fit_irm(cd, store_diagnostics=False)

    def _unexpected_get_df(*args, **kwargs):
        raise AssertionError("GATE unexpectedly reloaded the dataframe")

    monkeypatch.setattr(CausalData, "get_df", _unexpected_get_df)
    res = irm.estimate(score="GATE", groups=groups)

    assert isinstance(res, GateEstimate)


def test_gate_lightweight_mode_matches_full_mode_on_unchanged_data():
    cd, df = _make_synthetic_data(n=260, seed=18)
    groups = _groups_from_df(df)

    res_full = _fit_irm(cd, store_diagnostics=True).estimate(score="GATE", groups=groups)
    res_light = _fit_irm(cd, store_diagnostics=False).estimate(score="GATE", groups=groups)

    np.testing.assert_allclose(res_light.values, res_full.values, atol=1e-12)


def test_gate_lightweight_mode_is_stable_after_data_reordering():
    cd, df = _make_synthetic_data(n=240, seed=22)
    groups = _groups_from_df(df)
    irm = _fit_irm(cd, store_diagnostics=False)

    first = irm.estimate(score="GATE", groups=groups)
    cd.df = cd.df.sample(frac=1.0, random_state=31).reset_index(drop=True)
    second = irm.estimate(score="GATE", groups=groups)

    np.testing.assert_allclose(second.values, first.values, atol=1e-12)


def test_gate_requires_fitted_model():
    cd, df = _make_synthetic_data(n=120, seed=5)
    irm = IRM(
        data=cd,
        ml_g=LinearRegression(),
        ml_m=LogisticRegression(max_iter=2000, solver="lbfgs"),
        n_folds=3,
        random_state=11,
    )
    groups = _groups_from_df(df)

    with pytest.raises(NotFittedError):
        irm.estimate(score="GATE", groups=groups)


def test_gate_groups_fallback_uses_causaldata_attribute():
    cd, df = _make_synthetic_data(n=260, seed=6)
    groups = _groups_from_df(df)
    object.__setattr__(cd, "gate_groups", groups)
    irm = _fit_irm(cd)

    res = irm.estimate(score="GATE")
    assert isinstance(res, GateEstimate)
    assert len(res.group_names) == 2


def test_gate_raises_when_groups_missing_and_no_fallback():
    cd, _ = _make_synthetic_data(n=180, seed=8)
    irm = _fit_irm(cd)

    with pytest.raises(ValueError, match="GATE requires pre-defined groups"):
        irm.estimate(score="GATE")


def test_gate_method_is_thin_wrapper_over_estimate():
    cd, df = _make_synthetic_data(n=220, seed=10)
    groups = _groups_from_df(df)
    irm = _fit_irm(cd, random_state=101)

    res_gate = irm.gate(groups=groups, alpha=0.1, cov_type="HC2")
    res_estimate = irm.estimate(score="GATE", groups=groups, alpha=0.1, cov_type="HC2")

    np.testing.assert_allclose(res_gate.values, res_estimate.values, atol=1e-12)
    np.testing.assert_allclose(res_gate.std_errors, res_estimate.std_errors, atol=1e-12)
    assert res_gate.model_options["cov_type"] == "HC2"


def test_gate_deterministic_with_fixed_random_state():
    cd1, df1 = _make_synthetic_data(n=300, seed=9)
    groups1 = _groups_from_df(df1)
    irm1 = _fit_irm(cd1, random_state=123)
    res1 = irm1.estimate(score="GATE", groups=groups1, cov_type="HC2")

    cd2, df2 = _make_synthetic_data(n=300, seed=9)
    groups2 = _groups_from_df(df2)
    irm2 = _fit_irm(cd2, random_state=123)
    res2 = irm2.estimate(score="GATE", groups=groups2, cov_type="HC2")

    np.testing.assert_allclose(res1.values, res2.values, atol=1e-12)
    np.testing.assert_allclose(res1.std_errors, res2.std_errors, atol=1e-12)



def test_gate_raises_for_groups_without_treated_and_control_support():
    cd, df = _make_synthetic_data(n=240, seed=15)
    irm = _fit_irm(cd, random_state=33)
    groups = pd.Series(df["d"].to_numpy(dtype=float), name="d")

    irm.m_hat_ = np.where(irm._d == 1.0, 0.965, 0.010)

    with pytest.raises(
        ValueError,
        match="Each GATE group must contain at least one treated and one control observation",
    ):
        irm.estimate(score="GATE", groups=groups, cov_type="HC3")


def test_gate_signal_non_finite_values_raise_runtime_error():
    cd, _ = _make_synthetic_data(n=180, seed=14)
    irm = _fit_irm(cd, random_state=91)

    irm.m_hat_ = np.ones_like(irm.m_hat_)
    with pytest.raises(RuntimeError, match="non-finite values"):
        _compute_gate_signal_from_irm(irm)
