import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from causalis.dgp.causaldata import CausalData
from causalis.scenarios.unconfoundedness.model import IRM


def _make_synth(n=120, seed=123):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    logits = 0.8 * X[:, 0] - 0.5 * X[:, 1] + 0.2 * X[:, 2]
    p = 1 / (1 + np.exp(-logits))
    D = rng.binomial(1, p)
    Y = 1.5 * D + X[:, 0] + 0.5 * X[:, 1] + rng.normal(scale=1.0, size=n)
    df = pd.DataFrame({
        "y": Y,
        "d": D,
        "x1": X[:, 0],
        "x2": X[:, 1],
        "x3": X[:, 2],
    })
    data = CausalData(df=df, treatment="d", outcome="y", confounders=["x1", "x2", "x3"])
    return data


def test_fit_store_diagnostics_controls_ate_payload():
    data = _make_synth(n=140, seed=99)
    ml_g = LinearRegression()
    ml_m = LogisticRegression(max_iter=1000)

    res_default = IRM(data, ml_g=ml_g, ml_m=ml_m, n_folds=3).fit().estimate()
    assert res_default.diagnostic_data is not None

    res_off = IRM(data, ml_g=ml_g, ml_m=ml_m, n_folds=3).fit(store_diagnostics=False).estimate()
    assert res_off.diagnostic_data is None


def test_fit_store_diagnostics_controls_atte_payload():
    data = _make_synth(n=150, seed=77)
    ml_g = LinearRegression()
    ml_m = LogisticRegression(max_iter=1000)

    res_default = IRM(data, ml_g=ml_g, ml_m=ml_m, n_folds=3).fit().estimate(score="ATTE")
    assert res_default.diagnostic_data is not None

    res_off = IRM(data, ml_g=ml_g, ml_m=ml_m, n_folds=3).fit(store_diagnostics=False).estimate(score="ATTE")
    assert res_off.diagnostic_data is None


def test_diagnostics_use_cached_x_without_reloading_dataframe(monkeypatch):
    data = _make_synth(n=120, seed=55)
    model = IRM(
        data,
        ml_g=LinearRegression(),
        ml_m=LogisticRegression(max_iter=1000),
        n_folds=3,
    ).fit()

    def _unexpected_get_df(*args, **kwargs):
        raise AssertionError("estimate() unexpectedly reloaded the dataframe")

    monkeypatch.setattr(CausalData, "get_df", _unexpected_get_df)
    result = model.estimate(score="ATE")

    assert result.diagnostic_data is not None
    np.testing.assert_allclose(result.diagnostic_data.x, model._X)


def test_repeated_estimate_calls_are_stable_with_reused_intermediates():
    data = _make_synth(n=160, seed=11)
    model = IRM(
        data,
        ml_g=LinearRegression(),
        ml_m=LogisticRegression(max_iter=1000),
        n_folds=3,
    ).fit()

    first = model.estimate(score="ATE")
    second = model.estimate(score="ATE")

    assert second.diagnostic_data is not None
    assert first.value == pytest.approx(second.value)
    assert first.model_options["std_error"] == pytest.approx(second.model_options["std_error"])
    assert first.ci_lower_absolute == pytest.approx(second.ci_lower_absolute)
    assert first.ci_upper_absolute == pytest.approx(second.ci_upper_absolute)


def test_lightweight_mode_skips_model_level_diagnostic_caches():
    data = _make_synth(n=140, seed=17)
    model = IRM(
        data,
        ml_g=LinearRegression(),
        ml_m=LogisticRegression(max_iter=1000),
        n_folds=3,
    ).fit(store_diagnostics=False)

    assert model._X is None
    assert model._y is not None
    assert model._d is not None
    assert model.m_hat_raw_ is None
    assert model.folds_ is None

    result = model.estimate(score="ATE")
    assert result.diagnostic_data is None
    assert np.isfinite(result.value)


def test_lightweight_mode_estimate_uses_cached_targets_without_reloading_dataframe(monkeypatch):
    data = _make_synth(n=130, seed=29)
    model = IRM(
        data,
        ml_g=LinearRegression(),
        ml_m=LogisticRegression(max_iter=1000),
        n_folds=3,
    ).fit(store_diagnostics=False)

    def _unexpected_get_df(*args, **kwargs):
        raise AssertionError("estimate() unexpectedly reloaded the dataframe")

    monkeypatch.setattr(CausalData, "get_df", _unexpected_get_df)
    result = model.estimate(score="ATE")

    assert result.diagnostic_data is None
    assert np.isfinite(result.value)


@pytest.mark.parametrize("score", ["ATE", "ATTE"])
def test_lightweight_mode_matches_full_mode_on_unchanged_data(score):
    data = _make_synth(n=180, seed=41)
    kwargs = dict(
        data=data,
        ml_g=LinearRegression(),
        ml_m=LogisticRegression(max_iter=1000),
        n_folds=3,
        random_state=13,
    )

    result_full = IRM(**kwargs).fit(store_diagnostics=True).estimate(score=score)
    result_light = IRM(**kwargs).fit(store_diagnostics=False).estimate(score=score)

    assert result_light.value == pytest.approx(result_full.value)


def test_lightweight_mode_estimate_is_stable_after_data_reordering():
    data = _make_synth(n=170, seed=1234)
    model = IRM(
        data,
        ml_g=LinearRegression(),
        ml_m=LogisticRegression(max_iter=1000),
        n_folds=3,
    ).fit(store_diagnostics=False)

    first = model.estimate(score="ATE")
    data.df = data.df.sample(frac=1.0, random_state=7).reset_index(drop=True)
    second = model.estimate(score="ATE")

    assert second.value == pytest.approx(first.value)
    assert second.model_options["std_error"] == pytest.approx(first.model_options["std_error"])


def test_lightweight_mode_legacy_reload_fallback_rejects_changed_data():
    data = _make_synth(n=170, seed=4321)
    model = IRM(
        data,
        ml_g=LinearRegression(),
        ml_m=LogisticRegression(max_iter=1000),
        n_folds=3,
    ).fit(store_diagnostics=False)

    model._y = None
    model._d = None
    data.df = data.df.sample(frac=1.0, random_state=19).reset_index(drop=True)

    with pytest.raises(RuntimeError, match="changed after fit"):
        model.estimate(score="ATE")
