import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from causalis.dgp.causaldata import CausalData
from causalis.scenarios.unconfoundedness.model import IRM
from causalis.scenarios.unconfoundedness.refutation.overlap import (
    plot_feature_importance,
)


def _make_data(n: int = 240, seed: int = 123) -> CausalData:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    logits = 1.2 * x[:, 0] - 0.8 * x[:, 1] + 0.2 * x[:, 2]
    p = 1.0 / (1.0 + np.exp(-logits))
    d = rng.binomial(1, p)
    y = 1.5 * d + 2.0 * x[:, 0] - 0.7 * x[:, 1] + rng.normal(0.0, 0.25, n)
    df = pd.DataFrame(
        {
            "y": y,
            "d": d,
            "x1": x[:, 0],
            "x2": x[:, 1],
            "x3": x[:, 2],
        }
    )
    return CausalData(
        df=df,
        treatment="d",
        outcome="y",
        confounders=["x1", "x2", "x3"],
    )


def _assert_native_payload(feature_importance: dict) -> None:
    assert feature_importance["method"] == "native"
    assert feature_importance["feature_names"] == ["x1", "x2", "x3"]
    assert feature_importance["n_features"] == 3
    assert set(feature_importance["nuisances"]) == {"m", "g0", "g1"}

    for key in ("m", "g0", "g1"):
        payload = feature_importance["nuisances"][key]
        assert payload["available"] is True
        assert payload["n_folds"] == 3
        assert payload["mean"].shape == (3,)
        assert payload["std"].shape == (3,)
        assert np.all(np.isfinite(payload["mean"]))
        assert np.all(payload["mean"] >= 0.0)
        assert np.isclose(float(np.sum(payload["mean"])), 1.0)


def test_feature_importance_is_collected_with_default_diagnostics():
    data = _make_data()
    model = IRM(
        data,
        ml_g=RandomForestRegressor(n_estimators=10, random_state=1),
        ml_m=RandomForestClassifier(n_estimators=10, random_state=1),
        n_folds=3,
        random_state=1,
    )

    estimate = model.fit().estimate(score="ATE")

    _assert_native_payload(model.feature_importance_)
    _assert_native_payload(estimate.diagnostic_data.feature_importance)


def test_feature_importance_payload_uses_tree_native_importances():
    data = _make_data()
    model = IRM(
        data,
        ml_g=RandomForestRegressor(n_estimators=20, random_state=2),
        ml_m=RandomForestClassifier(n_estimators=20, random_state=2),
        n_folds=3,
        random_state=2,
    )

    estimate = model.fit().estimate(score="ATE")

    _assert_native_payload(estimate.diagnostic_data.feature_importance)


def test_feature_importance_payload_uses_coef_native_importances():
    data = _make_data()
    model = IRM(
        data,
        ml_g=LinearRegression(),
        ml_m=LogisticRegression(max_iter=1000),
        n_folds=3,
        random_state=3,
    )

    estimate = model.fit().estimate(score="ATE")

    _assert_native_payload(estimate.diagnostic_data.feature_importance)


def test_plot_feature_importance_returns_figure():
    data = _make_data()
    model = IRM(
        data,
        ml_g=RandomForestRegressor(n_estimators=10, random_state=4),
        ml_m=RandomForestClassifier(n_estimators=10, random_state=4),
        n_folds=3,
        random_state=4,
    )
    estimate = model.fit().estimate(score="ATE")

    fig = plot_feature_importance(estimate, top_k=2)

    assert isinstance(fig, Figure)


def test_plot_feature_importance_raises_without_diagnostic_payload():
    data = _make_data()
    model = IRM(
        data,
        ml_g=RandomForestRegressor(n_estimators=10, random_state=5),
        ml_m=RandomForestClassifier(n_estimators=10, random_state=5),
        n_folds=3,
        random_state=5,
    )
    estimate = model.fit(store_diagnostics=False).estimate(score="ATE")

    with pytest.raises(ValueError, match="Missing estimate.diagnostic_data"):
        plot_feature_importance(estimate)


def test_plot_feature_importance_raises_when_no_native_importance_available():
    data = _make_data()
    model = IRM(
        data,
        ml_g=KNeighborsRegressor(n_neighbors=5),
        ml_m=KNeighborsClassifier(n_neighbors=5),
        n_folds=3,
        random_state=6,
    )
    estimate = model.fit().estimate(score="ATE")

    with pytest.raises(ValueError, match="No native feature importance was available"):
        plot_feature_importance(estimate)


def test_feature_importance_is_skipped_when_diagnostics_are_disabled():
    data = _make_data()
    model = IRM(
        data,
        ml_g=RandomForestRegressor(n_estimators=10, random_state=7),
        ml_m=RandomForestClassifier(n_estimators=10, random_state=7),
        n_folds=3,
        random_state=7,
    )

    estimate = model.fit(store_diagnostics=False).estimate(score="ATE")

    assert model.feature_importance_ is None
    assert estimate.diagnostic_data is None
