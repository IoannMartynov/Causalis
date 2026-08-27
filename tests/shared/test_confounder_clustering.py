from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from causalis.dgp.causaldata import CausalData
from causalis.shared import cluster_confounders, rank_confounder_clusters


def _make_data(*, n: int = 2_000, seed: int = 42, n_features: int = 3) -> CausalData:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = -x1 + rng.normal(scale=0.02, size=n)
    x3 = rng.normal(size=n)
    d = rng.binomial(1, 1.0 / (1.0 + np.exp(-0.4 * x1)))
    y = d + x1 + 0.2 * x3 + rng.normal(size=n)
    df = pd.DataFrame({"y": y, "d": d, "x1": x1, "x2": x2, "x3": x3})
    confounders = [f"x{i + 1}" for i in range(n_features)]
    for feature in confounders[3:]:
        df[feature] = rng.normal(size=n)
    return CausalData(
        df=df,
        treatment="d",
        outcome="y",
        confounders=confounders,
    )


def _make_feature_importance(
    *,
    m: list[float],
    g0: list[float],
    g1: list[float],
) -> dict:
    n_features = len(m)
    return {
        "method": "native",
        "feature_names": [f"x{i + 1}" for i in range(n_features)],
        "n_features": n_features,
        "nuisances": {
            key: {
                "available": True,
                "mean": np.asarray(values, dtype=float),
                "std": np.zeros(n_features),
                "n_folds": 3,
            }
            for key, values in {"m": m, "g0": g0, "g1": g1}.items()
        },
    }


def test_cluster_confounders_uses_absolute_correlation_and_keeps_order():
    data = _make_data()

    clusters = cluster_confounders(
        data,
        min_abs_correlation=0.95,
        max_samples=None,
    )

    assert clusters == [["x1", "x2"], ["x3"]]


def test_cluster_confounders_sampling_is_deterministic():
    data = _make_data(n=5_000, seed=7)

    first = cluster_confounders(data, max_samples=250, random_state=19)
    second = cluster_confounders(data, max_samples=250, random_state=19)

    assert first == second


@pytest.mark.parametrize("correlation_method", ["pearson", "spearman"])
@pytest.mark.parametrize("linkage_method", ["average", "complete", "single"])
def test_cluster_confounders_supports_documented_methods(
    correlation_method: str,
    linkage_method: str,
):
    data = _make_data(seed=10)

    clusters = cluster_confounders(
        data,
        correlation_method=correlation_method,  # type: ignore[arg-type]
        linkage_method=linkage_method,  # type: ignore[arg-type]
        max_samples=500,
    )

    assert sorted(feature for group in clusters for feature in group) == [
        "x1",
        "x2",
        "x3",
    ]


def test_cluster_confounders_validates_configuration():
    data = _make_data()

    with pytest.raises(ValueError, match="min_abs_correlation"):
        cluster_confounders(data, min_abs_correlation=1.1)
    with pytest.raises(ValueError, match="correlation_method"):
        cluster_confounders(data, correlation_method="kendall")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="linkage_method"):
        cluster_confounders(data, linkage_method="ward")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least 2"):
        cluster_confounders(data, max_samples=1)
    with pytest.raises(TypeError, match="integer or None"):
        cluster_confounders(data, max_samples=100.5)  # type: ignore[arg-type]


def test_cluster_confounders_keeps_single_feature_cluster():
    data = _make_data()
    single = CausalData(
        df=data.df,
        treatment="d",
        outcome="y",
        confounders=["x3"],
    )

    assert cluster_confounders(single) == [["x3"]]


@pytest.mark.parametrize("ranking_method", ["mixed", "joint"])
def test_rank_confounder_clusters_sorts_by_joint_importance(ranking_method):
    data = _make_data()
    feature_importance = _make_feature_importance(
        m=[0.3, 0.3, 0.4],
        g0=[0.3, 0.3, 0.4],
        g1=[0.3, 0.3, 0.4],
    )
    estimate = SimpleNamespace(
        diagnostic_data=SimpleNamespace(feature_importance=feature_importance)
    )

    ranking = rank_confounder_clusters(
        data,
        estimate,
        min_abs_correlation=0.95,
        max_samples=None,
        ranking_method=ranking_method,
    )

    assert list(ranking.columns) == [
        "cluster_id",
        "features",
        "n_features",
        "importance_d",
        "importance_g0",
        "importance_g1",
        "importance_y",
        "score",
    ]
    assert ranking.loc[0, "features"] == ["x1", "x2"]
    assert ranking.loc[1, "features"] == ["x3"]
    assert float(ranking.loc[0, "importance_d"]) == pytest.approx(0.6)
    assert float(ranking.loc[0, "importance_y"]) == pytest.approx(0.6)
    assert float(ranking.loc[0, "score"]) == pytest.approx(0.36)


@pytest.mark.parametrize(
    "m, g, joint_order, mixed_order",
    [
        pytest.param(
            [1, 1, 3, 4, 11, 1],
            [5, 5, 6, 3, 1, 1],
            [0, 1, 2, 3, 4],
            [0, 1, 3, 2, 4],
            id="promote-treatment-cluster",
        ),
        pytest.param(
            [10, 10, 15, 4, 11, 1],
            [5, 5, 6, 3, 1, 1],
            [0, 1, 2, 3, 4],
            [0, 1, 3, 2, 4],
            id="strongest-treatment-already-selected",
        ),
        pytest.param(
            [1, 1, 3, 4, 6, 6],
            [10, 10, 12, 4, 1, 2],
            [0, 1, 2, 4, 3],
            [0, 1, 4, 2, 3],
            id="treatment-tie-prefers-joint-score",
        ),
        pytest.param(
            [1, 1, 3, 4, 6, 6],
            [10, 10, 12, 4, 1, 1],
            [0, 1, 2, 3, 4],
            [0, 1, 3, 2, 4],
            id="treatment-and-score-tie-prefers-original-order",
        ),
    ],
)
def test_rank_confounder_clusters_mixed_default(m, g, joint_order, mixed_order):
    data = _make_data(n_features=len(m))
    importance = _make_feature_importance(m=m, g0=g, g1=g)

    mixed = rank_confounder_clusters(data, importance, min_abs_correlation=0.95)
    joint = rank_confounder_clusters(
        data, importance, min_abs_correlation=0.95, ranking_method="joint"
    )

    assert joint["cluster_id"].tolist() == joint_order
    assert mixed["cluster_id"].tolist() == mixed_order
    assert mixed.index.tolist() == list(range(len(mixed)))
    assert mixed["cluster_id"].is_unique
    assert joint["score"].is_monotonic_decreasing
    pd.testing.assert_frame_equal(
        mixed.sort_values("cluster_id").reset_index(drop=True),
        joint.sort_values("cluster_id").reset_index(drop=True),
    )


@pytest.mark.parametrize("n_features", [1, 2, 3, 4])
def test_rank_confounder_clusters_mixed_keeps_small_rankings(n_features):
    data = _make_data(n_features=n_features)
    importance = _make_feature_importance(
        m=list(range(1, n_features + 1)),
        g0=[1.0] * n_features,
        g1=[1.0] * n_features,
    )

    mixed = rank_confounder_clusters(data, importance, min_abs_correlation=0.95)
    joint = rank_confounder_clusters(
        data, importance, min_abs_correlation=0.95, ranking_method="joint"
    )

    assert len(mixed) == max(1, n_features - 1)
    pd.testing.assert_frame_equal(mixed, joint)


def test_rank_confounder_clusters_validates_ranking_method():
    with pytest.raises(ValueError, match="ranking_method must be 'mixed' or 'joint'"):
        rank_confounder_clusters(
            _make_data(), SimpleNamespace(), ranking_method="treatment"
        )


def test_rank_confounder_clusters_supports_model_and_max_outcome_importance():
    data = _make_data(seed=9)
    feature_importance = _make_feature_importance(
        m=[0.2, 0.2, 0.6],
        g0=[0.45, 0.45, 0.1],
        g1=[0.05, 0.05, 0.9],
    )
    model = SimpleNamespace(feature_importance_=feature_importance)

    ranking = rank_confounder_clusters(
        data,
        model,
        min_abs_correlation=0.95,
        max_samples=None,
        outcome_importance="max",
    )

    grouped = ranking.loc[ranking["features"].map(len).eq(2)].iloc[0]
    assert float(grouped["importance_g0"]) == pytest.approx(0.9)
    assert float(grouped["importance_g1"]) == pytest.approx(0.1)
    assert float(grouped["importance_y"]) == pytest.approx(0.9)


def test_rank_confounder_clusters_validates_importance_payload():
    data = _make_data(seed=11)

    with pytest.raises(ValueError, match="Feature importance is unavailable"):
        rank_confounder_clusters(data, SimpleNamespace())

    mismatched = _make_feature_importance(
        m=[0.3, 0.3, 0.4],
        g0=[0.3, 0.3, 0.4],
        g1=[0.3, 0.3, 0.4],
    )
    mismatched["feature_names"] = ["x1", "x2", "other"]
    with pytest.raises(ValueError, match="must match data.confounders"):
        rank_confounder_clusters(data, mismatched)

    with pytest.raises(ValueError, match="outcome_importance"):
        rank_confounder_clusters(
            data,
            _make_feature_importance(
                m=[0.3, 0.3, 0.4],
                g0=[0.3, 0.3, 0.4],
                g1=[0.3, 0.3, 0.4],
            ),
            outcome_importance="sum",  # type: ignore[arg-type]
        )
