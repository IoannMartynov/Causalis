import matplotlib
import numpy as np
import pandas as pd
import pytest

from sklearn.linear_model import LinearRegression, LogisticRegression

matplotlib.use("Agg")

from causalis.data_contracts.gate_estimate import GateEstimate
from causalis.dgp.causaldata import CausalData
from causalis.scenarios.gate.gate_plot import gate_plot
from causalis.scenarios.gate.model import (
    _coerce_groups_to_partition,
    _compute_gate_signal_from_irm,
    _compute_gatet_signal_from_irm,
    _estimate_gatet_groupwise_summary_from_partition,
    estimate_gatet_from_irm,
)
from causalis.scenarios.unconfoundedness.model import IRM


def _make_synthetic_data(n: int = 500, seed: int = 7) -> tuple[CausalData, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    p = 1.0 / (1.0 + np.exp(-(0.6 * x1 - 0.4 * x2)))
    d = rng.binomial(1, p)
    tau = 1.2 + 0.5 * (x1 > 0.0).astype(float)
    y = 1.0 + 0.4 * x1 - 0.2 * x2 + tau * d + rng.normal(scale=1.0, size=n)
    user_id = pd.Index([f"u_{i:04d}" for i in range(n)], name="user_id")

    df = pd.DataFrame({"user_id": user_id, "y": y, "d": d, "x1": x1, "x2": x2})
    cd = CausalData(df=df, treatment="d", outcome="y", confounders=["x1", "x2"], user_id="user_id")
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
    return pd.Series(
        np.where(df["x1"] >= 0.0, "high_x1", "low_x1"),
        index=pd.Index(df["user_id"], name="user_id"),
        name="segment",
    )


def _groups_from_row_index(df: pd.DataFrame) -> pd.Series:
    return pd.Series(np.where(df["x1"] >= 0.0, "high_x1", "low_x1"), index=df.index, name="segment")


def _quantile_groups_from_df(df: pd.DataFrame, q: int = 4) -> pd.Series:
    groups = pd.qcut(df["x1"], q=q, duplicates="drop").rename("x1_bin")
    groups.index = pd.Index(df["user_id"], name="user_id")
    return groups


def _manual_gatet_stats(
    *,
    z: np.ndarray,
    d: np.ndarray,
    codes: np.ndarray,
    k: int,
    cov_type: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_obs = z.shape[0]
    n_group = np.bincount(codes, minlength=k).astype(int)
    n_treated = np.rint(np.bincount(codes, weights=np.asarray(d, dtype=float), minlength=k)).astype(int)
    values = np.bincount(codes, weights=z, minlength=k) / n_treated

    variances = np.full(k, np.nan, dtype=float)
    estimable_mask = n_group > 1
    d_float = np.asarray(d, dtype=float)
    residual = z - d_float * values[codes]
    sum_u2 = np.bincount(codes, weights=np.square(residual), minlength=k)
    hc0 = np.full(k, np.nan, dtype=float)
    hc0[estimable_mask] = sum_u2[estimable_mask] / np.square(n_treated[estimable_mask].astype(float))

    if cov_type == "HC0":
        variances[estimable_mask] = hc0[estimable_mask]
    elif cov_type == "HC1":
        variances[estimable_mask] = hc0[estimable_mask] * (n_obs / (n_obs - k))
    elif cov_type == "HC2":
        hcx_mask = estimable_mask & (n_treated > 1)
        obs_mask = hcx_mask[codes]
        adjusted_u2 = np.zeros(n_obs, dtype=float)
        leverage_denom = 1.0 - (d_float[obs_mask] / n_treated[codes[obs_mask]].astype(float))
        adjusted_u2[obs_mask] = np.square(residual[obs_mask]) / leverage_denom
        sum_adjusted_u2 = np.bincount(codes, weights=adjusted_u2, minlength=k)
        variances[hcx_mask] = sum_adjusted_u2[hcx_mask] / np.square(n_treated[hcx_mask].astype(float))
    elif cov_type == "HC3":
        hcx_mask = estimable_mask & (n_treated > 1)
        obs_mask = hcx_mask[codes]
        adjusted_u2 = np.zeros(n_obs, dtype=float)
        leverage_denom = 1.0 - (d_float[obs_mask] / n_treated[codes[obs_mask]].astype(float))
        adjusted_u2[obs_mask] = np.square(residual[obs_mask]) / np.square(leverage_denom)
        sum_adjusted_u2 = np.bincount(codes, weights=adjusted_u2, minlength=k)
        variances[hcx_mask] = sum_adjusted_u2[hcx_mask] / np.square(n_treated[hcx_mask].astype(float))
    else:
        raise ValueError(cov_type)

    share_treated = n_treated / n_group.astype(float)
    transformed_signal = z / share_treated[codes]
    return values, variances, transformed_signal, n_treated


def test_irm_estimate_gatet_dispatches_to_new_module(monkeypatch):
    cd, df = _make_synthetic_data(n=180, seed=1)
    irm = _fit_irm(cd, store_diagnostics=False)
    groups = _groups_from_df(df)
    marker = object()
    captured = {}

    def _fake_gatet_estimator(**kwargs):
        captured.update(kwargs)
        return marker

    monkeypatch.setattr("causalis.scenarios.unconfoundedness.model.estimate_gatet_from_irm", _fake_gatet_estimator)

    res = irm.estimate(
        score="GATET",
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


def test_gatet_accepts_same_group_input_forms_as_gate():
    cd, df = _make_synthetic_data(n=260, seed=2)
    irm = _fit_irm(cd)
    groups_series = _groups_from_df(df)
    groups_row = _groups_from_row_index(df)
    groups_df_1col = groups_series.to_frame()
    groups_dummy = pd.DataFrame(
        {
            "g_high": (groups_series == "high_x1").astype(int),
            "g_low": (groups_series == "low_x1").astype(int),
        },
        index=groups_series.index,
    )

    res_series = irm.estimate(score="GATET", groups=groups_series)
    res_row = irm.estimate(score="GATET", groups=groups_row)
    res_df = irm.estimate(score="GATET", groups=groups_df_1col)
    res_dummy = irm.estimate(score="GATET", groups=groups_dummy)

    assert isinstance(res_series, GateEstimate)
    assert isinstance(res_row, GateEstimate)
    assert isinstance(res_df, GateEstimate)
    assert isinstance(res_dummy, GateEstimate)
    assert sorted(res_series.group_names) == ["segment=high_x1", "segment=low_x1"]
    np.testing.assert_allclose(np.sort(res_series.values), np.sort(res_row.values), atol=1e-10)
    np.testing.assert_allclose(np.sort(res_series.values), np.sort(res_df.values), atol=1e-10)


def test_gatet_is_stable_to_group_reordering_and_data_reordering():
    cd, df = _make_synthetic_data(n=240, seed=22)
    groups = _groups_from_df(df)
    irm = _fit_irm(cd, store_diagnostics=False)

    first = irm.estimate(score="GATET", groups=groups)
    shuffled_groups = groups.sample(frac=1.0, random_state=41)
    second = irm.estimate(score="GATET", groups=shuffled_groups)
    np.testing.assert_allclose(second.values, first.values, atol=1e-12)
    np.testing.assert_allclose(second.std_errors, first.std_errors, atol=1e-12)

    cd.df = cd.df.sample(frac=1.0, random_state=31).reset_index(drop=True)
    third = irm.estimate(score="GATET", groups=groups)
    np.testing.assert_allclose(third.values, first.values, atol=1e-12)
    np.testing.assert_allclose(third.std_errors, first.std_errors, atol=1e-12)


def test_gatet_groups_fallback_uses_causaldata_attribute():
    cd, df = _make_synthetic_data(n=260, seed=6)
    groups = _groups_from_df(df)
    object.__setattr__(cd, "gate_groups", groups)
    irm = _fit_irm(cd)

    res = irm.estimate(score="GATET")
    assert isinstance(res, GateEstimate)
    assert len(res.group_names) == 2


@pytest.mark.parametrize("cov_type", ["HC0", "HC1", "HC2", "HC3"])
def test_gatet_matches_manual_groupwise_formula(cov_type):
    cd, df = _make_synthetic_data(n=320, seed=4)
    irm = _fit_irm(cd, random_state=19)
    groups = _quantile_groups_from_df(df, q=4)

    res = irm.estimate(score="GATET", groups=groups, cov_type=cov_type)
    z, d, _ = _compute_gatet_signal_from_irm(irm)
    partition = _coerce_groups_to_partition(groups, n_obs=z.shape[0])
    manual_values, manual_variances, transformed_signal, _ = _manual_gatet_stats(
        z=z,
        d=d,
        codes=partition.codes,
        k=len(partition.group_names),
        cov_type=cov_type,
    )

    np.testing.assert_allclose(res.values, manual_values, atol=1e-10)
    np.testing.assert_allclose(np.square(res.std_errors), manual_variances, atol=1e-10, equal_nan=True)
    np.testing.assert_allclose(
        np.diag(res.covariance.loc[res.group_names, res.group_names].to_numpy(dtype=float)),
        manual_variances,
        atol=1e-10,
        equal_nan=True,
    )

    group_codes = partition.codes
    for group_idx, group_name in enumerate(res.group_names):
        mask = group_codes == group_idx
        assert res.mean_phi[group_idx] == pytest.approx(float(np.mean(transformed_signal[mask])))
        if np.sum(mask) > 1:
            assert res.std_phi[group_idx] == pytest.approx(float(np.std(transformed_signal[mask], ddof=1)))
        else:
            assert np.isnan(res.std_phi[group_idx])
        assert group_name in partition.group_names


def test_gatet_hc3_uses_treated_leverage_not_group_size():
    z = np.asarray([2.0, 4.0, -1.0, 1.0, 3.0, 6.0, 9.0, -3.0])
    d = np.asarray([1, 1, 0, 0, 1, 1, 1, 0], dtype=float)
    m_hat = np.full(z.shape, 0.5, dtype=float)
    groups = pd.Series(["a", "a", "a", "a", "b", "b", "b", "b"], name="segment")
    partition = _coerce_groups_to_partition(groups, n_obs=z.shape[0])

    stats = _estimate_gatet_groupwise_summary_from_partition(
        z=z,
        d=d,
        m_hat=m_hat,
        partition=partition,
        cov_type="HC3",
        alpha=0.05,
    )

    np.testing.assert_array_equal(stats["n_group"], np.asarray([4, 4]))
    np.testing.assert_array_equal(stats["n_treated"], np.asarray([2, 3]))
    np.testing.assert_allclose(stats["values"], np.asarray([3.0, 5.0]))
    np.testing.assert_allclose(np.square(stats["std_errors"]), np.asarray([2.5, 6.25]))

    old_group_leverage_variances = np.asarray([16.0 / 9.0, 160.0 / 27.0])
    assert not np.allclose(np.square(stats["std_errors"]), old_group_leverage_variances)


def test_gatet_hc3_sets_one_treated_group_inference_to_nan():
    z = np.asarray([2.0, 4.0, -1.0, 1.0, 5.0, -2.0, 1.0, 3.0])
    d = np.asarray([1, 1, 0, 0, 1, 0, 0, 0], dtype=float)
    m_hat = np.full(z.shape, 0.5, dtype=float)
    groups = pd.Series(["a", "a", "a", "a", "b", "b", "b", "b"], name="segment")
    partition = _coerce_groups_to_partition(groups, n_obs=z.shape[0])

    with pytest.warns(RuntimeWarning, match="only one treated observation"):
        stats = _estimate_gatet_groupwise_summary_from_partition(
            z=z,
            d=d,
            m_hat=m_hat,
            partition=partition,
            cov_type="HC3",
            alpha=0.05,
        )

    one_treated_idx = stats["group_names"].index("segment=b")
    assert stats["n_treated"][one_treated_idx] == 1
    assert np.isfinite(stats["values"][one_treated_idx])
    assert np.isnan(stats["std_errors"][one_treated_idx])
    assert np.isnan(stats["p_values"][one_treated_idx])
    assert np.isnan(stats["ci_lower"][one_treated_idx])
    assert np.isnan(stats["ci_upper"][one_treated_idx])


def test_gatet_ignores_normalize_ipw_and_cov_kwds():
    cd, df = _make_synthetic_data(n=320, seed=3)
    irm = _fit_irm(cd, normalize_ipw=True)
    groups = _groups_from_df(df)

    with pytest.warns(RuntimeWarning) as recorded:
        res = irm.estimate(score="GATET", groups=groups, cov_type="HC3", cov_kwds={"use_correction": True})

    messages = [str(w.message) for w in recorded]
    assert any("ignored for GATET" in msg for msg in messages)
    assert any("cov_kwds are ignored for GATET" in msg for msg in messages)
    assert res.model_options["normalize_ipw_requested"] is True
    assert res.model_options["normalize_ipw_effective"] is False
    assert res.model_options["cov_kwds"] == {}
    assert res.model_options["cov_kwds_requested"] == {"use_correction": True}


def test_gatet_payload_follows_fit_store_diagnostics_setting():
    cd, df = _make_synthetic_data(n=220, seed=14)
    groups = _groups_from_df(df)

    irm_diag = _fit_irm(cd, store_diagnostics=True)
    irm_light = _fit_irm(cd, store_diagnostics=False)

    diagnostic_payload = irm_diag.estimate(score="GATET", groups=groups).diagnostic_data
    assert diagnostic_payload is not None
    assert "group_codes" in diagnostic_payload
    assert "raw_treated_signal" in diagnostic_payload
    assert "basis" not in diagnostic_payload
    assert irm_light.estimate(score="GATET", groups=groups).diagnostic_data is None


def test_gatet_allows_treated_only_group_with_warning():
    cd, df = _make_synthetic_data(n=260, seed=18)
    irm = _fit_irm(cd, random_state=29)
    groups = pd.Series(
        np.where((df["d"] == 1) & (df["x1"] > 0.0), "treated_only", "mixed"),
        index=pd.Index(df["user_id"], name="user_id"),
        name="segment",
    )

    with pytest.warns(RuntimeWarning, match="have no control observations"):
        res = irm.estimate(score="GATET", groups=groups, cov_type="HC2")

    treated_only_idx = res.group_names.index("segment=treated_only")
    assert res.n_control[treated_only_idx] == 0
    assert res.n_treated[treated_only_idx] > 0
    assert np.isfinite(res.values[treated_only_idx])


def test_gatet_rejects_groups_without_treated_support():
    cd, df = _make_synthetic_data(n=240, seed=15)
    irm = _fit_irm(cd, random_state=33)
    groups = pd.Series(
        np.where((df["d"] == 0) & (df["x1"] > 0.0), "control_only", "mixed"),
        index=pd.Index(df["user_id"], name="user_id"),
        name="segment",
    )

    with pytest.raises(ValueError, match="at least one treated observation"):
        irm.estimate(score="GATET", groups=groups, cov_type="HC3")


def test_atte_is_treated_share_mixture_of_gatet():
    cd, df = _make_synthetic_data(n=320, seed=41)
    irm = _fit_irm(cd, random_state=44)
    groups = _quantile_groups_from_df(df, q=4)

    atte = irm.estimate(score="ATTE")
    gatet = irm.estimate(score="GATET", groups=groups, cov_type="HC2")

    weights = gatet.n_treated.astype(float) / float(np.sum(gatet.n_treated))
    assert float(np.dot(weights, gatet.values)) == pytest.approx(float(atte.value), abs=1e-10)


def test_gatet_differs_from_naive_gate_phi_mean_over_treated():
    cd, df = _make_synthetic_data(n=320, seed=42)
    irm = _fit_irm(cd, random_state=45)
    groups = _quantile_groups_from_df(df, q=4)

    gate_phi, d, _ = _compute_gate_signal_from_irm(irm)
    gatet = irm.estimate(score="GATET", groups=groups, cov_type="HC3")
    partition = _coerce_groups_to_partition(groups, n_obs=gate_phi.shape[0])

    naive = []
    for group_idx in range(len(partition.group_names)):
        mask = partition.codes == group_idx
        naive.append(float(np.mean(gate_phi[mask & (d == 1)])))
    naive = np.asarray(naive, dtype=float)

    assert np.max(np.abs(gatet.values - naive)) > 1e-3


def test_gatet_result_surface_is_estimand_aware():
    cd, df = _make_synthetic_data(n=320, seed=43)
    irm = _fit_irm(cd, random_state=46)
    groups = _quantile_groups_from_df(df, q=4)

    res = irm.gatet(groups=groups, alpha=0.1, cov_type="HC2")
    assert isinstance(res, GateEstimate)
    assert res.estimand == "GATET"

    contrast = res.contrast(res.group_names[0], res.group_names[1])
    assert contrast.estimand == "GATET_CONTRAST"

    pairwise = res.pairwise_summary()
    assert pairwise.shape[0] == len(res.group_names) * (len(res.group_names) - 1) // 2

    fig = gate_plot(res)
    assert fig.axes[0].get_title() == "GATET Estimates and Confidence Intervals"


def test_estimate_gatet_from_irm_export_matches_irm_method():
    cd, df = _make_synthetic_data(n=260, seed=52)
    irm = _fit_irm(cd, random_state=18)
    groups = _groups_from_df(df)

    direct = estimate_gatet_from_irm(irm, groups=groups, cov_type="HC1")
    via_method = irm.gatet(groups=groups, cov_type="HC1")

    np.testing.assert_allclose(direct.values, via_method.values, atol=1e-12)
    np.testing.assert_allclose(direct.std_errors, via_method.std_errors, atol=1e-12)
