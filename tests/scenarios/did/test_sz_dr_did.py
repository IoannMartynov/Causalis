import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize
from scipy.special import expit

from causalis.data_contracts import PanelDIDEstimate, PanelDataDID
from causalis.scenarios.did import SantAnnaZhaoDID


def _panel_from_units(units, *, covariates=None, cluster_col=None) -> PanelDataDID:
    rows = []
    pre = pd.Period("2020-01", freq="M")
    post = pd.Period("2020-02", freq="M")
    for unit in units:
        base = {
            "unit_id": unit["unit_id"],
            "x": unit.get("x", 0.0),
            "z": unit.get("z", 0.0),
        }
        if cluster_col is not None:
            base[cluster_col] = unit.get(cluster_col)
        rows.append(
            {
                **base,
                "time_id": pre,
                "y": unit["y_pre"],
                "treated_time": 0,
            }
        )
        rows.append(
            {
                **base,
                "time_id": post,
                "y": unit["y_post"],
                "treated_time": int(unit["treated"]),
            }
        )

    return PanelDataDID(
        df=pd.DataFrame(rows),
        y="y",
        unit_col="unit_id",
        time_col="time_id",
        treated_time="treated_time",
        covariates=[] if covariates is None else covariates,
        cluster_col=cluster_col,
    )


def _canonical_no_cov_panel() -> PanelDataDID:
    return _panel_from_units(
        [
            {"unit_id": "T1", "treated": True, "y_pre": 10.0, "y_post": 12.0},
            {"unit_id": "C1", "treated": False, "y_pre": 8.0, "y_post": 9.0},
        ],
    )


def _covariate_panel(*, cluster_col=None) -> PanelDataDID:
    units = [
        {"unit_id": "T1", "treated": True, "x": -1.0, "z": 0.5, "y_pre": 1.0, "y_post": 3.4, "cluster": "A"},
        {"unit_id": "T2", "treated": True, "x": 0.5, "z": 2.5, "y_pre": 1.5, "y_post": 3.6, "cluster": "A"},
        {"unit_id": "T3", "treated": True, "x": 2.0, "z": 3.0, "y_pre": 2.0, "y_post": 4.9, "cluster": "B"},
        {"unit_id": "C1", "treated": False, "x": -2.0, "z": 1.0, "y_pre": 0.5, "y_post": 0.8, "cluster": "B"},
        {"unit_id": "C2", "treated": False, "x": -0.5, "z": 3.0, "y_pre": 0.7, "y_post": 1.1, "cluster": "C"},
        {"unit_id": "C3", "treated": False, "x": 0.2, "z": 1.5, "y_pre": 1.1, "y_post": 1.7, "cluster": "C"},
        {"unit_id": "C4", "treated": False, "x": 1.5, "z": 2.5, "y_pre": 1.4, "y_post": 2.2, "cluster": "D"},
        {"unit_id": "C5", "treated": False, "x": 3.0, "z": 2.0, "y_pre": 1.9, "y_post": 2.9, "cluster": "D"},
    ]
    return _panel_from_units(units, covariates=["x", "z"], cluster_col=cluster_col)


def _solve_logit_gamma(x: np.ndarray, d: np.ndarray) -> np.ndarray:
    def objective(gamma):
        eta = x @ gamma
        return float(np.mean(np.logaddexp(0.0, eta) - d * eta))

    def gradient(gamma):
        eta = x @ gamma
        return x.T @ (expit(eta) - d) / x.shape[0]

    res = minimize(objective, np.zeros(x.shape[1]), jac=gradient, method="BFGS")
    assert res.success
    return np.asarray(res.x, dtype=float)


def test_no_covariate_2x2_matches_manual_did_and_returns_panel_did_estimate():
    panel = _canonical_no_cov_panel()

    model = SantAnnaZhaoDID().fit(panel)
    estimate = model.estimate()

    assert model.is_fitted is True
    assert isinstance(estimate, PanelDIDEstimate)
    assert estimate.estimand == "ATT"
    assert estimate.model == "SantAnnaZhaoImprovedPanelDRDID"
    assert estimate.att == pytest.approx(1.0)
    assert estimate.value == pytest.approx(1.0)
    assert estimate.n_treated == 1
    assert estimate.n_control == 1
    assert estimate.diagnostic_data is not None
    assert estimate.diagnostic_data.propensity_score.tolist() == pytest.approx([0.5, 0.5])
    assert estimate.summary().loc["value", "value"] == (
        f"{estimate.att:.4f} (ci_abs: {estimate.ci_lower:.4f}, {estimate.ci_upper:.4f})"
    )


def test_ipt_first_order_condition_holds_and_differs_from_logit_mle():
    panel = _covariate_panel()

    estimate = SantAnnaZhaoDID().fit(panel).estimate()
    diag = estimate.diagnostic_data
    x = np.column_stack([np.ones(len(diag.d)), diag.x])
    d = diag.d

    gamma_ipt = diag.gamma_hat
    ipt_score = x.T @ ((1.0 - d) * np.exp(x @ gamma_ipt) - d) / x.shape[0]
    assert np.max(np.abs(ipt_score)) < 1e-6

    gamma_logit = _solve_logit_gamma(x, d)
    e_logit = expit(x @ gamma_logit)
    assert not np.allclose(diag.propensity_score, e_logit, atol=1e-3, rtol=1e-3)


def test_wls_control_outcome_evolution_matches_manual_weighted_least_squares():
    panel = _covariate_panel()

    estimate = SantAnnaZhaoDID().fit(panel).estimate()
    diag = estimate.diagnostic_data
    x = np.column_stack([np.ones(len(diag.d)), diag.x])
    control = diag.d == 0.0
    weights = diag.propensity_score[control] / (1.0 - diag.propensity_score[control])
    sqrt_w = np.sqrt(weights)

    beta_manual, *_ = np.linalg.lstsq(
        x[control] * sqrt_w[:, None],
        diag.delta_y[control] * sqrt_w,
        rcond=None,
    )

    assert diag.beta_hat == pytest.approx(beta_manual)
    assert diag.control_outcome_evolution == pytest.approx(x @ beta_manual)


def test_clustered_inference_matches_cluster_score_formula():
    panel = _covariate_panel(cluster_col="cluster")

    estimate = SantAnnaZhaoDID().fit(panel).estimate()
    diag = estimate.diagnostic_data
    psi = diag.influence_scores
    cluster_scores = diag.cluster_scores
    n = len(psi)
    g = len(cluster_scores)
    expected_var = g / (g - 1) * np.sum((cluster_scores - cluster_scores.mean()) ** 2) / (n ** 2)

    assert estimate.inference == "clustered_influence"
    assert estimate.se == pytest.approx(np.sqrt(expected_var))


def test_non_canonical_panels_are_rejected():
    rows = []
    periods = pd.period_range("2020-01", periods=3, freq="M")
    for unit, treated_unit in [("T1", True), ("C1", False)]:
        for idx, period in enumerate(periods):
            rows.append(
                {
                    "unit_id": unit,
                    "time_id": period,
                    "y": float(idx + int(treated_unit)),
                    "treated_time": int(treated_unit and period >= pd.Period("2020-02", freq="M")),
                }
            )
    panel = PanelDataDID(
        df=pd.DataFrame(rows),
        y="y",
        unit_col="unit_id",
        time_col="time_id",
        treated_time="treated_time",
    )

    with pytest.raises(ValueError, match="canonical 2x2"):
        SantAnnaZhaoDID().fit(panel)


def test_incomplete_unit_pairs_are_rejected():
    rows = [
        {"unit_id": "T1", "time_id": pd.Period("2020-01", freq="M"), "y": 1.0, "treated_time": 0},
        {"unit_id": "T1", "time_id": pd.Period("2020-02", freq="M"), "y": 3.0, "treated_time": 1},
        {"unit_id": "T2", "time_id": pd.Period("2020-02", freq="M"), "y": 4.0, "treated_time": 1},
        {"unit_id": "C1", "time_id": pd.Period("2020-01", freq="M"), "y": 0.5, "treated_time": 0},
        {"unit_id": "C1", "time_id": pd.Period("2020-02", freq="M"), "y": 1.0, "treated_time": 0},
        {"unit_id": "C2", "time_id": pd.Period("2020-01", freq="M"), "y": 0.7, "treated_time": 0},
    ]
    panel = PanelDataDID(
        df=pd.DataFrame(rows),
        y="y",
        unit_col="unit_id",
        time_col="time_id",
        treated_time="treated_time",
    )

    with pytest.raises(ValueError, match="same units"):
        SantAnnaZhaoDID().fit(panel)


def test_unstable_cluster_column_is_rejected():
    panel = _canonical_no_cov_panel()
    panel_with_time_cluster = PanelDataDID(
        df=panel.df_analysis(),
        y="y",
        unit_col="unit_id",
        time_col="time_id",
        treated_time="treated_time",
        cluster_col="time_id",
    )

    with pytest.raises(ValueError, match="cluster_col must be stable"):
        SantAnnaZhaoDID().fit(panel_with_time_cluster)


def test_invalid_alpha_input_type_and_estimate_before_fit_are_rejected():
    with pytest.raises(ValueError, match="PanelDataDID"):
        SantAnnaZhaoDID().fit(pd.DataFrame())

    with pytest.raises(ValueError, match="alpha"):
        SantAnnaZhaoDID(alpha=1.0)

    with pytest.raises(RuntimeError, match="fit"):
        SantAnnaZhaoDID().estimate()


def test_estimate_overrides_alpha_and_diagnostic_payload():
    model = SantAnnaZhaoDID(alpha=0.2, diagnostic_data=True).fit(_canonical_no_cov_panel())

    estimate = model.estimate(alpha=0.1, diagnostic_data=False)

    assert estimate.alpha == pytest.approx(0.1)
    assert estimate.diagnostic_data is None
