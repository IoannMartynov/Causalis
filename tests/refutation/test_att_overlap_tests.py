import numpy as np

from causalis.scenarios.unconfoundedness.refutation import run_overlap_diagnostics
from tests.refutation._overlap_test_utils import make_overlap_data_and_estimate


def test_run_overlap_diagnostics_structure_and_basic_flags():
    rng = np.random.default_rng(123)
    n = 500
    m = np.clip(rng.uniform(0.05, 0.95, size=n), 1e-6, 1.0 - 1e-6)
    d = rng.binomial(1, m)
    data, estimate = make_overlap_data_and_estimate(
        m_hat=m,
        d=d,
        normalize_ipw=True,
        overlap_threshold=0.05,
    )

    out = run_overlap_diagnostics(data, estimate)

    assert isinstance(out, dict)
    assert "edge_mass" in out
    assert "ks" in out
    assert "auc" in out
    assert "ate_ess" in out
    assert "flags" in out
    assert out["flags"]["auc"] in {"GREEN", "YELLOW", "RED", "NA"}


def test_run_overlap_diagnostics_marks_flip_for_att_style_anti_prediction():
    n = 200
    d = np.array([1, 0] * (n // 2))
    m = np.clip(1.0 - d, 1e-6, 1.0 - 1e-6)
    data, estimate = make_overlap_data_and_estimate(m_hat=m, d=d)

    out = run_overlap_diagnostics(data, estimate)

    assert out["flags"]["auc"] == "RED"
    assert out["flags"].get("auc_flip_suspected") == "YELLOW"
