import numpy as np

from causalis.scenarios.multi_unconfoundedness.dgp import generate_multi_dml_cx_26


def test_multi_dml_cx_26_is_one_hot_and_returns_expected_metadata():
    data = generate_multi_dml_cx_26(
        n=4000,
        seed=314,
        include_oracle=False,
        return_causal_data=True,
    )

    assert data.treatment_names == ["control", "neg_contact_flg", "error_flg", "neg_contact_flg_error_flg"]
    assert data.control_treatment == "control"
    assert data.outcome == "y"

    treatment_frame = data.df[data.treatment_names]
    assert treatment_frame.sum(axis=1).eq(1.0).all()
    assert all(treatment_frame[col].sum() > 0 for col in data.treatment_names)


def test_multi_dml_cx_26_oracle_effects_match_notebook_story():
    df = generate_multi_dml_cx_26(
        n=4000,
        seed=2718,
        include_oracle=True,
        return_causal_data=False,
    )

    required = {
        "g_control",
        "g_neg_contact_flg",
        "g_error_flg",
        "g_neg_contact_flg_error_flg",
        "cate_neg_contact_flg",
        "cate_error_flg",
        "cate_neg_contact_flg_error_flg",
        "tau_link_control",
        "tau_link_neg_contact_flg",
        "tau_link_error_flg",
        "tau_link_neg_contact_flg_error_flg",
    }
    assert required.issubset(df.columns)

    cate_contact = df["cate_neg_contact_flg"].to_numpy(dtype=float)
    cate_repeat = df["cate_error_flg"].to_numpy(dtype=float)
    cate_both = df["cate_neg_contact_flg_error_flg"].to_numpy(dtype=float)

    assert np.allclose(cate_contact, 0.0, atol=1e-12, rtol=0.0)
    assert np.all(cate_repeat < 0.0)
    assert np.all(cate_both < 0.0)
    assert np.allclose(cate_repeat, cate_both, atol=1e-12, rtol=0.0)
