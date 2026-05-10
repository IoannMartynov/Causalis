import numpy as np

from causalis.data_contracts import IVCausalData
from causalis.dgp import InstrumentalGenerator, generate_iv_data
from causalis.dgp.causaldata_instrumental import IVCausalDatasetGenerator


def test_instrumental_generator_produces_binary_iv_dataset():
    gen = InstrumentalGenerator(
        k=2,
        theta=1.2,
        first_stage=1.5,
        beta_y=np.array([0.4, -0.2]),
        beta_d=np.array([0.6, 0.3]),
        beta_z=np.array([0.2, -0.1]),
        target_z_rate=0.45,
        target_d_rate=0.55,
        u_strength_d=0.7,
        u_strength_y=0.5,
        seed=3141,
    )

    df = gen.generate(1_500)

    assert {"y", "d", "z", "x1", "x2"}.issubset(df.columns)
    assert {"m", "r_z0", "r_z1", "g_z0", "g_z1", "late"}.issubset(df.columns)
    assert df["d"].isin([0.0, 1.0]).all()
    assert df["z"].isin([0.0, 1.0]).all()
    assert abs(df["z"].mean() - 0.45) < 0.08
    assert abs(df["d"].mean() - 0.55) < 0.08
    assert df["iv_first_stage"].mean() > 0.1
    assert np.isfinite(df["late"]).all()


def test_generate_iv_data_returns_iv_causal_data_contract():
    data = generate_iv_data(
        n=800,
        k=2,
        random_state=123,
        return_causal_data=True,
        include_oracle=True,
    )

    assert isinstance(data, IVCausalData)
    assert data.instruments == ["z"]
    assert data.confounders == ["x1", "x2"]
    assert list(data.df.columns) == ["y", "d", "z", "x1", "x2"]
    assert set(data.Z.columns) == {"z"}


def test_iv_generator_alias_and_no_oracle_path():
    assert IVCausalDatasetGenerator is InstrumentalGenerator

    df = generate_iv_data(
        n=300,
        k=1,
        random_state=456,
        include_oracle=False,
        return_causal_data=False,
    )

    assert list(df.columns) == ["y", "d", "z", "x1"]
    assert df["d"].nunique() == 2
    assert df["z"].nunique() == 2
