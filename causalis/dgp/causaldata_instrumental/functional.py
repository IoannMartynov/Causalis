"""
Instrumental-variable DGP helpers.

This module currently exposes a placeholder IV generator so downstream code can
import a stable public entry point while the richer IV DGP is still under
construction.

Notes
-----
At the moment ``generate_iv_data()`` returns an empty ``DataFrame`` because the
underlying :class:`InstrumentalGenerator` is a stub. The docstring is kept
explicit so users do not mistake the current behavior for a full IV benchmark.

Examples
--------
>>> from causalis.dgp.causaldata_instrumental.functional import generate_iv_data
>>> df = generate_iv_data(n=100)
>>> df.empty
True
"""

from __future__ import annotations
import pandas as pd
from .base import InstrumentalGenerator

def generate_iv_data(n: int = 1000) -> pd.DataFrame:
    """
    Generate synthetic dataset with instrumental variables.

    Placeholder implementation.

    Parameters
    ----------
    n : int, default=1000
        Number of samples to generate.

    Returns
    -------
    pandas.DataFrame
        Synthetic IV dataset.

    Notes
    -----
    This function is intentionally a placeholder. It returns the current output
    of :class:`InstrumentalGenerator`, which is an empty ``DataFrame`` until the
    structural IV generator is implemented.

    Examples
    --------
    >>> from causalis.dgp.causaldata_instrumental.functional import generate_iv_data
    >>> generate_iv_data(10).shape
    (0, 0)
    """
    gen = InstrumentalGenerator()
    return gen.generate(n)
