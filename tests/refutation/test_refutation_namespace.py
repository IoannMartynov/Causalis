def test_refutation_namespace_exports():
    import causalis.scenarios.unconfoundedness.refutation as ref

    # Overlap exports
    assert hasattr(ref, "run_overlap_diagnostics")
    assert hasattr(ref, "plot_m_overlap")
    assert hasattr(ref, "plot_propensity_reliability")

    # Score diagnostics
    assert hasattr(ref, "run_score_diagnostics")
    assert hasattr(ref, "plot_influence_instability")
    assert hasattr(ref, "plot_residual_diagnostics")

    # Unconfoundedness balance plot
    assert hasattr(ref, "love_plot")
