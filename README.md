# Causalis
[![PyPI version](https://img.shields.io/pypi/v/causalis.svg)](https://pypi.org/project/causalis/)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/causalis?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/causalis)
![Python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12%20|%203.13%20|%203.14-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Code quality](https://img.shields.io/badge/code%20quality-A-brightgreen)
[![Docs](https://img.shields.io/badge/docs-causalis.causalcraft.com-blue)](https://causalis.causalcraft.com/)

<a href="https://causalis.causalcraft.com/"><img src="https://raw.githubusercontent.com/causalis-causalcraft/Causalis/main/notebooks/new_logo_big.svg" alt="Causalis logo" width="80" style="float: left; margin-right: 10px;" /></a>

Robust causal inference for experiments and observational studies in Python, organized around **scenarios** (e.g., Classic RCT, CUPED, Unconfoundedness) with a consistent `fit() → estimate()` workflow.

- 📚 Documentation & notebooks: https://causalis.causalcraft.com/
- 🔎 API reference: https://causalis.causalcraft.com/api-reference

## Why Causalis?
Causalis focuses on:
- Scenario-first workflows (you pick the study design; Causalis provides best-practice defaults).
- Extensive robustness tests that reveal issues in the study design or model specification
- Pydantic data contracts 
- An advanced DGP (Data Generating Process) with heterogeneous treatment effects, latent variables, and correlated confounders
- A website with notebooks based on real-world cases

## Installation
### Recommended
```bash
pip install causalis
```

# Quickstart: Classic RCT (difference in means + inference)

```python
from causalis.dgp import generate_classic_rct_26
from causalis.scenarios.classic_rct import DiffInMeans, check_srm

# Synthetic RCT data as a validated CausalData object
data = generate_classic_rct_26(seed=42, return_causal_data=True)

# Optional: Sample Ratio Mismatch check
srm = check_srm(data, target_allocation={0: 0.5, 1: 0.5}, alpha=1e-3)
print("SRM detected?", srm.is_srm, "p=", srm.p_value, "chi2=", srm.chi2)

# Estimate treatment effect with t-test inference (or bootstrap / conversion_ztest)
result = DiffInMeans().fit(data).estimate(method="ttest", alpha=0.05)
result.summary()
```
# Quickstart: Observational study (Unconfoundedness / DML IRM)
```python
from causalis.scenarios.unconfoundedness.dgp import generate_obs_hte_26
from causalis.scenarios.unconfoundedness import IRM
from causalis.data_contracts import CausalData

causaldata = generate_obs_hte_26(return_causal_data=True, include_oracle=False)

from causalis.scenarios.unconfoundedness import IRM

model = IRM().fit(causaldata)
result = model.estimate(score='ATTE')
result.summary()
```

# Pick your scenario

| Scenario                                                                                   | Estimator                                     | Assumptions                                                                                                                     |
|--------------------------------------------------------------------------------------------|-----------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------|
| [Classic RCT](https://causalis.causalcraft.com/articles/classic_rct)                       | Difference in means (ttest, ztest, bootstrap) | Random assignment, no sample ratio mismatch, SUTVA                                                                              |
| [CUPED](https://causalis.causalcraft.com/articles/cuped)                                   | CUPED-adjusted difference in means            | Random assignment, no sample ratio mismatch, SUTVA, valid pre-period metrics                                                    |
| [Unconfoundedness](https://causalis.causalcraft.com/articles/unconfoundedness)             | DML IRM                                       | Unconfoundedness, Overlap, SUTVA, No leakage, Score stability                                                                   |
| [GATE](https://causalis.causalcraft.com/articles/gate)                                     | DML IRM (GATE and GATET)                      | Same assumptions as unconfoundedness, plus meaningful pre-specified or validated subgroup definitions.                          |
| [Multi Unconfoundedness](https://causalis.causalcraft.com/articles/multi_unconfoundedness) | Multi DML IRM                                 | Unconfoundedness, Multi class Overlap, SUTVA, No leakage, Score stability                                                       |
| [Synthetic Control](https://causalis.causalcraft.com/articles/synthetic_control)           | ASCM                                          | No interference / spillovers, No anticipation, The treated unit’s untreated outcome path is well approximated by the donor pool |
| [Difference in Difference](https://causalis.causalcraft.com/articles/did)                  | CallawaySantAnnaDID                           | Parallel trends, no anticipation, stable group composition, no spillovers between treated and control groups.                   |
| [Uplift / CATE scoring](https://causalis.causalcraft.com/articles/uplift)                  | DML IRM (CATE)                                | Identified treatment effects from randomized or unconfounded data, overlap, calibrated individual-level predictions.            |

[Introduction to Causal Inference](https://causalis.causalcraft.com/articles/introduction-to-causal-inference): guide

See scenario notebooks: https://causalis.causalcraft.com/explore-scenarios

# [Contributing guidelines](https://github.com/causalis-causalcraft/Causalis?tab=contributing-ov-file)

# Maintainers

[Ioann Martynov](https://www.linkedin.com/in/ioannmartynov/)

# References

https://github.com/DoubleML/doubleml-for-py

## Search terms / supported methods

Causalis covers methods often searched as:

- causal inference Python
- causal machine learning Python
- treatment effect estimation
- A/B testing Python
- randomized controlled trial analysis
- CUPED Python
- Double Machine Learning Python
- DML / IRM
- CATE estimation
- uplift modeling
- propensity score diagnostics
- synthetic control Python
- difference-in-differences Python