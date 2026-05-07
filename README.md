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

**[Classic RCT](https://causalis.causalcraft.com/articles/classic_rct)**: randomized assignment (no pre-period metric).

**[CUPED](https://causalis.causalcraft.com/articles/cuped)**: randomized assignment with pre-period metric for variance reduction.

**[Unconfoundedness](https://causalis.causalcraft.com/articles/uncofoundedness)**: observational study adjusting for measured confounders (DML IRM).

**[GATE](https://causalis.causalcraft.com/articles/gate)**: Subgroup treatment effects built on top of an observational IRM workflow.

**[Multi Unconfoundedness](https://causalis.causalcraft.com/articles/multi_unconfoundedness)**: Multi Unconfoundedness extends observational identification to multiple treatment arms. We estimate causal contrasts across arms by adjusting for observed confounders and modeling generalized propensity scores.

**[Synthetic Control*](https://causalis.causalcraft.com/articles/synthetic_control)*: Single treated-unit panel setups matched against a weighted synthetic donor pool.

**[Difference in Difference*](https://causalis.causalcraft.com/articles/did)*: causal effects by comparing the changes in outcomes over time between a treatment group and a control group based on parallel trends

**[Uplift / CATE scoring](https://causalis.causalcraft.com/articles/uplift)**: Uplift modeling estimates the Conditional Average Treatment Effect (CATE) for individual units, enabling optimal targeting and personalized decision making.

[Introduction to Causal Inference](https://causalis.causalcraft.com/articles/introduction-to-causal-inference): guide

See scenario notebooks: https://causalis.causalcraft.com/explore-scenarios

# [Contributing guidelines](https://github.com/causalis-causalcraft/Causalis?tab=contributing-ov-file)

# Acknowledgements

https://github.com/DoubleML/doubleml-for-py
