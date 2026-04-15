# Causalis
![Python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12%20|%203.13%20|%203.14-blue)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/causalis?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/causalis)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Code quality](https://img.shields.io/badge/code%20quality-A-brightgreen)

<a href="https://causalis.causalcraft.com/"><img src="https://raw.githubusercontent.com/causalis-causalcraft/Causalis/main/notebooks/new_logo_big.svg" alt="Causalis logo" width="80" style="float: left; margin-right: 10px;" /></a>

Robust causal inference for experiments and observational studies in Python, organized around **scenarios** (e.g., Classic RCT, CUPED, Unconfoundedness) with a consistent `fit() → estimate()` workflow.

- 📚 Documentation & notebooks: https://causalis.causalcraft.com/
- 🔎 API reference: https://causalis.causalcraft.com/api-reference

## Why Causalis?
Causalis focuses on:
- Scenario-first workflows (you pick the study design; Causalis provides best-practice defaults).
- Guardrails and diagnostics (e.g., SRM checks, balance checks).
- Typed data contracts (`CausalData`) to fail fast on schema issues. 

## Installation
### Recommended
```bash
pip install causalis
```

## Building A Production Release

Use the helper below when you want `dist/*` to contain a stable release instead of a `setuptools_scm` dev build:

```bash
./scripts/build_release.sh 0.3.1
twine upload dist/*
```

The script removes older `dist/causalis-*` artifacts before building and, by default, stops if the checkout has tracked changes. It uses the current environment's `python`, or `PYTHON_BIN=/path/to/python` if you want to force a specific interpreter.

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

Classic RCT: randomized assignment (no pre-period metric).

CUPED: randomized assignment with pre-period metric for variance reduction.

Unconfoundedness: observational study adjusting for measured confounders (DML IRM).

See scenario notebooks: https://causalis.causalcraft.com/explore-scenarios

# Responsible use / limitations

Causal estimates require identification assumptions (e.g., randomization or unconfoundedness + overlap).
Causalis can help with diagnostics, but it cannot guarantee assumptions hold in your data.

# Contributing

Contributions are welcome—bug reports, docs fixes, notebooks, and new estimators.
Please read CONTRIBUTING.md and follow the Code of Conduct.

# Getting help

Questions: GitHub Discussions

Bugs: GitHub Issues (include minimal repro + versions)


# License

MIT (see LICENSE).

# Acknowledgements

https://github.com/DoubleML/doubleml-for-py

