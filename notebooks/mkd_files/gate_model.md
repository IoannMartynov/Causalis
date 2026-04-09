# GATE / GATET (from IRM)

This note documents the strict subgroup estimators implemented on top of a fitted IRM.
Sections 0-8 describe **GATE**. Section 9 adds the implemented **GATET**
variant, which targets treatment effects among treated units inside each
pre-treatment subgroup.

# 0) Assumptions

* **SUTVA / consistency:** observed outcomes correspond to the realized treatment, with no interference across units.
* **Unconfoundedness:** conditional on observed covariates $X$,
  $$
  (Y(1), Y(0)) \perp D \mid X.
  $$
* **Overlap / positivity:** for relevant covariate values and inside each estimable group,
  $$
  0 < \Pr(D=1 \mid X) < 1.
  $$
* Group membership must be **pre-treatment**. Groups defined by treatment itself, post-treatment outcomes, or post-treatment covariates are not valid causal subgroups.
* Cross-fitted nuisance models are assumed accurate enough that the orthogonal score behaves like a stable pseudo-outcome for subgroup averaging.

---

# 1) Data, notation, and estimand

For each observation $i=1,\dots,n$, observe
$$
W_i = (Y_i, D_i, X_i, G_i),
$$
where
* $Y_i$ is the outcome,
* $D_i \in \{0,1\}$ is treatment,
* $X_i$ are observed confounders,
* $G_i \in \{1,\dots,K\}$ is a pre-specified subgroup label.

The target is the **Group Average Treatment Effect**
$$
\theta_g = \mathbb{E}[Y(1)-Y(0) \mid G=g].
$$

Write the subgroup basis as
$$
B_{ig} = \mathbf{1}\{G_i = g\},
$$
and stack it into a matrix
$$
B \in \{0,1\}^{n \times K}.
$$

Because this implementation enforces a strict partition,
$$
\sum_{g=1}^K B_{ig} = 1
\qquad \text{for every } i.
$$

---

# 2) Start from the fitted IRM nuisances

The GATE estimator does **not** refit a separate causal model from scratch. Instead, it reuses the fitted IRM nuisance functions:
$$
\hat g_1(X) \approx \mathbb{E}[Y \mid X, D=1],
\qquad
\hat g_0(X) \approx \mathbb{E}[Y \mid X, D=0],
\qquad
\hat m(X) \approx \Pr(D=1 \mid X).
$$

These predictions are cross-fitted, meaning each observation receives nuisance predictions from models trained on other folds.

Define
$$
h_{1i} = \frac{D_i}{\hat m(X_i)},
\qquad
h_{0i} = \frac{1-D_i}{1-\hat m(X_i)}.
$$

The implementation then builds the canonical doubly robust orthogonal signal
$$
\hat\phi_i
= \hat g_1(X_i) - \hat g_0(X_i) + \bigl(Y_i - \hat g_1(X_i)\bigr) h_{1i} - \bigl(Y_i - \hat g_0(X_i)\bigr) h_{0i}.
$$

Under the IRM assumptions, this signal satisfies
$$
\mathbb{E}[\phi(W;\eta_0) \mid G=g] = \theta_g.
$$

So GATE can be estimated by averaging the orthogonal score within each subgroup.

## Important implementation note

Even if the IRM was fit with `normalize_ipw=True`, GATE intentionally ignores that option and uses the **canonical unnormalized** Horvitz-Thompson-style score above.

---

# 3) Convert user-supplied groups into a strict dummy basis

Let the user pass either:
* a single subgroup label column, or
* a full dummy basis.

## Case A: one-column subgroup labels

If `groups` has one column, convert labels into subgroup dummies:
$$
B_{ig} = \mathbf{1}\{G_i = g\}.
$$

## Case B: multi-column subgroup indicators

If `groups` already has $K$ columns, interpret it as a candidate basis matrix
$$
B = [B_1,\dots,B_K].
$$

Then verify:
$$
B_{ig} \in \{0,1\},
\qquad
\sum_{g=1}^K B_{ig} = 1 \ \forall i.
$$

This excludes overlapping subgroup definitions. Overlapping bases correspond to a more general BLP-style projection, not the strict GATE estimator implemented here.

## Alignment step

Before any estimation, align rows of `groups` to the **fit-time observation ids** used by the IRM.

```text
Input:
  fitted IRM
  groups indexed by ids or row index

1. Resolve fit-time observation ids from:
     - irm_model._fit_index_, if present
     - else data.user_id
     - else fail

2. If groups.index matches fit-time ids:
     use as-is

3. Else if groups.index is a permutation of fit-time ids:
     reindex to fit-time order

4. Else try row-index alignment:
     only allowed if the original fit-time row index still maps to the same user_id values

5. Otherwise:
     raise alignment error
```

---

# 4) GATE point estimation

Once the orthogonal signal $\hat\phi_i$ and subgroup basis $B$ are available, the estimator solves a saturated no-intercept linear projection:
$$
\hat\theta
=
\arg\min_{\beta \in \mathbb{R}^K}
\sum_{i=1}^n
\left(\hat\phi_i - \sum_{g=1}^K B_{ig}\beta_g\right)^2.
$$

Because the basis is a disjoint partition, the design is block-diagonal:
$$
B^\top B = \mathrm{diag}(n_1,\dots,n_K),
$$
where
$$
n_g = \sum_{i=1}^n B_{ig}.
$$

Therefore the estimator reduces to the groupwise sample mean:
$$
\hat\theta_g
=
\frac{1}{n_g}\sum_{i:B_{ig}=1}\hat\phi_i.
$$

Equivalently, if
$$
\hat\theta = (\hat\theta_1,\dots,\hat\theta_K)^\top,
$$
then
$$
\hat\theta = (B^\top B)^{-1} B^\top \hat\phi.
$$

## Pseudocode

```text
Input:
  fitted IRM with cross-fitted nuisances:
    g0_hat, g1_hat, m_hat
  groups

Output:
  one GATE estimate per subgroup

1. Validate:
     - IRM is fitted
     - alpha in (0,1)
     - cov_type in {HC0, HC1, HC2, HC3}
     - groups exist

2. Align groups to the fit-time observation ids.

3. Convert groups to a strict subgroup basis B.

4. Check support:
     for each group g:
       require at least one treated and one control unit

5. Compute orthogonal score for each observation i:
     phi_i =
       g1_hat_i - g0_hat_i
       + (Y_i - g1_hat_i) * D_i / m_hat_i
       - (Y_i - g0_hat_i) * (1-D_i) / (1-m_hat_i)

6. For each group g:
     mask_g   = {i : B_ig = 1}
     n_g      = number of observations in group g
     theta_g  = mean(phi_i over i in mask_g)

7. Return theta = (theta_1, ..., theta_K)
```

---

# 5) Closed-form HCx inference

For each group $g$, define residuals
$$
\hat u_i = \hat\phi_i - \hat\theta_g
\qquad \text{for } i \text{ such that } B_{ig}=1.
$$

Let the within-group residual sum of squares be
$$
\mathrm{SSE}_g = \sum_{i:B_{ig}=1} \hat u_i^2.
$$

Because the design is a partition, the covariance matrix is diagonal, so each group variance can be computed in closed form.

## HC0

$$
\widehat{\mathrm{Var}}_{HC0}(\hat\theta_g)
=
\frac{\mathrm{SSE}_g}{n_g^2}.
$$

## HC1

Let $K$ be the number of groups. Then
$$
\widehat{\mathrm{Var}}_{HC1}(\hat\theta_g)
=
\frac{n}{n-K}
\cdot
\frac{\mathrm{SSE}_g}{n_g^2},
$$
provided $n > K$. If $n \le K$, the implementation falls back to HC0 scaling.

## HC2

Since leverage within a no-intercept subgroup cell is
$$
h_{ii} = \frac{1}{n_g},
$$
HC2 becomes
$$
\widehat{\mathrm{Var}}_{HC2}(\hat\theta_g)
=
\frac{\mathrm{SSE}_g}{n_g^2 \left(1-\frac{1}{n_g}\right)}.
$$

## HC3

Similarly,
$$
\widehat{\mathrm{Var}}_{HC3}(\hat\theta_g)
=
\frac{\mathrm{SSE}_g}{n_g^2 \left(1-\frac{1}{n_g}\right)^2}.
$$

If $n_g=1$, these variance formulas are not estimable, so standard errors and interval-based inference are returned as `NaN`.

## Wald inference

For each subgroup $g$,
$$
\widehat{SE}(\hat\theta_g)
=
\sqrt{\widehat{\mathrm{Var}}(\hat\theta_g)},
$$
$$
z_g
=
\frac{\hat\theta_g}{\widehat{SE}(\hat\theta_g)}.
$$

The code uses normal-reference inference:
$$
p_g = 2 \Phi(-|z_g|),
$$
and
$$
\mathrm{CI}_{1-\alpha,g}
=
\left[
\hat\theta_g - z_{1-\alpha/2}\widehat{SE}(\hat\theta_g),
\hat\theta_g + z_{1-\alpha/2}\widehat{SE}(\hat\theta_g)
\right].
$$

---

# 6) Output object

The function returns a `GateEstimate` object containing one row per group.

Main fields:
* `value`: subgroup treatment effect estimate $\hat\theta_g$.
* `std_error`: robust HCx standard error.
* `wald_stat` / `test_stat`: subgroup-vs-zero Wald statistic.
* `p_value`: two-sided p-value for
  $$
  H_0:\theta_g = 0.
  $$
* `ci_lower`, `ci_upper`: $(1-\alpha)$ confidence interval.
* `is_significant`: indicator that `p_value < alpha`.

Support and overlap diagnostics:
* `n_group`: total observations in the subgroup.
* `n_treated`, `n_control`: within-group treatment counts.
* `share_treated`: empirical treatment share in the group.
* `mean_phi`, `std_phi`: mean and spread of the orthogonal signal within the group.
* `mean_propensity`, `min_propensity`, `max_propensity`: average and range of $\hat m(X)$ within the subgroup.

The object also stores:
* `covariance`: diagonal covariance matrix across group estimates.
* `summary_table`: table combining estimates and diagnostics.
* `diagnostic_data`: optional payload with the full orthogonal signal, aligned basis, and group warnings.

---

# 7) Interpretation of GATE results

## 7.1 What does `value` mean?

For subgroup $g$,
$$
\hat\theta_g \approx \mathbb{E}[Y(1)-Y(0)\mid G=g].
$$

So:
* if `value > 0`, treatment is estimated to increase the outcome in that subgroup;
* if `value < 0`, treatment is estimated to decrease the outcome in that subgroup;
* if `value \approx 0`, the estimated subgroup effect is small relative to the outcome scale.

This is a **within-group causal effect**, not just a descriptive difference in observed means.

## 7.2 What does the confidence interval tell you?

The interval
$$
[\mathrm{ci\_lower}_g,\ \mathrm{ci\_upper}_g]
$$
describes uncertainty around the subgroup effect estimate.

Practical reading:
* If the interval excludes $0$, the data provide evidence that the treatment effect in that subgroup is nonzero at level $\alpha$.
* If the interval is wide, the subgroup effect is estimated imprecisely, usually because the group is small, noisy, or has weak overlap.
* If the interval includes both substantively positive and negative values, the sign of the subgroup effect is not well resolved.

## 7.3 What `p_value` does and does not mean

The reported `p_value` tests
$$
H_0:\theta_g = 0.
$$

It does **not** test whether two groups differ from one another.

For example, suppose:
* group A has a significant positive effect,
* group B has a non-significant effect.

That alone does **not** imply
$$
\theta_A \ne \theta_B.
$$

To test subgroup heterogeneity directly, use formal contrasts such as:
```text
gate.contrast("group_A", "group_B")
gate.pairwise_summary()
```

## 7.4 How to interpret support diagnostics

The subgroup effect is only as credible as the support behind it.

Read these columns together:
* `n_group`: very small groups are unstable.
* `n_treated`, `n_control`: both must be positive; otherwise the subgroup is not identified here.
* `share_treated`: very extreme treatment shares suggest weak within-group overlap.
* `min_propensity`, `max_propensity`: values close to $0$ or $1$ indicate practical positivity problems.

In this implementation:
* a group is small (`n_group < 10`),
* a group has extreme estimated propensity support,
* a group with no treated or no control observations is rejected before estimation.

## 7.5 How to interpret differences across groups

If two groups have different `value` estimates, that is evidence of possible treatment-effect heterogeneity, but it should be interpreted carefully.

Useful rule:
* `value` answers: "What is the treatment effect inside this subgroup?"
* `contrast(...)` answers: "Is the treatment effect different between subgroup A and subgroup B?"

So subgroup heterogeneity claims should be based on **contrasts**, not on visual comparison of point estimates alone.

## 7.6 What can go wrong in interpretation

Common mistakes:
* Treating post-treatment group definitions as causal subgroups.
* Interpreting subgroup-vs-zero significance as proof of group-vs-group difference.
* Ignoring extreme propensity scores inside a subgroup.
* Over-interpreting very small groups with unstable standard errors.
* Reading GATE as if it were a conditional effect for every value of $X$; here it is an average effect over a **coarse partition**.

---

# 8) Compact end-to-end pseudocode

```text
Input:
  fitted IRM
  subgroup definition groups
  alpha
  cov_type ∈ {HC0, HC1, HC2, HC3}

1. Validate fitted IRM and GATE arguments.

2. Align groups to fit-time observation ids.

3. Convert groups into strict dummy basis B.
   Require:
     - one active group per row
     - no empty groups

4. Compute cross-fitted orthogonal signal:
     phi_i =
       g1_hat_i - g0_hat_i
       + (Y_i - g1_hat_i) * D_i / m_hat_i
       - (Y_i - g0_hat_i) * (1-D_i) / (1-m_hat_i)

5. For each group g:
     n_g         = sum_i B_ig
     theta_hat_g = (1/n_g) * sum_i B_ig * phi_i
     residual_i  = phi_i - theta_hat_g   for i in group g
     SSE_g       = sum residual_i^2
     var_g       = HCx formula(SSE_g, n_g, n, K)
     se_g        = sqrt(var_g)
     z_g         = theta_hat_g / se_g
     p_g         = 2 * Phi(-|z_g|)
     CI_g        = theta_hat_g ± z_(1-alpha/2) * se_g

6. Build summary table with:
     estimate + inference + support diagnostics

7. Return GateEstimate
```

---

# 9) GATET (group average treatment effect on the treated)

GATET uses the **same subgroup alignment and strict partition logic** as GATE,
but it changes the target estimand, the orthogonal score, and the support
requirements.

## 9.1 Target estimand

For subgroup $g$,
$$
\theta_g^{\mathrm{GATET}}
=
\mathbb{E}[Y(1)-Y(0)\mid G=g,\ D=1].
$$

So GATET answers:
"among the treated units inside subgroup $g$, what is the average causal
effect?"

When the groups form an exhaustive partition, ATTE is the treated-share mixture
of subgroup GATETs:
$$
\mathrm{ATTE}
=
\sum_g \Pr(G=g\mid D=1)\,\theta_g^{\mathrm{GATET}}.
$$

Empirically, this is the weighted average of subgroup estimates using
within-sample treated shares:
$$
\widehat{\mathrm{ATTE}}
=
\sum_g
\frac{n_{\mathrm{treated},g}}{\sum_h n_{\mathrm{treated},h}}
\hat\theta_g^{\mathrm{GATET}}.
$$

## 9.2 Orthogonal signal used by the implementation

Unlike ordinary GATE, GATET does **not** reuse the ATE-style score
$$
\hat\phi_i = \hat g_1(X_i) - \hat g_0(X_i) + \bigl(Y_i - \hat g_1(X_i)\bigr)\frac{D_i}{\hat m(X_i)}- \bigl(Y_i - \hat g_0(X_i)\bigr)\frac{1-D_i}{1-\hat m(X_i)}.
$$

Instead it builds the canonical ATT-style subgroup signal
$$
z_i
=
D_i\bigl(Y_i-\hat g_0(X_i)\bigr) - \hat m(X_i)(1-D_i)\frac{Y_i-\hat g_0(X_i)}{1-\hat m(X_i)}.
$$

Only $\hat g_0(X)$ and $\hat m(X)$ enter this score. As with GATE,
`normalize_ipw=True` on the fitted IRM is intentionally ignored so the
estimator uses the canonical unnormalized orthogonal signal above.

This matters because, in general,
$$
\hat\theta_g^{\mathrm{GATET}}
\ne
\frac{1}{n_{\mathrm{treated},g}}
\sum_{i:G_i=g,\ D_i=1}\hat\phi_i^{\mathrm{GATE}}.
$$

So GATET is **not** "GATE restricted to treated rows." It is a different
orthogonal score matched to the subgroup ATT estimand.

## 9.3 Point estimation

Let
$$
n_{\mathrm{treated},g}
=
\sum_{i=1}^n \mathbf{1}\{G_i=g\}D_i.
$$

Then the implementation estimates subgroup ATT as
$$
\hat\theta_g^{\mathrm{GATET}}
=
\frac{1}{n_{\mathrm{treated},g}}
\sum_{i:G_i=g} z_i.
$$

Equivalently, it solves the groupwise moment condition
$$
\sum_{i:G_i=g} \bigl(z_i - D_i\theta_g\bigr) = 0.
$$

## 9.4 Closed-form HCx inference

Define the subgroup residual
$$
u_{ig}
=
\mathbf{1}\{G_i=g\}\bigl(z_i - D_i\hat\theta_g^{\mathrm{GATET}}\bigr).
$$

Let
$$
U_g
=
\sum_{i:G_i=g}\bigl(z_i - D_i\hat\theta_g^{\mathrm{GATET}}\bigr)^2.
$$

Write $n_g$ for total subgroup size and $K$ for the number of groups. Then the
closed-form robust variances implemented in code are:

**HC0**
$$
\widehat{\mathrm{Var}}_{HC0}(\hat\theta_g)
=
\frac{U_g}{n_{\mathrm{treated},g}^2}.
$$

**HC1**
$$
\widehat{\mathrm{Var}}_{HC1}(\hat\theta_g)
=
\frac{n}{n-K}
\cdot
\frac{U_g}{n_{\mathrm{treated},g}^2},
$$
with fallback to HC0 scaling if $n \le K$.

**HC2**
$$
\widehat{\mathrm{Var}}_{HC2}(\hat\theta_g)
=
\frac{U_g}{n_{\mathrm{treated},g}^2\left(1-\frac{1}{n_g}\right)}.
$$

**HC3**
$$
\widehat{\mathrm{Var}}_{HC3}(\hat\theta_g)
=
\frac{U_g}{n_{\mathrm{treated},g}^2\left(1-\frac{1}{n_g}\right)^2}.
$$

The same normal-reference Wald inference is then used:
$$
z_g=\frac{\hat\theta_g}{\widehat{SE}(\hat\theta_g)},
\qquad
p_g = 2\Phi(-|z_g|),
$$
with confidence interval
$$
\hat\theta_g \pm z_{1-\alpha/2}\widehat{SE}(\hat\theta_g).
$$

## 9.5 Support rules

GATET uses the same strict partition requirement as GATE:
every observation must belong to exactly one subgroup.

But the support checks are different:

* Every GATET group must contain **at least one treated observation**.
* A group with **zero treated units** is rejected.
* A group with **zero control units** is still accepted, but the code emits a
  warning because within-group overlap is degenerate and identification relies
  on $\hat g_0(X)$ and $\hat m(X)$ learned outside that subgroup.
* If a group has only one total observation (`n_group = 1`), standard errors
  and interval-based inference are returned as `NaN`.

## 9.6 Output and diagnostics

The return type is still `GateEstimate`, but with
`estimand="GATET"`. The same helper surface is available:
```text
irm.estimate(score="GATET", groups=groups)
irm.gatet(groups=groups)
gatet.summary()
gatet.contrast("group_A", "group_B")
gatet.pairwise_summary()
```

Interpretation of the main columns:
* `value`: estimated treatment effect among treated units in the subgroup.
* `n_treated`: treated support actually identifying that subgroup ATT.
* `n_control`: controls available in the subgroup; may be zero for GATET.
* `share_treated`: empirical treated share in the subgroup.

When diagnostics are stored, the payload includes:
* `orthogonal_signal`: the transformed subgroup signal used for diagnostics,
  whose within-group mean equals the reported GATET estimate.
* `raw_treated_signal`: the raw ATT-style score $z_i$ before subgroup scaling.

So the practical distinction is:
* **GATE:** average treatment effect within subgroup $g$.
* **GATET:** average treatment effect among the treated units within subgroup
  $g$.
