  # 0) Assumptions  
  
## Implementation / observed-data requirements  
  
* Exactly one treated unit in the analysis sample.  
* At least 2 donor units in the donor pool.  
* Donors are never treated during the analysis horizon.  
* The treated unit is untreated in all pre-periods and treated in all post-periods.  
* There is at least one pre-treatment period and at least one post-treatment period.  
* Pre and post periods are disjoint, ordered, and satisfy  
  $$  
  \max(\text{pre}) < \min(\text{post}).  
  $$  
* The treated+donor analysis block is balanced over all analysis periods.  
* There are no duplicate ((unit,time)) rows.  
* There are no missing outcomes in the treated+donor analysis block used for estimation.  
  
## Identification / modeling assumptions  
  
* **No interference / spillovers:** donor outcomes are unaffected by the treated unit’s treatment.  
* **No anticipation:** treatment does not affect the treated unit in pre-periods.  
* The treated unit’s untreated outcome path is well approximated by the donor pool, at least in the pre-treatment period.  
* Good pre-treatment fit is informative about the treated unit’s untreated post-treatment path.  
* In ASCM, residual imbalance left by SCM is adequately captured by the augmentation model, allowing controlled extrapolation beyond the donor convex hull.  
* The untreated outcome relationship is stable from pre- to post-period absent treatment.  
  
## Inference assumptions  
  
* For CWZ-style pointwise conformal inference, the residual process under the null is sufficiently stable / approximately shift-invariant for circular time shifts to generate a valid or approximately valid reference distribution.  
---  
# ASCM pseudocode with math

```text
Input:
  PanelDataSCM with:
    - one treated unit
    - donor pool {1, ..., J}
    - pre-treatment periods pre = {1, ..., T0}
    - post-treatment periods post = {T0+1, ..., T0+T1}

  Hyperparameters:
    lambda_sc >= 0
    lambda_aug >= 0
    alpha in (0,1)
    average_att_n_folds >= 2
    enforce_sum_to_one_augmented in {True, False}
    optional conformal grid settings

Notation:
  Y1_pre  ∈ R^{T0}
  X0_pre  ∈ R^{T0 × J}
  Y1_all  ∈ R^{T0+T1}
  X0_all  ∈ R^{(T0+T1) × J}

  where:
    Y1_pre  = treated outcomes in pre-periods
    X0_pre  = donor outcomes in pre-periods
    Y1_all  = treated outcomes in all periods
    X0_all  = donor outcomes in all periods

Output:
  PanelEstimate with:
    - effect_by_time
    - optional ci_lower_by_time, ci_upper_by_time
    - optional p_value_by_time, is_significant_by_time
    - donor_weights_augmented
    - diagnostics
```

---

# 1) Build balanced panel matrices

```text
1.1 Extract treated unit and donor units from PanelDataSCM.

1.2 Extract ordered pre-times and post-times.
    Require:
      T0 = |pre| >= 1
      T1 = |post| >= 1
      max(pre) < min(post)

1.3 Build balanced treated + donor outcome block:

      Y1_all = (Y_{1t})_{t ∈ pre ∪ post}             ∈ R^{T0+T1}
      X0_all = (Y_{jt})_{t ∈ pre ∪ post, j=1..J}     ∈ R^{(T0+T1) × J}

1.4 Restrict to pre-period rows:

      Y1_pre = (Y_{1t})_{t ∈ pre}                    ∈ R^{T0}
      X0_pre = (Y_{jt})_{t ∈ pre, j=1..J}            ∈ R^{T0 × J}
```

---

# 2) Fit simplex SCM anchor ( w_{sc} )

We first estimate simplex-constrained SCM weights.

## Optimization problem

$$  
\hat w_{sc}
=
\arg\min_{w\in\mathbb{R}^J}  
|Y_1^{pre}-X_0^{pre}w|_2^2  
+  
\lambda_{sc}|w|_2^2  
$$

subject to

$$  
w_j \ge 0,  
\qquad  
\sum_{j=1}^{J} w_j = 1  
$$

Define

$$  
G_{sc} = (X_0^{pre})^\top X_0^{pre} + \lambda_{sc} I_J  
$$

$$  
b = (X_0^{pre})^\top Y_1^{pre}  
$$

Then the objective becomes

$$  
\hat w_{sc}
=
\arg\min_{w\in\Delta_J}  
\left(  
w^\top G_{sc} w - 2 b^\top w  
\right)  
$$

where

$$
\Delta_J =
\{\, w \in \mathbb{R}^J :
w_j \ge 0,\;
\mathbf{1}^\top w = 1 \,\}
$$

## Pseudocode

```text
2.1 Compute:
      G_sc = X0_pre' X0_pre + lambda_sc I_J
      b    = X0_pre' Y1_pre

2.2 Solve constrained optimization:

      w_sc = argmin_{w ∈ Delta_J} [ w' G_sc w - 2 b' w ]

2.3 Use SLSQP with:
      equality constraint: 1' w = 1
      bounds: 0 ≤ w_j ≤ 1

2.4 If SLSQP fails → projected gradient:

      initialize w^(0) = (1/J,...,1/J)

      gradient:
        grad f(w) = 2(G_sc w - b)
        
	  Lipschitz constant of grad:  
		L = 2 * lambda_max(G_sc)

      step size:
        eta = 1 / max(L, 1e-12)

      iterate:
        w^(m+1) = Π_{Delta_J}( w^(m) - eta * grad f(w^(m)) )
        
      until coverage

2.5 Return:
      w_sc
```

---

# 3) Fit augmented weights ( w_{aug} )

Now fit ridge-augmented weights centered around the simplex anchor.

## Optimization problem

$$  
\hat w_{aug}
=
\arg\min_{w\in\mathbb{R}^J}  
|Y_1^{pre}-X_0^{pre}w|_2^2  
+  
\lambda_{aug}|w-\hat w_{sc}|_2^2  
$$

First-order condition:

$$  
(X_0^{pre})^\top(X_0^{pre}) w  
+  
\lambda_{aug} w
=
(X_0^{pre})^\top Y_1^{pre}  
+  
\lambda_{aug}\hat w_{sc}  
$$

Define

$$  
G_{aug} = (X_0^{pre})^\top X_0^{pre} + \lambda_{aug} I  
$$

$$  
r = (X_0^{pre})^\top Y_1^{pre} + \lambda_{aug} \hat w_{sc}  
$$

Unconstrained solution:

$$  
\tilde w_{aug} = G_{aug}^{-1} r  
$$

---

## Optional sum-to-one correction

If enforcing

$$  
\mathbf{1}^\top w = 1  
$$

then

$$  
\hat w_{aug}
=
\tilde w_{aug}
-
G_{aug}^{-1}\mathbf{1}  
 
\frac{\mathbf{1}^\top \tilde w_{aug}-1}  
{\mathbf{1}^\top G_{aug}^{-1}\mathbf{1}}  
$$

Otherwise

$$  
\hat w_{aug} = \tilde w_{aug}  
$$

---

## Pseudocode

```text
3.1 Compute:
      G_aug = X0_pre' X0_pre + lambda_aug I_J
      r     = X0_pre' Y1_pre + lambda_aug w_sc

3.2 Solve linear system:
      w_aug_tilde = solve(G_aug, r)
    or fallback:
      w_aug_tilde = least_squares(G_aug, r)

3.3 If enforce_sum_to_one_augmented:
      q     = G_aug^{-1} 1
      denom = 1' q
      w_aug = w_aug_tilde - q * ((1' w_aug_tilde - 1) / denom)
    Else:
      w_aug = w_aug_tilde

3.4 Output:
      w_aug
```

---

# 4) Build synthetic path and dynamic treatment effects

Synthetic untreated outcome:

$$  
\hat Y_0^{all}
=
X_0^{all} \hat w_{aug}  
$$

Gap path:

$$  
\hat g^{all}
=
Y_1^{all} - \hat Y_0^{all}  
$$

Dynamic treatment effects:

$$  
\hat\tau_t
=
Y_{1t}
-
X_{0t}^\top \hat w_{aug},  
\qquad  
t \in post  
$$

```
4.1 Compute synthetic untreated path:
      Y0_hat_all = X0_all w_aug

4.2 Compute gap path:
      gap_all = Y1_all - Y0_hat_all

4.3 Restrict to post periods:
      effect_by_time = (gap_t)_{t in post}

4.4 Store:
      observed_outcome  = Y1_all
      synthetic_outcome = Y0_hat_all
      gap               = gap_all
      effect_by_time    = gap_all[post]
```

---

## 5) Default inference: average ATT t-test

This layer targets an aggregate post-treatment effect using fold-based bias correction.

---

### 5.1 Build holdout blocks

Let requested number of folds be (K_{req}). Define:  
$$ 
K=\min(K_{req}, T_0),  
\qquad  
r=\min\left(\left\lfloor\frac{T_0}{K}\right\rfloor, T_1\right)  
$$

Construct contiguous pre-treatment holdout blocks  
$$ 
B_k={{(k-1)r,\ldots,kr-1}},  
\qquad k=1,\ldots,K.  
$$

Each block has length (r).

---

### 5.2 Fold-specific refits

For each fold (k):

- training pre-periods:  
    $$ 
    pre_{train}^{(k)} = pre \setminus B_k  
    $$
    
- fit fold-specific weights:  
    $$  
    \hat w_{aug}^{(k)}  
    =  
    \arg\min_w  
    |Y_{1,train}^{pre}-X_{0,train}^{pre}w|_2^2
    

\lambda_{aug}|w-\hat w_{sc}^{(k)}|_2^2  
$$  
with the same anchor-then-augment procedure applied on the training subset.

- compute post mean gap:  
    $$  
    \bar g_{post}^{(k)}  
    =  
    \frac{1}{T_1}  
    \sum_{t\in post}  
    \left(Y_{1t}-X_{0t}^\top \hat w_{aug}^{(k)}\right)  
    $$
    
- compute holdout pre mean gap:  
    $$  
    \bar g_{hold}^{(k)}  
    =  
    \frac{1}{r}  
    \sum_{t\in B_k}  
    \left(Y_{1t}-X_{0t}^\top \hat w_{aug}^{(k)}\right)  
    $$
    
- fold ATT estimate:  
    $$  
    \hat\tau_k  
    =  
    \bar g_{post}^{(k)}-\bar g_{hold}^{(k)}  
    $$
    

# Aggregate across folds:  
$$  
\hat\tau

\frac{1}{K}\sum_{k=1}^K \hat\tau_k  
$$

---

### 5.3 Self-normalized t-statistic

# Compute fold sample standard deviation:  
$$  
s_K

\sqrt{  
\frac{1}{K-1}\sum_{k=1}^K(\hat\tau_k-\hat\tau)^2  
}  
$$

# Scale adjustment:  
$$ 
\hat\sigma_\tau

\sqrt{1+\frac{Kr}{T_1}} s_K.  
$$

# Estimated standard error:  
$$  
\widehat{SE}

\frac{\hat\sigma_\tau}{\sqrt K}  
$$

# Test statistic:  
$$  
t

\frac{\sqrt K,\hat\tau}{\hat\sigma_\tau}  
$$

# Two-sided p-value:  
$$  
p
=
2\Pr\left(|t_{K-1}|\ge |t|\right).  
$$

# Confidence interval:  
$$  
CI_{1-\alpha}
=
\left[  
\hat\tau - t_{1-\alpha/2,K-1}\widehat{SE},  
  
\hat\tau + t_{1-\alpha/2,K-1}\widehat{SE}  
\right].  
$$

### Pseudocode

```text
5.1 Build K contiguous pre holdout blocks B_1, ..., B_K of size r.

5.2 For each fold k = 1, ..., K:
      - define train_idx = pre \ B_k
      - refit anchor + augmented weights on train_idx:
            w_sc^(k), w_aug^(k)

      - compute post mean gap:
            gbar_post^(k)
            = (1 / T1) sum_{t in post} [ Y1_t - X0_t' w_aug^(k) ]

      - compute holdout pre mean gap:
            gbar_hold^(k)
            = (1 / r) sum_{t in B_k} [ Y1_t - X0_t' w_aug^(k) ]

      - fold estimate:
            tau_k = gbar_post^(k) - gbar_hold^(k)

5.3 Aggregate:
      tau = (1 / K) sum_{k=1}^K tau_k

5.4 Compute fold dispersion:
      s_K = sqrt( (1/(K-1)) sum_k (tau_k - tau)^2 )

5.5 Compute scale and standard error:
      sigma_hat = sqrt(1 + K*r/T1) * s_K
      se        = sigma_hat / sqrt(K)

5.6 Compute test statistic:
      t_stat = sqrt(K) * tau / sigma_hat

5.7 Compute p-value:
      p_value = 2 * StudentT_sf(|t_stat|, df=K-1)

5.8 Compute confidence interval:
      crit = StudentT_quantile(1 - alpha/2, df=K-1)
      CI   = [tau - crit * se, tau + crit * se]
```


---

## 6) Optional pointwise conformal inference for each post period

For each post period $t^\star \in post$, perform grid inversion of null hypotheses  
$$ 
H_0:\tau_{t^\star}=\theta.  
$$

---

### 6.1 Reduced sample for one target post period

For a fixed post period $t^\star$, keep only:

- all pre periods,
- the single target post period $t^\star$.

So the reduced sample size is  
$$  
T_0+1.  
$$

### For candidate null effect (\theta), modify the treated outcome at the target period:  
$$  
Y_{1,t^\star}^{(\theta)}
=
Y_{1,t^\star}-\theta.  
$$

### Let  
$$  
Y_1^{(\theta)}
=
\big(  
Y_{1,1},\ldots,Y_{1,T_0},Y_{1,t^\star}-\theta  
\big).  
$$

### Refit the augmented weights on the reduced sample:  
$$  
\hat w_{aug}^{(\theta)}
=
\arg\min_w  
|Y_1^{(\theta)}-X_0 w|_2^2  
+  
\lambda_{aug}|w-\hat w_{sc}^{(\theta)}|_2^2.  
$$

### Residual vector under the null:  
$$ 
u^{(\theta)}
=
Y_1^{(\theta)}-X_0 \hat w_{aug}^{(\theta)}  
\in \mathbb R^{T_0+1}.  
$$

---

### 6.2 CWZ statistic

Let the last coordinate correspond to the single reduced-sample post period.

# Define the CWZ statistic:  
$$  
S(u)
=
\frac{\left|\sum_{post}u_t\right|}{\sqrt{n_{post}}}.  
$$

Since here $n_{post}=1$,  
$$  
S(u)=|u_{post}|.  
$$

---

### 6.3 Circular-shift randomization p-value

Let $\pi_m$ denote the $m$-th circular shift of the residual vector, for  
$$  
m=1,\ldots,T_0+1.  
$$

# Compute  
$$
\hat p(\theta)
=
\frac{1}{T_0+1}
\sum_{m=1}^{T_0+1}
\mathbf{1}\!\left(
S(\pi_m u^{(\theta)})
\ge
S(u^{(\theta)}) - 10^{-12}
\right)
$$

This is the randomization p-value for the null effect $\theta$.

---

### 6.4 Invert the test over a grid

Choose a grid  
$$  
\Theta={\theta_1,\ldots,\theta_G}.  
$$

For each (\theta_g\in\Theta), compute (\hat p(\theta_g)).

### Accepted set:  
$$  
A_{t^\star}
=
{\theta\in\Theta:\hat p(\theta)>\alpha}.  
$$

### Convert accepted grid points into contiguous accepted segments:  
$$  
A_{t^\star}
=
\bigcup_{\ell=1}^{L_{t^\star}}  
[a_{\ell}, b_{\ell}].  
$$

### If there is exactly one accepted segment, define the pointwise CI as  
$$  
CI_{t^\star}
=
[a_1,b_1].  
$$

Otherwise keep the full accepted-set representation.

Pointwise p-value for no effect:  
$$  
p_{t^\star}=\hat p(0).  
$$

Pointwise significance indicator:  
$$  
\mathbf  { p_{t^\star}\le \alpha}  
$$

Minimum attainable p-value:  
$$  
p_{min}=\frac{1}{T_0+1}.  
$$

### Pseudocode

```text
6.1 For each post period t* in post:

      Let point estimate:
        tau_hat_t* = effect_by_time[t*]

6.2 Build grid Theta = {theta_1, ..., theta_G}

6.3 For each theta in Theta:
      - construct reduced sample using all pre periods + period t*
      - replace treated target outcome:
            y_t*^(theta) = y_t* - theta
      - refit anchor + augmented weights on reduced sample
      - compute residual vector:
            u(theta) = y^(theta) - X w_aug^(theta)
      - compute statistic:
            S(u(theta)) = |sum(post residuals)| / sqrt(n_post)
        here n_post = 1, so:
            S(u(theta)) = |u_post(theta)|
      - compute circular-shift p-value:
            p(theta)
            = (1/(T0+1)) sum_{m=1}^{T0+1}
                1{ S(pi_m u(theta)) >= S(u(theta)) - 1e-12 }

6.4 Acceptance set:
      A_t* = { theta in Theta : p(theta) > alpha }

6.5 Convert accepted grid points to contiguous accepted segments.

6.6 If exactly one segment:
      pointwise CI at t* = [lower_t*, upper_t*]
    Else:
      CI endpoints = NaN
      keep full accepted-set representation

6.7 Compute pointwise p-value for zero effect:
      p_value_t* = p(0)

6.8 Compute significance flag:
      is_significant_t* = 1{ p_value_t* <= alpha }
```

---

# 7) Return PanelEstimate

```text
7.1 Return a PanelEstimate object containing:

    Core outputs:
      estimand            = "dynamic_effect_path"
      effect_by_time      = {tau_hat_t}_{t in post}
      observed_outcome    = Y1_all
      synthetic_outcome   = Y0_hat_all
      donor_weights_augmented = w_aug

    If pointwise conformal was computed:
      ci_lower_by_time
      ci_upper_by_time
      p_value_by_time
      is_significant_by_time
      confidence_set_by_time

    Diagnostics:
      - number of donors
      - number of pre/post periods
      - pre-fit RMSE
      - sum / min / max / L1 norm of augmented weights
      - condition number of augmented Gram matrix
      - fallback-solver information
      - average ATT t-test outputs:
          att
          ci
          p-value
          t-stat
          standard error
          sigma_hat
          fold estimates
          fold blocks
      - pointwise conformal metadata:
          grid by time
          grid p-values by time
          minimum attainable p-value
          multiple-testing note
          warnings / stability messages
```
## Very compact math summary

If you want a shorter “formula-only” version for docs, this is the essence:

$$ 
\hat w_{sc}

\arg\min_{w\in\Delta_J}  
|Y_1^{pre}-X_0^{pre}w|_2^2+\lambda_{sc}|w|_2^2  
$$

$$  
\hat w_{aug}

\arg\min_{w\in\mathbb R^J}  
|Y_1^{pre}-X_0^{pre}w|_2^2+\lambda_{aug}|w-\hat w_{sc}|_2^2  
$$

$$  
\hat Y_0^{all}=X_0^{all}\hat w_{aug},  
\qquad  
\hat\tau_t

Y_{1t}-\hat Y_{0t},  
\quad t\in post  
$$

$$  
\hat\tau_k

\bar g_{post}^{(k)}-\bar g_{hold}^{(k)},  
\qquad  
\hat\tau=\frac{1}{K}\sum_{k=1}^K \hat\tau_k  
$$

$$ 
\hat\sigma_\tau

\sqrt{1+\frac{Kr}{T_1}}  
\sqrt{\frac{1}{K-1}\sum_{k=1}^K(\hat\tau_k-\hat\tau)^2}  
$$

$$ 
t=\frac{\sqrt K,\hat\tau}{\hat\sigma_\tau},  
\qquad  
CI_{1-\alpha}

\hat\tau\pm t_{1-\alpha/2,K-1}\frac{\hat\sigma_\tau}{\sqrt K}  
$$

$$ 
u^{(\theta)}=Y_1^{(\theta)}-X_0\hat w_{aug}^{(\theta)},  
\qquad  
\hat p(\theta)

\frac{1}{T_0+1}  
\sum_{m=1}^{T_0+1}  
\mathbf 1{S(\pi_m u^{(\theta)})\ge S(u^{(\theta)})}  
$$

$$ 
CI_t = {\theta:\hat p_t(\theta)>\alpha},  
\qquad  
p_t=\hat p_t(0)  
$$
# References  
  
- **[AG2003]** Abadie, A., and Gardeazabal, J. (2003). _The Economic Costs of Conflict: A Case Study of the Basque Country_. _American Economic Review_, 93(1), 113–132. ([American Economic Association](https://www.aeaweb.org/articles?id=10.1257%2F000282803321455188&utm_source=chatgpt.com "A Case Study of the Basque Country"))  
      
- **[ADH2010]** Abadie, A., Diamond, A., and Hainmueller, J. (2010). _Synthetic Control Methods for Comparative Case Studies: Estimating the Effect of California’s Tobacco Control Program_. _Journal of the American Statistical Association_, 105(490), 493–505. ([JSTOR](https://www.jstor.org/stable/pdf/29747059.pdf?utm_source=chatgpt.com "Synthetic Control Methods for Comparative Case Studies"))  
      
- **[ADH2015]** Abadie, A., Diamond, A., and Hainmueller, J. (2015). _Comparative Politics and the Synthetic Control Method_. _American Journal of Political Science_, 59(2), 495–510. ([MIT Economics](https://economics.mit.edu/sites/default/files/publications/Comparative%20Politics%20and%20the%20Synthetic%20Control.pdf?utm_source=chatgpt.com "Comparative Politics and the Synthetic Control Method"))  
      
- **[Abadie2021]** Abadie, A. (2021). _Using Synthetic Controls: Feasibility, Data Requirements, and Methodological Aspects_. _Journal of Economic Literature_, 59(2), 391–425. ([NBER](https://www.nber.org/system/files/working_papers/w34550/w34550.pdf?utm_source=chatgpt.com "Working Paper 34550"))  
      
- **[DI2016]** Doudchenko, N., and Imbens, G. W. (2016). _Balancing, Regression, Difference-in-Differences and Synthetic Control Methods: A Synthesis_. NBER Working Paper 22791 / arXiv:1610.07748. ([NBER](https://www.nber.org/system/files/working_papers/w22791/w22791.pdf?utm_source=chatgpt.com "Balancing, Regression, Difference-In- ..."))  
      
- **[BFR2021]** Ben-Michael, E., Feller, A., and Rothstein, J. (2021). _The Augmented Synthetic Control Method_. _Journal of the American Statistical Association_, 116(536), 1789–1803. ([NBER](https://www.nber.org/papers/w28885?utm_source=chatgpt.com "The Augmented Synthetic Control Method"))  
      
- **[CWZ2021]** Chernozhukov, V., Wüthrich, K., and Zhu, Y. (2021). _An Exact and Robust Conformal Inference Method for Counterfactual and Synthetic Controls_. _Journal of the American Statistical Association_, 116(536), 1849–1864. ([OUP Academic](https://academic.oup.com/biometrics/article/80/2/ujae055/7685803?utm_source=chatgpt.com "Doubly robust proximal synthetic controls - Oxford Academic"))  
      
- **[CWZ-T]** Chernozhukov, V., Wüthrich, K., and Zhu, Y. _A t-test for synthetic controls_ / _Debiasing and t-tests for synthetic control inference on average causal effects_. arXiv:1812.10820. ([arXiv](https://arxiv.org/abs/1812.10820?utm_source=chatgpt.com "A $t$-test for synthetic controls"))