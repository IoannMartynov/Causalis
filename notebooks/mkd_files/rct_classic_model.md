
# Estimand: Average Treatment Effect (ATE)  
The model assumes random assignment of treatment $D \in \{0, 1\}$. The ATE ($\tau$) is defined as:  
$$\tau = E[Y | D=1] - E[Y | D=0]$$  
The sample estimator is the difference in group means:  
$$\hat{\tau} = \bar{Y}_1 - \bar{Y}_0$$  
  
# Inference Methods  
  
**Welch's T-Test (`ttest`)**  
Used for continuous outcomes without assuming equal variances.  
*   **Standard Error:** $SE = \sqrt{\frac{s_1^2}{n_1} + \frac{s_0^2}{n_0}}$, where $s^2$ is the sample variance.  
*   **Degrees of Freedom (Satterthwaite):**   
    $$\nu \approx \frac{\left(\frac{s_1^2}{n_1} + \frac{s_0^2}{n_0}\right)^2}{\frac{(s_1^2/n_1)^2}{n_1-1} + \frac{(s_0^2/n_0)^2}{n_0-1}}$$  
*   **Confidence Interval:** $\hat{\tau} \pm t_{1-\alpha/2, \nu} \cdot SE$  
*   **P-value:** Two-sided test based on the $t$-distribution with $\nu$ degrees of freedom.  
  
**Conversion Z-Test (`conversion_ztest`)**  
Optimized for binary conversion outcomes.  
*   **Proportions:** $p_1 = \frac{X_1}{n_1}, p_0 = \frac{X_0}{n_0}$  
*   **Standard Error (Pooled):** $SE_{pooled} = \sqrt{\hat{p}(1-\hat{p})\left(\frac{1}{n_1} + \frac{1}{n_0}\right)}$ where $\hat{p} = \frac{X_1 + X_0}{n_1 + n_0}$  
*   **Absolute CI (Newcombe-style):** Uses the difference of Wilson score intervals:  
    $$CI = [L_1 - U_0, U_1 - L_0]$$  
    where $[L_i, U_i]$ are the Wilson score bounds for proportion $p_i$.  
  
**Welch Permutation T-Test (`welch_permutation_t_test`)**  
Uses the Welch statistic, but estimates the p-value by repeatedly permuting treatment labels while preserving group sizes.  
*   **Statistic:** $t = \frac{\bar{Y}_1 - \bar{Y}_0}{\sqrt{s_1^2/n_1 + s_0^2/n_0}}$.  
*   **Absolute CI:** Welch-Satterthwaite interval for $\hat{\tau}$.  
*   **P-value:** Monte Carlo permutation p-value with a +1 correction:  
    $$p = \frac{1 + \sum_{b=1}^{B} I(|t_b^*| \ge |t_{obs}|)}{B + 1}$$  
  
# Relative Lift and Delta Method  
The relative lift is calculated as:  
$$Lift (\%) = 100 \cdot \left(\frac{\bar{Y}_1}{\bar{Y}_0} - 1\right)$$  
Its variance is estimated via the **Delta Method**:  
$$Var(Lift/100) \approx \frac{1}{\bar{Y}_0^2} Var(\bar{Y}_1) + \frac{\bar{Y}_1^2}{\bar{Y}_0^4} Var(\bar{Y}_0)$$  
  
---  
  
# Pseudo-code  
  
#### Model Wrapper 

```python  
class DiffInMeans:  
    def fit(data: CausalData):
        self.data = data        
        self.is_fitted = True
        
    def estimate(method="ttest", alpha=0.05, **kwargs):        
	    if method == "ttest":            
		    res = ttest_inference(self.data, alpha)        
	    elif method == "welch_permutation_t_test":            
		    res = welch_permutation_t_test(self.data, alpha, **kwargs)        
	    elif method == "conversion_ztest":            
		    res = ztest_inference(self.data, alpha, **kwargs)                
	    return CausalEstimate(
	    value=res.absolute_difference,
	    p_value=res.p_value,            
        ci=(res.absolute_ci_lower, res.absolute_ci_upper),
        #... additional metadata        
        )  
```  
  
#### Welch's T-Test Inference 

```python  
def ttest_inference(data, alpha):
    y1, y0 = data.split_by_treatment()

    diff = mean(y1) - mean(y0)
    se = sqrt(var(y1) / n1 + var(y0) / n0)

    df_abs = calculate_welch_df(y1, y0)
    tcrit_abs = (
        stats.t.ppf(1 - alpha / 2, df_abs)
        if df_abs_valid
        else stats.norm.ppf(1 - alpha / 2)
    )

    p_val = 2 * (1 - stats.t.cdf(abs(diff / se), df))
    abs_ci = (diff - tcrit_abs * se, diff + tcrit_abs * se)

    # Delta method for relative CI (guard near-zero control mean)
    if abs(mean(y0)) < eps:
        rel_diff = sign(diff) * inf if diff != 0 else 0
        rel_ci = (nan, nan)
    else:
        rel_diff = (mean(y1) / mean(y0) - 1) * 100
        rel_se = 100 * sqrt(
            (1 / mean(y0)) ** 2 * (var(y1) / n1)
            + (mean(y1) / mean(y0) ** 2) ** 2 * (var(y0) / n0)
        )
        df_rel = satterthwaite_df_for_relative(...)
        tcrit_rel = (
            stats.t.ppf(1 - alpha / 2, df_rel)
            if df_rel_valid
            else stats.norm.ppf(1 - alpha / 2)
        )
        rel_ci = (rel_diff - tcrit_rel * rel_se, rel_diff + tcrit_rel * rel_se)

    return {diff, p_val, abs_ci, rel_diff, rel_ci}
  
```  

#### Conversion Z-Test
```python  
def conversion_ztest(data, alpha, ci_method="newcombe", se_for_test="pooled"):
    y1, y0 = data.split_by_treatment()  # y is binary {0, 1}

    p1, p0 = mean(y1), mean(y0)
    diff = p1 - p0
    z_crit = norm.ppf(1 - alpha / 2)

    # 1) P-value (two-sided Z-test)
    if se_for_test == "pooled":
        p_pool = (sum(y1) + sum(y0)) / (len(y1) + len(y0))
        se_test = sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n0))
    else:
        se_test = sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0)

    p_val = 2 * (1 - norm.cdf(abs(diff / se_test)))

    # 2) Absolute CI (Newcombe-style using Wilson score bounds)
    if ci_method == "newcombe":
        l1, u1 = wilson_score_interval(p1, n1, z_crit)
        l0, u0 = wilson_score_interval(p0, n0, z_crit)
        abs_ci = (l1 - u0, u1 - l0)
    else:  # Wald fallback
        abs_ci = (diff - z_crit * se_test, diff + z_crit * se_test)

    # 3) Relative lift (delta method for proportions)
    rel_diff = (p1 / p0 - 1) * 100
    se_rel = 100 * sqrt(
        (1 / p0) ** 2 * (p1 * (1 - p1) / n1)
        + (p1 / p0**2) ** 2 * (p0 * (1 - p0) / n0)
    )
    rel_ci = (rel_diff - z_crit * se_rel, rel_diff + z_crit * se_rel)

    return {
        "p_value": p_val,
        "absolute_difference": diff,
        "absolute_ci": abs_ci,
        "relative_difference": rel_diff,
        "relative_ci": rel_ci,
        # ...
    }

```
#### Welch Permutation T-Test Inference 

```python  
def welch_permutation_t_test(data, alpha, B=10000, alternative="two-sided", seed=None):
    y1, y0 = data.split_by_treatment()

    abs_diff = mean(y1) - mean(y0)
    t_obs = welch_t_stat(y1, y0)

    extreme = 0
    z = concatenate(y1, y0)
    for _ in range(B):
        z_perm = random_permutation(z)
        y1_perm = z_perm[:len(y1)]
        y0_perm = z_perm[len(y1):]
        t_perm = welch_t_stat(y1_perm, y0_perm)

        if alternative == "two-sided" and abs(t_perm) >= abs(t_obs):
            extreme += 1
        elif alternative == "greater" and t_perm >= t_obs:
            extreme += 1
        elif alternative == "less" and t_perm <= t_obs:
            extreme += 1

    p_val = (extreme + 1) / (B + 1)
    abs_ci = welch_satterthwaite_ci(y1, y0, alpha)

    return {abs_diff, t_obs, p_val, abs_ci, ...}
  
```


# References
## Estimand / difference-in-means ATE under random assignment

- **Neyman (1923; English translation 1990)** — potential outcomes framework for randomized experiments; difference-in-means and its sampling properties under randomization. ([ics.uci.edu](https://www.ics.uci.edu/~sternh/courses/265/neyman_statsci1990.pdf?utm_source=chatgpt.com "On the Application of Probability Theory to Agricultural ..."))
    
- **Rubin (1974)** — formalizes causal effects via potential outcomes for randomized (and nonrandomized) studies; motivates ATE as an estimand. ([Ovid](https://www.ovid.com/journals/jedup/pdf/00004760-197410000-00007~estimating-causal-effects-of-treatments-in-randomized-and?utm_source=chatgpt.com "ESTIMATING CAUSAL EFFECTS OF TREATMENTS IN ..."))
    
- **Freedman (2008)** — uses the Neyman/Rubin potential-outcomes model to discuss inference and common adjustments in experiments (helpful for “design-based” framing around diff-in-means). ([causal.unc.edu](https://causal.unc.edu/wp-content/uploads/sites/1553/2011/08/Freedman2008.pdf?utm_source=chatgpt.com "On regression adjustments to experimental data"))
    
- **Imbens & Rubin (2015)** — book, but extremely standard citation for ATE estimands + sampling variances for average causal effects. ([Cambridge University Press & Assessment](https://www.cambridge.org/core/books/causal-inference-for-statistics-social-and-biomedical-sciences/71126BE90C58F1A431FE9B2DD07938AB?utm_source=chatgpt.com "Causal Inference for Statistics, Social, and Biomedical ..."))
    

## Welch’s t-test + Satterthwaite df

- **Student (1908)** — original t distribution and small-sample inference motivation. ([JSTOR](https://www.jstor.org/stable/2331554?utm_source=chatgpt.com "The Probable Error of a Mean"))
    
- **Welch (1947)** — the unequal-variance two-sample t-test (the modern “Welch’s t-test”). DOI: 10.1093/biomet/34.1-2.28. ([OUP Academic](https://academic.oup.com/biomet/article-abstract/34/1-2/28/210174?utm_source=chatgpt.com "THE GENERALIZATION OF 'STUDENT'S' PROBLEM WHEN ..."))
    
- **Satterthwaite (1946)** — effective degrees of freedom approximation (the “Welch–Satterthwaite” df you wrote). ([JSTOR](https://www.jstor.org/stable/3002019?utm_source=chatgpt.com "An Approximate Distribution of Estimates of Variance ..."))
    

## Conversion z-test + “Newcombe/Wilson-style” absolute CI for two proportions

**Wilson score interval (single proportion):**

- **Wilson (1927)** — the score interval for a binomial proportion (better coverage than Wald). ([JSTOR](https://www.jstor.org/stable/2276774?utm_source=chatgpt.com "Probable Inference, the Law of Succession, and Statistical ..."))
    

**Newcombe’s CI for difference of independent proportions (the one you’re implementing as “Newcombe-style / Wilson score bounds then subtract”):**

- **Newcombe (1998)** — “Interval estimation for the difference between independent proportions: comparison of eleven methods.” This is the go-to citation for the Wilson-score-based risk-difference CI family. DOI: 10.1002/(SICI)1097-0258(19980430)17:8<873::AID-SIM779>3.0.CO;2-I. ([PubMed](https://pubmed.ncbi.nlm.nih.gov/9595617/?utm_source=chatgpt.com "Interval estimation for the difference between independent ..."))
    

(If you also want a citation for Wilson/score intervals generally outperforming “exact”/Wald-style intervals in practice:)

- **Agresti & Coull (1998)** — “Approximate is Better than ‘Exact’…” (single-proportion intervals, but commonly cited in the same discussion). ([math.unm.edu](https://math.unm.edu/~james/Agresti1998.pdf?utm_source=chatgpt.com "Approximate is Better than “Exact” for Interval Estimation of ..."))
    
- **Agresti & Caffo (2000)** — simple adjusted intervals for proportions _and differences_ (alternative to Newcombe; useful “related work” citation). ([Statistics](https://users.stat.ufl.edu/~aa/articles/agresti_caffo_2000.pdf?utm_source=chatgpt.com "Simple and Effective Confidence Intervals for Proportions and ..."))
    

## Permutation tests

- **Fisher (1935)** — randomization/permutation testing as exact design-based inference under random assignment.
    
- **Good (2005)** — practical reference for permutation, parametric, and bootstrap tests, including Monte Carlo permutation p-values.

## Relative lift (ratio) + Delta method variance

- **Oehlert (1992)** — clean, standard citation for delta method approximations. DOI: 10.1080/00031305.1992.10475842. ([Taylor & Francis Online](https://www.tandfonline.com/doi/abs/10.1080/00031305.1992.10475842?utm_source=chatgpt.com "A Note on the Delta Method: The American Statistician"))
    

And since your pseudo-code explicitly guards when the control mean is near zero (ratio instability), it’s also common to cite ratio-CI alternatives:

- **Fieller (1954)** — Fieller-type confidence intervals for ratios (classic reference when denominators can be near 0). ([JSTOR](https://www.jstor.org/stable/2984043?utm_source=chatgpt.com "Some Problems in Interval Estimation"))
    
