---
module: Interview Prep
topic: Ml
subtopic: Canonical Stats Questions
status: unread
tags: [interviewprep, ml, ml-canonical-stats-questions]
---
# Canonical Statistics Interview Questions

Statistical foundations that show up in every ML interview: probability, inference, experimental design, and the traps interviewers set. See [Probability & Statistics](10-probability-and-statistics.md) for the underlying theory (Bayes, MLE/MAP, CLT, bootstrap) — this file focuses on the Q&A framing and topics not covered there (Simpson's paradox, multiple testing, causal design).

---

## 1. Multiple Testing (Bonferroni and Benjamini-Hochberg)

**Problem:** test 20 independent hypotheses at α=0.05 → expected false positives = 1, even if all $H_0$ are true. Family-wise error rate (FWER) = $1-(1-\alpha)^m \approx 1$ for large m.

**Bonferroni (conservative):** $\alpha_{adjusted} = \alpha/m$. Simple but causes many false negatives when tests are correlated.

**Benjamini-Hochberg (FDR control):** controls the expected fraction of false positives among rejections, not FWER.
1. Sort p-values ascending.
2. Find largest $k$ such that $p_{(k)} \leq \frac{k}{m}\alpha$.
3. Reject all hypotheses up to $k$.

```python
from statsmodels.stats.multitest import multipletests
reject_bh, p_corrected_bh, _, _ = multipletests(p_values, method='fdr_bh', alpha=0.05)
```

**When to use which:** Bonferroni for confirmatory studies where FWER control is critical (clinical trials); BH for exploratory studies with many hypotheses (genomics, feature selection).

**ML application:** A/B testing 50 metrics at once → BH to avoid declaring winners from noise.

---

## 2. Simpson's Paradox

A trend in subgroups reverses in the aggregate, caused by a confounder correlated with both the grouping variable and the outcome.

**Classic example (UC Berkeley 1973):** aggregate admit rate looked higher for men (44% vs 35%), but women had higher admit rates in most individual departments — women applied more to competitive departments (department is the confound).

**Rule:** always check aggregated vs. stratified results before making a causal claim. Simpson's paradox is a signal a confounder exists.

---

## 3. Causal Study Design

| Design | Causal strength | Example |
|---|---|---|
| RCT (A/B test) | Gold standard | Feature flag experiment |
| Quasi-experiment (IV, DiD) | Strong | Rollout by region |
| Regression discontinuity | Strong (local) | Cutoff threshold effects |
| Propensity score matching | Moderate | Observational with controls |
| Regression on observational data | Weak | Correlational analysis |

**Instrumental variables:** used when treatment is endogenous (correlated with unobserved confounders). Need an instrument $Z$ that (1) correlates with treatment, (2) affects outcome only through treatment, (3) is independent of unobserved confounders. Example: distance to nearest college as an instrument for education when estimating effect on wages.

**Difference-in-differences:**
$$\text{ATT} = (\bar Y_{treated,post}-\bar Y_{treated,pre}) - (\bar Y_{control,post}-\bar Y_{control,pre})$$
Requires the parallel trends assumption (treated and control would have moved together absent treatment).

---

## 4. Common Statistical Traps in ML Interviews

1. **Accuracy on imbalanced data** — 99% accuracy by predicting all-negative on a 1% positive class. Fix: precision/recall/F1/PR-AUC.
2. **Data leakage** — scaling using statistics from the full dataset including test. Fix: fit scaler on train only.
3. **Multiple comparisons without correction** — "we ran 20 A/B tests and this one won." Fix: Bonferroni/BH, pre-register the primary metric.
4. **Correlation vs. causation** — ice cream sales and drownings correlate (confounder: summer). Fix: draw the DAG, identify confounders.
5. **p-hacking / HARKing** — hypothesizing after seeing results, stopping collection right when p<0.05. Fix: pre-registration, alpha-spending for sequential tests.
6. **Survivorship bias** — evaluating a model only on cases that "survived" to be observable (e.g. only confirmed cancer cases, only stocks still listed). Fix: match the eval set to the real deployment population.

---

## Canonical Interview Q&As

**Q: Explain p-values to a PM who wants to know if the experiment worked.**
A: "Imagine the feature has zero effect. If we reran the experiment 100 times, about p×100 of those would show a result this extreme purely by chance. p=0.03 means only 3 in 100 would look like this if nothing was going on — so it's unlikely to be noise. But check the effect size and CI too — that tells you if it's actually worth shipping."

**Q: How do you handle testing 50 metrics in one A/B experiment?**
A: Pick one primary metric before running the test. Apply Benjamini-Hochberg to the other 49 at 5% FDR. Treat any surprising secondary-metric win as a hypothesis for a dedicated follow-up test, not a shipping decision.

**Q: Treated users have higher average income than control — how do you handle this in a revenue analysis?**
A: Income is a confounder. In order of rigor: randomize (RCT); propensity score matching/weighting; DiD if rollout timing gives a quasi-instrument; or control for income in regression. If treatment was truly randomized, the imbalance is likely chance; if not, you must adjust for it.

**Q: Confidence intervals vs. credible intervals?**
A: CI (frequentist): a procedure that contains the true parameter 95% of the time under repeated sampling — you can't say "95% probability it's in this interval." Credible interval (Bayesian): given data and prior, there's a 95% probability the parameter is in this interval — a more direct statement. For large n with weak priors the two are numerically similar.
