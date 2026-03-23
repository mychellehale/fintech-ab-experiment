# Data Science Interview Prep: A/B Testing & Statistical Analysis

Based on a fintech feature adoption experiment using the UCI Bank Marketing dataset.

**Experiment setup:**
- **Treatment** — customers previously contacted by the bank (`previous > 0`)
- **Control** — customers not previously contacted
- **Outcome** — subscribed to a term deposit (`y == 'yes'`)
- **Dataset** — ~45,000 rows, UCI Bank Marketing

---

## The Three Questions You Always Need to Answer

| Question | Tool | What it tells you |
|---|---|---|
| Is the lift real or noise? | p-value (chi-squared) | Statistical significance |
| Is the lift big enough to matter? | Effect size (Cohen's h) | Practical significance |
| Did we have enough data to trust either answer? | Power analysis | Reliability of the experiment |

---

## Q1: p=0.003, stakeholder wants to scale. What do you push back on?

**What to check before agreeing:**

1. **Effect size** — a significant p-value with a tiny lift (e.g. 0.1%) may not justify the cost of the campaign. Statistical significance ≠ practical significance
2. **Power** — was the sample large enough to reliably detect the effect?
3. **Selection bias** — groups aren't randomly assigned. Treatment is defined by `previous > 0`, meaning treatment users self-selected by having been contacted before. They may already be more engaged customers. The significant result could reflect pre-existing differences, not the campaign effect
4. **Multiple comparisons** — if segmentation tests were also run, you'd expect ~1 in 20 to come back significant by chance at α=0.05. A significant global result is more trustworthy if it holds after correction and in the regression

---

## Q2: Why does logistic regression add value over chi-squared?

**Chi-squared** is unadjusted — it sees that treatment converts more but can't isolate why. It has no way to ask "but what if I hold everything else constant?"

**Logistic regression** models the probability of converting as a function of multiple variables simultaneously. By including covariates alongside `treatment_flag`, it asks: *among users identical on balance, job, marital status, etc., what is the independent contribution of being in treatment?*

**The practical consequence:** the balance check flags that groups are imbalanced on balance, job, housing, etc. The chi-squared result could be picking up those pre-existing differences rather than the campaign effect. The regression controls for observed differences; chi-squared cannot.

**Key limitation of both:** logistic regression can only control for *observed* covariates. Unobserved confounders (e.g. willingness to engage with a rep) still bias the result. The gold standard fix is random assignment.

---

## Q3: Lift is only significant for 30-39 year olds. Stakeholder wants to target them.

**What to raise:**

1. **Multiple comparisons** — you tested 5 age bands at α=0.05, meaning one would come back significant by chance even if there's no real effect. A single significant segment out of many tests is weak evidence on its own
2. **Imbalance bias** — other variables are imbalanced between groups. The segment result could be driven by covariate differences within that age band, not the campaign
3. **Right response** — treat 30-39 as a hypothesis to validate in a prospective experiment, not a finding to act on. "This is interesting and worth investigating, but this analysis wasn't designed to find segment-specific effects"

---

## Q4: Treatment effect is still biased after regression controls. Is your colleague right?

**Yes, they are right.**

The regression controls for *observed* covariates — balance, job, housing etc. But **willingness to engage with a rep** is an unobserved behavioral trait not in the data. It almost certainly correlates with likelihood to convert independently of the campaign.

This is **confounding on unobservables** — the regression can't fix what it can't see. The treatment effect estimate absorbs that unobserved difference and is likely overstated.

**Fixes:**
- **Random assignment** — distributes unobservables equally across groups by design
- **Tighter experiment definition** — filter on `contact != 'unknown'` to get a cleaner proxy for "not treated"

---

## Q5: What is Cohen's h and why not use raw conversion rate difference?

**The problem with raw difference:** it's baseline-dependent. A 5% lift from 5%→10% is much more meaningful than 50%→55%. The same absolute difference means something very different depending on the baseline.

**Cohen's h** applies an arcsine transformation first, which stabilizes variance across different baseline rates. The formula stretches the scale near 0 and 1 (where variance is naturally compressed), making effect sizes comparable regardless of baseline.

```python
cohens_h = 2 * (arcsin(sqrt(t_rate)) - arcsin(sqrt(c_rate)))
```

**Conventional thresholds:** h=0.2 small, h=0.5 medium, h=0.8 large

**Why it matters here:** Cohen's h feeds directly into the power analysis formula for proportions — it's the standard input to sample size calculations.

**Alternative effect size measures:**

| Measure | Formula | When to use |
|---|---|---|
| Absolute lift | `t_rate - c_rate` | Simple, honest, executive-facing |
| Relative lift | `(t_rate - c_rate) / c_rate` | Business-friendly but inflated at low baselines |
| Odds ratio | `(t/(1-t)) / (c/(1-c))` | Natural when running logistic regression |
| Risk ratio | `t_rate / c_rate` | Epidemiology standard, easier to interpret than OR |
| Cohen's h | `2(arcsin√t - arcsin√c)` | Best for power analysis with proportions |

---

## Q6: Why 80% power, and when would you raise it?

**The asymmetry behind 80%:**
- **Alpha (5%)** = tolerance for a false positive (saying the campaign worked when it didn't) — kept low because acting on a false positive wastes money
- **Power (80%)** = tolerance for a false negative (missing a real effect) — more lenient because missing a real effect is considered less costly than a false alarm
- The implicit assumption is that **false positives are 4x more costly than false negatives**

80% is a convention established by Jacob Cohen in the 1970s, not a law. Raise to 90%+ when the cost of missing a real effect is high — e.g. not rolling out a feature that genuinely drives significant revenue, or in clinical trials where missing a real treatment effect has serious consequences.

**How power is calculated:**
```python
z_alpha = norm.ppf(1 - α/2)          # critical value for significance threshold
z_power = h * sqrt((n_t * n_c) / (n_t + n_c)) - z_alpha   # standardized effect
power   = norm.cdf(z_power)           # probability of detecting the effect
```
**Required sample size for 80% power:**
```python
n_required = ((z_alpha + z_beta) / h) ** 2 * 2   # where z_beta = norm.ppf(0.80)
```
Power increases with larger effect size and larger sample size.

---

## Q7: The power analysis is retrospective. What's the problem?

**The circularity problem:** the retrospective power calculation uses the observed effect size from the data, which is itself noisy. If you happened to measure a larger lift than the true underlying effect by chance, power looks great. If smaller, power looks terrible. You're using a noisy estimate to evaluate whether you had enough data to detect that noisy estimate.

**The right approach — prospective power analysis:**
1. Decide the minimum effect size that would be business-meaningful (e.g. "we only care if the campaign drives at least a 2% lift")
2. Plug that into the power formula to get the required sample size
3. Run the experiment until you hit that sample size

That way power is based on what you *need* to detect, not what you happened to observe.

---

## Q8: Odds ratio is 1.4 but raw lift is 2%. Stakeholder wants to say "40% more likely to convert."

**Why it's wrong:** odds ratio and probability are only similar when baseline rates are very low. At 9% baseline they're close but not equal — at higher baselines they diverge dramatically. "40% more likely" implies a relative risk framing, which would actually be `11/9 = 1.22` (22% more likely), not 40%.

**What to lead with by audience:**

| Audience | Metric | Why |
|---|---|---|
| Executives | Absolute lift ("2 percentage point increase") | Simple, honest, hard to misinterpret |
| Data team | Odds ratio from regression | Accounts for covariates |
| Never | Odds ratio framed as "X% more likely" to non-technical audience at baseline rates above ~10% | Systematically overstates the effect |

---

## Q9: All segments go non-significant after Bonferroni correction. How do you handle it?

**What it means:** the original findings were likely false positives — you ran enough tests that one came up significant by chance, exactly the problem Bonferroni is designed to catch.

**How to handle it:**
1. Don't report uncorrected segments as findings
2. Treat them as hypotheses to validate in a prospective experiment
3. Consider **Benjamini-Hochberg** instead of Bonferroni — Bonferroni assumes all tests are independent, but segments overlap (age and job aren't independent), so it overcorrects. Benjamini-Hochberg controls false discovery rate and is more appropriate for exploratory segmentation
4. Report honestly to stakeholders who saw uncorrected results

**Bonferroni correction:** divide α by number of tests. e.g. 20 tests → require p < 0.05/20 = p < 0.0025

---

## Q10: Confidence intervals barely overlap but p=0.04. Colleague says result isn't significant. Are they right?

**No, they are wrong.**

Each CI is built around one group's rate independently — "where does treatment's true rate likely fall?" The significance test asks a different question: "is the *difference* between the two rates significant?" That's its own sampling distribution, narrower than either individual CI.

Two groups can have visually overlapping 95% CIs and still have p<0.05. The right tool is a CI around the *difference itself* — if that interval excludes zero, the result is significant regardless of whether individual CIs overlap.

---

## Q11: n<30 filter in segmented analysis. Why 30, and is it right?

**Why 30:** it's a proxy for the chi-squared assumption that expected cell counts ≥ 5 in each cell of the contingency table. Below that, chi-squared produces unreliable p-values.

**Why it's not quite right:** n=30 doesn't account for effect size or power. The right threshold is the **minimum sample size required to detect your minimum meaningful effect at 80% power** — the same logic as prospective power analysis applied at the segment level.

- A segment with n=30 and a 40% lift might be perfectly detectable
- A segment with n=500 and a 0.5% lift might be completely underpowered

**Also relevant — Wilson vs normal approximation for CIs:** the normal approximation (`1.96 * se`) is fine at the global level (45,000 rows, 9-11% conversion). It breaks down in segmented analysis for small subgroups with extreme conversion rates — can produce intervals below 0, which is nonsensical for a proportion. Wilson interval stays bounded between 0 and 1 by construction and is the safer choice for segments.

---

## Q12: Which covariates should be excluded from the logistic regression?

**Framework — include if:**
- Measured before treatment
- Predicts conversion independently of treatment
- Not collinear with treatment assignment

| Variable | Include? | Reason |
|---|---|---|
| `duration` | ❌ | **Post-treatment** — only exists because a call happened. Absorbs treatment effect, biases coefficient downward |
| `poutcome` | ❌ | **Collinear with treatment** — customers with `poutcome = success` almost certainly have `previous > 0` |
| `pdays` | ❌ | **Collinear with treatment** — customers contacted before overlap heavily with treatment group |
| `previous` | ❌ | **Is treatment** — including it as a covariate makes no sense |
| `default` | ✅ | Pre-treatment, measures creditworthiness, predicts conversion independently |
| `balance`, `job`, `marital`, `education`, `housing` | ✅ | Pre-treatment, flagged as imbalanced, predict conversion independently |

**Two distinct failure modes:**
- **Post-treatment bias** (`duration`) — variable is caused by treatment, so controlling for it absorbs the treatment effect
- **Collinearity with treatment** (`poutcome`, `pdays`) — variable is so correlated with treatment assignment that the regression can't separate their effects

---

## Key Concepts Cheat Sheet

| Concept | One-line definition |
|---|---|
| p-value | Probability of seeing this result by chance if there's no real effect |
| Effect size | How big is the effect, independent of sample size |
| Power | Probability of detecting a real effect if it exists |
| False positive (Type I) | Saying the campaign worked when it didn't — controlled by alpha |
| False negative (Type II) | Missing a real effect — controlled by power |
| Selection bias | Groups differ in ways unrelated to treatment because assignment wasn't random |
| Confounding on unobservables | Bias from variables you can't measure or control for |
| Multiple comparisons | Running many tests inflates false positive rate — correct with Bonferroni or Benjamini-Hochberg |
| Post-treatment bias | Including a variable caused by treatment absorbs the treatment effect |
| Collinearity | Two variables are so correlated the model can't separate their effects |
| Prospective power analysis | Calculate required sample size *before* the experiment based on minimum meaningful effect |
| Retrospective power analysis | Calculate power *after* using observed effect size — circular and unreliable |
