# Feature Adoption Experiment Analysis

An end-to-end A/B experiment analysis pipeline built on the UCI Bank Marketing dataset. It tests whether prior customer engagement predicts feature adoption, and covers the full analytical lifecycle: balance checking, significance testing, explainability, and drift detection.

**Stack:** Python · scipy · statsmodels · scikit-learn · SHAP · Evidently · uv

---

## The Problem

Does contacting a customer in a previous campaign make them more likely to adopt a new product? This is the core question behind campaign targeting decisions at retail banks. Answering it requires more than a p-value — you need to verify group comparability, control for confounders, measure effect size, and check whether the model holds over time.

---

## Data

- **Source:** UCI Bank Marketing dataset (`fetch_ucirepo(id=222)`)
- **Size:** 45,211 records, 17 features
- **Group assignment:** Users with prior campaign contact (`previous > 0`) = treatment; no prior contact = control
- **Outcome:** Subscribed to a term deposit (`y == 'yes'`)
- **Known limitation:** Groups are not randomly assigned. Treatment users self-selected by being reachable in a prior campaign, introducing selection bias that motivates the covariate controls in Step 2.

---

## Methods

Data is ingested via a generator-based streaming pipeline simulating chunk-by-chunk Kafka consumption. All analysis modules live in `src/analysis/` and are called by thin entry-point scripts.

**Step 1 — Balance check** (`scripts/run_balance_check.py`): T-tests on numeric features and chi-squared tests on categoricals verify group comparability before any significance testing.

**Step 2 — Significance testing** (`scripts/run_significance.py`): Chi-squared test, 95% confidence intervals, power analysis (Cohen's h), segmentation across age bands and job types, and logistic regression controlling for imbalanced covariates. `duration` is excluded from all models — it is a post-treatment variable that would leak outcome information.

**Step 3 — SHAP explainability** (`scripts/run_shap.py`): A `GradientBoostingClassifier` is trained to predict conversion. SHAP values rank feature contributions globally and at the individual level. Relevant to EU AI Act Article 86 requirements for automated decision explanations.

**Step 4 — Drift detection** (`scripts/run_drift.py`): Dataset split into reference (first 60%) and current (last 40%) periods. Evidently compares feature distributions and model predictions across periods.

---

## Key Results

| Metric | Value |
|--------|-------|
| Treatment conversion rate | 23.1% |
| Control conversion rate | 9.2% |
| Raw lift | +13.9% |
| Chi-squared p-value | 0.000000 |
| Cohen's h (effect size) | 0.387 (medium) |
| Observed power | 100.0% |
| Adjusted odds ratio (logistic regression) | 2.930x |
| Adjusted p-value | 0.000000 |

The lift is **statistically significant** at alpha=0.05.

After controlling for imbalanced covariates identified in the balance check, the adjusted odds ratio of 2.93x confirms the treatment effect survives covariate adjustment.

![Summary](outputs/summary.png)

---

## Project Structure

```
src/
    pipeline.py              # Shared fetch, stream, and transform functions
    analysis/
        balance.py           # Balance check logic
        significance.py      # Chi-squared, power, segmentation, logistic regression
        shap.py              # SHAP training and explainability
        drift.py             # Evidently drift detection
        summary.py           # Summary visualization and README generation
scripts/
    run_balance_check.py
    run_significance.py
    run_shap.py
    run_drift.py
    run_summary.py
outputs/                     # Generated charts and reports
```

---

## How to Run

```bash
git clone https://github.com/mychellehale/fintech-ab-experiment.git
cd fintech-ab-experiment
uv sync
uv run python scripts/run_balance_check.py
uv run python scripts/run_significance.py
uv run python scripts/run_shap.py
uv run python scripts/run_drift.py
uv run python scripts/run_summary.py
```

---

## Compliance

The SHAP explainability layer addresses EU AI Act Article 86, which requires meaningful explanations for automated decisions affecting individuals — a live compliance requirement for credit, fraud, and personalisation models at regulated financial institutions.
