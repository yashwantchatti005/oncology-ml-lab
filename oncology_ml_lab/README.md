# Oncology ML Lab — Explainable Breast Cancer Classification (Research Only)

**End-to-end, reproducible Data Science portfolio project** demonstrating problem formulation, data auditing, exploratory analysis, modeling, model selection, threshold optimization, uncertainty, explainability, testing, and interactive presentation.

> **Safety**: Educational research only. This model is NOT medically validated and must never be used for diagnosis, screening, treatment, or any clinical decision.

## Research question
Can cell-nucleus measurements in the classic Wisconsin Diagnostic Breast Cancer dataset distinguish malignant from benign samples? How robust are predictions on an untouched holdout, and which measurement features influence results?

## Dataset and reproducibility
The **569-row, 30-feature** Wisconsin Diagnostic Breast Cancer dataset is bundled with scikit-learn (original dataset from UCI). No download or patient data collection is required. The project exports its input to `data/wisconsin_breast_cancer.csv`. Labels are explicitly recoded to **malignant = 1**, benign = 0. Source: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic and scikit-learn's `load_breast_cancer`.

## Technical highlights
- **Data quality**: shape, missingness, duplicate measurement audit and class distribution.
- **EDA**: class balance and feature distributions, with reproducible figures.
- **Leakage prevention**: no diagnosis or post-outcome information among predictors; all preprocessing inside scikit-learn Pipelines.
- **Evaluation design**: stratified **60% train / 20% validation / 20% untouched test**, fixed seed 42.
- **Model selection**: stratified 5-fold grid-search CV on training only, comparing regularized logistic regression and random forest by ROC AUC.
- **Threshold**: selected on validation only to maximize F2 (greater weight on malignant-class recall). Test is untouched until final reporting.
- **Metrics**: test ROC AUC, average precision, precision, recall, F1/F2, Brier score, confusion matrix and bootstrap 95% CI for ROC AUC.
- **Explainability**: permutation importance on holdout, descriptive and **not causal**.
- **Deployment**: Streamlit demonstration using existing research samples; clearly not a medical device.
- **Engineering**: modular Python, automated tests, saved model artifact, deterministic seeds and pinned major dependencies.

## Run locally
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python -m src.eda
python -m src.train --out .
pytest -q
streamlit run app.py
```
Run all commands from the repository root. The model artifact bundled with this handoff can also be retrained from source.

## Results
See **`reports/metrics.json`** for the actual measured held-out results generated with seed 42, and `reports/figures/` for charts. Never report validation CV as final test accuracy. With only ~114 test records, figures are uncertain; see the bootstrap interval.

## Files
```
app.py                         Streamlit research demo
src/common.py                  Dataset loading and validation
src/eda.py                     Reproducible EDA
src/train.py                   End-to-end modeling and evaluation
tests/test_project.py          Automated checks
data/wisconsin_breast_cancer.csv Local copy of benchmark data
models/model.joblib            Trained sklearn pipeline + threshold
reports/metrics.json          Test metrics and CV comparison
reports/holdout_predictions.csv  Test predictions for audit
reports/figures/              EDA and model charts
```

## Research limitations and next steps
1. Small, curated benchmark with limited real-world representativeness; strong benchmark results are not clinical performance.
2. Random splits may overstate generalization if the target deployment differs in time, location, instruments, or population. **External and prospective validation** are necessary for meaningful clinical claims.
3. The validation threshold chosen on a small validation set is unstable. Assess on larger independent cohorts.
4. Compare probability calibration, subgroup robustness (only where appropriate metadata exist), distribution shift, and CI stability using multi-center data under suitable governance.
5. Do not present permutation importance as causality or the predicted score as a patient's clinical probability.

## Suggested university SOP / GitHub description
> I developed an end-to-end explainable machine-learning research pipeline using the Wisconsin Diagnostic Breast Cancer dataset (569 observations, 30 numerical features). I compared regularized logistic regression and random forest via stratified cross-validation, optimized an F2 decision threshold on a dedicated validation set, and evaluated the selected system once on an untouched test set. I also implemented bootstrapped uncertainty intervals, calibration analysis, permutation feature importance, reproducibility checks, and a Streamlit demonstration. The project highlights rigorous evaluation and the limitations of small clinical benchmarks rather than claiming medical utility.

**Authorship integrity:** Review and understand every code section, reproduce the runs yourself, and describe any AI assistance honestly. This is stronger in interviews than presenting the repository without being able to explain model choices.
