# Bank Marketing – DBA5106 Project 2

Predict whether a bank customer will subscribe to a term deposit, and choose who to call
so that the business cost (wasted calls vs missed subscribers) is as low as possible.

## Dataset

- UCI Bank Marketing, `bank-additional-full.csv` (Moro, Rita & Cortez, 2014): https://doi.org/10.24432/C5K306
- Also on Kaggle, e.g. https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing
- The code downloads it once from
  https://raw.githubusercontent.com/fabiorcampos/Bank-Marketing/master/data/bank-additional-full.csv
  (identical to the UCI file) and keeps a local copy in `data/`.
- 41,188 calls (May 2008 – Nov 2010), 20 inputs, target `y` (11.3% "yes"). The separator is `;`.

## Layout

```
src/bank.py                          shared data prep: load, clean, split, cost function
notebooks/01_models_and_threshold.ipynb   4 models, AUC, calibration, threshold, test result
notebooks/01_models_and_threshold.py      same analysis as a plain script (python notebooks/01_models_and_threshold.py)
notebooks/02_shap.ipynb              Tree SHAP on the boosting model (to do)
models/boosting_for_shap.joblib      fitted boosting pipeline for SHAP
models/final_model.joblib            selected model + calibration map + threshold
models/feature_names.csv             feature names after one-hot encoding
figures/                             ROC, calibration, cost-vs-threshold plots
results/                             tables behind the slides (CSV)
starter.py                           first version of the data prep (kept for reference)
```

## Getting started

```
pip install -r requirements.txt
jupyter notebook notebooks/01_models_and_threshold.ipynb
```

`scikit-learn` is pinned because the saved models are pickled with that version.

## Method (notebook 01)

- Data prep: drop `duration` (only known after the call, so it leaks the outcome), replace `pdays`
  with `contacted_before` (999 = never contacted), keep "unknown" as a category.
- Split: stratified random 60 / 20 / 20 (train / validation / test). The time-ordered split shows strong
  drift (see `results/time_split_drift.csv`) and is reported as a limitation. Section 9b of the notebook
  explains why a model trained on 2008 calls scores AUC ≈ 0.5 on later calls (economic indicators act as
  date stamps whose effect reverses, the best months are missing from training, previous-campaign history
  did not exist yet); see `results/time_split_*.csv` and `figures/time_split_feature_groups.png`.
- Evaluation loss: average cost per customer, `COST_FP` per wasted call and `COST_FN` per missed
  subscriber (placeholders 5 and 50 in `src/bank.py`).
- Models: ridge regression (L2) vs logistic regression (log loss) as glass-box; gradient boosting with
  squared error vs log loss as black-box. Each tuned separately by 5-fold CV AUC on train.
- Compare by validation AUC; calibrate with Platt or isotonic (fitted on out-of-fold train
  predictions); choose the threshold that minimises the cost on validation; report once on test.

## Using the boosting model for SHAP

```python
import sys, joblib; sys.path.insert(0, "src")
import bank, shap
train, val, test = bank.split(bank.clean(bank.load_raw()))   # same split as notebook 01
X_test, y_test = bank.xy(test)
pipe = joblib.load("models/boosting_for_shap.joblib")
X = pipe[:-1].transform(X_test)
names = pipe[:-1].get_feature_names_out()
explainer = shap.TreeExplainer(pipe[-1])
```
## SHAP Analysis 

## Open decisions

- Cost numbers: `COST_FP` and `COST_FN` in `src/bank.py` are placeholders. Only their ratio changes
  the decision; `results/cost_sensitivity_validation.csv` shows other ratios.

## Work split

- Modelling: data prep, cost function, 4 models, validation AUC, calibration, threshold.
- Interpretation and slides: Tree SHAP (global and local) on the boosting model, slide deck.
