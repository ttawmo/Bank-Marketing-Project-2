# Bank Marketing – DBA5106 Project 2

Predict whether a bank customer will subscribe to a term deposit, and choose who to call
so that the business cost (wasted calls vs missed subscribers) is as low as possible.

## Dataset

- Direct CSV used by the code: https://raw.githubusercontent.com/fabiorcampos/Bank-Marketing/master/data/bank-additional-full.csv
- Original source (UCI Bank Marketing, Moro, Rita & Cortez, 2014): https://doi.org/10.24432/C5K306
- 41,188 rows, 20 inputs, target `y` (11.3% "yes"). The separator is `;`.

## Getting started

```
pip install -r requirements.txt
python starter.py
```

`starter.py` loads the data, drops `duration` (only known after the call, so it leaks the
outcome), recodes `pdays`, splits by time into train / validation / test, builds the
preprocessing step and defines the business cost function.

## Open decisions

- Cost numbers: `COST_FP` and `COST_FN` in `starter.py` are placeholders.
- Split: the time-based split gives subscription rates of 4.8% (train), 11.1% (validation)
  and 30.8% (test). Decide whether to keep it or use a random stratified split.

## Work split

- Modelling: data prep, cost function, 4 models (glass-box and black-box, each with L2 and
  another surrogate loss), validation AUC, calibration, threshold.
- Interpretation and slides: Tree SHAP (global and local) on the boosting model, slide deck.
