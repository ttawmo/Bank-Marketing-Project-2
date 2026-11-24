# Phase 4 — Interpretability: Global and Local Tree SHAP

## 1. Model and analysis scope

This analysis explains the repository's selected black-box model: **scikit-learn `GradientBoostingClassifier(loss="log_loss")`** (Gradient Boosting with Log Loss). It is not an `XGBClassifier` from the XGBoost package.

- Validation AUC recorded in `results/model_comparison.csv`: **0.8060476** (rounded: **0.8060**).
- Validation set: **8,238** observations from the repository's stratified random 60/20/20 split (`random_state=42`).
- Model input: **62 transformed features** (9 numerical columns plus one-hot encoded categorical columns).
- Original predictors: 19 after dropping `duration` and replacing `pdays` with `contacted_before`.
- SHAP explains the model's **predicted probability of subscription (`y=1`)**.

The saved model was loaded from `models/boosting_for_shap.joblib`. The original data preparation and split were reproduced using `src/bank.py`. The saved pipeline's validation AUC was recomputed as **0.806047602**, matching the recorded value.

## 2. SHAP method

- Explainer: `shap.TreeExplainer`
- Feature perturbation: `interventional`
- Model output: `probability`
- Background data: a reproducible random sample of **100 rows** from the transformed training set (`random_state=42`).
- Explanation set for the global analysis: all **8,238 validation observations**.
- Output space: probability, so each observation's SHAP values add to the model's predicted subscription probability (up to floating-point precision).

The mean model output on the selected background sample (SHAP base value) was **0.139566**.

### Feature grouping

The fitted preprocessing pipeline expands categorical variables into one-hot encoded columns. The beeswarm plot therefore displays encoded features. For easier business interpretation, the grouped global bar chart aggregates the mean absolute SHAP values of the encoded columns belonging to each original predictor. This is a descriptive aggregation; predictors with more encoded categories can accumulate importance across more columns.

## 3. Global Tree SHAP

Files:
- `figures/global_tree_shap_beeswarm.png`
- `figures/global_tree_shap_grouped_features.png`
- `figures/global_tree_shap_bar.png`
- `results/global_shap_feature_importance.csv`
- `results/global_shap_grouped_importance.csv`

### Main findings

The most influential original predictors by grouped mean absolute SHAP value were:

- **nr.employed**: 0.06319
- **poutcome**: 0.01816
- **contacted_before**: 0.01659
- **cons.conf.idx**: 0.01536
- **euribor3m**: 0.01309
- **contact**: 0.01089
- **day_of_week**: 0.01070
- **job**: 0.00817

`nr.employed` is the strongest driver by a substantial margin in this fitted model. Previous campaign outcome (`poutcome`), whether the customer had been contacted before (`contacted_before`), consumer confidence index (`cons.conf.idx`), and the 3-month Euribor rate (`euribor3m`) are also among the most influential predictors.

The beeswarm plot shows the distribution and direction of encoded-feature SHAP values across the validation set. For example, lower `nr.employed` values tend to push the model's predicted subscription probability upward, while higher values tend to push it downward in this dataset/model. This is an association learned by the model, not evidence that changing employment levels would cause an individual to subscribe.

## 4. Local Tree SHAP

Files:
- `figures/local_tree_shap_waterfall.png` (encoded-feature waterfall)
- `figures/local_tree_shap_grouped_waterfall.png` (grouped to original predictors)
- `results/local_shap_contributions.csv`
- `results/local_shap_grouped_contributions.csv`
- `results/local_customer_profile.csv`

### Selected validation observation

For a concrete case, we selected the highest-scoring true positive in the validation set. The observation is anonymised and identified only by its original row index in the public dataset.

- Original row index: **39498**
- Actual outcome: **Yes (`y=1`)**
- Predicted subscription probability: **0.9516**
- Validation threshold recorded for GB (log loss): **0.0825**
- Predicted class at that threshold: **Yes**
- SHAP base value: **0.1396**
- Sum of SHAP contributions: **0.8120**

The additive explanation is approximately:

`0.1396 (base value) + 0.8120 (sum of SHAP values) = 0.9516 (predicted probability)`

The largest grouped positive contributions for this customer were:

- **nr.employed**: +0.3046
- **contacted_before**: +0.1438
- **euribor3m**: +0.1413
- **poutcome**: +0.0714
- **age**: +0.0541
- **campaign**: +0.0353
- **previous**: +0.0274
- **job**: +0.0109

For this individual, the model was strongly pushed toward a positive subscription prediction by the customer's `nr.employed` and `euribor3m` values, the `contacted_before` indicator, and previous campaign information (`poutcome` and `previous`). The waterfall plot shows how these contributions move the prediction from the baseline to the final probability.

This is one selected example, not a summary of all customers. The feature values and SHAP contributions explain this model's prediction; they do not establish causal effects.

## 5. Business value

The model can support campaign prioritisation by ranking customers according to their predicted probability of subscribing. The bank could use this ranking, together with a business-cost threshold and available call capacity, to prioritise outreach rather than calling every customer indiscriminately.

The SHAP results can help:
- identify which variables the model relies on most across the validation population;
- explain why a particular customer received a high or low predicted probability;
- communicate model behaviour to marketing and campaign teams;
- flag reliance on macroeconomic variables that may become unstable over time.

SHAP explanations should be used as model diagnostics and communication aids, not as causal claims or as a substitute for campaign experimentation.

## 6. Model limitations and risks

1. **Historical and geographic scope.** The dataset represents a Portuguese bank's telemarketing campaigns from May 2008 to November 2010. The learned relationships may not transfer to current campaigns, other countries, or digital marketing channels.
2. **Strong temporal drift.** The repository's time-ordered evaluation reports AUC **0.4652** on the next 20% and **0.5417** on the last 20%, compared with **0.8060** on the stratified random validation split. The subscription rate also rises from **4.81%** in the first 60% to **11.07%** in the next 20% and **30.83%** in the last 20%. This indicates substantial distribution and relationship changes over time.
3. **Macroeconomic dependence.** `nr.employed` dominates the global SHAP ranking. Macroeconomic indicators may act partly as proxies for the historical period. Their predictive relationships can change when economic conditions change.
4. **Association, not causation.** SHAP attributes the fitted model output to input features under the chosen explainer assumptions. It does not show that intervening on a feature will cause subscription.
5. **One-hot encoding.** Categorical variables are represented by multiple encoded columns. Grouped SHAP importance sums mean absolute values across those columns and should be interpreted as a descriptive grouping, not a perfectly comparable causal or intrinsic importance score.
6. **Single local example.** The local explanation is for one validation observation selected because it was the highest-scoring true positive. It is illustrative and not representative of every customer.
7. **Prediction-time feature availability.** `duration` was dropped because call duration is only known after the call and would leak information into a pre-call prediction. `pdays` was recoded into `contacted_before` because 999 means the customer had not previously been contacted.

## 7. Reproducibility

Run the script from the repository root in the pinned project environment:

```bash
pip install -r requirements.txt
python notebooks/02_shap_analysis.py
```

The script recreates the SHAP figures and CSV summaries under `figures/` and `results/`.

