# Phase 4 — Interpretability: Global and Local Tree SHAP

## 1. Model and analysis scope

This analysis explains the repository's selected black-box model: **scikit-learn `GradientBoostingClassifier(loss="log_loss")`** (Gradient Boosting with Log Loss). It is not an `XGBClassifier` from the XGBoost package.

- Validation AUC recorded in `results/model_comparison.csv`: **0.8060476** (rounded: **0.8060**).
- Validation set: **8,238** observations from the repository's stratified random 60/20/20 split (`random_state=42`).
- Model input: **62 transformed features** (9 numerical columns plus one-hot encoded categorical columns).
- Original predictors: 19 after dropping `duration` and replacing `pdays` with `contacted_before`.

The saved model was loaded from `models/boosting_for_shap.joblib`. The original data preparation and split were reproduced using `src/bank.py`. The saved pipeline's validation AUC was recomputed as **0.806047602**, matching the recorded value.

### What SHAP explains: raw score vs final probability

The final model (`models/final_model.joblib`) has three parts: this boosting model, an **isotonic calibration map**, and a **threshold of 0.0825**. The threshold was chosen on the *calibrated* probabilities.

- Tree SHAP explains the tree model itself, so the SHAP values add up to the **raw boosting score**.
- The calibration map is applied afterwards. It only rescales the score monotonically (a higher score always means a higher calibrated probability), so it does not change which features push a customer up or down.
- All call / no-call decisions in this analysis use the calibrated probability and the 0.0825 threshold, exactly as the final model does.

## 2. SHAP method

- Explainer: `shap.TreeExplainer`
- Feature perturbation: `interventional`
- Model output: `probability` (the raw boosting score)
- Background data: a reproducible random sample of **1,000 rows** from the transformed training set (`random_state=42`). A 100-row sample was tried first, but its base value (0.140) and feature ranking moved noticeably with the sample; 1,000 rows gives a stable result.
- Explanation set for the global analysis: all **8,238 validation observations**.

The SHAP base value (mean raw score on the background) is **0.0977**; the mean raw score on the validation set is 0.1122.

### Feature grouping

The fitted preprocessing pipeline expands categorical variables into one-hot encoded columns, so the beeswarm plot shows encoded features. For the grouped results, each customer's SHAP values are first **added up over the one-hot columns of the same original predictor** (SHAP values are additive, so this is that predictor's contribution for that customer), and then the mean absolute value is taken over customers. Adding up the per-column mean absolute values instead would overstate predictors with many categories.

## 3. Global Tree SHAP

Files:
- `figures/shap/global_tree_shap_beeswarm.png`
- `figures/shap/global_tree_shap_grouped_features.png`
- `figures/shap/global_tree_shap_bar.png`
- `results/global_shap_feature_importance.csv`
- `results/global_shap_grouped_importance.csv`

### Main findings

The most influential original predictors by grouped mean absolute SHAP value were:

| Rank | Predictor | Mean \|SHAP\| | Type |
|---:|---|---:|---|
| 1 | **nr.employed** | 0.0457 | economic indicator |
| 2 | **euribor3m** | 0.0126 | economic indicator |
| 3 | **cons.conf.idx** | 0.0125 | economic indicator |
| 4 | **contacted_before** | 0.0101 | contact history |
| 5 | **contact** (cellular / telephone) | 0.0097 | contact channel |
| 6 | **month** | 0.0081 | timing |
| 7 | **poutcome** | 0.0068 | previous campaign |
| 8 | **day_of_week** | 0.0056 | timing |

`nr.employed` is the strongest driver by a wide margin, and the three economic indicators take the top three places. Customer contact history (`contacted_before`, `poutcome`) and the contact channel are the strongest customer-level predictors.

The beeswarm plot shows the direction of each encoded feature's effect. For example, lower `nr.employed` values push the predicted subscription probability up and higher values push it down; customers contacted before, reached on a mobile phone, or with a successful previous campaign get higher scores. These are associations learned by the model, not evidence that changing a feature would cause a customer to subscribe.

### Why the economic indicators dominate

`nr.employed` takes only **11 different values** in the whole dataset, one per period of the campaign (May 2008 – November 2010). The subscription rate in those periods ranges from about 3% to 57%. So the model largely uses `nr.employed` (and, to a lesser extent, `euribor3m` and `cons.conf.idx`) to recognise **which period a call was made in**, rather than anything about the customer. This matches the time-split analysis in `notebooks/01_models_and_threshold.ipynb` (section 9b): the effect of these indicators reverses over time, and boosting drops to an AUC of about 0.47–0.54 when trained on early periods and tested on later ones.

## 4. Local Tree SHAP

Two validation customers are explained: the clearest positive case and a borderline call.

Files:
- `figures/shap/local_tree_shap_waterfall.png`, `figures/shap/local_tree_shap_grouped_waterfall.png` (top case)
- `figures/shap/local_tree_shap_waterfall_borderline.png`, `figures/shap/local_tree_shap_grouped_waterfall_borderline.png` (borderline case)
- `results/local_shap_contributions.csv`, `results/local_shap_grouped_contributions.csv` (top case)
- `results/local_shap_contributions_borderline.csv`, `results/local_shap_grouped_contributions_borderline.csv` (borderline case)
- `results/local_customer_profile.csv` (both customers)

The observations are anonymised and identified only by their original row index in the public dataset.

### 4.1 Top case: highest-scoring true positive

| Item | Value |
|---|---|
| Original row index | 39498 |
| Profile | 80, retired, divorced, basic 4y education, mobile phone, previous campaign = success |
| Actual outcome | Yes (`y=1`) |
| Raw boosting score | 0.9516 |
| Calibrated probability (final model) | 1.00 |
| Decision at threshold 0.0825 | Call |

`0.0977 (base value) + 0.8539 (sum of SHAP values) = 0.9516 (raw boosting score)`

Largest grouped contributions:

| Predictor | Value for this customer | SHAP contribution |
|---|---|---:|
| nr.employed | 5008.7 | +0.3381 |
| euribor3m | 0.644 | +0.1493 |
| contacted_before | 1 (yes) | +0.1482 |
| poutcome | success | +0.0769 |
| age | 80 | +0.0498 |
| campaign | 1 | +0.0304 |
| previous | 2 | +0.0280 |

About half of this customer's score comes from the two economic indicators, i.e. from *when* they were called. The customer-level reasons are a successful previous campaign, having been contacted before, their age, and needing only one call this campaign.

### 4.2 Borderline case: the marginal call

This is the called customer with the lowest score, i.e. the customer just above the threshold.

| Item | Value |
|---|---|
| Original row index | 31131 |
| Profile | 39, blue-collar, married, professional course, mobile phone, never contacted before |
| Actual outcome | No (`y=0`) |
| Raw boosting score | 0.0802 |
| Calibrated probability (final model) | 0.0828 |
| Decision at threshold 0.0825 | Call (just above the threshold) |

`0.0977 (base value) − 0.0175 (sum of SHAP values) = 0.0802 (raw boosting score)`

The largest contribution is again `nr.employed` (−0.028), pulling the score down; never having been contacted before (−0.006) and age (−0.005) also pull it down, while calling on a Wednesday (+0.007) and `euribor3m` (+0.007) push it up. All the contributions are small, so this customer is close to average and only just clears the threshold. The call turned out to be wasted (the customer did not subscribe), which is the expected trade-off at the margin: with a wasted call costing 5 and a missed subscriber 50, calling customers with a probability of about 8–9% is still worth it on average.

These are two selected examples, not a summary of all customers. The contributions explain the model's prediction; they do not establish causal effects.

## 5. Business value

The model can support campaign prioritisation by ranking customers by predicted subscription probability and calling those above the cost-based threshold, instead of calling everyone.

The SHAP results can help:
- show which customer-level information the model uses: contact history (`contacted_before`, `poutcome`), contact channel, age and number of calls this campaign;
- explain to the marketing team why a particular customer was or was not selected for a call;
- **flag a risk**: the model leans most heavily on economic indicators that mainly identify the time period. This is not something the bank can act on in campaign planning. It means the model must be monitored and retrained as conditions change, and a model without these indicators should be considered if stability matters more than accuracy on the random split.

SHAP explanations should be used as model diagnostics and communication aids, not as causal claims or as a substitute for campaign experimentation.

## 6. Model limitations and risks

1. **Historical and geographic scope.** The dataset represents a Portuguese bank's telemarketing campaigns from May 2008 to November 2010. The learned relationships may not transfer to current campaigns, other countries, or digital marketing channels.
2. **Strong temporal drift.** The repository's time-ordered evaluation reports AUC **0.4652** on the next 20% and **0.5417** on the last 20%, compared with **0.8060** on the stratified random validation split. The subscription rate also rises from **4.81%** in the first 60% to **11.07%** in the next 20% and **30.83%** in the last 20%.
3. **Economic indicators act as period labels.** `nr.employed` dominates the global SHAP ranking but takes only 11 values, one per period. Its effect reflects which period a call was made in and reverses over time (notebook 01, section 9b).
4. **Association, not causation.** SHAP attributes the fitted model output to input features under the chosen explainer assumptions. It does not show that intervening on a feature will cause subscription.
5. **Correlated features share credit.** `contacted_before`, `poutcome` and `previous` carry overlapping information, as do the economic indicators. SHAP splits the credit between correlated features, so their individual ranks should not be over-interpreted.
6. **Raw score, not calibrated probability.** SHAP values add up to the raw boosting score. The final decision uses the calibrated probability, which is a monotone rescaling of that score.
7. **Selected local examples.** The two local explanations are illustrative and not representative of every customer.
8. **Prediction-time feature availability.** `duration` was dropped because call duration is only known after the call and would leak information into a pre-call prediction. `pdays` was recoded into `contacted_before` because 999 means the customer had not previously been contacted.

## 7. Reproducibility

Run the script from the repository root in the pinned project environment:

```bash
pip install -r requirements.txt
python notebooks/02_shap_analysis.py
```

The script recreates the SHAP figures under `figures/shap/` and the CSV summaries under `results/`.
