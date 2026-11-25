"""
Phase 4 — Global and Local Tree SHAP for the selected GB (log loss) model.

Run from the repository root:
    python notebooks/02_shap_analysis.py

Requires the repository's pinned environment and:
    data/bank-additional-full.csv

Outputs: figures in figures/shap/, CSV summaries in results/.
"""
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import bank

MODEL_PATH = ROOT / "models" / "boosting_for_shap.joblib"
FINAL_MODEL_PATH = ROOT / "models" / "final_model.joblib"
DATA_PATH = ROOT / "data" / "bank-additional-full.csv"
FIGURES = ROOT / "figures" / "shap"
RESULTS = ROOT / "results"
FIGURES.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(exist_ok=True)

BACKGROUND_SIZE = 1000   # rows of the training set used as the SHAP background
EPS = 1e-6               # same probability clipping as notebook 01

# 1. Recreate the exact data cleaning and stratified random split from bank.py.
raw = pd.read_csv(DATA_PATH, sep=";")
clean = bank.clean(raw)
train, val, test = bank.split(clean, seed=42)
X_train, _ = bank.xy(train)
X_val, y_val = bank.xy(val)

# 2. Load the saved fitted pipeline. Do not retrain the model.
pipe = joblib.load(MODEL_PATH)
preprocessor = pipe[:-1]
# Compatibility for loading a ColumnTransformer pickled under a different sklearn release.
if not hasattr(preprocessor, "force_int_remainder_cols"):
    preprocessor.force_int_remainder_cols = False
tree_model = pipe[-1]

X_train_transformed = preprocessor.transform(X_train)
X_val_transformed = preprocessor.transform(X_val)
feature_names = preprocessor.get_feature_names_out()

X_train_df = pd.DataFrame(X_train_transformed, columns=feature_names)
X_val_df = pd.DataFrame(X_val_transformed, columns=feature_names)

# Verify that this is the saved model associated with the reported validation AUC.
validation_score = tree_model.predict_proba(X_val_transformed)[:, 1]   # raw boosting score
validation_auc = roc_auc_score(y_val, validation_score)
print(f"Reproduced validation AUC: {validation_auc:.9f}")

# The final model = this boosting model + a calibration map + a threshold (notebook 01).
# The threshold was chosen on CALIBRATED probabilities, so decisions below use them.
final = joblib.load(FINAL_MODEL_PATH)
assert np.allclose(final["pipeline"].predict_proba(X_val)[:, 1], validation_score), \
    "final_model.joblib does not contain the same boosting model"
threshold = float(final["threshold"])
validation_probability = np.clip(final["calibrator"].predict(validation_score), EPS, 1 - EPS)
print(f"Final model: {final['name']}, calibration = {final['calibration']}, threshold = {threshold}")

# 3. Tree SHAP on the raw boosting score (probability space), with a fixed training background.
#    Tree SHAP explains the tree model itself; the isotonic calibration map is applied afterwards
#    and only rescales the score monotonically, so it does not change which features drive it.
background = X_train_df.sample(n=BACKGROUND_SIZE, random_state=42)
explainer = shap.TreeExplainer(
    tree_model,
    data=background,
    feature_perturbation="interventional",
    model_output="probability",
)
shap_values = explainer(X_val_df)
base_value = float(np.mean(shap_values.base_values))
print(f"Base value (mean raw score on the background): {base_value:.6f} | "
      f"mean raw score on validation: {validation_score.mean():.6f}")

# 4. Global SHAP: encoded-feature beeswarm and mean-absolute bar plot.
plt.figure()
shap.plots.beeswarm(shap_values, max_display=20, show=False)
plt.title("Global Tree SHAP — Gradient Boosting (Log Loss)")
plt.tight_layout()
plt.savefig(FIGURES / "global_tree_shap_beeswarm.png", dpi=220, bbox_inches="tight")
plt.close()

plt.figure()
shap.plots.bar(shap_values, max_display=20, show=False)
plt.title("Global Tree SHAP — Mean Absolute SHAP Value")
plt.tight_layout()
plt.savefig(FIGURES / "global_tree_shap_bar.png", dpi=220, bbox_inches="tight")
plt.close()

# Map one-hot encoded columns back to their original input variable.
categorical_columns = X_train.select_dtypes(exclude="number").columns.tolist()
def original_feature(encoded_name):
    if encoded_name.startswith("num__"):
        return encoded_name[len("num__"):]
    if encoded_name.startswith("cat__"):
        remainder = encoded_name[len("cat__"):]
        for column in sorted(categorical_columns, key=len, reverse=True):
            if remainder == column or remainder.startswith(column + "_"):
                return column
    return encoded_name

groups = [original_feature(str(name)) for name in feature_names]
mean_abs = np.abs(shap_values.values).mean(axis=0)

encoded_importance = pd.DataFrame({
    "encoded_feature": feature_names,
    "mean_abs_shap": mean_abs,
    "mean_shap": shap_values.values.mean(axis=0),
}).sort_values("mean_abs_shap", ascending=False)
encoded_importance.to_csv(RESULTS / "global_shap_feature_importance.csv", index=False)

# Grouped importance: first add up each customer's SHAP values over the one-hot columns of the
# same original feature (SHAP values are additive, so this is that feature's contribution for
# that customer), then take the mean absolute value over customers. Summing the per-column
# mean |SHAP| instead would overstate features that have many categories.
per_customer_grouped = pd.DataFrame(shap_values.values, columns=groups).T.groupby(level=0).sum().T
grouped_importance = (
    per_customer_grouped.abs().mean()
    .rename("mean_abs_shap").rename_axis("original_feature").reset_index()
    .sort_values("mean_abs_shap", ascending=False)
)
grouped_importance.to_csv(RESULTS / "global_shap_grouped_importance.csv", index=False)

plot_data = grouped_importance.sort_values("mean_abs_shap", ascending=True)
fig, ax = plt.subplots(figsize=(9, 7))
ax.barh(plot_data["original_feature"], plot_data["mean_abs_shap"])
ax.set_xlabel("Mean |SHAP| per original feature (one-hot columns added up per customer)")
ax.set_title("Global Tree SHAP — Grouped by Original Feature")
fig.tight_layout()
fig.savefig(FIGURES / "global_tree_shap_grouped_features.png", dpi=220, bbox_inches="tight")
plt.close(fig)

# 5. Local SHAP for two validation customers, using the final model's decision
#    (calibrated probability >= threshold):
#    - "top":        the highest-scoring true positive (clearest case);
#    - "borderline": the called customer with the lowest score (the marginal call).
actual = np.asarray(y_val)
predicted_class = (validation_probability >= threshold).astype(int)
true_positive_positions = np.where((actual == 1) & (predicted_class == 1))[0]
called_positions = np.where(predicted_class == 1)[0]
if len(true_positive_positions) == 0:
    raise RuntimeError("No true-positive validation observation found at the recorded threshold.")

examples = {
    "top": ("Validation Customer (Highest-Scoring True Positive)",
            true_positive_positions[np.argmax(validation_score[true_positive_positions])]),
    "borderline": ("Validation Customer (Borderline Call, Just Above the Threshold)",
                   called_positions[np.argmin(validation_score[called_positions])]),
}

profiles = []
for key, (label, position) in examples.items():
    suffix = "" if key == "top" else "_borderline"
    explanation = shap_values[position]

    plt.figure()
    shap.plots.waterfall(explanation, max_display=15, show=False)
    plt.title(f"Local Tree SHAP — {label}")
    plt.tight_layout()
    plt.savefig(FIGURES / f"local_tree_shap_waterfall{suffix}.png", dpi=220, bbox_inches="tight")
    plt.close()

    # Aggregate one-hot columns to original variables for an easier-to-read local plot.
    local_grouped = (
        pd.DataFrame({"original_feature": groups, "shap_value": explanation.values})
        .groupby("original_feature")["shap_value"].sum()
    )
    local_grouped = local_grouped.reindex(local_grouped.abs().sort_values(ascending=False).index)
    raw_customer = X_val.iloc[position]
    grouped_explanation = shap.Explanation(
        values=local_grouped.values,
        base_values=float(explanation.base_values),
        data=np.asarray([raw_customer[name] for name in local_grouped.index], dtype=object),
        feature_names=local_grouped.index.tolist(),
    )
    plt.figure(figsize=(10, 8))
    shap.plots.waterfall(grouped_explanation, max_display=15, show=False)
    plt.title(f"Local Tree SHAP — {label}\n(grouped by original feature; raw boosting score)")
    plt.tight_layout()
    plt.savefig(FIGURES / f"local_tree_shap_grouped_waterfall{suffix}.png", dpi=220, bbox_inches="tight")
    plt.close()

    local_grouped.rename("grouped_shap_value").to_csv(
        RESULTS / f"local_shap_grouped_contributions{suffix}.csv"
    )
    local_encoded = pd.DataFrame({
        "encoded_feature": feature_names,
        "transformed_feature_value": X_val_transformed[position],
        "shap_value": explanation.values,
    })
    local_encoded["abs_shap"] = local_encoded["shap_value"].abs()
    local_encoded.sort_values("abs_shap", ascending=False).to_csv(
        RESULTS / f"local_shap_contributions{suffix}.csv", index=False
    )

    profile = raw_customer.to_frame().T
    profile.insert(0, "example", key)
    profile.insert(1, "validation_original_row_index", int(X_val.index[position]))
    profile["actual_y"] = int(actual[position])
    profile["raw_boosting_score"] = float(validation_score[position])
    profile["calibrated_probability"] = float(validation_probability[position])
    profile["validation_threshold"] = threshold
    profile["predicted_class"] = int(predicted_class[position])
    profile["shap_base_value"] = float(explanation.base_values)
    profiles.append(profile)

    print(f"\n[{key}] original row index {int(X_val.index[position])} | actual y {int(actual[position])}")
    print(f"  raw boosting score {validation_score[position]:.6f} | "
          f"calibrated probability {validation_probability[position]:.6f} | threshold {threshold}")
    print(f"  base value {float(explanation.base_values):.6f} + sum of SHAP "
          f"{float(explanation.values.sum()):.6f} = {float(explanation.base_values) + float(explanation.values.sum()):.6f}")

pd.concat(profiles).to_csv(RESULTS / "local_customer_profile.csv", index=False)
print("\nSHAP figures saved to figures/shap/ and CSV summaries to results/.")
