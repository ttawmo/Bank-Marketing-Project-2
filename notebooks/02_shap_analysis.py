"""
Phase 4 — Global and Local Tree SHAP for the selected GB (log loss) model.

Run from the repository root:
    python notebooks/02_shap_analysis.py

Requires the repository's pinned environment and:
    data/bank-additional-full.csv
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
DATA_PATH = ROOT / "data" / "bank-additional-full.csv"
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"
FIGURES.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

# 1. Recreate the exact data cleaning and stratified random split from bank.py.
raw = pd.read_csv(DATA_PATH, sep=";")
clean = bank.clean(raw)
train, val, test = bank.split(clean, seed=42)
X_train, _ = bank.xy(train)
X_val, y_val = bank.xy(val)

# 2. Load the saved fitted pipeline. Do not retrain the model.
pipe = joblib.load(MODEL_PATH)
preprocessor = pipe[:-1]
# Compatibility for loading a ColumnTransformer pickled under a newer sklearn release.
if not hasattr(preprocessor, "force_int_remainder_cols"):
    preprocessor.force_int_remainder_cols = False
tree_model = pipe[-1]

X_train_transformed = preprocessor.transform(X_train)
X_val_transformed = preprocessor.transform(X_val)
feature_names = preprocessor.get_feature_names_out()

X_train_df = pd.DataFrame(X_train_transformed, columns=feature_names)
X_val_df = pd.DataFrame(X_val_transformed, columns=feature_names)

# Verify that this is the saved model associated with the reported validation AUC.
validation_probability = tree_model.predict_proba(X_val_transformed)[:, 1]
validation_auc = roc_auc_score(y_val, validation_probability)
print(f"Reproduced validation AUC: {validation_auc:.9f}")

# 3. Tree SHAP in probability space, using a fixed training background sample.
background = X_train_df.sample(n=100, random_state=42)
explainer = shap.TreeExplainer(
    tree_model,
    data=background,
    feature_perturbation="interventional",
    model_output="probability",
)
shap_values = explainer(X_val_df)

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

grouped_importance = (
    pd.DataFrame({"original_feature": groups, "mean_abs_shap": mean_abs})
    .groupby("original_feature", as_index=False)["mean_abs_shap"].sum()
    .sort_values("mean_abs_shap", ascending=False)
)
grouped_importance.to_csv(RESULTS / "global_shap_grouped_importance.csv", index=False)

plot_data = grouped_importance.sort_values("mean_abs_shap", ascending=True)
fig, ax = plt.subplots(figsize=(9, 7))
ax.barh(plot_data["original_feature"], plot_data["mean_abs_shap"])
ax.set_xlabel("Sum of mean absolute SHAP values across encoded columns")
ax.set_title("Global Tree SHAP — Grouped by Original Feature")
fig.tight_layout()
fig.savefig(FIGURES / "global_tree_shap_grouped_features.png", dpi=220, bbox_inches="tight")
plt.close(fig)

# 5. Local SHAP: choose the highest-scoring true positive in validation.
threshold = 0.0825  # GB (log loss) threshold recorded in results/threshold_validation.csv
actual = np.asarray(y_val)
predicted_class = (validation_probability >= threshold).astype(int)
true_positive_positions = np.where((actual == 1) & (predicted_class == 1))[0]
if len(true_positive_positions) == 0:
    raise RuntimeError("No true-positive validation observation found at the recorded threshold.")

local_position = true_positive_positions[
    np.argmax(validation_probability[true_positive_positions])
]
local_explanation = shap_values[local_position]

plt.figure()
shap.plots.waterfall(local_explanation, max_display=15, show=False)
plt.title("Local Tree SHAP — Validation Customer (True Positive)")
plt.tight_layout()
plt.savefig(FIGURES / "local_tree_shap_waterfall.png", dpi=220, bbox_inches="tight")
plt.close()

# Also aggregate one-hot columns to original variables for an easier-to-read local plot.
local_grouped = (
    pd.DataFrame({"original_feature": groups, "shap_value": local_explanation.values})
    .groupby("original_feature")["shap_value"].sum()
)
local_grouped = local_grouped.reindex(
    local_grouped.abs().sort_values(ascending=False).index
)
raw_customer = X_val.iloc[local_position]
raw_values = [raw_customer[name] for name in local_grouped.index]
grouped_explanation = shap.Explanation(
    values=local_grouped.values,
    base_values=float(local_explanation.base_values),
    data=np.asarray(raw_values, dtype=object),
    feature_names=local_grouped.index.tolist(),
)
plt.figure(figsize=(10, 8))
shap.plots.waterfall(grouped_explanation, max_display=15, show=False)
plt.title("Local Tree SHAP — Grouped by Original Feature")
plt.tight_layout()
plt.savefig(FIGURES / "local_tree_shap_grouped_waterfall.png", dpi=220, bbox_inches="tight")
plt.close()

local_grouped.rename("grouped_shap_value").to_csv(
    RESULTS / "local_shap_grouped_contributions.csv"
)
local_encoded = pd.DataFrame({
    "encoded_feature": feature_names,
    "transformed_feature_value": X_val_transformed[local_position],
    "shap_value": local_explanation.values,
})
local_encoded["abs_shap"] = local_encoded["shap_value"].abs()
local_encoded.sort_values("abs_shap", ascending=False).to_csv(
    RESULTS / "local_shap_contributions.csv", index=False
)

profile = raw_customer.to_frame().T
profile.insert(0, "validation_original_row_index", int(X_val.index[local_position]))
profile["actual_y"] = int(actual[local_position])
profile["predicted_probability"] = float(validation_probability[local_position])
profile["validation_threshold"] = threshold
profile["predicted_class"] = int(predicted_class[local_position])
profile.to_csv(RESULTS / "local_customer_profile.csv", index=False)

print(f"Local example original row index: {int(X_val.index[local_position])}")
print(f"Actual y: {int(actual[local_position])}")
print(f"Predicted probability: {validation_probability[local_position]:.6f}")
print(f"Base value: {float(local_explanation.base_values):.6f}")
print(f"Sum of SHAP values: {float(local_explanation.values.sum()):.6f}")
print("SHAP figures and CSV summaries saved.")
