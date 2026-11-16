"""DBA5106 Project 2 - Bank Marketing: starter code (load, clean, split, cost function)."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_URL = ("https://raw.githubusercontent.com/fabiorcampos/Bank-Marketing/"
            "master/data/bank-additional-full.csv")

# ---- 1. Load ---------------------------------------------------------------
df = pd.read_csv(DATA_URL, sep=";")          # note: separator is ';'
print("Shape:", df.shape)
print("Subscription rate: {:.1%}".format((df["y"] == "yes").mean()))

# ---- 2. Clean --------------------------------------------------------------
df["y"] = (df["y"] == "yes").astype(int)      # 1 = subscribed
df = df.drop(columns=["duration"])            # leakage: only known after the call
df["contacted_before"] = (df["pdays"] != 999).astype(int)   # 999 = never contacted
df = df.drop(columns=["pdays"])
# "unknown" is kept as its own category (no rows dropped)

# ---- 3. Time-based split (rows are in date order) --------------------------
n = len(df)
train = df.iloc[: int(0.6 * n)]
val   = df.iloc[int(0.6 * n): int(0.8 * n)]
test  = df.iloc[int(0.8 * n):]
for name, part in [("train", train), ("val", val), ("test", test)]:
    print(f"{name:5s} n={len(part):6d}  subscription rate={part['y'].mean():.1%}")

X_train, y_train = train.drop(columns="y"), train["y"]
X_val,   y_val   = val.drop(columns="y"),   val["y"]
X_test,  y_test  = test.drop(columns="y"),  test["y"]

# ---- 4. Preprocessing (fit on train only) ----------------------------------
cat_cols = X_train.select_dtypes(exclude="number").columns.tolist()
num_cols = [c for c in X_train.columns if c not in cat_cols]
prep = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])
Xtr = prep.fit_transform(X_train)
Xva = prep.transform(X_val)
Xte = prep.transform(X_test)
print("Features after encoding:", Xtr.shape[1])

# ---- 5. Evaluation loss (to agree as a team) -------------------------------
COST_FP = 5     # cost of calling a customer who does not subscribe (placeholder)
COST_FN = 50    # margin lost by not calling a customer who would subscribe (placeholder)

def business_loss(y_true, p_hat, threshold):
    """Average cost per customer when we call everyone with p_hat >= threshold."""
    y_true = np.asarray(y_true)
    call = np.asarray(p_hat) >= threshold
    fp = np.sum(call & (y_true == 0))
    fn = np.sum(~call & (y_true == 1))
    return (COST_FP * fp + COST_FN * fn) / len(y_true)

# Baselines to beat on validation
print("Call everyone:", business_loss(y_val, np.ones(len(y_val)), 0.5))
print("Call no one  :", business_loss(y_val, np.zeros(len(y_val)), 0.5))
