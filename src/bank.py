"""Shared data preparation for DBA5106 Project 2 (Bank Marketing).

Both notebooks import from here so they use exactly the same cleaning, split
and cost function.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "bank-additional-full.csv"
DATA_URL = ("https://raw.githubusercontent.com/fabiorcampos/Bank-Marketing/"
            "master/data/bank-additional-full.csv")
# Same file as UCI Bank Marketing (Moro, Cortez & Rita, 2014),
# https://doi.org/10.24432/C5K306 - checked byte-for-byte against the UCI archive.

SEED = 42

# ---- Evaluation loss (placeholders until agreed as a team) ------------------
COST_FP = 5     # calling a customer who does not subscribe (wasted call)
COST_FN = 50    # not calling a customer who would have subscribed (lost margin)

MACRO_COLS = ["emp.var.rate", "cons.price.idx", "cons.conf.idx",
              "euribor3m", "nr.employed"]


def load_raw():
    """Raw data in its original (date) order. Downloads once, then reads the local copy."""
    if not DATA_PATH.exists():
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        pd.read_csv(DATA_URL, sep=";").to_csv(DATA_PATH, sep=";", index=False)
    return pd.read_csv(DATA_PATH, sep=";")


def clean(df):
    """Target to 0/1, drop the leaking `duration`, recode `pdays`."""
    df = df.copy()
    df["y"] = (df["y"] == "yes").astype(int)               # 1 = subscribed
    df = df.drop(columns=["duration"])                     # only known after the call
    df["contacted_before"] = (df["pdays"] != 999).astype(int)  # 999 = never contacted
    df = df.drop(columns=["pdays"])
    return df                                              # "unknown" kept as a category


def split(df, seed=SEED):
    """Stratified random 60 / 20 / 20 split into train / validation / test."""
    train, rest = train_test_split(df, test_size=0.4, stratify=df["y"], random_state=seed)
    val, test = train_test_split(rest, test_size=0.5, stratify=rest["y"], random_state=seed)
    return train, val, test


def time_split(df):
    """Row-order (= date-order) 60 / 20 / 20 split, used only to show drift."""
    n = len(df)
    return df.iloc[: int(0.6 * n)], df.iloc[int(0.6 * n): int(0.8 * n)], df.iloc[int(0.8 * n):]


def xy(part):
    return part.drop(columns="y"), part["y"].to_numpy()


def make_preprocessor(X):
    """Scale numeric columns, one-hot encode categorical ones (fit on train only)."""
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]
    return ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ])


def business_loss(y_true, p_hat, threshold, cost_fp=COST_FP, cost_fn=COST_FN):
    """Average cost per customer when we call everyone with p_hat >= threshold."""
    y_true = np.asarray(y_true)
    call = np.asarray(p_hat) >= threshold
    fp = np.sum(call & (y_true == 0))
    fn = np.sum(~call & (y_true == 1))
    return (cost_fp * fp + cost_fn * fn) / len(y_true)
