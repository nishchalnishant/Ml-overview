"""Phase 1 — Feature Engineering.

SAY OUT LOUD before coding:
- "I'll split before fitting any transform, so nothing leaks from val/test into training stats."
- "Numeric features get scaled — neural nets are scale-sensitive, unlike trees."
- "High-cardinality categoricals (user ID, item ID) get embeddings later; low-cardinality ones
   get one-hot/ordinal encoding now."
- "Missing values get an explicit is_missing flag, not silent imputation — sparsity itself can be
   informative (e.g. a new user with no history)."

Runs as-is on the bundled toy dataset (sklearn breast-cancer). Swap `load_raw_data()` and the
NUMERIC_FEATURES / CATEGORICAL_FEATURES / TARGET constants for the real dataset in the interview.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_breast_cancer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# TODO(interview): replace with the real feature lists for the given dataset.
NUMERIC_FEATURES = None   # e.g. ["age", "tenure_months", "monthly_charge"]
CATEGORICAL_FEATURES = []  # e.g. ["contract_type", "payment_method"]
TARGET = None              # e.g. "churn"


def load_raw_data():
    """TODO(interview): replace with pd.read_csv(...) on the real dataset."""
    data = load_breast_cancer(as_frame=True)
    df = data.frame.rename(columns={"target": "label"})
    return df


def check_leakage(df: pd.DataFrame, target: str, label_time_col: str | None = None) -> list[str]:
    """Say out loud: name every column that could only be known after the label is decided.

    TODO(interview): inspect columns for post-label-window timestamps, IDs that encode the
    outcome, or aggregates computed over a window that includes the label period.
    """
    suspicious = [c for c in df.columns if "future" in c.lower() or "post_" in c.lower()]
    if suspicious:
        print(f"LEAKAGE WARNING — dropping suspicious columns: {suspicious}")
    return suspicious


def add_missingness_flags(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Explicit is_missing flags before imputation — sparsity can be signal, not noise."""
    df = df.copy()
    for c in cols:
        if df[c].isna().any():
            df[f"{c}_is_missing"] = df[c].isna().astype(int)
    return df


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    """Single leakage-safe preprocessing contract — fit only on the training fold."""
    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ])


def main():
    df = load_raw_data()
    target = TARGET or "label"
    numeric = NUMERIC_FEATURES or [c for c in df.columns if c != target]
    categorical = CATEGORICAL_FEATURES or []

    check_leakage(df, target)
    df = add_missingness_flags(df, numeric)

    # Split BEFORE fitting any transform — this ordering is the single most important line here.
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(columns=[target]), df[target], test_size=0.2, random_state=42,
        stratify=df[target] if df[target].nunique() < 20 else None,
    )

    preprocessor = build_preprocessor(numeric, categorical)
    preprocessor.fit(X_train)  # fit only on train
    X_train_t = preprocessor.transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    print(f"Train shape: {X_train_t.shape}, Test shape: {X_test_t.shape}")
    print(f"Target balance (train): {np.bincount(y_train.astype(int)) if y_train.nunique() < 20 else y_train.describe()}")
    return X_train, X_test, y_train, y_test, preprocessor


if __name__ == "__main__":
    main()
