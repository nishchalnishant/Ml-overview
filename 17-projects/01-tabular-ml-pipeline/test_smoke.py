"""Smoke tests: catch broken imports/APIs without running full training.

Run: pytest test_smoke.py -v
"""
import numpy as np
import pandas as pd
import pytest

from generate_data import generate
from train import NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET, build_preprocessor


def test_generate_data_shape_and_columns():
    df = generate()
    expected_cols = {"customer_id", *NUMERIC_FEATURES, *CATEGORICAL_FEATURES, TARGET}
    assert expected_cols.issubset(df.columns)
    assert len(df) > 0


def test_generate_data_has_missingness_and_imbalance():
    df = generate()
    assert df["total_charge"].isna().any()
    churn_rate = df[TARGET].mean()
    assert 0.05 < churn_rate < 0.30


def test_preprocessor_fits_and_transforms():
    df = generate()
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].head(200)
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(X)
    assert transformed.shape[0] == 200


def test_preprocessor_handles_unseen_category():
    df = generate()
    X_train = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].head(200)
    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)

    X_new = X_train.head(1).copy()
    X_new["contract_type"] = "never-seen-plan"
    # Should not raise — OneHotEncoder(handle_unknown="ignore")
    preprocessor.transform(X_new)


def test_end_to_end_tiny_fit_predict():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    df = generate()
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].head(300)
    y = df[TARGET].head(300)

    pipe = Pipeline(
        [
            ("preprocess", build_preprocessor()),
            ("model", LogisticRegression(max_iter=200)),
        ]
    )
    pipe.fit(X, y)
    proba = pipe.predict_proba(X)[:, 1]
    assert proba.shape[0] == 300
    assert ((proba >= 0) & (proba <= 1)).all()
