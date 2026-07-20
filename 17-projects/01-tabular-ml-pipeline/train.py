"""Train and compare a baseline vs. gradient-boosted churn classifier.

Every preprocessing step lives inside a single sklearn Pipeline fit only on
the training fold — see 01-foundations/04-data-processing-and-eda.md section 6
(Data Leakage) and section 8 (Pipelines) for why this structure matters.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

NUMERIC_FEATURES = ["age", "tenure_months", "monthly_charge", "total_charge", "num_support_calls"]
CATEGORICAL_FEATURES = ["contract_type", "payment_method", "internet_service", "tech_support"]
TARGET = "churn"


def build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ]
    )


def main() -> None:
    df = pd.read_csv("data/churn.csv")
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    # Split BEFORE fitting any transform — the preprocessor below only ever
    # sees X_train during .fit().
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    preprocessor = build_preprocessor()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    candidates = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "xgboost": XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="auc",
            random_state=42,
        ),
    }

    results = {}
    fitted_pipelines = {}
    for name, model in candidates.items():
        pipe = Pipeline([("preprocess", preprocessor), ("model", model)])
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
        results[name] = scores
        pipe.fit(X_train, y_train)  # refit on full train for the held-out comparison below
        fitted_pipelines[name] = pipe
        print(f"{name:20s} CV ROC-AUC: {scores.mean():.4f} +/- {scores.std():.4f}")

    best_name = max(results, key=lambda k: results[k].mean())
    best_pipe = fitted_pipelines[best_name]

    test_proba = best_pipe.predict_proba(X_test)[:, 1]
    print(f"\nBest model: {best_name}")
    print(f"Test ROC-AUC: {roc_auc_score(y_test, test_proba):.4f}")
    print(f"Test PR-AUC:  {average_precision_score(y_test, test_proba):.4f}")

    joblib.dump(best_pipe, "model.joblib")
    X_test.assign(churn=y_test).to_csv("data/test_holdout.csv", index=False)
    print("Saved model.joblib and data/test_holdout.csv")


if __name__ == "__main__":
    main()
