"""Phase 2 — Baseline model (before deep learning).

SAY OUT LOUD before coding:
- "I always fit a fast baseline first — linear/logistic regression and a gradient-boosted tree —
   before touching a neural net. It gives me a number to beat and catches pipeline bugs early."
- "On tabular data specifically, GBTs are a strong prior. If the DL model in `03` doesn't clearly
   beat this, that's a valid finding to report, not a failure of the DL model."
- "I'm reusing the exact preprocessor from `01` so the comparison is apples-to-apples."

Runs standalone on the bundled toy dataset.
"""
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# TODO(interview): import build_preprocessor from 01_feature_engineering instead of this
# inline scaler once the real dataset's categorical/numeric split is known.


def load_raw_data():
    data = load_breast_cancer(as_frame=True)
    df = data.frame.rename(columns={"target": "label"})
    return df


def fit_baselines(X_train, y_train):
    linear = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    linear.fit(X_train, y_train)

    # TODO(interview): swap for XGBoost/LightGBM if available — sklearn's GBT is the
    # dependency-free stand-in so this file runs anywhere.
    gbt = GradientBoostingClassifier(random_state=42)
    gbt.fit(X_train, y_train)

    return {"logistic_regression": linear, "gbt": gbt}


def evaluate(models: dict, X_test, y_test):
    results = {}
    for name, model in models.items():
        proba = model.predict_proba(X_test)[:, 1]
        results[name] = {
            "roc_auc": roc_auc_score(y_test, proba),
            "pr_auc": average_precision_score(y_test, proba),
        }
    return results


def main():
    df = load_raw_data()
    X, y = df.drop(columns=["label"]), df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = fit_baselines(X_train, y_train)
    results = evaluate(models, X_test, y_test)

    for name, metrics in results.items():
        print(f"{name}: ROC-AUC={metrics['roc_auc']:.4f}  PR-AUC={metrics['pr_auc']:.4f}")

    print("\nThis is the number the deep learning model in 03 needs to beat.")
    return results


if __name__ == "__main__":
    main()
