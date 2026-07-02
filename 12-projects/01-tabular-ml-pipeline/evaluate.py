"""Evaluate the saved pipeline on the held-out test set."""
import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

TARGET = "churn"


def main() -> None:
    pipe = joblib.load("model.joblib")
    df = pd.read_csv("data/test_holdout.csv")
    X_test = df.drop(columns=[TARGET])
    y_test = df[TARGET]

    proba = pipe.predict_proba(X_test)[:, 1]
    preds = (proba > 0.5).astype(int)

    print(f"ROC-AUC: {roc_auc_score(y_test, proba):.4f}\n")
    print("Classification report:")
    print(classification_report(y_test, preds, target_names=["retained", "churned"]))
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(confusion_matrix(y_test, preds))


if __name__ == "__main__":
    main()
