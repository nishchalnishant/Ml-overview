"""Phase 4 — Hyperparameter selection.

SAY OUT LOUD before coding:
- "I use random search over grid search — with N hyperparameters, random search covers the space
   far more efficiently for the same trial budget, since not all hyperparameters matter equally."
- "I state the search space and trial budget up front rather than searching indefinitely."
- "I tune on a validation split (or CV, see 05), never on the test set — test is touched exactly
   once, at the end."
- "For the DL model, learning rate and weight decay usually matter most; for GBTs, n_estimators x
   learning_rate trade-off and max_depth matter most."

Runs standalone on the bundled toy dataset.
"""
import numpy as np
from scipy.stats import loguniform, randint
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split

# TODO(interview): state the budget out loud before running — e.g. "20 trials, 3-fold CV,
# optimizing ROC-AUC, ~2 minutes on this machine."
N_TRIALS = 20

SEARCH_SPACE = {
    "n_estimators": randint(50, 300),
    "learning_rate": loguniform(1e-3, 3e-1),
    "max_depth": randint(2, 6),
    "subsample": [0.6, 0.8, 1.0],
}


def main():
    data = load_breast_cancer(as_frame=True)
    df = data.frame.rename(columns={"target": "label"})
    X, y = df.drop(columns=["label"]), df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    search = RandomizedSearchCV(
        estimator=GradientBoostingClassifier(random_state=42),
        param_distributions=SEARCH_SPACE,
        n_iter=N_TRIALS,
        scoring="roc_auc",
        cv=3,  # TODO(interview): swap for the CV strategy chosen in 05 if not plain k-fold
        random_state=42,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)

    print(f"Best params: {search.best_params_}")
    print(f"Best CV ROC-AUC: {search.best_score_:.4f}")
    test_auc = search.score(X_test, y_test)
    print(f"Held-out test ROC-AUC with best params: {test_auc:.4f}")

    # TODO(interview): for the DL model, mention Bayesian optimization (Optuna) as the
    # natural upgrade from random search once the trial budget is large enough to matter,
    # and mention learning-rate range test / warmup+cosine schedule as DL-specific tactics.
    return search


if __name__ == "__main__":
    main()
