"""Phase 5 — Cross-validation strategy.

SAY OUT LOUD before coding:
- "The CV strategy is determined by the row grain established in `00_problem_framing.md`, not
   chosen by default."
- "Imbalanced classification -> StratifiedKFold, so every fold preserves the class ratio."
- "Multiple rows share an entity (user, patient, device) -> GroupKFold on that entity ID, or the
   split leaks identity across train/val and inflates the score."
- "Time-dependent data -> TimeSeriesSplit / a fixed time-based cutoff, never random shuffling,
   or the model gets to 'see the future'."

Runs standalone on the bundled toy dataset (i.i.d. rows -> StratifiedKFold is correct here).
"""
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import (
    GroupKFold,
    StratifiedKFold,
    TimeSeriesSplit,
    cross_val_score,
)


def choose_cv_strategy(row_grain: str, is_temporal: bool, is_imbalanced: bool):
    """TODO(interview): call this out loud with the actual answers from 00_problem_framing.md."""
    if is_temporal:
        return TimeSeriesSplit(n_splits=5)
    if row_grain == "grouped":
        return GroupKFold(n_splits=5)
    if is_imbalanced:
        return StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def main():
    data = load_breast_cancer(as_frame=True)
    df = data.frame.rename(columns={"target": "label"})
    X, y = df.drop(columns=["label"]), df["label"]

    # TODO(interview): set these from the answers in 00_problem_framing.md.
    cv = choose_cv_strategy(row_grain="i.i.d.", is_temporal=False, is_imbalanced=False)

    model = GradientBoostingClassifier(random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")

    print(f"CV strategy: {type(cv).__name__}")
    print(f"Fold ROC-AUC scores: {np.round(scores, 4)}")
    print(f"Mean +/- std: {scores.mean():.4f} +/- {scores.std():.4f}")

    # TODO(interview): if using GroupKFold, pass groups=<entity_id_column> to cross_val_score.
    return scores


if __name__ == "__main__":
    main()
