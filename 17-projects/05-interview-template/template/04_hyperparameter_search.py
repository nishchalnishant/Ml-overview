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
import torch
from scipy.stats import loguniform, randint
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn

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


# ---------------------------------------------------------------------------
# DL-specific HPO: Optuna (Bayesian/TPE) tuning a PyTorch training loop.
# ---------------------------------------------------------------------------
# SAY OUT LOUD: "For the neural net I'd use Optuna instead of RandomizedSearchCV — it's
# Bayesian (TPE by default), so each trial uses the results of prior trials to pick the next
# point, converging faster than random search when trials are expensive (each one is a full
# training run). I also use a pruner so bad trials get killed early instead of running to
# completion."

def build_mlp(n_features, hidden, dropout):
    return nn.Sequential(
        nn.Linear(n_features, hidden),
        nn.ReLU(),
        nn.BatchNorm1d(hidden),
        nn.Dropout(dropout),
        nn.Linear(hidden, 1),
    )


def optuna_objective(trial, X_train, y_train, X_val, y_val):
    # TODO(interview): state the space out loud before defining it.
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    hidden = trial.suggest_categorical("hidden", [32, 64, 128])
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

    model = build_mlp(X_train.shape[1], hidden, dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    x_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    x_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    for epoch in range(30):
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(model(x_train_t).squeeze(-1), y_train_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(x_val_t).squeeze(-1), y_val_t).item()

        # Prune unpromising trials early — saves compute vs. letting every trial run to completion.
        trial.report(val_loss, epoch)
        if trial.should_prune():
            import optuna
            raise optuna.TrialPruned()

    return val_loss


def run_optuna_search(n_trials=15):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    data = load_breast_cancer(as_frame=True)
    df = data.frame.rename(columns={"target": "label"})
    X, y = df.drop(columns=["label"]).values, df["label"].values

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler().fit(X_train)
    X_train, X_val = scaler.transform(X_train), scaler.transform(X_val)

    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
    study.optimize(
        lambda trial: optuna_objective(trial, X_train, y_train, X_val, y_val),
        n_trials=n_trials,
    )

    print(f"Best trial params: {study.best_trial.params}")
    print(f"Best val loss: {study.best_value:.4f}")
    return study


if __name__ == "__main__":
    main()
    print("\n--- DL-specific HPO with Optuna ---")
    run_optuna_search()
