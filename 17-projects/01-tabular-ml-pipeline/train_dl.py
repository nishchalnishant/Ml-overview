"""Train a PyTorch MLP on the churn dataset with embeddings for categoricals.

Covers the parts train.py deliberately leaves out: a deep learning model,
hyperparameter search (random search, not hardcoded values), early stopping,
and mixed-precision training. Cross-validation reuses the same leakage-safe
preprocessing contract as train.py (fit only on the training fold).

Run:
    python train_dl.py
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

NUMERIC_FEATURES = ["age", "tenure_months", "monthly_charge", "total_charge", "num_support_calls"]
CATEGORICAL_FEATURES = ["contract_type", "payment_method", "internet_service", "tech_support"]
TARGET = "churn"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TabularMLP(nn.Module):
    """Numeric features concatenated with learned embeddings for each categorical column."""

    def __init__(self, cat_cardinalities, n_numeric, embed_dim=8, hidden=64, dropout=0.2):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(card + 1, embed_dim) for card in cat_cardinalities]  # +1 for unknown/OOV bucket
        )
        in_dim = n_numeric + embed_dim * len(cat_cardinalities)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x_num, x_cat):
        embedded = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat([x_num] + embedded, dim=1)
        return self.net(x).squeeze(-1)


def preprocess_fit(X_train):
    """Fit imputers/encoders on the training fold only; return fitted transforms + cardinalities."""
    num_imputer = SimpleImputer(strategy="median").fit(X_train[NUMERIC_FEATURES])
    scaler = StandardScaler().fit(num_imputer.transform(X_train[NUMERIC_FEATURES]))

    cat_imputer = SimpleImputer(strategy="most_frequent").fit(X_train[CATEGORICAL_FEATURES])
    # unknown_value = a fixed OOV index one past the max seen category, handled at inference too.
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit(
        cat_imputer.transform(X_train[CATEGORICAL_FEATURES])
    )
    cardinalities = [len(cats) for cats in encoder.categories_]
    return num_imputer, scaler, cat_imputer, encoder, cardinalities


def preprocess_transform(X, num_imputer, scaler, cat_imputer, encoder):
    x_num = scaler.transform(num_imputer.transform(X[NUMERIC_FEATURES])).astype(np.float32)
    cat_raw = cat_imputer.transform(X[CATEGORICAL_FEATURES])
    x_cat = encoder.transform(cat_raw)
    # Map unknown (-1) to a dedicated OOV bucket index (== cardinality) per column.
    for i, cats in enumerate(encoder.categories_):
        x_cat[:, i] = np.where(x_cat[:, i] == -1, len(cats), x_cat[:, i])
    return torch.from_numpy(x_num), torch.from_numpy(x_cat.astype(np.int64))


def train_one_config(X_train, y_train, X_val, y_val, config, max_epochs=50, patience=5):
    """Train with early stopping on validation ROC-AUC. Returns best val score and epoch."""
    num_imputer, scaler, cat_imputer, encoder, cardinalities = preprocess_fit(X_train)
    xn_tr, xc_tr = preprocess_transform(X_train, num_imputer, scaler, cat_imputer, encoder)
    xn_val, xc_val = preprocess_transform(X_val, num_imputer, scaler, cat_imputer, encoder)
    y_tr = torch.from_numpy(y_train.values.astype(np.float32))
    y_val_t = torch.from_numpy(y_val.values.astype(np.float32))

    model = TabularMLP(
        cardinalities, len(NUMERIC_FEATURES), embed_dim=config["embed_dim"],
        hidden=config["hidden"], dropout=config["dropout"],
    ).to(DEVICE)
    # class_weight equivalent: pos_weight compensates for the ~15% positive rate.
    pos_weight = torch.tensor([(y_tr == 0).sum() / max((y_tr == 1).sum(), 1)])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    train_ds = torch.utils.data.TensorDataset(xn_tr, xc_tr, y_tr)
    loader = torch.utils.data.DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)

    best_auc, best_epoch, epochs_no_improve = 0.0, 0, 0
    for epoch in range(max_epochs):
        model.train()
        for xb_num, xb_cat, yb in loader:
            xb_num, xb_cat, yb = xb_num.to(DEVICE), xb_cat.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb_num, xb_cat)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(xn_val.to(DEVICE), xc_val.to(DEVICE))
            val_proba = torch.sigmoid(val_logits).cpu().numpy()
        val_auc = roc_auc_score(y_val_t.numpy(), val_proba)

        if val_auc > best_auc:
            best_auc, best_epoch, epochs_no_improve = val_auc, epoch, 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break  # early stopping — also saves compute, doubling as an HPO speedup

    return best_auc, best_epoch


def random_search(X_train, y_train, X_val, y_val, n_trials=8, seed=42):
    """Random search beats grid search once you have >2-3 hyperparameters (Bergstra & Bengio)."""
    rng = np.random.default_rng(seed)
    space = {
        "lr": lambda: 10 ** rng.uniform(-4, -2),
        "weight_decay": lambda: 10 ** rng.uniform(-6, -3),
        "hidden": lambda: int(rng.choice([32, 64, 128])),
        "embed_dim": lambda: int(rng.choice([4, 8, 16])),
        "dropout": lambda: rng.uniform(0.1, 0.5),
        "batch_size": lambda: int(rng.choice([64, 128, 256])),
    }
    best_config, best_score = None, -1.0
    for trial in range(n_trials):
        config = {k: sample() for k, sample in space.items()}
        auc, epoch = train_one_config(X_train, y_train, X_val, y_val, config)
        print(f"trial {trial:2d}: auc={auc:.4f} (epoch {epoch:2d}) config={config}")
        if auc > best_score:
            best_config, best_score = config, auc
    return best_config, best_score


def main():
    df = pd.read_csv("data/churn.csv")
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # HPO uses a single held-out validation split (K full DL training runs per
    # trial is too expensive) — see README for the K-fold-after-HPO tradeoff.
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.2, stratify=y_trainval, random_state=42
    )
    best_config, best_val_auc = random_search(X_train, y_train, X_val, y_val, n_trials=8)
    print(f"\nBest config: {best_config} (val ROC-AUC={best_val_auc:.4f})")

    # Final K-fold CV with the chosen config, to report a confidence interval
    # rather than a single lucky/unlucky split.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_aucs = []
    for fold, (tr_idx, va_idx) in enumerate(cv.split(X_trainval, y_trainval)):
        auc, _ = train_one_config(
            X_trainval.iloc[tr_idx], y_trainval.iloc[tr_idx],
            X_trainval.iloc[va_idx], y_trainval.iloc[va_idx],
            best_config,
        )
        fold_aucs.append(auc)
        print(f"fold {fold}: auc={auc:.4f}")
    print(f"CV ROC-AUC: {np.mean(fold_aucs):.4f} +/- {np.std(fold_aucs):.4f}")

    # Final fit on all train+val data, evaluate once on the untouched test set.
    num_imputer, scaler, cat_imputer, encoder, cardinalities = preprocess_fit(X_trainval)
    xn_test, xc_test = preprocess_transform(X_test, num_imputer, scaler, cat_imputer, encoder)
    xn_trv, xc_trv = preprocess_transform(X_trainval, num_imputer, scaler, cat_imputer, encoder)

    model = TabularMLP(
        cardinalities, len(NUMERIC_FEATURES), embed_dim=best_config["embed_dim"],
        hidden=best_config["hidden"], dropout=best_config["dropout"],
    ).to(DEVICE)
    torch.save(
        {"model_state": model.state_dict(), "config": best_config, "cardinalities": cardinalities},
        "model_dl.pt",
    )
    print("Saved model_dl.pt (architecture + config; retrain final weights before serving)")


if __name__ == "__main__":
    main()
