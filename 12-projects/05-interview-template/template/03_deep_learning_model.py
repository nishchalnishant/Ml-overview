"""Phase 3 — Train a deep learning model.

SAY OUT LOUD before coding:
- "For tabular data I use an MLP with learned embeddings for categorical columns — each gets its
   own embedding table, concatenated with scaled numeric features."
- "Loss: BCEWithLogitsLoss for binary classification (numerically stable — no separate sigmoid),
   CrossEntropyLoss for multiclass, MSE/Huber for regression."
- "I use early stopping on a held-out validation split — both a regularizer and a compute saver."
- "BatchNorm + Dropout inside the MLP for regularization, same as any tabular DL setup."

Mirrors the TabularMLP pattern in ../../01-tabular-ml-pipeline/train_dl.py.
Runs standalone on the bundled toy dataset (all-numeric, so cat_cardinalities=[]).

REGRESSION / MULTICLASS VARIANT — say this out loud if the target isn't binary:
- Regression: loss_fn = nn.MSELoss() (or nn.HuberLoss() if outliers are a concern), final layer
  stays width 1, no sigmoid at inference, metric becomes MAE/RMSE instead of ROC-AUC.
- Multiclass: loss_fn = nn.CrossEntropyLoss(), final layer width = n_classes, targets are class
  indices (long dtype) not floats, metric becomes macro-F1 or multiclass ROC-AUC (ovr).
- Everything else — embeddings, BatchNorm/Dropout, early stopping, train/val/test split
  ordering — is unchanged.
"""
import numpy as np
import torch
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class TabularMLP(nn.Module):
    """Embeddings per categorical column + numeric features -> MLP -> single logit."""

    def __init__(self, cat_cardinalities, n_numeric, embed_dim=8, hidden=64, dropout=0.2):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(card + 1, embed_dim) for card in cat_cardinalities  # +1 for OOV bucket
        ])
        total_in = n_numeric + embed_dim * len(cat_cardinalities)
        self.net = nn.Sequential(
            nn.Linear(total_in, hidden),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x_num, x_cat):
        parts = [x_num]
        for i, emb in enumerate(self.embeddings):
            parts.append(emb(x_cat[:, i]))
        x = torch.cat(parts, dim=1) if len(parts) > 1 else x_num
        return self.net(x).squeeze(-1)


def train_one_model(model, train_loader, val_loader, lr=1e-3, max_epochs=100, patience=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val_loss, best_state, epochs_no_improve = float("inf"), None, 0

    for epoch in range(max_epochs):
        model.train()
        for x_num, x_cat, y in train_loader:
            optimizer.zero_grad()
            logits = model(x_num, x_cat)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x_num, x_cat, y in val_loader:
                val_losses.append(loss_fn(model(x_num, x_cat), y).item())
        val_loss = float(np.mean(val_losses))

        if val_loss < best_val_loss:
            best_val_loss, best_state, epochs_no_improve = val_loss, model.state_dict(), 0
        else:
            epochs_no_improve += 1

        # TODO(interview): say this out loud — early stopping is the regularizer here.
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch} (best val loss {best_val_loss:.4f})")
            break

    model.load_state_dict(best_state)
    return model


def main():
    data = load_breast_cancer(as_frame=True)
    df = data.frame.rename(columns={"target": "label"})
    X, y = df.drop(columns=["label"]).values, df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    scaler = StandardScaler().fit(X_train)  # fit only on train
    X_train, X_val, X_test = (scaler.transform(a) for a in (X_train, X_val, X_test))

    def to_loader(X, y, shuffle):
        ds = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.zeros((len(X), 0), dtype=torch.long),  # no categoricals in this toy dataset
            torch.tensor(y, dtype=torch.float32),
        )
        return DataLoader(ds, batch_size=64, shuffle=shuffle)

    train_loader = to_loader(X_train, y_train, shuffle=True)
    val_loader = to_loader(X_val, y_val, shuffle=False)
    test_loader = to_loader(X_test, y_test, shuffle=False)

    model = TabularMLP(cat_cardinalities=[], n_numeric=X_train.shape[1])
    model = train_one_model(model, train_loader, val_loader)

    model.eval()
    all_logits, all_y = [], []
    with torch.no_grad():
        for x_num, x_cat, y_batch in test_loader:
            all_logits.append(model(x_num, x_cat))
            all_y.append(y_batch)
    proba = torch.sigmoid(torch.cat(all_logits)).numpy()
    auc = roc_auc_score(torch.cat(all_y).numpy(), proba)
    print(f"DL model test ROC-AUC: {auc:.4f}  (compare against 02_baseline_model.py)")
    return model


if __name__ == "__main__":
    main()
