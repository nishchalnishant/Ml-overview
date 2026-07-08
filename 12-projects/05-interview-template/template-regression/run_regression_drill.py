"""Second interview drill: same 6-phase shape as ../template/, but REGRESSION.

Use this as a second practice rep once ../template/ (binary classification) feels automatic —
drilling only one task type builds false confidence. Deltas from the classification template are
called out explicitly; everything not mentioned (split ordering, leakage-safe preprocessing
contract, early stopping) is identical.

Dataset: sklearn California housing (bundled, regression target = median house value). California
housing ships fully numeric, so a synthetic categorical column (`density_bucket`, binned from
`AveOccup`) is added in `load_data()` purely so this drill exercises the categorical branch of the
preprocessing contract — swap it for whatever real categoricals the interviewer gives you.

SAY OUT LOUD (deltas from classification):
- Metric: MAE/RMSE, not ROC-AUC/PR-AUC. State whether the business cares about relative error
  (MAPE) — matters more for regression than classification.
- Loss: nn.MSELoss() or nn.HuberLoss() (Huber if outliers in the target are a concern), not
  BCEWithLogitsLoss.
- Final layer: still width 1, but no sigmoid anywhere — raw output IS the prediction.
- CV: plain KFold (or GroupKFold/TimeSeriesSplit if grouped/temporal) — no StratifiedKFold,
  since there's no class balance to preserve for a continuous target.
- Categorical handling is the same contract as the classification template: impute
  (most-frequent) + one-hot inside a ColumnTransformer, fit only on train. Regression doesn't
  change this at all — only the numeric side (scaling before an MLP) and the target differ.
- Calibration (phase 6) doesn't apply in the classification sense; the regression analogue is
  checking residual distribution / prediction intervals (conformal prediction), mentioned but
  not implemented here — see 02-classical-ml/14-calibration-and-uncertainty.md.
"""
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn

NUMERIC_FEATURES = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "Latitude", "Longitude"]
CATEGORICAL_FEATURES = ["density_bucket"]


def load_data():
    data = fetch_california_housing(as_frame=True)
    df = data.frame.rename(columns={"MedHouseVal": "target"})
    # TODO(interview): replace with the real categorical columns for the given dataset.
    df["density_bucket"] = pd.cut(
        df["AveOccup"], bins=[0, 2, 4, np.inf], labels=["low", "medium", "high"]
    ).astype(str)
    return df.drop(columns=["AveOccup"])


# --- Phase 1: feature engineering (numeric -> impute+scale, categorical -> impute+one-hot) ---
def build_preprocessor():
    """Same leakage-safe contract as ../template/01: fit only on the training fold."""
    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])


def split_and_scale(df):
    X, y = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES], df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    preprocessor = build_preprocessor().fit(X_train)  # fit only on train
    return (
        preprocessor.transform(X_train), preprocessor.transform(X_test),
        y_train.values, y_test.values,
    )


# --- Phase 2: baseline ---
def fit_baselines(X_train, y_train, X_test, y_test):
    ridge = Ridge().fit(X_train, y_train)
    gbt = GradientBoostingRegressor(random_state=42).fit(X_train, y_train)
    for name, model in [("ridge", ridge), ("gbt", gbt)]:
        pred = model.predict(X_test)
        print(f"{name}: MAE={mean_absolute_error(y_test, pred):.4f} "
              f"RMSE={root_mean_squared_error(y_test, pred):.4f}")
    return ridge, gbt


# --- Phase 3: deep learning model (regression head: width 1, no sigmoid, MSE/Huber loss) ---
class RegressionMLP(nn.Module):
    def __init__(self, n_features, hidden=64, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_dl(X_train, y_train, X_val, y_val, max_epochs=100, patience=10):
    model = RegressionMLP(X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.HuberLoss()  # robust to outlier house prices vs. plain MSE

    x_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    x_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    best_val, best_state, no_improve = float("inf"), None, 0
    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(model(x_train_t), y_train_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(x_val_t), y_val_t).item()

        if val_loss < best_val:
            best_val, best_state, no_improve = val_loss, model.state_dict(), 0
        else:
            no_improve += 1
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch} (best val loss {best_val:.4f})")
            break

    model.load_state_dict(best_state)
    return model


# --- Phase 4: cross-validation (plain KFold — no stratification for continuous targets) ---
def cross_validate(X, y):
    """X is the raw (unencoded) feature frame — preprocessing must live inside the CV pipeline,
    otherwise the encoder/scaler would be fit on data outside each fold and leak."""
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    pipeline = Pipeline([
        ("preprocess", build_preprocessor()),
        ("model", GradientBoostingRegressor(random_state=42)),
    ])
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="neg_mean_absolute_error")
    print(f"CV MAE: {-scores.mean():.4f} +/- {scores.std():.4f}")


def main():
    df = load_data()
    X_train, X_test, y_train, y_test = split_and_scale(df)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    print("--- Baselines ---")
    fit_baselines(X_train, y_train, X_test, y_test)

    print("\n--- Deep learning model ---")
    model = train_dl(X_train, y_train, X_val, y_val)
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(X_test, dtype=torch.float32)).numpy()
    print(f"DL test MAE={mean_absolute_error(y_test, pred):.4f} "
          f"RMSE={root_mean_squared_error(y_test, pred):.4f}")

    print("\n--- Cross-validation ---")
    X_full, y_full = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES], df["target"].values
    cross_validate(X_full, y_full)

    # Phase 5 (HPO) and phase 6 (perf/scaling) are identical in shape to ../template/04 and
    # ../template/06 — only the metric (MAE/RMSE) and loss (Huber/MSE) change, so they are not
    # re-implemented here. Narrate them by pointing at those files with the deltas above.


if __name__ == "__main__":
    main()
