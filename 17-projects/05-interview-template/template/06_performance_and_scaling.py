"""Phase 6 — Performance optimization + scaling in production.

SAY OUT LOUD before coding:
- "Performance optimization" means two different things and I'll address both:
  1. MODEL performance: regularization, class-weighting, calibration.
  2. COMPUTE performance: mixed precision, batch size, profiling the real bottleneck.
- "Scaling in production" is a separate concern from either: serving architecture, monitoring,
   rollback.

Runs standalone: demonstrates calibration + a profiled/AMP-ready training step on the toy dataset,
then prints a production serving/monitoring checklist (no infra to actually stand up here).
"""
import time

import numpy as np
import torch
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split
from torch import nn


# ---------------------------------------------------------------------------
# 1. Model-performance optimization: calibration
# ---------------------------------------------------------------------------
def calibrate_model(model, X_train, y_train, X_test, y_test):
    """Say out loud: if predictions feed a threshold/decision, calibration matters as much as AUC."""
    uncalibrated_proba = model.predict_proba(X_test)[:, 1]
    uncalibrated_brier = brier_score_loss(y_test, uncalibrated_proba)

    calibrated = CalibratedClassifierCV(model, method="isotonic", cv=3)
    calibrated.fit(X_train, y_train)
    calibrated_proba = calibrated.predict_proba(X_test)[:, 1]
    calibrated_brier = brier_score_loss(y_test, calibrated_proba)

    print(f"Brier score before calibration: {uncalibrated_brier:.4f}")
    print(f"Brier score after calibration:  {calibrated_brier:.4f}")
    return calibrated


# ---------------------------------------------------------------------------
# 2. Compute-performance optimization: profile before optimizing, then AMP
# ---------------------------------------------------------------------------
def profile_training_step(model, x, y, loss_fn, optimizer):
    """TODO(interview): say — 'I profile before optimizing; don't guess the bottleneck.'"""
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.perf_counter()
    optimizer.zero_grad()
    loss = loss_fn(model(x), y)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.perf_counter() - start


def train_step_with_amp(model, x, y, loss_fn, optimizer, scaler):
    """Mixed precision — halves memory and speeds up matmuls on modern GPUs (no-op on CPU)."""
    optimizer.zero_grad()
    with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu",
                         enabled=torch.cuda.is_available()):
        loss = loss_fn(model(x), y)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return loss.item()


# ---------------------------------------------------------------------------
# 3. Production scaling checklist (say out loud, no infra to stand up here)
# ---------------------------------------------------------------------------
PRODUCTION_CHECKLIST = """
Serving architecture:
  - Batch scoring (nightly/hourly job) vs. online serving (<100ms, single-request) — decided in
    00_problem_framing.md by the latency constraint.
  - Online: package the fitted Pipeline (preprocessor + model) behind a REST/gRPC endpoint;
    the same preprocessing code must run in training and serving to avoid training-serving skew.
  - Batch: score a snapshot table, write predictions to a feature/serving store.

Training-serving skew:
  - Guarantee it by construction: serialize the *same* sklearn Pipeline object (joblib) used in
    training, don't reimplement preprocessing in the serving language.

Deployment safety:
  - Version every model artifact; keep the previous version hot for instant rollback.
  - Canary or shadow-deploy the new model against a slice of traffic before full rollout.

Monitoring:
  - Input drift: compare live feature distributions to training distributions (e.g. PSI).
  - Prediction drift: monitor the output score distribution over time.
  - Delayed ground truth: log predictions with IDs so real labels can be joined back later to
    compute live AUC/calibration, not just proxy signals.

Scaling:
  - Horizontal: stateless scoring service behind a load balancer, autoscale on request volume.
  - Caching: cache predictions for frequently-requested entities if the feature vector rarely changes.
"""


def main():
    data = load_breast_cancer(as_frame=True)
    df = data.frame.rename(columns={"target": "label"})
    X, y = df.drop(columns=["label"]), df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = GradientBoostingClassifier(random_state=42).fit(X_train, y_train)
    calibrate_model(model, X_train, y_train, X_test, y_test)

    net = nn.Sequential(nn.Linear(X_train.shape[1], 64), nn.ReLU(), nn.Linear(64, 1), nn.Flatten(0))
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(net.parameters())
    x_t = torch.tensor(X_train.values, dtype=torch.float32)
    y_t = torch.tensor(y_train.values, dtype=torch.float32)
    step_time = profile_training_step(net, x_t, y_t, loss_fn, optimizer)
    print(f"Single training-step wall time: {step_time*1000:.2f} ms (profile before optimizing)")

    print(PRODUCTION_CHECKLIST)


if __name__ == "__main__":
    main()
