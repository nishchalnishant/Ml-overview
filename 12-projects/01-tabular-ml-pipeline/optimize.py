"""Performance optimization pass: profile the training loop, then compare
baseline fp32 training against mixed-precision (AMP) on the same config.

This is the "performance optimization" phase of the interview — the point
isn't the specific numbers, it's demonstrating you profile before optimizing
and can name the actual bottleneck (data loading vs. GPU/CPU compute).

Run:
    python optimize.py
"""
import time

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from train_dl import (
    CATEGORICAL_FEATURES,
    DEVICE,
    NUMERIC_FEATURES,
    TARGET,
    TabularMLP,
    preprocess_fit,
    preprocess_transform,
)

CONFIG = {"lr": 1e-3, "weight_decay": 1e-5, "hidden": 64, "embed_dim": 8, "dropout": 0.2, "batch_size": 128}


def time_epoch(model, loader, optimizer, criterion, use_amp):
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and DEVICE.type == "cuda")
    data_time, compute_time = 0.0, 0.0
    t_prev = time.perf_counter()
    for xb_num, xb_cat, yb in loader:
        t_loaded = time.perf_counter()
        data_time += t_loaded - t_prev

        xb_num, xb_cat, yb = xb_num.to(DEVICE), xb_cat.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        with torch.autocast(device_type=DEVICE.type, enabled=use_amp and DEVICE.type == "cuda"):
            logits = model(xb_num, xb_cat)
            loss = criterion(logits, yb)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        t_prev = time.perf_counter()
        compute_time += t_prev - t_loaded
    return data_time, compute_time


def main():
    df = pd.read_csv("data/churn.csv")
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    num_imputer, scaler_, cat_imputer, encoder, cardinalities = preprocess_fit(X_train)
    xn, xc = preprocess_transform(X_train, num_imputer, scaler_, cat_imputer, encoder)
    yt = torch.from_numpy(y_train.values.astype(np.float32))

    ds = torch.utils.data.TensorDataset(xn, xc, yt)
    loader = torch.utils.data.DataLoader(ds, batch_size=CONFIG["batch_size"], shuffle=True)

    for use_amp in (False, True):
        model = TabularMLP(
            cardinalities, len(NUMERIC_FEATURES),
            embed_dim=CONFIG["embed_dim"], hidden=CONFIG["hidden"], dropout=CONFIG["dropout"],
        ).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
        pos_weight = torch.tensor([(yt == 0).sum() / max((yt == 1).sum(), 1)]).to(DEVICE)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        model.train()
        start = time.perf_counter()
        data_time, compute_time = time_epoch(model, loader, optimizer, criterion, use_amp)
        total = time.perf_counter() - start

        label = "mixed-precision" if use_amp else "fp32 baseline"
        print(f"{label:16s} total={total:.3f}s  data-loading={data_time:.3f}s  compute={compute_time:.3f}s")

    print(
        "\nNote: AMP only helps on CUDA GPUs — on CPU-only machines (like this "
        "environment likely is) the fp32/AMP times will be nearly identical. "
        "State that caveat explicitly rather than claiming a speedup you can't observe."
    )


if __name__ == "__main__":
    main()
