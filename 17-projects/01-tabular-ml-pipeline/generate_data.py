"""Generate a synthetic customer-churn dataset.

Deterministic (fixed seed) so the pipeline is reproducible without a network
download. Mimics realistic quirks: missing values, a skewed numeric feature,
mixed categorical cardinality, and a ~15% imbalanced target.
"""
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N = 5000


def generate() -> pd.DataFrame:
    tenure_months = RNG.integers(1, 72, N)
    monthly_charge = RNG.normal(70, 25, N).clip(10, 200)
    total_charge = monthly_charge * tenure_months + RNG.normal(0, 50, N)
    num_support_calls = RNG.poisson(1.5, N)
    contract_type = RNG.choice(["month-to-month", "one-year", "two-year"], N, p=[0.55, 0.25, 0.20])
    payment_method = RNG.choice(
        ["electronic-check", "mailed-check", "bank-transfer", "credit-card"], N
    )
    internet_service = RNG.choice(["dsl", "fiber", "none"], N, p=[0.35, 0.45, 0.20])
    tech_support = RNG.choice(["yes", "no", "no-internet"], N)
    age = RNG.integers(18, 85, N)

    # Latent churn propensity — deterministic function of features + noise,
    # then thresholded to hit ~15% positive rate.
    logit = (
        -1.5
        + 0.03 * (72 - tenure_months)
        + 0.015 * (monthly_charge - 70)
        + 0.35 * num_support_calls
        + np.where(contract_type == "month-to-month", 0.9, 0.0)
        + np.where(payment_method == "electronic-check", 0.4, 0.0)
        + RNG.normal(0, 1, N)
    )
    prob = 1 / (1 + np.exp(-logit))
    churn = (prob > np.quantile(prob, 0.85)).astype(int)

    df = pd.DataFrame(
        {
            "customer_id": np.arange(1, N + 1),
            "age": age,
            "tenure_months": tenure_months,
            "monthly_charge": monthly_charge.round(2),
            "total_charge": total_charge.round(2),
            "num_support_calls": num_support_calls,
            "contract_type": contract_type,
            "payment_method": payment_method,
            "internet_service": internet_service,
            "tech_support": tech_support,
            "churn": churn,
        }
    )

    # Inject missingness (MAR-ish: more likely missing for newer customers).
    missing_mask = RNG.random(N) < (0.08 + 0.05 * (tenure_months < 6))
    df.loc[missing_mask, "total_charge"] = np.nan
    support_missing = RNG.random(N) < 0.04
    df.loc[support_missing, "tech_support"] = np.nan

    return df


if __name__ == "__main__":
    import os

    os.makedirs("data", exist_ok=True)
    df = generate()
    df.to_csv("data/churn.csv", index=False)
    print(f"Wrote data/churn.csv — {len(df)} rows, churn rate={df['churn'].mean():.2%}")
