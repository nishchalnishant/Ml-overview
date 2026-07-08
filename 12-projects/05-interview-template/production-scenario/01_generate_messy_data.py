"""Generates the "real data" for the fraud-scoring production scenario.

Run this once, then treat transactions.csv / chargebacks.csv as data handed to you by the
interviewer — inspect it fresh, don't peek at this generator's internals until after the drill.

Traps baked in on purpose (see 00_scenario_brief.md):
- Chargebacks resolve 14-45 days after the transaction (label lag). Training naively on "all
  transactions with a known label as of today" silently drops recent fraud that hasn't resolved
  yet, understating the true fraud rate on recent data and, if joined carelessly, can also leak
  future information if "today" isn't fixed correctly relative to a train/test cutoff.
- Payment method schema changes 70% of the way through the time range: an early column is renamed,
  and a new payment_method value ("wallet") appears only in the later period.
- One merchant is responsible for a disproportionate share of both volume and chargebacks.
- Class balance is realistic (~1.5% fraud), not 50/50.
- A handful of columns are only populated for a subset of rows (nulls), simulating optional
  fields from a schema that evolved over time.
"""
import csv
import random

N_TRANSACTIONS = 60_000
START_DAY = 0
END_DAY = 180
SCHEMA_CHANGE_DAY = 126  # 70% through the range

MERCHANTS = [f"m_{i:03d}" for i in range(40)]
DOMINANT_MERCHANT = "m_000"  # disproportionate share of volume + chargebacks

random.seed(1234)


def make_transaction(txn_id, day):
    is_dominant = random.random() < 0.18
    merchant = DOMINANT_MERCHANT if is_dominant else random.choice(MERCHANTS[1:])

    amount = round(random.lognormvariate(3.5, 1.1), 2)

    if day < SCHEMA_CHANGE_DAY:
        # early-period schema: column name is "card_type"
        payment_method = random.choice(["visa", "mastercard", "amex"])
    else:
        # later-period schema: renamed to "payment_method", new "wallet" type appears
        payment_method = random.choice(["visa", "mastercard", "amex", "wallet"])

    country = random.choice(["US", "US", "US", "CA", "GB", "DE", "BR"])

    # base fraud propensity: dominant merchant and 'wallet' method are riskier
    base_risk = 0.008
    if is_dominant:
        base_risk += 0.03
    if payment_method == "wallet":
        base_risk += 0.02
    if amount > 500:
        base_risk += 0.01

    is_fraud = random.random() < base_risk
    # resolution lag: fraud confirmed 14-45 days after the transaction, if at all
    resolves_after_days = random.randint(14, 45) if is_fraud else None

    device_id = f"d_{random.randint(1, 25000)}" if random.random() > 0.05 else ""  # 5% missing

    return {
        "txn_id": txn_id,
        "txn_day": day,
        "merchant_id": merchant,
        "amount": amount,
        "payment_method_raw": payment_method,
        "country": country,
        "device_id": device_id,
        "_is_fraud": is_fraud,
        "_resolves_after_days": resolves_after_days,
    }


def main():
    transactions = []
    chargebacks = []

    for i in range(N_TRANSACTIONS):
        day = random.randint(START_DAY, END_DAY)
        txn = make_transaction(f"t_{i:07d}", day)
        transactions.append(txn)
        if txn["_is_fraud"]:
            chargebacks.append({
                "txn_id": txn["txn_id"],
                "chargeback_day": txn["txn_day"] + txn["_resolves_after_days"],
            })

    transactions.sort(key=lambda r: r["txn_day"])

    with open("transactions.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["txn_id", "txn_day", "merchant_id", "amount",
                          "card_type", "payment_method", "country", "device_id"])
        for t in transactions:
            if t["txn_day"] < SCHEMA_CHANGE_DAY:
                card_type, payment_method = t["payment_method_raw"], ""
            else:
                card_type, payment_method = "", t["payment_method_raw"]
            writer.writerow([t["txn_id"], t["txn_day"], t["merchant_id"], t["amount"],
                              card_type, payment_method, t["country"], t["device_id"]])

    with open("chargebacks.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["txn_id", "chargeback_day"])
        for c in chargebacks:
            writer.writerow([c["txn_id"], c["chargeback_day"]])

    fraud_rate = len(chargebacks) / len(transactions)
    print(f"Wrote {len(transactions)} rows to transactions.csv")
    print(f"Wrote {len(chargebacks)} rows to chargebacks.csv")
    print(f"True fraud rate: {fraud_rate:.3%}")
    print(f"Schema change at txn_day={SCHEMA_CHANGE_DAY} "
          f"(card_type -> payment_method, 'wallet' introduced)")
    print("Note: chargebacks resolve 14-45 days after txn_day — "
          "mind label lag when picking a train/test time cutoff.")


if __name__ == "__main__":
    main()
