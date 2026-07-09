# Solution Walkthrough — read only after attempting the drill

This is the answer key. If you haven't run the drill yourself against
`00_scenario_brief.md` yet, stop and do that first — the value here is in finding these traps
under time pressure, not in reading about them.

---

## 1. Clarifying questions to ask (and why each matters)

| Question | Why it matters |
|---|---|
| "Is scoring at checkout (hard real-time, <~200ms) or can we hold for a short review window?" | Resolves the brief's internal contradiction ("checkout time" vs. "review-time"). Determines sync-serving vs. queue-based architecture. |
| "What's the cost of a false negative (missed fraud) vs. false positive (declined good customer)?" | Without this you can't justify a threshold or a metric — say you'll assume a ratio (e.g. "I'll assume a missed fraud costs ~10x a false decline, common in payments, correct me if that's off") rather than silently picking one. |
| "Is there a review queue, and what's its daily capacity?" | Changes the decision policy from a single threshold to a 3-tier policy (auto-approve / queue for review / auto-decline) capacity-constrained by the queue. |
| "Should the model generalize across merchants, or do we care most about performance on our largest merchants?" | Determines whether merchant_id is a feature, a CV grouping key, or both — and whether you need per-merchant calibration. |
| "How fresh does a chargeback need to be before we call a transaction confirmed-fraud for training?" | Surfaces the label-lag problem directly — gets you permission to define a training cutoff explicitly instead of guessing. |

If running solo, state each assumption explicitly and move on — e.g. "I'll assume checkout-time
scoring under 200ms, a 10:1 false-negative:false-positive cost ratio, and merchant-agnostic
generalization, and flag that these need confirming with the real stakeholder."

## 2. Traps in the data, and how you'd find them

Run `01_generate_messy_data.py`, then actually inspect before modeling:

```python
import pandas as pd
txns = pd.read_csv("transactions.csv")
cb = pd.read_csv("chargebacks.csv")

txns.isna().mean()          # device_id ~5% missing
txns.groupby("merchant_id").size().sort_values(ascending=False).head()  # m_000 dominates
txns[["card_type", "payment_method"]].apply(lambda c: c.notna().sum())  # both partially populated -> schema change
txns.loc[txns["payment_method"] == "wallet", "txn_day"].min()           # 'wallet' only appears after some day
len(cb) / len(txns)          # ~1.5% fraud rate, confirms severe imbalance
```

**Trap 1 — schema drift (`card_type` vs `payment_method`).** Naively using only `card_type` drops
all rows after the schema change (they're blank there); using only `payment_method` drops all rows
before it. Correct handling: coalesce into one `payment_method` column
(`txns["card_type"].fillna(txns["payment_method"])`), and note "wallet" is a value that only
exists in recent data — a model trained mostly on early data may underperform once wallet volume
grows, worth flagging as a monitoring item later.

**Trap 2 — label lag.** A transaction on `txn_day=170` (near the end of the 180-day window) whose
fraud hasn't resolved yet (resolves at day 170+14..45, i.e. after day 180) has **no chargeback row
yet**, but that does not mean it's legitimately non-fraud — it means the label is not yet
observed. If you pick a training cutoff at day 180 (the whole dataset) and treat "no chargeback
present" as the negative class, you are injecting **label noise**: some of those "negatives" are
actually unresolved frauds. The correct fix is to define a **feature cutoff** and a separate,
earlier **label cutoff** with the full lag window subtracted — e.g. only use transactions with
`txn_day <= 180 - 45 = 135` as training candidates, so every included row has had the full 45-day
window to reveal a chargeback if one exists. State this out loud even if you don't fully implement
it: "I'd only train on transactions old enough that their label has had time to fully resolve —
here that's `txn_day <= day_max - max_lag`."

**Trap 3 — merchant concentration.** `m_000` accounts for a large share of both volume and fraud.
A random row-level train/test split will place many of the same merchant's transactions on both
sides, letting the model partly memorize merchant-specific patterns rather than generalizing — and
inflates validation metrics. If the earlier clarifying answer was "must generalize across
merchants," use **GroupKFold on merchant_id**; if the answer was "we mainly care about our biggest
merchants," it's fine to include merchant_id as a strong feature and do a normal time-based split
instead — say which one you're choosing and why.

**Trap 4 — severe class imbalance (~1.5%).** Rules out accuracy as a metric immediately;
ROC-AUC and PR-AUC (PR-AUC is more informative at this imbalance level) are the right offline
metrics. Also rules out a single 0.5 threshold — needs an explicit cost-based threshold or
multi-tier policy (see below).

**Trap 5 — missing `device_id`.** ~5% missing, plausibly meaningful (some checkout flows don't
capture it) — impute with an explicit `is_missing` flag rather than dropping the column or
silently filling, consistent with the leakage-safe contract used in
[`../template/01_feature_engineering.py`](../template/01_feature_engineering.py).

## 3. Leakage-safe pipeline shape

```
1. Fix a label cutoff: only include txns with txn_day <= (max_day - max_observed_lag).
2. Time-based train/test split on txn_day (not random) — this is a "predict the future" problem.
3. Coalesce card_type/payment_method into one column before anything else touches it.
4. Join chargebacks by txn_id -> label = 1 if a chargeback exists with chargeback_day present.
5. ColumnTransformer: scale amount, one-hot payment_method/country, target/frequency-encode
   merchant_id (high-cardinality) with encoding fit ONLY on the training fold.
6. is_missing flag for device_id; device_id itself as a high-cardinality categorical (embedding
   if going DL, frequency-encoded if GBT) — or drop it and say why if time-boxed.
```

## 4. Model choice — the tradeoff to state out loud

"This is tabular, structured, moderate feature count, severe imbalance, and a hard latency
constraint if it's checkout-time scoring. I'd start with a gradient-boosted tree (LightGBM/XGBoost)
as both baseline and likely final model: it handles imbalance well (`scale_pos_weight`), needs no
GPU, trains fast enough to iterate live, and is easier to explain to a fraud/risk team who will
ask 'why was this declined.' I'd only reach for a DL model if there's high-cardinality
sequence/behavioral data (e.g. a user's full event stream) that a tree can't represent well — that
would justify the extra complexity and latency cost. Given what's in this dataset, I'd default to
GBT and say so explicitly rather than defaulting to DL because the round mentions 'deep learning.'"

This is the correct move even though the earlier interview template centers a DL model — round 3
is scored on judgment, and picking DL here without justification is a signal of the opposite.

## 5. Decision policy — not just a threshold

Given the asymmetric cost (assume 10:1 false-negative:false-positive), propose 3 tiers instead of
one cutoff:
- **score < low_threshold** → auto-approve
- **low_threshold ≤ score < high_threshold** → route to human review queue (capacity-constrained —
  tune `low_threshold`/`high_threshold` so queue volume matches the stated review capacity)
- **score ≥ high_threshold** → auto-decline

State that thresholds should be chosen by walking the precision-recall curve against the stated
cost ratio and queue capacity, not picked arbitrarily.

## 6. Serving architecture (matched to the latency answer)

- If checkout-time (<200ms): synchronous scoring call behind a strict timeout, with a fallback
  (e.g. simple velocity-rule heuristic) if the model call times out — never let fraud scoring be a
  hard blocking dependency with no fallback.
- Precompute the slow-changing features (merchant risk history, device reputation) into a feature
  store updated on a schedule; compute only request-time features (amount, payment method,
  country of this transaction) on the fly — this is the batch-precompute + online-compute-on-request
  hybrid, justified by the mixed nature of the features here (see
  [ROUND3-tradeoff-drills.md](../../../07-interview-prep/ROUND3-tradeoff-drills.md#feature-pipeline)).

## 7. Monitoring plan

- **Input drift**: distribution of `payment_method` (wallet share growing over time — this dataset
  bakes that in deliberately), `amount`, `country` mix per merchant.
- **Label lag-aware evaluation**: you cannot compute live precision/recall until the label window
  (45 days here) has passed — track a rolling "delayed" AUC/PR-AUC computed only on
  fully-resolved cohorts, not naive same-day accuracy.
- **Prediction drift**: score distribution shift, especially around the schema-change-like events
  (new payment method types, new merchants onboarding).
- **Retrain trigger**: given fraud patterns adapt adversarially, prefer trigger-based retrain
  (drift or delayed-PR-AUC drop beyond a threshold) over a fixed weekly cadence — say this
  explicitly, it's the correct flip from the tradeoff bank's default.

## 8. What a strong close sounds like

"To summarize: I found a label-lag issue and a schema drift issue in the data before modeling
anything, which would have silently corrupted a naive pipeline. I'd ship a GBT baseline behind a
three-tier decision policy tuned to the cost ratio and review-queue capacity, serve it with
precomputed merchant/device features plus fast request-time features, and monitor with
lag-aware delayed evaluation rather than same-day accuracy, retraining on a drift trigger rather
than a fixed schedule given fraud is adversarial."

## Where to Next

- **Tradeoff bank** → [../../../07-interview-prep/ROUND3-tradeoff-drills.md](../../../07-interview-prep/ROUND3-tradeoff-drills.md)
- **Full fraud system design reference** → [../../../06-production-ml/system-design/20-fraud-detection-full-system.md](../../../06-production-ml/system-design/20-fraud-detection-full-system.md)
- **Clean-dataset drill (round 2 style)** → [../template/](../template/)
