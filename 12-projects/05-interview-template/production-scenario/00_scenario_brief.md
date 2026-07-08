# Scenario Brief — Fraud Risk Scoring for a Payments Platform

Read this once, then close it. Everything below is what "the interviewer" would tell you verbally
across the round — deliberately incomplete and slightly contradictory in places, because that's
what a real intake looks like. Your first job is to ask the clarifying questions that resolve the
gaps, out loud, before writing code. Don't fix the ambiguity by silently assuming — say what you're
assuming and why.

Unlike [`../template/`](../template/), there is no clean feature/target split handed to you. You
get a data description and a business ask. Practice starting from `01_generate_messy_data.py`
(run it once to get the raw files) and building your own framing → features → model → tradeoffs
narrative on top, out loud, exactly as you would live.

---

## The ask (as the interviewer would state it)

"We process payments for online merchants. Some transactions are fraudulent — a chargeback comes
in weeks later. We want a model that scores each transaction at checkout time with a fraud
probability, so we can decline, hold for review, or approve automatically. We have transaction
logs and a table of confirmed chargebacks. Build something end to end and walk me through the
decisions you'd make putting it in production. We're not expecting a perfect model in this
session — we want to see how you think."

That's it. No feature list, no schema doc, no explicit metric, no latency number. All of the
following you have to surface by asking or by inspecting the data once generated.

## Deliberately unresolved tensions (find these yourself)

- The chargeback (label) table lags the transaction table by **weeks** — if you naively join on
  transaction time and train, are you sure every row's label had actually resolved when you're
  supposedly "predicting"? (Label latency / leakage trap.)
- The interviewer said "review-time scoring" but also said "checkout time" — these imply very
  different latency budgets (seconds vs. can-tolerate-a-short-delay). Ask which.
- The transaction table has a schema change partway through the data (a column gets renamed /
  a new payment method type appears later in time) — this simulates real schema drift. Don't
  assume the loader you write for early rows also works unmodified for later rows without
  checking.
- Merchant-level behavior varies wildly (one merchant might be 80% of chargebacks) — is the unit
  of generalization "any transaction" or "this specific merchant"? Changes your CV strategy.
- Class balance is heavily skewed and not stated — check it yourself, don't assume 50/50.
- Cost of a false negative (missed fraud → chargeback, plus fees) vs. false positive (declined
  good customer → lost revenue, angry merchant) are asymmetric and unstated — ask, or state an
  assumption and justify a threshold/metric choice based on it.

## What "good" looks like in this round

1. You ask 3-5 sharp clarifying questions before touching code (latency, cost asymmetry, label
   lag, unit of generalization, what "review queue" capacity looks like).
2. You inspect the generated data for the traps above rather than trusting the schema at face
   value — this is the single biggest signal of production maturity.
3. You build a leakage-safe pipeline (point-in-time correct join, no label-lag leakage), a
   reasonable baseline, and state — don't necessarily fully build — the DL/GBT model choice with
   a justified tradeoff.
4. You propose a serving architecture (sync scoring + fallback, or precomputed risk features +
   lightweight online model) matched to the latency answer you got.
5. You close with monitoring: what drifts, what triggers a retrain, and how you'd handle the
   asymmetric cost via a threshold or multi-tier (auto-approve / review-queue / auto-decline)
   decision policy instead of a single cutoff.

## Files in this drill

| File | Purpose |
|---|---|
| `00_scenario_brief.md` | this file — the ambiguous ask, read once before starting |
| `01_generate_messy_data.py` | generates `transactions.csv` and `chargebacks.csv` with the traps above baked in (label lag, schema drift, skew, merchant concentration) — run this first, then treat its output as "the real data" |
| `02_solution_walkthrough.md` | reference walkthrough: the questions to ask, the traps found, the pipeline built, tradeoffs justified, and the production plan — **read only after attempting the drill yourself**, this is the answer key |

## How to run this drill

```bash
cd production-scenario
python 01_generate_messy_data.py     # writes transactions.csv, chargebacks.csv
```

Then, time-boxed to ~45-60 minutes and narrating out loud the whole time:
1. Ask your clarifying questions (write down what you'd ask; assume reasonable answers if solo).
2. Load both CSVs, find the traps, write down what you found before fixing anything.
3. Build a point-in-time-correct join + leakage-safe feature pipeline.
4. Fit a baseline, state the DL-vs-GBT tradeoff, pick one and justify it.
5. Propose serving architecture, decision policy (not just a threshold), and monitoring plan.
6. Only then open `02_solution_walkthrough.md` and compare.

## Where to Next

- **Clean-dataset live-coding drill (do this one first if you haven't)** → [../template/](../template/)
- **Tradeoff bank for justifying every choice above** → [../../../07-interview-prep/ROUND3-tradeoff-drills.md](../../../07-interview-prep/ROUND3-tradeoff-drills.md)
- **Full fraud system design reference** → [../../../06-production-ml/system-design/20-fraud-detection-full-system.md](../../../06-production-ml/system-design/20-fraud-detection-full-system.md)
