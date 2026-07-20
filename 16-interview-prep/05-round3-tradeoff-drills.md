---
module: Interview Prep
topic: Round 3 - Production Problem + Tradeoffs
subtopic: ""
status: unread
tags: [interviewprep, production, tradeoffs, system-design, coding]
---
# Round 3 Prep — "Solve a real production problem end to end (with coding). Make and justify tradeoffs."

This round is not the clean-dataset live-coding drill in
[12-projects/05-interview-template/](../12-projects/05-interview-template/) — expect an
**ambiguous, production-flavored scenario** (constraints stated verbally, messy/missing
requirements, no clean CSV handed to you) where the interviewer is scoring your **decisions and
justifications** as much as your code. Treat every design choice as "A vs. B, here's why A, here's
when I'd switch to B."

## How to run the round

1. **Clarify before coding** (2-3 min): scale (QPS/data volume), latency SLA, label availability
   and lag, online vs. offline, cost/infra constraints, team/data maturity. Don't assume — ask.
2. **State the tradeoff before making the choice**, not after. Pattern: *"Given [constraint], I'd
   pick [A] over [B] because [reason]. If [constraint changes], I'd switch to [B] because [reason]."*
3. **Code the slice that matters**, narrate the rest. You will not have time to build the whole
   system — pick the piece with the most technical risk (usually feature pipeline or
   serving/inference path) and code that; describe the rest at the same level of detail as the
   production checklist in
   [12-projects/05-interview-template/README.md](../17-projects/05-interview-template/README.md#from-trained-model-to-production-the-steps-in-between).
4. **Volunteer the failure mode** for every choice you make — this is what "justify" is testing.

---

## Tradeoff bank

Each entry: the question, the axis, a default answer, and when the default flips. Practice saying
the *whole* line (default + flip condition) out loud in under 20 seconds.

### Model choice

**GBT (XGBoost/LightGBM) vs. deep learning (MLP/embeddings) for tabular data**
- Default: GBT. Wins/ties on tabular data, trains faster, needs less data, easier to explain and calibrate.
- Flip to DL: very high-cardinality categoricals better handled by learned embeddings, need to fuse with unstructured inputs (text/image/sequence) in the same model, or you have a genuinely large dataset (>~10M rows) where DL's representational capacity pays off.
- Reference: [02-classical-ml/05-when-classical-ml-wins.md](../03-classical-ml/07-when-classical-ml-wins.md)

**Single global model vs. per-segment models** (e.g. per-region, per-product-line)
- Default: single global model with segment as a feature. Simpler to maintain, more data per model, avoids fragmenting rare segments.
- Flip to per-segment: segments have genuinely different feature-target relationships (not just different base rates) and each segment has enough data to train independently — check with an interaction-term ablation before splitting.

**Simple heuristic/rules vs. ML model as the very first version**
- Default: ship the heuristic first if one exists (e.g. "flag if >3 failed logins in 10 min"). Gets a baseline in front of users fast, gives you a comparison point, de-risks the "is ML even needed here" question.
- Flip to ML immediately: heuristic accuracy is provably far from acceptable, or the interviewer states this is a v2/replacement project with an existing heuristic already in prod (then ML must beat it, stated explicitly as the launch bar).

### Serving architecture

**Batch scoring vs. online/real-time inference**
- Default: batch (precompute predictions on a schedule, serve from a cache/lookup) whenever the feature set doesn't need last-second data and slight staleness (minutes-hours) is acceptable — much simpler, cheaper, no live model-serving infra.
- Flip to online: prediction depends on request-time context (current session, current cart, current query) that can't be precomputed, or SLA requires reacting to events within seconds.
- Reference: [06-production-ml/system-design/05-real-time-ml-systems.md](../13-production-ml/04-real-time-ml-systems.md), [06-production-ml/system-design/06-streaming-ml-pipeline.md](../13-production-ml/05-streaming-ml-pipeline.md)

**Synchronous model call vs. precomputed + fallback**
- Default: synchronous call behind a strict timeout with a fallback (cached prediction, simple heuristic, or default value) — never let the model be a hard dependency that can take the product down.
- Flip: if the model result is not decision-blocking (e.g. it's an offline report), skip the sync path entirely.

**Two-stage retrieval + ranking vs. single-stage scoring** (when candidate set is large)
- Default: two-stage (cheap retrieval/candidate generation → expensive ranking on a shortlist) once the candidate pool is too large to score exhaustively within latency budget.
- Flip to single-stage: candidate pool is already small (<a few hundred) — the extra stage adds complexity and a second training/serving surface for no latency win.
- Reference: [06-production-ml/system-design/24-search-ranking-system.md](../15-system-design/cases/09-search-ranking.md), [26-end-to-end-recommendation-system.md](../15-system-design/cases/11-recommendation-system.md)

### Feature pipeline

**Precomputed feature store vs. compute-on-request**
- Default: feature store (precomputed, point-in-time correct) for anything reused across models/services or requiring historical aggregates (rolling windows, recency/frequency).
- Flip to compute-on-request: feature depends purely on the current request payload (no historical join needed) — a store just adds latency and staleness risk for nothing.
- Reference: [06-production-ml/system-design/08-feature-store-architecture.md](../13-production-ml/07-feature-store-architecture.md)

**Streaming feature pipeline vs. batch (Airflow/cron) pipeline**
- Default: batch. Simpler, cheaper, easier to debug and backfill.
- Flip to streaming: features must reflect events within seconds/minutes (fraud, real-time bidding, live matchmaking) and staleness directly costs money or safety.
- Reference: [06-production-ml/system-design/07-data-engineering-for-ml.md](../13-production-ml/06-data-engineering-for-ml.md)

### Hyperparameter search & training

**Random/Bayesian search vs. manual tuning vs. grid search**
- Default: random search for cheap trials, Bayesian (Optuna/TPE) once each trial is expensive (large DL model). Grid search only justified for ≤2 hyperparameters.
- Flip to manual: extremely tight time-box (live-coding round) — tune 1-2 known highest-leverage knobs (learning rate, tree depth) by hand and say you'd automate given more time.
- Reference: [02-classical-ml/07-hyperparameter-optimization.md](../03-classical-ml/08-hyperparameter-optimization.md)

**Full K-fold CV vs. single holdout split**
- Default: single holdout for the ambiguous/large-model or time-boxed case; full K-fold only when the dataset is small enough that variance across folds matters and compute allows K full retrains.
- Flip: report a final confidence interval via K-fold or bootstrap once the config is locked, even if search itself used a single split.

### Deployment & rollout

**Shadow/canary rollout vs. full cutover**
- Default: shadow (mirror traffic, no user impact) or canary (small % of real traffic) before any full rollout — always, no exceptions, for anything customer-facing.
- Flip: never skip this by default; the only time to argue for faster rollout is an internal-only tool with trivial rollback and no user impact.
- Reference: [12-projects/05-interview-template/README.md](../17-projects/05-interview-template/README.md#after-deployment-what-happens-once-the-model-is-live)

**Fixed retrain cadence vs. trigger-based retrain (drift/perf threshold)**
- Default: fixed cadence (e.g. weekly) when the underlying distribution changes slowly and predictably, and pipeline cost is low — simpler to operate and reason about.
- Flip to trigger-based: distribution shifts are bursty/unpredictable (seasonal spikes, adversarial adaptation like fraud/cheating) — a fixed schedule would either retrain too rarely (miss the shift) or waste compute retraining on unchanged data.

**Build custom infra vs. use a managed/existing platform**
- Default: use what the team already has (existing feature store, existing serving platform, existing experimentation framework) — say this explicitly, it signals pragmatism over resume-driven design.
- Flip to build: the existing platform provably can't meet a hard constraint (e.g. latency SLA, cost ceiling) stated in the problem — name the specific gap, don't build custom by default.

### Metrics & evaluation

**Offline proxy metric vs. online/business metric**
- Default: pick the offline metric that best proxies the online business metric (e.g. AUC as a proxy for revenue lift), but explicitly flag it's a proxy and state the A/B test that will validate it before full rollout.
- Flip: if the interviewer says an online metric is already instrumented and cheap to test against, go straight to it — don't over-invest in a proxy nobody will validate.
- Reference: [06-production-ml/system-design/14-ab-testing-experimentation.md](../04-evaluation/06-ab-testing-experimentation.md)

**Optimize for accuracy vs. optimize for calibration**
- Default: calibration matters whenever the score feeds a downstream decision with a fixed threshold or cost (e.g. "flag if p>0.8", ranking blended with other signals, or the number itself is shown to a human). Accuracy/AUC alone can be fine when only the ranking order matters, not the raw score.
- Reference: [02-classical-ml/14-calibration-and-uncertainty.md](../04-evaluation/04-calibration-and-uncertainty.md)

---

## Closing pattern for any tradeoff question

Say this shape every time, it demonstrates the "justify" bar directly:

1. "The two options are X and Y."
2. "I'd default to X because [constraint/benefit]."
3. "The risk with X is [specific failure mode]."
4. "I'd switch to Y if [concrete condition changes]."

## Where to Next

- **Ambiguous production scenario to drill this against (fraud-scoring, messy data, find the traps yourself)** → [12-projects/05-interview-template/production-scenario/](../12-projects/05-interview-template/production-scenario/)
- **Live-coding pipeline drill (clean dataset, mechanics only)** → [12-projects/05-interview-template/](../12-projects/05-interview-template/)
- **Full production checklist (post-training → deployed → monitored)** → same template's README, "From trained model to production" and "After deployment" sections
- **System design references for a specific domain** → [06-production-ml/system-design/](../06-production-ml/system-design/)
- **EA-specific routing for the earlier rounds** → [EA-ml-deep-learning-interview.md](ea/00-ml-deep-learning-interview.md)
