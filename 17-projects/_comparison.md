---
module: Projects
topic: Comparison Cheat Sheet
subtopic: ""
status: unread
tags: [projects, cheatsheet, comparison, tradeoffs, hpo, models, deployment]
---
# 12-projects Comparison Cheat Sheet

Rapid-review, option-vs-option reference distilled from the actual code/docs in this
folder (01-tabular-ml-pipeline, 05-interview-template incl. production-scenario, 02-rag-pipeline,
03-llm-finetuning, 04-multi-agent-a2a-mcp, 09-mcp-agent-system). Grounded only in comparisons the
folder actually makes.

---

## Hyperparameter Search Strategies

### Grid Search vs. Random Search vs. Bayesian (Optuna/TPE)

**Random search** (used in `01-tabular-ml-pipeline/train_dl.py`'s hand-rolled 8-trial loop, and
`05-interview-template/template/04_hyperparameter_search.py`'s `RandomizedSearchCV`)
- What it is: sample hyperparameter configs independently at random (uniform/log-uniform/choice)
  for a fixed trial budget, keep the best by validation score.
- Pros: trivially parallelizable; no assumptions about the search space's smoothness; dominates
  grid search once you have >2-3 hyperparameters (Bergstra & Bengio) because not every dimension
  matters equally and grid wastes trials on unimportant combinations.
- Cons: doesn't learn from past trials — wastes budget re-sampling regions already known to be bad;
  sample-inefficient once each trial is expensive (e.g. a full NN training run).
- Pick over grid search when: >2-3 hyperparameters, or a continuous space (log-uniform lr, etc.)
  that a grid would coarsely discretize anyway.
- Pick over Bayesian/Optuna when: trials are cheap (sklearn `GradientBoostingClassifier` fits fast),
  dependency surface should stay minimal (Project 01 deliberately skips Optuna for this reason), or
  you just need a fast, dependency-light baseline search.
- Key detail: Project 01's `train_dl.py` explicitly does NOT run k full NN trainings per trial —
  HPO uses a single held-out validation split; K-fold CV is run once afterward, only on the winning
  config, to report a confidence interval (not for search) — because `n_trials × k_folds` full DL
  trainings is too expensive.

**Grid Search**
- What it is: exhaustively evaluate every combination of a discrete hyperparameter grid.
- Pros: fully reproducible/deterministic; simple to reason about coverage.
- Cons: combinatorial blow-up with >2-3 params; wastes evaluations on unimportant-dimension
  combinations; can't naturally express continuous log-uniform ranges.
- Pick over random search when: only 1-2 hyperparameters, each with a small discrete set of
  plausible values, and you want exhaustive certainty over that small space.

**Bayesian optimization / Optuna (TPE) + pruning**
- What it is: a sequential model-based search — each trial's result informs where to sample next
  (TPE by default in Optuna), with a pruner killing bad trials early instead of running to
  completion.
- Pros: converges faster than random search when each trial is expensive (a full training run);
  pruning saves compute by not finishing clearly-bad trials.
- Cons: added dependency; sequential-ish nature makes naive massive parallelization less
  straightforward than random search; overkill for cheap-to-fit models.
- Pick over random search when: trials are expensive (full NN training runs) and the budget is
  large enough that smarter sampling pays off — explicitly the repo's stated boundary: Project 01's
  GBT/random-search stays simple; the DL model and the interview template both narrate "Optuna/TPE
  + pruning" as the upgrade once trials get costly.
- Key detail: `05-interview-template`'s `04_hyperparameter_search.py` explicitly narrates
  Optuna/TPE as the DL answer but implements only `RandomizedSearchCV` for the GBT model in
  code — the repo's own stated position is "say Optuna for DL, don't necessarily wire it up live."

| Option | Best for | Avoid when | Key tradeoff |
|---|---|---|---|
| Grid search | 1-2 discrete hyperparams, need exhaustive coverage | >2-3 params or continuous ranges | Coverage guarantee vs. combinatorial cost |
| Random search | 3+ hyperparams, cheap-to-fit models (GBT/sklearn) | Trials are very expensive and budget is large | Simple/parallel vs. doesn't learn across trials |
| Bayesian (Optuna/TPE) | Expensive trials (full DL training runs), large budget | Trials are cheap, want minimal dependencies | Sample-efficient vs. added complexity/sequential bias |

---

## Cross-Validation Strategies

Covered in `01-tabular-ml-pipeline` (`StratifiedKFold`), `05-interview-template/template/05_cross_validation.py`
narration, and `production-scenario` (GroupKFold vs. time-based split tension).

- **StratifiedKFold** — preserves class ratio (e.g. ~15% churn positive rate) in every fold.
  Pick over plain KFold whenever the target is imbalanced classification; plain KFold can by chance
  produce a fold with almost no positive examples, making CV noisy.
- **Plain KFold** — no stratification. Pick for regression targets (nothing to stratify by) —
  used in `template-regression/run_regression_drill.py` instead of StratifiedKFold.
- **GroupKFold (on an entity like merchant_id or user_id)** — keeps all of one entity's rows on one
  side of the split. Pick over a random/stratified split when multiple rows share an entity (one
  merchant dominates volume/fraud, per the `production-scenario` fraud drill) and the model must
  generalize to *unseen* entities — otherwise the model partly memorizes entity-specific patterns
  and validation metrics are inflated.
- **Time-based split (not random)** — order-preserving split on a timestamp. Pick over any random
  or grouped split when the task is inherently "predict the future" (the fraud-scoring scenario) or
  when there's schema/behavior drift over time (e.g. a new payment method type appearing only in
  later data) — a random split would leak future information backward and hide the drift.
- **Held-out validation split (not K-fold) for HPO, K-fold only for final confidence interval** —
  Project 01's DL-specific pattern: full K-fold CV during search is too expensive (`n_trials ×
  k_folds` trainings), so search uses one split, and K-fold is reserved for reporting a CI on the
  already-chosen config.

| Option | Best for | Avoid when | Key tradeoff |
|---|---|---|---|
| StratifiedKFold | Imbalanced classification | Regression targets | Stable class ratio per fold vs. no benefit for continuous targets |
| Plain KFold | Regression, balanced classification | Imbalanced classes | Simplicity vs. noisy folds under imbalance |
| GroupKFold | Entity concentration (merchant/user), must generalize to new entities | You explicitly care most about known/major entities | True generalization estimate vs. discards entity-specific signal |
| Time-based split | Forecasting, drifting/non-stationary data | Data is i.i.d. with no time structure | Realistic "predict the future" eval vs. can't use standard k-fold tooling |

---

## Model Choices for Tabular Data: Logistic Regression / GBT vs. Deep Learning (MLP)

Covered across `01-tabular-ml-pipeline` (LogisticRegression + XGBoost baseline vs. `TabularMLP`),
`05-interview-template` (baseline-before-DL principle), and `production-scenario/02_solution_walkthrough.md`
(explicit GBT-over-DL justification for the fraud scenario).

**Linear/Logistic Regression baseline**
- What it is: a fast, interpretable linear classifier fit inside the same leakage-safe
  `ColumnTransformer` pipeline, used as the number to beat before anything fancier.
- Pros: trains in seconds; fully interpretable coefficients; good sanity check on
  preprocessing/leakage before investing in a heavier model.
- Cons: can't capture non-linear feature interactions; ceiling well below GBT/DL on most tabular
  problems with real signal.
- Pick over GBT/DL when: you need a same-session sanity baseline fast, or interpretability to a
  regulator/business stakeholder is a hard requirement and no non-linear signal has been shown to
  matter yet.

**Gradient-Boosted Trees (XGBoost/LightGBM/`GradientBoostingClassifier`)**
- What it is: an ensemble of shallow decision trees fit sequentially on residuals.
- Pros: handles heterogeneous tabular features and non-linear interactions natively (no
  scaling/embedding needed for numerics); handles class imbalance well via `scale_pos_weight`;
  trains fast with no GPU; easier to explain feature-driven decisions to a non-ML stakeholder
  (e.g. "why was this transaction declined") than a neural net.
- Cons: doesn't natively exploit unstructured/sequential/high-cardinality embedding-style signal
  (e.g. a user's full event stream) the way an embedding-based DL model can.
- Pick over DL when (explicit repo guidance, `production-scenario/02_solution_walkthrough.md`):
  tabular, structured, moderate feature count, severe imbalance, hard latency constraint, and/or
  you need to explain individual decisions to a risk/compliance team — "default to GBT and say so
  explicitly rather than defaulting to DL because the round mentions 'deep learning.'"

**Deep Learning MLP with categorical embeddings (`TabularMLP` in Project 01)**
- What it is: `nn.Embedding` per categorical column (+1 OOV bucket) concatenated with scaled
  numeric features, fed through `Linear→ReLU→BatchNorm→Dropout` blocks to a single logit;
  `BCEWithLogitsLoss` with a `pos_weight` computed from the batch's class ratio (the DL analogue of
  `class_weight="balanced"`).
- Pros: can represent high-cardinality categorical interactions via learned embeddings; natural
  extension point if later fed sequential/behavioral data.
- Cons: needs more tuning (architecture, LR schedule, embedding dims) than a GBT; harder to explain
  per-decision to a non-ML stakeholder; GPU helps but isn't required — still slower to iterate than
  GBT on a laptop; "if the DL model doesn't clearly beat GBT on tabular data, that's a valid and
  expected outcome to report, not a failure" (explicit README framing).
- Pick over GBT when: there's meaningful high-cardinality sequence/behavioral data a tree can't
  represent well (e.g. full user event streams) that would justify the added complexity and latency
  cost — not merely because the interview brief says "deep learning."

| Option | Best for | Avoid when | Key tradeoff |
|---|---|---|---|
| Logistic/linear baseline | Fast sanity check, hard interpretability requirement | Non-linear signal matters and interpretability isn't a hard constraint | Speed/clarity vs. ceiling |
| GBT (XGBoost/LightGBM) | Structured tabular, imbalance, latency-constrained, needs explainability | High-cardinality sequence/behavioral data a tree can't represent | Strong default, fast, explainable vs. can't exploit embedding-style signal |
| DL MLP w/ embeddings | High-cardinality categorical/sequential signal justifies the cost | Plain tabular data, tight latency budget, need per-decision explainability | Representational flexibility vs. tuning cost + weaker explainability |

---

## Decision Policy: Single Threshold vs. Multi-Tier

From `production-scenario/02_solution_walkthrough.md` (fraud scoring).

- **Single 0.5 (or any single) threshold**
  - Pros: simplest possible decision rule.
  - Cons: ignores asymmetric costs (false negative = missed fraud + chargeback fees; false
    positive = declined good customer, angry merchant) and ignores operational constraints like
    review-queue capacity.
  - Avoid when the cost of the two error types is meaningfully different or there's a
    capacity-constrained human review path.
- **Multi-tier policy (auto-approve / human review queue / auto-decline)**
  - What it is: two thresholds instead of one — `score < low` auto-approve, `low ≤ score < high`
    routed to a capacity-constrained review queue, `score ≥ high` auto-decline.
  - Pros: matches the asymmetric cost structure and respects a fixed review-queue capacity;
    thresholds are tunable by walking the PR curve against the stated cost ratio.
  - Cons: more moving parts to own/monitor (two thresholds, a queue, its capacity).
  - Pick over a single threshold when: false-negative and false-positive costs are meaningfully
    asymmetric and/or there's a human review capacity to size against — the repo's explicit
    guidance for the fraud scenario.

---

## Feature/Serving Architecture: Precompute vs. Online Compute

From `production-scenario/02_solution_walkthrough.md`'s serving-architecture recommendation.

- **Fully synchronous online scoring** (compute all features per request)
  - Pros: always fresh, simplest mental model.
  - Cons: slow-changing features (merchant risk history, device reputation) get recomputed on
    every request for no benefit — wastes latency budget.
  - Avoid when some features are expensive/slow-changing and the latency budget is tight
    (checkout-time, <200ms).
- **Batch-precompute + online-compute-on-request hybrid**
  - What it is: precompute slow-changing features (merchant risk history, device reputation) into
    a feature store on a schedule; compute only request-time features (amount, payment method,
    country of *this* transaction) synchronously.
  - Pros: keeps the hard-latency path minimal; avoids recomputing expensive aggregates per request.
  - Cons: precomputed features can be stale between refresh cycles — a drift/staleness monitoring
    concern.
  - Pick over pure synchronous scoring when: latency is tight (<200ms) and some features are both
    slow-changing and expensive to compute per-request — explicit repo recommendation for the
    fraud-scoring scenario.
- **Synchronous scoring with a fallback (e.g. velocity-rule heuristic) on timeout** — never let a
  model call be a hard blocking dependency with no fallback if the latency budget is checkout-time.

---

## Retrain Trigger: Fixed Cadence vs. Trigger-Based

From `05-interview-template/README.md` (post-deployment checklist) and the fraud
`production-scenario` walkthrough, which explicitly flips the default.

- **Fixed cadence (e.g. weekly retrain)**
  - Pros: predictable, easy to schedule and staff.
  - Cons: retrains even when nothing has changed (wasted compute), or too slow to react if drift
    happens faster than the cadence.
  - Pick when: the underlying data-generating process changes slowly/predictably and the interview
    template's default framing applies (no stated adversarial dynamic).
- **Trigger-based (drift exceeds a threshold, or delayed-eval metric drops below a floor)**
  - Pros: reacts exactly when needed; avoids both wasted retrains and slow reaction to fast drift.
  - Cons: needs reliable drift/metric monitoring already in place to trigger on.
  - Pick over fixed cadence when: the problem is adversarial (fraud patterns shift deliberately to
    evade the model) — the repo explicitly calls this "the correct flip from the tradeoff bank's
    default" for the fraud scenario.

---

## RAG Pipeline Design Choices

From `02-rag-pipeline/README.md` and its interview Q&A.

### Chunking: naive fixed-size vs. sentence-boundary-snapped with overlap
- **Naive fixed-size character chunking**
  - Cons: can cut a sentence in half, degrading both the embedding (no longer one coherent idea)
    and the eventual citation's readability.
- **Sentence-boundary-snapped chunking with overlap** (what `chunking.py` implements: greedily
  packs sentences to ~500 chars, carries back trailing sentences into up to 100 chars of overlap)
  - Pros: never splits a sentence; overlap prevents an answer spanning a chunk boundary from being
    silently dropped from both neighboring chunks' context.
  - Cons: slightly more complex than a raw character slice; chunk sizes vary a bit around the target.
  - Pick over naive fixed-size chunking whenever coherence of the retrieved unit and the quality of
    citations matter (i.e. essentially always for a citation-forced RAG system).

### Retrieval: single-stage dense vs. two-stage dense + rerank
- **Single-stage dense retrieval (top-k cosine similarity directly)**
  - Cons: pure embedding similarity can surface a topically-close but keyword-irrelevant chunk
    over a more lexically precise match.
- **Two-stage: dense top-20 → lexical-overlap rerank → top-5** (what `retrieval.py` implements: no
  real cross-encoder — `rerank_score = 0.7*dense_score + 0.3*keyword_overlap`)
  - Pros: cheap approximation of a cross-encoder rerank stage; can promote a keyword-exact chunk
    dense search alone ranked much lower into the final top-k.
  - Cons: not a real learned cross-encoder — can't catch pure paraphrases with zero keyword overlap;
    still O(n) brute-force numpy search under the hood, not an ANN index.
  - Pick two-stage over single-stage whenever precision of the final top-k matters more than raw
    simplicity, at negligible extra cost (no extra model weights needed for the lexical stage).
  - Key detail: for production/million-chunk scale, the repo's own hard-question answer recommends
    a real cross-encoder (`ms-marco-MiniLM` or hosted rerank API) and a real ANN index
    (FAISS HNSW/IVF, pgvector, Pinecone, Weaviate) instead of numpy brute-force + lexical overlap.

### Generation backend: extractive (no LLM) vs. LLM-backed (OpenAI/Anthropic)
- **Extractive fallback** (default, `_extractive_answer()`)
  - What it is: returns the single existing sentence (verbatim) from a retrieved chunk with the
    highest keyword overlap with the query, plus its citation tag — never calls an LLM.
  - Pros: runs fully offline with zero API keys/setup; hallucination is *structurally impossible*
    since the returned text always existed verbatim in a retrieved chunk; useful for evaluating
    retrieval quality in isolation from generation quality.
  - Cons: answer quality/fluency is capped — it's a single retrieved sentence, not a synthesized
    answer.
  - Pick over an LLM backend when: you want zero-setup reproducibility, or you specifically want to
    isolate/debug retrieval quality from generation quality.
- **LLM-backed (`--llm openai` / `--llm anthropic`)**
  - Pros: fluent, synthesized answers across multiple chunks.
  - Cons: requires an API key (raises an explicit `RuntimeError` if missing, rather than silently
    falling back); hallucination is only prompt-mitigated (forced `[source: file#chunk_id]`
    citations + "answer using ONLY the context"), not structurally prevented.
  - Pick over extractive when: answer fluency/synthesis quality matters more than the zero-setup,
    zero-hallucination guarantee.

| Option | Best for | Avoid when | Key tradeoff |
|---|---|---|---|
| Naive fixed-size chunking | Never really — always dominated here | — | Simplicity vs. broken sentence coherence |
| Sentence-snapped + overlap chunking | Any citation-forced RAG system | N/A (near-strictly better at negligible cost) | Slightly more logic vs. coherent, boundary-safe chunks |
| Single-stage dense retrieval | Very small corpus, latency-critical, simplicity valued | Precision of top-k matters | Fewer moving parts vs. misses lexically-exact matches |
| Two-stage dense+rerank | Precision-sensitive retrieval, still cheap | Need true semantic (paraphrase) reranking at scale | Cheap precision boost vs. not a real cross-encoder |
| Extractive generation | Zero-setup, zero-hallucination, retrieval-quality debugging | Need fluent synthesized answers | Guaranteed groundedness vs. capped fluency |
| LLM generation | Fluency/synthesis quality matters | No API key, need offline determinism | Better answers vs. only prompt-level hallucination mitigation |

---

## LLM Fine-Tuning: LoRA vs. Full Fine-Tuning (and vs. RAG)

From `03-llm-finetuning/README.md` and its Q&A.

- **Full fine-tuning**
  - Cons (implied by contrast): trains and stores every parameter — large memory footprint (full
    gradients/optimizer states) and large per-task artifact size (a full model copy per fine-tune).
- **LoRA (Low-Rank Adaptation)** (what the project implements: `r=8`, `lora_alpha=16`,
  `lora_dropout=0.05`, `target_modules=["c_attn"]` for GPT-2-style fused attention)
  - Pros: only low-rank adapter matrices train, base weights frozen — drastically cuts trainable
    parameter count and memory footprint; adapter-only save is a few MB, not a full model copy; lets
    one frozen base model serve many swappable per-task/per-customer adapters (`PeftModel.
    from_pretrained(base, adapter_dir)`) instead of maintaining N full model copies.
  - Cons: slightly less expressive than full fine-tuning in principle (adapter is low-rank);
    requires picking rank/target modules correctly per architecture (GPT-2's fused `c_attn` vs.
    Llama/Qwen's separate `q_proj`/`k_proj`/`v_proj`/`o_proj`).
  - Pick over full fine-tuning when: compute/memory is constrained, you need to serve many
    task/customer-specific variants cheaply, or you want fast iteration on a smoke-test-sized model.
- **QLoRA (quantized base + LoRA)** — not implemented in the project but named as the upgrade path:
  load the base model in 4-bit (`bitsandbytes`) via `prepare_model_for_kbit_training()`, then apply
  LoRA on top. Pick over plain LoRA when the base model itself is too large to fit in memory at
  full/half precision.
- **RAG vs. fine-tuning for knowledge injection** (cross-referenced, not elaborated in-repo): RAG
  injects knowledge at query time via retrieval; fine-tuning bakes behavior/style into weights.
  README positions RAG (Project 02) as "an alternative to fine-tuning for knowledge injection" —
  pick fine-tuning for behavior/format/style adaptation, RAG when the knowledge changes frequently
  or must be sourced/citable.

| Option | Best for | Avoid when | Key tradeoff |
|---|---|---|---|
| Full fine-tuning | Max expressiveness, ample compute | Limited GPU memory, multi-tenant adapter needs | Best quality ceiling vs. heavy memory/storage cost |
| LoRA | Constrained compute, multi-tenant/many adapters, fast iteration | Base model already fits and full FT compute is a non-issue | Small memory/storage footprint vs. slightly less expressive |
| QLoRA | Base model too large to fit even for inference at full precision | Base model already fits comfortably | Fits huge models on modest hardware vs. quantization overhead |
| RAG (vs. fine-tuning) | Frequently-changing or citable knowledge | Need to change model behavior/style/format itself | No retraining needed vs. no weight-level behavior change |

---

## Multi-Agent Protocol Choices: MCP vs. A2A, and Client Architecture

From `04-multi-agent-a2a-mcp/README.md` and `09-mcp-agent-system/README.md`.

### MCP vs. A2A (different layers, not substitutes)
- **MCP (Model Context Protocol)** — agent-to-*tool*. Fine-grained, stateless, low-level actions
  (`search_notes`, `compute_trend`) meant to be called many times as an agent reasons.
  - Pick MCP when: exposing tools/data sources/APIs to a single agent's reasoning loop.
- **A2A (Agent2Agent)** — agent-to-*agent*. Coarse-grained handoffs of an entire unit of work with
  its own goal and lifecycle (`submitted→working→completed/failed`), meant to cross a
  trust/ownership boundary.
  - Pick A2A when: handing work between agents that may be owned/deployed by different teams or
    vendors — "you'd A2A-call another company's agent, but you wouldn't want to hand it raw MCP
    tool access to your internal database."
  - Key detail: a single system commonly needs both, as Project 04 does (three OS processes, each
    with its own MCP client(s), exchanging A2A HTTP tasks).

### Single shared MCP client vs. Hierarchical Multi-Client
- **One shared MCP client with access to every server**
  - Cons: any agent can call any tool — e.g. the Research Agent could accidentally (or if
    compromised) call `write_report` and overwrite files it has no business touching.
- **Hierarchical Multi-Client** (what Project 04 implements: Research→`notes-server` only,
  Analysis→`stats-server` only, Report→both `filesystem-server` and `notes-server`)
  - Pros: bounds each agent's blast radius to exactly the servers its job requires; a single agent
    can also hold multiple clients scoped to different servers when its job genuinely spans two
    tool domains (Report Agent needs both write access and a notes re-check).
  - Cons: more wiring/bookkeeping (N clients instead of 1).
  - Pick over a shared client whenever agents have different trust levels or destructive-action
    capabilities (write access, spending money, etc.) — security/blast-radius reduction is the
    explicit stated reason.

### Local stdio MCP servers vs. remote SSE MCP servers
(`09-mcp-agent-system/README.md`, "Variation B," prose-only, not implemented in either project)
- **Local stdio** — server spawned as a subprocess on the same machine as the client.
  - Pros: simple, no network/auth surface, matches what Claude Desktop and other local MCP hosts do.
  - Cons: client and server must run co-located.
- **Remote SSE (Server-Sent Events + HTTP POST)**
  - Pros: allows a lightweight client (mobile app, browser extension) to use powerful, centralized
    tools without distributing API keys to the client itself.
  - Pick over stdio when: the client is untrusted/lightweight and shouldn't hold the credentials the
    tool needs, or the server needs to be centrally hosted/shared across many clients.

### Real external API tool vs. mocked tool (cost/complexity tradeoff, not a general recommendation)
- `09-mcp-agent-system`'s `get_weather` hits the real, free, keyless Open-Meteo API; `search_web` is
  hardcoded/mocked because a real search API (Tavily/Brave/Google) needs a key and billing.
  Deliberate choice to keep the demo zero-setup, explicitly flagged in the docstring rather than
  presented as if it were real — a reminder to state mocked-vs-real integration status honestly in
  a live system walkthrough.

| Option | Best for | Avoid when | Key tradeoff |
|---|---|---|---|
| MCP | Agent-to-tool, in-process/local tool calling loop | Handing off a whole task across a trust boundary | Fine-grained tool access vs. not meant for cross-agent handoff |
| A2A | Agent-to-agent handoff across ownership/trust boundaries | Simple in-loop tool calls within one agent | Coarse-grained, ownership-safe handoff vs. more overhead per call |
| Shared MCP client (all servers) | Quick prototype, all agents fully trusted | Any agent has destructive/write capability | Simplicity vs. no blast-radius containment |
| Hierarchical multi-client | Agents with different trust levels/capabilities | Single-agent, single-tool-domain systems (overkill) | Security isolation vs. more clients to wire/maintain |
| Local stdio MCP server | Co-located client/server, simplest setup | Client is remote/lightweight/untrusted with the tool's credentials | Zero network/auth surface vs. requires co-location |
| Remote SSE MCP server | Centralized tools, untrusted/lightweight clients | Simple local-only demo | Credential isolation from client vs. added network/auth complexity |

---

## Production Deployment / Serving Patterns

From `01-tabular-ml-pipeline/serve.py` and `05-interview-template/README.md`.

- **Online (synchronous) serving** — stateless FastAPI `/score` endpoint, model loaded once per
  worker (`uvicorn serve:app --workers 4`), horizontally scalable behind a load balancer with no
  per-request state. Pick for real-time, single-record scoring under a tight latency budget.
- **Batch scoring** — chunked (`chunksize=10_000`) CSV read/score/write so memory stays bounded
  regardless of input file size. Pick for nightly/offline scoring jobs where latency doesn't matter
  but throughput over a large file does.
- **Both share one model artifact** — same saved pipeline (preprocessing + model as one joblib/
  torch artifact) used by both paths, avoiding training-serving skew from a reimplemented
  preprocessing step in a different service/language.
- **Shadow/canary vs. full rollout** — route a small slice of live traffic (or mirror traffic with
  no user-facing effect) to a new model before it serves real decisions; ramp gradually with a hot
  rollback path. Pick over a full rollout whenever a new model version is going live at all —
  stated as the default, not a special case.

---

## Quick-Reference: Which Project Demonstrates Which Comparison

| Comparison | Project/file |
|---|---|
| Grid vs. random vs. Bayesian HPO | `01-tabular-ml-pipeline/train_dl.py`, `05-interview-template/template/04_hyperparameter_search.py` |
| StratifiedKFold vs. KFold vs. GroupKFold vs. time split | `01-tabular-ml-pipeline/train.py`, `production-scenario/02_solution_walkthrough.md` |
| Logistic/GBT vs. DL MLP for tabular | `01-tabular-ml-pipeline`, `production-scenario/02_solution_walkthrough.md` |
| Single threshold vs. multi-tier decision policy | `production-scenario/02_solution_walkthrough.md` |
| Precompute vs. online-compute features | `production-scenario/02_solution_walkthrough.md` |
| Fixed-cadence vs. trigger-based retrain | `05-interview-template/README.md`, `production-scenario/02_solution_walkthrough.md` |
| Chunking strategies, retrieval stages, generation backends | `02-rag-pipeline/README.md` |
| LoRA vs. full fine-tuning vs. QLoRA vs. RAG | `03-llm-finetuning/README.md` |
| MCP vs. A2A, shared vs. hierarchical client, stdio vs. SSE | `04-multi-agent-a2a-mcp/README.md`, `09-mcp-agent-system/README.md` |
| Online vs. batch serving, canary vs. full rollout | `01-tabular-ml-pipeline/serve.py`, `05-interview-template/README.md` |
