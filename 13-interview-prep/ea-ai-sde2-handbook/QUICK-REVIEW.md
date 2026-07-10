# Quick Review Index — EA AI/SDE2 Interview Handbook

Master index for the 30 condensed cheat sheets in `condensed/`. **Skim this file first** — it's the single most important page to reread right before the interview.

## How to use this in 1-2 days

- Each cheat sheet takes **~3 minutes** to read (they're 70-110 lines). All 30 = ~90 minutes total, so a full pass is doable even the night before.
- **Day 1 — Game/EA-specific systems (01-13):** matchmaking, churn, toxicity, NPCs, cheat detection, pricing, crash prediction, LiveOps, recommendations, voice, economy, support RAG, semantic search. These are the scenarios most likely to open the interview, since they're EA's actual domain.
- **Day 2 — General AI engineering (14-30):** feature stores, RAG pipelines, anomaly/drift detection, recsys at scale, edge serving, A/B testing, VLM agents, RL, GenAI content, knowledge graphs, distributed training, MLOps. These test whether you can generalize ML-system fundamentals beyond games.
- Before the interview: read the **Cross-Cutting Patterns** section below twice — it's the highest-density signal-to-time ratio in this whole handbook.
- Under time pressure, read only each file's **"Talking Points"** and **"Biggest Pitfall"** sections — that's ~80% of the interview signal in ~30% of the words.

## Topics — Game/EA-Specific (Day 1)

- [01 — Matchmaking](condensed/01-matchmaking.md) — real-time SBMM, LightGBM win-probability model, Platt calibration, fairness window vs queue time
- [02 — Churn Prediction](condensed/02-churn-prediction.md) — daily batch XGBoost churn classifier, cost-aware intervention tiers, uplift/CATE modeling
- [03 — Toxic Chat](condensed/03-toxic-chat.md) — sub-50ms text moderation funnel (regex → cache → DistilBERT/ONNX), fail-open, per-game thresholds
- [04 — NPC Assistant](condensed/04-npc-assistant.md) — streaming LLM dialogue brain, vLLM/PagedAttention, persona-scoped RAG, jailbreak guardrails
- [05 — Cheat Detection](condensed/05-cheat-detection.md) — server-side aimbot detection from telemetry, XGBoost + SHAP explainability, conservative ban threshold
- [06 — Dynamic Pricing](condensed/06-dynamic-pricing.md) — personalized coupons as contextual bandit, uplift over absolute LTV, epsilon-greedy exploration
- [07 — Crash Prediction](condensed/07-crash-prediction.md) — on-device XGBoost/Treelite crash prediction, PR-AUC under 1:10,000 imbalance, shadow-mode counterfactual eval
- [08 — LiveOps Agent](condensed/08-liveops-agent.md) — ReAct tool-calling agent over Snowflake, human-in-the-loop approval, backend-enforced guardrails
- [09 — Store Recommendations](condensed/09-store-recommendations.md) — Two-Tower retrieval+ranking for 50M users, real-time inventory filtering, position-bias correction
- [10 — Voice Moderation](condensed/10-voice-moderation.md) — streaming STT + text toxicity model, VAD gating, GDPR-safe ephemeral ring buffer
- [11 — Economy Prediction](condensed/11-economy-prediction.md) — Prophet-based faucet/sink inflation forecasting, dynamic seasonality-aware thresholds
- [12 — Support RAG](condensed/12-support-rag.md) — ACL-filtered hybrid search (BM25+dense) over Confluence/Jira, semantic caching, recency-biased reranking
- [13 — Semantic Search](condensed/13-semantic-search.md) — CLIP-based multimodal asset search, 2D-thumbnail bridge for 3D meshes, HNSW/quantization at 10M scale

## Topics — General AI Engineering (Day 2)

- [14 — Feature Store](condensed/14-feature-store.md) — online/offline dual-store design, point-in-time AS-OF joins, training-serving skew prevention
- [15 — LLM RAG Pipeline](condensed/15-llm-rag-pipeline.md) — Markdown/AST-aware chunking, hybrid search + reranking for code docs, hallucinated-API verification
- [16 — Anomaly Detection](condensed/16-anomaly-detection.md) — unsupervised Isolation Forest over streaming server telemetry, multivariate + trend anomalies
- [17 — Recommendation System](condensed/17-recommendation-system.md) — implicit-feedback ALS matrix factorization, item cold-start, popularity-bias correction
- [18 — Edge Model Serving](condensed/18-edge-model-serving.md) — PyTorch→ONNX/CoreML quantization pipeline for mobile, PTQ vs QAT, encrypted OTA updates
- [19 — A/B Testing](condensed/19-ab-testing.md) — switchback design to avoid SUTVA/interference, SRM detection, peeking/multiple-comparisons discipline
- [20 — VLM QA Agent](condensed/20-vlm-qa-agent.md) — Set-of-Mark visual grounding for game UI automation, self-hosted VLM for IP safety, action-trajectory caching
- [21 — Data Drift](condensed/21-data-drift.md) — PSI-based drift monitoring across 5,000 features, SHAP-value drift, data-quality gate before retraining
- [22 — RL NPC](condensed/22-rl-npc.md) — PPO actor-critic for shooter NPCs, reward-hacking prevention, domain randomization for generalization
- [23 — GenAI 3D Assets](condensed/23-genai-3d-assets.md) — ControlNet-constrained diffusion + feed-forward reconstruction (TripoSR), game-ready mesh post-processing
- [24 — Knowledge Graph](condensed/24-knowledge-graph.md) — LLM-based information extraction into Neo4j, entity resolution, epistemic tracking for conflicting lore
- [25 — Anti-Cheat](condensed/25-anti-cheat.md) — client-side LightGBM tripwire under <1% CPU budget, physics-based mouse features, ban waves over instant bans
- [26 — Generative Audio](condensed/26-generative-audio.md) — streaming VITS/XTTS voice cloning for 500 NPCs, sentence-level chunking, cross-lingual cloning
- [27 — Latency Prediction](condensed/27-latency-prediction.md) — GRU-based client-side position prediction, delta coordinates, server stays authoritative for hits
- [28 — Distributed Training](condensed/28-distributed-training.md) — FSDP sharding for a 10B-param model, BF16, elastic torchrun fault tolerance
- [29 — Dynamic Difficulty](condensed/29-dynamic-difficulty.md) — contextual bandit (Thompson Sampling) DDA, boredom vs frustration reward shaping, sandbagging detection
- [30 — MLOps Pipeline](condensed/30-mlops-pipeline.md) — notebook-to-production pipeline, model registry, shadow→canary→full rollout, drift monitoring

## Cross-Cutting Patterns — "if you only remember one list, remember this"

These talking points recur across most of the 30 scenarios. Volunteering them unprompted is what separates a senior-sounding answer from a merely-correct one.

- **Calibrate probabilistic outputs before gating decisions on them.** A [0.45, 0.55] fairness window or a 0.90 churn threshold is meaningless on raw, uncalibrated model scores (01, 02, 06, 30).
- **Shadow-mode before canary before full rollout.** Compare new model predictions against production silently first, then a small % of live traffic, before a full cutover (01, 09, 16, 21, 30).
- **PSI over KS-test / raw thresholds for drift detection at scale.** KS-test p-values collapse to significant on trivial shifts once N is large; PSI stays interpretable (01, 11, 21).
- **Fallback to a simple heuristic or cached/stale value when the model/service fails** — never let an ML outage break the product (Redis stale-but-safe reads, fail-open toxicity filters, most-popular-item fallback) (03, 06, 09, 10, 30).
- **Redis (or similar KV store) for shared, horizontally-scalable state**, not in-memory-per-pod — needed for HA queues, feature serving, bandit assignments, and dynamic config (01, 06, 09, 14, 19).
- **Survivorship bias / SUTVA in experiment design.** Tighter constraints (fairness windows, discounts) change who sticks around to be measured — always check retention/dropout, not just the primary metric; and never bucket by user in systems with network effects like matchmaking (01, 19).
- **Uplift/incremental modeling over predicting absolute outcomes.** Predicting who'll churn or who'll buy isn't the same as predicting who a treatment actually moves — apply this to churn interventions, pricing, and DDA (02, 06).
- **Separate cosmetic/visual ML from authoritative/safety-critical decisions.** Client-side predictions (position prediction, crash prediction) must never be the source of truth for hit detection, bans, or money — server stays authoritative (05, 25, 27).
- **Batch vs real-time is a business-need decision, not a default.** Don't reach for Kafka/Flink when a nightly Airflow batch job satisfies the actual SLA, and vice versa (02, 06, 11, 21).
- **Latency/compute budget determines model choice, not accuracy alone.** GBDT/trees over neural nets when sub-millisecond or on-device inference is required; state the constraint before proposing the model (01, 03, 05, 07, 25, 27).
- **Explicit false-positive vs false-negative tradeoff, tied to product/PR consequences**, not treated as a generic threshold-tuning exercise — bans, cosmetic offers, and auto-mutes all have asymmetric costs (03, 05, 06, 10, 25).
- **Human-in-the-loop / approval gates on any irreversible or high-stakes automated action** (bans, currency grants, model promotion to prod, campaign sends) — the model/agent proposes, a human or hardcoded backend rule disposes (08, 24, 30).
- **Point-in-time correctness / no label leakage** when constructing training data — AS-OF joins with a bounded tolerance, not naive joins on timestamp (02, 14).
- **Cold-start handling via metadata/content features**, not "collect more data" — applies to new items, new players, and new NPCs alike (09, 13, 17).
- **Concept drift is often structural (patches, seasonal, meta shifts), not gradual** — retrain on recency-weighted or truncated windows after known breaks, and gate retraining behind a data-quality check so pipeline bugs don't get baked in as "new normal" (01, 02, 11, 16, 21).
- **Name the interpretability/explainability requirement (SHAP) whenever a human downstream needs a reason** — ban appeals, cheat review, drift root-causing (05, 16, 21).
