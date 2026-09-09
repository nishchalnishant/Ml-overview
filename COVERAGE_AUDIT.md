# Coverage Audit — 2026 Trend Taxonomy

Audit date: 2026-08-11. Method: 4 parallel content-audit passes (LLM core/RAG/agents/MCP/A2A; eval/serving; MLOps/data/governance; classical ML) plus direct verification reads of the highest-priority files (RAG, agentic workflows, MCP, A2A). Read-only — no content changed in this phase.

Legend: ✅ well-covered · 🟡 partial/outdated (specifics noted) · ❌ missing

---

## 1. LLMs & GenAI core

| Item | Status | Notes |
|---|---|---|
| Attention variants, MoE, long-context methods | ✅ | `10-llms/03-architecture-deep-dive.md`, `08-kv-cache-and-mqa-gqa.md`, `09-moe-advanced-and-routing.md`, `07-context-window-extension.md` — code + math + Interview Angles throughout. Very high depth. |
| Prompt engineering, structured outputs/function calling/tool use | ✅ | `11-llm-applications/03-prompt-optimization-and-versioning.md` (APE, DSPy, versioning/registry) + tool-use section in `01-agentic-workflows.md`. |
| Fine-tuning: full vs PEFT (LoRA/QLoRA), when to FT vs RAG vs prompt | ✅ | `10-llms/06-fine-tuning-at-scale.md` (code+math for LoRA/QLoRA/GPTQ/AWQ/GGUF/ZeRO) and RAG-vs-FT decision framework in `11-llm-applications/02-rag.md`. |
| Model families landscape, open vs closed tradeoffs | 🟡 | Model tables across `10-llms/03-architecture-deep-dive.md` and elsewhere stop at LLaMA 4 / Claude 3.7 Sonnet / Gemini 2.0 / DeepSeek R1 / Qwen3 — frozen ~early/mid-2025. No 2026-era model generation represented. Vendor-neutral framing is fine; just needs a freshness pass on named examples/tables, not a rewrite. |

## 2. RAG

| Item | Status | Notes |
|---|---|---|
| Chunking strategies | ✅ | `11-llm-applications/02-rag.md` — fixed/sentence/recursive/semantic/structure-aware, sizing guidance. |
| Embedding models | 🟡 | Covered (`02-rag.md`, `16-interview-prep/.../06-vector-databases-and-embeddings.md`) but named examples (bge-m3, text-embedding-3-large, ada-002 dims) are ~2024-era with nothing newer referenced. |
| Vector DBs (pgvector, Qdrant, Weaviate, Milvus) | 🟡 | Milvus is present in `16-interview-prep/.../06-vector-databases-and-embeddings.md` but **absent** from the main `11-llm-applications/02-rag.md` vector-DB table (Chroma/Pinecone/Weaviate/Qdrant/pgvector/FAISS only). Inconsistent coverage across the two files. |
| Hybrid search, reranking, query rewriting | ✅ | `02-rag.md` — RRF math, HyDE, step-back prompting, cross-encoder reranking, all with code/formulas and failure modes. |
| GraphRAG | ✅ | Deep treatment in `16-interview-prep/.../03-retrieval-augmented-generation-rag.md` (local/global search, tool comparison); appropriately summarized with a cross-link in `02-rag.md`. |
| Agentic RAG | 🟡 | Deep, dedicated section in the interview-notes file (`16-interview-prep/.../03-retrieval-augmented-generation-rag.md`) but only a single line in the production-facing `11-llm-applications/02-rag.md` — depth mismatch between the two treatments of the same topic. |
| RAG evaluation & failure modes | ✅ | RAGAS metrics table + production debugging playbook in `02-rag.md`; consistent with interview-notes companion. |

## 3. Agents & orchestration

| Item | Status | Notes |
|---|---|---|
| Agent design patterns (ReAct, planning, reflection, multi-agent) | ✅ | `11-llm-applications/01-agentic-workflows.md` — CoT, ReAct, Reflection, ToT, multi-agent (orchestrator/supervisor/debate), routing. High depth with Interview Angles. |
| LangGraph / stateful agents, orchestration frameworks | 🟡 | LangGraph mentioned only as one row in a frameworks comparison table — no dedicated treatment of stateful graph orchestration (nodes/edges/state persistence/checkpointing) despite being named explicitly in the taxonomy. |
| Model Context Protocol (MCP) | ✅ | `11-llm-applications/12-mcp.md` — excellent, current depth: Host/Client/Server architecture, transports (stdio/HTTP+SSE/Streamable HTTP), tool-call lifecycle, security (prompt injection via tool results, sandboxing), multi-agent coordination. |
| Agent2Agent (A2A) | ✅ | `11-llm-applications/13-a2a.md` — matches MCP file's rigor: Agent Cards, task lifecycle, streaming, MCP-vs-A2A comparison, security. |
| Guardrails and agent reliability | ✅ | Dedicated section in `01-agentic-workflows.md` (max steps, timeouts, schema validation, content classifiers, human-in-the-loop, sandboxing) + observability section. |
| **Cross-cutting gap** | 🟡 | `01-agentic-workflows.md` (the main agentic-patterns file) does not mention or cross-link MCP or A2A at all, despite both having thorough dedicated files. A 2026 reader of the agentic-patterns doc would not learn these protocols exist. |

## 4. Evaluation & observability

| Item | Status | Notes |
|---|---|---|
| LLM eval (RAGAS, LLM-as-judge, offline vs online) | ✅ | RAGAS well covered in RAG files; LLM-as-judge bias mitigation in `03-prompt-optimization-and-versioning.md`. |
| DeepEval and other eval-framework alternatives | ❌ | RAGAS is named repeatedly; DeepEval is not named anywhere in the repo. |
| Tracing/observability (LangSmith, Langfuse, Arize Phoenix) | 🟡 | LangSmith and Arize Phoenix are named (`01-agentic-workflows.md` Observability section), but only as a one-line tooling mention, not a dedicated topic — no comparison, no instrumentation example. **Langfuse is not named anywhere.** |
| Regression testing for prompts and pipelines | ✅ | A/B testing, rollback triggers, prompt versioning/registry covered in `03-prompt-optimization-and-versioning.md`. |

## 5. Inference & serving

| Item | Status | Notes |
|---|---|---|
| vLLM, TGI, serving architectures | ✅ | Covered in `12-systems-and-scale/03-llm-serving.md` and `11-llm-applications/06-inference-optimization.md` per earlier cluster audit. |
| Quantization (GPTQ/AWQ/GGUF), KV cache, continuous batching | ✅ | Strong — GPTQ/AWQ/GGUF in `10-llms/06-fine-tuning-at-scale.md`, KV cache deep-dive in `08-kv-cache-and-mqa-gqa.md` (incl. PagedAttention, MLA, prefix caching). |
| Speculative decoding | ✅ | Per earlier cluster audit, covered in serving/inference-ops files. |
| Prefill/decode disaggregation | ❌ | Not covered as an explicit named architecture anywhere (Dynamo, DistServe, Mooncake-style disaggregated serving) — a significant 2025-2026 production serving pattern. |
| Multi-tenant inference, LoRA-swap serving, cost/latency tradeoffs | ❌ | Not covered as an explicit topic; general cost/latency tradeoffs exist but multi-tenant/LoRA-hot-swap serving patterns are absent. |

## 6. MLOps & platform

| Item | Status | Notes |
|---|---|---|
| Experiment tracking, model registry, CI/CD for ML | ✅ | `13-production-ml/09-model-registry-versioning.md` covers MLflow/W&B/DVC-class tools; registry comparison omits Databricks Unity Catalog for models (minor 🟡, not blocking). |
| Pipelines/orchestration (Airflow, Kubeflow, Dagster) | ✅ | Per earlier cluster audit. |
| Drift detection & monitoring (Evidently, Prometheus/Grafana) | 🟡 | `13-production-ml/01-mlops.md` uses the **deprecated Evidently `Dashboard`/`DataDriftTab` API** — current Evidently versions use the `Report`-based API. Concept coverage is fine; code example is stale. |
| Feature stores (Feast, Feathr), serving skew | ✅ | Per earlier cluster audit. |

## 7. Data engineering for AI

| Item | Status | Notes |
|---|---|---|
| Data prep/quality for ML & LLMs | ✅ | Per earlier cluster audit. |
| Synthetic data generation | 🟡 | `11-llm-applications/04-synthetic-data.md` anchored in 2022-2024 papers (Self-Instruct, Alpaca, LIMA, Phi); most recent reference is DeepSeek-R1. Missing PII/differential-privacy synthetic data angle and newer tooling (Nemotron, Distilabel, DataDreamer). |
| Streaming (Kafka), batch vs streaming feature computation | ✅ | Per earlier cluster audit. |

## 8. AI governance, safety & security

| Item | Status | Notes |
|---|---|---|
| Prompt injection/jailbreak defense, PII handling, output guardrails | ✅ | Strong — `14-responsible-ai/03-llm-safety-and-governance.md` plus prompt-injection coverage in `12-mcp.md` security section. |
| Model cards, lineage, auditability, regulatory context | 🟡 | `14-responsible-ai/03-llm-safety-and-governance.md` has explicit EU AI Act risk tiers (the strongest regulatory-currency file in the repo), but `13-production-ml/03-model-governance.md` (credit-scoring use case — a textbook EU AI Act "high-risk" example) relies only on GDPR/US regs (ECOA/FCRA/HIPAA) with **no EU AI Act cross-reference** despite the two files covering overlapping ground. |
| Bias/fairness metrics, explainability (SHAP) | ✅ | Per earlier cluster audit. |

## 9. Classical ML foundations

| Item | Status | Notes |
|---|---|---|
| Core algorithms, bias/variance, regularization, cross-validation, feature engineering | ✅ | Per earlier cluster audit — strong, evergreen coverage. |
| Boosting (XGBoost/LightGBM), imbalanced data, evaluation metrics | 🟡 | `03-classical-ml/01-supervised-learning.md` line ~362 uses the **deprecated `use_label_encoder=False` XGBoost parameter** — removed in current XGBoost versions. Minor, isolated fix. |

---

## Summary counts (original Phase 1 assessment, 2026-08-11)

- ✅ Well-covered: 20
- 🟡 Partial/outdated: 12
- ❌ Missing: 3

---

## Resolution status — all items closed as of 2026-09-09

**❌ Missing — closed in Phase 2 (2026-08-11):**

| # | Item | Resolution |
|---|---|---|
| 1 | DeepEval / eval-framework landscape | `11-llm-applications/11-eval-and-observability.md` created |
| 2 | Prefill/decode disaggregation | `12-systems-and-scale/04-llm-inference-ops.md` §10 (DistServe, Mooncake, Dynamo) |
| 3 | Multi-tenant / LoRA-swap serving | `12-systems-and-scale/03-llm-serving.md` multi-LoRA subsection |

**🟡 Partial/outdated — closed across Phase 2 and Phase 3 (2026-09-09):**

| # | Item | Resolution |
|---|---|---|
| 4 | Observability tooling depth (Langfuse absent) | Phase 2 — dedicated file, Langfuse/LangSmith/Phoenix instrumentation |
| 5 | Cross-link MCP/A2A from agentic-workflows | Phase 2 |
| 6 | LangGraph dedicated treatment | Phase 2 — stateful graph orchestration section |
| 7 | Agentic RAG depth mismatch | Phase 2 — full subsection in `02-rag.md` |
| 8 | Milvus missing from vector-DB table | Phase 2 |
| 9 | EU AI Act cross-ref in model-governance | Closed — `13-production-ml/03-model-governance.md` now carries the Annex III high-risk framing and a checklist item |
| 10 | Deprecated Evidently `Dashboard`/`DataDriftTab` | Closed — `Report`-based API, verified against evidently 0.7.x |
| 11 | Deprecated XGBoost `use_label_encoder` | Closed — removed, with an inline note on why |
| 12 | Synthetic-data anchors pre-2024 | Closed — DP/privacy-preserving synthesis section + Distilabel/DataDreamer/Nemotron tooling table |
| 13 | Model-family tables frozen ~mid-2025 | **Phase 3** — added a freshness disclaimer and §12.1 "What Actually Generalizes": six durable architectural axes (sparse-over-dense, KV cache as binding constraint, deliberate overtraining, test-time compute, hybrid positional encoding, early-fusion multimodality) plus interview framing for models past the knowledge cutoff. Structural fix rather than a name refresh, since any name list re-stales within months. |
| 14 | Embedding-model examples ~2024-era | **Phase 3** — replaced the name-drop with a four-axis selection framework (dimensionality/Matryoshka, sequence length, domain match, licence+hosting) and an MTEB-overfitting caveat directing readers to evaluate on their own labelled pairs |
| 15 | Model registry: Unity Catalog absent | **Phase 3** — added to the comparison table |

### Phase 3 also addressed (structural audit, outside the original queue)

- **20 broken relative links** repaired — stale paths from a pre-renumbering layout (`02-classical-ml`, `03-deep-learning`, `05-llms`, `06-production-ml`, `07-interview-prep`, `12-projects`), concentrated in `17-projects/*/README.md` and `16-interview-prep/ea/`.
- **63 files missing `status:` frontmatter** — restored the 100% invariant.
- **`08-generative/03-diffusion.md` depth gap** — the one folder whose thin coverage had no deeper home elsewhere in the repo. Added score matching / flow matching unification, the hierarchical-VAE/ELBO connection, conditioning mechanics (CFG's 2× inference cost, ControlNet zero-init), and the U-Net → DiT shift.

### Topic-coverage sweep (Phase 3)

~44 canonical ML-interview topics checked across statistics/classical ML, deep learning, LLMs, and systems/production. **No missing topics.** Thinnest but present: curriculum learning, CLT, survival analysis. Two apparent gaps confirmed as intentional design: `02-data/07-data-engineering.md` is a redirect stub, and the thin `06-architectures` attention file is layered above deep treatment in `10-llms/`.

---

*Phase 1 was read-only. Phases 2 and 3 made the changes recorded above and in `CHANGELOG.md`.*
