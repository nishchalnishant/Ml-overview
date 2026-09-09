# Changelog

Tracks additions/updates made against `COVERAGE_AUDIT.md`'s priority queue. Newest first.

## 2026-09-09 — Phase 3 (structural audit + queue closeout)

**Structural integrity**

- **Fixed** 20 broken relative links across `17-projects/*/README.md`, `16-interview-prep/ea/00-ml-deep-learning-interview.md`, and `10-llms/README.md`. All were stale paths from a pre-renumbering repo layout (`02-classical-ml`→`03-classical-ml`, `03-deep-learning`→`05-deep-learning-core`, `05-llms`→`10-llms`, `06-production-ml`→`13-production-ml`, `07-interview-prep`→`16-interview-prep`, `12-projects`→`17-projects`, `09-mcp-agent-system`→`11-llm-applications/12-mcp.md`, `10-llms/interview-notes/`→`16-interview-prep/llm-interview-notes/`).
- **Fixed** 63 files missing `status:` frontmatter, restoring the 100% coverage invariant.

**Content — closes remaining 🟡 queue items**

- **Updated** `10-llms/03-architecture-deep-dive.md` — closes 🟡 #13 (model tables frozen ~mid-2025). Added a freshness disclaimer and a new §12.1 "What Actually Generalizes": the six architectural axes that outlast any model generation, plus interview framing for questions about models past a knowledge cutoff. Chose a structural fix over a name refresh because a refreshed name list re-stales within months.
- **Updated** `11-llm-applications/02-rag.md` — closes 🟡 #14 (embedding examples ~2024-era). Replaced the name-drop with a four-axis selection framework (dimensionality/Matryoshka truncation, max sequence length, domain match, licence+hosting), reference points by category including biomedical encoders, and an MTEB-overfitting caveat.
- **Updated** `13-production-ml/09-model-registry-versioning.md` — closes 🟡 #15. Added Databricks Unity Catalog to the registry comparison table.
- **Updated** `08-generative/03-diffusion.md` — addresses the one genuine depth gap found by the Phase 3 topic sweep (102 → 173 lines). Added the score-matching/flow-matching unification (score ≡ rescaled noise prediction, probability-flow ODE, rectified flow), the hierarchical-VAE/ELBO connection, conditioning mechanics (CFG's 2× per-step inference cost, ControlNet zero-init parallel to LoRA), the U-Net → DiT architectural shift, and two interview angles (score matching; text-to-image latency cost breakdown).
- **Updated** `COVERAGE_AUDIT.md` — replaced the stale priority queue with a resolution table. Items #9-#12 were found already fixed by earlier work but never marked closed, which made them appear outstanding.

**Topic-coverage sweep:** ~44 canonical ML-interview topics verified across statistics/classical ML, deep learning, LLMs, and systems/production. No missing topics.

## 2026-08-11

- **Added** `11-llm-applications/11-eval-and-observability.md` — closes ❌ priority #1 (DeepEval / eval-framework landscape) and 🟡 priority #4 (observability tooling depth). Covers eval frameworks (RAGAS vs. DeepEval, offline vs. online eval), observability/tracing platforms (LangSmith, Langfuse, Arize Phoenix) with instrumentation guidance, regression testing for prompts/pipelines, and a production checklist. Fills the folder's `11-*` numbering gap.
- **Updated** `11-llm-applications/README.md` — added `11-eval-and-observability.md` to the "What Each File Helps You Do" index.
- **Updated** `12-systems-and-scale/04-llm-inference-ops.md` — closes ❌ priority #2 (prefill/decode disaggregation). Added §10 covering DistServe, Mooncake, and NVIDIA Dynamo, the disaggregated prefill/decode architecture, and when to use/not use it.
- **Updated** `12-systems-and-scale/03-llm-serving.md` — closes ❌ priority #3 (multi-tenant inference / LoRA-swap serving). Added a "Multi-Tenant Inference and LoRA Hot-Swapping" subsection under Production Deployment Patterns covering shared-base-model multi-LoRA serving, hot-swap mechanics, and vLLM's multi-LoRA support.
- **Updated** `11-llm-applications/01-agentic-workflows.md` — closes 🟡 priority #5 (cross-link MCP/A2A) and #6 (LangGraph dedicated treatment). Added an MCP/A2A cross-link in the Tool Use section, a new "LangGraph: Stateful Graph Orchestration" section (nodes/edges/state/checkpointing), and a Langfuse mention + cross-link to the new eval-and-observability file in the Observability section.
- **Updated** `11-llm-applications/02-rag.md` — closes 🟡 priority #7 (agentic RAG depth parity) and #8 (Milvus in vector-DB table). Added a full "Agentic RAG" subsection under Advanced Patterns (query decomposition, source routing, iterative retrieval, what breaks) matching the depth of Self-RAG/CRAG/GraphRAG, and added Milvus to the vector database comparison table.
