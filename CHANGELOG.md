# Changelog

Tracks Phase 2 additions/updates made against `COVERAGE_AUDIT.md`'s priority queue. Newest first.

## 2026-08-11

- **Added** `11-llm-applications/11-eval-and-observability.md` — closes ❌ priority #1 (DeepEval / eval-framework landscape) and 🟡 priority #4 (observability tooling depth). Covers eval frameworks (RAGAS vs. DeepEval, offline vs. online eval), observability/tracing platforms (LangSmith, Langfuse, Arize Phoenix) with instrumentation guidance, regression testing for prompts/pipelines, and a production checklist. Fills the folder's `11-*` numbering gap.
- **Updated** `11-llm-applications/README.md` — added `11-eval-and-observability.md` to the "What Each File Helps You Do" index.
- **Updated** `12-systems-and-scale/04-llm-inference-ops.md` — closes ❌ priority #2 (prefill/decode disaggregation). Added §10 covering DistServe, Mooncake, and NVIDIA Dynamo, the disaggregated prefill/decode architecture, and when to use/not use it.
- **Updated** `12-systems-and-scale/03-llm-serving.md` — closes ❌ priority #3 (multi-tenant inference / LoRA-swap serving). Added a "Multi-Tenant Inference and LoRA Hot-Swapping" subsection under Production Deployment Patterns covering shared-base-model multi-LoRA serving, hot-swap mechanics, and vLLM's multi-LoRA support.
- **Updated** `11-llm-applications/01-agentic-workflows.md` — closes 🟡 priority #5 (cross-link MCP/A2A) and #6 (LangGraph dedicated treatment). Added an MCP/A2A cross-link in the Tool Use section, a new "LangGraph: Stateful Graph Orchestration" section (nodes/edges/state/checkpointing), and a Langfuse mention + cross-link to the new eval-and-observability file in the Observability section.
- **Updated** `11-llm-applications/02-rag.md` — closes 🟡 priority #7 (agentic RAG depth parity) and #8 (Milvus in vector-DB table). Added a full "Agentic RAG" subsection under Advanced Patterns (query decomposition, source routing, iterative retrieval, what breaks) matching the depth of Self-RAG/CRAG/GraphRAG, and added Milvus to the vector database comparison table.
