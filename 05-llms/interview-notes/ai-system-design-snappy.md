# AI system design — Azure/DevOps-fluent templates

These are **system design skeletons** you can speak out loud. Short, structured, and production-minded.

**Answer rhythm:** requirements → architecture → data flow → evals → safety → cost/latency trade-offs.

---

# Q1: Design an AI-powered customer support chatbot.
- **Clarify (30s):** users, volume, latency, languages, privacy/PII, citations, failure tolerance.
- **Core architecture:** client → gateway → (router) → LLM/RAG/tools → post-process/validators → response.
- **Azure mapping:** API Management + App Insights + Key Vault + AKS/managed endpoint + storage + vector search.
- **Quality gates:** eval suite, canary, rollback; monitor TTFT/p95 + quality (faithfulness/safety).
- **Safety:** prompt injection defenses, allow-listed tools, HITL for high-risk actions.

---

# Q2: Design a document Q&A system for enterprise use.
- **Clarify (30s):** users, volume, latency, languages, privacy/PII, citations, failure tolerance.
- **Core architecture:** client → gateway → (router) → LLM/RAG/tools → post-process/validators → response.
- **Azure mapping:** API Management + App Insights + Key Vault + AKS/managed endpoint + storage + vector search.
- **Quality gates:** eval suite, canary, rollback; monitor TTFT/p95 + quality (faithfulness/safety).
- **Safety:** prompt injection defenses, allow-listed tools, HITL for high-risk actions.
- **RAG note:** chunking + hybrid search + reranking + citations; store metadata for audit.

---

# Q3: Design a code generation and review system.
- **Clarify (30s):** users, volume, latency, languages, privacy/PII, citations, failure tolerance.
- **Core architecture:** client → gateway → (router) → LLM/RAG/tools → post-process/validators → response.
- **Azure mapping:** API Management + App Insights + Key Vault + AKS/managed endpoint + storage + vector search.
- **Quality gates:** eval suite, canary, rollback; monitor TTFT/p95 + quality (faithfulness/safety).
- **Safety:** prompt injection defenses, allow-listed tools, HITL for high-risk actions.
- **Code note:** structured output + sandbox execution + policy checks; never auto-merge without review.

---

# Q4: Design a content moderation system using AI.
- **Clarify (30s):** users, volume, latency, languages, privacy/PII, citations, failure tolerance.
- **Core architecture:** client → gateway → (router) → LLM/RAG/tools → post-process/validators → response.
- **Azure mapping:** API Management + App Insights + Key Vault + AKS/managed endpoint + storage + vector search.
- **Quality gates:** eval suite, canary, rollback; monitor TTFT/p95 + quality (faithfulness/safety).
- **Safety:** prompt injection defenses, allow-listed tools, HITL for high-risk actions.

---

# Q5: Design a real-time AI recommendation system.
- **Clarify (30s):** users, volume, latency, languages, privacy/PII, citations, failure tolerance.
- **Core architecture:** client → gateway → (router) → LLM/RAG/tools → post-process/validators → response.
- **Azure mapping:** API Management + App Insights + Key Vault + AKS/managed endpoint + storage + vector search.
- **Quality gates:** eval suite, canary, rollback; monitor TTFT/p95 + quality (faithfulness/safety).
- **Safety:** prompt injection defenses, allow-listed tools, HITL for high-risk actions.
- **Decisioning note:** offline training + online serving; guardrails for feedback loops and abuse.

---

# Q6: Design a multi-modal search system (text, image, video).
- **Clarify (30s):** users, volume, latency, languages, privacy/PII, citations, failure tolerance.
- **Core architecture:** client → gateway → (router) → LLM/RAG/tools → post-process/validators → response.
- **Azure mapping:** API Management + App Insights + Key Vault + AKS/managed endpoint + storage + vector search.
- **Quality gates:** eval suite, canary, rollback; monitor TTFT/p95 + quality (faithfulness/safety).
- **Safety:** prompt injection defenses, allow-listed tools, HITL for high-risk actions.

---

# Q7: Design an AI-powered email assistant.
- **Clarify (30s):** users, volume, latency, languages, privacy/PII, citations, failure tolerance.
- **Core architecture:** client → gateway → (router) → LLM/RAG/tools → post-process/validators → response.
- **Azure mapping:** API Management + App Insights + Key Vault + AKS/managed endpoint + storage + vector search.
- **Quality gates:** eval suite, canary, rollback; monitor TTFT/p95 + quality (faithfulness/safety).
- **Safety:** prompt injection defenses, allow-listed tools, HITL for high-risk actions.

---

# Q8: Design a medical diagnosis assistant using AI.
- **Clarify (30s):** users, volume, latency, languages, privacy/PII, citations, failure tolerance.
- **Core architecture:** client → gateway → (router) → LLM/RAG/tools → post-process/validators → response.
- **Azure mapping:** API Management + App Insights + Key Vault + AKS/managed endpoint + storage + vector search.
- **Quality gates:** eval suite, canary, rollback; monitor TTFT/p95 + quality (faithfulness/safety).
- **Safety:** prompt injection defenses, allow-listed tools, HITL for high-risk actions.
- **Regulated note:** audit logs, explainability, refusal modes; prefer ‘assist’ not ‘decide’.

---

# Q9: Design a fraud detection system powered by LLMs.
- **Clarify (30s):** users, volume, latency, languages, privacy/PII, citations, failure tolerance.
- **Core architecture:** client → gateway → (router) → LLM/RAG/tools → post-process/validators → response.
- **Azure mapping:** API Management + App Insights + Key Vault + AKS/managed endpoint + storage + vector search.
- **Quality gates:** eval suite, canary, rollback; monitor TTFT/p95 + quality (faithfulness/safety).
- **Safety:** prompt injection defenses, allow-listed tools, HITL for high-risk actions.

---

# Q10: Design a data extraction pipeline from unstructured documents.
- **Clarify (30s):** users, volume, latency, languages, privacy/PII, citations, failure tolerance.
- **Core architecture:** client → gateway → (router) → LLM/RAG/tools → post-process/validators → response.
- **Azure mapping:** API Management + App Insights + Key Vault + AKS/managed endpoint + storage + vector search.
- **Quality gates:** eval suite, canary, rollback; monitor TTFT/p95 + quality (faithfulness/safety).
- **Safety:** prompt injection defenses, allow-listed tools, HITL for high-risk actions.
- **RAG note:** chunking + hybrid search + reranking + citations; store metadata for audit.

---

# Q11: Design a personalized learning assistant.
- **Clarify (30s):** users, volume, latency, languages, privacy/PII, citations, failure tolerance.
- **Core architecture:** client → gateway → (router) → LLM/RAG/tools → post-process/validators → response.
- **Azure mapping:** API Management + App Insights + Key Vault + AKS/managed endpoint + storage + vector search.
- **Quality gates:** eval suite, canary, rollback; monitor TTFT/p95 + quality (faithfulness/safety).
- **Safety:** prompt injection defenses, allow-listed tools, HITL for high-risk actions.

---

# Q12: Design an AI system for automated code migration.
- **Clarify (30s):** users, volume, latency, languages, privacy/PII, citations, failure tolerance.
- **Core architecture:** client → gateway → (router) → LLM/RAG/tools → post-process/validators → response.
- **Azure mapping:** API Management + App Insights + Key Vault + AKS/managed endpoint + storage + vector search.
- **Quality gates:** eval suite, canary, rollback; monitor TTFT/p95 + quality (faithfulness/safety).
- **Safety:** prompt injection defenses, allow-listed tools, HITL for high-risk actions.
- **Code note:** structured output + sandbox execution + policy checks; never auto-merge without review.

---

# Q13: Design a legal document review system.
- **Clarify (30s):** users, volume, latency, languages, privacy/PII, citations, failure tolerance.
- **Core architecture:** client → gateway → (router) → LLM/RAG/tools → post-process/validators → response.
- **Azure mapping:** API Management + App Insights + Key Vault + AKS/managed endpoint + storage + vector search.
- **Quality gates:** eval suite, canary, rollback; monitor TTFT/p95 + quality (faithfulness/safety).
- **Safety:** prompt injection defenses, allow-listed tools, HITL for high-risk actions.
- **RAG note:** chunking + hybrid search + reranking + citations; store metadata for audit.

---

# Q14: Design a conversational AI system with memory across sessions.
- **Clarify (30s):** users, volume, latency, languages, privacy/PII, citations, failure tolerance.
- **Core architecture:** client → gateway → (router) → LLM/RAG/tools → post-process/validators → response.
- **Azure mapping:** API Management + App Insights + Key Vault + AKS/managed endpoint + storage + vector search.
- **Quality gates:** eval suite, canary, rollback; monitor TTFT/p95 + quality (faithfulness/safety).
- **Safety:** prompt injection defenses, allow-listed tools, HITL for high-risk actions.

---

# Q15: Design for latency vs quality trade-offs.
- **Clarify (30s):** users, volume, latency, languages, privacy/PII, citations, failure tolerance.
- **Core architecture:** client → gateway → (router) → LLM/RAG/tools → post-process/validators → response.
- **Azure mapping:** API Management + App Insights + Key Vault + AKS/managed endpoint + storage + vector search.
- **Quality gates:** eval suite, canary, rollback; monitor TTFT/p95 + quality (faithfulness/safety).
- **Safety:** prompt injection defenses, allow-listed tools, HITL for high-risk actions.

---

# Q16: Implement caching strategies for LLM apps.
- **Clarify (30s):** users, volume, latency, languages, privacy/PII, citations, failure tolerance.
- **Core architecture:** client → gateway → (router) → LLM/RAG/tools → post-process/validators → response.
- **Azure mapping:** API Management + App Insights + Key Vault + AKS/managed endpoint + storage + vector search.
- **Quality gates:** eval suite, canary, rollback; monitor TTFT/p95 + quality (faithfulness/safety).
- **Safety:** prompt injection defenses, allow-listed tools, HITL for high-risk actions.

---

# Q17: Design rate limiting and cost management for AI APIs.
- **Clarify (30s):** users, volume, latency, languages, privacy/PII, citations, failure tolerance.
- **Core architecture:** client → gateway → (router) → LLM/RAG/tools → post-process/validators → response.
- **Azure mapping:** API Management + App Insights + Key Vault + AKS/managed endpoint + storage + vector search.
- **Quality gates:** eval suite, canary, rollback; monitor TTFT/p95 + quality (faithfulness/safety).
- **Safety:** prompt injection defenses, allow-listed tools, HITL for high-risk actions.

---

# Q18: Handle failover and fallback strategies.
- **Clarify (30s):** users, volume, latency, languages, privacy/PII, citations, failure tolerance.
- **Core architecture:** client → gateway → (router) → LLM/RAG/tools → post-process/validators → response.
- **Azure mapping:** API Management + App Insights + Key Vault + AKS/managed endpoint + storage + vector search.
- **Quality gates:** eval suite, canary, rollback; monitor TTFT/p95 + quality (faithfulness/safety).
- **Safety:** prompt injection defenses, allow-listed tools, HITL for high-risk actions.

---

# Q19: High availability and fault tolerance for AI systems.
- **Clarify (30s):** users, volume, latency, languages, privacy/PII, citations, failure tolerance.
- **Core architecture:** client → gateway → (router) → LLM/RAG/tools → post-process/validators → response.
- **Azure mapping:** API Management + App Insights + Key Vault + AKS/managed endpoint + storage + vector search.
- **Quality gates:** eval suite, canary, rollback; monitor TTFT/p95 + quality (faithfulness/safety).
- **Safety:** prompt injection defenses, allow-listed tools, HITL for high-risk actions.
- **Reliability:** multi-region + fallback models + queueing + graceful degradation.

---

# Q20: Graceful degradation when the model is unavailable.
- **Clarify (30s):** users, volume, latency, languages, privacy/PII, citations, failure tolerance.
- **Core architecture:** client → gateway → (router) → LLM/RAG/tools → post-process/validators → response.
- **Azure mapping:** API Management + App Insights + Key Vault + AKS/managed endpoint + storage + vector search.
- **Quality gates:** eval suite, canary, rollback; monitor TTFT/p95 + quality (faithfulness/safety).
- **Safety:** prompt injection defenses, allow-listed tools, HITL for high-risk actions.

---

# Q21: Multi-region deployment considerations.
- **Clarify (30s):** users, volume, latency, languages, privacy/PII, citations, failure tolerance.
- **Core architecture:** client → gateway → (router) → LLM/RAG/tools → post-process/validators → response.
- **Azure mapping:** API Management + App Insights + Key Vault + AKS/managed endpoint + storage + vector search.
- **Quality gates:** eval suite, canary, rollback; monitor TTFT/p95 + quality (faithfulness/safety).
- **Safety:** prompt injection defenses, allow-listed tools, HITL for high-risk actions.
- **Reliability:** multi-region + fallback models + queueing + graceful degradation.

---

# Q22: Design an AI-powered e-commerce search engine.
- **Clarify (30s):** users, volume, latency, languages, privacy/PII, citations, failure tolerance.
- **Core architecture:** client → gateway → (router) → LLM/RAG/tools → post-process/validators → response.
- **Azure mapping:** API Management + App Insights + Key Vault + AKS/managed endpoint + storage + vector search.
- **Quality gates:** eval suite, canary, rollback; monitor TTFT/p95 + quality (faithfulness/safety).
- **Safety:** prompt injection defenses, allow-listed tools, HITL for high-risk actions.

---

# Q23: Design an AI gateway/proxy for org-wide LLM access.
- **Clarify (30s):** users, volume, latency, languages, privacy/PII, citations, failure tolerance.
- **Core architecture:** client → gateway → (router) → LLM/RAG/tools → post-process/validators → response.
- **Azure mapping:** API Management + App Insights + Key Vault + AKS/managed endpoint + storage + vector search.
- **Quality gates:** eval suite, canary, rollback; monitor TTFT/p95 + quality (faithfulness/safety).
- **Safety:** prompt injection defenses, allow-listed tools, HITL for high-risk actions.

---

# Q24: RAG with conflicting sources.
- **Clarify (30s):** users, volume, latency, languages, privacy/PII, citations, failure tolerance.
- **Core architecture:** client → gateway → (router) → LLM/RAG/tools → post-process/validators → response.
- **Azure mapping:** API Management + App Insights + Key Vault + AKS/managed endpoint + storage + vector search.
- **Quality gates:** eval suite, canary, rollback; monitor TTFT/p95 + quality (faithfulness/safety).
- **Safety:** prompt injection defenses, allow-listed tools, HITL for high-risk actions.

---

# Q25: Capacity planning for an AI system.
- **Clarify (30s):** users, volume, latency, languages, privacy/PII, citations, failure tolerance.
- **Core architecture:** client → gateway → (router) → LLM/RAG/tools → post-process/validators → response.
- **Azure mapping:** API Management + App Insights + Key Vault + AKS/managed endpoint + storage + vector search.
- **Quality gates:** eval suite, canary, rollback; monitor TTFT/p95 + quality (faithfulness/safety).
- **Safety:** prompt injection defenses, allow-listed tools, HITL for high-risk actions.

---

# Q26: Multi-tenant custom chatbot platform.
- **Clarify (30s):** users, volume, latency, languages, privacy/PII, citations, failure tolerance.
- **Core architecture:** client → gateway → (router) → LLM/RAG/tools → post-process/validators → response.
- **Azure mapping:** API Management + App Insights + Key Vault + AKS/managed endpoint + storage + vector search.
- **Quality gates:** eval suite, canary, rollback; monitor TTFT/p95 + quality (faithfulness/safety).
- **Safety:** prompt injection defenses, allow-listed tools, HITL for high-risk actions.

---

# Q27: Meeting summarizer at scale.
- **Clarify (30s):** users, volume, latency, languages, privacy/PII, citations, failure tolerance.
- **Core architecture:** client → gateway → (router) → LLM/RAG/tools → post-process/validators → response.
- **Azure mapping:** API Management + App Insights + Key Vault + AKS/managed endpoint + storage + vector search.
- **Quality gates:** eval suite, canary, rollback; monitor TTFT/p95 + quality (faithfulness/safety).
- **Safety:** prompt injection defenses, allow-listed tools, HITL for high-risk actions.

---

# Q28: Notification prioritizer (not broadcaster).
- **Clarify (30s):** users, volume, latency, languages, privacy/PII, citations, failure tolerance.
- **Core architecture:** client → gateway → (router) → LLM/RAG/tools → post-process/validators → response.
- **Azure mapping:** API Management + App Insights + Key Vault + AKS/managed endpoint + storage + vector search.
- **Quality gates:** eval suite, canary, rollback; monitor TTFT/p95 + quality (faithfulness/safety).
- **Safety:** prompt injection defenses, allow-listed tools, HITL for high-risk actions.

---

# Q29: Anomaly detection for cloud infra.
- **Clarify (30s):** users, volume, latency, languages, privacy/PII, citations, failure tolerance.
- **Core architecture:** client → gateway → (router) → LLM/RAG/tools → post-process/validators → response.
- **Azure mapping:** API Management + App Insights + Key Vault + AKS/managed endpoint + storage + vector search.
- **Quality gates:** eval suite, canary, rollback; monitor TTFT/p95 + quality (faithfulness/safety).
- **Safety:** prompt injection defenses, allow-listed tools, HITL for high-risk actions.

---

# Q30: Document processing for financial institutions.
- **Clarify (30s):** users, volume, latency, languages, privacy/PII, citations, failure tolerance.
- **Core architecture:** client → gateway → (router) → LLM/RAG/tools → post-process/validators → response.
- **Azure mapping:** API Management + App Insights + Key Vault + AKS/managed endpoint + storage + vector search.
- **Quality gates:** eval suite, canary, rollback; monitor TTFT/p95 + quality (faithfulness/safety).
- **Safety:** prompt injection defenses, allow-listed tools, HITL for high-risk actions.
- **RAG note:** chunking + hybrid search + reranking + citations; store metadata for audit.
- **Regulated note:** audit logs, explainability, refusal modes; prefer ‘assist’ not ‘decide’.

---

# Q31: Dynamic pricing engine.
- **Clarify (30s):** users, volume, latency, languages, privacy/PII, citations, failure tolerance.
- **Core architecture:** client → gateway → (router) → LLM/RAG/tools → post-process/validators → response.
- **Azure mapping:** API Management + App Insights + Key Vault + AKS/managed endpoint + storage + vector search.
- **Quality gates:** eval suite, canary, rollback; monitor TTFT/p95 + quality (faithfulness/safety).
- **Safety:** prompt injection defenses, allow-listed tools, HITL for high-risk actions.
- **Decisioning note:** offline training + online serving; guardrails for feedback loops and abuse.

---

# Q32: Resume screening at 100K/week.
- **Clarify (30s):** users, volume, latency, languages, privacy/PII, citations, failure tolerance.
- **Core architecture:** client → gateway → (router) → LLM/RAG/tools → post-process/validators → response.
- **Azure mapping:** API Management + App Insights + Key Vault + AKS/managed endpoint + storage + vector search.
- **Quality gates:** eval suite, canary, rollback; monitor TTFT/p95 + quality (faithfulness/safety).
- **Safety:** prompt injection defenses, allow-listed tools, HITL for high-risk actions.

---

# Q33: Voice assistant architecture.
- **Clarify (30s):** users, volume, latency, languages, privacy/PII, citations, failure tolerance.
- **Core architecture:** client → gateway → (router) → LLM/RAG/tools → post-process/validators → response.
- **Azure mapping:** API Management + App Insights + Key Vault + AKS/managed endpoint + storage + vector search.
- **Quality gates:** eval suite, canary, rollback; monitor TTFT/p95 + quality (faithfulness/safety).
- **Safety:** prompt injection defenses, allow-listed tools, HITL for high-risk actions.

---

# Q34: Multi-agent workflow collaboration system.
- **Clarify (30s):** users, volume, latency, languages, privacy/PII, citations, failure tolerance.
- **Core architecture:** client → gateway → (router) → LLM/RAG/tools → post-process/validators → response.
- **Azure mapping:** API Management + App Insights + Key Vault + AKS/managed endpoint + storage + vector search.
- **Quality gates:** eval suite, canary, rollback; monitor TTFT/p95 + quality (faithfulness/safety).
- **Safety:** prompt injection defenses, allow-listed tools, HITL for high-risk actions.

---

# Q35: Real-time transcription for many streams.
- **Clarify (30s):** users, volume, latency, languages, privacy/PII, citations, failure tolerance.
- **Core architecture:** client → gateway → (router) → LLM/RAG/tools → post-process/validators → response.
- **Azure mapping:** API Management + App Insights + Key Vault + AKS/managed endpoint + storage + vector search.
- **Quality gates:** eval suite, canary, rollback; monitor TTFT/p95 + quality (faithfulness/safety).
- **Safety:** prompt injection defenses, allow-listed tools, HITL for high-risk actions.

---

# Q36: Live streaming content moderation.
- **Clarify (30s):** users, volume, latency, languages, privacy/PII, citations, failure tolerance.
- **Core architecture:** client → gateway → (router) → LLM/RAG/tools → post-process/validators → response.
- **Azure mapping:** API Management + App Insights + Key Vault + AKS/managed endpoint + storage + vector search.
- **Quality gates:** eval suite, canary, rollback; monitor TTFT/p95 + quality (faithfulness/safety).
- **Safety:** prompt injection defenses, allow-listed tools, HITL for high-risk actions.

---

