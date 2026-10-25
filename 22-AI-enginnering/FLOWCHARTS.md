---
module: AI Engineering
topic: Flowcharts
subtopic: ""
status: unread
tags: [ai-engineering, flowcharts, decision-trees, debugging, mermaid]
---
# AI Engineering — Flowcharts

Decision and debugging flows. Each links to the lesson with the detail. Numbers and defaults are in the [cheatsheet](CHEATSHEET.md).

## 1. Choosing a Technique (prompt → RAG → tools → fine-tune)
```mermaid
flowchart TD
    S[New requirement] --> E{Eval set exists?}
    E -->|no| E1[Build 50–100 examples<br/>+ success criteria] --> P
    E -->|yes| P[Prompt engineering<br/>clear instructions · few-shot · structured output]
    P --> Q{Passes evals?}
    Q -->|yes| SHIP[Ship + monitor]
    Q -->|no| W{Why does it fail?}
    W -->|missing / private / fresh knowledge| RAG[RAG]
    W -->|needs live data or actions| TOOLS[Tool use]
    W -->|multi-step task| WF{Steps known in advance?}
    WF -->|yes| FLOW[Workflow: chain · router · parallel]
    WF -->|no| AGENT[Agent loop with budgets]
    W -->|format / style / behaviour inconsistent| FT[Fine-tune: SFT → DPO]
    W -->|works but too slow or costly| OPT[Route · cache · distill · smaller model]
    RAG & TOOLS & FLOW & AGENT & FT & OPT --> Q
```
See [what is AI engineering](01-introduction/01-what-is-ai-engineering.md), [when to fine-tune](17-fine-tuning/01-when-to-fine-tune.md).

## 2. Workflow or Agent?
```mermaid
flowchart TD
    A[Task] --> B{Can you write the steps<br/>as a fixed graph?}
    B -->|yes| C{Steps independent?}
    C -->|yes| PAR[Parallelization / map-reduce]
    C -->|no| D{Input types differ?}
    D -->|yes| RT[Router → specialised chains]
    D -->|no| CH[Prompt chain with gates]
    B -->|no| F{Errors cheap and reversible?}
    F -->|yes| AG[Autonomous agent<br/>step + token budget]
    F -->|no| HAG[Agent + human approval<br/>on risky actions]
    AG & HAG --> M{One agent overloaded?<br/>> ~15 tools or distinct skills}
    M -->|yes| MA[Orchestrator–workers / multi-agent]
    M -->|no| SA[Single agent]
```
See [building AI workflows](10-agents/08-building-ai-workflows.md), [multi-agent systems](10-agents/03-multi-agent-systems.md).

## 3. Debugging a RAG System
```mermaid
flowchart TD
    BAD[Bad answer] --> R{Is the needed chunk<br/>in the top-k retrieved?}
    R -->|no| IDX{Is it in the index at all?}
    IDX -->|no| ING[Fix ingestion: parsing · OCR ·<br/>chunking · sync · ACL metadata]
    IDX -->|yes| RET[Fix retrieval: hybrid BM25 ·<br/>query rewrite · HyDE · larger k ·<br/>better embeddings · metadata filters]
    R -->|yes| RANK{Is it in the final context<br/>after rerank/truncation?}
    RANK -->|no| RR[Add / tune reranker ·<br/>context budget · dedupe]
    RANK -->|yes| GEN{Does the answer use it correctly?}
    GEN -->|ignores it| PR[Prompt: answer only from sources ·<br/>order chunks · fewer chunks]
    GEN -->|adds unsupported claims| HAL[Citation check · NLI verify ·<br/>abstain when unsupported]
    GEN -->|multi-hop / reasoning wrong| MH[Decompose query · agentic RAG ·<br/>GraphRAG · stronger model]
    GEN -->|yes, answer is right| EV[Eval label is wrong → fix golden set]
```
See [evaluating RAG](07-rag/05-evaluating-rag-systems.md), [advanced retrieval](07-rag/03-advanced-retrieval-techniques.md).

## 4. Model Selection and Routing
```mermaid
flowchart TD
    T[Task] --> D{Data may leave your boundary?}
    D -->|no| SH[Self-host open-weights<br/>or private deployment]
    D -->|yes| L{Latency budget}
    L -->|< 300 ms| SM[Small model / self-hosted / distilled]
    L -->|seconds| C{Task difficulty on your evals}
    C -->|small model passes| SMALL[Small / cheap tier]
    C -->|only frontier passes| FR[Frontier model]
    C -->|mixed traffic| ROUTE[Router: classifier or cascade<br/>cheap first, escalate on low confidence]
    FR --> V{Volume high and task stable?}
    V -->|yes| DIST[Distill / fine-tune smaller model<br/>on frontier outputs]
    V -->|no| KEEP[Keep frontier + caching]
```
See [model selection and routing](13-llm-optimization/03-model-selection-and-routing.md), [self-hosting](13-llm-optimization/05-self-hosting-models.md).

## 5. Fine-Tuning Decision and Method
```mermaid
flowchart TD
    A[Prompting + RAG plateaued on evals] --> B{Failure is knowledge?}
    B -->|yes| RAG[Fix RAG — fine-tuning is a poor<br/>way to add facts]
    B -->|no: format · style · skill · cost| C{≥ ~500 quality examples?}
    C -->|no| GEN[Collect from prod / synthesize<br/>with teacher + filter]
    GEN --> C
    C -->|yes| D{Goal}
    D -->|imitate outputs| SFT[SFT with LoRA]
    D -->|prefer A over B| DPO[DPO on preference pairs]
    D -->|verifiable correctness| RL[RL with verifier rewards]
    D -->|cheaper copy of a working prompt| DI[Distillation]
    SFT & DPO & RL & DI --> GPU{GPU memory tight?}
    GPU -->|yes| QL[QLoRA 4-bit base]
    GPU -->|no| LO[LoRA / full FT]
    QL & LO --> EV[Compare vs prompted baseline<br/>+ check regressions on general evals]
```
See [LoRA/QLoRA](17-fine-tuning/03-lora-qlora-and-peft.md), [preference tuning](17-fine-tuning/04-preference-tuning-dpo-and-rl.md).

## 6. Latency Debugging
```mermaid
flowchart TD
    SLOW[p95 latency too high] --> TR[Look at trace spans]
    TR --> W{Where is the time?}
    W -->|before first token| TTFT{Long prompt?}
    TTFT -->|yes| PC[Prompt caching · trim context ·<br/>fewer retrieved chunks]
    TTFT -->|no| QUEUE[Provider queueing → other region /<br/>provisioned throughput / fallback]
    W -->|token generation| GENT[Shorter outputs · smaller model ·<br/>speculative decoding · quantization]
    W -->|retrieval / tools| RT[Parallelize · cache · index tuning ·<br/>tool timeouts · prefetch]
    W -->|many sequential LLM calls| SEQ[Merge steps · parallelize ·<br/>small model for routing steps]
    W -->|client / network| NET[Stream to UI · colocate · keep-alive]
```
See [latency optimization](13-llm-optimization/04-latency-optimization.md).

## 7. Production Incident Triage
```mermaid
flowchart TD
    AL[Alert or user reports] --> SEV{User harm, data leak,<br/>or unsafe action?}
    SEV -->|yes| KILL[Kill switch / disable feature ·<br/>rotate creds · preserve traces]
    SEV -->|no| K{What changed?}
    K -->|our deploy: prompt · model · code| RB[Roll back via flag]
    K -->|provider: outage · latency · model update| FB[Fallback provider/model ·<br/>pin versions]
    K -->|traffic: spike · new input type · abuse| TL[Rate limit · scale ·<br/>route unknowns to safe path]
    K -->|data: index stale · upstream source changed| DATA[Re-ingest · fix connector]
    K -->|nothing obvious| DRIFT[Sample traces → error analysis ·<br/>compare to eval distribution]
    KILL & RB & FB & TL & DATA & DRIFT --> PM[Postmortem → add failing cases<br/>to eval set + new alert]
```
See [handling production incidents](12-production-deployment/04-handling-production-incidents.md).

## 8. The Eval-Driven Development Loop
```mermaid
flowchart LR
    SPEC[Define success<br/>criteria] --> DS[Eval dataset<br/>golden + edge + adversarial]
    DS --> BASE[Baseline run]
    BASE --> CHG[Change prompt /<br/>retrieval / model]
    CHG --> RUN[Run evals<br/>code checks + judge]
    RUN --> G{Better and no<br/>regressions?}
    G -->|no| ERR[Error analysis<br/>cluster failures] --> CHG
    G -->|yes| SHIP[Ship behind flag<br/>→ online A/B]
    SHIP --> MON[Monitor + collect<br/>failures & feedback]
    MON --> DS
```
See [eval-driven development](16-evaluation/01-eval-driven-development.md), [error analysis](16-evaluation/02-building-eval-datasets-and-error-analysis.md).

## 9. Guardrail Placement
```mermaid
flowchart LR
    U[User input] --> IR[Input rails<br/>injection · PII · topic · abuse]
    IR --> RET[Retrieval]
    RET --> RR[Retrieval rails<br/>ACL filter · injection scan]
    RR --> LLM[LLM]
    LLM --> TC{Tool call?}
    TC -->|yes| TR[Tool rails<br/>authz in code · limits · approval] --> LLM
    TC -->|no| OR[Output rails<br/>PII · policy · grounding · schema]
    OR --> OUT[Response]
```
See [guardrails architecture](18-safety-and-governance/01-guardrails-architecture.md).

## 10. Handling an LLM API Call Robustly
```mermaid
flowchart TD
    CALL[Call model] --> RES{Response}
    RES -->|429 / 5xx / timeout| RETRY{Retries left?}
    RETRY -->|yes| BO[Exponential backoff + jitter] --> CALL
    RETRY -->|no| FBK[Fallback model/provider] --> CALL2[Call fallback]
    CALL2 -->|fails| DEG[Degrade: cached answer ·<br/>simpler path · honest error]
    RES -->|200| VAL{Valid schema & passes rails?}
    VAL -->|yes| OK[Return]
    VAL -->|no, retries left| REP[Re-ask with validation error] --> CALL
    VAL -->|no, exhausted| DEG
```
See [designing for reliability](11-ai-system-architecture/02-designing-for-reliability.md), [AI gateway](11-ai-system-architecture/05-the-ai-gateway-pattern.md).

## 11. AI System Design Interview Flow
```mermaid
flowchart LR
    A[Clarify: users · task ·<br/>quality bar · constraints] --> B[Estimate: QPS ·<br/>tokens · cost · latency]
    B --> C[High-level design:<br/>gateway · retrieval · model · tools]
    C --> D[Deep dive: the hardest part<br/>ACLs · latency · agent safety]
    D --> E[Evaluation: offline ·<br/>online · launch gates]
    E --> F[Failure modes &<br/>guardrails]
    F --> G[Cost, scale,<br/>iteration plan]
```
See [interview framework](19-ai-system-design-cases/01-ai-system-design-interview-framework.md).

## Related
- [Index](README.md) · [Cheatsheet](CHEATSHEET.md) · [Revision](REVISION.md) · [Glossary](GLOSSARY.md) · Repo: [FLOWCHARTS](../FLOWCHARTS.md)
