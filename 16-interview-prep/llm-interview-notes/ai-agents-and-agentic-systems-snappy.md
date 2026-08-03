---
module: LLMs
topic: Interview Notes
subtopic: Ai Agents And Agentic Systems Snappy
status: unread
tags: [llms, ml, interview-notes-ai-agents-and-]
---

> _Quick-recall companion. For the full deep-dive, see [ai-agents-and-agentic-systems.md](04-ai-agents-and-agentic-systems.md)._

# AI agents & agentic systems — workflow orchestration (Azure/DevOps-fluent)

Agents are what happens when you stop asking for a single answer and start running a **loop**: plan → call tools → verify → repeat.

**One-line mental model:** Agent = LLM + tools + memory + control loop + guardrails.

---

# Q1: What is an AI agent, and how does it differ from a simple LLM call?
- **Direct answer:** An agent runs a **closed-loop** process (think-act-observe) with tools and state; a simple LLM call is one-shot text generation.
- **DevOps bridge:** LLM call = a single script. Agent = a pipeline/job runner with retries, state, and external steps.

---

# Q2: Explain the ReAct (Reasoning + Acting) agent architecture.
- **Direct answer:** ReAct interleaves reasoning with tool calls: reason → act → observe → reason.
- **Why it works:** tools provide ground truth; reasoning stitches tool outputs into a plan.

---

# Q3: What is the Plan-and-Execute agent pattern?
- **Direct answer:** First create a plan (task graph), then execute steps with tool calls.
- **MI analogy:** match plan first, then over-by-over adjustments.

---

# Q4: What is tool use (function calling) in LLMs?
- **Direct answer:** The model outputs a structured tool invocation (name + args) instead of free text.
- **DevOps bridge:** typed APIs beat “stringly-typed” bash.

---

# Q5: How do you design tools for an agent?
- **Direct answer:** Small, explicit, least-privilege functions with clear schemas.
- **Checklist:** allow-lists, timeouts, idempotency, dry-run support, audit logs.

---

# Q6: Single-agent vs multi-agent systems?
- **Direct answer:** Single-agent = one controller; multi-agent = specialists that coordinate.
- **Trade-off:** multi-agent adds coordination cost and failure modes.

---

# Q7: What is Model Context Protocol (MCP)?
- **Direct answer:** A standard for connecting models to tools/resources in a consistent, discoverable way.
- **DevOps bridge:** like a common interface spec so integrations don’t become bespoke glue.

---

# Q8: Types of agent memory (short-term, long-term, episodic)?
- **Short-term:** recent messages + working state.
- **Long-term:** stored knowledge (vector DB / DB).
- **Episodic:** “what happened last time,” traces, lessons.

---

# Q9: Handling failures and error recovery?
- **Direct answer:** Treat tool calls like unreliable dependencies.
- **Patterns:** retries with backoff, circuit breakers, fallbacks, escalation.

---

# Q10: What is an agent loop? How does it decide when to stop?
- **Direct answer:** A loop that runs until a stop condition: goal met, budget hit, max steps hit, or human approval required.

---

# Q11: How do you evaluate and test agents?
- **Direct answer:** scenario suites + tool-mocking + regression tests + safety tests.
- **DevOps bridge:** evals are your CI gates.

---

# Q12: Security risks of agentic systems? Mitigations?
- **Risks:** prompt injection, data exfiltration, tool abuse, privilege escalation.
- **Mitigations:** sandboxing, least privilege, allow-lists, secrets isolation, human approvals.

---

# Q13: Reactive vs proactive agents?
- **Reactive:** respond to requests.
- **Proactive:** watch signals, trigger actions.
- **Caution:** proactive agents need stronger guardrails to avoid “automation surprise.”

---

# Q14: Token consumption and cost in long workflows?
- **Fixes:** summarize state, store memory externally, retrieve selectively, compress tool outputs.

---

# Q15: Human-in-the-loop (HITL) pattern?
- **Direct answer:** require explicit approval for risky actions or low-confidence decisions.
- **DevOps bridge:** approval gates before prod deploy.

---

# Q16: Guardrails to prevent harmful actions?
- **Controls:** allow-list tools, parameter validation, policy layer, rate limits, irreversible-action blocks.

---

# Q17: Agent reflection?
- **Direct answer:** post-step critique + improvement (“did I follow rules?” “did I cite evidence?”).
- **Trade-off:** more tokens/latency.

---

# Q18: Code-generating vs tool-calling agents?
- **Code-gen:** writes code to do work; riskier, harder to sandbox.
- **Tool-calling:** uses predefined functions; safer and more auditable.

---

# Q19: Multi-modal inputs/outputs?
- **Direct answer:** support images/audio/etc via specialized models + tool routing; store artifacts with metadata.

---

# Q20: State management in complex workflows?
- **Direct answer:** explicit state machine / DAG; persist state; make steps idempotent.
- **DevOps bridge:** workflow engines > ad-hoc loops.

---

# Q21: Customer support agent with escalation logic?
- **Pattern:** triage → retrieve policy → answer → if uncertain/risky → escalate to human.
- **Mini prompt:** what’s the escalation signal? → low confidence, policy conflict, PII, refunds.

---

# Q22: Agent orchestration with LangGraph?
- **Direct answer:** model the agent as a graph of nodes (tools/LLM steps) with edges (conditions).

---

# Q23: Code execution agent safely?
- **Direct answer:** sandbox (container/VM), no network by default, CPU/mem limits, timeouts, filesystem jail.

---

# Q24: Agent stuck in an infinite loop — detect & break?
- **Controls:** max steps, repeated-state detection, “no progress” heuristic, budget enforcement.

---

# Q25: Conflicting answers from tools — reconcile?
- **Pattern:** prefer authoritative source, compare timestamps, ask follow-up queries, cite and surface uncertainty.

---

# Q26: Too many tokens per task — reduce consumption?
- **Fixes:** shorter prompts, summarization, smaller top-k retrieval, structured outputs, avoid verbose reflection.

---

# Q27: Enforce budget limits?
- **Direct answer:** hard caps on tokens, tool calls, wall-clock time; stop with partial result + next steps.

---

# Q28: Hallucinates tool capabilities / wrong inputs — fix?
- **Fixes:** tool schema in prompt, constrained decoding, examples, tool-result validation.

---

# Q29: Deleted a production DB — prevent irreversible actions?
- **Hard rule:** no destructive ops without human approval.
- **Tech:** read-only creds by default, “dry run,” multi-party approval, blast-radius limits.

---

# Q30: Many tools but picks wrong one — improve selection?
- **Fixes:** tool descriptions, routing rules, tool-choice evals, reduce tool set, hierarchical tool menus.

---

# Q31: Agent takes too long — speed it up?
- **Fixes:** parallelize retrieval/tool calls, caching, reduce model size for routing, cut steps.

---

# Q32: Right tool but wrong parameters — fix extraction?
- **Fixes:** schema-first design, structured outputs, validation + repair, canonicalization of inputs.
