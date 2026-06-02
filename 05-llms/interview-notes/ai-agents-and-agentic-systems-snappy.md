---
module: Llms
topic: Interview Notes
subtopic: Ai Agents And Agentic Systems Snappy
status: unread
tags: [llms, ml, interview-notes-ai-agents-and-]
---
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

## Rapid Recall

### Direct answer
- Direct Answer: An agent runs a closed-loop process (think-act-observe) with tools and state; a simple LLM call is one-shot text generation.
- Why: This matters because it tells you how to reason about direct answer.
- Pitfall: Don't answer "Direct answer" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: An agent runs a closed-loop process (think-act-observe) with tools and state; a simple LLM call is one-shot text generation.

### DevOps bridge
- Direct Answer: LLM call = a single script. Agent = a pipeline/job runner with retries, state, and external steps.
- Why: This matters because it tells you how to reason about devops bridge.
- Pitfall: Don't answer "DevOps bridge" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: LLM call = a single script. Agent = a pipeline/job runner with retries, state, and external steps.

### Direct answer
- Direct Answer: ReAct interleaves reasoning with tool calls: reason → act → observe → reason.
- Why: This matters because it tells you how to reason about direct answer.
- Pitfall: Don't answer "Direct answer" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: ReAct interleaves reasoning with tool calls: reason → act → observe → reason.

### Why it works
- Direct Answer: tools provide ground truth; reasoning stitches tool outputs into a plan.
- Why: This matters because it tells you how to reason about why it works.
- Pitfall: Don't answer "Why it works" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: tools provide ground truth; reasoning stitches tool outputs into a plan.

### Direct answer
- Direct Answer: First create a plan (task graph), then execute steps with tool calls.
- Why: This matters because it tells you how to reason about direct answer.
- Pitfall: Don't answer "Direct answer" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: First create a plan (task graph), then execute steps with tool calls.

### MI analogy
- Direct Answer: match plan first, then over-by-over adjustments.
- Why: This matters because it tells you how to reason about mi analogy.
- Pitfall: Don't answer "MI analogy" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: match plan first, then over-by-over adjustments.

### Direct answer
- Direct Answer: The model outputs a structured tool invocation (name + args) instead of free text.
- Why: This matters because it tells you how to reason about direct answer.
- Pitfall: Don't answer "Direct answer" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: The model outputs a structured tool invocation (name + args) instead of free text.

### DevOps bridge
- Direct Answer: typed APIs beat “stringly-typed” bash.
- Why: This matters because it tells you how to reason about devops bridge.
- Pitfall: Don't answer "DevOps bridge" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: typed APIs beat “stringly-typed” bash.

### Direct answer
- Direct Answer: Small, explicit, least-privilege functions with clear schemas.
- Why: This matters because it tells you how to reason about direct answer.
- Pitfall: Don't answer "Direct answer" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Small, explicit, least-privilege functions with clear schemas.

### Checklist
- Direct Answer: allow-lists, timeouts, idempotency, dry-run support, audit logs.
- Why: This matters because it tells you how to reason about checklist.
- Pitfall: Don't answer "Checklist" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: allow-lists, timeouts, idempotency, dry-run support, audit logs.

### Direct answer
- Direct Answer: Single-agent = one controller; multi-agent = specialists that coordinate.
- Why: This matters because it tells you how to reason about direct answer.
- Pitfall: Don't answer "Direct answer" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Single-agent = one controller; multi-agent = specialists that coordinate.

### Trade-off
- Direct Answer: multi-agent adds coordination cost and failure modes.
- Why: This matters because it tells you how to reason about trade-off.
- Pitfall: Don't answer "Trade-off" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: multi-agent adds coordination cost and failure modes.

### Direct answer
- Direct Answer: A standard for connecting models to tools/resources in a consistent, discoverable way.
- Why: This matters because it tells you how to reason about direct answer.
- Pitfall: Don't answer "Direct answer" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: A standard for connecting models to tools/resources in a consistent, discoverable way.

### DevOps bridge
- Direct Answer: like a common interface spec so integrations don’t become bespoke glue.
- Why: This matters because it tells you how to reason about devops bridge.
- Pitfall: Don't answer "DevOps bridge" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: like a common interface spec so integrations don’t become bespoke glue.

### Short-term
- Direct Answer: recent messages + working state.
- Why: This matters because it tells you how to reason about short-term.
- Pitfall: Don't answer "Short-term" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: recent messages + working state.

### Long-term
- Direct Answer: stored knowledge (vector DB / DB).
- Why: This matters because it tells you how to reason about long-term.
- Pitfall: Don't answer "Long-term" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: stored knowledge (vector DB / DB).

### Episodic
- Direct Answer: “what happened last time,” traces, lessons.
- Why: This matters because it tells you how to reason about episodic.
- Pitfall: Don't answer "Episodic" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: “what happened last time,” traces, lessons.

### Direct answer
- Direct Answer: Treat tool calls like unreliable dependencies.
- Why: This matters because it tells you how to reason about direct answer.
- Pitfall: Don't answer "Direct answer" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Treat tool calls like unreliable dependencies.

### Patterns
- Direct Answer: retries with backoff, circuit breakers, fallbacks, escalation.
- Why: This matters because it tells you how to reason about patterns.
- Pitfall: Don't answer "Patterns" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: retries with backoff, circuit breakers, fallbacks, escalation.

### Direct answer
- Direct Answer: A loop that runs until a stop condition: goal met, budget hit, max steps hit, or human approval required.
- Why: This matters because it tells you how to reason about direct answer.
- Pitfall: Don't answer "Direct answer" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: A loop that runs until a stop condition: goal met, budget hit, max steps hit, or human approval required.

### Direct answer
- Direct Answer: scenario suites + tool-mocking + regression tests + safety tests.
- Why: This matters because it tells you how to reason about direct answer.
- Pitfall: Don't answer "Direct answer" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: scenario suites + tool-mocking + regression tests + safety tests.

### DevOps bridge
- Direct Answer: evals are your CI gates.
- Why: This matters because it tells you how to reason about devops bridge.
- Pitfall: Don't answer "DevOps bridge" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: evals are your CI gates.

### Risks
- Direct Answer: prompt injection, data exfiltration, tool abuse, privilege escalation.
- Why: This matters because it tells you how to reason about risks.
- Pitfall: Don't answer "Risks" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: prompt injection, data exfiltration, tool abuse, privilege escalation.

### Mitigations
- Direct Answer: sandboxing, least privilege, allow-lists, secrets isolation, human approvals.
- Why: This matters because it tells you how to reason about mitigations.
- Pitfall: Don't answer "Mitigations" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: sandboxing, least privilege, allow-lists, secrets isolation, human approvals.

### Reactive
- Direct Answer: respond to requests.
- Why: This matters because it tells you how to reason about reactive.
- Pitfall: Don't answer "Reactive" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: respond to requests.

### Proactive
- Direct Answer: watch signals, trigger actions.
- Why: This matters because it tells you how to reason about proactive.
- Pitfall: Don't answer "Proactive" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: watch signals, trigger actions.

### Caution
- Direct Answer: proactive agents need stronger guardrails to avoid “automation surprise.”
- Why: This matters because it tells you how to reason about caution.
- Pitfall: Don't answer "Caution" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: proactive agents need stronger guardrails to avoid “automation surprise.”

### Fixes
- Direct Answer: summarize state, store memory externally, retrieve selectively, compress tool outputs.
- Why: This matters because it tells you how to reason about fixes.
- Pitfall: Don't answer "Fixes" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: summarize state, store memory externally, retrieve selectively, compress tool outputs.

### Direct answer
- Direct Answer: require explicit approval for risky actions or low-confidence decisions.
- Why: This matters because it tells you how to reason about direct answer.
- Pitfall: Don't answer "Direct answer" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: require explicit approval for risky actions or low-confidence decisions.

### DevOps bridge
- Direct Answer: approval gates before prod deploy.
- Why: This matters because it tells you how to reason about devops bridge.
- Pitfall: Don't answer "DevOps bridge" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: approval gates before prod deploy.

### Controls
- Direct Answer: allow-list tools, parameter validation, policy layer, rate limits, irreversible-action blocks.
- Why: This matters because it tells you how to reason about controls.
- Pitfall: Don't answer "Controls" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: allow-list tools, parameter validation, policy layer, rate limits, irreversible-action blocks.

### Direct answer
- Direct Answer: post-step critique + improvement (“did I follow rules?” “did I cite evidence?”).
- Why: This matters because it tells you how to reason about direct answer.
- Pitfall: Don't answer "Direct answer" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: post-step critique + improvement (“did I follow rules?” “did I cite evidence?”).

### Trade-off
- Direct Answer: more tokens/latency.
- Why: This matters because it tells you how to reason about trade-off.
- Pitfall: Don't answer "Trade-off" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: more tokens/latency.

### Code-gen
- Direct Answer: writes code to do work; riskier, harder to sandbox.
- Why: This matters because it tells you how to reason about code-gen.
- Pitfall: Don't answer "Code-gen" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: writes code to do work; riskier, harder to sandbox.

### Tool-calling
- Direct Answer: uses predefined functions; safer and more auditable.
- Why: This matters because it tells you how to reason about tool-calling.
- Pitfall: Don't answer "Tool-calling" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: uses predefined functions; safer and more auditable.

### Direct answer
- Direct Answer: support images/audio/etc via specialized models + tool routing; store artifacts with metadata.
- Why: This matters because it tells you how to reason about direct answer.
- Pitfall: Don't answer "Direct answer" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: support images/audio/etc via specialized models + tool routing; store artifacts with metadata.

### Direct answer
- Direct Answer: explicit state machine / DAG; persist state; make steps idempotent.
- Why: This matters because it tells you how to reason about direct answer.
- Pitfall: Don't answer "Direct answer" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: explicit state machine / DAG; persist state; make steps idempotent.

### DevOps bridge
- Direct Answer: workflow engines > ad-hoc loops.
- Why: This matters because it tells you how to reason about devops bridge.
- Pitfall: Don't answer "DevOps bridge" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: workflow engines > ad-hoc loops.

### Pattern
- Direct Answer: triage → retrieve policy → answer → if uncertain/risky → escalate to human.
- Why: This matters because it tells you how to reason about pattern.
- Pitfall: Don't answer "Pattern" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: triage → retrieve policy → answer → if uncertain/risky → escalate to human.

### Mini prompt
- Direct Answer: what’s the escalation signal? → low confidence, policy conflict, PII, refunds.
- Why: This matters because it tells you how to reason about mini prompt.
- Pitfall: Don't answer "Mini prompt" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: what’s the escalation signal? → low confidence, policy conflict, PII, refunds.

### Direct answer
- Direct Answer: model the agent as a graph of nodes (tools/LLM steps) with edges (conditions).
- Why: This matters because it tells you how to reason about direct answer.
- Pitfall: Don't answer "Direct answer" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: model the agent as a graph of nodes (tools/LLM steps) with edges (conditions).

### Direct answer
- Direct Answer: sandbox (container/VM), no network by default, CPU/mem limits, timeouts, filesystem jail.
- Why: This matters because it tells you how to reason about direct answer.
- Pitfall: Don't answer "Direct answer" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: sandbox (container/VM), no network by default, CPU/mem limits, timeouts, filesystem jail.

### Controls
- Direct Answer: max steps, repeated-state detection, “no progress” heuristic, budget enforcement.
- Why: This matters because it tells you how to reason about controls.
- Pitfall: Don't answer "Controls" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: max steps, repeated-state detection, “no progress” heuristic, budget enforcement.

### Pattern
- Direct Answer: prefer authoritative source, compare timestamps, ask follow-up queries, cite and surface uncertainty.
- Why: This matters because it tells you how to reason about pattern.
- Pitfall: Don't answer "Pattern" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: prefer authoritative source, compare timestamps, ask follow-up queries, cite and surface uncertainty.

### Fixes
- Direct Answer: shorter prompts, summarization, smaller top-k retrieval, structured outputs, avoid verbose reflection.
- Why: This matters because it tells you how to reason about fixes.
- Pitfall: Don't answer "Fixes" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: shorter prompts, summarization, smaller top-k retrieval, structured outputs, avoid verbose reflection.

### Direct answer
- Direct Answer: hard caps on tokens, tool calls, wall-clock time; stop with partial result + next steps.
- Why: This matters because it tells you how to reason about direct answer.
- Pitfall: Don't answer "Direct answer" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: hard caps on tokens, tool calls, wall-clock time; stop with partial result + next steps.

### Fixes
- Direct Answer: tool schema in prompt, constrained decoding, examples, tool-result validation.
- Why: This matters because it tells you how to reason about fixes.
- Pitfall: Don't answer "Fixes" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: tool schema in prompt, constrained decoding, examples, tool-result validation.

### Hard rule
- Direct Answer: no destructive ops without human approval.
- Why: This matters because it tells you how to reason about hard rule.
- Pitfall: Don't answer "Hard rule" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: no destructive ops without human approval.

### Tech
- Direct Answer: read-only creds by default, “dry run,” multi-party approval, blast-radius limits.
- Why: This matters because it tells you how to reason about tech.
- Pitfall: Don't answer "Tech" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: read-only creds by default, “dry run,” multi-party approval, blast-radius limits.

### Fixes
- Direct Answer: tool descriptions, routing rules, tool-choice evals, reduce tool set, hierarchical tool menus.
- Why: This matters because it tells you how to reason about fixes.
- Pitfall: Don't answer "Fixes" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: tool descriptions, routing rules, tool-choice evals, reduce tool set, hierarchical tool menus.

### Fixes
- Direct Answer: parallelize retrieval/tool calls, caching, reduce model size for routing, cut steps.
- Why: This matters because it tells you how to reason about fixes.
- Pitfall: Don't answer "Fixes" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: parallelize retrieval/tool calls, caching, reduce model size for routing, cut steps.

### Fixes
- Direct Answer: schema-first design, structured outputs, validation + repair, canonicalization of inputs.
- Why: This matters because it tells you how to reason about fixes.
- Pitfall: Don't answer "Fixes" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: schema-first design, structured outputs, validation + repair, canonicalization of inputs.
