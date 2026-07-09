# PART 9: AGENT DECISION FRAMEWORK

## Goal
To teach candidates when to build agents vs. simpler systems, and how to architect reliable, safe, production-grade agentic systems.

## Mental Model
**An agent is a loop: Think → Act → Observe → Repeat.**
Do NOT build an agent when a deterministic pipeline will do. Agents add nondeterminism, cost, and debugging complexity. Build the simplest thing that works.

---

## 9.1 Agent vs Pipeline Decision Tree

```text
Is the task sequence FIXED and known in advance?
├── YES → Use a deterministic pipeline (DAG), not an agent.
│   └── Example: ETL pipeline, batch scoring job.
└── NO (Requires dynamic decision-making mid-task) →
    ├── Does it need to USE TOOLS based on context? → Single Agent.
    ├── Does it require MULTIPLE ROLES or SPECIALIZED EXPERTISE? → Multi-Agent.
    └── Does it require HUMAN APPROVAL at key steps? → Human-in-the-Loop Agent.
```

---

## 9.2 Single vs. Multi-Agent Decision

```text
How complex is the task?
├── ONE well-defined goal + limited tools (< 5) → Single Agent.
├── MULTIPLE sub-tasks with different domains → Multi-Agent.
│   └── Example: Research Agent + Code Agent + QA Agent.
├── Tasks that need PARALLELISM → Parallel Multi-Agent (fan-out).
└── Tasks needing PEER REVIEW → Multi-Agent with Critic.
```

### Multi-Agent Architectures

| Pattern | Description | Use Case |
| :--- | :--- | :--- |
| **Supervisor** | One orchestrator delegates to specialist sub-agents. | General-purpose coding assistant |
| **Swarm** | Agents hand off to each other based on context. | Customer support routing |
| **Parallel** | Multiple agents run simultaneously, results merged. | Research synthesis |
| **Hierarchical** | Nested supervisor → sub-supervisors → workers. | Complex enterprise workflows |

---

## 9.3 Tool Design Framework

### Tool Design Principles
1. **One tool = one responsibility.** Do not build a mega-tool that does 10 things.
2. **Tools must be idempotent where possible.** Calling a tool twice should be safe.
3. **Tools must return structured outputs.** JSON, not free text. The agent parses it.
4. **Tools must have timeouts.** Never let a tool block the agent indefinitely.
5. **Tools must have error messages.** Return descriptive errors so the agent can self-correct.

### Tool Calling vs Function Calling
- **Function Calling (OpenAI):** Structured JSON schema defines available functions. Model selects and populates arguments. Deterministic argument format.
- **Tool Calling (Anthropic/LangGraph):** Same concept, different API naming. Treat as equivalent.
- **MCP (Model Context Protocol):** An open protocol that standardizes how models connect to tools and external context sources. Use MCP when you need interoperability across different model providers and tool ecosystems.

---

## 9.4 Memory Architecture

```text
What type of information needs to be remembered?
├── IN-CONTEXT (single conversation) → In-context window (no external store).
├── WITHIN-SESSION (multi-turn chat) → Conversation history buffer (last N turns).
├── ACROSS-SESSIONS (user preferences, history) → External memory store.
│   ├── Structured facts → Relational DB / Redis.
│   ├── Semantic memory (past conversations) → Vector DB (retrieve by similarity).
│   └── Episodic memory (specific past events) → Structured log + retrieval.
└── LONG-TERM KNOWLEDGE → RAG (not agent memory, treat as external KB).
```

### Memory Types
| Type | Storage | Retrieval | Best For |
| :--- | :--- | :--- | :--- |
| **In-Context** | Prompt | Instant | Short single sessions |
| **Summary Buffer** | Summarize old turns | In-context | Long conversations |
| **Semantic (Vector)** | Vector DB | Similarity search | User history, past tasks |
| **Key-Value** | Redis | Exact key lookup | User preferences, state |

---

## 9.5 Reflection & Self-Correction

### Framework
```text
Reflection Loop:
[Agent Output] → [Critic/Evaluator] → [Is output correct?]
                                       ├── YES → Return output.
                                       └── NO → Feedback → Agent revises → Repeat.
```

- **Inline Reflection:** The same LLM checks its own output ("Step back and verify your reasoning").
- **External Critic:** A separate LLM (possibly a stronger model) reviews the output.
- **Rule-based Validator:** For structured outputs (JSON, code), use a deterministic validator. Always prefer deterministic validators over LLM-based ones where possible.

### When NOT to Use Reflection
- Cost-sensitive applications (reflection doubles token usage).
- Simple tasks where the output can be validated deterministically (run the code, check if it passes tests).

---

## 9.6 Human-in-the-Loop (HITL)

### When to Add HITL
- **Irreversible actions:** Deleting records, sending emails, executing payments.
- **Low-confidence decisions:** When the agent's confidence score falls below a threshold.
- **Compliance-required tasks:** Legal, medical, financial approvals.
- **Novel situations:** When the agent encounters a scenario outside its training distribution.

### HITL Patterns
```text
[Agent proposes action]
       │
       ▼
[Is action IRREVERSIBLE?] ── YES ──> [Human review required before executing]
       │
       NO
       ▼
[Is confidence LOW?] ────── YES ──> [Show to human for confirmation]
       │
       NO
       ▼
[Execute autonomously]
```

---

## 9.7 Retries & Error Handling

### Framework
```text
Tool call failed?
├── TRANSIENT ERROR (timeout, rate limit) → Retry with exponential backoff.
│   └── Max 3 retries. Log each retry.
├── INVALID ARGUMENTS → Agent self-corrects arguments using error message.
│   └── Max 2 self-correction attempts.
├── TOOL UNAVAILABLE → Gracefully skip tool, attempt alternative path.
└── UNRECOVERABLE → Escalate to HITL or return partial result with error flag.
```

### Retry Best Practices
- Add jitter to exponential backoff to prevent thundering herd.
- Set a maximum total timeout per agent run (e.g., 120 seconds).
- Log the exact error message and tool arguments on every failure.

---

## 9.8 Agent Security Framework

### Attack Vectors
| Threat | Description | Mitigation |
| :--- | :--- | :--- |
| **Prompt Injection** | Malicious user input hijacks agent instructions. | Separate system prompt from user data. Validate tool outputs before re-injecting into context. |
| **Tool Abuse** | Agent calls a destructive tool (delete DB) unexpectedly. | Allowlist tools per agent role. Require HITL for destructive operations. |
| **Privilege Escalation** | Agent acquires permissions beyond its role. | Principle of least privilege for all tool permissions. |
| **Data Exfiltration** | Agent leaks sensitive data via tool calls. | Scan tool outputs for PII before passing to next step. |

### Security Checklist for Agents
- [ ] Is each tool's permission scope minimal (read-only unless write is explicitly needed)?
- [ ] Is user-provided content separated from agent instructions in the prompt?
- [ ] Are all external tool outputs validated before being re-injected into the context?
- [ ] Is there a maximum cost/token budget per agent run?
- [ ] Is there an audit log of every tool call and its arguments?

---

## 9.9 LangGraph vs Custom Agent Architecture

| Approach | Pros | Cons | Use When |
| :--- | :--- | :--- | :--- |
| **LangGraph** | Visual graph, state management, built-in HITL, streaming. | Framework dependency, learning curve. | Complex multi-step workflows with branching logic. |
| **OpenAI Assistants API** | Managed threads, file retrieval, code interpreter. | Vendor lock-in, limited customization. | Quick prototyping with OpenAI models. |
| **Custom (bare LLM)** | Full control, minimal dependencies. | Need to build everything yourself. | Simple single-agent, cost-critical, unique requirements. |
| **CrewAI** | Easy multi-agent role definition. | Less production-hardened. | Rapid multi-agent prototyping. |

---

## Engineering Checklist

- [ ] Have I verified an agent is necessary (vs. a deterministic pipeline)?
- [ ] Have I defined a maximum step/token budget per run?
- [ ] Have I designed tools with structured outputs and descriptive errors?
- [ ] Have I added HITL gates for all irreversible actions?
- [ ] Is there an audit log of every tool invocation and result?
- [ ] Have I tested adversarial inputs (prompt injection attempts)?
- [ ] Is there a graceful degradation path if the agent fails?

## Production Considerations

- **Cost Monitoring:** Track token usage per agent run. Set hard limits to prevent runaway agents from generating $1000 API bills.
- **Observability:** Trace every step of every agent run (LangSmith, Phoenix Arize). This is non-negotiable in production.
- **Idempotency:** Ensure that re-running a failed agent task doesn't result in duplicate actions (e.g., sending the same email twice).

## Interview Follow-up Questions & Best Answers

**Q: "Your agent for a game support bot keeps calling the wrong tools and going in circles. How do you debug it?"**
*Best Answer:* "I trace the exact sequence of thoughts, tool calls, and observations using a full observability framework like LangSmith. Circular behavior usually has three causes: 
1. **Tool descriptions are ambiguous** — the agent can't distinguish between tools and picks randomly. Fix: rewrite tool descriptions with explicit examples of when to use each tool. 
2. **The task is too complex** — the agent gets lost in a long horizon. Fix: decompose into sub-tasks with a supervisor.
3. **The termination condition is unclear** — the agent doesn't know when it's done. Fix: add an explicit 'task_complete' tool and define it clearly in the system prompt."
