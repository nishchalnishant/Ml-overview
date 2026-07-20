---
module: LLMs
topic: Applications
subtopic: Agentic Workflows
status: unread
tags: [llms, ml, applications-agentic-workflows]
---
# Agentic Workflows

---

## Chain-of-Thought (CoT)

**The problem:** LLMs produce a final answer from a single forward pass over the prompt. For multi-step problems, any error in the implicit reasoning is invisible and unrecoverable — the wrong intermediate state propagates directly into the answer, and the model produces a confident wrong output with no way to identify where things broke.

**The core insight:** intermediate steps are both verifiable checkpoints and self-supplied context. When the model writes out a reasoning step, that step enters its own context window, making subsequent tokens conditioned on the (hopefully correct) intermediate result rather than reconstructed from distributed weights alone.

**The mechanics:**
- Zero-shot: append "Let's think step by step." — the model imitates reasoning patterns from training data.
- Few-shot: provide 2–5 worked examples with explicit step-by-step traces. The model imitates the pattern.
- The generated scratchpad becomes part of the model's own context for the final answer.

**What breaks:** the trace can look structurally correct while being factually wrong. The model can hallucinate a coherent step sequence that arrives at a wrong answer confidently. High-confidence wrong reasoning is more dangerous than admitted uncertainty. CoT reduces multi-step error rate; it does not eliminate it.

---

## ReAct (Reasoning + Acting)

**The problem:** CoT keeps reasoning inside the model's parametric memory. Any question requiring current facts, live data, or external computation will either be refused or hallucinated — the model cannot discover facts it does not already know, no matter how well it reasons.

**The core insight:** interleave thoughts with actions. Let the model emit a `Thought:` (what it needs), an `Action:` (a tool call), then an `Observation:` (the tool's actual result) — then reason again from that grounded observation. The model provides only reasoning about what to ask and how to interpret results; factual content comes from the tool.

**The mechanics:**
```
Thought: I need the current AAPL price.
Action: search_web(query="AAPL stock price today")
Observation: AAPL is trading at $189.42 as of 2:30 PM EST.
Thought: Now I can answer.
Answer: AAPL is currently trading at $189.42.
```

**What breaks:** the model can hallucinate tool names or arguments that do not exist. It can loop — each observation spawns a new action indefinitely. Long observation strings consume the context window. Every tool call is a latency and cost hit; chains of 10+ calls are common for complex tasks.

---

## Reflection and Self-Critique

**The problem:** a model's first response is its best guess. That guess may contain errors, but calling the same model with the same prompt will produce the same answer. There is no intrinsic correction mechanism.

**The core insight:** critiquing an existing solution is a lower-difficulty task than producing a perfect solution from scratch. It is easier to notice a flaw in a proposed answer than to produce a flawless answer in one pass. Making the critique explicit surfaces the error in context, which the model can then use to produce a revised answer.

**The mechanics:**
1. Generate a first answer.
2. Pass the problem and first answer back to the model with instructions to identify errors or gaps.
3. Pass the problem, first answer, and critique back for a revised answer.

This costs 2–3× more tokens. Revision is not guaranteed to improve the answer.

**What breaks:** the model can agree with its own errors ("the solution looks correct to me") when the error is subtle or in a domain where the model is systematically miscalibrated. Self-critique is most reliable for code — where execution can be traced mentally — and least reliable for subtle factual errors.

---

## Tree of Thought (ToT)

**The problem:** reflection is linear — one critique, one revision. For problems with multiple plausible solution paths (planning, combinatorial tasks), a single path may be a local optimum. The model commits to a direction early and cannot backtrack.

**The core insight:** convert generation from a greedy linear walk to a best-first search over the reasoning space. Generate several candidate next-steps at each node, score their promise, prune, and expand only the best branches. This recovers from early bad choices.

**The mechanics:**
1. From the current state, generate k possible next reasoning steps.
2. Score each step (using the model as an evaluator or a heuristic).
3. Expand only the top-scoring branches.
4. Continue until a complete answer is reached or the compute budget is exhausted.

**What breaks:** cost scales exponentially with branching factor and depth. This is only practical when accuracy justifies running the model dozens of times per query. The evaluation step is itself unreliable — a model that produces wrong reasoning can also misjudge which reasoning branch is better.

---

## Tool Use

**The problem:** an LLM generates text. It cannot read from a database, execute code, call an API, or look up a live fact. Generation can describe what should happen, but it cannot cause things to happen.

**The core insight:** give the model a set of callable functions with typed schemas. The model emits structured call requests as part of its output; a host process executes them; results come back as context. The model stays in the text-in, text-out paradigm it was trained on. The tools handle all real-world interaction.

**The mechanics (OpenAI function calling):**
- Define tools as JSON schemas (name, description, parameter types, required fields).
- Pass tool definitions alongside the prompt.
- The model outputs a structured tool-call object instead of prose when it decides a tool is needed.
- The host parses the call, executes the function, appends the result as a message, and calls the model again.

**What breaks:**
- **Hallucinated tool calls:** the model invents a tool name or argument not in the schema. Mitigated by strict schema validation with required fields and enum constraints.
- **Wrong argument values:** the model infers the right tool but the wrong parameter (wrong city, wrong date). Mitigated by unambiguous parameter descriptions.
- **Cascade errors:** a wrong tool call returns a bad observation; the model reasons correctly from bad data and produces a confidently wrong answer.
- **Infinite loops:** the model calls tools indefinitely without concluding. Always enforce a hard maximum iteration limit.

---

## Memory Systems

**The problem:** each LLM API call is stateless — the model starts fresh every time. A user who stated their preferences in a prior session cannot rely on the model remembering them. The context window is finite, expensive, and old messages eventually fall off.

**The core insight:** different kinds of "memory" have different retrieval patterns and lifetimes, and need different storage backends. Relevant past context is surfaced into the current context window at query time — the model does not "remember"; it reads.

**The mechanics:**

| Memory type | What it holds | Implementation |
|:---|:---|:---|
| In-context | Recent turns | Sliding message window |
| Short-term external | Session state, task progress | Redis or in-memory key-value store |
| Long-term semantic | Past interactions, user facts | Vector database (retrieve by similarity) |
| Structured state | Workflow variables, task steps | Relational DB or document store |

For semantic memory: embed each interaction summary, store in a vector index. At the start of each new query, retrieve the top-k most relevant past items and prepend them to the context.

**What breaks:** retrieved memories can be stale (old facts that are no longer true), irrelevant (retrieved by surface similarity but not actually useful), or contradictory. Without TTL (time-to-live) policies and freshness scoring, semantic memory accumulates outdated context the model will treat as current.

---

## Multi-Agent Patterns

**The problem:** some tasks exceed the scope of a single context window. A task like "research this topic, write a report, then write code to analyze the findings" requires different capabilities and more context than fits in one pass. Stuffing everything into one context causes quality degradation and cost explosion.

**The core insight:** decompose the task into roles and route each subtask to a specialized agent. An orchestrator holds the high-level plan; worker agents execute bounded subtasks and return results. Agents communicate through structured messages rather than shared state.

**The mechanics:**

*Orchestrator + subagents:* the orchestrator receives the goal, decomposes it into steps, dispatches each to a specialist (research, code, writing), receives results, and assembles final output.

*Supervisor / worker (LangGraph style):* a central supervisor node in a state graph decides which worker to invoke next based on current state. Workers update state and return control to the supervisor. The graph terminates when the supervisor emits a stop signal.

*Debate / critic:* two agents argue opposing positions; a third judges. Useful for decisions where confirmation bias is a risk.

**What breaks:**
- Multi-agent systems multiply every single-agent failure mode. One bad tool call in a subagent poisons the orchestrator's context.
- Coordination adds latency: 2–4× more LLM calls than a single agent.
- Inter-agent communication formats must be precise. If subagent A returns prose and subagent B expects JSON, the system breaks silently.
- Attribution is hard: tracing which agent introduced an error requires full-trace observability.

---

## Routing

**The problem:** not every query needs the same treatment. A question answerable from the model's training data does not need retrieval. A simple factual lookup does not need a 10-step reasoning chain. Sending every query through the most expensive path wastes time and money.

**The core insight:** classify the query first, then dispatch to the appropriate handler. A cheap classifier — even a small-model prompt with a constrained output — can save the cost of retrieval or tool calls on the majority of queries that do not need them.

**The mechanics:**
```python
class Route(str, Enum):
    RAG = "rag"       # needs document retrieval
    CODE = "code"     # needs code execution
    SEARCH = "search" # needs web search
    DIRECT = "direct" # answerable from model weights
```
The router classifies the query and dispatches to the matching handler. The router itself should be fast — a fine-tuned small model or a structured LLM call with constrained output.

**What breaks:** router errors are invisible in the final output. If a query that needs retrieval is misclassified as DIRECT, the model answers from potentially stale parametric memory with no warning. Router accuracy degrades on queries that legitimately span multiple categories.

---

## Guardrails and Safety

**The problem:** agentic systems run autonomously. A bug in the stopping condition, a misbehaving tool, or adversarial input can send the system into an infinite loop, cause irreversible real-world actions (sending emails, deleting files), or produce harmful outputs — all without human notice until damage is done.

**The core insight:** enumerate the ways the system can fail catastrophically, then enforce hard limits that cannot be overridden by the model's own reasoning. The model should not be able to escape guardrails by generating clever arguments for why it should.

**The mechanics:**

| Guardrail | Failure it prevents | Implementation |
|:---|:---|:---|
| Max steps | Infinite loops | Counter in state; raise after N steps |
| Per-tool timeout | Hung tool calls | Timeout decorator on every tool |
| Output schema validation | Malformed tool arguments | Pydantic models on all tool inputs/outputs |
| Content classifiers | Harmful output | Input/output filter (Llama Guard, etc.) |
| Human-in-the-loop | Irreversible high-stakes actions | Pause execution and require explicit approval |
| Sandboxed code execution | Host compromise from generated code | Docker / E2B containers |

**What breaks:** guardrails that are too strict make the system useless for legitimate tasks. Guardrails that are too loose let dangerous actions through. The right calibration requires red-teaming — deliberately trying to break the guardrails — before deployment.

---

## Observability

**The problem:** multi-step systems fail at multi-step boundaries. A wrong answer from a 10-step agent could have originated at step 2, 5, or 9. Without a trace of every LLM call and tool call, debugging is guessing. Logs that record only the final output are useless.

**The core insight:** instrument the full trace, not just the output. Every LLM call, every tool call, every routing decision, and every state transition must be recorded with inputs, outputs, latency, and cost. A trace viewer lets you replay the exact event sequence that led to a failure.

**The mechanics:**
- Every LLM call: record model name, prompt tokens, completion tokens, latency, full output.
- Every tool call: record tool name, input arguments, output, latency, success/failure.
- Full agent trace: record the sequence of steps, state at each step, and final answer.
- Cost: sum token costs across all LLM calls in a run.

Tooling: LangSmith, Weights & Biases Weave, Arize Phoenix, or custom OpenTelemetry traces.

**What breaks:** high-cardinality tracing increases storage cost non-linearly at scale. Sampling strategies (log 100% of failures, 1% of successes) help. PII in traces is a compliance risk — sanitize before logging.

---

## Production Failure Modes

| Failure | Cause | Fix |
|:---|:---|:---|
| Infinite loops | No stopping condition; goal never detected as met | Hard max-step limit + explicit termination prompt |
| Tool hallucination | Model invents tool names or arguments not in schema | Schema validation; constrained output format |
| Context overflow | Long tool outputs fill the context window | Summarize observations before appending to context |
| Stale memory | Retrieved past memories contain outdated facts | TTL on memory entries; freshness scoring |
| Cascade failures | One bad tool call poisons all downstream reasoning | Per-step output validation; graceful error handling |
| Prompt injection | Tool output contains adversarial instructions | Sanitize all external inputs before adding to context |

---

## Frameworks

| Framework | Strengths | Best for |
|:---|:---|:---|
| LangChain / LangGraph | Comprehensive; explicit state graphs | Complex multi-step workflows |
| LlamaIndex | Strong retrieval and document ingestion | RAG-heavy agentic systems |
| AutoGen (Microsoft) | Multi-agent conversations | Research, multi-agent experiments |
| CrewAI | Role-based agent definition | Team-style task decomposition |
| Direct API | Full control, no abstractions | Simple tool use; production-critical paths |

*Related: [RAG](02-rag.md) | [Hallucination Mitigation](05-hallucination-mitigation.md) | [Tuning and Optimization](10-tuning-optimization.md)*


## Flashcards

**Why does Chain-of-Thought reduce (but not eliminate) reasoning errors?** #flashcard
Writing out intermediate steps puts them into the model's own context, so later tokens condition on those intermediate results instead of implicit weights alone. But the trace can look structurally correct while being factually wrong — CoT lowers the error rate, it doesn't guarantee correctness.

**How does ReAct differ from plain Chain-of-Thought?** #flashcard
CoT reasons only from parametric memory. ReAct interleaves Thought → Action (tool call) → Observation, grounding each reasoning step in a real external result instead of the model's internal knowledge — necessary for live data or facts the model doesn't already know.

**Why is self-critique/reflection unreliable for factual errors but more reliable for code?** #flashcard
Critiquing an existing answer is an easier task than generating a perfect one, so it can surface flaws. But the model can also agree with its own subtle factual errors when it's systematically miscalibrated in that domain. Code errors are more traceable because execution can be mentally simulated, making self-critique more trustworthy there.

**When is Tree of Thought worth its exponential cost over linear reflection?** #flashcard
ToT explores multiple candidate reasoning branches with scoring and pruning instead of committing to one linear path — useful for planning/combinatorial problems with multiple plausible solutions. Cost scales exponentially with branching factor and depth, so it's only justified when accuracy gains from backtracking outweigh running the model dozens of times per query.

**What are the main tool-use failure modes and their mitigations?** #flashcard
Hallucinated tool calls (invented names/args) — mitigate with strict schema validation. Wrong argument values — mitigate with unambiguous parameter descriptions. Cascade errors (bad tool output reasoned over correctly, producing a confidently wrong answer) — mitigate with output validation. Infinite loops — mitigate with a hard max iteration limit.

**Why do agent memory systems need different backends for different memory types?** #flashcard
In-context (recent turns) uses a sliding window; short-term state uses Redis/key-value store; long-term semantic memory (past facts) uses a vector DB retrieved by similarity; structured workflow state uses a relational DB. Each has different retrieval patterns and lifetimes. Risk: without TTL and freshness scoring, semantic memory accumulates stale facts the model treats as current.

**What causes multi-agent systems to fail more than single-agent ones, despite handling more complex tasks?** #flashcard
Multi-agent systems multiply every single-agent failure mode — one bad tool call in a subagent poisons the orchestrator's context. They add 2-4x more LLM calls (latency/cost), require precise inter-agent message formats (a prose/JSON mismatch breaks silently), and make error attribution hard without full-trace observability.

**Why route queries instead of sending everything through the most capable/expensive path?** #flashcard
Not every query needs retrieval or multi-step reasoning — a cheap classifier (small model or constrained-output LLM call) can dispatch simple queries directly and save cost on the majority that don't need the expensive path. Risk: router errors are invisible in the output — a misclassified query silently answers from stale parametric memory with no warning.

**What guardrails are non-negotiable for autonomous agentic systems, and why must they not be model-overridable?** #flashcard
Max step counters (prevent infinite loops), per-tool timeouts (prevent hung calls), output schema validation (prevent malformed tool args), content classifiers (block harmful output), human-in-the-loop approval (for irreversible actions), and sandboxed execution (contain generated code). These must be hard-enforced outside the model's own reasoning, since the model could otherwise talk itself past a soft guardrail.

**Why is full-trace observability essential for debugging multi-step agents?** #flashcard
A wrong final answer from a 10-step agent could have originated at any step. Logging only the final output makes debugging guesswork. Full tracing (every LLM call, tool call, routing decision, state transition with inputs/outputs/latency/cost) lets you replay the exact sequence that led to a failure — at the cost of non-linear storage growth, mitigated by sampling (100% of failures, 1% of successes) and PII sanitization.
