# Agentic Design Patterns — First-Principles Notes

Each pattern below is introduced by the specific failure mode it exists to fix.

---

## Chapter 1: Prompt Chaining (The Pipeline Pattern)

### The Problem
Ask an LLM to simultaneously summarize a document, identify three trends, and draft a stakeholder email. What breaks?

- **Instruction neglect**: With many constraints in one prompt, the model skips some.
- **Context drift**: Long outputs cause the model to lose the thread of earlier constraints.
- **Error propagation**: A wrong assumption in paragraph one compounds through the rest.
- **Hallucination surface area**: The more the model must invent in one pass, the more it invents incorrectly.

The single-prompt approach fails because the cognitive load of a complex task exceeds what the model can reliably hold in one forward pass.

### The Core Insight
A model that does one thing at a time is a model that can be correct at each step. You decompose the task, not just the prompt.

### The Mechanics
Each step in the chain is assigned a single responsibility. The output of step N becomes the *only* input to step N+1 — nothing else leaks through.

```
Step 1 (Market Analyst)  →  { "summary": "..." }
Step 2 (Trend Analyst)   →  { "trends": [...] }
Step 3 (Writer)          →  Final email
```

**Structured intermediate outputs** (JSON, XML) are essential. If step 2 returns prose, step 3 must parse natural language, re-introducing ambiguity. If step 2 returns `{"trend_name": "...", "supporting_data": "..."}`, step 3 has a contract to work from.

### What Breaks
- **Under-decomposed chains**: If a single step still carries two responsibilities, you have the original problem with extra wiring.
- **Context contamination**: Passing too much history from prior steps overloads context and re-introduces drift.
- **Silent schema violations**: If step 2 emits malformed JSON, step 3 fails silently or hallucinates a fix, and the error becomes invisible.

---

## Chapter 2: Routing

### The Problem
A single sequential pipeline assumes all inputs follow the same path. A customer service agent that routes "I want to book a hotel" and "What is the tallest mountain?" through the same steps wastes compute, applies wrong tools, and produces wrong answers. Linear rigidity breaks whenever input variability is high.

### The Core Insight
Before executing, classify. Insert a decision node that evaluates *what kind of problem this is* before choosing *which pipeline handles it*. Deterministic execution (`A → B`) becomes conditional execution (`if X then A, else B`).

### The Mechanics
Two implementation strategies, each with a different cost-accuracy profile:

**LLM-Based Routing**: Instruct the model to emit *only* a category label:
> "Classify this query as exactly one of: 'Order Status', 'Product Info', 'Technical Support'. Output only the label."

High accuracy, but costs a full LLM call per route decision.

**Embedding-Based (Semantic) Routing**: Encode the user query and predefined route descriptions as vectors. Route to the highest cosine similarity. No LLM call required — just a dot product. Faster and cheaper, but blind to edge cases that require reasoning.

Framework examples:
- **LangGraph**: Conditional edges between graph nodes encode the routing logic explicitly.
- **Google ADK**: A Coordinator agent with registered sub-agents routes implicitly — the LLM decides which sub-agent's scope matches the request.

### What Breaks
- **Ambiguous boundaries**: If "billing issue" overlaps with both "Order Status" and "Technical Support", the router misclassifies consistently at the boundary.
- **Embedding drift**: Semantic routing fails when the user's phrasing is unusual. "My parcel hasn't shown up" may not embed near "Order Status" if the training distribution didn't include informal language.
- **Router hallucination**: LLM-based routers instructed to output a category sometimes output a sentence instead. Enforce structured output or parse with a fallback.

---

## Chapter 3: Parallelisation

### The Problem
In a sequential pipeline, independent sub-tasks block each other. If researching a company requires checking news, pulling stock data, and reading the CEO's bio, sequential execution takes 3× the time of the slowest step. The tasks share no dependency, yet they queue as if they do.

### The Core Insight
Dependency is a property of data flow, not of time. Tasks that do not consume each other's outputs have *no reason* to be sequential. Identify the dependency graph; run independent branches simultaneously.

### The Mechanics
Split the workflow into branches where no branch reads the output of another, run them concurrently, then aggregate:

```
                          ┌─ Branch A: Fetch news → Summarize news ──┐
User request ──► Router ──┤                                           ├─► Aggregate → Final answer
                          └─ Branch B: Pull stock data → Analyze ────┘
```

Framework primitives:
- **LangChain LCEL**: `RunnableParallel({...})` — a dict of chains that execute concurrently, results collected as a map.
- **LangGraph**: A single state node can transition to multiple child nodes; convergence is handled at an aggregation node.
- **Google ADK**: `ParallelAgent` class manages concurrent sub-agent execution natively.

### What Breaks
- **Rate limits**: Parallel agents fire multiple API calls simultaneously. Burst usage can hit provider rate limits and cause all branches to fail at once.
- **Aggregation logic**: Parallel branches produce unordered results. If the aggregation step assumes a specific result order, it will misread inputs.
- **False independence**: If Branch B actually needs a fact from Branch A to compute correctly, parallelising them produces silently wrong results. Dependency analysis must be done before parallelising.
- **Cost amplification**: Parallelisation reduces wall-clock time but not total token spend. Costs accumulate across all branches simultaneously.

---

## Chapter 4: Reflection

### The Problem
LLMs produce fluent text confidently regardless of correctness. A model asked to write code or summarize a document outputs a plausible-looking result with no internal check. There is no "pause to verify" step in a standard forward pass. Bugs, missed constraints, and factual errors all emerge as polished prose.

### The Core Insight
Correctness requires a separate evaluation pass. The agent that generates output is poorly positioned to evaluate it — it is primed to continue in the direction it just generated. A separate critic role, with a different prompt and explicit evaluation criteria, can catch what the generator cannot.

### The Mechanics
The Producer-Critic (Generator-Reviewer) loop:

```
1. Generate  →  Producer creates initial draft
2. Evaluate  →  Critic checks against explicit criteria (security, correctness, tone, completeness)
3. Refine    →  Producer rewrites using the Critic's specific feedback
4. Repeat    →  Until Critic passes, or max_retries is reached
```

The Critic's prompt must encode *what good looks like* precisely. "Check for security bugs" is more effective than "Improve this code."

**Memory dependency**: The feedback loop only works if the Producer can see *why* a prior attempt failed. Without conversation history, the Producer will re-generate the same error. Memory (Chapter 8) is not optional for multi-turn reflection.

**Goal dependency**: The Critic needs a benchmark to judge against. Goals (Chapter 11) provide that benchmark — reflection without a defined success criterion is circular.

### What Breaks
- **Missing exit condition**: Without `max_retries`, an agent that can never satisfy the Critic loops infinitely.
- **Critic hallucination**: A critic LLM can generate plausible-sounding criticism that is itself wrong. If the producer obeys, output quality degrades.
- **Role collapse**: When the same model instance plays both Producer and Critic in the same prompt, it tends to validate its own work. Separate prompts, ideally separate model instances, are needed.
- **Latency multiplication**: N reflection cycles = N× the generation latency and N× the API cost of a single call.

---

## Chapter 5: Tool Use (Function Calling)

### The Problem
A pre-trained LLM's knowledge is frozen at training time. It cannot tell you today's stock price, whether a SKU is in stock, or what is in a specific user's database record. Asked to compute `438,291 × 7`, it will produce a plausible but unreliable answer. The model is stateless with respect to the real world.

### The Core Insight
The LLM should not *compute* real-world facts — it should *request* them. Give the model the ability to generate a structured request for an external system, have that system execute the request, and inject the result back into context. The model becomes an *orchestrator of tools* rather than a calculator.

### The Mechanics
A five-step cycle:

1. **Tool definition**: Developer describes available tools in structured form (name, parameters, types, purpose).
2. **Intent recognition**: LLM receives a user prompt, identifies which tool is needed, outputs a structured call object — `{"tool": "get_stock_price", "ticker": "AAPL"}`.
3. **Client routing**: The host application intercepts the structured output and routes it to the appropriate service.
4. **Execution**: The external service runs the real operation.
5. **Context injection**: The result is fed back to the LLM as a new message, allowing it to formulate the final response.

**MCP vs Function Calling**:
- Function Calling is vendor-specific. OpenAI's schema differs from Google's. The integration is hardcoded.
- Model Context Protocol (MCP) is an open standard. The LLM client queries the MCP server dynamically at runtime: "What tools do you expose?" The server responds with a live tool manifest. This separates tool definition from agent logic and allows hot-swapping.

### What Breaks
- **Malformed structured output**: If the LLM emits a malformed JSON tool call, the client cannot parse it. Enforce structured output schemas or add a parsing fallback.
- **Data format incompatibility**: An MCP server that returns a raw PDF is useless to an LLM that needs text. Agent-friendliness requires the server to return parsed, readable data.
- **Blind trust**: If the tool returns an error and the agent silently ignores it, subsequent reasoning is grounded in nothing. Tool results must be validated before being used as facts.
- **Google ADK distinction**: Extensions are executed automatically by the Vertex AI platform. Function Calls require the client application to execute manually. Confusing these leads to agents that "call" tools but never see results.

---

## Chapter 6: Planning

### The Problem
A reactive agent processes each input as it arrives and responds. Ask it to "Organize a team offsite," and it either asks a single clarifying question and stops, or immediately calls a tool without knowing what the end state should be. Without a map of what success looks like, the agent cannot sequence its steps or detect when it has gone off track.

### The Core Insight
Before acting, generate the map. Force the agent to decompose the goal into an ordered sequence of sub-tasks, establishing dependencies and checkpoints before any tool is called. Action becomes implementation of a plan, not reaction to a stimulus.

### The Mechanics
A four-stage loop:

1. **Goal ingestion**: Agent receives high-level, ambiguous objective.
2. **Plan generation**: LLM decomposes it into structured steps, output as JSON array or numbered list.
3. **Execution**: System iterates through plan, calling tools or sub-agents per step.
4. **Refinement** (optional): If a step fails or returns unexpected data, the agent replans from that point.

Explicit prompting is required:
> "First, create a numbered plan. Then execute the plan one step at a time."

Without this instruction, most models skip straight to execution.

Plans should be serializable (JSON arrays, not prose) so the system can programmatically track which steps have completed.

**ReAct** (Reason + Act) is the foundational framework: the model emits a `Thought`, then an `Action`, then observes the `Observation`, then reasons again. This cycle *is* planning over execution.

### What Breaks
- **Stale plans**: If step 3 fails and the agent continues executing steps 4-7 as if step 3 succeeded, it propagates an invalid state through the rest of the plan.
- **Over-decomposition**: Excessively granular plans waste tokens and introduce more points of failure.
- **Dependency blindness**: A plan that lists "Book the venue" before "Confirm the date" will fail in execution. The planning prompt must instruct the model to identify dependencies.
- **Plan hallucination**: The agent generates a plan that includes steps it has no tools to execute. Validate the plan against available tools before starting execution.

---

## Chapter 7: Multi-Agent Collaboration

### The Problem
A monolithic agent given a complex, multi-domain task (research + write + edit + verify + check legal compliance) must hold contradictory objectives simultaneously. "Be creative" and "be strictly factual" conflict. The context window fills with competing instructions. One failure in any sub-task cascades into the whole response. Specialization is impossible when a single agent must do everything.

### The Core Insight
Decompose not just the task but the agent. Assign narrow, non-conflicting responsibilities to separate agents. Let each agent be excellent at one thing. Coordination is then a problem of communication, not a problem of prompt engineering.

### The Mechanics
Four collaboration architectures:

**Orchestrator/Worker (most common)**:
- A Manager agent breaks down the goal, delegates tasks to Worker agents, aggregates their outputs.
- Workers report results back; they do not communicate with each other directly.

**Sequential Handoffs**:
- `Researcher → Writer → Editor`. Each agent receives the prior agent's full output.
- Simple but brittle: a bad handoff at any step corrupts all subsequent agents.

**Debate/Consensus**:
- Multiple agents (or the same model playing different roles) propose and critique solutions.
- The adversarial dynamic filters out hallucinations and weak reasoning before the final answer is committed.

**Hierarchical Teams**:
- Chief Editor → Section Editors → Writers. Useful for large-scale document generation.

Framework primitives:
- **CrewAI**: Designed specifically for this pattern. Define `Agent` (role, backstory, goal), `Task` (assignment), `Crew` (team + execution order).
- **Google ADK**: Coordinator + sub-agents with auto-flow delegation.
- **LangGraph**: Explicit state machine where edges carry handoffs between agents.

### What Breaks
- **Context window starvation**: Each agent has its own context window. If the Manager tries to pass the entire conversation history to every Worker, all workers hit their limits simultaneously.
- **Silent failure in workers**: If a Worker fails and returns a plausible-looking error embedded in natural language, the Manager may treat it as a valid result.
- **Attribution difficulty**: When a multi-agent output is wrong, tracing the error to the responsible agent requires tracing message provenance through the entire chain.
- **Cost explosion**: N agents × M turns = N×M LLM calls per user request. The economics must be validated before deploying this architecture at scale.

---

## Chapter 8: Memory Management

### The Problem
Every LLM API call is stateless. The model that answered a question two minutes ago has no record of it. A user who says "Actually, change that to Tuesday" gets a confused response if the agent has no access to what "that" refers to. Without memory, every interaction is a first conversation.

### The Core Insight
State must be managed explicitly. The developer is responsible for deciding what to store, where to store it, and what to inject into each new context. There are two distinct storage problems: what the agent is working on *right now* (short-term), and what the agent should remember across sessions (long-term).

### The Mechanics
**Short-Term (Session State)**:
- Holds current conversation history, tool outputs, and workflow progress.
- Limited by the context window. Once full, older items are evicted unless explicitly saved.
- Implemented as `InMemorySessionService` (Google ADK) or `ConversationBufferMemory` (LangChain).

**Long-Term (Persistent)**:
- Survives across sessions. Accessed via retrieval, not by stuffing everything into context.
- Three cognitive subtypes:
  - **Episodic**: What happened in past sessions. ("User asked about refund policy last Tuesday.")
  - **Semantic**: Domain facts and knowledge base. ("Company travel policy: max $250/night hotel.")
  - **Procedural**: Rules for *how* to behave. Can be updated by Reflection — the agent rewrites its own system prompt based on past mistakes.

Long-term memory is implemented via a Vector DB + RAG: store past interactions as embeddings; retrieve by semantic similarity when context is relevant.

### What Breaks
- **Context window overflow**: Injecting too much history defeats the purpose of memory management. Only inject what is *relevant*, retrieved by similarity.
- **Stale memory**: A fact stored in long-term memory may be outdated. Without a TTL (time-to-live) or versioning, the agent cites obsolete information confidently.
- **Procedural memory corruption**: If Reflection incorrectly rewrites the agent's own instructions, the degraded instructions persist and worsen over time.
- **Session/knowledge confusion**: Treating ephemeral session state as permanent knowledge (or the reverse) causes agents to forget ongoing tasks or over-persist irrelevant details.

---

## Chapter 9: Learning and Adaptation

### The Problem
A static agent will make the same mistake the second time it encounters a case its initial design didn't anticipate. It cannot learn that a user always prefers aisle seats. It cannot internalize that a particular API returns unreliable data. Without feedback loops that change behavior, every improvement requires a human developer to rewrite a prompt.

### The Core Insight
Performance information generated during execution is training signal. The agent that logs what worked and what failed has the raw material to improve. The question is whether it has the architecture to act on that signal.

### The Mechanics
Three levels of adaptation:

**Reinforcement Learning**: Agent receives reward/penalty signals; gradually learns policies that maximize reward. Requires explicit reward function design.

**Knowledge Base Learning (RAG-based)**: Store successful strategies and known failure patterns in a searchable database. Before acting, retrieve analogous past cases. Effectively: the agent learns by giving itself a richer few-shot prompt derived from real experience.

**Self-Modification**: Advanced agents rewrite their own internal code or system prompts (procedural memory update). SICA (Self-Improving Coding Agent) demonstrated this concretely: through iterative self-modification, it autonomously developed specialized tools (a Smart Editor, an AST Symbol Locator) that it had not been initially given.

**Architecture**: A two-level system is typically required:
- **Sub-agents**: Execute specific tasks, produce performance data.
- **Overseer agent**: Monitors sub-agent performance, manages the learning pipeline, prevents runaway self-modification.

### What Breaks
- **Reward hacking**: The agent finds ways to maximize the reward signal that do not correspond to the intended behavior.
- **Context saturation**: Learned examples accumulated in the system prompt or retrieved context can crowd out the actual task.
- **Runaway self-modification**: An agent that rewrites its own instructions without an overseer can degrade its own behavior irreversibly in a single session.
- **False generalization**: An agent that learned "always retry three times" from one API may apply that rule to every tool, including ones where retries cause side effects (double-posting, double-charging).

---

## Chapter 10: Model Context Protocol (MCP)

### The Problem
Every LLM provider has a different tool-calling format. OpenAI's function calling schema differs from Anthropic's differs from Google's. Every new tool an agent needs to use requires a custom integration written against each provider's API. The result is a combinatorial explosion: N models × M tools = N×M bespoke adapters. Switching providers requires rewriting every integration.

### The Core Insight
Standardize the interface, not the implementation. If every tool speaks the same protocol, any agent can use any tool without custom code. This is the same insight as USB: standardize the port, let devices handle their own internals.

### The Mechanics
MCP operates on a three-role architecture:

- **MCP Host** (the application, e.g., Claude Desktop, an IDE): Owns the connection lifecycle and permissions.
- **MCP Client** (the LLM/agent): Queries servers for capabilities, sends structured requests.
- **MCP Server** (wraps a tool or data source): Exposes three things:
  - **Resources**: Static data the agent can read (files, DB records, logs).
  - **Prompts**: Pre-written templates for using the server effectively.
  - **Tools**: Executable functions the agent can invoke.

**Dynamic Discovery** is the critical difference from Function Calling: the client asks the server at runtime "What can you do?" and the server returns a live manifest. Adding a new tool to the server makes it immediately available to all connected agents — no prompt update required.

**Transport**: Local tools use `stdio`; remote APIs use `SSE` (Server-Sent Events).

**FastMCP** (Python library) removes boilerplate: decorate a Python function with `@mcp.tool()` and it is automatically exposed with JSON-RPC handling, error management, and schema generation.

**MCP vs A2A**: MCP is for agent-to-tool communication. A2A (Chapter 15) is for agent-to-agent communication. They operate at different levels of the stack.

### What Breaks
- **Data format mismatch**: An MCP server that returns raw bytes or a binary PDF is protocol-compliant but agent-useless. Servers must return agent-readable formats (text, Markdown).
- **Dynamic discovery latency**: If an agent queries the server's tool manifest on every request, that round-trip adds latency. Cache the manifest and invalidate on server version change.
- **Permissions ambiguity**: The Host manages permissions, but if the Host grants broad access, a misbehaving agent can call destructive tools it should not reach.

---

## Chapter 11: Goal Setting and Monitoring

### The Problem
A reactive agent that completes each sub-task correctly can still fail the overall objective. Completing "search for flights," "book cheapest flight," and "confirm hotel" correctly, but in the wrong sequence, or without checking that the hotel and flight dates match, produces a useless trip. Without an explicit objective and a check against it, the agent is executing steps, not achieving goals.

### The Core Insight
Define what done looks like before starting. Give the agent a machine-readable success criterion, not just a task description. Then give it a monitoring loop that continuously compares current state against that criterion.

### The Mechanics
**SMART Goals** as the goal specification framework:
- **Specific**: Clear, unambiguous objective.
- **Measurable**: A state the system can check programmatically.
- **Achievable**: Within the agent's tool inventory.
- **Relevant**: Aligned with the user's actual need.
- **Time-bound**: With a deadline or step budget.

**Monitoring Loop**:
1. **Observe**: Track tool call results, state variables, and intermediate outputs.
2. **Evaluate**: Compare observed state against the measurable success criteria.
3. **Adapt**: If drifting, revise the plan. If stuck, escalate to human.

Google ADK implements this via session state variables. Tool call outcomes (success/error codes) serve as the primary monitoring signal.

### What Breaks
- **Underspecified goals**: "Help the user plan a trip" has no measurable completion condition. The agent can loop indefinitely or stop too early.
- **Metric gaming**: An agent optimized for a proxy metric (e.g., "number of tasks completed") can satisfy the metric while missing the real objective.
- **Monitoring overhead**: Continuous state evaluation adds latency to every step. For high-frequency actions, lightweight monitoring is required.
- **Goal drift**: In long sessions, an agent's accumulated context can shift its interpretation of the original goal. The goal must be re-injected into context at regular intervals.

---

## Chapter 12: Exception Handling and Recovery

### The Problem
Real-world APIs fail. Rate limits hit. Files are missing. The LLM generates a parameter that doesn't exist. An agent built only for the happy path will crash or emit broken output the moment any external dependency misbehaves. In a pipeline of ten steps, the probability of encountering at least one failure is high.

### The Core Insight
Failures are not exceptional — they are expected. An agent designed for production must have explicit handling for every failure mode: what to detect, what to try next, and how to return to a stable state without crashing the entire session.

### The Mechanics
**Three-phase resilience model**:

1. **Detection**: Identify that a failure occurred.
   - Programmatic: catch HTTP 4xx/5xx, parse errors, timeouts.
   - Semantic: a Critic agent flags that the output makes no logical sense.

2. **Handling**: Choose a response strategy based on failure type:
   - **Log**: Always. Visibility is prerequisite to debugging.
   - **Retry with exponential backoff**: For transient failures (rate limits, network blips).
   - **Fallback**: If primary method fails, try a secondary. Graceful degradation: provide a partial answer rather than nothing.

3. **Recovery**: Return to a stable state — update session state to reflect the failure, resume the plan from the last valid checkpoint.

Self-correction example (code agent): agent writes code → code fails to execute → error traceback is injected into context → agent reasons about the error and rewrites the code.

### What Breaks
- **Silent failures**: If a tool returns a plausible-looking error embedded in natural language, the agent may not detect it programmatically and proceeds on false data.
- **Infinite retry loops**: Retrying a structurally broken request (wrong endpoint, malformed payload) will always fail. Distinguish transient from permanent failures before retrying.
- **Cascading fallbacks**: If fallback A fails and triggers fallback B, which triggers fallback C, the agent may end up multiple hops from its original intent with no record of how it got there.

---

## Chapter 13: Human-in-the-Loop

### The Problem
Agents make confident errors. A customer service agent might promise a refund that violates policy. A legal agent might mis-cite a statute. A 99% accuracy rate means 1% of actions are wrong — in high-stakes domains, that remaining 1% can be legally or physically catastrophic. Full autonomy is not appropriate for every decision.

### The Core Insight
Human oversight is not a fallback for broken agents — it is a deliberate design choice for cases where the cost of an undetected error exceeds the cost of human review time. Build the handoff to humans into the architecture, not as an afterthought.

### The Mechanics
Three integration patterns:

**Approval (Gatekeeper)**: Agent prepares a proposed action; a human must approve before execution. Blocks the happy path intentionally. Used for: email sends, code commits, database writes.

**Feedback (Teacher)**: Human corrects the agent's output. Correction is stored and used as training signal for future runs (RLHF). The agent improves from real-world mistakes rather than only synthetic training data.

**Escalation (Safety Net)**: Agent detects its own uncertainty (low confidence score, user frustration via sentiment analysis, policy-triggering content) and hands the conversation to a human without prompting. The human sees the agent's reasoning log, not just the final output.

**Active vs. Passive HITL**:
- **Active**: Human is a required step in every execution of this workflow.
- **Passive**: Human monitors dashboards; only intervenes when an alert fires.

### What Breaks
- **Approval bottleneck**: If every action requires human sign-off, the agent provides no latency benefit over a human doing the work directly. Reserve approval gates for high-stakes, irreversible actions only.
- **Context-free review**: If the human sees only the agent's proposed action, not its reasoning, they cannot meaningfully evaluate it. HITL requires a reasoning log UI, not just a Yes/No button.
- **Trust erosion**: If humans approve agent actions without reviewing them (rubber-stamping), the oversight provides legal cover but not actual safety.

---

## Chapter 14: Knowledge Retrieval (RAG)

### The Problem
LLMs hallucinate. When the model lacks specific knowledge, it generates a plausible-sounding answer rather than admitting ignorance. Additionally, training data has a cutoff date and cannot include private organizational knowledge. An agent asked about current inventory, internal policy, or recent legal precedent will invent answers it cannot actually know.

### The Core Insight
Separate *what the model knows* from *what the model says*. Before generating, retrieve actual documents from a verified source. Ground the generation in retrieved text. Hallucination is not a reasoning failure — it is a knowledge gap. Fill the gap with retrieval before generation runs.

### The Mechanics
Three required technical components:

**Embeddings**: Text → dense vector (semantic representation). The model encodes *meaning*, so "canine" and "dog" produce nearby vectors despite sharing no characters.

**Chunking**: Large documents must be split into smaller segments before embedding. The retrieval system returns the specific paragraph relevant to the query, not an entire 50-page document. Chunk size is a tunable parameter: too small loses context; too large reduces precision.

**Vector Search**: The user's query is embedded, and the system finds the K stored chunks with highest cosine similarity. This is semantic matching, not keyword matching — the query "What is the reimbursement limit?" correctly retrieves the paragraph containing "Employees may claim up to $250 per night."

**Full RAG loop**:
```
Query → Embed query → Vector search → Retrieve top-K chunks
→ Inject chunks into prompt → LLM generates answer grounded in retrieved text
→ Optional: cite retrieved sources
```

### What Breaks
- **Retrieval quality ceiling**: RAG cannot improve an answer if the correct information is not in the database. The retrieval system is only as good as its data.
- **Chunk boundary artifacts**: If the answer spans two chunks that are retrieved separately, the model may see each half but not the complete picture.
- **Query-answer embedding gap**: Questions ("What is the CEO's salary?") and answers ("$4.2M annually") may not embed near each other because they are semantically different in form. Bi-encoder fine-tuning on Q-A pairs addresses this.
- **Hallucination despite retrieval**: If the retrieval returns irrelevant chunks, the model may still hallucinate rather than returning "I don't know." Add a relevance threshold: if no chunk exceeds it, return a "no information found" response.

---

## Chapter 15: Inter-Agent Communication (A2A)

### The Problem
Multi-agent systems built with different frameworks cannot communicate. An agent built in Google ADK cannot delegate tasks to an agent built in CrewAI. Every multi-agent system becomes a silo. As agent ecosystems grow, the combinatorial cost of custom integrations becomes prohibitive — the same problem MCP solved for tools, but at the agent collaboration layer.

### The Core Insight
Define a standard protocol for agents to discover each other, describe their capabilities, and exchange tasks. The protocol must be framework-agnostic and HTTP-based so any agent, anywhere, can participate.

### The Mechanics
Three architectural components:

**Agent Card**: An agent's identity document. Published at a well-known URL. Contains: name, description, supported task types, communication endpoint. Used by other agents to discover and evaluate whether to delegate.

**Communication Patterns**:
- **Request/Response (Polling)**: Agent A sends a task, waits for Agent B's reply. Simple but ties up resources during wait.
- **Webhooks (Event-driven)**: Agent B notifies Agent A when work is complete. Agent A can do other work in the meantime.

**State Management**: A2A maintains shared task context during multi-turn interactions. Follow-up messages from Agent A carry the task ID so Agent B can retrieve prior history.

**A2A vs MCP comparison**:
| | A2A | MCP |
|---|---|---|
| What connects | Agent ↔ Agent | Agent ↔ Tool/Data Source |
| Level | Task delegation, workflow | Tool invocation, data access |
| Unit of exchange | Tasks, intents | Function calls, resource reads |

Tools like Trickle AI visualize inter-agent message flow — essential for debugging multi-agent traffic that is otherwise invisible.

### What Breaks
- **Agent Card staleness**: If an agent's capabilities change but its Agent Card is not updated, other agents delegate tasks it can no longer handle.
- **Trust between agents**: In an open A2A ecosystem, a malicious agent could present a fraudulent Agent Card and intercept delegated tasks. Authentication and capability verification are required.
- **Cascading latency**: A task delegated A → B → C has the combined latency of three network hops plus three LLM generations. Depth of delegation chains must be bounded.

---

## Chapter 16: Resource-Aware Optimization

### The Problem
Using a frontier reasoning model for every query is slow and expensive. Asking GPT-4-class models "Is this sentence in English?" wastes 99% of its capability and incurs high latency and cost at scale. Production systems serving millions of users cannot absorb this inefficiency.

### The Core Insight
Match model capability to task complexity at call time, not at design time. A routing decision — "is this query hard or easy?" — made before each call can direct simple tasks to small, fast, cheap models and complex tasks to large, capable, expensive ones.

### The Mechanics
Four optimization strategies:

**Dynamic Model Switching**: Classify query complexity before generation. Route simple queries (greetings, factual lookups) to small models (Gemini Flash). Route multi-step reasoning or ambiguous tasks to large models (Gemini Pro). Net effect: dramatically lower average cost and latency without sacrificing quality on hard cases.

**Adaptive Tool Use**: Rather than giving every agent every tool (which fills context and increases confusion), provide only the tools relevant to the current task category. Reduces context size and improves tool selection accuracy.

**Contextual Pruning**: Conversation history grows without bound if not managed. Summarize or prune older, less relevant turns to keep context focused. Prevents context bloat from degrading response quality and increasing cost.

**Proactive Resource Prediction**: Pre-classify task difficulty to allocate compute before starting. An agent that knows "this will require web search + multi-step reasoning" can pre-warm the correct tools and allocate the appropriate model tier.

**Context Caching**: For repeated queries against the same large document (e.g., a 500-page manual), cache the processed key-value representations. Avoid re-processing on every query.

### What Breaks
- **Misclassification**: If the complexity router routes a hard question to a small model, the answer is wrong and the optimization backfires. Router accuracy is a system requirement, not an optimization.
- **Switching overhead**: The latency of the routing decision itself must be smaller than the latency savings it produces. Heavyweight routing models negate the benefit.
- **Pruning loss**: Aggressive context pruning can discard a detail the model needs 10 turns later. Pruning strategy must distinguish between ephemeral context and load-bearing facts.

---

## Chapter 17: Reasoning Techniques

### The Problem
Standard LLM generation is a single forward pass: input goes in, output comes out. For math problems, multi-step planning, or questions requiring hypothesis evaluation, this is insufficient. The model cannot hold and compare multiple partial solutions. It commits to a direction early and generates coherently in that direction, even if the direction was wrong.

### The Core Insight
Performance scales with structured thinking time. Forcing the model to externalize intermediate steps — rather than jumping to conclusions — gives it a scaffold to reason correctly. More computation at inference time, directed at structured problem decomposition, produces accuracy gains equivalent to using a larger model.

### The Mechanics
Four reasoning architectures:

**Chain of Thought (CoT)**:
- Prompt the model to "think step by step." Each intermediate step is generated as text, and that text becomes input to the next step.
- Benefit: The model cannot skip a logical step without it being visible. Errors in step 3 can be caught by the logic of steps 4 and 5.
- Zero-Shot CoT: append "Let's think step by step" — no examples required.

**Tree of Thoughts (ToT)**:
- Non-linear. The model generates multiple possible next steps from the current state, evaluates each branch, and prunes unpromising ones.
- Analogy: a chess player simulating three candidate moves before choosing.
- Appropriate for problems where the optimal path is non-obvious and backtracking has value.

**ReAct (Reason + Act)**:
- The loop: `Thought → Action (tool call) → Observation (tool result) → Thought → ...`
- Grounds reasoning in real-world feedback. The model does not just plan; it updates the plan based on what tools actually return.
- Considered the gold standard for autonomous agents.

**Graph of Debates (GoD) / Chain of Debates**:
- Multi-agent adversarial reasoning. One agent proposes; another attacks; a judge decides.
- Filters hallucinations and weak logic through structured disagreement before committing to an answer.

**Inference-Time Compute Scaling**: Allowing a model to generate thousands of internal reasoning steps (hidden from the user) before producing output can allow a smaller model to outperform a larger model that generates instantaneously. Performance is not only a function of model size.

### What Breaks
- **CoT hallucinated steps**: The model can generate plausible-sounding intermediate steps that are logically incorrect. The chain produces a confident wrong answer.
- **ToT computational explosion**: Evaluating every branch of a reasoning tree is expensive. Without effective pruning, ToT costs grow exponentially.
- **ReAct tool loops**: An agent in a ReAct loop that encounters repeated tool failures can loop indefinitely, generating Thought/Action cycles that never converge.
- **Debate convergence failure**: In GoD, if the judge agent has a systematic bias, it will consistently select wrong answers regardless of the quality of the debate.

---

## Chapter 18: Guardrails / Safety Patterns

### The Problem
Autonomous agents can be prompted, tricked, or simply wrong in ways that cause real harm: generating illegal advice, leaking user PII, calling destructive tools, or producing output that violates policy. An agent capable of taking action in the world with no safety layer is a liability, not a product.

### The Core Insight
Safety is a multi-layer property, not a single check. No single guardrail is sufficient because each layer can be bypassed or fail. Effective safety is defense-in-depth: filter at input, constrain at behavior, restrict at tool use, and filter again at output.

### The Mechanics
Four distinct guardrail layers, applied sequentially:

**1. Input Validation**: Filter requests before they reach the core LLM.
- Detect jailbreak attempts (prompts engineered to bypass safety rules).
- Block requests that violate policy on first contact.
- Use a fast, cheap model for this — latency cost must be minimal.

**2. Behavioral Constraints (Prompt-Level)**: Embed hard rules in the system prompt.
- "Do not provide financial, legal, or medical advice."
- "If the user asks about competitors, acknowledge and redirect."
- These are soft constraints — they can be overridden by sufficiently adversarial prompts. Not the only safety layer.

**3. Tool Use Restrictions**: Limit what the agent can do, not just what it can say.
- Read-only database access prevents accidental deletions.
- Restricted API scopes prevent the agent from calling endpoints it should not reach.
- Principle of least privilege applied to tool permissions.

**4. Output Filtering**: Analyze the generated response *before* showing it to the user.
- A secondary reviewer model checks for toxicity, bias, and PII.
- If the check fails: block the response and regenerate, or substitute a safe fallback.
- CrewAI pattern: a Policy Enforcer agent outputs `{"compliance_status": "compliant/non-compliant", "evaluation_summary": "..."}`, validated by Pydantic schema before being acted upon.

**5. Human-in-the-Loop for irreversible actions** (see Chapter 13).

### What Breaks
- **Layer bypass**: A user who defeats the input filter with an indirect jailbreak still faces the behavioral constraints, the tool restrictions, and the output filter. No single bypass defeats all layers.
- **Guardrail latency**: Each safety layer adds a model call. Four layers = 4× the baseline latency. Use fast models (Flash-tier) for guardrails specifically.
- **Over-restriction**: Guardrails set too aggressively block legitimate requests. The cost of false positives (rejected valid requests) must be weighed against the cost of false negatives (passed harmful requests).
- **Pydantic validation dependency**: If the Policy Enforcer agent outputs malformed JSON, Pydantic schema validation fails and the pipeline stalls. The enforcer must be prompted with strict output format requirements.

---

## Chapter 19: Evaluation and Monitoring

### The Problem
AI agents are non-deterministic. The same prompt produces different outputs on different runs. Traditional software testing (`assert output == expected`) fails because two correct answers may be different strings. Without a methodology for evaluating probabilistic, open-ended outputs, you cannot know whether your agent is improving or regressing.

### The Core Insight
Evaluate the trajectory, not just the output. An agent that gets the right answer via a hallucinated reasoning chain is not a reliable agent. Metrics must cover: correctness of the final answer, efficiency of the path taken, and operational costs incurred.

### The Mechanics
**Four evaluation dimensions**:

1. **Response Accuracy**: Does the output match the "golden answer"? Measured via LLM-as-a-Judge — a stronger model evaluates whether the agent's answer is semantically equivalent to the expected answer. Binary string matching is insufficient.

2. **Latency**: Total time from query to final output, across all tool calls and reflection loops.

3. **Token Usage (Cost)**: Tokens consumed per task. Unmonitored, agents can accumulate costs that make the product economically unviable.

4. **Trajectory Analysis**: Examine the sequence of `Thoughts → Actions → Observations`. Did the agent use tools correctly? Did it take unnecessary loops? Did it hallucinate intermediate steps? The path matters, not just the destination.

**Testing methodologies**:

- **Golden Datasets (Evalsets)**: Curated input/expected-outcome pairs. Agent is run against the full set; pass rate is the primary quality metric.
- **A/B Testing**: Two agent versions run against real traffic simultaneously. Version with higher metric scores wins.
- **Drift Detection**: Continuous monitoring for performance degradation. If model provider updates a base model or external data sources change, agent behavior can degrade silently.

Google ADK structures evaluation as: Unit Tests (individual tools in isolation) + Integration Tests (full workflow against evalset files).

### What Breaks
- **Evalset staleness**: A golden dataset curated at launch stops being representative as user behavior evolves. Evalsets must be continuously updated with real production inputs.
- **LLM-as-Judge bias**: The judge model has its own preferences and biases. If the judge model and the agent model are from the same training lineage, the judge will systematically favor the agent's style regardless of correctness.
- **Trajectory gaming**: An agent can be optimized to produce good trajectories on the evalset while failing on novel real-world inputs. Evalset diversity is a requirement, not an afterthought.
- **Vanity metrics**: Monitoring only token counts or response time misses quality degradation. All four dimensions must be monitored simultaneously.

---

## Chapter 20: Prioritisation

### The Problem
An agent that executes tasks in the order they were received will complete low-value work before high-value work whenever the queue is non-empty. In a dynamic environment where new tasks arrive continuously, this produces the wrong outcomes even when every individual task is handled correctly.

### The Core Insight
Priority is a function of urgency, importance, and dependency — not arrival time. Before executing, rank. Continuously re-rank as new information arrives.

### The Mechanics
**Three prioritization criteria**:
- **Urgency**: How time-sensitive is this? (SLA deadline, user frustration signal)
- **Importance**: How much does this move the primary objective?
- **Dependency**: Does this task unblock other tasks?

**Two levels of operation**:
- **Strategic**: What overall goals should the agent pursue first?
- **Tactical**: In the current execution step, which tool call or action has highest priority?

**Dynamic re-prioritization**: Priority is not set once at task intake. New information (a tool call returns unexpected data; a user sends an urgent follow-up) triggers a re-ranking of the remaining task queue.

### What Breaks
- **Priority inversion**: A low-priority task that blocks a high-priority task must be treated as high-priority. Static ranking that ignores dependencies causes deadlocks.
- **Starvation**: Low-priority tasks that are never prioritized above urgent ones are never completed. Aging (gradually increasing the priority of old tasks) prevents indefinite starvation.
- **Reprioritization overhead**: Continuously re-ranking a large task queue using an LLM is expensive. Use lightweight heuristics for frequent re-ranking; reserve LLM-based ranking for initial priority assessment.

---

## Chapter 21: Exploration and Discovery

### The Problem
Standard agents optimize within a known solution space. They retrieve existing answers, combine known patterns, and execute established workflows. For genuinely novel problems — new scientific hypotheses, undiscovered market strategies — there is no existing answer to retrieve. Optimization over the known fails when the answer must be invented.

### The Core Insight
The agent must be able to treat its own ignorance as a starting point, not a terminal condition. Exploration means generating and testing hypotheses, not retrieving and applying facts. The loop is: hypothesize → experiment → observe → refine.

### The Mechanics
**Discovery workflow**:
1. **Hypothesis Generation**: Agent scans existing knowledge to identify gaps or anomalies. Formulates a candidate claim.
2. **Experiment Design**: Translates the hypothesis into a testable procedure (code to run, simulation to execute, literature to verify against).
3. **Analysis**: Evaluates experimental output against the hypothesis. Validated, refuted, or inconclusive.
4. **Evolution**: Updates the knowledge base and generates the next hypothesis.

**Agent Laboratory framework** (multi-agent implementation):
- **Researcher Agent**: Literature review and hypothesis generation.
- **Software Engineer Agent**: Data preparation code.
- **ML Engineer Agent**: Model training code.
- **AgentRxiv**: A shared repository where agents deposit and retrieve research artifacts, enabling cumulative building across sessions.

Use case: Google Co-Scientist applies this pattern to autonomously design chemistry and biology experiments.

### What Breaks
- **Hallucinated experiments**: An agent that generates experiments it cannot actually run (requires physical lab equipment, access to restricted data) produces hypotheses that are untestable in its operating environment.
- **Circular discovery**: Without a mechanism to track which hypotheses have already been tested, the agent re-discovers the same finding repeatedly.
- **Quality control at scale**: Automated hypothesis generation without human review can flood AgentRxiv-type repositories with low-quality findings that corrupt future retrieval.
- **Exploitation-exploration balance**: An agent that always explores never delivers useful output. An agent that always exploits never discovers anything new. Balancing requires an explicit exploration budget.
