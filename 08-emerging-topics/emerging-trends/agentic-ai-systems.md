---
module: Emerging Topics
topic: Emerging Trends
subtopic: Agentic Ai Systems
status: unread
tags: [emergingtopics, ml, emerging-trends-agentic-ai-sys]
---
# Agentic AI Systems

> *Snapshot: June 2026 — frontier topic, moves fast. Treat as a current-state map, not settled canon.*

How AI agents actually work in production — from the core loop and memory architecture to multi-agent orchestration, tool use, and the failure modes that kill real deployments.

---

## 1. What Makes Something an Agent

An LLM call is not an agent. An agent is a system that:
1. **Perceives** an environment (reads files, calls APIs, observes results)
2. **Reasons** about what to do next (LLM generates a plan or action)
3. **Acts** to change state (executes tools with real-world effects)
4. **Maintains state** across multiple action steps (memory)
5. **Has a stopping condition** (terminates when goal is achieved or fails)

The minimum viable agent is a for-loop:

```python
def agent_loop(task: str, tools: list[Tool], max_steps: int = 30) -> str:
    memory = [{"role": "user", "content": task}]
    
    for step in range(max_steps):
        response = llm.generate(memory + tool_descriptions)
        memory.append({"role": "assistant", "content": response})
        
        if is_final_answer(response):
            return extract_answer(response)
        
        tool_call = parse_tool_call(response)
        if tool_call:
            result = execute_tool(tool_call)  # real-world action
            memory.append({"role": "tool", "content": result})
    
    return "Max steps reached without answer"
```

Everything else in agentic AI is solving the problems that arise in this loop.

---

## 2. The ReAct Pattern

ReAct (Reasoning + Acting) is the foundational prompting pattern for agents. The model interleaves:
- **Thought**: what does it need and why
- **Action**: tool call with structured parameters
- **Observation**: tool result (grounded in reality)

```
Thought: I need to find the current price of NVDA stock.
Action: web_search(query="NVDA stock price today")
Observation: NVDA is trading at $875.40 as of 3:15 PM EST.

Thought: Now I need to compare this to the 52-week high.
Action: web_search(query="NVDA 52 week high 2024")
Observation: NVDA 52-week high: $974.00

Thought: NVDA is 10.1% below its 52-week high. I have enough information to answer.
Answer: NVDA is currently at $875.40, which is approximately 10.1% below its 52-week high of $974.00.
```

**Why ReAct works**: the Observation grounds the next Thought in actual retrieved data, not the model's parametric memory. Each reasoning step is conditioned on verified information.

---

## 3. Memory Architecture

Memory is the hardest engineering problem in agentic systems. Four types:

### In-Context Memory (Working Memory)
The current conversation window. Fast, directly accessible, but bounded by context length. At 128K context with ReAct traces averaging 500 tokens/step: ~256 steps before context is full.

**Context compression**: periodically summarize older steps and replace them with a compressed summary. The full trace is stored externally; the context window keeps only the recent + summary.

### External Memory (Long-Term Memory)
Vector store of facts, experiences, or tool results. Retrieved via semantic search when relevant.

```python
class AgentMemory:
    def __init__(self):
        self.vector_store = VectorStore()   # semantic retrieval
        self.episodic_store = KeyValueDB()  # past task outcomes
    
    def store(self, observation: str, context: str):
        embedding = embed(observation)
        self.vector_store.upsert(embedding, {"text": observation, "context": context})
    
    def retrieve(self, query: str, k: int = 5) -> list[str]:
        query_embedding = embed(query)
        results = self.vector_store.search(query_embedding, k=k)
        return [r["text"] for r in results]
```

### Tool/State Memory (Procedural)
The actual state of external systems the agent interacts with: files written, API calls made, database rows created. This state persists independently of the agent's context window.

### Episodic Memory (Experience)
Records of past tasks: what worked, what failed, how long tasks took. Allows agents to avoid repeating mistakes across sessions.

> For advanced retrieval patterns within agentic pipelines — including Self-RAG (the agent decides when to retrieve and critiques its own retrieved passages), Agentic RAG (multi-step retrieval loops), RAPTOR (hierarchical document summarization), MemGPT (virtual context management), and Mem0 (user-level persistent memory) — see [advanced-rag-and-memory.md](advanced-rag-and-memory.md). This section covers the memory architecture abstractions; that file covers the retrieval system internals.

---

## 4. Tool Design Principles

Tools are the most critical agent design decision. Bad tool design is the #1 cause of agent failures.

**Principle 1: Tools must have clear, unambiguous descriptions.**
The LLM reads tool descriptions and decides which to call. Ambiguous descriptions cause wrong tool selection.

```python
# Bad: ambiguous
@tool
def search(query: str) -> str:
    """Search for information."""  # Search WHERE? Returns WHAT?
    ...

# Good: specific
@tool  
def web_search(query: str) -> str:
    """Search the web using Google and return the top 3 results as text snippets.
    Use this to find current information, news, prices, or facts not in your training data.
    Returns: JSON with keys 'title', 'url', 'snippet' for each result."""
    ...
```

**Principle 2: Tools should be atomic and composable.**
Each tool does one thing. The agent composes them. A tool that "searches the web AND saves the result to a file" breaks composability and makes error handling impossible.

**Principle 3: Tools must return structured, parseable output.**
JSON is better than prose. Consistent schema is better than variable output.

**Principle 4: Tools must be idempotent where possible.**
An agent might retry a tool if it gets confused. Sending an email twice is a disaster. Writing a file twice is fine.

**Principle 5: Tool errors must be informative.**
```python
# Bad: agent doesn't know what to do next
raise Exception("Error")

# Good: agent can reason about the failure and try a different approach
return {
    "success": False,
    "error": "Rate limit exceeded",
    "retry_after_seconds": 60,
    "suggestion": "Wait 60 seconds then retry, or use a different data source"
}
```

---

## 5. Planning Strategies

### Zero-Shot Planning
Let the LLM decide what to do at each step. Simple, flexible, but inconsistent for complex tasks.

### ReWOO (Reasoning Without Observation)
For tasks where tool results are predictable, plan all steps upfront without interleaving observations. Then execute the plan, substituting actual results for placeholders. More efficient (fewer LLM calls) but fails when any step produces unexpected output.

### Tree of Thoughts (ToT)
Generate multiple possible next steps, evaluate each, and pursue the most promising branch. More expensive but handles problems where the right path isn't obvious.

```python
def tree_of_thoughts(problem, depth=3, branching_factor=3):
    """Explore multiple reasoning paths, keep the best."""
    if depth == 0:
        return evaluate(problem)
    
    # Generate branching_factor candidate next steps
    candidates = [llm_generate_step(problem) for _ in range(branching_factor)]
    
    # Evaluate each candidate
    scores = [llm_evaluate(problem, c) for c in candidates]
    
    # Recurse on best candidate
    best = candidates[scores.index(max(scores))]
    return tree_of_thoughts(problem + best, depth - 1, branching_factor)
```

### Plan-and-Execute
Generate a full plan first, then execute step by step with an executor agent. The planner and executor can be different models (cheap planner, capable executor).

---

## 6. Multi-Agent Systems

Single agents fail on tasks that require parallelism, specialization, or work exceeding a context window. Multi-agent systems coordinate multiple agents.

### Orchestrator-Worker Pattern
```
Orchestrator LLM
    │
    ├── Research Agent (web search, summarize)
    ├── Code Agent (write + test code)
    ├── Review Agent (critique code, find bugs)
    └── File Agent (read/write filesystem)
```

The orchestrator receives the high-level task, decomposes it into subtasks, assigns each to a specialized agent, collects results, and assembles the final output.

### Supervisor Pattern
Similar to orchestrator-worker but the supervisor can override worker decisions and reassign tasks.

### Peer-to-Peer (Multi-Agent Debate)
Multiple agents work on the same problem independently, then debate their answers. Used in Constitutional AI and some reasoning systems to reduce hallucination.

```python
def multi_agent_debate(question: str, n_agents: int = 3, rounds: int = 2) -> str:
    # Round 0: each agent answers independently
    answers = [agent.answer(question) for agent in agents]
    
    for round in range(rounds):
        # Each agent sees all other agents' answers and updates their view
        for i, agent in enumerate(agents):
            others_answers = [a for j, a in enumerate(answers) if j != i]
            answers[i] = agent.update(question, others_answers)
    
    # Aggregate: majority vote or meta-judge
    return aggregate(answers)
```

---

## 7. Failure Modes and How to Prevent Them

### Hallucinated Tool Calls
Model invents tool names or parameters that don't exist. Fix: strict schema validation — only allow calls with exact tool name matches and validated parameter types. Return structured error if schema doesn't match.

### Infinite Loops
Agent loops on the same tool call repeatedly. Fix: track action history and abort if the same (tool, params) appears twice within 10 steps.

### Context Window Overflow
Long tasks fill the context window and the model loses track of the goal. Fix: maintain a "task summary" that's prepended to every context; compress older tool results into summaries.

### Irreversible Actions
Agent deletes a file, sends an email, deploys code — without user confirmation. Fix: categorize actions as reversible/irreversible; require human-in-the-loop confirmation before irreversible actions.

```python
IRREVERSIBLE_TOOLS = {"send_email", "delete_file", "deploy_to_production", "make_payment"}

def execute_tool(tool_name: str, params: dict) -> str:
    if tool_name in IRREVERSIBLE_TOOLS:
        confirmation = request_human_confirmation(tool_name, params)
        if not confirmation:
            return "Action cancelled by user"
    return TOOL_REGISTRY[tool_name](**params)
```

### Prompt Injection from Tool Results
Retrieved web content or document content contains "Ignore previous instructions." Fix: structural separation — wrap all tool results in a designated XML block and instruct the model to treat them as untrusted data.

### Cost Explosion
Unconstrained agents can make hundreds of tool calls and generate massive amounts of tokens. Fix: hard limits on max_steps, max_tokens, and total cost per task.

---

## 8. Evaluation Framework for Agents

Agents are harder to evaluate than static models — correctness depends on the trajectory, not just the final answer.

**Task completion rate**: did the agent complete the task? Binary for simple tasks, graded for complex.

**Trajectory efficiency**: did the agent use the minimum necessary steps? Extra steps = higher cost.

**Tool call accuracy**: did the agent call the right tools in the right order? Measured against a reference trajectory.

**Error recovery rate**: when a tool returns an error, does the agent recover gracefully?

```python
class AgentEvaluator:
    def evaluate(self, task, agent_trajectory, reference_trajectory, final_answer):
        return {
            "task_success": self.check_task_completion(task, final_answer),
            "efficiency": len(reference_trajectory) / len(agent_trajectory),
            "tool_precision": self.compute_tool_precision(agent_trajectory, reference_trajectory),
            "error_recovery_rate": self.count_recovered_errors(agent_trajectory),
            "hallucinated_tool_calls": self.count_invalid_calls(agent_trajectory),
        }
```

**Benchmarks**: WebArena (web browser tasks), SWE-bench (GitHub issue fixing), GAIA (general assistant tasks), OSWorld (desktop computer tasks).

---

## 9. Production Patterns

### Human-in-the-Loop
For high-stakes tasks, the agent pauses at decision points and requests human confirmation. Reduces autonomy but dramatically reduces catastrophic failures.

### Sandboxed Execution
Run agent actions in isolated environments (Docker containers for code, staging environments for APIs). Validate results before applying to production.

### Checkpointing
Save agent state after each step. If the agent fails mid-task, resume from the last checkpoint rather than restarting.

### Cost Budgeting
```python
class BudgetedAgent:
    def __init__(self, max_cost_usd: float):
        self.budget = max_cost_usd
        self.spent = 0.0
    
    def execute_step(self, step):
        step_cost = estimate_cost(step)
        if self.spent + step_cost > self.budget:
            return self.graceful_termination("Budget exhausted")
        result = execute(step)
        self.spent += actual_cost(result)
        return result
```

---

## 10. Agent Framework Ecosystem

Everything in §1–9 describes agent *patterns*. In practice, teams rarely implement the loop from scratch — they build on a framework that handles orchestration, tool-calling boilerplate, and state management. Framework choice is a real architectural decision, not just tooling preference.

### 10.1 MCP (Model Context Protocol)
MCP is not a framework — it's a standardized protocol (originated at Anthropic, now widely adopted) for connecting LLM applications to external tools and data sources. Before MCP, every agent framework had its own bespoke tool-integration format; an integration built for LangChain couldn't be reused in AutoGen. MCP defines a common client-server interface: an **MCP server** exposes tools/resources/prompts over a standard schema, and any **MCP client** (Claude Desktop, an IDE, a custom agent) can discover and call them without framework-specific glue code.

```
Agent/Host (MCP client) ──JSON-RPC──> MCP Server ──> Tool/API/DB/filesystem
```
**Why it matters:** MCP turns "N frameworks × M tools" integration work into "N clients + M servers," each written once. It's the closest thing agentic AI has to a USB-C standard for tool access — the same filesystem or database MCP server works whether the calling agent is built on LangGraph, a custom loop, or a chat client.

### 10.2 Framework Comparison

| Framework | Core abstraction | Best for | Trade-off |
|---|---|---|---|
| **LangChain** | Chains and Agents composed from modular components (LLMs, retrievers, tools, memory) | Rapid prototyping, broad ecosystem/integration coverage | Heavy abstraction layers; harder to debug or customize control flow at scale |
| **LangGraph** | Agent workflows as an explicit state graph (nodes = steps, edges = transitions, can be cyclic) | Complex, stateful, multi-step agents needing explicit control flow (loops, conditional branches, human-in-the-loop pauses) | More upfront design work than a simple chain; steeper learning curve |
| **LlamaIndex** | Data-framework-first: indices, retrievers, query engines, with agent capabilities layered on top | RAG-centric agents where retrieval quality is the primary concern | Less mature multi-agent orchestration than LangGraph/CrewAI |
| **CrewAI** | "Crews" of role-based agents (e.g. researcher, writer, reviewer) collaborating on a shared task | Multi-agent workflows modeled on human team roles, fast to set up | Less fine-grained control over inter-agent communication than LangGraph |
| **AutoGen** (Microsoft) | Conversable agents that communicate via structured multi-agent dialogue | Research-oriented multi-agent conversation patterns, code-generation agents | More conversation-centric than task/graph-centric; orchestration logic lives in dialogue patterns rather than an explicit graph |

### 10.3 Choosing Among Them
- **Need explicit, debuggable control flow with cycles and conditionals** (retry loops, human approval gates) → LangGraph. Its state-graph model maps directly onto §5 (Planning Strategies) and §7 (Failure Modes) — every failure-mode fix (dedup checks, budget cutoffs, checkpointing) is a graph node/edge.
- **Need to move fast with broad tool/vector-store integrations already built** → LangChain (or LlamaIndex if the task is retrieval-heavy).
- **Need several specialized agents playing distinct roles with minimal orchestration code** → CrewAI.
- **Need agents primarily reasoning through structured back-and-forth dialogue** (e.g. code review between a coder agent and a critic agent) → AutoGen.
- **Need agent-to-tool integration decoupled from any specific framework** → build/consume MCP servers regardless of which orchestration framework sits on top; MCP and these frameworks are not mutually exclusive (LangGraph and CrewAI agents can both call tools over MCP).

### 10.4 Common Failure Modes Specific to Frameworks
- **Framework lock-in**: bespoke tool/memory formats make migrating between frameworks expensive — a reason MCP adoption is accelerating (§10.1 decouples the tool layer from the orchestration layer).
- **Debugging opacity**: heavily abstracted chains (LangChain's higher-level constructs especially) can obscure exactly which prompt was sent at which step — pair with tracing (LangSmith, or framework-agnostic tools) rather than relying on print debugging.
- **Version churn**: this ecosystem moves fast; pin framework versions and test upgrades against a fixed eval set (§8, Evaluation Framework for Agents) before rolling forward.

---

## Canonical Interview Q&As

**Q: What are the key differences between a single LLM call and a production agent, and what are the main failure modes?**
A: A single LLM call is stateless, single-step, and cannot take real-world actions. A production agent adds: a for-loop that runs multiple LLM calls sequentially; tools that connect to real systems (web, databases, code executors); memory that persists state across steps; and stopping conditions. The main failure modes are: (1) Hallucinated tool calls — model invents non-existent tools or wrong parameters; fix with strict schema validation; (2) Context overflow — long tasks fill the context window and the model loses track of the goal; fix with context compression and task summary; (3) Infinite loops — agent repeats the same action; fix with action deduplication checks; (4) Irreversible actions without confirmation — agent sends email or deletes data; fix with an irreversible-action allowlist requiring human confirmation; (5) Prompt injection from tool results — malicious content in retrieved data overrides instructions; fix with structural separation of trusted instructions vs untrusted data.

**Q: Design a multi-agent system to automatically review and fix GitHub issues. What are the key design decisions?**
A: Orchestrator-worker architecture: (1) Triage agent reads the issue, classifies it (bug/feature/question), extracts relevant files and error messages. (2) Research agent searches the codebase for relevant code sections using semantic search. (3) Code agent writes the fix in an isolated sandbox, running tests iteratively. (4) Review agent reads the original issue, the proposed fix, and all test results — flags if the fix doesn't address root cause or introduces regressions. (5) Orchestrator decides if the review passes; if not, sends the code agent back with reviewer feedback. Key design decisions: code execution must be sandboxed (Docker container, no internet access, resource limits); each agent gets a scoped context (code agent sees only relevant files, not entire codebase); failures should be informative ("tests failed: test_auth.py:42 TypeError") not opaque; human-in-the-loop before merging (agent opens a PR, human approves). Evaluation: SWE-bench verified — resolved rate is the primary metric.

**Q: How do you handle an agent that keeps failing on a subtask and getting stuck in a retry loop?**
A: Several layered defenses: (1) Exponential backoff with max retries (3-5 attempts, not infinite) — after max retries, emit a structured failure and let the orchestrator decide whether to try an alternative approach or fail gracefully. (2) Action deduplication — if the same (tool, params) pair appears twice in recent history, skip it and try something different (the same action is unlikely to succeed a second time without any change). (3) Diverse retry strategies — don't retry with identical input; modify the approach: different search query, different decomposition of the problem, different tool. (4) Failure escalation — after 3 failed retries on a subtask, escalate to the orchestrator with a summary of what was tried; the orchestrator can reassign to a different worker agent or request human guidance. (5) Timeout budgets — each subtask has a wall-clock time budget; exceeding it triggers graceful failure with partial results rather than running indefinitely.

**Q: When would you choose LangGraph over LangChain, or CrewAI over both?**
A: LangGraph when the workflow needs explicit, debuggable control flow — cycles, conditional branches, human-in-the-loop pauses — modeled as a state graph rather than a linear chain; it maps directly onto the failure-mode fixes in §7 (retry loops, checkpointing) as graph nodes/edges. LangChain when moving fast matters more than fine-grained control flow and the task fits a more linear chain-of-components pattern, leveraging its broad integration ecosystem. CrewAI when the problem naturally decomposes into a handful of role-based agents (researcher, writer, reviewer) collaborating with minimal custom orchestration code. These aren't mutually exclusive with MCP — tool access via MCP servers is a separate, orthogonal decision from which orchestration framework sits on top (§10.1, §10.3).

## Flashcards

**Thought?** #flashcard
what does it need and why

**Action?** #flashcard
tool call with structured parameters

**Observation?** #flashcard
tool result (grounded in reality)
