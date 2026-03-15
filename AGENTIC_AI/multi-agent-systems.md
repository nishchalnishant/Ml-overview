# Multi-Agent Systems

In **multi-agent** setups, several agents (or roles) collaborate: each can have its own LLM, tools, and task. An orchestrator or a "manager" agent assigns work and aggregates results.

---

## When to use multiple agents

- **Specialization**: One agent for search, one for code, one for summarization.
- **Scale**: Break a big task into sub-tasks handled by different agents.
- **Debate / critique**: One agent proposes, another critiques; iterate to improve quality.
- **Roles**: Simulate user, reviewer, coder, tester in a pipeline.

---

## Simple architecture

```
                    ┌─────────────────┐
                    │  Orchestrator   │
                    │  (or Manager)   │
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         ▼                   ▼                   ▼
   ┌───────────┐       ┌───────────┐       ┌───────────┐
   │  Agent A  │       │  Agent B  │       │  Agent C  │
   │  (search) │       │  (code)   │       │  (review) │
   └───────────┘       └───────────┘       └───────────┘
```

- Orchestrator receives the user request, decomposes it, and assigns sub-tasks to agents.
- Each agent may call tools and return a result.
- Orchestrator aggregates and may send follow-up tasks or return the final answer.

---

## Patterns

- **Sequential**: Agent 1 → output → Agent 2 → … → final output.
- **Parallel**: Multiple agents work on different sub-tasks; results merged (e.g. summarizer).
- **Debate**: Two (or more) agents argue or critique; manager decides or asks for another round.
- **Hierarchical**: Manager delegates to sub-managers, which delegate to workers.

---

## Challenges

- **Latency and cost**: More agents and rounds mean more LLM calls.
- **Consistency**: Ensure shared context (e.g. task description, constraints) and clear handoff format.
- **Failure handling**: Define what happens when one agent fails or times out.

---

## Quick revision

- **Multi-agent** = several agents (possibly with different tools/roles) coordinated by an orchestrator or manager.
- **Use**: specialization, task decomposition, debate, role-based pipelines.
- **Design**: clear interfaces, shared context, and policies for failure and aggregation.
