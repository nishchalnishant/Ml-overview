# Agentic AI

**Agentic AI** refers to systems where an **autonomous agent** uses reasoning, planning, and **tools** (APIs, code, search, retrieval) to accomplish multi-step goals, rather than answering in a single turn.

This section covers:

- [Overview and core concepts](overview.md)
- [Autonomous agents and reasoning](autonomous-agents-reasoning.md)
- [Tool-using agents](tool-using-agents.md)
- [Multi-agent systems](multi-agent-systems.md)
- [Memory and orchestration](memory-and-orchestration.md)
- [Frameworks and implementation](frameworks-and-implementation.md)

---

## Why agents?

- **Single-turn LLMs** are limited to one response; they cannot run code, call APIs, or iterate based on results.
- **Agents** loop: reason → decide action (e.g. tool call) → observe result → repeat until the task is done or a final answer is produced.
- Use cases: coding assistants, research, data analysis, customer workflows, document handling, and any task that requires external data or tools.

---

## High-level architecture

```
User goal → Agent loop:
  1. Reason / Plan (LLM)
  2. Choose action (tool call or answer)
  3. Execute tool / get observation
  4. Update memory / state
  5. Repeat or return answer
```

See [Overview and core concepts](overview.md) for diagrams and [Frameworks and implementation](frameworks-and-implementation.md) for code-level patterns.
