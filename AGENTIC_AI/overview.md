# Agentic AI — Overview and Core Concepts

## What is an AI agent?

An **AI agent** is a system that:

1. **Perceives** context (user message, history, tool results).
2. **Reasons** about what to do next (planning, decomposition).
3. **Acts** by calling tools (APIs, code, search, RAG) or producing a final answer.
4. **Observes** the results and repeats until the goal is met or a stopping condition is reached.

Unlike a single-call chatbot, an agent operates in a **loop** and can use **external tools** and **memory**.

---

## Core components

| Component | Role |
|-----------|------|
| **LLM (brain)** | Reasoning, planning, deciding which tool to call and with what arguments; generating final answer. |
| **Tools** | Actions the agent can take: search, code execution, API calls, RAG retrieval, calculators, etc. |
| **Memory** | Short-term (current conversation), long-term (past sessions or facts), and sometimes episodic (key events). |
| **Orchestrator** | Runs the loop: prompt → LLM → parse output → execute tool or return answer → append to context → repeat. |

---

## Agent workflow (simplified)

```
┌─────────────────────────────────────────────────────────────────┐
│  User: "What's the weather in Paris and should I take an umbrella?"  │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│  Agent loop:                                                      │
│  1. LLM reasons: "I need weather for Paris"                       │
│  2. Tool call: get_weather(city="Paris")                         │
│  3. Observation: "22°C, rain expected"                           │
│  4. LLM reasons: "Rain expected → recommend umbrella"             │
│  5. Final answer: "22°C with rain; yes, take an umbrella."        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Reasoning patterns

- **ReAct** (Reason + Act): Interleave reasoning steps and tool calls in natural language; model outputs "Thought", "Action", "Observation" in sequence.
- **Plan-and-execute**: First produce a plan (steps), then execute each step with tools and possibly refine the plan.
- **Chain-of-thought (CoT)**: Encourage step-by-step reasoning in the response before committing to a tool or answer; improves reliability.

These can be combined: e.g. plan → ReAct over each step → final answer.

---

## Strengths and limitations

**Strengths**

- Can use up-to-date and private data (tools, RAG).
- Multi-step tasks (research, coding, analysis) are natural.
- Same LLM can be reused with different tools and prompts.

**Limitations**

- Latency and cost increase with more steps.
- Tool use and parsing can fail (hallucinated tools, bad arguments).
- Need guardrails and limits (e.g. which tools, how many steps).

---

## Quick revision

- **Agent** = LLM + tools + memory + loop (reason → act → observe).
- **Orchestrator** runs the loop and parses LLM output into tool calls or answers.
- **ReAct** and **plan-and-execute** are common reasoning patterns.
- Agents enable coding, search, API use, and RAG in one system.
