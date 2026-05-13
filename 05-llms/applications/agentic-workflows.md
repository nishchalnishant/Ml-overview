# Agentic Workflows

An agentic workflow turns an LLM from a one-shot text generator into a system that observes state, decides actions, calls tools, and loops until a goal is met. The LLM is the reasoning engine; everything else — tools, memory, routing, stopping rules — is engineering.

---

## 1. The Core Loop

```
Observe → Think → Act → Observe → ... → Stop
```

Concretely:

1. **Input:** user request + context (conversation history, tool results, memory)
2. **Plan:** LLM reasons about what to do next (Chain-of-Thought or structured output)
3. **Tool call:** LLM invokes a function with structured arguments
4. **Observation:** tool result appended to context
5. **Repeat** until the LLM emits a final answer or a stopping condition triggers

---

## 2. Prompting Patterns

### Chain-of-Thought (CoT)

Encourage the model to reason step by step before answering:

```
Q: If a store has 3 packs of 8 apples and sells 5, how many remain?
A: Let me think step by step.
   3 packs × 8 apples = 24 apples total.
   24 - 5 = 19 apples remain.
```

**Zero-shot CoT:** append "Let's think step by step." to the prompt. Works surprisingly well on GPT-4-class models.

**Few-shot CoT:** provide 2–5 worked examples in the prompt. Better for smaller models or domain-specific reasoning.

**Limitations:** reasoning traces can look correct while being wrong. High-confidence incorrect reasoning is dangerous in production.

### ReAct (Reasoning + Acting)

Interleaves chain-of-thought with tool calls:

```
Thought: I need to find the current stock price of AAPL.
Action: search_web(query="AAPL stock price today")
Observation: AAPL is trading at $189.42 as of 2:30 PM EST.
Thought: Now I can answer the user's question.
Answer: AAPL is currently trading at $189.42.
```

ReAct reduces hallucination on factual tasks by grounding reasoning in tool observations. Key paper: Yao et al. (2022).

### Reflection / Self-Critique

After generating an answer, prompt the model to critique and revise it:

```python
# First pass
answer = llm.complete(f"Solve: {problem}")

# Reflection
critique = llm.complete(f"""
Problem: {problem}
Proposed solution: {answer}
Identify any errors or gaps in this solution.
""")

# Revision
final_answer = llm.complete(f"""
Problem: {problem}
Solution: {answer}
Critique: {critique}
Provide a corrected final answer.
""")
```

Useful for code generation, math, and multi-step tasks. Costs 2–3× more tokens.

### Tree of Thought (ToT)

Explore multiple reasoning branches in parallel and select the best path. Useful for search-like problems. Expensive — generally reserved for hard reasoning tasks where accuracy justifies the cost.

---

## 3. Tool Use

Tool use is the most reliable way to move from "fancy autocomplete" to a useful system.

### Function Calling (OpenAI-style)

Define tools as JSON schemas; the model outputs structured call arguments:

```python
import openai
import json

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["city"]
            }
        }
    }
]

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What's the weather in London?"}],
    tools=tools,
    tool_choice="auto"
)

if response.choices[0].message.tool_calls:
    call = response.choices[0].message.tool_calls[0]
    args = json.loads(call.function.arguments)
    result = get_weather(**args)   # your actual function
```

### Tool Design Principles

1. **Narrow scope:** each tool should do one thing well
2. **Idempotency:** safe to call multiple times (avoid side effects when possible)
3. **Structured outputs:** return JSON, not prose — the model needs to parse results
4. **Error messages:** be specific; "City not found" helps the model retry; "Error 400" does not
5. **Rate limits and timeouts:** always set hard limits — infinite tool loops will happen

---

## 4. Memory Systems

| Memory type | What it stores | Implementation |
| :--- | :--- | :--- |
| **In-context** | Recent conversation turns | Sliding window of messages |
| **External short-term** | Session state across API calls | Redis, in-memory store |
| **Long-term semantic** | Facts, preferences, past interactions | Vector database (retrieval by similarity) |
| **Structured state** | Task progress, workflow variables | JSON / database rows |

### Semantic Memory with Vector Store

```python
from langchain.memory import VectorStoreRetrieverMemory
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

vectorstore = Chroma(embedding_function=OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
memory = VectorStoreRetrieverMemory(retriever=retriever)

# Save an interaction
memory.save_context(
    {"input": "My budget for the project is $10k"},
    {"output": "Noted. I'll keep the $10k budget in mind."}
)

# Retrieve relevant context before responding
relevant = memory.load_memory_variables({"prompt": "How much can I spend?"})
```

---

## 5. Multi-Agent Patterns

### Orchestrator + Subagent

A central orchestrator agent decomposes tasks and delegates to specialized subagents:

```
User request
    │
    ▼
Orchestrator (GPT-4o)
    ├── Research Agent  → web search, doc retrieval
    ├── Code Agent      → code execution, testing
    └── Writer Agent    → drafts, edits, formats
```

**When to use:** tasks with distinct specializations that would overload a single context window.

**Cost:** 2–4× more LLM calls, more latency, more coordination complexity.

### Supervisor / Worker (LangGraph style)

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    next_agent: str

def supervisor(state):
    # Decides which worker to call next, or END
    ...

def researcher(state):
    # Retrieves information
    ...

def writer(state):
    # Generates final answer
    ...

workflow = StateGraph(AgentState)
workflow.add_node("supervisor", supervisor)
workflow.add_node("researcher", researcher)
workflow.add_node("writer", writer)
workflow.set_entry_point("supervisor")
workflow.add_conditional_edges("supervisor", lambda s: s["next_agent"])
workflow.add_edge("researcher", "supervisor")
workflow.add_edge("writer", END)

app = workflow.compile()
```

### Debate / Critic Pattern

Two agents argue opposite sides of a question; a judge synthesizes. Useful for decisions where confirmation bias is a risk. Expensive — use sparingly.

---

## 6. Routing

Not every query needs the same workflow. A router classifies the request and dispatches to the appropriate path:

```python
from enum import Enum

class Route(str, Enum):
    RAG = "rag"           # needs document retrieval
    CODE = "code"         # needs code execution
    SEARCH = "search"     # needs web search
    DIRECT = "direct"     # model can answer from weights

def route_query(query: str) -> Route:
    response = llm.complete(f"""
    Classify this query into one routing category:
    - rag: needs information from internal documents
    - code: needs code to be written or executed
    - search: needs current web information
    - direct: general question answerable from training knowledge

    Query: {query}
    Category:""")
    return Route(response.strip().lower())
```

Routing is one of the cheapest optimizations — a fast classifier (even a fine-tuned small model) can save significant cost by avoiding expensive retrieval or tool calls.

---

## 7. Guardrails and Safety

Every production agentic system needs hard limits:

| Guardrail | Why it matters | Implementation |
| :--- | :--- | :--- |
| **Max steps** | Prevent infinite loops | Counter in state; raise after N steps |
| **Timeout** | Prevent hung agents | Per-tool and per-run timeouts |
| **Output validation** | Catch malformed tool calls | Pydantic schemas on tool outputs |
| **Content filters** | Prevent harmful outputs | Input/output classifiers (Llama Guard, etc.) |
| **Human-in-the-loop** | High-stakes decisions | Pause and await approval before irreversible actions |
| **Sandboxed execution** | Code runs don't compromise the host | Docker containers, E2B sandboxes |

```python
MAX_ITERATIONS = 15

for step in range(MAX_ITERATIONS):
    result = agent.step(state)
    if result.is_final:
        return result.answer
    state = result.next_state

# If we exit the loop without finishing:
return fallback_response("Agent exceeded step limit. Please rephrase your request.")
```

---

## 8. Observability

Multi-step systems fail at multi-step boundaries. You need traces, not just logs.

**What to instrument:**
- Every LLM call: model, prompt tokens, completion tokens, latency
- Every tool call: name, inputs, output, latency, success/fail
- Full agent trace: step sequence, decisions, final answer
- Cost: token cost per run

**Tooling options:** LangSmith, Weights & Biases Weave, Phoenix (Arize), custom OpenTelemetry traces.

```python
import langsmith

@langsmith.traceable(run_type="chain")
def run_agent(query: str):
    # All nested LLM and tool calls are automatically traced
    return agent.run(query)
```

---

## 9. Production Failure Modes

| Failure | Cause | Fix |
| :--- | :--- | :--- |
| **Infinite loops** | No stopping condition, weak goal detection | Max step limit + explicit termination prompt |
| **Tool hallucination** | Model invents tool names or arguments | Strict function calling schema validation |
| **Context overflow** | Long tool outputs fill context window | Summarize observations before adding to context |
| **Stale memory** | Retrieved memories are outdated | TTL on memory entries, freshness scoring |
| **Cascade failures** | One bad tool call poisons downstream steps | Tool output validation, graceful error handling |
| **Prompt injection** | Tool output contains adversarial instructions | Sanitize all external inputs before adding to context |

---

## 10. Frameworks Overview

| Framework | Strengths | Best for |
| :--- | :--- | :--- |
| **LangChain / LangGraph** | Comprehensive, large ecosystem | Complex multi-step workflows with state |
| **LlamaIndex** | Strong retrieval, document ingestion | RAG-heavy agentic systems |
| **AutoGen (Microsoft)** | Multi-agent conversations | Research and multi-agent experiments |
| **CrewAI** | Role-based agent definition | Team-style multi-agent tasks |
| **Direct API** | Full control, no abstractions | Simple tool-use, production-critical paths |

> [!TIP]
> **Interview structure:** For any agentic system question — explain (1) the loop: observe/think/act/stop, (2) how tools work, (3) memory strategy, (4) failure modes and guardrails. The last two separate candidates who have shipped agents from those who have only read about them.
