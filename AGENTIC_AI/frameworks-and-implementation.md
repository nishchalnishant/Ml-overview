# Agent Frameworks and Implementation

Frameworks simplify building agents by handling tool registration, prompt construction, parsing, and the reasoning loop.

---

## Common frameworks

| Framework | Language | Highlights |
|-----------|----------|------------|
| **LangChain / LangGraph** | Python/JS | Chains, agents, tools, RAG; LangGraph for cycles and state. |
| **LlamaIndex** | Python | RAG-first; query engines, agents, tool integration. |
| **CrewAI** | Python | Multi-agent roles and tasks; sequential or hierarchical. |
| **AutoGen** (Microsoft) | Python | Multi-agent conversations and code execution. |
| **Semantic Kernel** | C#, Python | Planners, plugins (tools), and orchestration. |
| **OpenAI Assistants API** | API | Threads, tools, file search; managed loop. |

---

## Minimal agent loop (Python pseudocode)

```python
from openai import OpenAI

client = OpenAI()
tools = [{"name": "get_weather", "description": "...", "parameters": {...}}]
messages = [{"role": "user", "content": "Weather in Paris?"}]

while True:
    resp = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )
    msg = resp.choices[0].message
    if not msg.tool_calls:
        print(msg.content)
        break
    for tc in msg.tool_calls:
        result = run_tool(tc.function.name, tc.function.arguments)
        messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
    messages.append(msg)  # assistant message with tool_calls
```

---

## Building a simple RAG + tool agent

1. **Tools**: Register a retrieval tool (e.g. query vector store, return top-k chunks) plus any other tools (search, code).
2. **Prompt**: System message explains the tools and that the agent should retrieve when it needs internal docs.
3. **Loop**: As above; when the agent calls the retrieval tool, run the vector search and append chunks as the observation.
4. **Evaluation**: Test with questions that require retrieved docs; check citation and relevance.

---

## Quick revision

- **LangChain/LangGraph**, **LlamaIndex**, **CrewAI**, **AutoGen**, **Semantic Kernel**, and **Assistants API** are common choices for building agents.
- **Core loop**: messages + tools → LLM → parse tool calls → execute → append results → repeat until answer.
- **RAG as a tool**: expose retrieval as a function; agent calls it when it needs document context.
