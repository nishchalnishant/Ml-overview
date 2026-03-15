# Tool-Using Agents

Agents call **tools** (functions) to get information or perform actions they cannot do with text alone: search, code execution, API calls, RAG retrieval, calculators, etc.

---

## Why tools?

- LLMs have no direct access to live data, private docs, or execution environments.
- **Tools** expose these as callable functions; the agent chooses which tool to call and with what arguments, then uses the result in the next reasoning step.

---

## Tool-calling mechanism

1. **Tool schema**: Each tool is described by name, description, and parameters (e.g. JSON Schema). The LLM sees these in the prompt or via a system message.
2. **Model output**: The model returns a structured **tool call** (tool name + arguments) instead of or in addition to plain text. Many APIs support a dedicated tool-call format (e.g. OpenAI function calling).
3. **Execution**: The orchestrator runs the chosen function with the given arguments in a sandboxed or controlled environment.
4. **Observation**: The return value (or error) is injected back into the conversation as an "observation"; the LLM then reasons and may call another tool or produce the final answer.

---

## Example tool definitions (pseudocode)

```python
tools = [
    {
        "name": "get_weather",
        "description": "Get current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["city"]
        }
    },
    {
        "name": "search_web",
        "description": "Search the web for up-to-date information.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
            "required": ["query"]
        }
    }
]
```

The LLM receives these (or a summary) and outputs e.g. `get_weather(city="Paris")` or the API-equivalent structure.

---

## Design considerations

- **Minimal set**: Fewer, well-described tools reduce confusion and latency.
- **Safety**: Validate arguments, rate-limit, sandbox execution (e.g. code run in a container).
- **Observations**: Keep tool results concise; long responses consume context and can distract the model.
- **Retries**: Define behavior when a tool fails (retry, ask user, or give up).

---

## Connection to RAG

- **RAG** can be exposed as a tool: e.g. `retrieve(query, top_k=5)` returns relevant chunks; the agent calls it when it needs internal documents.
- Hybrid flows: agent decides when to search (vector DB) vs when to call other APIs or answer directly.

---

## Quick revision

- **Tools** = functions the agent can call; described by name, description, and parameters.
- **Flow**: LLM outputs tool name + arguments → orchestrator executes → observation fed back → LLM continues.
- **Safety**: validate inputs, sandbox execution, limit which tools are available and how often they can be called.
