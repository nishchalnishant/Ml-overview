# Build an AI Agent

Minimal **tool-using agent**: LLM + tools + loop until answer or max steps.

---

## Agent loop (pseudocode)

```
messages = [system, user_message]
while not done and steps < max_steps:
    response = llm.chat(messages, tools=tool_schemas)
    if response has no tool_calls:
        return response.content  # final answer
    for each tool_call in response.tool_calls:
        result = run_tool(tool_call.name, tool_call.arguments)
        messages.append(tool_result(tool_call.id, result))
    messages.append(response)  # assistant message with tool_calls
    steps += 1
```

---

## Tool schema (OpenAI-style)

```python
tools = [
    {
        "type": "function",
        "function": {
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
        }
    }
]
```

---

## Minimal Python (OpenAI API)

```python
import json
from openai import OpenAI

client = OpenAI()

def run_tool(name: str, arguments: str):
    args = json.loads(arguments)
    if name == "get_weather":
        return f"Weather in {args['city']}: 22°C, sunny"  # mock
    return "Unknown tool"

def agent(user_message: str, max_steps: int = 5):
    messages = [
        {"role": "system", "content": "You have access to get_weather. Use it when needed."},
        {"role": "user", "content": user_message}
    ]
    for _ in range(max_steps):
        resp = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        msg = resp.choices[0].message
        if not msg.tool_calls:
            return msg.content
        for tc in msg.tool_calls:
            result = run_tool(tc.function.name, tc.function.arguments)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result
            })
        messages.append({
            "role": "assistant",
            "content": msg.content or None,
            "tool_calls": msg.tool_calls
        })
    return "Max steps reached."
```

---

## RAG as a tool

Expose retrieval as a function; agent calls it when it needs docs.

```python
tools.append({
    "type": "function",
    "function": {
        "name": "search_docs",
        "description": "Search internal documentation. Use when you need company or product info.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"]
        }
    }
})

def run_tool(name, arguments):
    # ...
    if name == "search_docs":
        q = json.loads(arguments)["query"]
        chunks = vector_store.search(q, k=5)
        return "\n".join(chunks)
```

---

## Diagram

```
User → [System + tools] + history + user msg → LLM
                    ↑                              ↓
                    │                    tool_calls or answer
                    │                              ↓
                    │              execute tools → observations
                    └──────────── append to messages, repeat
```

See [AGENTIC_AI](../AGENTIC_AI/README.md) and [Tool-using agents](../AGENTIC_AI/tool-using-agents.md) for more.
