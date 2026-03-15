# Memory and Agent Orchestration

Agents need **memory** to use past turns and tool results, and **orchestration** to run the reasoning loop and manage context.

---

## Memory systems

| Type | Purpose | Example |
|------|--------|--------|
| **Short-term (conversation)** | Current dialog and recent tool outputs | Sliding window of messages in the prompt |
| **Long-term** | Facts or summaries across sessions | Vector store of important facts; retrieved when relevant |
| **Episodic** | Key events or decisions | Stored “memories” (e.g. “User prefers Python”) |

- **Implementation**: Short-term = keep last N messages (or tokens). Long-term = embed and store in a vector DB; at each step, retrieve top-k and add to context. Episodic = similar retrieval over “memory” items that can be updated.

---

## Context management

- **Window limit**: LLMs have a fixed context size; once exceeded, older content must be dropped, summarized, or retrieved.
- **Strategies**: (1) Truncate oldest messages. (2) Summarize old turns into a short paragraph. (3) Retrieve relevant past chunks from a vector store and prepend them.
- **Tool results**: Keep observations concise; optionally summarize long tool outputs before appending.

---

## Orchestration

The **orchestrator**:

1. Builds the prompt (system message, tools schema, conversation history, current user message).
2. Calls the LLM.
3. Parses the response: plain text vs tool call(s).
4. If tool call: executes tools, appends observations to conversation, goes to step 2.
5. If final answer (or max steps): returns to user and optionally updates long-term memory.

---

## Reasoning loop (pseudocode)

```python
messages = [system_prompt, *history, user_message]
while not done and steps < max_steps:
    response = llm.create(messages, tools=tools)
    if response.has_tool_calls:
        for call in response.tool_calls:
            result = execute_tool(call.name, call.args)
            messages.append(tool_result(call.id, result))
    else:
        done = True
        return response.content
    steps += 1
```

---

## Quick revision

- **Short-term**: current conversation (and recent tool results) in the prompt.
- **Long-term / episodic**: stored facts or events; retrieved (e.g. by embedding) and injected when relevant.
- **Orchestration**: build prompt → LLM → parse → execute tools or return answer → update context → repeat.
- **Context limit**: truncate, summarize, or retrieve to stay within the model’s context window.
