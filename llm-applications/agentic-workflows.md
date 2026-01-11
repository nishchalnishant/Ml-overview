# Agentic Workflows & Multi-Agent Systems

## Executive Summary
Moving from "Chatbots" to "Autonomous Agents".

| Concept | Definition | Key Pattern |
|---------|------------|-------------|
| **Tool Use** | LLM calling external APIs (Search, Calculator) | Function Calling / JSON mode |
| **Reasoning** | Thinking before acting | Chain-of-Thought (CoT) |
| **Agentic Loop** | Self-correction and planning | ReAct (Reason + Act) |
| **Multi-Agent** | Orchestrating specialized models | Hierarchical / Peer architecture |

---

## 1. Core Logic Patterns

### Chain-of-Thought (CoT)
Encouraging the model to output its intermediate reasoning steps. 
- **Few-Shot CoT**: Providing examples with reasoning.
- **Zero-Shot CoT**: Simply adding "Let's think step by step" to the prompt.

### The ReAct Pattern
A loop of **Thought $\rightarrow$ Action $\rightarrow$ Observation**.
1. **Thought**: "I need to find the current weather in Paris."
2. **Action**: `get_weather(city="Paris")`
3. **Observation**: "15C and Cloudy."
4. **Conclusion**: "The weather in Paris is 15C."

---

## 2. Compound AI Systems
Instead of one massive model, use multiple components.
- **Routing**: A small model decides which specialized model to use (e.g., Coding agent vs. Summarization agent).
- **Consensus**: Multiple agents vote on an answer to reduce hallucinations.

---

## 3. Cognitive Architectures
- **Memory**: Giving agents a "working memory" (Short-term context) and "long-term memory" (Vector DB retrieval).
- **Planning**: Breaking a complex user goal into sub-tasks (e.g., Task Decomposition).

---

## Interview Questions

**1. "What is an 'Agent' in the context of LLMs?"**
> An agent is a system where the LLM is the "brain" that uses a planning loop and external tools (APIs, code execution) to accomplish a non-trivial goal. Unlike a simple RAG, an agent can decide *how* to use the information it retrieves.

**2. "How do you prevent an Agent from getting stuck in an infinite loop?"**
> 1. Set a **Max Iteration** limit ($N=10$). 2. Implement a "Final Answer" override if the model repeats the same thought pattern 3 times. 3. Use an external "Monitor" model to evaluate the loop.

**3. "Explain Function Calling."**
> It's a structured way for the model to output its intent to use a tool. Instead of prose, the model outputs a valid JSON object matching a predefined schema which the developer's code then executes.

---

## Agentic Pseudo-code
```python
while not finished:
    thought = model.generate(f"Goal: {goal}\nHistory: {history}\nNext Step?")
    if "FINAL ANSWER" in thought: break
    tool, params = extract_action(thought)
    observation = execute(tool, params)
    history.append(observation)
```
