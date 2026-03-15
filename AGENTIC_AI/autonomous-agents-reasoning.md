# Autonomous Agents and Reasoning

Autonomous agents **plan** and **reason** over multiple steps, deciding what to do next (e.g. which tool to call) instead of producing a single response.

---

## Autonomy in practice

- **Autonomous** here means: the system decides the next action (tool or answer) at each step, within guardrails (allowed tools, max steps, safety checks).
- The **LLM** is the decision maker: given the goal and history (including tool results), it outputs the next thought and/or action.

---

## Reasoning patterns

### ReAct (Reason + Act)

- Model outputs alternating **Thought**, **Action**, **Observation**.
- Example:
  - Thought: I need the current weather in Paris.
  - Action: `get_weather(city="Paris")`
  - Observation: 22°C, rain expected.
  - Thought: User asked about umbrella; rain expected → yes.
  - Answer: Yes, take an umbrella.

### Plan-and-execute

- **Plan**: First output a list of steps (e.g. 1. Get weather 2. Compare to threshold 3. Recommend).
- **Execute**: For each step, call tools or reason, then move to the next; optionally refine the plan if something fails.

### Chain-of-thought (CoT)

- Encourage step-by-step reasoning in the model’s reply before committing to a tool or final answer.
- Improves accuracy on math and multi-step tasks; can be combined with ReAct (reasoning in “Thought” blocks).

---

## Planning

- **High-level plan**: Break the user goal into sub-goals (e.g. “research topic” → search, read, summarize).
- **Execution**: For each sub-goal, run the agent loop (reason + tool + observe) until that sub-goal is satisfied, then proceed.
- **Recovery**: If a tool fails or the answer is wrong, the agent can retry, try another tool, or ask the user.

---

## Quick revision

- **Autonomous** = agent chooses next action (tool or answer) at each step within guardrails.
- **ReAct**: alternate Thought, Action, Observation. **Plan-and-execute**: plan first, then execute steps. **CoT**: step-by-step reasoning in text.
- **Planning** = decompose goal into sub-goals; execute and optionally refine.
