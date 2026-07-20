---
module: LLMs
topic: Interview Notes
subtopic: Coding And Practical Implementation Snappy
status: unread
tags: [llms, ml, interview-notes-coding-and-pra]
---

> _Quick-recall companion. For the full deep-dive, see [coding-and-practical-implementation.md](13-coding-and-practical-implementation.md)._

# Coding & practical implementation — patterns you can type under pressure

This file is intentionally **code-forward** but not bloated. Each question gives a minimal, interview-safe skeleton.

**DevOps note:** treat every LLM call like an unreliable dependency: timeouts, retries, budgets, logs.

---

# Q1: Implement a basic RAG pipeline using an embedding model and a vector database.

```python
def rag_answer(query, embed, vdb, llm, k=5):
    qv = embed(query)
    hits = vdb.search(qv, k=k)
    context = "\n\n".join(h.text for h in hits)
    prompt = (
        "Use ONLY this context. If missing, say INSUFFICIENT_DATA.\n"
        f"<context>\n{context}\n</context>\n"
        f"Q: {query}"
    )
    return llm(prompt, temperature=0)
```

---

# Q2: Build a simple AI agent with tool use (e.g., calculator, web search).

```python
TOOLS = {"calc": calc, "search": search}

def agent_loop(task, llm, max_steps=8):
    state = {"task": task, "steps": []}
    for _ in range(max_steps):
        action = llm.plan(state)  # returns {tool, args} or {final}
        if action.get("final"):
            return action["final"]
        tool = TOOLS[action["tool"]]
        obs = tool(**action["args"])
        state["steps"].append({"action": action, "obs": obs})
    return "STOP: budget reached"
```

---

# Q3: Implement semantic search using embeddings and cosine similarity.

```python
import numpy as np

def cosine(a, b):
    a, b = np.array(a), np.array(b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

def semantic_search(query_vec, docs, top_k=5):
    scored = [(cosine(query_vec, d.vec), d) for d in docs]
    return [d for _, d in sorted(scored, reverse=True)[:top_k]]
```

---

# Q4: Write code for different text chunking strategies (fixed-size, recursive, semantic).

```python
def fixed(text, n=800):
    return [text[i:i+n] for i in range(0, len(text), n)]

def recursive(text, sizes=(2000, 800, 300)):
    chunks = [text]
    for s in sizes:
        nxt = []
        for c in chunks:
            if len(c) <= s:
                nxt.append(c)
                continue
            parts = c.split("\n\n") if "\n\n" in c else [c[i:i+s] for i in range(0, len(c), s)]
            nxt.extend([p.strip() for p in parts if p.strip()])
        chunks = nxt
    return chunks
```

---

# Q5: Implement a prompt template system with variable substitution.

```python
from string import Template

t = Template("You are $role. Return ONLY JSON with keys: $keys. Text: $text")

def render(role, keys, text):
    return t.substitute(role=role, keys=keys, text=text)
```

---

# Q6: Build an evaluation pipeline for LLM outputs using LLM-as-a-judge.

```python
def judge(judge_llm, prompt, output):
    rubric = (
        "Score 0-10 for correctness and faithfulness. "
        "Return JSON {score:int, reason:str}."
    )
    return judge_llm(f"{rubric}\nPROMPT: {prompt}\nOUTPUT: {output}")
```

---

# Q7: Implement streaming responses for an LLM API.

```python
# pseudocode: yield tokens as server-sent events
async def stream(llm, prompt):
    async for tok in llm.stream(prompt):
        yield f"data: {tok}\n\n"
```

---

# Q8: Build a simple vector similarity search from scratch.

```python
def knn(q, vecs, k=5):
    scored = [(sum(qi*vi for qi,vi in zip(q,v)), i) for i,v in enumerate(vecs)]
    return [i for _, i in sorted(scored, reverse=True)[:k]]
```

---

# Q9: Implement a conversation memory for a chatbot (sliding window, summary, buffer).

```python
def build_context(turns, window=6, summary=None):
    recent = turns[-window:]
    parts = []
    if summary:
        parts.append(f"SUMMARY: {summary}")
    parts.extend([f"{t['role']}: {t['text']}" for t in recent])
    return "\n".join(parts)
```

---

# Q10: Write code to detect and handle hallucinations in LLM outputs.

```python
def grounded_answer(llm, question, context):
    prompt = (
        "Answer ONLY using <context>. If missing, say INSUFFICIENT_DATA.\n"
        f"<context>{context}</context>\nQ:{question}"
    )
    return llm(prompt, temperature=0)
```

---

# Q11: Implement a retry mechanism with exponential backoff for LLM API calls.

```python
import time, random

def call_with_retry(fn, tries=5, base=0.5):
    for i in range(tries):
        try:
            return fn()
        except Exception:
            time.sleep(base * (2**i) + random.random()*0.1)
    raise
```

---

# Q12: Write a function calling (tool use) handler for an LLM API.

```python
TOOLS = {"calc": calc}

def handle(tool_name, args):
    if tool_name not in TOOLS:
        raise ValueError("tool not allowed")
    return TOOLS[tool_name](**args)
```

---

# Q13: Implement a simple re-ranker for search results.

```python
# score() could be a cross-encoder or a small LLM

def rerank(query, docs, score, k=5):
    return sorted(docs, key=lambda d: score(query, d.text), reverse=True)[:k]
```

---

# Q14: Build a basic document parser that extracts text from PDFs and splits it into chunks.

```python
# pseudocode: use a PDF lib to extract text then chunk

def parse_and_chunk(pdf_bytes):
    text = extract_text(pdf_bytes)
    return recursive(text)
```

---

# Q15: Implement cosine similarity, dot product, and Euclidean distance from scratch.

```python
import math

def dot(a, b):
    return sum(x*y for x,y in zip(a,b))

def norm(a):
    return math.sqrt(dot(a,a))

def cosine(a, b):
    return dot(a,b) / (norm(a)*norm(b) + 1e-9)

def euclid(a, b):
    return math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))
```

---

# Q16: Write code to implement token counting and context window management.

```python
def budget(messages, tokenizer, max_tokens):
    n = len(tokenizer.encode(str(messages)))
    if n > max_tokens:
        raise ValueError("context too large")
    return n
```

---

# Q17: Build a simple prompt versioning system.

```python
PROMPTS = {"v1": "...", "v2": "..."}

def get_prompt(version):
    return PROMPTS[version]
```

---

# Q18: Implement a caching layer for LLM responses.

```python
CACHE = {}

def cached_call(key, fn):
    if key in CACHE:
        return CACHE[key]
    CACHE[key] = fn()
    return CACHE[key]
```

---

# Q19: Implement semantic caching for LLM queries.

```python
# store (vec, answer) and reuse if cosine > threshold

def semantic_cache(q_vec, cache, threshold=0.92):
    best = None
    for v, a in cache:
        s = cosine(q_vec, v)
        if s > threshold and (best is None or s > best[0]):
            best = (s, a)
    return best[1] if best else None
```

---

# Q20: Write code to detect prompt injection attempts in user inputs.

```python
BAD = ["ignore previous", "reveal system prompt", "bypass"]

def looks_injected(user_text):
    t = user_text.lower()
    return any(b in t for b in BAD)
```

---

# Q21: Implement an LLM output guardrails system (off-topic + PII).

```python

def guardrails(output, pii_detector):
    if pii_detector(output):
        return "BLOCKED_PII"
    return output
```

---

# Q22: Build a multi-agent system where agents collaborate on a task.

```python

def collaborate(planner, researcher, writer, task):
    plan = planner(task)
    notes = researcher(plan)
    return writer(task, notes)
```
