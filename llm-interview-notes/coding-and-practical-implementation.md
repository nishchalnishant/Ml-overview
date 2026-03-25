# Q1: Implement a basic RAG pipeline using an embedding model and a vector database.

## 1. 🔹 Direct Answer
A basic RAG pipeline: embed documents into chunks, upsert vectors + metadata into a vector DB, embed the query at retrieval time, run similarity search (top-k), concatenate retrieved text as context, then call the LLM with a grounded prompt.

## 2. 🔹 Intuition
Retrieve first, generate second—so answers can cite what was actually in your corpus.

## 3. 🔹 Deep Dive
Steps: (1) chunk documents, (2) `embed(text)` per chunk, (3) store `(id, vector, text, metadata)` in the index, (4) `embed(query)`, (5) `search(query_vec, k)`, (6) build `context` from hits, (7) prompt: answer only from context + optional citations.

## 4. 🔹 Practical Perspective
Use when knowledge changes often and you need grounding. Trade-off: retrieval quality dominates; tune chunk size and top-k before prompt tweaks.

## 5. 🔹 Code Snippet
```python
import numpy as np

def rag_answer(query, chunks, embed_fn, index, llm):
    qv = embed_fn(query)
    hits = index.search(qv, top_k=5)
    context = "\n\n".join(chunks[i] for i in hits.ids)
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer using only the context."
    return llm.generate(prompt)
```

## 6. 🔹 Interview Follow-ups
1. Q: What if retrieval returns nothing?  
   A: Abstain or widen search / hybrid BM25; do not invent facts.
2. Q: How do you evaluate?  
   A: Recall@k on held-out QA pairs + faithfulness checks.

## 7. 🔹 Common Mistakes
Chunking without overlap so sentences split across chunks.

## 8. 🔹 Comparison / Connections
Connects to chunking, vector DBs, and RAG evaluation.

## 9. 🔹 One-line Revision
RAG = embed chunks → index → retrieve top-k → prompt LLM with context only.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q2: Build a simple AI agent with tool use (e.g., calculator, web search).

## 1. 🔹 Direct Answer
Loop: LLM receives tools as JSON schemas, returns tool calls or final text; you execute allowed tools, append results as `tool` messages, repeat until done or max steps.

## 2. 🔹 Intuition
The model proposes actions; your code runs them safely and feeds observations back.

## 3. 🔹 Deep Dive
Define `TOOLS` with name, description, parameters (JSON Schema). Parse `tool_calls` from the API response, validate args, dispatch `calculator(expr)` / `web_search(q)`, sanitize outputs, cap iterations, refuse destructive tools.

## 4. 🔹 Practical Perspective
Use for assistants that need fresh data or computation. Trade-off: tool hallucination and infinite loops—use budgets and allowlists.

## 5. 🔹 Code Snippet
```python
def run_agent(messages, tools, llm, max_steps=8):
    for _ in range(max_steps):
        resp = llm.chat(messages, tools=tools)
        if not resp.tool_calls:
            return resp.content
        for tc in resp.tool_calls:
            obs = dispatch_tool(tc.name, tc.arguments)  # validate + sandbox
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": obs})
    return "Max steps exceeded."
```

## 6. 🔹 Interview Follow-ups
1. Q: How do you validate tool args?  
   A: JSON schema validation before execution.
2. Q: Parallel tool calls?  
   A: Run independent tools concurrently; merge results in order.

## 7. 🔹 Common Mistakes
Executing tools without schema validation or user permission for side effects.

## 8. 🔹 Comparison / Connections
ReAct, function calling, agent guardrails.

## 9. 🔹 One-line Revision
Agent loop: LLM → tool_calls → execute → tool messages → repeat with bounds.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q3: Implement semantic search using embeddings and cosine similarity.

## 1. 🔹 Direct Answer
Embed queries and documents into vectors, normalize if using cosine, compute similarity against a corpus (matrix multiply or ANN index), return top-k by score.

## 2. 🔹 Intuition
“Semantic” means closeness in meaning space, not keyword overlap.

## 3. 🔹 Deep Dive
`cosine(a,b) = dot(a,b)/(||a||*||b||)`. For normalized vectors, cosine equals dot product. Store `doc_embeddings` as matrix `D`; scores = `query_emb @ D.T`; argsort top-k.

## 4. 🔹 Practical Perspective
Brute-force OK for small N; use FAISS/HNSW for large scale.

## 5. 🔹 Code Snippet
```python
import numpy as np

def cosine_scores(q, D):  # D: [n, d], q: [d]
    qn = q / (np.linalg.norm(q) + 1e-12)
    Dn = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-12)
    return Dn @ qn

def top_k(q, texts, embed_fn, D, k=5):
    scores = cosine_scores(embed_fn(q), D)
    idx = np.argsort(-scores)[:k]
    return [(texts[i], float(scores[i])) for i in idx]
```

## 6. 🔹 Interview Follow-ups
1. Q: Cosine vs dot product?  
   A: Same if vectors are normalized; dot on raw vectors mixes magnitude and angle.
2. Q: ANN?  
   A: Approximate nearest neighbors for speed at scale.

## 7. 🔹 Common Mistakes
Forgetting to normalize when claiming “cosine.”

## 8. 🔹 Comparison / Connections
Vector DBs, hybrid search with BM25.

## 9. 🔹 One-line Revision
Semantic search = embed → cosine/dot similarity → top-k (or ANN at scale).

## 10. 🔹 Difficulty Tag
🟢 Easy

---

# Q4: Write code for different text chunking strategies (fixed-size, recursive, semantic).

## 1. 🔹 Direct Answer
Fixed-size: split every N chars/tokens with optional overlap. Recursive: split on separators (paragraphs → sentences) until chunks fit. Semantic: split on embedding similarity drops or sentence boundaries, then merge/split to target size.

## 2. 🔹 Intuition
Fixed is simple; recursive respects structure; semantic tries to keep one topic per chunk.

## 3. 🔹 Deep Dive
Recursive often uses `["\n\n", "\n", " "]`. Semantic: embed sentences or windows, compare consecutive cosine distances, break when drop exceeds threshold.

## 4. 🔹 Practical Perspective
RAG usually combines recursive + size limits; pure fixed-size is fastest but can cut mid-sentence.

## 5. 🔹 Code Snippet
```python
def fixed_chunks(text, size=500, overlap=50):
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i+size])
        i += size - overlap
    return out

def recursive_chunk(text, max_size, seps=("\n\n", "\n", " ")):
    if len(text) <= max_size:
        return [text]
    for sep in seps:
        if sep in text:
            parts = text.split(sep)
            # merge small pieces up to max_size ...
            break
    return [text[:max_size]]  # simplified fallback
```

## 6. 🔹 Interview Follow-ups
1. Q: Chunk size choice?  
   A: Match embedding model limits and retrieval granularity; evaluate recall.

## 7. 🔹 Common Mistakes
Zero overlap so context is lost at boundaries.

## 8. 🔹 Comparison / Connections
Parent-child chunking, RAG ingestion.

## 9. 🔹 One-line Revision
Chunk by fixed windows, recursive separators, or semantic embedding boundaries—often combined.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q5: Implement a prompt template system with variable substitution.

## 1. 🔹 Direct Answer
Store templates with placeholders (`{var}` or `{{var}}`), validate required keys, escape user content, render with a safe formatter, version templates.

## 2. 🔹 Intuition
Treat prompts like parameterized SQL: structure fixed, data injected.

## 3. 🔹 Deep Dive
Use `str.format` or Jinja2 for conditionals; whitelist variables; never `eval` user strings; optionally type-check (e.g., schema for `context`).

## 4. 🔹 Practical Perspective
Production needs template IDs, versions, and tests that snapshot rendered prompts.

## 5. 🔹 Code Snippet
```python
TEMPLATES = {
    "qa_v1": "You are a helper.\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer:",
}

def render(name, **kwargs):
    if missing := set(_placeholders(TEMPLATES[name])) - kwargs.keys():
        raise ValueError(f"Missing: {missing}")
    return TEMPLATES[name].format(**kwargs)
```

## 6. 🔹 Interview Follow-ups
1. Q: Prevent injection in variables?  
   A: Label blocks as untrusted; do not let variables override system instructions in code.

## 7. 🔹 Common Mistakes
Using f-strings with raw user input for the whole prompt.

## 8. 🔹 Comparison / Connections
Prompt versioning, LLM gateway.

## 9. 🔹 One-line Revision
Templates = versioned strings + required keys + safe substitution + validation.

## 10. 🔹 Difficulty Tag
🟢 Easy

---

# Q6: Build an evaluation pipeline for LLM outputs using LLM-as-a-judge.

## 1. 🔹 Direct Answer
For each example: run task → get output → feed judge prompt with rubric (1–5, criteria) → parse judge score/JSON → aggregate metrics; optionally human-calibrate the judge.

## 2. 🔹 Intuition
Automate grading at scale; judge is another LLM with a fixed rubric.

## 3. 🔹 Deep Dive
Include: instruction, criteria, optional reference, model output, output format (JSON). Temperature 0 for judges. Track correlation with human labels on a subset.

## 4. 🔹 Practical Perspective
Judge bias is real; use different model family for judge or ensemble.

## 5. 🔹 Code Snippet
```python
def judge_eval(question, output, reference, judge_llm):
    rubric = "Score 1-5 on correctness, completeness, safety. Return JSON only."
    raw = judge_llm.generate(f"{rubric}\nQ: {question}\nRef: {reference}\nOut: {output}")
    return json.loads(raw)
```

## 6. 🔹 Interview Follow-ups
1. Q: Grounding for RAG?  
   A: Judge must see retrieved context, not only the answer.

## 7. 🔹 Common Mistakes
Judging without the same evidence the user saw.

## 8. 🔹 Comparison / Connections
G-Eval, offline eval suites.

## 9. 🔹 One-line Revision
Eval pipeline: generate → judge with rubric + structured parse → aggregate + calibrate.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q7: Implement streaming responses for an LLM API.

## 1. 🔹 Direct Answer
Use the provider’s streaming API (SSE/WebSocket), iterate over chunks, yield tokens to the client, accumulate full text for logging/validation after stream ends.

## 2. 🔹 Intuition
First token arrives fast (TTFT); user sees progress.

## 3. 🔹 Deep Dive
Handle: `stream=True`, parse delta events, UTF-8 boundaries, cancellation on client disconnect, final message may include usage; run safety/format checks on full text when done.

## 4. 🔹 Practical Perspective
Essential for chat UX; buffer for JSON if you need structured output at end.

## 5. 🔹 Code Snippet
```python
def stream_chat(llm, messages):
    buf = []
    for chunk in llm.stream(messages):
        if chunk.content:
            buf.append(chunk.content)
            yield chunk.content
    full = "".join(buf)
    # validate(full) ...
```

## 6. 🔹 Interview Follow-ups
1. Q: Structured JSON while streaming?  
   A: Stream to user optionally; parse full buffer at end or use constrained decoding.

## 7. 🔹 Common Mistakes
Calling tools on partial JSON mid-stream.

## 8. 🔹 Comparison / Connections
LLM serving, TTFT metrics.

## 9. 🔹 One-line Revision
Stream chunks to client; accumulate and validate full output at end.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q8: Build a simple vector similarity search from scratch.

## 1. 🔹 Direct Answer
Represent corpus as matrix `D`, query `q`; compute similarities (dot or cosine), sort indices descending, return top-k. For cosine, normalize rows and query first.

## 2. 🔹 Intuition
Brute-force linear scan is the baseline before ANN.

## 3. 🔹 Deep Dive
Complexity O(nd) per query for n vectors of dim d. Use float32, batch queries as matrix multiply.

## 4. 🔹 Practical Perspective
Fine for thousands of vectors; switch to FAISS/HNSW beyond that.

## 5. 🔹 Code Snippet
```python
import numpy as np

def brute_force_search(q, D, k, normalize=True):
    if normalize:
        q = q / (np.linalg.norm(q) + 1e-12)
        D = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-12)
    scores = D @ q
    idx = np.argpartition(-scores, k)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return idx, scores[idx]
```

## 6. 🔹 Interview Follow-ups
1. Q: Speed up?  
   A: ANN, or GPU matmul for medium n.

## 7. 🔹 Common Mistakes
Using Python loops per vector instead of vectorized matmul.

## 8. 🔹 Comparison / Connections
FAISS IVF, HNSW.

## 9. 🔹 One-line Revision
Brute-force search = score all vectors with dot/cosine, argpartition top-k.

## 10. 🔹 Difficulty Tag
🟢 Easy

---

# Q9: Implement a conversation memory for a chatbot (sliding window, summary, buffer).

## 1. 🔹 Direct Answer
Sliding window: keep last N messages. Summary: periodically compress older turns into a summary string. Buffer: store raw list with optional max token budget—evict oldest or summarize when over budget.

## 2. 🔹 Intuition
Infinite history does not fit context; you truncate or compress.

## 3. 🔹 Deep Dive
Implement `Memory` class: `append(role, content)`, `get_messages()` returns window + optional `system` summary. Token counting: use tokenizer for accurate budget.

## 4. 🔹 Practical Perspective
Summary can drop constraints; test multi-turn evals; consider entity store for facts.

## 5. 🔹 Code Snippet
```python
class ChatMemory:
    def __init__(self, max_turns=10):
        self.turns = []
        self.max_turns = max_turns
    def append(self, role, content):
        self.turns.append({"role": role, "content": content})
        while len(self.turns) > self.max_turns * 2:
            self.turns.pop(0)
    def as_messages(self):
        return list(self.turns)
```

## 6. 🔹 Interview Follow-ups
1. Q: Summary memory?  
   A: Every K turns, replace prefix with `Summary: ...` via LLM.

## 7. 🔹 Common Mistakes
Sliding window without system prompt re-stating constraints.

## 8. 🔹 Comparison / Connections
Long context, RAG for episodic memory.

## 9. 🔹 One-line Revision
Memory = sliding window and/or rolling summary under a token budget.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q10: Write code to detect and handle hallucinations in LLM outputs.

## 1. 🔹 Direct Answer
If RAG: extract claims, check entailment against retrieved chunks; if unsupported, replace with abstention or “not in context.” Optionally use NLI model or second-pass verification.

## 2. 🔹 Intuition
Hallucination = confident text without evidence; verify claims against sources.

## 3. 🔹 Deep Dive
Pipeline: `claims = extract_claims(answer)` → for each claim, `max_entail(claim, evidence)` → if below threshold, flag. Handle: rewrite answer removing unsupported parts or return canned abstention.

## 4. 🔹 Practical Perspective
Claim extraction can be LLM-based; keep deterministic checks where possible.

## 5. 🔹 Code Snippet
```python
def grounded_or_abstain(answer, evidence_chunks, nli):
    claims = extract_claims(answer)
    for c in claims:
        if max(nli(c, e) for e in evidence_chunks) < 0.5:
            return "I cannot verify that from the provided sources."
    return answer
```

## 6. 🔹 Interview Follow-ups
1. Q: No retrieval?  
   A: Abstain on uncertainty, or external fact-check tool.

## 7. 🔹 Common Mistakes
Using BLEU only—does not measure faithfulness.

## 8. 🔹 Comparison / Connections
RAG evaluation, entailment.

## 9. 🔹 One-line Revision
Detect hallucinations by claim-level entailment vs evidence; abstain or strip unsupported claims.

## 10. 🔹 Difficulty Tag
🟣 Hard

---

# Q11: Implement a retry mechanism with exponential backoff for LLM API calls.

## 1. 🔹 Direct Answer
Retry on transient errors (429, 503, timeouts) with exponential delay `base * 2^attempt`, jitter, max attempts, and do not retry non-idempotent side effects without idempotency keys.

## 2. 🔹 Intuition
Back off so you don’t hammer a struggling API.

## 3. 🔹 Deep Dive
Use `time.sleep`, cap max delay, distinguish `Retry-After` header. For 401/400, fail fast.

## 4. 🔹 Practical Perspective
Combine with circuit breaker at gateway scale.

## 5. 🔹 Code Snippet
```python
import time, random

def call_with_backoff(fn, max_retries=5, base=0.5):
    for attempt in range(max_retries):
        try:
            return fn()
        except TransientError as e:
            if attempt == max_retries - 1:
                raise
            delay = base * (2 ** attempt) + random.uniform(0, 0.1)
            time.sleep(delay)
```

## 6. 🔹 Interview Follow-ups
1. Q: Jitter why?  
   A: Prevents synchronized retries (thundering herd).

## 7. 🔹 Common Mistakes
Retrying 400 errors indefinitely.

## 8. 🔹 Comparison / Connections
Rate limiting, resilience patterns.

## 9. 🔹 One-line Revision
Exponential backoff + jitter + max retries on transient failures only.

## 10. 🔹 Difficulty Tag
🟢 Easy

---

# Q12: Write a function calling (tool use) handler for an LLM API.

## 1. 🔹 Direct Answer
Parse `tool_calls` from response, map name to Python function, validate arguments with JSON Schema, execute, return tool messages with matching `tool_call_id`.

## 2. 🔹 Intuition
Bridge from LLM structured output to real code execution.

## 3. 🔹 Deep Dive
Registry: `TOOLS = {"calc": calc_fn, ...}`. Validate args, catch exceptions, return error string to model. Enforce allowlist and timeouts.

## 4. 🔹 Practical Perspective
Never pass raw strings to `eval`; use safe parsers for expressions.

## 5. 🔹 Code Snippet
```python
def handle_tool_calls(response, registry, schema_validate):
    out = []
    for tc in response.tool_calls:
        fn = registry.get(tc.function.name)
        args = json.loads(tc.function.arguments)
        schema_validate(tc.function.name, args)
        result = fn(**args)
        out.append({"role": "tool", "tool_call_id": tc.id, "content": str(result)})
    return out
```

## 6. 🔹 Interview Follow-ups
1. Q: Parallel tools?  
   A: `asyncio.gather` if independent.

## 7. 🔹 Common Mistakes
Executing tools before validating arguments.

## 8. 🔹 Comparison / Connections
OpenAI tools format, agents.

## 9. 🔹 One-line Revision
Tool handler = parse → validate → dispatch → return tool message payload.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q13: Implement a simple re-ranker for search results.

## 1. 🔹 Direct Answer
Retrieve broad top-K with fast retriever, then re-score pairs `(query, doc)` with a heavier cross-encoder or LLM, sort by new score, return top-k′.

## 2. 🔹 Intuition
Two-stage: recall then precision.

## 3. 🔹 Deep Dive
Cross-encoder: concat query+doc, single forward, relevance score. Cheaper: small bi-encoder reranker or BM25 fusion.

## 4. 🔹 Practical Perspective
Reranking 50–200 candidates is common; full corpus rerank is too slow.

## 5. 🔹 Code Snippet
```python
def rerank(query, docs, score_fn, top_k=10):
    scored = [(score_fn(query, d), d) for d in docs]
    scored.sort(key=lambda x: -x[0])
    return [d for _, d in scored[:top_k]]
```

## 6. 🔹 Interview Follow-ups
1. Q: `score_fn`?  
   A: Cross-encoder logits or dot product of separate query/doc towers.

## 7. 🔹 Common Mistakes
Reranking without enough recall from first stage.

## 8. 🔹 Comparison / Connections
RAG pipelines, ColBERT.

## 9. 🔹 One-line Revision
Rerank = rescore candidate list with a stronger relevance model and truncate.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q14: Build a basic document parser that extracts text from PDFs and splits it into chunks.

## 1. 🔹 Direct Answer
Use a library (e.g., `pypdf`, `pdfplumber`) to extract text per page, normalize whitespace, then apply chunking (fixed or recursive) with metadata `{source, page}`.

## 2. 🔹 Intuition
PDFs are layout-heavy; preserve page numbers for citations.

## 3. 🔹 Deep Dive
Handle: scanned PDFs need OCR (Tesseract/cloud). Store `(text, page, offset)` for traceability.

## 4. 🔹 Practical Perspective
Evaluate extraction quality on sample docs; tables may need special handling.

## 5. 🔹 Code Snippet
```python
def parse_pdf_chunks(path, chunk_size=800):
    from pypdf import PdfReader
    reader = PdfReader(path)
    chunks = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        for j in range(0, len(text), chunk_size):
            chunks.append({"text": text[j:j+chunk_size], "page": i})
    return chunks
```

## 6. 🔹 Interview Follow-ups
1. Q: Bad extraction?  
   A: Try pdfplumber, OCR, or layout models.

## 7. 🔹 Common Mistakes
Dropping page metadata so RAG cannot cite.

## 8. 🔹 Comparison / Connections
Document understanding, ingestion pipelines.

## 9. 🔹 One-line Revision
PDF parse → per-page text → chunk with page metadata for RAG.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q15: Implement cosine similarity, dot product, and Euclidean distance from scratch.

## 1. 🔹 Direct Answer
Dot: sum of elementwise products. Cosine: dot divided by product of L2 norms. Euclidean: L2 norm of difference.

## 2. 🔹 Intuition
Dot measures alignment; cosine removes magnitude; Euclidean measures geometric distance.

## 3. 🔹 Deep Dive
Add epsilon for numerical stability in division.

## 4. 🔹 Practical Perspective
Use vectorized NumPy; for batch, use broadcasting.

## 5. 🔹 Code Snippet
```python
import numpy as np

def dot(a, b):
    return float(np.dot(a, b))

def cosine(a, b, eps=1e-12):
    return dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + eps)

def euclidean(a, b):
    return float(np.linalg.norm(a - b))
```

## 6. 🔹 Interview Follow-ups
1. Q: When is dot ≈ cosine?  
   A: When vectors are normalized.

## 7. 🔹 Common Mistakes
Using cosine on non-normalized vectors without dividing norms.

## 8. 🔹 Comparison / Connections
Vector search, clustering.

## 9. 🔹 One-line Revision
Dot = inner product; cosine = dot/(||a||·||b||); Euclidean = ||a−b||₂.

## 10. 🔹 Difficulty Tag
🟢 Easy

---

# Q16: Write code to implement token counting and context window management.

## 1. 🔹 Direct Answer
Use the model’s tokenizer (`tiktoken`, HuggingFace tokenizer) to count tokens per message; sum until under budget; truncate from oldest or summarize when over.

## 2. 🔹 Intuition
Char length ≠ token length; always use the same tokenizer as the model.

## 3. 🔹 Deep Dive
Reserve tokens for completion (`max_tokens`). Strip or compress middle messages if needed (lost-in-the-middle aware strategies).

## 4. 🔹 Practical Perspective
Log token counts per request for cost dashboards.

## 5. 🔹 Code Snippet
```python
def count_messages(messages, encode):
    return sum(len(encode(m["content"])) for m in messages)

def trim_to_budget(messages, encode, max_tokens, reserve_output=1024):
    budget = max_tokens - reserve_output
    while count_messages(messages, encode) > budget and len(messages) > 1:
        messages.pop(1)  # keep system, drop oldest user/assistant
    return messages
```

## 6. 🔹 Interview Follow-ups
1. Q: tiktoken?  
   A: OpenAI-compatible BPE counting for GPT models.

## 7. 🔹 Common Mistakes
Using `len(text.split())` as token count.

## 8. 🔹 Comparison / Connections
Long context, prefix caching.

## 9. 🔹 One-line Revision
Count with model tokenizer; trim or summarize to stay under context + output reserve.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q17: Build a simple prompt versioning system.

## 1. 🔹 Direct Answer
Store prompts as named versioned records (`id`, `version`, `body`, `created_at`); load by `(name, version)` or “latest”; log which version was used per request; run evals before bumping version.

## 2. 🔹 Intuition
Same as API versioning: immutable versions, explicit rollout.

## 3. 🔹 Deep Dive
YAML/JSON in repo or DB; CI checks for required variables; feature flag selects version in production.

## 4. 🔹 Practical Perspective
Tie to model version and eval results in metadata.

## 5. 🔹 Code Snippet
```python
PROMPTS = {
    ("support", 1): "You are helpful. {context}",
    ("support", 2): "You are helpful and cite sources. {context}",
}

def get_prompt(name, version):
    return PROMPTS[(name, version)]
```

## 6. 🔹 Interview Follow-ups
1. Q: Rollback?  
   A: Flip flag to previous version.

## 7. 🔹 Common Mistakes
Editing “latest” in place without version bump.

## 8. 🔹 Comparison / Connections
LLMOps, CI/CD for AI.

## 9. 🔹 One-line Revision
Prompt versioning = immutable (name, version) artifacts + logging + flags.

## 10. 🔹 Difficulty Tag
🟢 Easy

---

# Q18: Implement a caching layer for LLM responses.

## 1. 🔹 Direct Answer
Key cache by hash of `(model, temperature, prompt_template_version, normalized_prompt)`; store TTL; optional Redis; return cached response on hit; invalidate on prompt/model change.

## 2. 🔹 Intuition
Identical requests should not pay twice.

## 3. 🔹 Deep Dive
Include all parameters that affect output in the key; respect PII—do not cache sensitive payloads in shared caches without encryption.

## 4. 🔹 Practical Perspective
Great for FAQ and repeated evaluations.

## 5. 🔹 Code Snippet
```python
import hashlib, json

def cache_key(model, prompt, params):
    blob = json.dumps({"model": model, "prompt": prompt, **params}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()

def get_or_set(cache, key, ttl, compute_fn):
    hit = cache.get(key)
    if hit is not None:
        return hit
    val = compute_fn()
    cache.set(key, val, ttl=ttl)
    return val
```

## 6. 🔹 Interview Follow-ups
1. Q: Stale cache?  
   A: TTL + version bump on prompt change.

## 7. 🔹 Common Mistakes
Keying only on user text and ignoring temperature/max_tokens.

## 8. 🔹 Comparison / Connections
Semantic caching next.

## 9. 🔹 One-line Revision
Exact-match cache with stable keys including model and decoding params + TTL.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q19: Implement semantic caching for LLM queries (semantically similar queries).

## 1. 🔹 Direct Answer
Embed query, search a cache index of past `(embedding, response)` with cosine similarity; if best similarity ≥ threshold, return cached answer; else compute, store embedding + response.

## 2. 🔹 Intuition
“Similar question” hits cache even if wording differs.

## 3. 🔹 Deep Dive
Use ANN for many entries; store policy version in metadata; optionally verify answer still applies (cheap classifier).

## 4. 🔹 Practical Perspective
Risk: wrong cache hit—tune threshold and log near-misses.

## 5. 🔹 Code Snippet
```python
def semantic_cache_lookup(q_emb, cache_embs, cache_vals, threshold=0.92):
    sims = cache_embs @ q_emb  # normalized
    i = int(np.argmax(sims))
    if sims[i] >= threshold:
        return cache_vals[i]
    return None
```

## 6. 🔹 Interview Follow-ups
1. Q: Invalidate?  
   A: When knowledge base updates, clear or version semantic cache.

## 7. 🔹 Common Mistakes
Same threshold for all domains—should be calibrated.

## 8. 🔹 Comparison / Connections
Embeddings, RAG freshness.

## 9. 🔹 One-line Revision
Semantic cache = embed query → nearest neighbor above threshold → return else compute and store.

## 10. 🔹 Difficulty Tag
🟣 Hard

---

# Q20: Write code to detect prompt injection attempts in user inputs.

## 1. 🔹 Direct Answer
Combine heuristics (regex for “ignore previous instructions”, role overrides) and a lightweight classifier; never treat detection as perfect—enforce trust boundaries in code.

## 2. 🔹 Intuition
Flag obvious jailbreak patterns; block or sanitize before model sees untrusted text.

## 3. 🔹 Deep Dive
Return `risk_score`; if high, strip/redact or refuse; log for red team. Also scan retrieved documents for indirect injection.

## 4. 🔹 Practical Perspective
Layer with tool allowlists and ACL on retrieval.

## 5. 🔹 Code Snippet
```python
INJECTION_PATTERNS = [
    r"ignore (all )?(previous|prior) instructions",
    r"system prompt",
    r"you are now",
]

def injection_score(text):
    import re
    return sum(1 for p in INJECTION_PATTERNS if re.search(p, text, re.I))
```

## 6. 🔹 Interview Follow-ups
1. Q: False positives?  
   A: Tune patterns; use classifier with human review on borderline.

## 7. 🔹 Common Mistakes
Relying only on regex without backend enforcement.

## 8. 🔹 Comparison / Connections
Guardrails, prompt injection defenses.

## 9. 🔹 One-line Revision
Heuristic + classifier scoring; code-side policy is the real defense.

## 10. 🔹 Difficulty Tag
🟡 Medium

---

# Q21: Implement an LLM output guardrails system that checks for off-topic responses and PII leakage.

## 1. 🔹 Direct Answer
Pipeline: run output through topic classifier (or entailment vs task spec), PII detector (regex + NER); fail closed: redact PII or replace response with safe template.

## 2. 🔹 Intuition
Validate machine output before showing users or storing logs.

## 3. 🔹 Deep Dive
Chain: `if not on_topic(output, spec): refuse` → `if pii_detected(output): redact`. Log decisions without leaking PII in logs.

## 4. 🔹 Practical Perspective
Tune thresholds per product; human review for regulated domains.

## 5. 🔹 Code Snippet
```python
def guardrail(output, task_spec, pii_model):
    if not topic_ok(output, task_spec):
        return "[Response withheld: off-topic]"
    if pii_model.has_pii(output):
        return pii_model.redact(output)
    return output
```

## 6. 🔹 Interview Follow-ups
1. Q: Define on-topic?  
   A: Entailment with system task or keyword/embedding similarity to allowed intents.

## 7. 🔹 Common Mistakes
Only checking user input, not model output.

## 8. 🔹 Comparison / Connections
Content moderation, safety filters.

## 9. 🔹 One-line Revision
Guardrails = topic/PII checks on outputs with redaction or refusal.

## 10. 🔹 Difficulty Tag
🟣 Hard

---

# Q22: Build a multi-agent system where agents have different roles and collaborate on a task.

## 1. 🔹 Direct Answer
Define agents with roles (system prompts), a shared message bus or orchestrator, and a handoff protocol: planner → workers → reviewer; each step is an LLM call with bounded iterations.

## 2. 🔹 Intuition
Specialize prompts per role; one process coordinates and merges outputs.

## 3. 🔹 Deep Dive
State: `messages` or structured `Workspace`. Orchestrator routes by phase. Validate final output with a reviewer agent or schema. Cap steps and cost.

## 4. 🔹 Practical Perspective
LangGraph/CrewAI patterns; keep tool permissions per agent.

## 5. 🔹 Code Snippet
```python
def multi_agent(task, agents, orchestrator_llm):
    state = {"task": task, "notes": []}
    plan = agents["planner"].run(task)
    for step in plan.steps:
        out = agents[step.role].run(step.prompt, state)
        state["notes"].append(out)
    final = agents["reviewer"].run("Synthesize and verify.", state)
    return final
```

## 6. 🔹 Interview Follow-ups
1. Q: Conflict between agents?  
   A: Reviewer resolves or escalate to human.

## 7. 🔹 Common Mistakes
Unbounded agent chatter without termination condition.

## 8. 🔹 Comparison / Connections
Multi-agent systems, orchestration.

## 9. 🔹 One-line Revision
Multi-agent = role-specific agents + orchestrator + shared state + reviewer + step/cost limits.

## 10. 🔹 Difficulty Tag
🟣 Hard

---
