---
module: Practice
topic: System Design Mocks
subtopic: LLM Serving
status: unread
tags: [practice, system-design, mock-interview, llm-serving, inference]
---
# LLM Serving Mock Scripts

Four scripts targeting AI-infrastructure and inference-data-plane rounds. Background:
[20-ai-infrastructure/](../../20-ai-infrastructure/).

These rounds differ from classical ML design in one specific way: **the numbers are
checkable**. An interviewer can and will ask you to compute KV cache size or all-reduce
volume on the spot, and a wrong order of magnitude ends the round. Arithmetic is part of
the rubric here in a way it is not for a recsys design.

---

## Mock 1 — Multi-tenant inference platform

> **Prompt (read at 0:00).** We serve a 70B model to internal teams. Some send 200-token
> chat turns, some send 100k-token document analyses. Today one team's batch job makes
> everyone else's latency unusable. You have 8×H100 (80 GB) nodes. Design the platform.

<details>
<summary><b>Stage 1 — open at 0:15</b></summary>

1. "How many GPUs for one replica of a 70B model? Show the arithmetic."
2. "Of the 80 GB per GPU, how much is left for KV cache after weights?"
3. "How many concurrent 100k-token requests fit in that?"

**Listening for:** 70B at bf16 is ~140 GB of weights, so two GPUs minimum, realistically
four with activation and KV headroom. KV per token is `2 × layers × kv_heads × head_dim ×
bytes` — for a 70B GQA model roughly 0.3 MB/token, so 100k tokens is around 30 GB for a
*single* request. That number is the entire point of the mock: it explains why one team's
long-context job destroys everyone else's latency, and a candidate who computes it has
already found the root cause.

Do not accept "we'll use vLLM" as a substitute for the arithmetic.
</details>

<details>
<summary><b>Stage 2 — open at 0:25</b></summary>

> "Given that, how do you stop the long-context team from starving the chat team?"

Then:
> "You proposed separate pools. Now the chat pool is idle at night while the batch pool
> is saturated. Defend the waste."

**Listening for:** the isolation mechanisms — separate replica pools, per-tenant KV
budgets, admission control on estimated KV footprint, priority scheduling with
preemption, and request-length routing. See
[20-ai-infrastructure/06-multi-tenancy-and-isolation.md](../../20-ai-infrastructure/06-multi-tenancy-and-isolation.md).

The defense question is the real test: physical pool separation buys hard isolation and
costs utilization. The strong answer states that tradeoff explicitly and proposes
reclaiming the idle capacity for preemptible batch work with checkpointing, rather than
either defending the waste blindly or immediately collapsing the pools.

**Calibration:**
- *Weak:* "Rate limit per tenant." (Requests-per-second does not bound KV usage — one
  100k-token request costs more than a thousand chat turns.)
- *Solid:* separate pools plus token-based limits.
- *Strong:* admission control keyed on *estimated KV footprint*, dual limits on tokens
  per minute and concurrency, and preemptible batch backfill on idle interactive capacity.
</details>

<details>
<summary><b>Stage 3 — open at 0:35</b></summary>

> "Your control plane — the thing that tracks replica health and routes — goes down.
> What happens to in-flight requests?"

**Listening for:** the control-plane / data-plane invariant. The data plane must keep
serving with its last known good routing state; a design where every request synchronously
consults the controller has made the controller a hard dependency on the request path,
which is the anti-pattern this question exists to surface. Routers should cache endpoint
state and degrade to stale routing, not to failure.
</details>

---

## Mock 2 — RAG at scale

> **Prompt (read at 0:00).** 50 million documents, 10,000 queries per second, p95 under
> 800 ms including generation. Answers must cite sources and must not fabricate. Design it.

<details>
<summary><b>Stage 1 — open at 0:15</b></summary>

1. "Break the 800 ms down by stage."
2. "50M documents — what is your index, and what does it cost in RAM?"
3. "Chunking strategy, and what breaks with the naive one?"

**Listening for:** the budget roughly splits into embedding the query (~10 ms), retrieval
(~50 ms), reranking (~100 ms if used), and generation as the remaining bulk — which means
generation length is the dominant lever and streaming TTFT is what the user actually
perceives. On the index: HNSW is fast but memory-hungry (a rough gigabyte per million
768-dim float32 vectors before graph overhead, so 50M vectors is well over 100 GB and
demands sharding or product quantization). On chunking: fixed-size splits cut sentences
and tables in half; the follow-up is that retrieval quality is usually bounded by chunking
long before it is bounded by the embedding model.
</details>

<details>
<summary><b>Stage 2 — open at 0:25</b></summary>

> "Users report the system confidently cites a document that does not say what the
> answer claims. Diagnose."

Then:
> "How would you have caught that before users did?"

**Listening for:** decomposing retrieval failure from generation failure — did the right
chunk get retrieved (measurable: recall@k against a labeled set) and did the model stay
faithful to it (measurable: groundedness/faithfulness scoring). Conflating the two is the
common weak answer. The second question wants an offline eval set with per-stage metrics
plus an online sample scored by an LLM judge with human spot-checks, not "more testing."
See [11-llm-applications/](../../11-llm-applications/).
</details>

<details>
<summary><b>Stage 3 — open at 0:35</b></summary>

> "Cost is $40k/month and the CFO wants it halved. What do you cut, in order?"

**Listening for:** an *ordered* list with the quality cost of each step named — cache
semantically similar queries first (free if hit rates are real), then shrink the generator
or quantize it, then drop the reranker, then reduce k, then quantize the index. The strong
answer starts by asking what fraction of the spend is retrieval versus generation, because
optimizing the wrong one is the usual mistake, and notes that prefix caching over shared
system prompts is often the single largest free win.
</details>

---

## Mock 3 — KV cache tier

> **Prompt (read at 0:00).** Our chat product has long multi-turn conversations. Users
> come back hours later and we recompute the entire history's prefill every time. Design
> something better.

<details>
<summary><b>Stage 1 — open at 0:15</b></summary>

1. "What exactly are you caching, and how big is it per conversation?"
2. "Where does it live — GPU HBM, host RAM, or remote storage? What does each cost you?"
3. "What is the cache key?"

**Listening for:** the storage hierarchy with rough bandwidth figures — HBM in TB/s, host
RAM over PCIe in tens of GB/s, network storage an order of magnitude below that — and the
break-even reasoning: fetching a KV block is worth it only when it is faster than
recomputing the prefill for those tokens. That comparison is the core of the design. The
key must cover the exact token prefix *and* the model version, since KV from a different
model or quantization is not reusable.
</details>

<details>
<summary><b>Stage 2 — open at 0:25</b></summary>

> "Two users send the same system prompt but different questions. What do you share?"

Then:
> "One of those users is at a different company than the other. Now what?"

**Listening for:** prefix sharing on the common prefix — a radix-tree structure over
prefixes rather than flat exact-match keys, so partial prefixes hit. The second question
is a security question: a shared cross-tenant prefix cache is a timing side channel, since
a fast TTFT reveals that someone else has already submitted that exact prefix. Mitigation
is namespacing the cache per tenant, which costs hit rate. A candidate who does not see
the side channel has missed the point of the follow-up.
</details>

<details>
<summary><b>Stage 3 — open at 0:35</b></summary>

> "Routing: a request arrives whose prefix is cached on replica 3, but replica 3 is the
> most loaded. Where do you send it?"

**Listening for:** the score is a weighted combination of prefix-match length against load
factor, not a hard preference for either. Always honoring cache affinity produces a
hotspot; ignoring it wastes the cache entirely. The strong answer adds that the weights
depend on the ratio of prefill cost saved to expected queuing delay, so the right tradeoff
shifts with prompt length — long prefixes justify tolerating more load.
</details>

---

## Mock 4 — Agent platform

> **Prompt (read at 0:00).** We are building a platform where internal teams deploy
> LLM agents that call tools, run for minutes, and sometimes call other agents. Design
> the serving and orchestration layer.

<details>
<summary><b>Stage 1 — open at 0:15</b></summary>

1. "An agent turn takes four minutes and makes twelve LLM calls. What does that do to
   your connection model and your capacity planning?"
2. "How does an agent discover and call a tool? What about another team's agent?"
3. "One agent enters a loop and calls a tool 10,000 times. What stops it?"

**Listening for:** long-running work cannot sit on a synchronous request — you need a task
abstraction with streaming status updates, and capacity is governed by concurrent *agent
sessions* rather than requests per second. Tools versus peer agents is the MCP/A2A
distinction: MCP is vertical, standardizing an agent's access to tools and data sources;
A2A is horizontal, letting agents delegate across org boundaries with capability
discovery. Loop protection needs step budgets, wall-clock deadlines, and per-tool call
quotas — all three, since any one alone is escapable.
</details>

<details>
<summary><b>Stage 2 — open at 0:25</b></summary>

> "A user cancels at minute three. What actually happens on the GPU?"

Then:
> "Your bill shows 30% of tokens are generated for requests nobody is waiting on."

**Listening for:** cancellation must propagate all the way to the inference engine so the
sequence is evicted and its KV blocks are freed — a client disconnect that does not reach
the scheduler means you keep generating tokens into a closed socket, which is exactly the
30% on the bill. In gRPC this is context cancellation flowing through every hop; the weak
answer stops at "we close the connection."
</details>

<details>
<summary><b>Stage 3 — open at 0:35</b></summary>

> "An agent is slow. Tell me how you find out why."

**Listening for:** distributed tracing with a trace ID spanning the agent's whole session
across every LLM call and tool invocation — without it, the twelve calls are twelve
unrelated log lines. The breakdown must separate queueing from prefill from decode from
tool latency, because the fixes are entirely different: queueing means add capacity or
route better, prefill means the prompt is too long or the cache is missing, decode means
the output is long or batching is poor, tool latency is someone else's service. Also
listen for goodput — SLO-conformant throughput — rather than raw tokens per second, and
for KV occupancy as the saturation signal, since GPU utilization stays near 100% during
decode regardless of how well the system is actually doing.
</details>
