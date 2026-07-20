# Interview 08 — LiveOps Campaign Agent (Cheat Sheet)

LiveOps managers manually write SQL, emails, and JSON configs to run campaigns (e.g. double-XP weekends). Build a conversational LLM agent that finds a player cohort, drafts an email, and prepares an item-grant/campaign trigger from a single natural-language prompt.

## Core Architecture
- LiveOps Manager UI → Agent Orchestrator (FastAPI + tool-calling loop).
- LLM Engine = **ReAct-style tool-calling agent** (GPT-4o/Claude) — chosen because it interleaves reasoning and DB querying, letting step 2 depend on step 1's results.
- Tools: `execute_sql` (read-only, row-limited), `draft_campaign` (writes a PENDING_APPROVAL record, never executes).
- SQL guardrails: enforce `LIMIT`, ban `SELECT *`, block missing `WHERE`.
- Human Approval Queue: manager reviews SQL cohort, email copy, API payload before "Approve & Execute."
- Execution Engine: only fires after explicit human approval, calls real LiveOps microservices.
- Observability via LLM tracing (LangSmith/Arize) — standard APM can't trace multi-step tool loops.

## Talking Points That Signal Seniority
- Proactively propose Human-in-the-Loop approval for all write actions, without being asked.
- Flag that LLMs will silently **summarize/rewrite SQL strings** when passing them between tool calls — fix by having `execute_sql` return a `query_id` and have `draft_campaign` accept that ID, not raw SQL text.
- Note that 50+ tools or 5,000+ tables blow the context window — propose **RAG for tools** and **RAG for schema** (embed tool/table descriptions, retrieve top-k relevant ones per prompt).
- Say safety must live in the backend, not the prompt: hardcode business-rule guardrails (e.g. currency-grant caps) in the `draft_campaign` API itself.
- Propose a Two-Person Rule / RBAC on the approval UI for high-value grants — treats the LLM as just an interface, not a trust boundary.
- Mention streaming intermediate agent steps ("Executing SQL...") over WebSockets so multi-round-trip latency (5-10s) doesn't read as a frozen UI.
- Suggest treating prompts and tool schemas as versioned code in git, gated by automated LLM evals on change.
- Mention a "Draft Preview" — auto-run the cohort SQL and show a demographic histogram before approval, so mistakes are visually obvious pre-execution.

## Top 3 Tradeoffs
- **ReAct vs Plan-and-Execute** — ReAct adapts step-by-step when query 2 depends on query 1's result; Plan-and-Execute is faster but brittle if an early step fails.
- **GPT-4o vs open-source (Llama-3)** — proprietary models are meaningfully more reliable at complex JSON function-calling; open models degrade fast on schema adherence.
- **Autonomous vs HITL** — autonomous is cheaper/faster but unacceptable for brand-safe comms or economy-affecting grants; HITL adds friction but is the only safe default here.

## Biggest Pitfall
Proposing a fully autonomous agent that writes to the database/production systems directly (or trying to solve this by training a custom text-to-SQL model from scratch) — signals no grasp of LLM safety boundaries and is an instant No Hire.
