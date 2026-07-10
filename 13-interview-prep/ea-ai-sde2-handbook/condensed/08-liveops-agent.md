# Interview 08 — LiveOps Campaign Agent (Cheat Sheet)

LiveOps managers manually write SQL, emails, and JSON configs to run campaigns (e.g. double-XP weekends). Build a conversational LLM agent that finds a player cohort, drafts an email, and prepares an item-grant/campaign trigger from a single natural-language prompt.

## Clarifying Questions to Ask
- Autonomous execution or human approval? → HITL required for all writes (email send, item grant); reads (SQL) can auto-run.
- How does agent hit the DB? → Read-only Snowflake warehouse; agent generates SQL against it.
- How does agent trigger actions? → Internal REST APIs for email campaigns + item grants.
- Which LLM is allowed? → GPT-4o or Claude 3.5 Sonnet via internal secure gateway.
- What happens on SQL failure? → Agent should read the error and self-correct the query.

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

## Toughest Follow-ups
**"LLM hallucinates table name `users` instead of `dim_player_profiles` — permanent fix?"**
Inject explicit schema/DDL in the system prompt with an instruction to only use listed tables. Add a self-correction loop: catch "table not found" SQL errors and feed them back to the LLM to retry. Doesn't require retraining — it's a context + feedback-loop problem.

**"Warehouse has 5,000 tables — can't fit all schemas in the prompt. How does the agent find the right ones?"**
Two-stage pattern: a "Schema Agent" embeds the user's intent, retrieves top-5 relevant table schemas from a vector DB, then injects only those into the main "Execution Agent" that writes the SQL. This is RAG applied to schema retrieval, not just documents.

**"Manager approves a request to grant 1,000,000 points to one player — how do you prevent insider abuse?"**
The AI is just an interface; real controls belong in the backend API — hardcoded policy checks (e.g. reject currency grants over a threshold) that raise regardless of what the LLM drafted, plus RBAC and a two-person approval rule for high-value actions.

## Biggest Pitfall
Proposing a fully autonomous agent that writes to the database/production systems directly (or trying to solve this by training a custom text-to-SQL model from scratch) — signals no grasp of LLM safety boundaries and is an instant No Hire.
