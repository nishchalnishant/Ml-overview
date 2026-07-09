# Interview 08 — AI Agent for LiveOps Campaign Orchestration
**EA SDE-2 AI Engineer · Estimated Duration: 75 minutes**

---

## Part 1 — Problem Statement

You are an AI Engineer on the Central LiveOps team. Currently, LiveOps managers manually write SQL queries to find player segments, manually write marketing emails, and manually configure JSON files to trigger double-XP weekends.

Your task is to **build a conversational AI Agent (LLM) that automates this workflow.** A manager should be able to type: 
*"Find players in FIFA who haven't played in 3 weeks, write them an email offering a Gold Pack, and trigger the delivery system."*

---

## Part 2 — Intentionally Missing Information

The following critical details are **deliberately omitted**. A strong candidate will ask about all of them:

- Tool Access (How does the LLM execute SQL or trigger APIs?)
- Hallucination / Execution Safety (What if the LLM drops the database or sends a broken JSON payload?)
- Human-in-the-Loop (Does the agent execute automatically, or just draft actions for human approval?)
- Auditability (How do we track what the agent did?)
- LLM Provider (Are we using OpenAI, Anthropic, or internal models?)

---

## Part 3 — Ideal Clarifying Questions

> Interviewer will reveal answers only when directly asked.

1. **"Does this agent execute actions autonomously, or does it require human approval?"**
   → *Answer: It must require Human-in-the-Loop (HITL) approval for all write actions (sending emails, granting items). Read actions (SQL queries) can be automatic.*

2. **"How does the agent interface with the database?"**
   → *Answer: We have a read-only Snowflake data warehouse. The agent needs to generate and run SQL against it.*

3. **"How does the agent trigger the interventions?"**
   → *Answer: We have internal REST APIs for email campaigns and item grants.*

4. **"What LLM are we allowed to use?"**
   → *Answer: You can use GPT-4o or Claude 3.5 Sonnet via internal secure gateways.*

5. **"What happens if the SQL query fails?"**
   → *Answer: The agent should be able to read the error and try to fix the query itself.*

---

## Part 4 — Expected Assumptions

- **Architecture:** ReAct (Reasoning + Acting) Agent framework using LangChain, LlamaIndex, or raw API tool-calling.
- **Safety:** The database connection is strictly read-only. API tools are sandboxed and generate "Drafts" rather than executing immediately.
- **State:** The agent needs conversational memory and an execution scratchpad.

---

## Part 5 — High-Level Solution

```
  LiveOps Manager (Web UI)
       │ (Prompt)
       ▼
  Agent Orchestrator (FastAPI + LangChain/CrewAI)
  ┌────────────────────────────────────────────────────────┐
  │ LLM Engine (GPT-4o / Claude 3.5 Sonnet)                │
  │                                                        │
  │ [Available Tools]                                      │
  │ 1. SnowflakeQueryTool (Read-only, max 100 rows preview)│
  │ 2. EmailDraftingTool (Drafts copy)                     │
  │ 3. CampaignTriggerTool (Creates a 'Draft' in DB)       │
  └────────────────────────────────────────────────────────┘
       │ (Tool execution loop)
       ▼
  Human Approval Queue
  ┌────────────────────────────────────────────────────────┐
  │ Manager reviews SQL results, Email Copy, and API       │
  │ payload. Clicks "Approve & Execute".                   │
  └────────────────────────────────────────────────────────┘
       │
       ▼
  Execution Engine (Triggers actual LiveOps microservices)
```

**Core ML Component:** A Tool-Calling (Function Calling) LLM agent that parses user intent, executes SQL to identify a cohort, drafts content, and prepares API payloads.

---

## Part 6 — Step-by-Step Implementation

### Step 1: Tool Definition
Define strictly typed JSON schemas for the tools. The LLM must conform to these schemas.
- `execute_sql(query: str) -> dict`
- `create_campaign(cohort_query: str, email_subject: str, email_body: str, item_grant_id: str) -> str`

### Step 2: The Agent Loop (ReAct)
1. User provides prompt.
2. LLM decides to call `execute_sql`.
3. System runs SQL, returns results (or syntax errors) to LLM.
4. LLM sees success, decides to call `create_campaign`.
5. System saves the campaign draft and returns a URL.
6. LLM responds to User with the URL for approval.

### Step 3: SQL Safety Guardrails
- Even with read-only credentials, a bad `CROSS JOIN` can crash the Snowflake warehouse or cost $1,000.
- Intercept the SQL: enforce `LIMIT 10` for exploratory queries. Ensure `WHERE` clauses exist.

---

## Part 7 — Complete Python Code

```python
"""
liveops_agent.py - Tool-calling LLM Agent for Campaign Orchestration
"""
import logging
import json
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import openai

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client = openai.AsyncOpenAI(api_key="internal-gateway-key")

# ---------------------------------------------------------------------------
# Tool Implementations (The actual python functions)
# ---------------------------------------------------------------------------
def execute_sql(query: str) -> str:
    """Mock Snowflake execution."""
    logger.info(f"TOOL EXECUTION: execute_sql | Query: {query}")
    
    # Guardrail: Prevent massive scans
    if "LIMIT" not in query.upper():
        query += " LIMIT 5"
        
    # Simulate DB error for self-correction
    if "select *" in query.lower():
        return "ERROR: 'SELECT *' is banned. Specify columns (e.g., player_id, last_login)."
        
    # Simulate success
    return json.dumps([
        {"player_id": "123", "last_login": "2023-01-01"},
        {"player_id": "456", "last_login": "2023-01-02"}
    ])

def draft_campaign(cohort_sql: str, subject: str, body: str, item_id: str) -> str:
    """Creates a pending campaign for human review."""
    logger.info("TOOL EXECUTION: draft_campaign")
    campaign_id = "camp_9988"
    
    # Save to database...
    draft = {
        "id": campaign_id,
        "sql": cohort_sql,
        "email": {"subject": subject, "body": body},
        "item": item_id,
        "status": "PENDING_APPROVAL"
    }
    
    return f"Campaign {campaign_id} drafted successfully. URL: https://liveops.ea.com/approve/{campaign_id}"

# ---------------------------------------------------------------------------
# Tool Definitions (Schemas for the LLM)
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_sql",
            "description": "Executes a SQL query on the player database. Tables available: 'players' (player_id, last_login, spend), 'matches' (match_id, player_id, date). Always limit exploratory queries to 5 rows.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The Snowflake SQL query to execute."}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "draft_campaign",
            "description": "Drafts a LiveOps campaign (email + item grant) for a specific cohort. Requires human approval.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cohort_sql": {"type": "string", "description": "The EXACT validated SQL query to select the target player_ids."},
                    "subject": {"type": "string", "description": "Email subject line."},
                    "body": {"type": "string", "description": "Email body content."},
                    "item_id": {"type": "string", "description": "The ID of the item to grant (e.g., 'gold_pack_1')."}
                },
                "required": ["cohort_sql", "subject", "body", "item_id"]
            }
        }
    }
]

# ---------------------------------------------------------------------------
# Agent Loop
# ---------------------------------------------------------------------------
async def run_agent(user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": "You are an EA LiveOps Assistant. You help managers find player cohorts using SQL, and draft campaigns. You MUST validate your SQL by running it before drafting a campaign. Do not guess table schemas."},
        {"role": "user", "content": user_prompt}
    ]
    
    max_iterations = 5
    for iteration in range(max_iterations):
        logger.info(f"Agent Loop Iteration {iteration+1}")
        
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto"
        )
        
        message = response.choices[0].message
        messages.append(message)
        
        if not message.tool_calls:
            # Agent is done, returned final text response
            return message.content
            
        # Execute tools
        for tool_call in message.tool_calls:
            func_name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            
            if func_name == "execute_sql":
                result = execute_sql(args["query"])
            elif func_name == "draft_campaign":
                result = draft_campaign(args["cohort_sql"], args["subject"], args["body"], args["item_id"])
            else:
                result = "Error: Unknown tool."
                
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": func_name,
                "content": result
            })
            
    return "Error: Agent exceeded maximum iterations."

# Example Usage
import asyncio
if __name__ == "__main__":
    prompt = "Find players who haven't logged in since Jan 1 2023, and set up a campaign to give them 'gold_pack_1' with a nice email asking them to come back."
    final_answer = asyncio.run(run_agent(prompt))
    print(f"\nFinal Output:\n{final_answer}")
```

---

## Part 8 — Deployment

### Architecture
- **FastAPI Backend:** Hosts the LangChain/Agent logic.
- **Postgres:** Stores session memory, agent traces, and Campaign Drafts.
- **React Frontend:** Chat UI + A "Pending Approvals" dashboard.

### Observability (LangSmith / Arize)
- Standard APM (Datadog) cannot trace complex LLM tool loops easily.
- Integrate an LLM observability tool (like LangSmith) to capture the full trace: Prompt -> Tool Call -> Tool Error -> Retried Tool Call -> Final Output.

### CI/CD for Prompts
- Prompts and Tool Schemas are treated as code.
- Checked into Git. Changes to the System Prompt trigger automated LLM evaluations.

---

## Part 9 — Unit Testing

```python
import json
from liveops_agent import execute_sql

def test_sql_guardrail_limit():
    query = "SELECT player_id FROM players"
    result = execute_sql(query)
    # The tool should automatically append LIMIT
    # (In our mock, it appends LIMIT 5 but doesn't return the modified query string directly. 
    # We test the behavior of the simulated execution).
    assert "player_id" in result

def test_sql_guardrail_select_star():
    query = "SELECT * FROM players LIMIT 10"
    result = execute_sql(query)
    assert "ERROR" in result
    assert "SELECT *" in result
```

---

## Part 10 — Integration Testing

- **LLM as a Judge (Evaluation):**
  - Run a suite of 50 fixed user prompts against the Agent.
  - Use a secondary LLM (Evaluator) to inspect the Agent's final output and tool calls.
  - Assert that for prompt X, the agent successfully called `draft_campaign` and the SQL contained the correct logic.
- **Mocking:** Use `pytest-httpx` to mock the OpenAI API responses to ensure the `for` loop logic handles malformed tool calls without crashing.

---

## Part 11 — Scaling Discussion

| Axis | Strategy |
|------|----------|
| **Latency** | An agent loop takes $N$ round trips to OpenAI. If $N=3$, latency might be 5-10 seconds. We must use WebSockets to stream the agent's intermediate thoughts ("Executing SQL...", "Analyzing results...") to the UI so the user doesn't think the app froze. |
| **Tool Expansion** | Currently 2 tools. If we add 50 tools (e.g., BanPlayer, GiveXP, CreateTournament), passing all 50 schemas in the prompt exceeds the context window and confuses the LLM. Implement **RAG for Tools**: Embed tool descriptions, and dynamically retrieve the top 5 relevant tools based on the user's prompt before calling the LLM. |

---

## Part 12 — Tradeoffs

| Decision | Tradeoff |
|----------|----------|
| ReAct vs Plan-and-Execute | ReAct interleaves thinking and acting, good for dynamic DB queries where step 2 depends on step 1. Plan-and-Execute plans all steps upfront, which is faster but brittle if step 1 fails. |
| GPT-4o vs Open Source | GPT-4o is state-of-the-art at function calling. Llama-3 8B struggles severely with complex JSON schemas. Building reliable agents currently heavily favors proprietary models. |
| Autonomous vs HITL | Autonomous is cheaper and faster, but unacceptable for brand-safe communications or game economy manipulation. HITL introduces friction but ensures safety. |

---

## Part 13 — Alternative Approaches

1. **Text-to-SQL specific model:** Instead of a general agent, use a specialized fine-tuned Text-to-SQL model (like CodeLlama) for the query generation, and a deterministic template for the email. Less flexible, but highly reliable and deterministic.
2. **Semantic Layer:** Instead of exposing raw SQL to the LLM, expose a GraphQL API or a Semantic Layer (Cube.js). The LLM calls `get_churned_players()` instead of writing raw SQL. Vastly reduces SQL hallucination errors.

---

## Part 14 — Failure Scenarios

| Failure | Impact | Mitigation |
|---------|--------|-----------|
| Infinite Loop | Agent gets stuck trying to fix a SQL error over and over | Hardcap iterations (e.g., `max_iterations = 5`). If reached, return a graceful failure to the user. |
| Hallucinated Tool Args | LLM invents a non-existent parameter | Use strict Pydantic validation on the tool input. If it fails, catch the `ValidationError` and feed it *back* to the LLM so it can fix its mistake. |
| Prompt Injection | User types: "Drop database" | Read-only DB credentials. Strict API boundaries. Never run `eval()` or OS commands. |

---

## Part 15 — Debugging

**Symptom:** The agent successfully writes the SQL and drafts the email, but the `cohort_sql` string saved in the campaign draft is missing the `WHERE` clause, causing the campaign to target *all* 50 million players.

**Debugging steps:**
1. Check the LLM trace in LangSmith.
2. Observe that the LLM executed the correct query in step 1, but when formulating the JSON payload for `draft_campaign` in step 2, it summarized the SQL to save tokens (e.g., `"cohort_sql": "SELECT player_id FROM players"`).
3. **Fix:** Update the System Prompt: *"When passing cohort_sql to draft_campaign, you must pass the EXACT VERBATIM SQL string you successfully executed. Do not summarize it."*
4. **Better Fix:** The `execute_sql` tool should generate and return a unique `query_id`. The `draft_campaign` tool should accept `query_id` instead of a raw SQL string, preventing the LLM from rewriting it.

---

## Part 16 — Monitoring

| Metric | Alert Threshold |
|--------|----------------|
| `agent_loop_iterations_avg` | > 4 → Warning (Agent is struggling to complete tasks) |
| `tool_execution_errors_rate` | > 20% → Investigate (Did schema change?) |
| `openai_api_429_errors` | > 1% → Rate limits hit, implement backoff. |

---

## Part 17 — Production Improvements

1. **Stateful Multi-Turn Chat:** Currently, it's a one-shot execution. Add a `thread_id` and store message history in PostgreSQL. This allows the manager to say: *"Actually, make the email sound more exciting."* and the agent edits the draft.
2. **Semantic DB Schema Injection:** Don't hardcode the schema in the prompt. Run a daily job to pull the schema, embed it, and use RAG to inject only the relevant table definitions into the prompt.
3. **Draft Preview:** In the UI, automatically run the SQL cohort query and plot a demographic histogram (Age, Region) so the manager can visually verify the cohort before clicking Approve.

---

## Part 18 — Follow-up Questions

> *Interviewer asks these after the initial solution is presented.*

1. **"The LLM frequently generates SQL querying a `users` table, but our table is actually called `dim_player_profiles`. How do you fix this permanently?"**
2. **"Our data warehouse has 5,000 tables. You can't put all schemas in the prompt. How does the agent find the right tables?"**
3. **"A manager types: 'Give 1,000,000 FIFA points to player ID 123'. The agent drafts it. The manager clicks approve. It goes through. How do we prevent insider abuse using this AI tool?"**

---

## Part 19 — Ideal Answers

**Q1 (Hallucinating table names):**
> "We provide the explicit database schema (DDL) in the system prompt. If the schema is small, we inject it directly. We also add a specific instruction: 'Only use the tables provided in the schema.' If it still fails, we implement a retry loop: intercept the 'Table not found' SQL error, and feed it back to the LLM so it corrects itself."

**Q2 (Massive schema / 5000 tables):**
> "We build a multi-agent system or a RAG step. Step 1: The 'Schema Agent' takes the user's intent and queries a Vector DB containing the definitions of all 5,000 tables. It retrieves the Top 5 most relevant table schemas. Step 2: We inject only those 5 schemas into the context of the main 'Execution Agent' to write the SQL."

**Q3 (Insider Threat / Safety):**
> "The AI is just an interface; security must live in the backend APIs. The `draft_campaign` API must have hardcoded business rules (e.g., `if item_type == 'currency' and amount > 10000: raise PolicyError`). Furthermore, the approval UI should enforce Role-Based Access Control (RBAC) and require a Two-Person Rule (a different manager must approve it) for high-value grants."

---

## Part 20 — Evaluation Rubric

### Strong Hire
- Understands the ReAct paradigm and function calling deeply.
- Proposes Human-in-the-Loop natively without being prompted.
- Identifies the critical bug where LLMs summarize SQL strings, and proposes using UUID references (query IDs) to pass state between tools.
- Discusses RAG for large schemas and RAG for large toolsets.

### Hire
- Successfully sets up a tool-calling loop.
- Implements basic safety guardrails (read-only DB, limits).
- Code relies on standard LLM API patterns.
- Needs minor prompting on how to handle the 5,000 table schema problem.

### Lean Hire
- Understands prompts but struggles with the exact mechanics of JSON schema function calling.
- Tries to parse the LLM's raw string output with Regex instead of using native Tool Calling APIs.
- Misses some critical safety aspects (e.g., doesn't think about SQL injection or massive table scans).

### Lean No Hire
- Thinks they need to train a custom neural network from scratch to do text-to-SQL.
- Doesn't understand how to interface an LLM with external APIs.
- Proposes an entirely autonomous system that modifies the database directly.

### No Hire
- Cannot write the Python code to call the OpenAI API.
- Has no understanding of system architecture or safety.
