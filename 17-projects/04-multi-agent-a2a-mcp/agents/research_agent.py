"""
Research Agent — stage 1 of the pipeline.

- Owns one MCP client, connected to one MCP server: notes-server (its
  restricted "view" of the world — it cannot compute stats or write
  files, only search/read raw notes).
- Speaks A2A as a server: exposes an Agent Card and a /tasks endpoint so
  the outer pipeline driver can hand it the initial goal.
- Speaks A2A as a client: once it has gathered raw findings, it hands
  a new Task to the Analysis Agent over HTTP.
"""
import asyncio
import re
import sys
import uuid

import uvicorn

from a2a_protocol import A2AMessage, AgentCard, AgentSkill, MessagePart, Task, make_agent_app, send_task
from mcp_client import MultiServerMCPClient

ANALYSIS_AGENT_URL = "http://127.0.0.1:8002"

CARD = AgentCard(
    name="Research Agent",
    description="Gathers raw findings from company notes relevant to a research goal.",
    url="http://127.0.0.1:8001",
    skills=[AgentSkill(id="research", name="research", description="Search and summarize raw notes on a topic.")],
)

mcp_client: MultiServerMCPClient | None = None


def extract_keywords(goal: str) -> list[str]:
    stopwords = {"the", "a", "an", "on", "of", "in", "and", "for", "to", "why", "what", "is", "are", "report"}
    words = re.findall(r"[a-zA-Z]+", goal.lower())
    return [w for w in words if w not in stopwords and len(w) > 2]


async def handle_task(task: Task) -> Task:
    goal = task.goal
    print(f"[Research Agent] Received A2A task {task.id}: goal='{goal}'")

    keywords = extract_keywords(goal) or ["revenue"]
    findings = []
    for kw in keywords:
        result = await mcp_client.call_tool("search_notes", {"keyword": kw})
        if "No notes found" not in result:
            findings.append(f"-- matches for '{kw}' --\n{result}")

    if not findings:
        full_text = await mcp_client.call_tool("read_all_notes", {})
        findings.append(full_text)

    findings_text = "\n\n".join(findings)
    print(f"[Research Agent] MCP tool calls complete. Found {len(findings)} keyword hit groups.")

    task.status = "completed"
    task.message = A2AMessage(parts=[MessagePart(kind="text", content=findings_text)])

    next_task = Task(
        id=str(uuid.uuid4()),
        goal=goal,
        message=A2AMessage(parts=[MessagePart(kind="text", content=findings_text)]),
    )
    analysis_result = await send_task(ANALYSIS_AGENT_URL, next_task)

    task.message.parts.append(MessagePart(kind="text", content="[Forwarded to Analysis Agent via A2A; see downstream task for final result.]"))
    task.message.parts.append(MessagePart(kind="data", content=analysis_result.model_dump()))
    return task


app = make_agent_app(CARD, handle_task)


@app.on_event("startup")
async def startup():
    global mcp_client
    mcp_client = MultiServerMCPClient(["notes_server.py"])
    await mcp_client.connect_all()
    print(f"[Research Agent] MCP client connected. Tools available: {mcp_client.available_tools()}")


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8001
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
