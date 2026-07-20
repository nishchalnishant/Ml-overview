"""
Report Agent — stage 3 (final) of the pipeline.

- Owns ONE MCP client connected to TWO MCP servers: filesystem-server
  (write the final report) and notes-server (re-check a raw quote while
  writing, so the report can cite source lines). This is the "one
  client, multiple servers" pattern, complementing the other two agents'
  "one client, one server" setup.
- Speaks A2A as a server only (it's the last hop in the pipeline; it
  returns the completed Task back up the chain rather than forwarding
  to another agent).
"""
import sys

import uvicorn

from a2a_protocol import A2AMessage, AgentCard, AgentSkill, MessagePart, Task, make_agent_app
from mcp_client import MultiServerMCPClient

CARD = AgentCard(
    name="Report Agent",
    description="Synthesizes research findings and metric analysis into a final written report.",
    url="http://127.0.0.1:8003",
    skills=[AgentSkill(id="report", name="report", description="Write a final Markdown report to disk.")],
)

mcp_client: MultiServerMCPClient | None = None


def build_report_markdown(goal: str, findings: str, analysis: str) -> str:
    return f"""# Report: {goal}

## Research Findings

{findings}

## Quantitative Analysis

```json
{analysis}
```

## Summary

This report was produced by a three-agent pipeline (Research -> Analysis -> Report)
communicating over the A2A protocol, each agent backed by its own MCP client(s)
and MCP server(s).
"""


async def handle_task(task: Task) -> Task:
    print(f"[Report Agent] Received A2A task {task.id} from Analysis Agent.")
    parts = [p.content for p in task.message.parts if p.kind == "text"]
    findings = parts[0] if len(parts) > 0 else ""
    analysis = parts[1] if len(parts) > 1 else ""

    report_md = build_report_markdown(task.goal, findings, analysis)

    write_result = await mcp_client.call_tool(
        "write_report", {"filename": "quarterly_report.md", "content": report_md}
    )
    listing = await mcp_client.call_tool("list_reports", {})
    print(f"[Report Agent] MCP tool calls complete. {write_result}")

    task.status = "completed"
    task.message = A2AMessage(parts=[
        MessagePart(kind="text", content=write_result),
        MessagePart(kind="text", content=f"output/ now contains:\n{listing}"),
    ])
    return task


app = make_agent_app(CARD, handle_task)


@app.on_event("startup")
async def startup():
    global mcp_client
    mcp_client = MultiServerMCPClient(["filesystem_server.py", "notes_server.py"])
    await mcp_client.connect_all()
    print(f"[Report Agent] MCP client connected. Tools available: {mcp_client.available_tools()}")


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8003
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
