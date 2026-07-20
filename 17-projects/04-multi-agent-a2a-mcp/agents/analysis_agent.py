"""
Analysis Agent — stage 2 of the pipeline.

- Owns one MCP client, connected to one MCP server: stats-server (its
  restricted "view" of the world — it can compute trends over metrics,
  but cannot read raw notes or write files).
- Speaks A2A as a server: receives the Research Agent's findings as an
  incoming Task.
- Speaks A2A as a client: hands its analysis to the Report Agent.
"""
import sys
import uuid

import uvicorn

from a2a_protocol import A2AMessage, AgentCard, AgentSkill, MessagePart, Task, make_agent_app, send_task
from mcp_client import MultiServerMCPClient

REPORT_AGENT_URL = "http://127.0.0.1:8003"

CARD = AgentCard(
    name="Analysis Agent",
    description="Computes quantitative trends over company metrics to support research findings.",
    url="http://127.0.0.1:8002",
    skills=[AgentSkill(id="analyze", name="analyze", description="Compute trends/projections for relevant metrics.")],
)

mcp_client: MultiServerMCPClient | None = None

METRICS_TO_CHECK = [
    "revenue_growth_pct",
    "churn_rate_pct",
    "csat_score",
    "support_ticket_volume",
    "eu_revenue_share_pct",
]


async def handle_task(task: Task) -> Task:
    print(f"[Analysis Agent] Received A2A task {task.id} from Research Agent.")
    research_findings = "\n".join(p.content for p in task.message.parts if p.kind == "text")

    relevant_metrics = [m for m in METRICS_TO_CHECK if any(
        token in research_findings.lower() for token in m.split("_")
    )] or METRICS_TO_CHECK

    analyses = []
    for metric in relevant_metrics:
        trend_json = await mcp_client.call_tool("compute_trend", {"metric_name": metric})
        analyses.append(trend_json)

    print(f"[Analysis Agent] MCP tool calls complete. Computed trends for {len(analyses)} metrics.")

    analysis_text = "\n".join(analyses)
    task.status = "completed"
    task.message = A2AMessage(parts=[
        MessagePart(kind="text", content=research_findings),
        MessagePart(kind="text", content=analysis_text),
    ])

    next_task = Task(
        id=str(uuid.uuid4()),
        goal=task.goal,
        message=task.message,
    )
    report_result = await send_task(REPORT_AGENT_URL, next_task)

    task.message.parts.append(MessagePart(kind="data", content=report_result.model_dump()))
    return task


app = make_agent_app(CARD, handle_task)


@app.on_event("startup")
async def startup():
    global mcp_client
    mcp_client = MultiServerMCPClient(["stats_server.py"])
    await mcp_client.connect_all()
    print(f"[Analysis Agent] MCP client connected. Tools available: {mcp_client.available_tools()}")


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8002
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
