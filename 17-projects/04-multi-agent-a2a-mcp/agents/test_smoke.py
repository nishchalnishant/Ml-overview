"""Smoke tests for the A2A protocol wiring (agent card + task lifecycle),
using FastAPI's in-process TestClient — no live server or network needed.

Run from 17-projects/04-multi-agent-a2a-mcp: pytest agents/test_smoke.py -v
"""
from fastapi.testclient import TestClient

from a2a_protocol import A2AMessage, AgentCard, AgentSkill, MessagePart, Task, make_agent_app

CARD = AgentCard(
    name="test-agent",
    description="A test agent",
    url="http://testserver",
    skills=[AgentSkill(id="echo", name="Echo", description="Echoes the task back")],
)


async def echo_handler(task: Task) -> Task:
    task.status = "completed"
    return task


def build_client() -> TestClient:
    app = make_agent_app(CARD, echo_handler)
    return TestClient(app)


def test_agent_card_endpoint():
    client = build_client()
    resp = client.get("/.well-known/agent-card.json")
    assert resp.status_code == 200
    body = resp.json()
    assert body["name"] == "test-agent"
    assert body["skills"][0]["id"] == "echo"


def test_task_submission_completes():
    client = build_client()
    task = Task(
        id="t1",
        message=A2AMessage(parts=[MessagePart(kind="text", content="hello")]),
        goal="say hello",
    )
    resp = client.post("/tasks", json=task.model_dump())
    assert resp.status_code == 200
    result = resp.json()
    assert result["status"] == "completed"
    assert result["id"] == "t1"


def test_task_defaults_to_submitted_status():
    task = Task(
        id="t2",
        message=A2AMessage(parts=[MessagePart(content="x")]),
        goal="test default",
    )
    assert task.status == "submitted"
