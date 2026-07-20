"""
Minimal A2A (Agent2Agent) protocol implementation.

Real A2A (the open spec Google published in 2025) defines: Agent Cards
(JSON self-description of an agent's skills, published at a well-known
URL), Tasks (units of work with a lifecycle: submitted -> working ->
completed/failed), and Messages (the actual content exchanged, made of
typed Parts). This implements the same three concepts, trimmed to what's
needed for one linear handoff chain, over plain JSON-over-HTTP instead
of the full spec's JSON-RPC/SSE transport.

Every agent process:
  1. Runs a tiny FastAPI server exposing GET /.well-known/agent-card.json
     and POST /tasks.
  2. Uses `send_task()` below as its A2A *client* to hand a task to the
     next agent in the pipeline.

This is deliberately a subset of the real spec — enough to see A2A's
actual wire shape (Agent Card discovery + Task submission + Message
parts), not a spec-complete implementation.
"""
from typing import Any, Literal

import httpx
from fastapi import FastAPI
from pydantic import BaseModel


class AgentSkill(BaseModel):
    id: str
    name: str
    description: str


class AgentCard(BaseModel):
    """A2A Agent Card: an agent's public, discoverable self-description."""
    name: str
    description: str
    url: str
    skills: list[AgentSkill]


class MessagePart(BaseModel):
    kind: Literal["text", "data"] = "text"
    content: Any


class A2AMessage(BaseModel):
    role: Literal["agent"] = "agent"
    parts: list[MessagePart]


class Task(BaseModel):
    """A2A Task: a unit of work handed from one agent to another."""
    id: str
    status: Literal["submitted", "working", "completed", "failed"] = "submitted"
    message: A2AMessage
    goal: str


def make_agent_app(card: AgentCard, handle_task) -> FastAPI:
    """Build the FastAPI app every A2A-speaking agent runs.

    `handle_task(task: Task) -> Task` is the agent's own logic, injected
    by the caller, and is where it uses its MCP client(s) to do the work
    before returning the completed task.
    """
    app = FastAPI(title=card.name)

    @app.get("/.well-known/agent-card.json")
    async def agent_card() -> AgentCard:
        return card

    @app.post("/tasks")
    async def receive_task(task: Task) -> Task:
        task.status = "working"
        result_task = await handle_task(task)
        return result_task

    return app


async def send_task(agent_url: str, task: Task, timeout: float = 60.0) -> Task:
    """A2A client call: discover the peer's Agent Card, then submit a Task to it."""
    async with httpx.AsyncClient(timeout=timeout) as client:
        card_resp = await client.get(f"{agent_url}/.well-known/agent-card.json")
        card_resp.raise_for_status()
        peer_card = AgentCard(**card_resp.json())

        task_resp = await client.post(f"{agent_url}/tasks", json=task.model_dump())
        task_resp.raise_for_status()
        completed = Task(**task_resp.json())

    print(f"[A2A] -> {peer_card.name} ({agent_url}) : task {task.id} -> status={completed.status}")
    return completed
