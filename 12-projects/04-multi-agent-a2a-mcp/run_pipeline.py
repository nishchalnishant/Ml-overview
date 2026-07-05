"""
Pipeline driver.

Starts all three agent processes (each an independent OS process, each
running its own MCP client(s) + FastAPI/A2A server), waits for them to
come up, then submits the initial A2A Task to the Research Agent —
which kicks off the Research -> Analysis -> Report chain, each hop a
real A2A message over local HTTP.

Usage:
    python run_pipeline.py "Why did churn increase and what should we do about it?"
"""
import asyncio
import subprocess
import sys
import time
import uuid
from pathlib import Path

import httpx

from agents.a2a_protocol import A2AMessage, MessagePart, Task, send_task

AGENTS_DIR = Path(__file__).resolve().parent / "agents"
VENV_PYTHON = Path(__file__).resolve().parent / ".venv" / "bin" / "python"

AGENT_PROCS = [
    ("research_agent.py", 8001),
    ("analysis_agent.py", 8002),
    ("report_agent.py", 8003),
]


def wait_for(url: str, timeout: float = 15.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = httpx.get(url, timeout=1.0)
            if r.status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.3)
    raise RuntimeError(f"Timed out waiting for {url}")


async def main():
    goal = sys.argv[1] if len(sys.argv) > 1 else "Why did churn increase in Q3 and what should we do about it?"

    procs = []
    python_bin = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable
    print("Starting agent processes (Research :8001, Analysis :8002, Report :8003)...\n")
    for script, port in AGENT_PROCS:
        proc = subprocess.Popen(
            [python_bin, str(AGENTS_DIR / script), str(port)],
            cwd=str(AGENTS_DIR),
        )
        procs.append(proc)

    try:
        for _, port in AGENT_PROCS:
            wait_for(f"http://127.0.0.1:{port}/.well-known/agent-card.json")
        print("All three agents are up and their Agent Cards are discoverable.\n")

        initial_task = Task(
            id=str(uuid.uuid4()),
            goal=goal,
            message=A2AMessage(parts=[MessagePart(kind="text", content=goal)]),
        )
        print(f"Submitting initial A2A task to Research Agent. Goal: '{goal}'\n")
        final = await send_task("http://127.0.0.1:8001", initial_task)

        print("\n=== Pipeline complete ===")
        for part in final.message.parts:
            if part.kind == "text":
                print(f"\n{part.content}")
        print("\nSee output/quarterly_report.md for the final report.")

    finally:
        print("\nShutting down agent processes...")
        for proc in procs:
            proc.terminate()
        for proc in procs:
            proc.wait(timeout=5)


if __name__ == "__main__":
    asyncio.run(main())
