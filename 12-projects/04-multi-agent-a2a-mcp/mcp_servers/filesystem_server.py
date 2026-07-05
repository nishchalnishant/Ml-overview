"""
MCP Server #3 — Filesystem / Report server.

Exposes the Report Agent's "external world": the ability to write the
final deliverable to disk. This is the only server with write access,
kept isolated in its own process/tool surface. Transport: stdio.
"""
from pathlib import Path

from mcp.server.fastmcp import FastMCP

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

mcp = FastMCP("filesystem-server")


@mcp.tool()
def write_report(filename: str, content: str) -> str:
    """Write the final report content to a file in the project's output/ directory.

    filename should be a plain name like 'quarterly_report.md' (no path traversal).
    """
    safe_name = Path(filename).name
    path = OUTPUT_DIR / safe_name
    path.write_text(content)
    return f"Report written to {path}"


@mcp.tool()
def list_reports() -> str:
    """List all report files currently in the output directory."""
    files = sorted(p.name for p in OUTPUT_DIR.glob("*"))
    return "\n".join(files) if files else "No reports written yet."


if __name__ == "__main__":
    mcp.run(transport="stdio")
