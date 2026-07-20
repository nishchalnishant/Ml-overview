"""
MCP Server #1 — Notes / Search server.

Exposes the Research Agent's "external world": a flat-file corpus of raw
company notes it can keyword-search and read, as if it were a search API
or a knowledge base. Transport: stdio (spawned as a subprocess by the
Research Agent's MCP client).
"""
from pathlib import Path

from mcp.server.fastmcp import FastMCP

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "notes.txt"

mcp = FastMCP("notes-server")


@mcp.tool()
def search_notes(keyword: str) -> str:
    """Search the company notes corpus for lines containing a keyword (case-insensitive).

    Returns matching lines, each prefixed with its line number.
    """
    lines = DATA_FILE.read_text().splitlines()
    hits = [
        f"{i + 1}: {line}"
        for i, line in enumerate(lines)
        if keyword.lower() in line.lower()
    ]
    if not hits:
        return f"No notes found matching '{keyword}'."
    return "\n".join(hits)


@mcp.tool()
def read_all_notes() -> str:
    """Return the full raw text of the company notes corpus."""
    return DATA_FILE.read_text()


if __name__ == "__main__":
    mcp.run(transport="stdio")
