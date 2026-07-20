"""
A minimal MCP Client wrapper.

Each Agent process owns its own instance of this class per MCP server it
needs (i.e. each agent gets its own restricted MCP client(s), connected
only to the servers relevant to its job — the "Hierarchical Multi-Client"
pattern). This client spawns the target server as a stdio subprocess,
performs the MCP handshake, and exposes list_tools()/call_tool() to the
agent's decision loop.
"""
import sys
from contextlib import AsyncExitStack
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVERS_DIR = Path(__file__).resolve().parent.parent / "mcp_servers"


class MCPClient:
    """One MCP client connection to one MCP server (spawned via stdio)."""

    def __init__(self, server_script: str):
        self.server_script = str(SERVERS_DIR / server_script)
        self._stack = AsyncExitStack()
        self.session: ClientSession | None = None

    async def connect(self):
        params = StdioServerParameters(command=sys.executable, args=[self.server_script])
        read, write = await self._stack.enter_async_context(stdio_client(params))
        self.session = await self._stack.enter_async_context(ClientSession(read, write))
        await self.session.initialize()
        return self

    async def list_tools(self) -> list[str]:
        result = await self.session.list_tools()
        return [t.name for t in result.tools]

    async def call_tool(self, name: str, arguments: dict) -> str:
        result = await self.session.call_tool(name, arguments)
        parts = []
        for block in result.content:
            if hasattr(block, "text"):
                parts.append(block.text)
        return "\n".join(parts)

    async def close(self):
        await self._stack.aclose()


class MultiServerMCPClient:
    """Fans out across several MCPClient connections and routes calls by tool name."""

    def __init__(self, server_scripts: list[str]):
        self.clients = [MCPClient(s) for s in server_scripts]
        self._tool_owner: dict[str, MCPClient] = {}

    async def connect_all(self):
        for client in self.clients:
            await client.connect()
            for tool_name in await client.list_tools():
                self._tool_owner[tool_name] = client
        return self

    def available_tools(self) -> list[str]:
        return list(self._tool_owner.keys())

    async def call_tool(self, name: str, arguments: dict) -> str:
        if name not in self._tool_owner:
            raise ValueError(f"No connected MCP server exposes tool '{name}'")
        return await self._tool_owner[name].call_tool(name, arguments)

    async def close_all(self):
        for client in self.clients:
            await client.close()
