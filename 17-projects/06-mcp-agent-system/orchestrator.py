"""
Multi-Agent Orchestrator (MCP Client)

This script acts as the central orchestrator and MCP client.
It spawns multiple MCP servers as subprocesses via stdio,
discovers their tools, and executes a simulated multi-agent loop.

If OPENAI_API_KEY is present in the environment (or .env), it will 
run an actual agent loop. Otherwise, it simulates the LLM decisions.
"""
import asyncio
import json
import os
import sys
from contextlib import AsyncExitStack

# Load environment variables (e.g. OPENAI_API_KEY)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from rich.console import Console
from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

console = Console()

def run_simulated_flow(weather_session, db_session):
    """Fallback flow if no LLM API key is provided."""
    console.print("\n[yellow]No OPENAI_API_KEY found. Running simulated agent flow...[/yellow]")
    
    async def _sim():
        console.print("\n[bold cyan]🧠 [Research Agent][/bold cyan] Requesting weather for Berlin...")
        result = await weather_session.call_tool(
            "get_weather", 
            arguments={"latitude": 52.52, "longitude": 13.41}
        )
        console.print(f"   [green]✅ Response:[/green] {result.content[0].text}")
        
        console.print("\n[bold cyan]🧠 [Data Analyst Agent][/bold cyan] Requesting active users from database...")
        db_result = await db_session.call_tool(
            "execute_sql",
            arguments={"query": "SELECT * FROM users WHERE active = 1"}
        )
        console.print(f"   [green]✅ Response:[/green] {db_result.content[0].text}")
        
        console.print("\n[bold red]🧠 [Malicious Agent][/bold red] Attempting SQL injection (DROP TABLE)...")
        hack_result = await db_session.call_tool(
            "execute_sql",
            arguments={"query": "DROP TABLE users;"}
        )
        console.print(f"   [red]⛔ Response:[/red] {hack_result.content[0].text}\n")
    
    return _sim()

async def run_llm_flow(weather_session, db_session, weather_tools, db_tools):
    """Real LLM execution flow using OpenAI API."""
    try:
        from openai import AsyncOpenAI
    except ImportError:
        console.print("[red]OpenAI package not installed. Cannot run LLM flow.[/red]")
        return await run_simulated_flow(weather_session, db_session)

    client = AsyncOpenAI()
    
    # Map MCP tools to OpenAI function calling format
    # In a production system, this mapping is done dynamically based on MCP JSON schema
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a specific latitude and longitude.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "latitude": {"type": "number"},
                        "longitude": {"type": "number"}
                    },
                    "required": ["latitude", "longitude"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "execute_sql",
                "description": "Execute a SELECT query on the users database.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"}
                    },
                    "required": ["query"]
                }
            }
        }
    ]

    messages = [
        {"role": "system", "content": "You are a helpful AI assistant connected to a multi-server MCP network. Use the provided tools to answer the user's questions."},
        {"role": "user", "content": "Can you tell me the weather in Berlin (52.52, 13.41), and also tell me the names of all active users in the engineering department?"}
    ]

    console.print("\n[bold green]Running live LLM agent loop with OpenAI...[/bold green]")
    console.print(f"[dim]User:[/dim] {messages[-1]['content']}\n")

    # Step 1: Send prompt to LLM
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    message = response.choices[0].message
    messages.append(message)
    
    # Step 2: Handle Tool Calls
    if message.tool_calls:
        for tool_call in message.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            
            console.print(f"[cyan]Agent invoking tool:[/cyan] {name}({args})")
            
            # Route to the appropriate MCP session
            if name == "get_weather":
                result = await weather_session.call_tool(name, arguments=args)
            elif name == "execute_sql":
                result = await db_session.call_tool(name, arguments=args)
            else:
                result = "Tool not found."
            
            tool_output = result.content[0].text if hasattr(result, 'content') else str(result)
            console.print(f"[dim]Tool Result:[/dim] {tool_output}")
            
            # Append result back to context
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": name,
                "content": tool_output
            })
            
        # Step 3: Get final response from LLM
        final_response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        console.print(f"\n[bold magenta]Final Agent Response:[/bold magenta]\n{final_response.choices[0].message.content}\n")
    else:
        console.print(f"\n[bold magenta]Agent Response (No tools used):[/bold magenta]\n{message.content}\n")


async def run_multi_agent_system():
    console.print("[bold]🤖 Booting Multi-Agent MCP Orchestrator...[/bold]\n")
    
    # 1. Define Server Parameters
    weather_server_params = StdioServerParameters(command="python", args=["server_weather.py"])
    db_server_params = StdioServerParameters(command="python", args=["server_database.py"])

    async with AsyncExitStack() as stack:
        console.print("🔌 Connecting to MCP Servers (Weather & Database)...")
        
        # 2. Connect to both servers
        read1, write1 = await stack.enter_async_context(stdio_client(weather_server_params))
        read2, write2 = await stack.enter_async_context(stdio_client(db_server_params))
        
        weather_session = await stack.enter_async_context(ClientSession(read1, write1))
        db_session = await stack.enter_async_context(ClientSession(read2, write2))
        
        # Initialize the MCP connections
        await weather_session.initialize()
        await db_session.initialize()
        
        # 3. Discover Tools
        console.print("🔍 Discovering available tools via MCP...")
        weather_tools = await weather_session.list_tools()
        db_tools = await db_session.list_tools()
        
        console.print(f"   ➔ [blue]Weather Server tools:[/blue] {[t.name for t in weather_tools.tools]}")
        console.print(f"   ➔ [blue]Database Server tools:[/blue] {[t.name for t in db_tools.tools]}")
        
        # Run execution flow
        if os.environ.get("OPENAI_API_KEY"):
            await run_llm_flow(weather_session, db_session, weather_tools, db_tools)
        else:
            await run_simulated_flow(weather_session, db_session)
            
        console.print("[bold green]🎉 Orchestrator successfully routed requests through isolated MCP servers.[/bold green]")

if __name__ == "__main__":
    # Ensure we run from the correct directory so the servers can find their files
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    try:
        asyncio.run(run_multi_agent_system())
    except KeyboardInterrupt:
        console.print("\n[red]Exiting...[/red]")
