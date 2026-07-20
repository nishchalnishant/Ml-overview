"""
MCP Server: Weather & Search API
Exposes external data sources to the LLM agent via MCP.

This server demonstrates how to wrap external REST APIs into standard MCP tools.
"""
from mcp.server.fastmcp import FastMCP
import httpx
import json

# Initialize FastMCP server
mcp = FastMCP("Weather & Search Server")

@mcp.tool()
def get_weather(latitude: float, longitude: float) -> str:
    """
    Get the current weather for a specific latitude and longitude using Open-Meteo API.
    This acts as an external real-world data source for the agent.
    """
    url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current_weather=true"
    
    try:
        response = httpx.get(url, timeout=5.0)
        if response.status_code == 200:
            data = response.json()
            current = data.get("current_weather", {})
            return json.dumps({
                "temperature_celsius": current.get("temperature"),
                "windspeed_kmh": current.get("windspeed"),
                "time": current.get("time")
            })
        return f"API returned status code: {response.status_code}"
    except Exception as e:
        return f"Could not fetch weather data. Error: {str(e)}"

@mcp.tool()
def search_web(query: str) -> str:
    """
    Simulates a web search API.
    In a production system, this would call Tavily, Brave, or Google Search API.
    """
    # Mocked response for demonstration purposes
    results = [
        {"title": f"Result 1 for {query}", "snippet": "This is a highly relevant result containing exact facts."},
        {"title": f"Result 2 for {query}", "snippet": "Alternative perspective on the topic. Mentions that Python is great for ML."}
    ]
    return json.dumps(results)

if __name__ == "__main__":
    # Runs the server using stdio transport by default.
    # When the client spawns this script as a subprocess, they communicate via stdin/stdout.
    mcp.run()
