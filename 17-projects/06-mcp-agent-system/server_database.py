"""
MCP Server: Database Access
Exposes local database access to the LLM agent via MCP.

Demonstrates establishing security boundaries — the server only permits SELECT queries,
safeguarding against malicious agent actions (e.g., DROP TABLE).
"""
from mcp.server.fastmcp import FastMCP
import sqlite3
import json
import os

mcp = FastMCP("Database Server")
DB_PATH = "mock_database.db"

def init_db():
    """Initialize a mock SQLite database for the agent to query."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT, role TEXT, active BOOLEAN, department TEXT)")
    
    # Insert some dummy data
    c.execute("INSERT OR IGNORE INTO users (id, name, role, active, department) VALUES (1, 'Alice Smith', 'Engineer', 1, 'Engineering')")
    c.execute("INSERT OR IGNORE INTO users (id, name, role, active, department) VALUES (2, 'Bob Jones', 'Manager', 1, 'Product')")
    c.execute("INSERT OR IGNORE INTO users (id, name, role, active, department) VALUES (3, 'Charlie Brown', 'Analyst', 0, 'Data')")
    c.execute("INSERT OR IGNORE INTO users (id, name, role, active, department) VALUES (4, 'Diana Prince', 'Data Scientist', 1, 'Data')")
    
    conn.commit()
    conn.close()

# Run initialization when the module loads
init_db()

@mcp.tool()
def execute_sql(query: str) -> str:
    """
    Execute a read-only SQL query against the local SQLite database.
    The database contains a 'users' table with columns: id, name, role, active, department.
    """
    # Security Boundary: Restrict to SELECT queries
    query_upper = query.strip().upper()
    if not query_upper.startswith("SELECT"):
        return json.dumps({"error": "Security policy violation: Only SELECT queries are permitted."})
    if ";" in query and not query.endswith(";"):
         return json.dumps({"error": "Security policy violation: Multiple statements detected."})
    
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute(query)
        rows = [dict(row) for row in c.fetchall()]
        conn.close()
        
        return json.dumps({"status": "success", "results": rows, "count": len(rows)})
    except Exception as e:
        return json.dumps({"error": f"Database error: {str(e)}"})

@mcp.tool()
def list_tables() -> str:
    """List all available tables in the database."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in c.fetchall()]
    conn.close()
    return json.dumps({"tables": tables})

if __name__ == "__main__":
    mcp.run()
