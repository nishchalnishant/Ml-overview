MCP is a protocol — a protocol is a set of rules for communication, it defines the format, order, and meaning of messages exchanged between two parties.

It is an open protocol, so it is publicly available for anyone to use or build on.

MCP -- open-source standard protocol for connecting AI applications to external tools and data. It acts as an abstraction over the implementation details of how data is fetched or logic is run, so the agent doesn't crash when a request fails — it just gets a tool/resource error back through the same standard interface.

## MCP participants

host -- the AI application (Claude Desktop, Cursor, etc.) that is interfacing with the AI model. Acts as a container/orchestrator whose aim is to get information to the LLM. A host can run many clients, and based on what it's doing it instantiates different clients (e.g. one client per server it needs to talk to).

client -- a running program within the host whose job is to get info from one server for the LLM. There's usually a 1:1 relationship between a client and a server it's connected to. A host can hold multiple clients, and one client can also be wired to multiple servers ("multi-server client").

server -- provides the context: pulling data from a DB, hitting an API, reading emails, reading files off disk, running some domain-specific computation, etc. A server doesn't know or care which host is calling it — it just exposes tools/resources/prompts over MCP.

## MCP primitives — smallest building blocks of an MCP application

tools
- a function on the server that performs an action. It enables the AI to ask the server to *do* things, which may or may not return data. Ex: querying a database, calling an API, computing a trend, writing a file.

resources
- contextual read-only info for the AI. It enables the AI to ask the server for information rather than to perform an action. Ex: files, database schemas, config values. Exposed via URI templates (e.g. `notes://all`).

prompts
- reusable prompt templates the server can offer, parameterized by arguments, so a client can pull a vetted prompt instead of the host having to hand-write it. Ex: a "summarize this ticket" template a support tool exposes.

(Rough mapping to REST verbs: tools ~ POST/action, resources ~ GET/data, prompts ~ canned templates.)

## Transport

Messages are exchanged in **JSON-RPC 2.0** format (`{"jsonrpc": "2.0", "method": ..., "params": ..., "id": ...}`), sent over a transport layer. Two common transports:

- **stdio** -- for local servers. The host/client spawns the server as a subprocess and talks to it over its stdin/stdout pipes. No network involved, no auth needed (it's local process-to-process). This is what Claude Desktop uses for local MCP servers, and what the notes/stats/filesystem servers in `12-projects/04-multi-agent-a2a-mcp/mcp_servers/` use.
- **remote HTTP (Streamable HTTP / SSE)** -- for servers running elsewhere (a separate machine, a hosted service). The client makes HTTP requests to the server's URL and reads responses (optionally as a stream via Server-Sent Events), so a server can be shared across many hosts/users at once. This is where auth (API keys, OAuth) actually matters, since it's now crossing a network/trust boundary.

## Advantages of MCP

- **integration problem (M×N -> M+N)** -- without a shared protocol, every AI app has to write a custom integration for every tool/data source it wants (M apps x N tools = M×N integrations), and that custom code often differs per model/vendor too. MCP standardizes the interface at the protocol level, so an app just needs to speak MCP once (M clients + N servers, not M×N glue code).
- **language-agnostic** -- without MCP you'd also end up rewriting integration code per language/framework. MCP is just JSON-RPC over a transport, so a server written in Python can be called from a TypeScript client with no shared code.

## Benefits of MCP [one protocol, any client, any server]

- we need one standard that connects any AI tool to any data source
- open protocol -- anyone can implement a client or a server against the same spec
- prebuilt intelligence -- servers can ship curated tools/prompts that encode domain expertise, not just raw data access
- standardized structure -- discovery (`list_tools`, `list_resources`) and invocation (`call_tool`) look the same regardless of what's on the other end
- client flexibility -- swap Claude Desktop for Cursor for a custom agent and the same MCP servers still work, unmodified
- reduces hallucination -- the model is grounded in real, live data/actions instead of guessing from training data alone

## MCP in action

- chatbot -- pull live data (docs, tickets, DB rows) into a conversation
- IDE assistants -- read/write files, run tests, query linters (Cursor, Claude Code)
- agents
  - access tools and data mid-task, not just once at the start
  - choose from available services (the model decides which tool fits the current step)
  - gather context as the task is carried out, instead of it all being front-loaded into the prompt

## Example hosts, clients, servers

host
- Claude Desktop
- Cursor
- Amazon Q
- Claude Code

client
- the built-in MCP client inside Claude Desktop/Cursor/Claude Code (one instantiated per configured server)
- a custom client you write yourself, e.g. `agents/mcp_client.py` in `12-projects/04-multi-agent-a2a-mcp/` — a minimal client that spawns a server subprocess over stdio, does the MCP handshake, calls `list_tools()`, then `call_tool()`
- a client can be scoped down on purpose: e.g. one agent's client only knows about a `notes-server`, another agent's client only knows about a `stats-server` — restricting blast radius rather than giving every client every tool

server
- Google Drive (read files)
- GitHub (issues, PRs)
- Slack (send/read messages)
- Postgres/SQLite (schema + query access)
- a custom server, e.g. `mcp_servers/notes_server.py` (`search_notes`, `read_all_notes`), `mcp_servers/stats_server.py` (`compute_trend`), `mcp_servers/filesystem_server.py` (`write_report`) from the same reference project

## MCP request/response flow

- host instantiates a client per server it needs

three phases:

1. **initialization** -- host + client establish the connection with the server: exchange protocol version and capabilities over JSON-RPC, server responds with what it supports. Ends with an `initialized` notification.
2. **operation** -- the actual back-and-forth while the agent runs, e.g.:
   - `list_tools` -> get issues
   - `call_tool("search_files", ...)`
   - `call_tool("send_message", ...)`
3. **shutdown** -- client closes the connection/subprocess cleanly when done.

## Lifecycle of context

LLM output is only as good as its context (garbage in, garbage out).

Sources of context:
- tools -- actions to perform
- resources -- data to read
- prompts -- templates to reuse

Flow: list of available tools -> MCP/model picks the relevant tool(s) for the current step -> server executes and sends back the context/result -> that result becomes part of the conversation context for the next step.

## Agent-to-agent protocol (A2A)

How agents communicate with *other agents* — each agent can work on its own, but a real system usually needs several agents (possibly built/owned by different teams or vendors) to hand work to one another. MCP doesn't cover this: MCP is about an agent calling a tool/data source, not one agent delegating to a peer agent.

A2A fills that gap and introduces the **Agent Card** -- a JSON document an agent publishes (conventionally at `/.well-known/agent-card.json`) describing:
- what the agent is called and what it does
- what skills/capabilities it exposes
- how to reach it and what it expects in a request

Other agents (or a human, via `curl`) can fetch this card to discover what an agent claims it can do *before* sending it work.

Core A2A concepts:
- **Agent Card** -- capability/discovery document (see above)
- **Task** -- the unit of work handed from one agent to another, with a lifecycle: `submitted` -> `working` -> `completed` / `failed`, plus a `goal`
- **Message / Parts** -- the actual payload inside a task, split into typed parts (`text` for prose, `data` for structured JSON), so one task can carry both a natural-language finding and machine-readable data together
- **send_task()** -- the client-side call: discover the peer's Agent Card, `POST` a `Task` to its `/tasks` endpoint, get back the completed `Task`

## MCP and A2A sit together

MCP connects agents to **tools and data**: the agent calls a database, reads a file, or hits an API through an MCP server. This is a **horizontal** connection (agent <-> tool), typically fine-grained and stateless — called many times as the agent reasons.

A2A connects agents to **each other**: one agent delegates a whole unit of work (with its own goal and lifecycle) to a peer agent and gets structured results back. This is a **vertical** connection (agent <-> agent), coarse-grained, and meant to cross a trust/ownership boundary — you'd A2A-call another team's or vendor's agent, but you wouldn't hand it raw MCP access to your own internal database.

Concrete worked example (`12-projects/04-multi-agent-a2a-mcp/`): a Research Agent, Analysis Agent, and Report Agent are three separate processes.
- Each owns its own MCP client(s), restricted to only the server(s) its job needs (Research -> notes-server only; Analysis -> stats-server only; Report -> filesystem-server + notes-server).
- They hand the task down the chain over real HTTP `POST /tasks` calls (A2A): driver -> Research -> Analysis -> Report, each agent publishing an Agent Card and exchanging `Task`/`Message`/`Part` objects.
- So MCP is the building block inside each agent (agent -> its own tools), and A2A is the connective tissue across agents (agent -> agent) — a real system commonly needs both layers at once.
