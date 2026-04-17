# MCP Notes

This file is the practical bridge between model-powered apps and structured tool use.

If Azure DevOps gives you a clean way to connect services through pipelines and contracts, MCP is that same spirit for LLM applications talking to tools, resources, and prompts.

Think:

- protocol
- capability discovery
- tool execution
- structured context flow

not random copy-paste glue.

---

# 1. The Core MCP Mental Model

There are three main players:

- **Host**
  The main application where the user interacts

- **Client**
  The part inside the host that connects to MCP servers

- **Server**
  The program that exposes tools, resources, and prompt templates

That separation matters because it makes capabilities discoverable and reusable instead of hard-wired into one app.

---

# 2. The Three Main Primitives

## Tools

Actions the model can invoke.

Think:

- search
- update
- call external service

## Resources

Read-only contextual data.

Think:

- files
- docs
- records
- dynamic state snapshots

## Prompt Templates

Reusable prompt structures exposed by the server.

These help move prompt logic out of the user's head and into something cleaner and repeatable.

---

# 3. Why MCP Feels Familiar If You Know DevOps

MCP is basically contract-driven integration for model-powered systems.

That is why it should feel familiar if you already think in:

- interfaces
- clients and servers
- discovery
- transport
- reusable workflows

Instead of every AI app wiring tools in a one-off way, MCP gives you a cleaner protocol layer.

That is the real value.

---

# 4. Common Transport Patterns

Two practical patterns matter most:

- local stdio-based server processes
- remote HTTP-based servers

The transport choice changes:

- deployment style
- state handling
- operational complexity

but not the core MCP idea.

---

# 5. Building an MCP Server

At a high level:

1. define the capability
2. expose it as a tool/resource/prompt
3. start the server
4. let a client discover and use it

That means the same Python function can stop being "just local helper code" and become a reusable capability in a broader LLM system.

That is a big shift.

---

# 6. Why MCP Matters for LLM Systems

Because real model-powered apps usually need more than raw generation.

They need:

- retrieval
- structured tools
- governed context
- reusable prompts
- safer orchestration

MCP helps formalize that layer.

So instead of:

> prompt -> hope -> output

you get something much more like:

> client -> discover capability -> invoke tool -> return structured result -> continue reasoning

Much better engineering.

---

# 7. Practical Use Cases

MCP works nicely for:

- research assistants
- database tools
- document retrieval
- workflow automation
- multi-tool agent systems

Any time you want the model to work with explicit capabilities rather than improvising everything from memory, MCP becomes relevant.

---

# Quick Thought Experiment

If you had to choose between:

- an LLM app with hand-wired local helper functions
- an LLM app with a protocol-based capability layer

which one scales better across teams and tools?

Exactly.

That is the MCP pitch in one question.
