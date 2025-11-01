# MCP

## MCP artitecture&#x20;

#### 💡 Core Architecture (0:43 - 1:27)

* Host: The main LLM application or environment (e.g., Claude desktop, Cursor, Windsurf) where the user interacts. It's responsible for managing clients and connections to servers.
* Client: Lives inside the Host. It manages the connection to an MCP server and is responsible for finding and using the tools, resources, and prompts the server offers.
* Server: A lightweight program that exposes specific capabilities (tools, resources, prompts) through the protocol. They can be local (like for SQLite) or remote.
* Goal: To help you understand what's happening "under the hood" when using powerful LLM applications.

***

#### 🛠️ Key Primitives (Fundamental Pieces of the Protocol) (1:36 - 3:10)

| **Primitive**    | **Description**                                                                                                                               | **Analogy/Use Case**                                                                                              |
| ---------------- | --------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| Tools            | Functions invoked by the client to retrieve, search, send messages, or update database records. Allow for modification (like a POST request). | Used to perform actions, e.g., `List Tables`, `Analyze data`.                                                     |
| Resources        | Read-only data or context exposed by the server. Clients can consume this data.                                                               | Similar to a GET request. Examples: database records, API responses, files, PDFs, or a dynamic memo that updates. |
| Prompt Templates | Predefined, thoroughly evaluated templates that live on the server. They remove the burden of detailed prompt engineering from the user.      | The client accesses and feeds this template to the user, who just provides dynamic data.                          |

* Client's Job: Find resources and find tools.
* Server's Job: Expose that information to the client.

***

#### 💻 SDK and Declaration (7:58 - 10:05)

* MCP provides Software Development Kits (SDKs) for building clients and servers (Python SDK is used in the course).
* Tools are declared by decorating a function, passing in arguments, and defining a return value to generate the tool schema.
* Resources are declared by specifying a URI (location) where the client finds the data. You decorate a function that returns the data (can be direct or templated).
* Prompt Templates are declared by decorating a function and defining a set of messages/text (e.g., user/assistant roles). They are designed to be user-controlled.

***

#### 🌐 Communication and Transport (10:08 - 14:50)

* Communication Lifecycle:
  1. Initialization: Client connects, requests and capabilities are exchanged.
  2. Message Exchange: Clients/servers send Requests (expect a response), Responses, and Notifications (one-way).
  3. Termination: Connection is closed.
* Transport: Handles the mechanics of how messages (JSON-RPC 2.0 format) are sent between client and server.

| **Transport**                   | **Use Case**                  | **Mechanism**                                                                                | **Key Notes**                                               |
| ------------------------------- | ----------------------------- | -------------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| Standard IO (stdio)             | Local servers, CLIs.          | Client launches server as a subprocess; communication over standard in/out.                  | Simplest, most common for local use.                        |
| HTTP + Server-Sent Events (SSE) | Remote servers.               | Opens a stateful connection (messages are remembered).                                       | Server can send events back to the client. Older transport. |
| Streamable HTTP                 | Remote servers (recommended). | Supports both stateful (using SSE) and stateless (using standard HTTP GET/POST) connections. | Newer, more flexible, and allows for stateless deployments. |



## Chatbot example

#### 1. Introduction & Goal

* Goal: Build a simple chatbot and code its tools to gain a foundation in tool use and prompting Large Language Models (LLMs) before moving to Model Context Protocol (MCP) servers.
* The application will use a chatbot to search for papers on arXiv (an open-source repository for published papers across domains like science and math).
* _Note:_ The instructor mentions if you are familiar with this, you can skip to the next lesson on building your first MCP server.

#### 2. Required Libraries and Setup

* Libraries Imported:
  * `arxiv` SDK: For searching for papers.
  * `json` module: For formatting.
  * `os` module: For environment variables (API keys).
  * `typing`: For code type-hinting.
  * `Anthropic` SDK: To interact with the Claude model.
* Constant Defined:
  * `paper_directory`: Set to the string `"papers"`. This is used for saving information to the local file system.

#### 3. Tool Functions

The lesson defines two key Python functions that will be exposed as tools to the LLM:

**A. `search_papers(topic, number_of_results=5)`**

* Purpose: Searches arXiv for papers related to a given `topic`.
* Process:
  1. Initializes the `arxiv` client.
  2. Searches for relevant articles.
  3. Creates the `paper_directory` (e.g., `"papers"`) if it doesn't exist.
  4. Processes the results into a dictionary with paper information (title, URL, summary).
  5. Saves this information locally to a file: `papers_info.json`.
  6. Returns: A list of paper IDs.
* Example Use: Searching for papers on "computers" and receiving a list of IDs.

**B. `extract_info(paper_id)`**

* Purpose: Retrieves detailed information about a paper using its ID.
* Process:
  1. Looks up the paper's ID in the locally saved `papers_info.json` file.
  2. If found, returns the detailed data (title, PDF URL, summary).
  3. If not found, returns a string indicating no saved information for that ID.
* Example Use: Calling the function with a specific paper ID to get its title, PDF URL, and summary.

#### 4. Tool Definition for the LLM

* A `tools` list is defined for Anthropic's Claude model.
* Each tool definition includes:
  * `name`: (e.g., `search_papers`, `extract_info`)
  * `description`: (Tells the model what the tool does and when to use it.)
  * `schema`: (Defines the required and optional arguments, like `topic` or `paper_id`.)
* Crucial Point: The LLM does _not_ call the functions itself; the developer must write the execution code to call the actual Python functions and pass the results back to the model.

#### 5. Chatbot Logic & Tool Execution

* Tool Mapping: A dictionary is set up to map the tool names (e.g., `"search_papers"`) to their corresponding Python functions. This is a helper for executing the correct function when the LLM requests a tool.
* Client Setup: Environment variables (API keys) are loaded, and an instance of the Anthropic client is created.
* `chat_loop` Function (Boilerplate): This is the main function for interacting with the model (Claude 3.7 Sonnet is used).
  * It maintains a list of messages.
  * The user's query is passed in.
  * It runs a loop that:
    1. Calls the model.
    2. Checks the response:
       * If it's text data, it's displayed to the user.
       * If it's a tool use request, the helper function executes the tool (calling `search_papers` or `extract_info`).
       * The tool result is then appended back to the message list, allowing the model to generate a final, informed response.
* Demonstration:
  1. A simple "hi" query works.
  2. A query like "search for papers on algebra" triggers the `search_papers` tool. The results are saved locally and the model returns the list of IDs and a follow-up question.
  3. The user asks to "extract information on the first two" IDs. The model calls `extract_info` twice, gets the data, and then uses that data to generate a summary for the user.
* Important Note: The current implementation does not have persistent memory. Each conversation must be thought of as a new one, so IDs must be passed back to the model when needed. The loop ends when the user types `"quit"`.

#### 6. Next Steps

* The next lesson will focus on refactoring this existing code to convert the functions into MCP tools (Model Context Protocol).
* This refactoring will allow the tools to be exposed via a server, which will then be tested.



## Creating a MCP server&#x20;



💡 Refactoring Chatbot Tools into an MCP Server with FastMCP

This lesson details the process of migrating existing chatbot tools (`search_papers` and `extract_info`) into a dedicated Model Context Protocol (MCP) server using the FastMCP library and the Standard IO (stdio) transport.

***

#### 🛠️ Key Steps & Implementation

1. Refactoring Goal: Abstract the definition and schema of the existing chatbot functions (tools) and wrap them in an MCP server for standardized access by LLMs/clients.
2. FastMCP Initialization:
   * Import the `FastMCP` class: `from mcp.server.fast_mcp import FastMCP` (The transcript shows `from MCP dot server dot fast MCP`).
   * Initialize the server instance, providing a name: `mcp = FastMCP("research")`
3. Tool Definition:
   * Define the existing functions (e.g., `search_papers`, `extract_info`) as MCP tools.
   * This is done simply by decorating each function with `@mcp.tool`. This decorator automatically infers the tool's description and parameter schema from the function's docstring and type hints.
4. Server Startup:
   * The server's execution is wrapped in the standard Python entry point: `if __name__ == "__main__":`
   * The server is started by calling the run method: `mcp.run(transport="standard_io")`
   * The Standard IO (stdio) transport is used, which is typical for local server environments.

***

#### 🧪 Environment Setup & Testing

1. File Creation: The code is executed to write a file named `ResearchServer.py`.
2. Terminal Setup: A new terminal is opened, and the environment is set up.
3. Dependency Management: The instructor uses the UV package manager (a faster alternative to `pip`) for dependency management:
   * `uv init` (Initializes the project/virtual environment).
   * `uv venv` (Creates the virtual environment, visible as a `.venv` folder).
   * `source .venv/bin/activate` (Activates the virtual environment).
   * `uv add mcp arxiv` (Installs the necessary `mcp` and `arxiv` dependencies).
4. Server Testing with MCP Inspector:
   * The server is tested using the MCP Inspector, a browser-based developer tool for exploring, testing, and debugging MCP primitives (Tools, Resources, Prompts).
   * The Inspector is launched using an `npx` command: `npx @ModelContextProtocol/inspector uv run research_server.py`.
   * The Inspector connects to the server running on the `standard_io` transport.
5. Tool Verification:
   * In the Inspector, the `tools/list` command is used to find available tools.
   * The tools can be directly executed/run within the Inspector's sandbox, testing the functionality (e.g., searching for "chemistry" papers) and verifying the returned data.
   * The Inspector confirms the initial handshake/initialization process between the client and server.

This entire process demonstrates how to cleanly separate tool logic into a compliant MCP server for robust integration with an LLM-based application, with the Inspector providing a vital testing interface.

***

For a visual guide on this entire process, you can watch Turn ANY Python Function into an MCP Tool Instantly (FastMCP Demo). This video demonstrates how to use FastMCP to instantly turn Python functions into MCP tools.



## Creating a MCP client&#x20;



👋 Time to get that MCP client wired up! This lesson focuses on the "under the hood" mechanics of creating an MCP client within your chatbot to talk to the server you built previously. It's the essential bridge for tool use.

Here are the detailed notes for your lesson:

***

## Creating an MCP Client in Your Chatbot Host

#### Goal

To build a host (your chatbot) that contains an MCP client. This client connects to the running MCP server, gets access to the server's tool definitions, and sends tool execution requests back to the server.

#### 1. Revisiting the Chatbot Structure

* The chatbot (e.g., in `mcp_chatbot.py`) initially uses the Anthropic SDK and Claude 3.7 Sonnet for conversation and tool-use logic.
* Key Difference: Unlike a standalone chatbot where tools are defined locally, this chatbot will not define any tools. Those are all managed by the MCP server.

#### 2. The MCP Client Essentials

The client's main jobs are:

1. Establish a connection to the server.
2. Initialize the session (the "handshake").
3. Query for available tools from the server.
4. Execute tools on the server's behalf when the LLM requests it.

* Lower-Level Focus: The code is more focused on the core library imports (`ClientSession`, `StdioServerParameters`, `stdio_client`, etc.) to show how the connection is fundamentally established. This is important for understanding what clients like Claude Desktop or Cursor are doing behind the scenes.
* Async Nature: Since the client might not want to block while waiting for the server, Python's `async` and `await` are heavily used.

#### 3. Key Code Components

**A. Server Parameters**

* You must specify how the client should start the server as a subprocess.
  * This includes the command (e.g., `uv run research_server.py`) and any necessary environment variables.
  * Concept: The client is responsible for _launching_ and _communicating_ with the server process, typically over Standard I/O (stdio).

**B. Connection and Session Management**

* A context manager (often within an `async` function named `run` or `connect_to_server_and_run`) is used to manage the connection lifecycle.
* The `stdio_client` function starts the server as a subprocess and provides a read and write stream for communication.
* The `ClientSession` class takes these read/write streams to establish the higher-level MCP session.
* Handshake: Call `session.initialize()` to perform the initial connection handshake.

**C. Tool Discovery and Use**

* List Tools: Call `session.list_tools()` to get the available tool definitions from the server.
  * The client then passes these definitions to the LLM (Claude) so it knows what it can use.
* Tool Invocation: The `process_query` function is modified:
  * If the LLM decides to use a tool, the client uses the established session (`session.execute_tool(...)`) to send a request back to the MCP server.
  * The server executes the tool's logic (defined in the previous lesson) and returns the result to the client.
  * The client then passes the result back to the LLM for a final, contextual response.

**D. Execution Environment**

* Since the entire client-server process is now `async`, you move from `mcp.run()` to `asyncio.run(main_function)` to start the main application loop.
* Dependency: The library `nest_asyncio` is often required to ensure compatibility with Python's event loop across different operating systems.

***

## Running the Chatbot (`mcp_chatbot.py`)

#### Setup Steps

1. Navigate to your project folder (e.g., `L5/mcp_project`).
2. Activate the virtual environment: `source .venv/bin/activate`.
3. Install Dependencies: You'll need the Anthropic SDK, `python-dotenv`, and `nest_asyncio` to make this work smoothly.
   * _Slightly cheeky tip: Typing `pip install anthropic python-dotenv nest_asyncio` is faster than waiting for a single dependency to install._

#### Execution and Interaction

1. Run the Client: Use `uv run mcp_chatbot.py`.
2. Connection: The client automatically:
   * Launches the MCP server as a subprocess.
   * Sends a `list_tools_request` to the server.
   * Once connected, it prints a message showing which tools it received.
3. Chat Loop: The interface starts, allowing you to converse with the Claude model.
4. Tool-Use Example:
   * User Query: "Can you search for papers around physics and find just two of them for me?"
   * Client Action: The client sends the tools list to Claude. Claude decides to call a tool (e.g., `search_papers`).
   * Communication: You see a `call_tool_request` being sent from the client to the server.
   * Result: The server executes the search, returns the results, and Claude uses that context to generate a summarized response.

#### The Bigger Picture

This process is the foundation for powerful agents:

* You can establish multiple client sessions to connect to different MCP servers, allowing for a diverse suite of tools.
* Future lessons will layer on more protocol primitives like Resources (read-only data) and Prompts (reusable templates) to scale this architecture.

Exit: You can type `quit` to exit the chatbot and gracefully close the connection.

***

## Connecting to Multiple Reference Servers

This lesson focuses on updating the MCP Chatbot client to dynamically connect to multiple MCP servers, including the custom server you built previously and Anthropic's official Reference Servers. This mirrors how professional AI tools like Claude Desktop or Cursor operate.

#### I. The Ecosystem of MCP Servers

* Goal: Move beyond a 1-to-1 client-server connection to an entire ecosystem where one client can talk to multiple servers simultaneously.
* Anthropic Reference Servers: These are official, open-source servers provided by the Anthropic team on GitHub to showcase the Model Context Protocol (MCP) features.
  * Any data source imaginable likely has an MCP server implementation (third-party or official).

***

#### II. Key Reference Servers

The lesson introduces two crucial reference servers:

**1. The Fetch Server (Python-based)**

* Purpose: Allows the LLM to retrieve content from web pages and convert HTML to Markdown for better consumption by the model.
* Execution Command: Since it's written in Python, the recommended way to run it is:
  * `uvx mcp-server-fetch`
  * `uvx` is used instead of `uv run`. It's a Python command runner that downloads the necessary package from PyPI and executes it in an isolated environment without needing to install it locally first.

**2. The File System Server (TypeScript-based)**

* Purpose: Provides tools for accessing the local file system (reading, writing, searching for files, getting metadata).
* Execution Command: Since it's written in TypeScript, the command is different:
  * `npx -y @modelcontextprotocol/server-filesystem --allowed-paths .`
  * `npx` is the Node Package Execute tool; it downloads and runs a Node.js package without a global install.
  * The `-y` flag automatically confirms installation prompts.
  * The `--allowed-paths .` argument is a critical security configuration, restricting the server's file access to only the current directory (`.`).

***

#### III. Server Configuration using JSON

To avoid hardcoding server parameters (name, command, arguments), you will configure all servers using a JSON file (e.g., `server_config.json`).

* Structure: The JSON file contains the necessary `command` and `args` (arguments) for each server, including your custom research server and the reference servers.
* Example Configuration Points:
  * File System Server: Requires the `allowed-paths` argument (`.`) for security.
  * Fetch Server: Uses the `uvx mcp-server-fetch` command.
  * Research Server: Uses the local `uv run` command for your custom-built server.

***

#### IV. Updating the MCP Chatbot Code

The MCP Chatbot needs significant updates to handle reading the JSON configuration and managing multiple concurrent connections.

| **Component**        | **Description**                                                                                                                                                                                                                                                   |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `connect_to_servers` | Function to read the `server_config.json`, parse the JSON into a dictionary, and iterate over each server, calling `connect_to_server` for each.                                                                                                                  |
| `AsyncExitStack`     | A powerful Python utility from `contextlib` used to manage multiple asynchronous context managers (`async with...`). It ensures all server connections are gracefully and reliably closed when the main application exits, regardless of errors.                  |
| Tool Mapping         | The chatbot must maintain a list of all discovered tools and, crucially, a map to link each tool back to the specific `ClientSession` (i.e., the server) that provides it. This ensures the correct server is called when the LLM decides to use a specific tool. |
| Connection Logic     | The connection logic is updated to use the `AsyncExitStack`'s `enter_async_context()` method to keep the `stdio_transport` (communication channel) and the `ClientSession` alive for the entire duration of the chat.                                             |
| Tool Execution       | The chat loop's logic is updated to first look up the tool in the map to find the correct `session` before executing the tool call on the appropriate server.                                                                                                     |

***

#### V. Demonstration

1. Setup: The chatbot is run using `uv run mcp_chatbot.py` after activating the virtual environment.
2. Connection: The output shows successful connection to all three servers:
   * `filesystem` (with allowed directory `.`)
   * `research-server`
   * `fetch`
3. Powerful Multi-Tool Prompt: A single query uses all three servers:
   * Prompt: _“Fetch the content of the Model Context Protocol, save the content to a file called MCP summary, and then create a visual diagram that summarizes the content.”_
   * Tool Usage:
     * Fetch Server: Retrieves the web content.
     * LLM (Claude): Summarizes the content and creates a visualization (using its internal knowledge/capabilities after getting the context).
     * File System Server: Writes the resulting summary and diagram to a file (`MCP_summary.md`).

This update allows the chatbot to harness the diverse capabilities of multiple, specialized servers, making it significantly more useful.

***

## Adding Resources and Prompt Templates to Your Chatbot

The goal of this lesson is to upgrade the server to provide read-only data (Resources) and pre-engineered prompts (Prompt Templates), and then update the client (chatbot) to discover and expose these new features to the user.

#### I. Server-Side Implementation (The Research Server)

The server is updated to offer two new primitives in addition to its existing tools.

**1. Resources (`@mcp.resource`)**

* Definition: Resources are read-only data that the application (client) can choose to use or pass directly to the LLM. They are the MCP equivalent of an HTTP `GET` request.
* Implementation: Resources are defined by decorating a function with `@mcp.resource` and assigning it a URI (Uniform Resource Identifier).
  * Example 1: Listing Folders
    * URI: `papers://folders`
    * Purpose: To list available folders in the server's papers directory (e.g., "computers," "math").
  * Example 2: Fetching Topic Info
    * URI: `papers://<topic>`
    * Purpose: To fetch specific information or content about a particular topic (e.g., `papers://math`).
  * Implementation Details: The decorated functions handle string manipulation, reading data from a JSON file (e.g., `papers_info.json`), and returning the content as text, including necessary error handling.

**2. Prompt Templates (`@mcp.prompt`)**

* Definition: Prompt templates are battle-tested, pre-written prompts created on the server and sent to the client. Their purpose is to help the user avoid complex prompt engineering by providing dynamic, high-quality instructions.
* Implementation: Prompts are defined by decorating a function with `@mcp.prompt`. The function returns the prompt template itself.
  * Example: `Generate Search Prompt`
  * Variables: The prompt defines dynamic fields the user must fill, such as `topic` (required) and `num_papers` (optional).
  * Server Benefit: The server developer can ensure the LLM receives an expertly crafted, complex prompt without the user needing to write it.

***

#### II. Client-Side Implementation (The MCP Chatbot)

The chatbot client must be updated to discover, manage, and present these new primitives to the user.

**1. Server Connection and Discovery**

* The `connect_to_server` function is updated to use the established `ClientSession` to list all available primitives:
  * `session.list_tools()`
  * `session.list_prompts()`
  * `session.list_resources()`
* Data Storage: The client now maintains separate lists to store:
  * Available tools (mapped to sessions).
  * Available prompts (mapped to sessions).
  * Available Resource URIs.
* Robustness: Error handling is included to manage servers that do not provide these new primitives.

**2. User Interface and Presentation (Client Logic)**

The lecture emphasizes that the presentation is entirely up to the developer of the host and client. MCP only dictates the _data_ format, not the UI . The chatbot uses simple string conventions for the command-line interface (CLI):

| **Command**                | **Purpose**       | **Client Action**                                                                                                        |
| -------------------------- | ----------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `@<URI>`                   | Access a Resource | The client parses the URI, calls `session.read_resource(uri)`, and prints the raw content to the user.                   |
| `/prompts`                 | List Prompts      | Lists all available prompt templates and their required arguments (from all connected servers).                          |
| `/prompt <name> <arg=val>` | Execute a Prompt  | The client parses the command to extract the prompt name and key-value arguments, then calls the prompt execution logic. |

**3. Prompt Execution Flow**

1. User enters `/prompt Generate Search Prompt topic=math num_papers=10`.
2. The client's `execute_prompt` function finds the correct prompt template and session.
3. The client substitutes the user-provided arguments (`topic=math`, `num_papers=10`) into the template.
4. The final, fully-formed prompt text is generated.
5. This prompt text is sent to the LLM (e.g., Claude) as the content of the message for processing.

#### III. Demonstration Summary (Bringing it Together)

The demo showcases the integrated system:

1. Checking Resources:
   * Typing `@papers://folders` lists the existing folders (`computers`).
   * Typing `@papers://computers` fetches the raw papers data for that topic.
2. Checking Prompts:
   * Typing `/prompts` lists the new `Generate Search Prompt` (from the Research Server) and a `fetch_url` prompt (from the Fetch Server).
3. Executing a Prompt:
   * The user executes a high-level command like `/prompt Generate Search Prompt topic=math`.
   * The client creates the full search query and sends it to the LLM.
   * The LLM executes the required `research_papers` tool.
   * The resulting papers are saved, dynamically updating the resources.
4. Resource Update:
   * Typing `@papers://folders` now shows both `computers` and `math`, confirming the resource data is dynamic and reflects application state changes.

This setup completes the core primitives of the MCP, demonstrating a powerful, context-rich application architecture.

***

## Integrating Your MCP Server with Claude Desktop

This lesson focuses on how developers can abstract away the low-level client code (the networking, session management, and UI building you did previously) by connecting your custom MCP servers to an existing, compliant AI application like Claude Desktop.

#### I. Preparing the Local MCP Server

1. Project Setup: The developer navigates to the folder containing the custom server (e.g., `research_server.py`).
2. Environment Setup: The standard virtual environment steps are followed:
   * Initialize the environment (`uv init`).
   * Create and activate the virtual environment (`uv venv` and `source venv/bin/activate`).
   * Install necessary dependencies (e.g., `pip install arxiv mcp`).
3. Server Execution: Crucially, the server is not run manually in the terminal. The client application (Claude Desktop) will manage the server's lifecycle via a subprocess.

#### II. Configuring Claude Desktop as an MCP Client

Claude Desktop (and many other compliant clients) uses a configuration file to discover and launch local MCP servers using the `stdio` (standard input/output) transport.

1. Accessing the Config File:
   * Navigate to Settings $$ $\to$ $$ Developer $$ $\to$ $$ Edit Config in Claude Desktop.
   * This opens a JSON configuration file (e.g., `claude_desktop_config.json`).
2. Adding the Server Configuration:
   * The developer pastes the server configuration into the JSON file.
   * The configuration specifies:
     * Name: A friendly name for the server (e.g., `"research_server"`).
     * Command: The executable used to run the server (e.g., `"python"` or the full path to the Python environment).
     * Arguments (`args`): The path to the server file and any required startup arguments.
       * Key Detail: For local servers, the exact file path (absolute path) to the server file (`research_server.py`) is specified so the client knows exactly what to launch.
3. Connection and Discovery:
   * The developer must Close and Reopen Claude Desktop to force the application to read the new configuration and establish the connections.
   * On restart, Claude Desktop launches the servers as subprocesses and lists all discovered Tools, Resources, and Prompts in its own user interface.

#### III. Using MCP Primitives in the Client UI

Once connected, Claude Desktop abstracts away the low-level code, providing a ready-made interface for the primitives you created:

* Tools: Available for Claude to use in its planning/reasoning.
* Resources: Available for Claude to fetch read-only data (like the papers directory list).
* Prompts: Available to the user/model to leverage pre-engineered instructions (like the `Generate Search Prompt`).

#### IV. The Power of MCP: Interoperability and Agentic Frameworks

The core benefit is demonstrated by combining tools from multiple sources to achieve a complex, agentic task.

1. Multi-Server Coordination: A single prompt triggers a chain reaction:
   * Prompt: Ask the LLM to research a topic, summarize papers, and generate a quiz.
   * Tool 1 (Fetch Server): Use the `fetch` tool to visit a website (e.g., DeepLearning.AI) to get a current topic (e.g., multi-modal LLMs).
   * Tool 2 (Local Research Server): Use the `search_papers` tool to find and store relevant papers based on the discovered topic.
2. Artifact Generation: Claude Desktop's built-in Artifacts feature is used to visualize the final output.
   * The LLM uses the summarized information to generate a web-based quiz (HTML/JS/React) and displays it in a separate, interactive window.
   * This shows the seamless integration of custom-built tools (MCP servers) with native client features (Artifacts) for a powerful, user-facing result.

#### V. Ecosystem Overview

The Model Context Protocol is designed for broad adoption, supported by a wide range of applications:

* Web Applications: Applications accessible via a browser.
* Agentic Frameworks: Tools designed for complex, multi-step tasks.
* Command Line Interfaces (CLIs): For text-based interactions.
* Integrated Development Environments (IDEs): Such as VS Code and JetBrains AI Assistant, allowing LLMs to interact with code, files, and debugging tools.

The lecture concludes by emphasizing that by building your own servers, you understand the fundamental mechanism that powers all these different, compatible client applications.

***

That lecture was all about taking your local Model Context Protocol (MCP) server and pushing it out into the wild as a remote server using Server-Sent Events (SSE) transport, followed by deployment on Render. It's a surprisingly small change to go from local to remote!

Here are the detailed notes:

#### 1. ⚙️ Modifying the MCP Server for Remote Access (0:25)

* Goal: To allow the server to be accessed remotely, which requires a change in the communication transport.
* Configuration: The core server logic (tools, resources, prompts) remains the same. The primary change is in specifying the transport.
* Transport: Since the Python SDK at the time didn't fully support HTTP streamable, the server was configured to use SSE (Server-Sent Events).
  * _Note:_ The speaker mentioned that switching to the newer HTTP streamable transport should be a quick change once it's fully supported in the SDKs.

#### 2. 🔎 Testing the Remote Server with the Inspector (1:05)

* The server is assumed to be running at a specific URL.
* Tool Used: The MCP Inspector tool is used to connect to and test the remote server.
* Connection Steps:
  1. Run the Inspector using `npx @ModelContextProtocol/inspector`.
  2. Visit the inspector's URL in a browser.
  3. Configure the connection in the Inspector:
     * Confirm the proxy address (if used).
     * Set the transport type to SSE.
     * Input the server's SSE URL.
  4. Verification (2:01): Once connected, you can list the server's primitives (resources, prompts, tools) to confirm the connection is initialized and functional.

#### 3. 🚀 Deploying the Server to Render (2:29)

The server is deployed to Render to make it publicly accessible. This requires using Git/GitHub for deployment.

**A. Prepare Code for Deployment (2:39)**

1. Initialize Git:
   * `git init` to start a new repository.
2. Create `.gitignore`:
   * Add a `.gitignore` file to exclude necessary files like the `.env` folder from the repository.
3. Generate `requirements.txt`:
   * Since Render doesn't support `uv` (the current dependency manager) for deployment, dependencies must be compiled into a standard `requirements.txt` file for Pip.
   * Command: `uv pip compile pyproject.toml > requirements.txt` (This takes the dependencies from your `pyproject.toml` and formats them for Pip.)
4. Specify Python Version:
   * Create a `runtime.txt` file to explicitly tell Render which Python version to use (e.g., `Python 3.11.11`).
5. Commit Changes:
   * Use `git status` to check the new files (`requirements.txt` and `runtime.txt`).
   * `git add .`
   * `git commit -m "ready for deployment"`

**B. Push to GitHub (4:40)**

1. Create a new remote repository on GitHub (e.g., _remote-research_).
2. Add the remote to the local Git configuration:
   * Copy and run the GitHub command to add the remote origin.
3. Push the code:
   * `git push origin main`

**C. Configure Render Web Service (5:50)**

1. Sign up/Log in to Render and select Deploy Web Service.
2. Connect to GitHub and select the new repository (remote-research).
3. Configuration Change: The only essential setting change is the Start Command.
   * Set the command to: `python research_server.py`
4. Select a plan (e.g., Free Plan) and Deploy.

**D. Final Verification (6:43)**

* Deployment: Render automatically uses `runtime.txt` and `requirements.txt` to set up the environment and install dependencies.
* Testing the Public URL:
  * The base URL will likely return a 404 error (which is expected).
  * The correct endpoint to check is the SSE endpoint (e.g., `[Render_URL]/sse`).
  * Success: Visiting the SSE endpoint should show a response from the server, including a session ID, confirming the server is deployed and accessible.

This walkthrough is a great, step-by-step guide for taking a project from a local development environment to a publicly available service.

***

You can learn more about deploying MCP servers to remote platforms like Render by watching [Exposing Your MCP Tools Remotely Using Server-Sent Events (SSE)](https://www.google.com/search?q=https://www.youtube.com/watch%3Fv%3DkYJ5XyI_52g).

## MCP: Beyond the Basics - Your Detailed Lecture Notes

Congratulations on completing the core concepts of the Model Context Protocol (MCP)! This final lesson covered advanced features and the exciting roadmap for the protocol's future, focusing on security, new primitives, and agentic architectures.

***

#### 🛡️ Authentication with OAuth 2.1 (0:56)

* Core Method: The Model Context Protocol (MCP) adopted OAuth 2.1 in the March specification update as the primary means for authentication with remote servers.
* Purpose: It allows clients and servers to authenticate securely and send authorized requests to data sources.
* The Flow:
  1. The client makes a request to the server.
  2. The server requires the user to authenticate (via the client).
  3. Upon successful authentication, the client and server exchange an access token.
  4. The client uses the token to make authenticated requests to the server, which then accesses the secure data source.
* Recommendation: This feature is optional but highly recommended for all remote servers. Standard I/O connections typically rely on environment variables instead.

***

#### ⚙️ Client-Exposed Primitives (2:09)

While you've learned about server primitives (tools, resources, prompts), clients can also expose capabilities to servers.

**1. Roots (2:17)**

* Definition: A URI (Uniform Resource Identifier) that a client suggests the server should operate within.
* Function: It's a way for the client to declare specific, relevant file paths (e.g., local folders) or other valid URIs (e.g., HTTP URLs) where the server should look for files or data.
* Benefits:
  * Security Limitations: It helps set a clear scope for the server.
  * Focus: Keeps the server concentrated on a relevant file path or location, preventing it from searching the entire file system or network.

**2. Sampling (3:03)**

* Definition: This primitive reverses the typical flow, allowing a server to request inference from a Large Language Model (LLM) via the client.
* Scenario Example: A server collecting performance logs and metrics (server logs, error logs, etc.) can send this data to the LLM via the client, asking the model to diagnose performance issues.
* Benefits:
  * Security: Prevents putting all sensitive server data into the context window and avoids potential security or boundary breaches by sending it back to the client/user.
  * Efficiency: The server gets the LLM's analysis and proposed steps _directly_, reducing the data transfer overhead for the client.
  * Agentic Capabilities: It's a powerful mechanism for agentic workflows, switching the direction of communication and enabling the server to leverage the LLM for complex tasks.

***

#### 🗺️ The Future: Agentic Capabilities and Discovery

**Multi-Agent Architecture (4:16)**

* Composability: MCP's design is composable and recursive, meaning a single agent can act as both an MCP client and an MCP server.
* Flow: An application and LLM can communicate with one agent, which can, in turn, connect to other specialized agents (for analysis, coding, research) that also operate as MCP servers/clients.
* Goal: To create architectures where multiple specialized agents all communicate using the same standardized protocol (MCP).

**Unified Registry API (5:47)**

* The Problem: The open-source community will see dozens of MCP servers for popular tools (like Google Drive or GitHub), which creates risks of malicious code and makes discovery difficult.
* The Solution: The Unified Registry API aims to be the centralized and standardized way for discovering, centralizing, and verifying trusted MCP servers, similar to NPM or PyPI for packages.
* Key Features:
  * Server discovery and centralization.
  * Verification and trust (servers trusted by the community/companies).
  * Versioning to lock in dependencies.
  * Dynamic Discovery: Agents can search the registry for the official server needed for a task, install it dynamically, and query it without being connected from the start.
* MCP JSON (7:24): The discovery process will often involve a `MCP JSON` file in a well-known folder, specifying the server's endpoint, exposed capabilities (primitives), and required authentication.

***

#### 🚀 Roadmap Highlights (8:08)

* HTTP Streamable Support: Aiming for a smoother transition between stateful and stateless capabilities.
* Collision Prevention: Addressing naming conflicts when multiple MCP servers use generic tool names (e.g., `fetch_users`). This requires creating logical groups for tools or servers.
* Popularizing Sampling: Continued work to make the proactive context request primitive (sampling) more popular and robust.
* Advanced Auth: Further development in authentication and authorization at scale, building on the initial OAuth 2.1 implementation.

The lecture concludes by encouraging continued building and research into the evolving MCP specification.

***

This video provides a deep dive into the MCP client, which is where the new primitives like Roots and Sampling reside.

Model Context Protocol - Part 5 of 10 - Client Deep Dive | Client Primitives Explained

Would you like a summary of a specific feature, like Sampling or the Unified Registry API, or do you have any other questions about the Model Context Protocol?



