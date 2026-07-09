# Interview 24 — Knowledge Graph Construction for Lore & Story
**EA SDE-2 AI Engineer · Estimated Duration: 75 minutes**

---

## Part 1 — Problem Statement

You are an AI Engineer on the BioWare narrative team (Mass Effect / Dragon Age). The studio has 20 years of lore spread across wikis, game design documents (GDDs), dialogue scripts, and novels. Writers frequently create plot holes because it's impossible to remember every character's history.

Your task is to **build an automated pipeline that ingests raw text documents and constructs a structured Knowledge Graph** (Entities and Relationships) so writers can query: *"Who are all the enemies of Commander Shepard that were born on Earth?"*

---

## Part 2 — Intentionally Missing Information

The following critical details are **deliberately omitted**. A strong candidate will ask about all of them:

- Entity Extraction methodology (Regex? NLP Models? LLMs?)
- Ontology / Schema (Is the schema fixed, e.g., `Person, Planet, Weapon`, or open-ended?)
- Graph Database choice (Neo4j, Amazon Neptune?)
- Deduplication / Entity Resolution (How do we know "Shepard" and "Commander Shepard" are the same node?)
- Query Interface (Do writers need to write Cypher/SQL, or use natural language?)

---

## Part 3 — Ideal Clarifying Questions

> Interviewer will reveal answers only when directly asked.

1. **"Is there a fixed schema (Ontology) for the entities and relationships, or should the model discover them dynamically?"**
   → *Answer: We have a loose ontology (Character, Faction, Location, Item), but relationships (e.g., `BETRAYED`, `MARRIED_TO`) should be extracted dynamically.*

2. **"How do we handle entity resolution? 'The Illusive Man' and 'Jack Harper' are the same person in the lore."**
   → *Answer: Excellent question. You will need to design an Entity Resolution (Coreference/Deduplication) step.*

3. **"Who is the end user? Will they write graph queries?"**
   → *Answer: Narrative writers. They cannot write Cypher/SQL. They need a natural language interface.*

---

## Part 4 — Expected Assumptions

- **Architecture:** NLP Pipeline ➔ Graph DB (Neo4j) ➔ Text-to-Graph QA Agent.
- **Extraction:** Use an LLM (GPT-4o) with strict JSON schema prompts for Information Extraction (IE) instead of training custom Spacy NER models, as LLMs excel at zero-shot relationship extraction.
- **Deduplication:** Embed entity names/descriptions and cluster them to merge duplicates.

---

## Part 5 — High-Level Solution

```
  [Raw Lore Documents (PDF/Markdown)]
       │
       ▼
  [Information Extraction (LLM)]
  Prompt: "Extract Entities (Type) and Relationships (Subject, Predicate, Object)."
  Output: {"entities": [...], "relationships": [...]}
       │
       ▼
  [Entity Resolution Module]
  Embeds "Shepard" and "Cmdr. Shepard". Cosine Similarity > 0.95 ➔ Merge Node.
       │
       ▼
  [Graph Database (Neo4j)]
  (Nodes: Entities | Edges: Relationships)
  
       =========================================================

  [Writer UI]
  User: "Which factions operate on Omega?"
       │
       ▼
  [Graph RAG Agent]
  Converts Text ➔ Cypher Query ➔ Executes on Neo4j ➔ Returns formatted answer.
```

**Core ML Component:** The zero-shot Information Extraction pipeline using LLMs, and the Entity Resolution logic to prevent the graph from becoming a disconnected mess of duplicate nodes.

---

## Part 6 — Step-by-Step Implementation

### Step 1: Information Extraction (IE)
- Chunk the documents.
- Pass chunks to an LLM with a strict JSON schema via function calling.
- Enforce the allowed node types (Character, Location, Faction).
- Allow dynamic predicates for edges (`KILLED`, `ALLIED_WITH`, `LOCATED_IN`).

### Step 2: Entity Resolution (Deduplication)
- If the LLM extracts `Node(Name: "Earth")` and later extracts `Node(Name: "Planet Earth")`, we must merge them.
- **Heuristic:** Exact string match (lowercased).
- **ML:** Generate a vector embedding for the Node Name + Context. If `cosine_sim(A, B) > 0.90`, merge them.

### Step 3: Graph Construction
- Load nodes and edges into Neo4j using the Cypher query language (e.g., `MERGE (a:Character {name: "Shepard"})`).

### Step 4: Natural Language Querying (Text2Cypher)
- Writers ask questions in English.
- Use a LangChain `GraphCypherQAChain`. It passes the DB schema and the user's question to the LLM, which writes the Cypher query, executes it against Neo4j, and summarizes the results.

---

## Part 7 — Complete Python Code

```python
"""
lore_graph_builder.py - Extracts Entities/Relationships and loads into Neo4j
"""
import logging
import json
import openai
from neo4j import GraphDatabase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Setup clients
openai_client = openai.OpenAI(api_key="mock_key")
neo4j_driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))

# ---------------------------------------------------------------------------
# 1. Information Extraction
# ---------------------------------------------------------------------------
def extract_knowledge(text_chunk: str) -> dict:
    """Uses LLM function calling to force structured Graph output."""
    
    tools = [{
        "type": "function",
        "function": {
            "name": "build_knowledge_graph",
            "description": "Extract entities and relationships from the lore.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entities": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string", "description": "Unique name of entity"},
                                "type": {"type": "string", "enum": ["Character", "Location", "Faction", "Item"]}
                            },
                            "required": ["id", "type"]
                        }
                    },
                    "relationships": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "source": {"type": "string", "description": "Entity ID of source"},
                                "target": {"type": "string", "description": "Entity ID of target"},
                                "predicate": {"type": "string", "description": "Relationship type, UPPERCASE (e.g., ALLIED_WITH)"}
                            },
                            "required": ["source", "target", "predicate"]
                        }
                    }
                },
                "required": ["entities", "relationships"]
            }
        }
    }]
    
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": text_chunk}],
        tools=tools,
        tool_choice={"type": "function", "function": {"name": "build_knowledge_graph"}}
    )
    
    args = response.choices[0].message.tool_calls[0].function.arguments
    return json.loads(args)

# ---------------------------------------------------------------------------
# 2. Graph Ingestion
# ---------------------------------------------------------------------------
def ingest_to_neo4j(graph_data: dict):
    """Loads the extracted data into Neo4j."""
    
    def _create_tx(tx, data):
        # Create Nodes using MERGE (prevents exact duplicates)
        for entity in data.get("entities", []):
            label = entity["type"]
            name = entity["id"].upper() # Simple normalization
            # Cypher query
            tx.run(f"MERGE (n:{label} {{id: $name}})", name=name)
            
        # Create Edges
        for rel in data.get("relationships", []):
            source = rel["source"].upper()
            target = rel["target"].upper()
            pred = rel["predicate"].upper().replace(" ", "_")
            
            # Match nodes and create relationship
            query = f"""
            MATCH (s {{id: $source}})
            MATCH (t {{id: $target}})
            MERGE (s)-[:{pred}]->(t)
            """
            tx.run(query, source=source, target=target)

    with neo4j_driver.session() as session:
        session.execute_write(_create_tx, graph_data)
        logger.info(f"Ingested {len(graph_data.get('entities', []))} entities.")

# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    lore_text = "Commander Shepard was born on Earth. Shepard allied with the Turian Hierarchy to defeat the Reapers."
    
    # 1. Extract
    graph_dict = extract_knowledge(lore_text)
    print("Extracted:", json.dumps(graph_dict, indent=2))
    
    # 2. Ingest
    # ingest_to_neo4j(graph_dict)
```

---

## Part 8 — Deployment

### Processing Pipeline
- **Apache Airflow / Celery:** Reading 10,000 pages of wikis and passing them through GPT-4 takes hours and handles rate limits.
- **Neo4j Aura:** Managed graph database optimized for Cypher queries.

### Writer Interface
- A simple web dashboard (Streamlit / React).
- Writers can view a visual representation of the graph (D3.js / NeoVis.js) or type natural language questions into a chatbox.

---

## Part 9 — Unit Testing

```python
import json
from lore_graph_builder import extract_knowledge
from unittest.mock import patch

# Mock OpenAI response
class MockOpenAI:
    class choices:
        class message:
            class tool_calls:
                class function:
                    arguments = '{"entities": [{"id": "Garrus", "type": "Character"}], "relationships": []}'
            tool_calls = [tool_calls()]
    choices = [choices()]

@patch('lore_graph_builder.openai_client.chat.completions.create', return_value=MockOpenAI())
def test_extraction_schema(mock_api):
    result = extract_knowledge("Garrus is a Turian.")
    
    # Assert schema adherence
    assert "entities" in result
    assert result["entities"][0]["id"] == "Garrus"
    assert result["entities"][0]["type"] == "Character"
```

---

## Part 10 — Integration Testing

- **Text-to-Cypher Test:**
  - Create a mock Neo4j DB with 3 nodes: `(Shepard)-[:BORN_ON]->(Earth)`.
  - Pass the question "Where was Shepard born?" to the Graph QA Agent.
  - Intercept the generated Cypher query.
  - Assert the query is `MATCH (c:Character {id: 'SHEPARD'})-[:BORN_ON]->(l:Location) RETURN l`.
  - Assert the final output contains "Earth".

---

## Part 11 — Scaling Discussion

| Axis | Strategy |
|------|----------|
| **Entity Resolution Scale** | Calculating pairwise cosine similarity for 100,000 entities is $O(N^2)$ (10 billion checks). We must use "Blocking". Only compare entities that share the same first 3 letters or same Type (don't compare a Location to a Character). |
| **Graph Density** | If everything connects to everything (e.g., "The Reapers" are connected to 500 characters), visualization becomes a useless hairball. We must implement PageRank or Betweenness Centrality on the graph, and filter the UI to only show the Top 20 most important relationships for a given node. |

---

## Part 12 — Tradeoffs

| Decision | Tradeoff |
|----------|----------|
| LLM Extraction vs Custom NLP (Spacy) | Training a custom Spacy NER/RE model is free to run, extremely fast, and deterministic. But it requires thousands of human-labeled training sentences. LLMs are zero-shot and require no training, but are expensive, slow, and sometimes hallucinate nodes. |
| Graph RAG vs Standard Vector RAG | Standard Vector RAG (embedding text chunks) fails miserably at multi-hop queries ("Who are the enemies of the friends of Shepard?"). Graph RAG excels at multi-hop, but requires a complex ETL pipeline to build the graph first. |

---

## Part 13 — Alternative Approaches

1. **Hybrid GraphRAG (Microsoft approach):** Don't just build a raw graph. Extract the entities, build the graph, and then use the LLM to write a text "Summary" of every community/cluster in the graph. When a user asks a question, embed the summaries, retrieve the relevant community, and generate the answer.
2. **Wikidata/DBpedia bridging:** Use a standard Entity Linking model (like REL) to link game characters to existing public fan-wikis, automatically inheriting all the structured infobox data from the wiki instead of parsing raw text.

---

## Part 14 — Failure Scenarios

| Failure | Impact | Mitigation |
|---------|--------|-----------|
| Predicate Explosion | The LLM creates `KILLED`, `MURDERED`, `ASSASSINATED`, `DESTROYED`. The graph has 100 different edge types that all mean the same thing, making querying impossible. | Force a strict taxonomy. Pass a list of 20 allowed Predicates in the LLM prompt. If it tries to use `MURDERED`, the function calling schema rejects it and forces it to use `KILLED`. |
| Coreference Failure | Text says: "Shepard entered the room. He shot the alien." The LLM extracts `(He)-[:SHOT]->(Alien)`. "He" is a useless node. | Run a Coreference Resolution model (like NeuralCoref) over the raw text to replace all pronouns with proper nouns *before* passing the text to the LLM. |

---

## Part 15 — Debugging

**Symptom:** Writers ask "Who killed Saren?" The Text-to-Cypher agent writes a valid query, executes it, gets zero results, and replies "I don't know." You manually check Neo4j and confirm `(Shepard)-[:KILLED]->(Saren)` exists.

**Debugging steps:**
1. Check the Cypher query generated by the LLM.
2. The LLM wrote: `MATCH (n)-[:KILLED]->(c:Character {id: "Saren"}) RETURN n`.
3. Check the casing. The LLM searched for `"Saren"`, but in your ingestion pipeline (Step 2), you uppercased all IDs to `"SAREN"`. 
4. **Fix:** The LLM doesn't know about your backend normalization rules. You must use Full-Text Search indexing in Neo4j (`db.index.fulltext.queryNodes()`) which is case-insensitive and handles slight misspellings, rather than relying on exact string matching in Cypher.

---

## Part 16 — Monitoring

| Metric | Alert Threshold |
|--------|----------------|
| `cypher_syntax_error_rate` | > 5% → The LLM is forgetting how to write Cypher. Update the prompt with more Few-Shot examples. |
| `orphan_node_count` | > 10% of DB → Extraction pipeline is failing to find relationships, creating disconnected nodes. |
| `average_node_degree` | Spikes massively → Entity resolution is failing, merging completely unrelated nodes together into a super-node. |

---

## Part 17 — Production Improvements

1. **Temporal Graph:** Add `start_time` and `end_time` properties to edges. Shepard and Ashley are `ALLIED_WITH` in Game 1, but `ENEMIES_WITH` in Game 3. Without temporal properties, writers will get conflicting answers.
2. **Source Citations:** Add a `source_doc` property to every relationship edge. When the agent answers "Shepard killed Saren," the UI provides a clickable link to the exact paragraph in the Game Design Document that proves it.

---

## Part 18 — Follow-up Questions

> *Interviewer asks these after the initial solution is presented.*

1. **"The LLM correctly extracts `(Reapers)-[:DESTROYED]->(Earth)`. However, later in the text, someone is telling a lie: 'The Turians destroyed Earth.' The LLM extracts `(Turians)-[:DESTROYED]->(Earth)`. How does the graph handle conflicting lore/lies?"**
2. **"To save money on GPT-4, you want to switch to a local open-source LLM (like Llama-3 8B) for extraction. What are the main challenges of using a small model for Information Extraction?"**
3. **"We want to expose this Knowledge Graph directly inside the Unreal Engine editor so level designers can use it. Neo4j is a heavy Java server. How do you integrate this?"**

---

## Part 19 — Ideal Answers

**Q1 (Conflicting Lore / Epistemology):**
> "We must implement Epistemic tracking (tracking *who* said it). Instead of asserting relationships as universal truths, we reify the edges. We extract: `(Turians)-[:DESTROYED]->(Earth) {stated_by: 'Saren', reliability: 'low'}`. When writers query the graph, they can filter for objective lore vs subjective character dialogue."

**Q2 (Small LLM Challenges):**
> "Small models struggle heavily with strict JSON formatting and function calling. They often forget brackets or hallucinate schema keys. To fix this, we must use constrained decoding (e.g., the `Outlines` or `Guidance` libraries), which forces the LLM at the token-generation level to strictly adhere to the JSON schema, guaranteeing parsable output even from an 8B model."

**Q3 (Engine Integration):**
> "We do not run Neo4j inside Unreal Engine. We keep Neo4j hosted on AWS. We build a lightweight C++ HTTP client plugin for Unreal Engine. The designer types a query into a UI panel in Unreal, the plugin sends a REST request to our Python FastAPI server (which holds the Neo4j driver and Langchain agent), and returns the results back to the Unreal UI."

---

## Part 20 — Evaluation Rubric

### Strong Hire
- Anticipates the need for Entity Resolution / Deduplication immediately.
- Solves the Predicate Explosion problem by forcing a schema constraint.
- Recommends Full-Text search or embeddings for retrieving nodes, rather than exact Cypher `MATCH` strings.
- Flawlessly handles the Coreference and Epistemic (truth) follow-up questions.

### Hire
- Successfully sets up an LLM extraction pipeline using function calling.
- Chooses a Graph DB (Neo4j) over a Relational DB.
- Uses Text-to-Cypher for the natural language UI.

### Lean Hire
- Suggests building the graph manually using standard Regex, vastly underestimating the complexity of human language.
- Creates a schema but ignores the problem of duplicate nodes (Shepard vs Cmdr Shepard).

### Lean No Hire
- Proposes using standard Vector RAG and completely ignores the explicit requirement to build a structured Knowledge Graph.

### No Hire
- Doesn't know what a Knowledge Graph or an Entity is.
- Cannot write a basic JSON extraction prompt.
