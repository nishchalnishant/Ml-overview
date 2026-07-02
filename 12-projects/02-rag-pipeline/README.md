---
module: Projects
topic: RAG Pipeline
subtopic: ""
status: unread
tags: [projects, rag, embeddings, vector-search, llm, hands-on]
---
# Project: Retrieval-Augmented Generation (RAG) Pipeline

**What this is:** a complete, runnable RAG system — chunk documents → embed → index → retrieve → rerank → generate a grounded answer with citations. Runs fully offline with local embeddings and a swappable LLM backend (defaults to a local/no-API-key extractive fallback so it works with zero setup; swap in an OpenAI/Anthropic call for generation quality).

This is the applied counterpart to the RAG coverage in [05-llms/](../../05-llms/) (retrieval, chunking, embeddings, agentic RAG) — here it's actually wired together and runnable.

## Why this project

RAG is the single most common production LLM pattern and the repo's study plans reference it repeatedly as a milestone, but no runnable implementation existed. This builds the full pipeline end-to-end, small enough to run on a laptop against a handful of markdown documents, with every stage (chunking, embedding, indexing, retrieval, reranking, generation, citation) implemented and inspectable.

## Setup

```bash
pip install -r requirements.txt
```

By default this uses `sentence-transformers` (local, no API key) for embeddings and a simple extractive answer synthesizer. To use a real LLM for generation, set `OPENAI_API_KEY` (or `ANTHROPIC_API_KEY`) and pass `--llm openai` / `--llm anthropic` to `query.py` — see `generate.py` for the swap point.

## Run

```bash
python build_index.py         # chunks docs/*.md, embeds, builds a local FAISS-like index (numpy fallback if faiss unavailable)
python query.py "How does attention scaling work?"
```

## Structure

| File | Purpose |
|---|---|
| `docs/` | Small corpus of sample markdown documents to retrieve over (self-contained, no download). |
| `chunking.py` | Fixed-size + overlap chunking with sentence-boundary snapping. |
| `embeddings.py` | Wraps `sentence-transformers` for embedding chunks and queries. |
| `build_index.py` | Chunks the corpus, embeds it, persists the index + metadata to disk. |
| `retrieval.py` | Cosine-similarity top-k retrieval, plus a simple keyword-overlap reranker (cross-encoder-style rerank without extra model weight). |
| `generate.py` | Prompt construction (context + citations) and pluggable generation backend (extractive fallback, OpenAI, Anthropic). |
| `query.py` | CLI entry point: retrieve → rerank → generate → print answer with source citations. |

## Design notes

- **Chunking** uses overlap (see `chunking.py`) so an answer spanning a chunk boundary in the source doc isn't silently dropped.
- **Retrieval is two-stage**: dense cosine-similarity retrieval (top-20) followed by a lightweight lexical-overlap rerank (top-5) — cheap approximation of a cross-encoder rerank stage, discussed in [05-llms/](../../05-llms/) under retrieval/reranking.
- **Citations are mandatory**: `generate.py` forces the prompt template to require `[source: filename]` tags, and the extractive fallback only ever returns text that exists verbatim in a retrieved chunk — this makes hallucination structurally harder to miss during inspection.
- **No vector DB dependency required**: falls back to a brute-force numpy cosine-similarity index if `faiss` isn't installed, so the project runs with the minimal requirements set.

## Where to Next

- **RAG theory, chunking strategies, reranking** → [05-llms/](../../05-llms/)
- **Agentic RAG / multi-step retrieval** → [08-emerging-topics/emerging-trends/agentic-ai-systems.md](../../08-emerging-topics/emerging-trends/agentic-ai-systems.md)
- **Next project: LLM fine-tuning** → [../03-llm-finetuning/](../03-llm-finetuning/)
