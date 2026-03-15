# Modern AI Infrastructure

How modern AI platforms are built: **LLM orchestration**, **vector databases**, **model serving**, and **GPU/distributed training**.

---

## LLM orchestration frameworks

- **LangChain / LangGraph**: Chains, agents, tools, RAG; LangGraph adds state and cycles for agent workflows.
- **LlamaIndex**: RAG-first; indices, query engines, and agent integrations.
- **Semantic Kernel**: Planners, plugins (tools), and orchestration (Microsoft).
- **CrewAI / AutoGen**: Multi-agent coordination and task delegation.
- **OpenAI Assistants API**: Managed threads, tools, and file search.

Use these to compose prompts, tools, retrieval, and multi-step flows without building everything from scratch.

---

## Vector database systems

- **Managed**: Pinecone, Weaviate, Qdrant, Milvus, Atlas (MongoDB), pgvector (Postgres).
- **Self-hosted / embedded**: Chroma, Vespa, LanceDB, FAISS on your own infra.
- **Features**: Similarity search (HNSW, IVF), filtering, replication, hybrid (keyword + vector).

Choose by scale, latency, filtering needs, and ops preference (managed vs self-hosted). See [Vector databases](../llm-applications/vector-databases.md).

---

## Model serving platforms

- **vLLM, TGI (Text Generation Inference), TensorRT-LLM**: High-throughput serving with continuous batching, KV-cache, PagedAttention.
- **Cloud APIs**: OpenAI, Anthropic, Google, Azure OpenAI — no GPU management; pay per token.
- **Considerations**: Latency (TTFT, time per token), throughput (batch size, replicas), cost (tokens, GPU hours).

---

## GPU infrastructure and distributed training

- **Single-node multi-GPU**: Data parallel (same model, split batch); NVLink for fast communication.
- **Multi-node**: Distributed data parallel (DDP); FSDP or tensor parallel for very large models that don’t fit on one node.
- **Efficiency**: Mixed precision (FP16/BF16), gradient checkpointing, and efficient kernels (FlashAttention, etc.) to train large transformers.
- **Orchestration**: Kubernetes with GPU nodes; Slurm or cloud job schedulers for long training runs.

---

## Quick revision

- **Orchestration**: LangChain, LlamaIndex, Semantic Kernel, etc. for composing LLMs, tools, and RAG.
- **Vector DBs**: Managed (Pinecone, Weaviate, pgvector) or self-hosted (Chroma, FAISS) for embedding search.
- **Serving**: vLLM, TGI, or cloud APIs; batching and KV-cache for throughput and latency.
- **Training**: Data parallel and FSDP/tensor parallel; mixed precision and attention optimizations for scale.
