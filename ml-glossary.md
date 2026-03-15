# ML Glossary

Short definitions for classical ML, deep learning, and modern AI systems.

---

## Classical ML

- **Regression**: Predicting a continuous target (e.g. price). **Linear regression** fits a line/plane; **logistic regression** predicts class probabilities.
- **Classification**: Predicting a discrete label. **Metrics**: accuracy, precision, recall, F1, ROC-AUC, confusion matrix.
- **SVM (Support Vector Machine)**: Finds a separating hyperplane that maximizes the margin; **kernel** allows non-linear boundaries.
- **Decision tree**: Splits on features by impurity (e.g. Gini, entropy); **ensemble**: many trees (Random Forest = bagging; boosting = AdaBoost, GBM, XGBoost).
- **Clustering**: Grouping unlabeled data (e.g. K-means). **Dimensionality reduction**: PCA, t-SNE — reduce features for visualization or preprocessing.
- **Overfitting / Underfitting**: Model too complex (fits noise) vs too simple (misses signal). **Bias–variance tradeoff**: balance approximation vs sensitivity to data.

---

## Deep learning

- **Neural network**: Layers of weighted sums + non-linear **activation** (ReLU, GELU, softmax). **Backpropagation**: compute gradients of loss w.r.t. weights by chain rule backward.
- **Loss function**: Objective to minimize (e.g. cross-entropy for classification, MSE for regression). **Optimizer**: SGD, Adam — update weights from gradients.
- **Regularization**: Dropout, weight decay (L2) to reduce overfitting. **Batch normalization**: normalize activations per batch to stabilize training.
- **Attention**: Weights over inputs from query–key similarity; **multi-head** = several such mechanisms. **Transformer**: architecture built on self-attention and feed-forward layers.
- **Encoder / Decoder**: Encoder reads input; decoder generates output (optionally with **cross-attention** to encoder). **Decoder-only**: no encoder; used in GPT-style LLMs.
- **Pretraining**: Train on large unlabeled/semi-labeled data (e.g. next-token prediction). **Fine-tuning**: Adapt on task-specific data. **Instruction tuning**: Fine-tune on (instruction, response) pairs.
- **RLHF**: Reinforcement Learning from Human Feedback — reward model from preferences, then optimize policy (LLM) with RL. **DPO**: Direct Preference Optimization — align from preferences without separate reward model.

---

## RAG and retrieval

- **RAG (Retrieval Augmented Generation)**: Retrieve relevant chunks from a store, inject into prompt, then generate with LLM. Enables proprietary/up-to-date knowledge and citations.
- **Embedding**: Dense vector representation of text (or item) from an embedding model. **Vector store**: Database that stores embeddings and supports similarity search.
- **Chunking**: Splitting documents into smaller segments for embedding and retrieval. **Reranking**: Second-stage model to score (query, chunk) and improve top-k.
- **Hybrid search**: Combine vector (semantic) and keyword (e.g. BM25) search; merge with RRF or weighted score.
- **ANN**: Approximate nearest neighbor — fast similarity search with slight recall tradeoff. **HNSW**: Graph-based index for ANN. **FAISS**: Library for efficient similarity search.

---

## Agents and tools

- **Agent**: System that reasons and acts in a loop (LLM + tools + memory). **Tool**: Callable function (API, search, code, RAG) the agent can invoke.
- **ReAct**: Pattern of alternating **Thought**, **Action** (tool call), **Observation**. **Plan-and-execute**: Plan steps first, then execute with tools.
- **Orchestrator**: Component that runs the agent loop: build prompt → LLM → parse tool calls → execute → append results → repeat.
- **Multi-agent**: Multiple agents (or roles) coordinated by an orchestrator or manager for specialization or debate.

---

## LLM and systems

- **Context window**: Maximum input tokens the model accepts. **Tokenization**: Splitting text into subword units (BPE, etc.); ~0.75 words per token (English).
- **Prompt engineering**: Designing system and user prompts; few-shot, chain-of-thought, structured output. **Streaming**: Emit output tokens as generated.
- **KV-cache**: Cache key/value from previous tokens to avoid recomputation during generation. **Batching**: Process multiple requests together for higher throughput.
- **Model serving**: vLLM, TGI, TensorRT-LLM, or cloud APIs for low-latency and high-throughput inference.
