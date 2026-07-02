# Attention and Scaling

The scaled dot-product attention mechanism computes attention weights as softmax(QK^T / sqrt(d_k))V. The scaling factor sqrt(d_k) is critical: without it, the dot products QK^T grow large in magnitude as the dimension d_k increases, pushing the softmax function into regions with extremely small gradients. This is because large logits cause softmax to saturate, assigning nearly all probability mass to a single element and producing gradients close to zero everywhere else.

Multi-head attention runs several attention operations in parallel, each with its own learned projection matrices for queries, keys, and values, then concatenates the results and projects them back to the model dimension. This lets different heads specialize in different types of relationships — some heads attend to syntactic structure, others to long-range coreference.

KV-cache is an inference-time optimization: during autoregressive generation, the key and value projections for previously generated tokens don't change, so they can be cached and reused instead of recomputed at every generation step. This turns a quadratic-per-token cost into a much cheaper linear append operation, at the cost of memory that scales with sequence length and number of layers.
