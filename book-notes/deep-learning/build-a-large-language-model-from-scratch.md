# Build a large language model from scratch

## **Chapter 1: Understanding Large Language Models**&#x20;

***

### **Chapter 1: Understanding Large Language Models**

#### **1.1 What is an LLM?**

* A **Large Language Model (LLM)** is a deep neural network designed to understand, generate, and process human-like text.
* LLMs are trained on vast datasets, often including internet text, books, and research papers.
* The term "large" refers to:
  * **Model Size**: LLMs contain billions of parameters.
  * **Training Data**: They learn from extensive text corpora.
* Training is based on **next-word prediction**, where the model learns to predict the following word in a sentence, capturing context and linguistic structure.
* LLMs rely on the **Transformer architecture**, which improves efficiency in processing sequences of words.

***

#### **1.2 Applications of LLMs**

* **Text Generation**: Creating original content, writing articles, summarizing text.
* **Machine Translation**: Converting text between languages.
* **Sentiment Analysis**: Determining emotional tone in text.
* **Chatbots & Virtual Assistants**: Powering AI assistants like ChatGPT and Google Gemini.
* **Code Generation**: Writing and debugging computer programs.
* **Knowledge Retrieval**: Extracting information from large document collections.

***

#### **1.3 Stages of Building and Using LLMs**

LLM development occurs in **two main phases**:

1. **Pretraining**:
   * The model learns general language patterns from a massive corpus of **unlabeled** text.
   * Example: GPT-3 was pretrained on billions of words.
2. **Fine-tuning**:
   * The pretrained model is specialized for tasks such as **classification** or **question answering** using **labeled datasets**.

Advantages of Custom LLMs:

* **Better Performance**: Custom models outperform general-purpose LLMs on domain-specific tasks.
* **Data Privacy**: Organizations can train models on **private data** rather than relying on third-party APIs.
* **Lower Latency**: Running models **locally** (e.g., on a laptop) can reduce response times and server costs.

***

#### **1.4 Introducing the Transformer Architecture**

* LLMs are built on the **Transformer** model, introduced in the 2017 paper **"Attention Is All You Need."**
* **Key Components of Transformers**:
  1. **Encoder**: Converts input text into vector representations.
  2. **Decoder**: Generates the output text based on learned representations.
  3. **Self-Attention Mechanism**: Allows the model to selectively focus on different parts of the input when making predictions.
* Variants of Transformers:
  * **BERT (Bidirectional Encoder Representations from Transformers)**: Used for classification tasks (e.g., sentiment analysis).
  * **GPT (Generative Pretrained Transformer)**: Used for text generation (e.g., ChatGPT).

***

#### **1.5 Utilizing Large Datasets**

* LLMs require **massive training datasets** for pretraining.
* Example: **GPT-3 dataset**
  * **CommonCrawl** (filtered web data) - 60%
  * **WebText2** (curated internet text) - 22%
  * **Books1 & Books2** (book corpora) - 16%
  * **Wikipedia** (high-quality reference text) - 3%
* Training requires **enormous computing power**, making open-source models essential for researchers.

***

#### **1.6 A Closer Look at the GPT Architecture**

* GPT models **use only the decoder** from the Transformer architecture.
* **Pretraining** involves predicting the next word in a sequence, a simple yet effective task for learning contextual relationships.
* GPT models exhibit **emergent behavior**, meaning they can perform tasks (e.g., translation) without being explicitly trained for them.
* They can perform:
  * **Zero-shot learning**: Answering queries without prior training.
  * **Few-shot learning**: Learning a task from a few examples.

***

#### **1.7 Building a Large Language Model**

* The book outlines a **three-stage approach** to building an LLM:
  1. **Stage 1: Implementing the LLM Architecture**
     * Preparing the dataset.
     * Designing the **attention mechanism**.
  2. **Stage 2: Pretraining the LLM**
     * Training on **unlabeled data** to create a **foundation model**.
  3. **Stage 3: Fine-tuning the LLM**
     * Specializing the model for tasks like **classification** or **personal assistants**.

***

#### **Summary**

* **LLMs have revolutionized NLP**, outperforming traditional rule-based and statistical models.
* **Pretraining on large datasets** allows LLMs to generalize across diverse language tasks.
* **Transformers** are the backbone of LLMs, enabling deep contextual learning.
* **GPT models are autoregressive**, generating text **one word at a time**.
* **Fine-tuning enables specialization**, improving performance on domain-specific tasks.

***

This chapter lays the foundation for building an LLM from scratch by introducing key concepts such as the Transformer model, data requirements, and the overall training process. The next chapter delves into **text data processing**, including **tokenization, embeddings, and data sampling techniques**.

***

### **Chapter 2: Working with Text Data**

#### **2.1 Understanding Word Embeddings**

* LLMs cannot process raw text directly; they need **word embeddings** to convert words into continuous numerical vectors.
* Embeddings map words into a **multi-dimensional space**, preserving relationships between words.
* Different types of embeddings:
  * **Word-level embeddings**: Represent individual words.
  * **Sentence/Paragraph embeddings**: Used in retrieval-augmented generation (RAG).
  * **Contextual embeddings**: Adapt based on sentence context (e.g., BERT, GPT).

***

#### **2.2 Tokenizing Text**

* **Tokenization** is the process of breaking text into smaller components (tokens).
* **Basic tokenization approaches**:
  1. **Whitespace-based tokenization**: Splits text using spaces.
  2. **Punctuation-aware tokenization**: Considers punctuation as separate tokens.
  3. **Custom tokenization**: Uses regex-based splitting.

**Example of punctuation-aware tokenization:**

```python
import re
text = "Hello, world. Is this a test?"
tokens = re.split(r'([,.!?]|\s)', text)
tokens = [t.strip() for t in tokens if t.strip()]
print(tokens)
```

**Output:**

```
['Hello', ',', 'world', '.', 'Is', 'this', 'a', 'test', '?']
```

* A simple tokenizer can convert text to **token IDs** using a vocabulary dictionary.
* **Tokenization challenges**:
  * Handling **out-of-vocabulary (OOV) words**.
  * Preserving **word relationships and context**.
  * Reducing **memory and computational cost**.

***

#### **2.3 Converting Tokens into Token IDs**

* Each token is mapped to a unique **token ID** using a predefined vocabulary.
*   Example:

    ```python
    vocab = {"Hello": 1, ",": 2, "world": 3, ".": 4, "test": 5}
    tokens = ["Hello", ",", "world", "."]
    token_ids = [vocab[t] for t in tokens]
    print(token_ids)  # Output: [1, 2, 3, 4]
    ```
*   **Reverse mapping** (Token IDs → Text):

    ```python
    inv_vocab = {v: k for k, v in vocab.items()}
    text = [inv_vocab[i] for i in token_ids]
    print(" ".join(text))  # Output: "Hello , world ."
    ```
* This process is essential for **training and inference** in LLMs.

***

#### **2.4 Adding Special Context Tokens**

* LLMs often include special tokens to **structure text inputs**:
  * `<|unk|>` (Unknown token) → Replaces unseen words.
  * `<|endoftext|>` (End of text) → Marks the boundary between different documents.
  * `<|pad|>` (Padding) → Ensures all inputs in a batch have the same length.
*   Example of tokenizing text with `<|endoftext|>`:

    ```
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces of the palace."
    ```

    * The `<|endoftext|>` token helps LLMs differentiate separate documents.

***

#### **2.5 Byte Pair Encoding (BPE)**

* BPE is an advanced tokenization technique used in **GPT-based models**.
* **How BPE works**:
  1. Start with **character-level tokens**.
  2. Find the most **frequent adjacent pair of tokens**.
  3. Merge the pair into a **new subword token**.
  4. Repeat until reaching the desired vocabulary size.
* Example:
  * Given text: `"low", "lowest", "newer", "wider"`
  * Initial tokens: `["l", "o", "w", "e", "s", "t", "n", "e", "w", "e", "r", "w", "i", "d", "e", "r"]`
  * Merge the most frequent pair `("l", "o") → "lo"` → `["lo", "w", "e", "s", "t", ...]`
  * Continue until getting **subword-level tokens**.
* **Benefits of BPE**:
  * **Efficient vocabulary compression**.
  * **Handles OOV words** by breaking them into subwords.
  * **Improves generalization** in LLMs.

***

#### **2.6 Data Sampling with a Sliding Window**

* **Sliding window approach** helps create training samples for LLMs.
* The dataset is divided into overlapping **input-output pairs**.

**Example:**

* Given sentence: `"The cat sat on the mat."`
*   **Window size = 3**, **stride = 1**:

    ```
    Input:  ["The", "cat", "sat"] → Target: "on"
    Input:  ["cat", "sat", "on"] → Target: "the"
    Input:  ["sat", "on", "the"] → Target: "mat"
    ```
* This method ensures **better learning of dependencies** between words.

***

#### **2.7 Creating Token Embeddings**

* Token IDs must be converted into **embedding vectors** before feeding them into an LLM.
* **Embedding layer** acts as a lookup table, mapping token IDs to dense vectors.

**Example:**

*   Given vocabulary:

    ```python
    vocab = {"hello": 0, "world": 1, "<|endoftext|>": 2}
    ```
*   Token IDs:

    ```python
    token_ids = [0, 1, 2]
    ```
*   Convert them into **embedding vectors**:

    ```python
    import torch
    embedding_layer = torch.nn.Embedding(num_embeddings=3, embedding_dim=4)
    embeddings = embedding_layer(torch.tensor(token_ids))
    print(embeddings.shape)  # Output: torch.Size([3, 4])
    ```
* **Final embeddings** are **input to the transformer model**.

***

#### **2.8 Encoding Word Positions**

* **Positional embeddings** are used to preserve **word order** in sequences.
* LLMs like GPT use **learned positional embeddings**, added to **token embeddings**.

**Example of positional embedding:**

```python
pos_embedding_layer = torch.nn.Embedding(num_embeddings=100, embedding_dim=4)
position_ids = torch.arange(3)  # Example for a sequence of length 3
pos_embeddings = pos_embedding_layer(position_ids)
print(pos_embeddings.shape)  # Output: torch.Size([3, 4])
```

*   **Final input to the model**:

    ```
    input_embeddings = token_embeddings + pos_embeddings
    ```
* **Ensures the model understands word order**.

***

#### **Summary**

* **LLMs need text to be converted into numerical vectors** for training.
* **Tokenization splits text into words/subwords**, followed by **mapping tokens to IDs**.
* **Byte Pair Encoding (BPE)** improves handling of rare words.
* **Sliding window sampling** creates input-target pairs for training.
* **Token embeddings + positional embeddings** form the **final model input**.

***

This chapter focuses on **text preprocessing** for training an LLM. The next chapter covers **implementing the attention mechanism**, a key component of transformer models.



***

### **Chapter 3: Coding Attention Mechanisms**

#### **3.1 The Problem with Modeling Long Sequences**

* Before **transformers**, **Recurrent Neural Networks (RNNs)** were commonly used for sequence-based tasks like language modeling and machine translation.
* RNNs process sequences step-by-step, maintaining a **hidden state** that captures previous inputs.
* **Limitations of RNNs**:
  * **Loss of long-range dependencies**: Earlier words in long texts fade in importance.
  * **Sequential processing**: Cannot be parallelized efficiently.
  * **Difficulty in learning complex dependencies**.
* To address these issues, **attention mechanisms** were introduced, allowing the model to selectively focus on relevant parts of input sequences.

***

#### **3.2 Capturing Data Dependencies with Attention Mechanisms**

* **Traditional RNN-based encoder-decoder models** require compressing an entire input sequence into a fixed-size vector, leading to information loss.
* **Attention mechanisms** allow models to **dynamically focus** on relevant input elements at each step.
* This idea was first introduced in **Bahdanau Attention (2014)** for sequence-to-sequence tasks like translation.
* **Transformers eliminate RNNs** by relying solely on attention mechanisms.

***

#### **3.3 Attending to Different Parts of Input with Self-Attention**

* **Self-Attention** is the key innovation in transformers:
  * Instead of processing tokens sequentially (like RNNs), **self-attention allows each token to consider all other tokens in the sequence simultaneously**.
* **How Self-Attention Works**:
  1. Each word is embedded into a **vector representation**.
  2. The model computes **attention scores** that determine how much focus each word should have on every other word in the sequence.
  3. These scores are used to compute a **weighted sum**, producing **context vectors** that capture dependencies between words.

**Implementation of Self-Attention**

*   **Step 1: Compute attention scores using dot product**

    ```python
    import torch
    inputs = torch.tensor([
        [0.43, 0.15, 0.89],  # Word 1
        [0.55, 0.87, 0.66],  # Word 2
        [0.57, 0.85, 0.64],  # Word 3
    ])

    query = inputs[1]  # Selecting second word as query
    attn_scores = torch.tensor([torch.dot(x, query) for x in inputs])
    print(attn_scores)  # Output: [dot-product scores]
    ```
*   **Step 2: Apply Softmax to Normalize Scores**

    ```python
    attn_weights = torch.softmax(attn_scores, dim=0)
    print(attn_weights)  # Normalized attention weights
    ```
*   **Step 3: Compute Context Vector**

    ```python
    context_vec = sum(attn_weights[i] * inputs[i] for i in range(len(inputs)))
    print(context_vec)  # Output: Weighted sum of inputs
    ```

***

#### **3.4 Implementing Self-Attention with Trainable Weights**

* In real LLMs, self-attention is implemented using **trainable weight matrices**.
* Instead of directly using token embeddings, we compute **Queries (Q), Keys (K), and Values (V)**.
  * **Query (Q)**: The vector representing the current token.
  * **Key (K)**: The vector representing other tokens in the sequence.
  * **Value (V)**: The information to be aggregated based on attention scores.

**Implementation of Scaled Dot-Product Attention**

```python
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out)
        self.W_key = nn.Linear(d_in, d_out)
        self.W_value = nn.Linear(d_in, d_out)
    
    def forward(self, x):
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.T / (keys.shape[-1] ** 0.5)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        context_vec = attn_weights @ values
        return context_vec
```

* **Key Takeaways**:
  * **Scaling by √d** prevents vanishing gradients.
  * **Softmax ensures attention scores sum to 1**.
  * **Dot-product attention is efficient and parallelizable**.

***

#### **3.5 Hiding Future Words with Causal Attention**

* **Causal Attention (Masked Attention)** ensures that a model does not "see" future tokens when predicting the next word.
* **In GPT models, causal masks prevent bidirectional context.**
* Implementation uses a **triangular mask** that sets attention scores to **-inf** for future words.

**Implementation of Causal Attention**

```python
def create_causal_mask(seq_len):
    mask = torch.tril(torch.ones(seq_len, seq_len))  # Lower triangular matrix
    mask[mask == 0] = float('-inf')  # Set future words to -inf
    return mask
```

```python
attn_scores = queries @ keys.T / (keys.shape[-1] ** 0.5)
mask = create_causal_mask(attn_scores.shape[0])
attn_scores += mask  # Apply causal mask
attn_weights = torch.softmax(attn_scores, dim=-1)
context_vec = attn_weights @ values
```

* **Effect**: Model can only use previous tokens when predicting next token.

***

#### **3.6 Extending Self-Attention to Multi-Head Attention**

* **Single-head attention** may miss important relationships in text.
* **Multi-head attention**:
  * Splits input into multiple smaller projections.
  * Each head learns **different attention patterns**.
  * Outputs from multiple heads are concatenated.

**Implementation of Multi-Head Attention**

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.W_query = nn.Linear(d_in, d_out * num_heads)
        self.W_key = nn.Linear(d_in, d_out * num_heads)
        self.W_value = nn.Linear(d_in, d_out * num_heads)
        self.out_proj = nn.Linear(d_out * num_heads, d_out)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        queries = self.W_query(x).view(batch_size, seq_len, self.num_heads, -1)
        keys = self.W_key(x).view(batch_size, seq_len, self.num_heads, -1)
        values = self.W_value(x).view(batch_size, seq_len, self.num_heads, -1)
        attn_scores = (queries @ keys.transpose(-2, -1)) / (keys.shape[-1] ** 0.5)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        context_vec = (attn_weights @ values).reshape(batch_size, seq_len, -1)
        return self.out_proj(context_vec)
```

* **GPT-2 (117M)**: **12 attention heads, embedding size 768**.
* **GPT-2 (1.5B)**: **25 attention heads, embedding size 1600**.

***

#### **Summary**

* **Attention mechanisms** improve long-sequence processing by focusing on relevant tokens.
* **Self-attention** computes **context vectors** using a **dot-product attention mechanism**.
* **Trainable weight matrices (Q, K, V)** allow the model to **learn contextual relationships**.
* **Causal attention ensures models predict words left-to-right**.
* **Multi-head attention enhances representation learning**.
* **Attention-based transformers replace RNNs for NLP tasks**.

***

This chapter focuses on **attention mechanisms**, a critical building block for LLMs. The next chapter covers **assembling the complete LLM architecture** and **training a GPT-like model**.



***

### **Chapter 4: Implementing a GPT Model from Scratch to Generate Text**

#### **4.1 Coding an LLM Architecture**

* **LLMs like GPT** generate text one word (token) at a time.
* GPT consists of **multiple transformer blocks**.
*   **Key model configurations**:

    ```python
    GPT_CONFIG_124M = {
        "vocab_size": 50257,   # Vocabulary size
        "context_length": 1024, # Maximum token sequence length
        "emb_dim": 768,        # Embedding dimension
        "n_heads": 12,         # Attention heads
        "n_layers": 12,        # Transformer layers
        "drop_rate": 0.1,      # Dropout probability
        "qkv_bias": False      # Query-Key-Value bias
    }
    ```
* **Components of GPT Architecture**:
  * **Token and Positional Embeddings** (convert tokens into dense representations).
  * **Transformer Blocks** (self-attention, feed-forward layers, normalization).
  * **Output Layer** (maps hidden states to vocabulary probabilities).

***

#### **4.2 Normalizing Activations with Layer Normalization**

* **LayerNorm (Layer Normalization)** stabilizes training by normalizing activations.
* Applied **before** attention and feed-forward layers (Pre-LayerNorm).
*   Code:

    ```python
    class LayerNorm(nn.Module):
        def __init__(self, normalized_shape, eps=1e-5):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
            self.eps = eps

        def forward(self, x):
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True)
            return self.weight * (x - mean) / (var + self.eps).sqrt() + self.bias
    ```

***

#### **4.3 Implementing a Feed-Forward Network with GELU Activation**

* Each transformer block contains a **Feed-Forward Network (FFN)**.
*   Uses **GELU (Gaussian Error Linear Unit)** activation:

    ```python
    import torch.nn.functional as F
    class FeedForward(nn.Module):
        def __init__(self, emb_dim):
            super().__init__()
            self.fc1 = nn.Linear(emb_dim, 4 * emb_dim)
            self.fc2 = nn.Linear(4 * emb_dim, emb_dim)

        def forward(self, x):
            return self.fc2(F.gelu(self.fc1(x)))
    ```
* The FFN expands embeddings **4x**, then projects back to the original size.

***

#### **4.4 Adding Shortcut (Residual) Connections**

* **Shortcut connections (Residual connections)** help prevent gradient vanishing.
* Formula: Output=x+f(x)\text{Output} = x + f(x)
* **Why?**
  * Allows gradients to flow through deep networks.
  * Helps training deeper transformers.

***

#### **4.5 Connecting Attention and FFN Layers in a Transformer Block**

* The **Transformer Block** combines:
  * **Multi-Head Attention**
  * **Feed-Forward Network**
  * **Layer Normalization**
  * **Residual Connections**
*   **Implementation of a Transformer Block**:

    ```python
    class TransformerBlock(nn.Module):
        def __init__(self, emb_dim, n_heads, drop_rate):
            super().__init__()
            self.ln1 = LayerNorm(emb_dim)
            self.attn = MultiHeadAttention(emb_dim, n_heads)
            self.ln2 = LayerNorm(emb_dim)
            self.ffn = FeedForward(emb_dim)
            self.drop = nn.Dropout(drop_rate)

        def forward(self, x):
            x = x + self.drop(self.attn(self.ln1(x)))
            x = x + self.drop(self.ffn(self.ln2(x)))
            return x
    ```
* **Key Features**:
  * Applies **LayerNorm before attention and FFN** (Pre-Norm).
  * Uses **residual connections** to improve gradient flow.
  * **Dropout regularization** prevents overfitting.

***

#### **4.6 Coding the GPT Model**

* A GPT model consists of:
  1. **Token Embeddings**: Converts tokens into vectors.
  2. **Positional Embeddings**: Adds positional information.
  3. **Multiple Transformer Blocks**: Main processing units.
  4. **Output Layer**: Maps hidden states to vocabulary probabilities.
*   **Implementation**:

    ```python
    class GPTModel(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
            self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
            self.drop_emb = nn.Dropout(cfg["drop_rate"])
            self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg["emb_dim"], cfg["n_heads"], cfg["drop_rate"]) for _ in range(cfg["n_layers"])])
            self.final_norm = LayerNorm(cfg["emb_dim"])
            self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

        def forward(self, in_idx):
            tok_embeds = self.tok_emb(in_idx)
            pos_embeds = self.pos_emb(torch.arange(in_idx.shape[1], device=in_idx.device))
            x = tok_embeds + pos_embeds
            x = self.drop_emb(x)
            x = self.trf_blocks(x)
            x = self.final_norm(x)
            return self.out_head(x)
    ```
* **Important Points**:
  * Uses **embeddings** for input tokens and positions.
  * Passes embeddings through **transformer blocks**.
  * Outputs **logits over vocabulary**.

***

#### **4.7 Generating Text**

* GPT generates text **one token at a time**.
* The model takes **previous tokens as context** and predicts the next token.

**Text Generation Process**

1. **Encode input text** into token IDs.
2. **Pass through GPT model** to get next-token probabilities.
3. **Select next token** using greedy decoding or sampling.
4. **Append token** to input and repeat.

*   **Implementation of Greedy Decoding**:

    ```python
    def generate_text_simple(model, idx, max_new_tokens, context_size):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_size:]
            with torch.no_grad():
                logits = model(idx_cond)
            logits = logits[:, -1, :]
            probas = torch.softmax(logits, dim=-1)
            idx_next = torch.argmax(probas, dim=-1, keepdim=True)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
    ```

**Example Output**

* **Input**: `"Hello, I am"`
* **Generated Output**: `"Hello, I am a model ready to help."`

***

#### **4.8 Memory and Parameter Requirements**

* **GPT-2 Small (124M parameters)**:
  * **163 million total parameters** (considering weight tying).
  * **Memory Usage**: \~621MB (assuming 32-bit float precision).
  * **Scaling Up**:
    * **GPT-2 Medium**: 345M parameters.
    * **GPT-2 Large**: 762M parameters.
    * **GPT-2 XL**: 1.5B parameters.

***

#### **Summary**

* **GPT models use transformer blocks** with **self-attention and feed-forward layers**.
* **Layer normalization, shortcut connections, and dropout** help training.
* **Text generation** follows an iterative process where GPT predicts one token at a time.
* **Scaling up GPT models** increases memory and computation needs.

***

This chapter covered **implementing GPT from scratch**. The next chapter focuses on **pretraining the model on unlabeled data**.

***

### **Chapter 5: Pretraining on Unlabeled Data**

#### **5.1 Evaluating Generative Text Models**

* Before training, we need ways to evaluate **text generation quality**.
* **Evaluation steps**:
  1. Generate text using GPT (from Chapter 4).
  2. Measure the **loss** (difference between predicted and actual tokens).
  3. Compare **training and validation loss** to monitor overfitting.

**Setting Up GPT for Evaluation**

```python
import torch
from chapter04 import GPTModel

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.eval()
```

* The **context length** is set to **256 tokens** instead of 1024 to reduce computational cost.

**Training vs Validation Loss**

* **Training loss**: Measures how well the model fits the training data.
* **Validation loss**: Assesses how well the model generalizes to unseen data.

***

#### **5.2 Training an LLM**

* **Pretraining involves minimizing loss** using backpropagation and optimization.
* The training loop follows **eight key steps**:
  1. **Iterate over epochs**.
  2. **Iterate over mini-batches**.
  3. **Reset gradients** before processing a batch.
  4. **Compute loss** between predicted and actual tokens.
  5. **Backpropagate loss** to update model weights.
  6. **Update weights** using an optimizer.
  7. **Print training/validation loss** periodically.
  8. **Generate sample text** for evaluation.

**Training Loop Implementation**

```python
def train_model_simple(model, train_loader, val_loader,
                       optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Epoch {epoch+1}, Step {global_step}: "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

    return train_losses, val_losses, track_tokens_seen
```

***

#### **5.3 Controlling Text Generation with Decoding Strategies**

* Text generation involves **choosing the next token** from a probability distribution.
* Strategies for controlling randomness:
  1. **Greedy Decoding**: Always selects the highest-probability token.
  2. **Temperature Scaling**: Adjusts randomness by modifying softmax probabilities.
  3. **Top-k Sampling**: Selects from the top-k most probable tokens.
  4. **Nucleus Sampling (Top-p Sampling)**: Chooses from a **probability mass p**.

**Temperature Scaling**

```python
temperature = 0.8  # Lower = more deterministic, Higher = more random
logits = model(idx_cond)
probas = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
idx_next = torch.argmax(probas, dim=-1, keepdim=True)
```

**Top-k Sampling**

```python
k = 50  # Consider only top 50 tokens
top_k_vals, top_k_idx = torch.topk(probas, k, dim=-1)
probas = top_k_vals / torch.sum(top_k_vals, dim=-1, keepdim=True)
idx_next = top_k_idx[torch.multinomial(probas, 1)]
```

***

#### **5.4 Saving and Loading Model Weights**

* **Saving weights ensures training can resume later**.
*   Uses `torch.save()` to store model parameters:

    ```python
    torch.save(model.state_dict(), "model.pth")
    ```
*   **Loading saved weights**:

    ```python
    model.load_state_dict(torch.load("model.pth"))
    model.eval()  # Switch to evaluation mode
    ```

***

#### **5.5 Loading Pretrained Weights from OpenAI**

* Instead of training from scratch, we can **load publicly available weights**.
*   **Example: Loading GPT-2 weights**:

    ```python
    from transformers import GPT2LMHeadModel

    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    ```
* Benefits:
  * Saves time and compute resources.
  * Provides a strong foundation for **fine-tuning** on specific tasks.

***

#### **Summary**

* **Pretraining LLMs** improves performance by learning from **unlabeled text data**.
* **Training loss and validation loss** track model performance.
* **Decoding strategies** like **top-k sampling** and **temperature scaling** improve text generation.
* **Loading pretrained weights from OpenAI** can save computation costs.

***

This chapter covers **training a GPT model from scratch**, **evaluating text generation**, and **optimizing LLM performance**. The next chapter discusses **fine-tuning for specific tasks like text classification**.



Here are detailed notes from **Chapter 6: Fine-Tuning for Classification** of _Build a Large Language Model (From Scratch)_ by Sebastian Raschka.

***

### **Chapter 6: Fine-Tuning for Classification**

#### **6.1 Different Categories of Fine-Tuning**

Fine-tuning an LLM can be done in two major ways:

1. **Instruction Fine-Tuning**
   * The model learns to follow specific instructions for various tasks.
   * Example: Translating text, summarizing documents, answering questions.
2. **Classification Fine-Tuning**
   * The model is trained to predict specific class labels (e.g., spam vs. not spam).
   * This is commonly used in **sentiment analysis, topic classification, medical diagnosis**.

**Key Differences**:

* **Instruction fine-tuning** is more **generalized** but requires **large datasets**.
* **Classification fine-tuning** is **task-specific** and more **efficient**.

***

#### **6.2 Preparing the Dataset**

* The example in this chapter focuses on a **spam classification task**.
* Uses the **SMS Spam Collection dataset** (downloaded from UCI Machine Learning Repository).
* **Preprocessing Steps**:
  1. Convert text to lowercase.
  2. Remove punctuation and special characters.
  3. Tokenize text.
  4. Convert text to token IDs using a tokenizer.
  5. Pad or truncate sequences to a fixed length.

**Downloading and Preprocessing Dataset in Python**:

```python
import urllib.request
import zipfile
import os
from pathlib import Path

url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
zip_path = "sms_spam_collection.zip"
extracted_path = "sms_spam_collection"

# Download and extract dataset
urllib.request.urlretrieve(url, zip_path)
with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(extracted_path)
```

***

#### **6.3 Creating Data Loaders**

* **Convert text dataset into PyTorch tensors**.
* Use `torch.utils.data.Dataset` and `torch.utils.data.DataLoader` for efficient batch loading.

**Creating the Dataset Class**:

```python
import torch
from torch.utils.data import Dataset, DataLoader

class SpamDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = self.tokenizer.encode(self.texts[idx])
        tokens = tokens[:self.max_length] + [0] * (self.max_length - len(tokens))
        return torch.tensor(tokens), torch.tensor(self.labels[idx])
```

* This **ensures all inputs are of equal length**.

**Loading Data**:

```python
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
```

***

#### **6.4 Initializing a Model with Pretrained Weights**

* Load the **pretrained GPT model** from Chapter 5.
* Freeze most of the model’s parameters to save compute.
* Replace the **output layer** to classify text into 2 categories: **spam (1) or not spam (0).**

**Freezing Model Parameters**:

```python
for param in model.parameters():
    param.requires_grad = False  # Freeze pretrained layers
```

**Replacing Output Layer for Classification**:

```python
import torch.nn as nn

num_classes = 2
model.out_head = nn.Linear(model.config["emb_dim"], num_classes)
```

***

#### **6.5 Adding a Classification Head**

* The **original GPT output layer** generates predictions for **50,257 vocabulary tokens**.
* For classification, **replace it with a layer that outputs only 2 values (spam/not spam).**

**Updated GPT Model Architecture**:

```
GPTModel(
  (tok_emb): Embedding(50257, 768)
  (pos_emb): Embedding(1024, 768)
  (trf_blocks): Sequential(
      TransformerBlock x 12
  )
  (final_norm): LayerNorm()
  (out_head): Linear(768 → 2)  # New classification layer
)
```

***

#### **6.6 Calculating the Classification Loss and Accuracy**

*   **Cross-Entropy Loss** is used for classification:

    ```python
    def calc_loss_batch(input_batch, target_batch, model, device):
        logits = model(input_batch)[:, -1, :]
        loss = torch.nn.functional.cross_entropy(logits, target_batch)
        return loss
    ```
*   **Accuracy Calculation**:

    ```python
    def calc_accuracy_loader(data_loader, model, device):
        correct_predictions, num_examples = 0, 0
        for input_batch, target_batch in data_loader:
            logits = model(input_batch)[:, -1, :]
            predicted_labels = torch.argmax(logits, dim=-1)
            correct_predictions += (predicted_labels == target_batch).sum().item()
            num_examples += predicted_labels.shape[0]
        return correct_predictions / num_examples
    ```

***

#### **6.7 Fine-Tuning the Model on Supervised Data**

* Uses **AdamW optimizer** with **weight decay**.
*   Training loop:

    ```python
    def train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs):
        for epoch in range(num_epochs):
            model.train()
            for input_batch, target_batch in train_loader:
                optimizer.zero_grad()
                loss = calc_loss_batch(input_batch, target_batch, model, device)
                loss.backward()
                optimizer.step()
            # Evaluate on validation set
            val_acc = calc_accuracy_loader(val_loader, model, device)
            print(f"Epoch {epoch+1}: Validation Accuracy: {val_acc:.2f}")
    ```

**Results after 5 epochs**:

* Training accuracy: **100%**
* Validation accuracy: **97.5%**

***

#### **6.8 Using the LLM as a Spam Classifier**

* Given an input text, the fine-tuned model predicts **spam or not spam**.

**Inference Code**:

```python
def classify_text(text, model, tokenizer, device):
    input_ids = tokenizer.encode(text)
    input_tensor = torch.tensor(input_ids).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]
    predicted_label = torch.argmax(logits, dim=-1).item()

    return "spam" if predicted_label == 1 else "not spam"
```

**Example Predictions**:

```python
print(classify_text("You won a $1000 cash prize!", model, tokenizer, device))  # Output: spam
print(classify_text("Hey, are we still meeting for dinner?", model, tokenizer, device))  # Output: not spam
```

***

#### **Summary**

* **Classification fine-tuning** adapts an LLM to specific tasks like **spam filtering**.
* **Data preparation** involves **tokenization, padding, and dataset conversion**.
* **Replacing the output layer** enables the model to **predict class labels**.
* **Cross-entropy loss and accuracy metrics** help evaluate model performance.
* **Fine-tuning only the last layers** saves computation while improving accuracy.

***

This chapter **fine-tuned a GPT model for text classification**. The next chapter explores **instruction fine-tuning**, where the LLM follows **natural language instructions**.



***

### **Chapter 7: Fine-Tuning to Follow Instructions**

#### **7.1 Introduction to Instruction Fine-Tuning**

* **Pretrained LLMs** are good at **text completion** but struggle with **following explicit instructions**.
* Fine-tuning on **instruction-response datasets** improves an LLM's ability to generate **helpful and structured responses**.
* **Key applications**:
  * Chatbots (e.g., ChatGPT, Google Gemini)
  * Personal assistants
  * Interactive AI tutors

***

#### **7.2 Preparing a Dataset for Supervised Instruction Fine-Tuning**

* Uses **1,100 instruction-response pairs** created for this book.
* **Alternative datasets**: Stanford’s **Alpaca dataset** (52,000+ entries).
* **Steps to prepare the dataset**:
  1. **Download dataset** (JSON format).
  2. **Inspect dataset**: Each entry contains an **instruction, input text, and expected response**.
  3. **Partition into train (85%), validation (5%), and test (10%) sets**.

**Example Entry:**

```json
{
  "instruction": "Identify the correct spelling of the following word.",
  "input": "Ocassion",
  "output": "The correct spelling is 'Occasion.'"
}
```

**Downloading & Loading Dataset**:

```python
import json, urllib

url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/instruction-data.json"
file_path = "instruction-data.json"

urllib.request.urlretrieve(url, file_path)
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

print("Dataset size:", len(data))  # Output: 1100 entries
```

***

#### **7.3 Organizing Data into Training Batches**

* LLMs require **batch processing** for efficient fine-tuning.
* **Custom collate function**:
  * **Tokenizes text** into token IDs.
  * **Pads sequences** to the same length.
  * **Masks padding tokens** to **avoid affecting loss calculations**.

**Tokenization & Padding Example**:

```python
import torch
from torch.utils.data import Dataset

class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = [tokenizer.encode(entry["instruction"] + " " + entry["input"]) for entry in data]

    def __getitem__(self, idx):
        return self.encoded_texts[idx]

    def __len__(self):
        return len(self.data)
```

***

#### **7.4 Creating Data Loaders**

* **Uses PyTorch DataLoader** to create batches.
* **Batch Size**: Typically **8–32**, depending on GPU memory.

**Creating DataLoaders**:

```python
from torch.utils.data import DataLoader

batch_size = 8
train_dataset = InstructionDataset(train_data, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
```

***

#### **7.5 Loading a Pretrained LLM**

* Instead of training from scratch, we **load a GPT-2 Medium model (355M parameters)**.
* **Pretrained models act as the foundation** for instruction fine-tuning.

**Loading GPT-2 Model**:

```python
from transformers import GPT2LMHeadModel

model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
model.train()
```

***

#### **7.6 Fine-Tuning the LLM on Instruction Data**

* Uses **AdamW optimizer** with a **low learning rate** (`0.00005`).
* Runs for **2–5 epochs** (larger models may require longer training).

**Training Loop:**

```python
import torch.optim as optim

optimizer = optim.AdamW(model.parameters(), lr=0.00005)

for epoch in range(2):
    for batch in train_loader:
        optimizer.zero_grad()
        loss = model(batch).loss
        loss.backward()
        optimizer.step()
```

**Training Loss Trend:**

* **Epoch 1**:
  * Step 0: **Train loss: 2.637, Val loss: 2.626**
  * Step 100: **Train loss: 0.857, Val loss: 0.906**
* **Epoch 2**:
  * Step 200: **Train loss: 0.438, Val loss: 0.670**
  * Step 300: **Train loss: 0.300, Val loss: 0.657**

***

#### **7.7 Extracting and Saving Model Responses**

* Fine-tuned LLM is tested on **unseen instructions** from the test set.

**Generating Model Responses**:

```python
def generate_response(text):
    input_ids = tokenizer.encode(text, return_tensors="pt")
    output_ids = model.generate(input_ids, max_length=256)
    return tokenizer.decode(output_ids[0])
```

**Example Test Cases**

| **Instruction**      | **Input**                  | **Expected Response**             |
| -------------------- | -------------------------- | --------------------------------- |
| Convert to passive   | "The chef cooks the meal." | "The meal is cooked by the chef." |
| Provide synonym      | "bright"                   | "radiant"                         |
| Convert km to meters | "45 kilometers"            | "45,000 meters"                   |

***

#### **7.8 Evaluating the Fine-Tuned LLM**

* Evaluation involves **quantifying the accuracy of generated responses**.
* **Ollama App** (Llama-3 model) **scores LLM responses**.

**GPT-2 Fine-Tuned Model Performance**:

| **Metric**            | **GPT-2 Base** | **GPT-2 Fine-Tuned** |
| --------------------- | -------------- | -------------------- |
| Accuracy              | 40%            | **85%**              |
| Fluency               | Medium         | **High**             |
| Instruction-following | Weak           | **Strong**           |

***

#### **7.9 Conclusions & Next Steps**

* **Fine-tuning significantly improves instruction-following capabilities**.
* **Next Steps**:
  * **Preference Fine-Tuning**: Tailor responses to **specific user preferences**.
  * **LoRA (Low-Rank Adaptation)**: **Faster fine-tuning** with fewer parameters.

**Future Exploration**:

* Check **Axolotl** for LLM fine-tuning:\
  🔗 [Axolotl GitHub](https://github.com/OpenAccess-AI-Collective/axolotl)
* Check **LitGPT** for lightweight training:\
  🔗 [LitGPT GitHub](https://github.com/Lightning-AI/litgpt)

***

#### **Summary**

✅ **Instruction fine-tuning adapts LLMs to generate structured responses.**\
✅ **Dataset preparation involves tokenizing instructions & batching data.**\
✅ **GPT-2 was fine-tuned using a small learning rate over multiple epochs.**\
✅ **Evaluation showed a significant improvement in instruction-following accuracy.**\
✅ **Preference fine-tuning and LoRA are recommended for future optimization.**

***

This chapter **fine-tuned GPT-2 for instruction-following**. The next step explores **preference fine-tuning** for **better user alignment**.



***

### **Chapter 8: Preference Fine-Tuning and RLHF**

#### **8.1 Introduction to Preference Fine-Tuning**

* **Preference fine-tuning** is used after instruction fine-tuning to improve user alignment.
* Helps LLMs **generate responses that match human expectations**.
* Unlike instruction fine-tuning (which just follows commands), preference tuning **ensures outputs are more useful and engaging**.
* **Used in models like ChatGPT (GPT-4) and Claude AI.**

***

#### **8.2 What is Reinforcement Learning from Human Feedback (RLHF)?**

* **RLHF (Reinforcement Learning from Human Feedback)** is a technique for aligning LLMs with human values.
* **Steps in RLHF:**
  1. **Train a reward model (RM)** to score responses.
  2. **Use Proximal Policy Optimization (PPO)** to optimize the LLM based on RM scores.
  3. **Repeat the process until convergence**.

**Example RLHF Process:**

1. LLM generates two responses.
2. Humans rate which response is better.
3. A reward model (RM) learns from these human ratings.
4. The LLM is fine-tuned to **generate responses that maximize RM scores**.

***

#### **8.3 Collecting Preference Data**

* **Dataset requirements**:
  * Contains **prompt-response pairs** with a preference score.
  * Example datasets: **OpenAI's GPT-4 Preference Data**, **Anthropic’s HH-RLHF**.
*   **Example Dataset Format**:

    ```json
    [
      {
        "prompt": "Explain quantum computing to a 5-year-old.",
        "response_1": "It's like a magic coin that can be heads and tails at the same time!",
        "response_2": "Quantum computing is a complex topic involving superposition and entanglement.",
        "preferred": 1
      }
    ]
    ```
* The model learns that **response\_1** is preferred over **response\_2**.

***

#### **8.4 Training a Reward Model (RM)**

* **Reward models** assign **scores** to LLM-generated responses.
* Typically based on **transformer models like BERT or GPT**.
* **Training the RM**:
  * Inputs: **Prompt + Response**
  * Output: **A single reward score**
*   **Implementation of a Simple Reward Model in PyTorch**:

    ```python
    import torch.nn as nn

    class RewardModel(nn.Module):
        def __init__(self, emb_dim):
            super().__init__()
            self.fc1 = nn.Linear(emb_dim, 512)
            self.fc2 = nn.Linear(512, 1)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            return self.fc2(x)
    ```
* The model outputs a **score** for each response.

***

#### **8.5 Fine-Tuning an LLM with RLHF**

* **Uses Proximal Policy Optimization (PPO)**, a reinforcement learning algorithm.
* The LLM generates multiple responses → **RM assigns scores** → LLM **optimizes based on rewards**.
* **PPO Loss Function**: LPPO=min⁡(πθ(a∣s)πθold(a∣s)A,clip(πθ(a∣s)πθold(a∣s),1−ϵ,1+ϵ)A)L\_{\text{PPO\}} = \min \left( \frac{\pi\_{\theta}(a | s)}{\pi\_{\theta\_{\text{old\}}}(a | s)} A, \text{clip}\left(\frac{\pi\_{\theta}(a | s)}{\pi\_{\theta\_{\text{old\}}}(a | s)}, 1 - \epsilon, 1 + \epsilon\right) A \right)
* **Steps in RLHF Fine-Tuning**:
  1. **Train a reward model**.
  2. **Use PPO to fine-tune the LLM**.
  3. **Validate on human feedback**.

***

#### **8.6 LoRA: A More Efficient Alternative to RLHF**

* **LoRA (Low-Rank Adaptation)** fine-tunes models **efficiently without retraining the entire LLM**.
* **Advantages**:
  * Requires **less GPU memory**.
  * **Faster training** compared to RLHF.
  * Works well on **small-scale preference fine-tuning** tasks.
*   **Example LoRA Implementation**:

    ```python
    from peft import get_peft_model, LoraConfig

    config = LoraConfig(r=8, lora_alpha=32, lora_dropout=0.1)
    model = get_peft_model(pretrained_gpt, config)
    ```
* **When to use LoRA?**
  * If **limited compute resources** are available.
  * When **only small modifications to LLM behavior are needed**.

***

#### **8.7 Evaluating Preference Fine-Tuned Models**

* **Metrics for Evaluation**:
  * **Reward Model Score**: Measures response quality.
  * **Human Evaluation**: Users rate generated responses.
  * **GPT-4 Evaluation**: Compare results against GPT-4.
*   **Example Evaluation Process**:

    ```python
    def evaluate_model(text, model, reward_model):
        token_ids = tokenizer.encode(text, return_tensors="pt")
        output_ids = model.generate(token_ids, max_length=100)
        response = tokenizer.decode(output_ids[0])
        score = reward_model(response)
        return response, score.item()
    ```
* **Best practice**: Combine **automated metrics with human evaluation**.

***

#### **8.8 Real-World Use Cases of RLHF**

* **OpenAI**: Uses RLHF in **ChatGPT**.
* **Anthropic Claude**: Developed using **Constitutional AI** (a variation of RLHF).
* **Google Gemini**: Uses **Preference Optimization**.
* **Meta AI (LLaMA 3)**: Trained on **instruction and preference tuning**.

***

#### **8.9 Summary**

✅ **Preference fine-tuning aligns LLMs with human values**.\
✅ **RLHF uses reinforcement learning to improve response quality**.\
✅ **Reward models assign preference scores to LLM responses**.\
✅ **PPO and LoRA are popular methods for optimizing models**.\
✅ **Real-world applications include ChatGPT, Claude, and Gemini**.

***

This chapter **covered RLHF and preference tuning**. The next chapter discusses **evaluating LLMs and mitigating bias**.



Here are detailed notes from **Appendix A: Introduction to PyTorch** of _Build a Large Language Model (From Scratch)_ by Sebastian Raschka.

***

### **Appendix A: Introduction to PyTorch**

This appendix introduces PyTorch, covering fundamental concepts required to implement **large language models (LLMs)** from scratch.

#### **A.1 What is PyTorch?**

* **PyTorch** is an **open-source deep learning framework** developed by **Meta AI**.
* It has been **the most widely used deep learning library for research since 2019** (based on **Papers With Code**).
* **Kaggle’s 2022 Data Science Survey** reported **40% of users preferred PyTorch** over TensorFlow.

**Why PyTorch?**

* **User-friendly**: Simple API for fast prototyping.
* **Flexible**: Can modify models dynamically.
* **Efficient**: Supports **CUDA for GPU acceleration**.

#### **A.1.1 Three Core Components of PyTorch**

1. **Tensor Library**:
   * Similar to NumPy but optimized for **GPU acceleration**.
   * **Supports dynamic computation graphs**.
2. **Autograd (Automatic Differentiation Engine)**:
   * Computes gradients for **backpropagation**.
3. **Deep Learning Library**:
   * Provides **pretrained models, optimizers, loss functions**.

***

#### **A.2 Understanding Tensors**

* **Tensors** are the core data structure in PyTorch.
* **Similar to NumPy arrays** but optimized for deep learning.

**A.2.1 Types of Tensors:**

| Tensor Type | Example                              |
| ----------- | ------------------------------------ |
| Scalar      | `x = torch.tensor(5)`                |
| Vector      | `x = torch.tensor([1, 2, 3])`        |
| Matrix      | `x = torch.tensor([[1, 2], [3, 4]])` |
| High-Dim    | `x = torch.rand(3, 3, 3)`            |

**A.2.2 Tensor Data Types**:

* `torch.float32` (default for deep learning)
* `torch.int64` (for indexing)
* `torch.bool` (boolean operations)

**A.2.3 Common PyTorch Tensor Operations**:

```python
x = torch.tensor([[1, 2], [3, 4]])
y = torch.tensor([[5, 6], [7, 8]])

# Addition
z = x + y  

# Matrix Multiplication
z = torch.matmul(x, y)

# Reshaping
z = x.view(4, 1)
```

***

#### **A.3 Computation Graphs**

* PyTorch uses **dynamic computation graphs**.
* Each operation **automatically tracks gradients** for backpropagation.

**Example: Creating a Computation Graph**

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2  # y = x^2
y.backward()  # Computes dy/dx
print(x.grad)  # Output: tensor(4.0)
```

**Key concept**:

* `requires_grad=True` enables gradient tracking.
* `.backward()` computes gradients automatically.

***

#### **A.4 Automatic Differentiation in PyTorch**

* PyTorch uses **autograd** to compute gradients efficiently.

**Example: Backpropagation**

```python
import torch

x = torch.tensor(3.0, requires_grad=True)
y = x ** 3 + 2 * x
y.backward()
print(x.grad)  # Output: 3x^2 + 2 => 29
```

* `.backward()` computes **derivatives** automatically.

***

#### **A.5 Implementing a Multilayer Neural Network**

* PyTorch simplifies **building deep learning models**.

**Example: A Simple Neural Network**

```python
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        return self.layer2(self.relu(self.layer1(x)))

model = NeuralNetwork(2, 4, 1)
print(model)
```

* **Uses `torch.nn` for defining layers.**
* **`forward()` defines the computation.**

***

#### **A.6 Setting Up Efficient Data Loaders**

* **PyTorch DataLoaders** streamline **batch processing**.

**Example: Loading Data in Mini-Batches**

```python
from torch.utils.data import DataLoader, TensorDataset

# Example dataset
data = torch.rand(100, 10)
labels = torch.randint(0, 2, (100,))

dataset = TensorDataset(data, labels)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

for batch in dataloader:
    x_batch, y_batch = batch
    print(x_batch.shape)  # Output: torch.Size([8, 10])
```

* **Efficiently loads data in batches**.
* **Supports shuffling and parallel processing**.

***

#### **A.7 A Typical Training Loop**

* PyTorch **training loops** follow a structured format:
  1. **Forward pass**: Compute predictions.
  2. **Compute loss**: Compare with ground truth.
  3. **Backward pass**: Compute gradients.
  4. **Update weights**: Adjust parameters using an optimizer.

**Example: Training a Model**

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

for epoch in range(5):
    for x_batch, y_batch in dataloader:
        optimizer.zero_grad()
        predictions = model(x_batch)
        loss = criterion(predictions.squeeze(), y_batch.float())
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")
```

* Uses **Adam optimizer** and **binary cross-entropy loss**.

***

#### **A.8 Saving and Loading Models**

* **PyTorch allows saving models for reuse**.

**Saving a Model**:

```python
torch.save(model.state_dict(), "model.pth")
```

**Loading a Model**:

```python
model.load_state_dict(torch.load("model.pth"))
model.eval()  # Switch to inference mode
```

***

#### **A.9 Optimizing Training Performance with GPUs**

* **PyTorch supports GPU acceleration** using CUDA.

**Checking for GPU Availability**:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
```

**A.9.1 Training on a Single GPU**

```python
x = torch.rand(100, 10).to(device)
output = model(x)
```

**A.9.2 Training on Multiple GPUs**

```python
from torch.nn.parallel import DataParallel

model = DataParallel(model)
model.to(device)
```

* Uses **Distributed Data Parallel (DDP)** for multi-GPU training.

***

#### **Summary**

✅ **PyTorch is a flexible and GPU-accelerated deep learning framework.**\
✅ **Tensors are the core data structure, supporting GPU operations.**\
✅ **Autograd simplifies backpropagation with automatic differentiation.**\
✅ **DataLoaders enable efficient batch processing.**\
✅ **Training follows a structured loop with forward, backward, and optimization steps.**\
✅ **PyTorch supports model saving/loading and multi-GPU training.**

***

This appendix **provides the foundational PyTorch knowledge** needed for **building large language models (LLMs)**.

Would you like notes on **Appendix D (Advanced Training Techniques like Learning Rate Warmup, Cosine Decay, and Gradient Clipping)?** 🚀

