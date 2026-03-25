# LLM Fundamentals

* What are foundation models, and how have they changed AI engineering?
* What is a Large Language Model (LLM), and how does it work?
* What are Transformer Models and how do they work?
* What are the key components of a Transformer model?
* What is tokenisation in LLMs?
* Explain BPE (Byte Pair Encoding).
* Explain WordPiece and SentencePiece.
* What is positional encoding, and why is it needed in Transformers?
* What is causal masking?
* What is self-attention, and how does it work in Transformers?
* Explain the Query(Q), Key(K), and Value(V) in attention.
* What are multi-head attention mechanisms? Why use multiple attention heads?
* What is the context window in LLMs, and why does it matter?
* What is temperature in the context of LLMs, and how does it affect output?
* Explain Top-p (nucleus) sampling and Top-k sampling. How do they differ?
* What are logits, and how are they used in text generation?
* What are skip connections (residual connections) in Transformers?
* What is the difference between open-source and closed-source LLMs? When would you choose one over the other?
* What is the difference between encoder-only, decoder-only, and encoder-decoder Transformer architectures?
* What is KV cache, and how does it speed up inference?
* Explain the difference between autoregressive and masked language modeling.
* What is model distillation, and how is it used with LLMs?
* What is Mixture of Experts (MoE), and how does it work in models like Mixtral?
* What is the difference between dense and sparse models?
* What is Flash Attention?
* What is Grouped-Query Attention (GQA), and how does it differ from Multi-Head Attention (MHA)?
* How does Rotary Position Embedding (RoPE) work, and why is it preferred over learned positional embeddings?
* Your LLM keeps ignoring your instructions. How do you make it follow structured output formats?
* Your LLM-powered tool hits the context window limit on long documents. How do you handle it?
* Your LLM does not admit when it does not know the answer. How do you make it say "I don't know"?
* Your LLM generates responses that are too verbose. How do you control response length?
* Your LLM memorised proprietary training data and leaks it in responses. How do you prevent this?
* Your LLM coding assistant generates outdated code using deprecated libraries. How do you fix it?
* Your tokenizer splits important domain terms into meaningless subword pieces. How do you fix it?
* Your Transformer's KV cache grows too large during long sequence generation. How do you manage memory?
* Your Transformer runs out of memory on long documents due to quadratic self-attention. How do you scale it?
* Your distilled student model fails on the complex reasoning that the teacher model handled. How do you close the gap?
* After RLHF alignment, your LLM became safer but lost capability on hard tasks. How do you manage the alignment tax?
* Your RLHF-trained LLM is gaming the reward model instead of being genuinely helpful. How do you fix reward hacking?
* Your chatbot loses context after 10 turns in a conversation. How do you maintain a long conversation context?
* Your chatbot fails when users switch topics mid-conversation. How do you handle topic switches?
* Your QA system always generates an answer even when no answer exists in the context. How do you detect unanswerable questions?
* Your summarization system hallucinated facts not in the original article. How do you fix it?
* Your text generation repeats phrases in long outputs. How do you fix repetition?
