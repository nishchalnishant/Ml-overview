---
module: Architectures
topic: Revision Guide
subtopic: "06 Architectures"
status: unread
tags: [revision, cheatsheet, rnn, lstm, gru, attention, transformers, cnn, gan, vae, diffusion, gnn]
---
# 06 Architectures — Revision Guide

Fast revision for the whole `06-architectures/` folder: an outline of every major concept, then graded Q&A (no answers — self-test). Use the outline to check you can explain each node aloud before drilling questions.

---

```markdown
- **06 Architectures**
  - **RNN / LSTM / GRU**
    - Vanilla RNN<br/>shared hidden state
      - Bidirectional RNN<br/>encoder-only, non-autoregressive
    - BPTT / Truncated BPTT
    - Vanishing gradients<br/>eigenvalues of W_h
      - Exploding gradients<br/>grad clipping
      - Orthogonal init
      - LSTM: 4 gates<br/>forget/input/candidate/output
    - Cell state = gradient highway<br/>Constant Error Carousel
    - Forget bias init = 1
    - GRU: reset + update gates
    - LSTM vs GRU tradeoffs
    - Stacked/Deep RNN
      - Sequence packing
    - Recurrent dropout
    - RNN vs Transformer tradeoffs
    - SSM / Mamba<br/>selective state, dual conv+recurrence
  - **Attention**
    - Fixed-context bottleneck<br/>encoder-decoder RNN limit
    - Query/Key/Value roles
    - Scaled dot-product<br/>softmax(QK^T/sqrt(d_k))V
      - Self-attention vs<br/>Cross-attention
      - Causal masking
      - Multi-head attention
      - Positional encodings
        - Sinusoidal PE
        - Learned PE
        - RoPE
        - ALiBi
      - Flash Attention<br/>IO-aware, online softmax
      - Complexity: standard vs<br/>sparse vs linear attention
    - Why sqrt(d_k) scaling
    - KV cache
      - GQA
    - MQA
  - **Transformers**
    - Sequential bottleneck<br/>motivation
    - Transformer block<br/>attn + FFN + residual + LN
      - Encoder-only (BERT)
      - Decoder-only (GPT)
      - Encoder-Decoder (T5)
      - Scaling laws<br/>Chinchilla D≈20N
    - Pre-norm vs Post-norm
    - FFN / MLP sublayer
      - Mixture-of-Experts FFN<br/>router, top-k, load balancing
    - SwiGLU activation
    - Architecture milestones<br/>GPT-2→LLaMA3→MoE
    - RMSNorm
  - **Other DL Architectures**
    - CNN motivation<br/>locality + translation invariance
      - GAN<br/>minimax, mode collapse
      - Diffusion models
    - Convolution op<br/>weight sharing
      - Equivariance vs Invariance
    - Conv building blocks<br/>BatchNorm/ReLU/Pool
    - AlexNet/VGG/ResNet/EfficientNet
    - ResNet skip connections<br/>y=F(x)+x
    - Autoencoder / VAE
    - Reparameterization trick
    - Posterior collapse
    - Architecture selection<br/>framework by inductive bias
    - CNN vs Transformer vs GNN<br/>vs MLP/GBT for tabular
  - **Graph Neural Networks**
    - Permutation invariance<br/>requirement
      - Message passing template<br/>message/aggregate/update
        - GCN<br/>symmetric normalization
        - GraphSAGE<br/>inductive, neighbor sampling
          - Scaling: neighbor sampling,<br/>Cluster-GCN
        - GAT<br/>attention over edges
    - GIN<br/>sum aggregation, WL test
    - Aggregator expressiveness<br/>sum > max > mean
    - Over-smoothing
    - Self-loops / residual update
```

---

## Questions

### Easy

1. What problem does the hidden state solve in a vanilla RNN?
2. Write the vanilla RNN recurrence equation. What does each term represent?
3. What is BPTT? What is truncated BPTT?
4. What is the difference between vanishing and exploding gradients, in terms of $W_h$'s eigenvalues?
5. What does gradient clipping do, and why clip by norm rather than by value?
6. Name the four gates in an LSTM and what each one does.
7. What are the two gates in a GRU?
8. Why is the LSTM forget gate bias typically initialized to 1?
9. What is a bidirectional RNN, and why can't it be used as an autoregressive decoder?
10. What are Query, Key, and Value in attention, in one sentence each?
11. Write the scaled dot-product attention formula.
12. What is causal masking and why is it needed for language modeling?
13. What is multi-head attention, at a high level?
14. Why do transformers need positional encodings at all?
15. What is a KV cache, and what problem does it solve?
16. What is the difference between self-attention and cross-attention?
17. What are MQA and GQA, one line each?
18. What does a residual connection do structurally in a transformer block?
19. What is the difference between an encoder-only, decoder-only, and encoder-decoder transformer? Give one example model of each.
20. What is the purpose of the feed-forward (FFN) sublayer in a transformer block?
21. What problem do convolutions solve that fully-connected layers don't, for images?
22. What is weight sharing in a CNN?
23. What does a ResNet skip connection compute, in one equation?
24. What is the reparameterization trick used for in a VAE?
25. In one sentence, what is "message passing" in a GNN?

### Medium

1. Derive why gradients vanish or explode in a vanilla RNN — what is the mathematical origin of the effect?
2. Explain the "constant error carousel": why does the LSTM cell state gradient not vanish, mechanically?
3. Why does GRU use 3 weight matrices vs LSTM's 4, and what performance/speed tradeoff results?
4. When would you prefer LSTM over GRU, and vice versa?
5. Why can truncated BPTT bias gradient estimates, and why is this usually acceptable in practice?
6. Derive why attention scores are divided by $\sqrt{d_k}$ — what breaks without it?
7. Compare MHA, MQA, and GQA in terms of KV cache memory and quality tradeoff. Why did LLaMA 3 pick GQA?
8. Explain why RoPE generalizes to longer sequences better than absolute sinusoidal or learned positional embeddings.
9. What is the difference between ALiBi and RoPE in how they encode position, and when would you prefer one over the other?
10. Why is Flash Attention faster than standard attention despite doing the same (or more) FLOPs?
11. Why is pre-norm preferred over post-norm in modern large transformers? What specifically goes wrong with post-norm at scale?
12. Explain why SwiGLU needs 3 weight matrices instead of 2, and why the FFN hidden dimension is scaled to 8/3·d_model to compensate.
13. What problem does Mixture-of-Experts solve, and what causes naive top-k routing to collapse?
14. Compare RNN and Transformer along parallelism, memory at inference, and training compute. When would you still choose an RNN today?
15. Explain how Mamba/SSMs achieve both parallel training and constant-memory inference — why can't a vanilla RNN do the same?
16. Explain the difference between translation equivariance and translation invariance in CNNs, and which one you want for detection vs classification.
17. Derive why ResNet skip connections mitigate the degradation problem — write the gradient expression through a residual block.
18. In a VAE, what is posterior collapse, and why is it the opposite failure mode from a plain (non-variational) autoencoder's degenerate latent space?
19. Derive why GCN needs symmetric normalization $\tilde D^{-1/2}\tilde A\tilde D^{-1/2}$ instead of using the raw adjacency matrix.
20. Why is GraphSAGE described as "inductive" while a standard GCN is "transductive"? What operational problem does this solve?
21. Explain why sum aggregation is strictly more expressive than mean or max aggregation in GNNs, and connect this to the Weisfeiler-Lehman test.
22. Why does stacking many GNN layers cause over-smoothing, and how is this different from vanishing gradients in RNNs?
23. Given a task needing long-range dependency resolution over 100K-token sequences with tight latency, would you choose a Transformer, RNN, or SSM? Justify with complexity tradeoffs.
24. Why does a Transformer resolve long-distance dependencies in one layer where an RNN needs many timesteps? Frame this in terms of path length.
25. What structural property of tabular data makes Transformers a poor default choice compared to gradient-boosted trees?

### Hard

1. A 500-token sequence is fed into a vanilla RNN. Derive the form of $\partial h_{500}/\partial h_1$ and explain precisely under what condition on $W_h$ the gradient vanishes vs explodes vs stays stable.
2. You initialize an LSTM's forget gate bias to 1.0 but observe the model still fails to learn dependencies beyond ~15 steps. What are three plausible causes beyond the forget-gate initialization itself?
3. Explain why orthogonal initialization of $W_h$ only helps "at initialization" and not throughout training. What actually drifts, and why doesn't it prevent vanishing gradients from re-emerging?
4. Compare the gradient path length for a dependency between token 1 and token $n$ in: vanilla RNN, LSTM, Transformer, and a depth-$k$ GNN. What determines the path length in each?
5. Derive the KV cache memory formula and compute the memory (in GB, fp16) for a 70B model with 64 heads, $d_k=128$, 80 layers, at 32K context. Then recompute with GQA using 8 KV heads. What does this tell you about why GQA became standard?
6. Explain precisely why the attention score variance scales with $d_k$ for random Q, K vectors, and derive why this pushes softmax toward one-hot outputs at high dimension without scaling.
7. Flash Attention reduces memory from $O(n^2)$ to $O(n)$ but does NOT reduce time complexity below $O(n^2 d)$. Explain the distinction between "IO-efficient" and "compute-efficient," and why this distinction matters for extremely long contexts (e.g. 1M tokens).
8. A model trained with RoPE at 4K context is deployed at 32K context without fine-tuning. What breaks, and how do NTK scaling and YaRN differ in their fix?
9. Explain why MoE routing requires an auxiliary load-balancing loss — what specific rich-get-richer dynamic occurs without it, and how does expert capacity (drop/reroute) interact with this at inference time?
10. Construct an argument for why decoder-only architectures (GPT-style) became dominant over encoder-decoder (T5-style) for general-purpose LLMs, addressing both training efficiency and task generality.
11. A Mamba/SSM model and a Transformer are both trained on the same data and compute budget. Under what sequence-length regime would you expect the SSM to win, and under what regime would the Transformer still win despite quadratic cost? Justify using complexity terms.
12. Derive the gradient of a residual block $y = F(x) + x$ with respect to $x$, and explain precisely why this prevents vanishing gradients even when $\partial F/\partial x \to 0$, contrasting this with the LSTM cell-state gradient mechanism — are these the same trick in different clothing?
13. Explain posterior collapse in a VAE as a specific pathology of the ELBO objective: what property of the decoder or the KL weighting causes $q_\phi(z|x)$ to become independent of $x$, and how does $\beta$-VAE's weighting interact with this failure mode?
14. Prove (informally) why no message-passing GNN using sum/mean/max aggregation can exceed 1-WL expressiveness. Give a pair of non-isomorphic graphs that a mean-aggregation GNN would fail to distinguish.
15. You have a 50M-node graph and full-batch GCN training doesn't fit in memory. Compare neighbor sampling (GraphSAGE) vs Cluster-GCN partitioning as solutions — what specific failure mode (neighbor explosion vs partition bias) does each risk?
16. Over-smoothing in GNNs and mode collapse in GANs are both described as "representational collapse" but arise from different mechanisms. Explain the mechanism behind each and why the fixes (residual/skip aggregation vs. minibatch discrimination/WGAN-GP) don't transfer between the two problem classes.
17. A production LLM needs both parallel training and streaming (token-by-token, constant-memory) inference. Compare how Transformers-with-KV-cache, RNNs, and SSMs each satisfy or fail to satisfy this dual requirement, and identify exactly which mathematical property (recurrence vs convolution duality) makes SSMs able to do both.
18. Why does pre-norm transformer training remain stable at extreme depth (100+ layers) while post-norm becomes untrainable, in terms of the gradient signal through the residual stream versus through the normalized branch? What is the remaining failure mode pre-norm doesn't fix, and how is it patched?
19. Design an ablation to determine whether a given long-context failure in a RoPE-based transformer is caused by (a) attention dilution/softmax peakiness at long context, (b) positional out-of-distribution generalization, or (c) insufficient training data at that length. What would you measure in each case?
20. Given the "architecture selection framework" (locality → CNN, arbitrary pairwise relations → Transformer, topology → GNN, no structure → MLP/GBT), construct a case where the data has graph structure but a GNN is still the wrong choice, and justify with a cost/benefit argument grounded in the folder's own reasoning.
