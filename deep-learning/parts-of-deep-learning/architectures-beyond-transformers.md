# Architectures Beyond Transformers

Beyond standard transformers, **state space models (SSM)** and **Mamba** offer efficient long-sequence modeling; research continues on **linear attention** and other alternatives.

---

## State space models (SSM)

**Continuous-time view:** \(h'(t) = A h(t) + B x(t)\), \(y(t) = C h(t)\). Discretize (e.g. zero-order hold) for discrete sequences: \(h_k = \bar{A} h_{k-1} + \bar{B} x_k\), \(y_k = C h_k\). **Recurrent:** O(1) per step for inference; **training:** can be parallelized via associative scan. **S4 (Structured State Space):** Specific parameterization of \(A\) (e.g. HiPPO initialization) for long-range dependency; strong on long-context benchmarks.

---

## Mamba

**Mamba:** SSM with **input-dependent** \(B, C\) (and sometimes \(A\)); selective copying of information into state. **Selective SSM:** Gating so the model can choose what to remember; improves performance over fixed S4. **Efficiency:** Linear in sequence length; fast in practice; competitive with transformers for language and long sequences. **Architecture:** Often stacked Mamba blocks (like transformer blocks) with residual connections; sometimes mixed with attention layers.

---

## Alternative sequence models

- **Linear attention:** Replace softmax attention with kernel feature maps so that \(O(N^2)\) becomes \(O(N)\); tradeoff in expressiveness. **RWKV:** Recurrent formulation with linear attention-like structure; efficient for long context.
- **Retentive networks (RetNet):** Parallel training, recurrent inference; proposed as transformer alternative. **Research:** Active area; transformers still dominant in production, but SSM/Mamba and variants are used in long-context and efficiency-sensitive settings.

---

## Quick revision

- **SSM:** Recurrent state update; S4 with structured \(A\) for long range. **Mamba:** Selective SSM; input-dependent; linear complexity; competitive with transformers. **Alternatives:** Linear attention, RWKV, RetNet; focus on efficiency and long context.
