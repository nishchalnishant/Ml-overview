# Long-Context Models

Long-context models handle sequences that exceed the typical 2K–8K token limit of early transformers. Approaches include **sparse attention**, **linear attention**, **state space models**, and **efficient attention** implementations.

---

## Sparse attention

**Longformer:** Sliding window (local) attention + global attention on selected tokens (e.g. [CLS]); linear in sequence length. **BigBird:** Combines random attention, window attention, and global tokens; theoretically approximates full attention for certain tasks. **Extended context:** 4K–32K+ tokens for documents.

---

## Flash Attention

**Flash Attention:** Recomputation and tiling to reduce GPU memory reads/writes; avoids materializing full \(N \times N\) attention matrix. **Result:** Faster and more memory-efficient attention; enables longer contexts and larger batches. **Flash Attention 2:** Further optimizations; standard in vLLM, TGI, and modern training stacks.

---

## State space models (SSM)

**Idea:** Model sequence with a continuous-time state space (e.g. \(h' = Ah + Bx\), \(y = Ch\)); discretize for computation. **Linear** in sequence length for recurrence; can be parallelized in training (scan). **S4, Mamba:** Structured state spaces for long-range dependency; competitive with transformers on long benchmarks. **Hybrid:** SSM layers alongside attention in some architectures.

---

## Position encodings for length

- **RoPE, ALiBi:** Allow extrapolation beyond training length (see [Transformers](../deep-learning/parts-of-deep-learning/transformers.md)). **NTK-aware, YaRN:** Scaling of RoPE for very long contexts (e.g. 128K) without full retraining.

---

## Quick revision

- **Longformer / BigBird:** Sparse + global attention; linear in length. **Flash Attention:** Memory-efficient implementation; enables longer contexts. **SSM (S4, Mamba):** Recurrent/scan; linear in length. **RoPE/ALiBi + scaling:** Better length extrapolation.
