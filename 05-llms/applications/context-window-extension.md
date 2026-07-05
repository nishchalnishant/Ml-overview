---
module: Llms
topic: Context Window Extension
subtopic: Applications
status: unread
tags: [llms, context-window, long-context, rope, alibi, sliding-window]
---
# Context Window Extension

> **📄 Full coverage is in the parent LLMs folder:**
> **[`05-llms/07-context-window-extension.md`](../07-context-window-extension.md)**

This page is a navigation redirect. The comprehensive deep dive lives one level up.

---

## Quick Summary (30-second version)

**The problem:** Transformers trained on short sequences fail catastrophically on longer ones. A model trained on 4K tokens shows degraded performance beyond that — because it has never seen certain positional encodings before.

**The core techniques:**

| Technique | Approach | Best for |
|---|---|---|
| **RoPE + NTK scaling** | Adjust rotation frequency at inference | LLaMA, Mistral family |
| **YaRN** | Non-uniform frequency interpolation | Most cost-efficient |
| **ALiBi** | Linear attention bias, no positional embedding | Bloom, MPT |
| **Sliding Window Attention** | Local + global attention (Longformer, Mistral) | Very long sequences |
| **Ring Attention** | Distributed sequence processing across GPUs | 1M+ token sequences |
| **Flash Attention** | Memory-efficient exact attention | All modern LLMs |

**Practical rule:** For 4× context extension, NTK-aware interpolation with fine-tuning is the most reliable. For 10×+, use YaRN or train from scratch with the target length.

---

→ For full derivations, interview Q&A, and production considerations: [context-window-extension.md](../07-context-window-extension.md)
