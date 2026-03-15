# Pretraining, Fine-tuning, and RLHF

Modern LLMs are built in stages: **pretraining** (learn language and world knowledge), **fine-tuning** (adapt to tasks or instructions), and **alignment** (RLHF/DPO so outputs match human preferences).

---

## Pretraining

- **Goal**: Learn general-purpose representations and next-token (or masked) prediction from large text (and optionally multimodal) corpora.
- **Objective**: For decoder-only models, maximize likelihood of next token given previous tokens (autoregressive). For encoder-only, often masked language modeling (predict masked tokens).
- **Data**: Web text, books, code, etc. Often heavily filtered and deduplicated.
- **Result**: A base model that can complete text and, with prompting, perform many tasks but is not yet instruction-following or “helpful.”

---

## Fine-tuning

- **Supervised fine-tuning (SFT)**: Train on (input, target output) pairs for specific tasks (e.g. summarization, QA) or on (instruction, response) pairs for general instruction following.
- **Adapter / LoRA**: Train small adapter weights or low-rank updates instead of full parameters; keeps base model frozen and reduces cost.
- **Instruction tuning**: Fine-tune on diverse (instruction, response) datasets so the model learns to follow user intents; bridges pretraining and helpful chat.

---

## RLHF (Reinforcement Learning from Human Feedback)

- **Steps**:
  1. **Collect preferences**: Humans rank or choose between model outputs for the same prompt.
  2. **Train reward model (RM)**: Supervised model that predicts which output humans prefer (or score per output).
  3. **Optimize policy**: Use RL (e.g. PPO) to update the LLM so it maximizes the reward from the RM, often with a KL penalty toward the SFT model to avoid drift.

- **Purpose**: Align model behavior with human preferences (helpfulness, harmlessness, honesty) without hand-coding rules.

---

## DPO (Direct Preference Optimization)

- **Idea**: Optimize the policy directly on preference data (A preferred over B) using a closed-form objective derived from the RLHF setup, without training a separate reward model or running PPO.
- **Benefits**: Simpler pipeline, often more stable; widely used for alignment after SFT.

---

## Typical pipeline

```
Pretraining (next-token) → SFT / Instruction tuning → RLHF or DPO → Deployed model
```

---

## Quick revision

- **Pretraining**: learn from large corpora with next-token or masked objective; produces base model.
- **Fine-tuning / instruction tuning**: adapt to tasks or instructions with (input, output) or (instruction, response) data.
- **RLHF**: reward model from human preferences + RL (e.g. PPO) to maximize reward; **DPO** does alignment from preferences without separate RM/RL.
- These stages are why modern LLMs are both capable (pretraining) and aligned and instruction-following (SFT + RLHF/DPO).
