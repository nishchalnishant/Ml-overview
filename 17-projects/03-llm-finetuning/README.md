---
module: Projects
topic: LLM Fine-Tuning
subtopic: ""
status: unread
tags: [projects, llm, fine-tuning, lora, peft, hands-on]
---
# Project: LoRA Fine-Tuning of a Small LLM

**What this is:** a complete, runnable parameter-efficient fine-tuning pipeline — load a small open-weight base model, apply LoRA adapters, fine-tune on an instruction-style dataset, evaluate before/after, and merge or save the adapter. Sized to run on a single consumer GPU or CPU (slowly) by defaulting to a small model (`sshleifer/tiny-gpt2` for a smoke-test-fast default, easily swapped for `Qwen2.5-0.5B-Instruct` or similar for a real result).

This is the applied counterpart to fine-tuning coverage in [05-llms/](../../05-llms/) (LoRA/QLoRA, PEFT, instruction tuning).

## Why this project

Fine-tuning is a repeatedly-referenced study-plan milestone with no runnable code anywhere in the repo. This builds the full loop — dataset formatting, LoRA config, training, before/after comparison — with a deliberately small default model so the whole thing runs as a smoke test in minutes, while remaining a real fine-tune when pointed at a larger base model and given a GPU.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python prepare_data.py                 # writes data/instructions.jsonl (small synthetic instruction-tuning set)
python train_lora.py                   # loads base model, applies LoRA, trains, saves adapter to ./adapter
python compare.py                      # generates from base model vs. fine-tuned model on held-out prompts, prints side by side
```

To use a larger, more capable base model instead of the tiny smoke-test default:

```bash
python train_lora.py --model Qwen/Qwen2.5-0.5B-Instruct --epochs 3
```

## Structure

| File | Purpose |
|---|---|
| `prepare_data.py` | Generates a small synthetic instruction/response dataset (JSONL), no external download needed. |
| `train_lora.py` | Loads base model + tokenizer, wraps with a LoRA adapter via `peft`, trains with `transformers.Trainer`, saves the adapter. |
| `compare.py` | Loads the base model and the LoRA-adapted model, generates on the same held-out prompts, prints both outputs for inspection. |
| `requirements.txt` | Dependencies (`transformers`, `peft`, `datasets`, `accelerate`, `torch`). |

## Design notes

- **LoRA, not full fine-tuning**: only low-rank adapter matrices are trained (`r=8` default), so the base model's weights stay frozen — see [05-llms/](../../05-llms/) for why this drastically cuts trainable-parameter count and memory footprint versus full fine-tuning.
- **Default model is intentionally tiny** (`sshleifer/tiny-gpt2`, ~2M params) so `train_lora.py` completes as a correctness smoke test on CPU in under a minute. The `--model` flag swaps in a real instruction-following base model for an actually useful fine-tune, at the cost of needing a GPU and more time.
- **Adapter-only save**: `train_lora.py` saves just the LoRA adapter weights (a few MB), not a full model copy — the standard PEFT deployment pattern, since the adapter is loaded on top of the (unchanged) base model at inference time.
- **Before/after comparison is the actual deliverable**: `compare.py` exists specifically to make the effect of fine-tuning visible, not just to prove training ran.

## Where to Next

- **LoRA/QLoRA/PEFT theory** → [05-llms/](../../05-llms/)
- **RAG as an alternative to fine-tuning for knowledge injection** → [../02-rag-pipeline/](../02-rag-pipeline/)
- **Distributed/multi-GPU training at scale** → [06-production-ml/](../../06-production-ml/)
