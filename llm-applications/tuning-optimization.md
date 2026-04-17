# Tuning and Optimization

This file is about a very practical question:

How do you make an LLM behave better without burning GPU budget, destroying quality, or solving the wrong problem?

That question comes up a lot in interviews because good teams do not fine-tune just because they can.

They fine-tune when it is the right lever.

## The First Decision

Before you touch training, ask:

- is this a **knowledge** problem?
- a **behavior** problem?
- a **latency/cost** problem?
- a **formatting consistency** problem?

That diagnosis matters more than the buzzword you pick next.

## Prompting vs RAG vs Fine-Tuning

This is the clean comparison to remember.

- **prompting**: fastest and cheapest way to steer behavior
- **RAG**: best when knowledge needs to stay fresh
- **fine-tuning**: best when behavior, tone, or output structure needs to change consistently

If the model keeps missing new company facts, reach for RAG first.
If it knows the facts but answers in the wrong style, fine-tuning becomes more interesting.

## Supervised Fine-Tuning (SFT)

SFT means training the model on prompt-response examples so it learns the behavior you want more directly.

Use it when you need:

- stronger instruction following
- better output formatting
- domain-specific response style
- more consistent task behavior

It is the classic adaptation path.

## Classical Music Analogy

SFT is like remastering a beloved Lata or Kishore track for modern speakers.

You are not changing the soul of the song.
You are making the delivery cleaner, clearer, and better aligned to the listening environment.

That is what good fine-tuning should feel like.

## PEFT

PEFT stands for **Parameter-Efficient Fine-Tuning**.

Instead of updating the whole model, you update a much smaller subset or add lightweight adapter structures.

Why teams love it:

- cheaper
- faster
- less memory-hungry
- easier to experiment with

This is where adaptation becomes operationally realistic.

## LoRA

LoRA is a PEFT method that learns low-rank updates instead of rewriting the full weight matrices.

The practical takeaway:

- very few trainable parameters
- surprisingly strong adaptation
- much more affordable than full fine-tuning

In interviews, that is the answer people usually want.

## QLoRA

QLoRA combines:

- a quantized base model
- LoRA-style training on top

That means you can adapt large models while using much less memory.

This is one of the reasons LLM tuning became much more accessible outside giant labs.

## RLHF

RLHF means **Reinforcement Learning from Human Feedback**.

High-level flow:

1. collect human preference data
2. train a reward model
3. optimize the model toward preferred behavior

Why it matters:

Pretraining teaches language patterns.
It does not automatically teach the model to be helpful, safe, and aligned with what humans actually want.

## DPO

DPO stands for **Direct Preference Optimization**.

It became popular because it gives teams a cleaner way to learn from preferences without the full complexity of a traditional RLHF stack.

Simple interview answer:

DPO uses preference pairs directly and is often easier to train and operate than full RLHF.

## Quantization

Quantization reduces numerical precision, like:

- FP16 to INT8
- FP16 to 4-bit

Why do it?

- lower memory usage
- faster inference
- cheaper deployment

This is usually a serving optimization, not a magical quality booster.

## When Quantization Is a Great Idea

Use it when:

- memory is tight
- inference cost matters
- latency matters
- the task can tolerate small quality tradeoffs

This is common in real production deployments.

## When Quantization Needs Caution

Be careful when:

- quality margins are already thin
- the task is precision-sensitive
- you have not measured the degradation

Do not quantize because a blog post made it sound fashionable.
Measure first.

## Azure / DevOps Bridge

Think of these levers like deployment strategies:

- **prompting** = config change
- **RAG** = runtime dependency update
- **SFT / LoRA** = rebuild a specialized artifact
- **quantization** = optimize the runtime footprint

That framing helps you choose the lightest intervention that solves the real issue.

## Mumbai Indians Analogy

Optimization is like changing field strategy mid-over.

You do not move every fielder just because one boundary happened.
You first ask what actually went wrong:

- bad line and length?
- wrong field placement?
- matchup issue?
- dew changing the game?

Same with LLM systems.
Diagnose first. Tune second.

## A Good Interview Answer

If someone asks, "When would you fine-tune instead of using RAG?" a strong answer is:

"I would fine-tune when I need consistent behavioral change, domain tone, or structured output patterns. If the issue is fresh knowledge or internal documents, I would prefer RAG because it updates faster and is operationally cheaper."

That answer is simple and mature.

## Mini Pop Quiz

The assistant keeps responding in the wrong JSON format, but its facts are mostly correct.

Better first move:

- improve prompting or fine-tune for format behavior

Not:

- rebuild the retrieval stack

## How Would You Deploy This with Azure Pipelines?

For tuning workflows, I would version and validate:

- training dataset snapshot
- base model version
- adapter or LoRA config
- evaluation metrics
- safety checks
- rollback path to previous model artifact

Because model adaptation without release discipline is just expensive improvisation.
