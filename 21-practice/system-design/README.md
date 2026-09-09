---
module: Practice
topic: System Design Mocks
status: unread
tags: [practice, system-design, mock-interview, interview-prep]
---
# System Design Mock Scripts

These are written from the **interviewer's side**. Each script contains the prompt, the
follow-up pressure applied in stages, the rubric being scored against, and calibrated
weak / solid / strong answers for the hardest moments.

## Why interviewer-side

Reading a worked system design teaches you the shape of a good answer. It does not
prepare you for the part that actually decides the round: the interviewer interrupting
your clean design with "traffic just went up 50×" or "that costs three times the budget,
what now?" These scripts make that pressure explicit so you can rehearse against it.

## Files

| File | Design | Role fit |
|---|---|---|
| [01-ml-platform-mocks.md](01-ml-platform-mocks.md) | Feature store, training pipeline, recsys, fraud | ML engineer, applied scientist |
| [02-llm-serving-mocks.md](02-llm-serving-mocks.md) | Multi-tenant inference, RAG at scale, KV cache tier, agent platform | AI infra, inference data plane |
| [03-rubrics.md](03-rubrics.md) | Scoring dimensions, calibration bands, self-assessment | All |

## How to use these

**Solo.** Set a 45-minute timer. Read *only* the prompt block. Design out loud into a
recording or onto paper. At the 15, 25, and 35-minute marks, open the corresponding
follow-up stage and absorb it as if it were just said to you. Do not read ahead. Score
yourself with [03-rubrics.md](03-rubrics.md) afterward.

**With a partner.** Hand them the whole file. They need no domain expertise — the script
tells them when to interrupt and what to say.

## What these cannot do

They cannot replicate an interviewer who follows *your* specific answer down a path this
script did not anticipate, which is where most real rounds go. Treat these as pressure
rehearsal, not as a question bank to memorize. The design content itself lives in
[15-system-design/](../../15-system-design/) and
[20-ai-infrastructure/](../../20-ai-infrastructure/); these scripts assume you have read it.
