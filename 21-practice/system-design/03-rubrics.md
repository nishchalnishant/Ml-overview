---
module: Practice
topic: System Design Mocks
subtopic: Rubrics
status: unread
tags: [practice, system-design, rubric, self-assessment]
---
# Scoring Rubrics and Self-Assessment

Use after each mock in [01-ml-platform-mocks.md](01-ml-platform-mocks.md) and
[02-llm-serving-mocks.md](02-llm-serving-mocks.md).

## The six dimensions

Interviewers vary in what they call these, but nearly every ML design debrief reduces to
some version of the following. Score each 1–4.

| # | Dimension | What a 4 looks like |
|---|---|---|
| 1 | **Requirement clarification** | Extracted scale, latency, and the actual business objective before drawing anything; identified what was *not* specified and made assumptions explicit |
| 2 | **Structure** | The design proceeded in a legible order and could be followed without backtracking; you controlled the whiteboard rather than the interviewer dragging you |
| 3 | **ML-specific depth** | Labels, metrics, training/serving skew, feedback loops, and evaluation were raised unprompted — not just boxes and arrows |
| 4 | **Quantitative grounding** | Sized things with real arithmetic; QPS, memory, and bandwidth were computed, not asserted |
| 5 | **Tradeoff reasoning** | Every choice came with what it costs and under what condition you would choose differently |
| 6 | **Failure and operations** | What breaks, how you detect it, how it degrades, what you do at 3 a.m. |

### Band interpretation

- **1 — Absent.** Not raised, or wrong when raised.
- **2 — Prompted.** Handled correctly, but only after the interviewer asked.
- **3 — Volunteered.** Raised unprompted and handled correctly.
- **4 — Volunteered with the tradeoff.** Raised unprompted, handled correctly, *and* you
  named what the alternative would have bought and when you would switch.

The gap between 2 and 3 is what most often separates a hire from a no-hire. Being able to
answer a question correctly once asked is table stakes; knowing that the question exists is
the signal.

## The specific failure modes

These are the ones that reliably sink otherwise-competent candidates.

**Designing the system you wanted to design.** The prompt says 5,000 rps and you build for
5 million because distributed systems are more fun to draw. Over-engineering scores as
poor judgment, not as ambition.

**Boxes without ML.** A diagram with Kafka, a feature store, and a model box is a *data*
architecture. If you never said what the label is, what metric you optimize, or how you
know the model is working, you have not done an ML design.

**Numbers as decoration.** "We'll need a lot of GPUs" is worth nothing. "70B at bf16 is
140 GB of weights, so four H100s per replica leaves roughly 180 GB for KV" is the answer.
In inference-infra rounds this dimension alone can decide the outcome.

**Ignoring the feedback loop.** Recsys, fraud, and ranking systems all train on data their
own decisions generated. Candidates who never mention this are visibly missing a layer of
experience.

**Defending rather than updating.** When the interviewer pushes, they are usually testing
whether you can revise. Digging in on a choice that has just been shown to be weak scores
far worse than "you're right, that breaks — here's what I'd do instead."

**Running out of road.** Spending 30 of 45 minutes on data ingestion and never reaching
serving or monitoring. Budget explicitly: roughly 5 minutes on requirements, 10 on
high-level design, 20 on the deep dive the interviewer steers toward, 10 on failure modes
and operations.

## Self-assessment sheet

After each mock, fill this in before reading any solution notes.

```
Mock:                                  Date:
Dimension scores:  1__  2__  3__  4__  5__  6__

Where I stalled (be specific — the exact question):

What I said that I now think was wrong:

What I did not know versus what I knew but did not say:
  didn't know:
  knew, didn't say:

The one number I could not compute:

Re-attempt on (date, 3+ days out):
```

The "knew but didn't say" line is the important one. Most system-design failure is
retrieval failure under pressure, not missing knowledge — and the fix for that is
rehearsal, which is different from the fix for a knowledge gap.

## Cadence

One mock per week is enough; these are expensive to do properly and worthless done
casually. Re-attempt any mock scoring below 3 on two or more dimensions after at least
three days, and do it out loud — silently reading the script a second time will feel
productive and teach you nothing.
