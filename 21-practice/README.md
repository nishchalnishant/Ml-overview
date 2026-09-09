---
module: Practice
topic: Active Recall
subtopic: ""
status: unread
tags: [practice, drills, interview-prep, active-recall]
---
# 21 — Practice

Every other section in this repo is written to be **read**. This one is written to be
**attempted**. That difference is the entire point of the section.

## Why this exists

Reading a worked solution produces the feeling of understanding without the ability to
reproduce it. This is the single most common way strong candidates fail interviews: they
have read about k-means fifty times and have never once written it on a blank page with a
clock running.

Everything here separates the **prompt** from the **solution** into different files. You are
meant to attempt first, badly, before you look. A failed attempt followed by reading the
answer builds far more durable recall than reading the answer twice.

## Start here

| File | Use it when |
|---|---|
| [DAY-ONE.md](DAY-ONE.md) | It is your first session — a 90-minute diagnostic that tells you which track you need |
| [INDEX.md](INDEX.md) | You want to pick a problem in ten seconds instead of opening four files |
| [PROGRESS.md](PROGRESS.md) | Every session — record what you got wrong and when to retry it |

## The three tracks

| Track | What it drills | Where the answers live |
|---|---|---|
| [coding/](coding/README.md) | Writing correct ML + DSA code under time pressure | `*-solutions.md` |
| [system-design/](system-design/README.md) | Defending design decisions under interviewer pushback | rubrics inside each mock |
| [go-inference/](go-inference/README.md) | Building a real streaming inference service in Go | `solutions/` per exercise |

## How to use this section

1. **Set a timer before you read the problem.** The constraint is the exercise. Without it
   you will drift into reading mode.
2. **Write the attempt somewhere real** — an editor, a terminal, paper. Not in your head.
   "I know how I'd do this" is the illusion this section exists to break.
3. **When stuck, note *where* you got stuck, then look.** The location of the block is the
   diagnostic information. Reading the solution without recording that throws it away.
4. **Re-attempt failed problems 2–3 days later**, not immediately. Immediate re-attempts
   test short-term memory; spaced ones test what you actually retained.

## What this section cannot do

Stated plainly, because the rest of the repo is honest about its limits and this section
should be too:

- **It cannot substitute for a human interviewer.** The system-design mocks are written so
  a friend can run them at you, and they are usable solo, but neither replaces real
  back-and-forth with someone who pushes where you are weakest.
- **It cannot give you project stories.** Interviewers dig into what *you* built. No
  problem bank supplies that.
- **The Go track will not make you fluent.** It will get you from "cannot write Go" to
  "can write a working streaming service and talk about it honestly." That is a real
  distance, and it is not the same as fluency.

## Related

- [INDEX.md](INDEX.md) — every problem in this section, with difficulty and the trap in each
- [16-interview-prep/](../16-interview-prep/README.md) — the reference material these drills test
- [20-ai-infrastructure/](../20-ai-infrastructure/README.md) — background for the Go track
- [17-projects/](../17-projects/README.md) — runnable end-to-end projects
