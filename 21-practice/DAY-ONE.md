---
module: Practice
topic: Active Recall
subtopic: First Session
status: unread
tags: [practice, onboarding, diagnostic]
---
# Day One

A 90-minute diagnostic. Its purpose is **not** to make progress on the material — it is to
find out which of the tracks you actually need, so you do not spend three weeks drilling
the one you were already fine at.

Do it cold. Preparing for a diagnostic destroys the diagnostic.

## Before you start

Open [PROGRESS.md](PROGRESS.md). You will write five rows into it. A diagnostic you do not
record is just an anxious afternoon.

## The 90 minutes

| Block | Time | Task |
|---|---|---|
| 1 | 15 min | ML **1.1** stable softmax — [problems](coding/01-ml-problems.md) |
| 2 | 25 min | ML **3.2** scaled dot-product attention |
| 3 | 15 min | Debug **D2** and **D9** — [drills](coding/03-debug-drills.md) |
| 4 | 25 min | DSA **2.3** minimum window substring — [problems](coding/02-dsa-problems.md) |
| 5 | 10 min | Read mock 1's prompt only, and **speak** your first two minutes out loud |

Timer on for each. Stop when it fires, whether or not you are done — an overrun is data,
not a reason to keep going.

## Reading the result

| What happened | What it means | Go here |
|---|---|---|
| Block 1 took over 15 min, or you did not shift by the max | Fundamentals are shakier than you think; do not skip Tier 1 | Tier 1, all six |
| Block 2 finished but you could not justify `sqrt(d_k)` | You have memorized shapes, not mechanisms | [10-llms/](../10-llms/README.md), then re-attempt |
| Block 2 did not finish | The highest-value gap you have — this is asked constantly | Tier 3, in order |
| Block 3 missed either bug | Leakage blindness; the most common silent disqualifier | All 12 debug drills first |
| Block 4 did not finish | Expect to be filtered by an algorithmic screen | Daily DSA, one per day |
| Block 5 felt unnatural or you froze | Retrieval-under-pressure problem, not a knowledge problem | Mocks weekly, [rubrics](system-design/03-rubrics.md) |

Most people fail block 3 or block 5 and are surprised by it. Both are pure practice
problems and both improve fast.

## Then pick a cadence

- **No interview scheduled:** 1 ML + 1 DSA daily, one mock per week, the Go track on
  weekends if inference roles are the target.
- **Two to four weeks out:** whatever the diagnostic flagged, every day. One mock every
  two days. Skip the Go track unless the role is explicitly infrastructure.
- **Under a week:** debug drills and mocks only. Attempting new material this late adds
  anxiety and not much recall. Also read
  [16-interview-prep/03-pre-interview-checklist.md](../16-interview-prep/03-pre-interview-checklist.md).

## The honest part

Ninety minutes tells you where you stand; it does not move you. The moving happens in the
weeks of unglamorous timed attempts afterward, and the single best predictor of whether
you do those is whether you actually fill in [PROGRESS.md](PROGRESS.md). Write the five
rows now.

## Related

- [INDEX.md](INDEX.md) — everything in this section on one page
- [16-interview-prep/study-plans/](../16-interview-prep/study-plans/README.md) — the 30-day sprint, if you want a fuller schedule
