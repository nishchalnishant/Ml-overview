---
module: Practice
topic: Active Recall
subtopic: Progress Tracker
status: unread
tags: [practice, tracker, spaced-repetition]
---
# Progress Tracker

Copy this file, or edit it in place and commit as you go. It exists because the
[root README](README.md) asks you to record *where* you got stuck, and advice to record
something with nowhere to record it does not get followed.

## How to fill this in

One line per attempt. The only column that matters is the last one.

- **Result:** `✅` finished in time and correct · `⚠️` finished but wrong or over time ·
  `❌` did not finish
- **Stuck at:** the *specific* line or step. "Blanked on the recurrence" is useful.
  "Was hard" is not.
- **Knew / Didn't:** the single most important distinction in this file. **K** = you knew
  it and did not retrieve it under time. **D** = you did not know it.

**K and D have different fixes.** D means go read the source material. K means you have a
retrieval problem, and rereading will not touch it — you need more timed attempts. Most
people diagnose every failure as D, go reread, feel productive, and do not improve. If
your log fills up with **K**, stop reading and start drilling.

---

## Attempt log

| Date | Problem | Time | Result | Stuck at | K/D |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

---

## Re-attempt queue

Anything marked ⚠️ or ❌ goes here with a date **at least 3 days out**. Immediate
re-attempts test short-term memory and inflate your sense of readiness.

| Re-attempt on | Problem | Original failure | Round 2 result |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

A problem leaves this queue after **two** clean passes, spaced. One pass right after
reading the solution proves nothing.

---

## System design mock scores

Dimensions and bands are defined in [system-design/03-rubrics.md](system-design/03-rubrics.md).
Score 1–4 each; the 2→3 boundary (prompted vs volunteered) is what separates outcomes.

| Date | Mock | Reqs | Structure | ML depth | Numbers | Tradeoffs | Failure modes | Total /24 |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

**Watch the column, not the total.** A flat 3 across the board is a stronger signal than a
4-4-4-1-4-3 with the same sum — one weak dimension is what an interviewer probes, and
quantitative grounding is the usual gap.

---

## Coverage

Tick as you complete a first clean attempt. The point is finding the untouched cluster,
not the count.

**ML coding** — Tier 1: `1.1` `1.2` `1.3` `1.4` `1.5` `1.6`
Tier 2: `2.1` `2.2` `2.3` `2.4` `2.5` `2.6` `2.7` `2.8` `2.9` `2.10`
Tier 3: `3.1` `3.2` `3.3` `3.4` `3.5` `3.6` `3.7` `3.8`
Tier 4: `4.1` `4.2` `4.3` `4.4` `4.5` `4.6`

**DSA** — T1: `1.1` `1.2` `1.3` `1.4` `1.5` `1.6`
T2: `2.1` `2.2` `2.3` `2.4` `2.5` `2.6`
T3: `3.1` `3.2` `3.3` `3.4` `3.5` `3.6` `3.7`
T4: `4.1` `4.2` `4.3` `4.4` `4.5` `4.6` `4.7` `4.8`

**Debug** — `D1` `D2` `D3` `D4` `D5` `D6` `D7` `D8` `D9` `D10` `D11` `D12`

**Mocks** — `1` `2` `3` `4` `5` `6` `7` `8`

**Go** — `1` `2` `3` `4` `5` `6` `7` `8`

---

## Readiness check

Not a score. Five questions to answer honestly before a loop.

1. Can you write attention from scratch, from a blank file, in under 20 minutes?
2. Did your last three mocks score ≥ 3 on quantitative grounding without being prompted?
3. Is your re-attempt queue empty, or is it hiding the things you keep avoiding?
4. Are your recent failures mostly **D**? If they are mostly **K**, you need reps, not reading.
5. Can you tell a two-minute story about something you built, with a number in it?

Question 5 is the one this section cannot help with. See
[17-projects/](../17-projects/README.md) and
[16-interview-prep/09-behavioral.md](../16-interview-prep/09-behavioral.md).
