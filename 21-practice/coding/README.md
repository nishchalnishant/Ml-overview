---
module: Practice
topic: Coding Drills
subtopic: ""
status: unread
tags: [practice, coding, drills, interview-prep]
---
# Coding Drills

Problems separated from solutions. Attempt cold, then check.

## Files

| File | Contents |
|---|---|
| [01-ml-problems.md](01-ml-problems.md) | 30 ML implementation problems, timed, no solutions |
| [01-ml-solutions.md](01-ml-solutions.md) | Worked solutions + what the interviewer scores |
| [02-dsa-problems.md](02-dsa-problems.md) | 24 DSA-screen problems — the round this repo otherwise skips |
| [02-dsa-solutions.md](02-dsa-solutions.md) | Approach, code, complexity |
| [03-debug-drills.md](03-debug-drills.md) | 12 broken snippets — find the bug (a real interview format) |
| [03-debug-solutions.md](03-debug-solutions.md) | The bug, why it is subtle, the fix |

## Why a DSA file exists here

Most ML job postings do not mention data structures. Most ML *loops* still contain one
algorithmic screen, especially at large companies and infrastructure-flavored roles like the
inference-data-plane track in [20-ai-infrastructure/](../../20-ai-infrastructure/README.md).
Candidates who prepare only ML material get filtered by a round they did not know was
coming. The set here is deliberately small and biased toward what actually appears in ML
loops — arrays, hashing, heaps, two pointers, intervals, light graphs — not a full
competitive-programming curriculum.

## Suggested cadence

- **Daily:** 1 ML problem (25 min) + 1 DSA problem (25 min)
- **Weekly:** re-attempt every problem you failed that week, cold
- **Pre-loop:** debug drills only — they are the fastest signal on whether your fundamentals
  are sharp

## Related

- [16-interview-prep/10-coding.md](../../16-interview-prep/10-coding.md) — worked reference with narration scripts
- [16-interview-prep/11-ml-coding-patterns.md](../../16-interview-prep/11-ml-coding-patterns.md) — failure-mode analysis of the same territory
