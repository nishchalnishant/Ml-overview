---
module: Practice
topic: Go for Inference
subtopic: Context
status: unread
tags: [practice, go, context, cancellation]
---
# Exercise 6 — Cancellation All the Way Down

**Time:** 75 minutes. **Solution:** [solutions/06-context.md](solutions/06-context.md)

## Build

A three-hop request path — HTTP handler → queue → backend call → simulated GPU work —
where cancelling at the top frees every resource at the bottom.

## Requirements

1. A client disconnect at the HTTP layer cancels the whole chain.
2. Each hop has its own deadline, tighter than the one above it.
3. The simulated "GPU work" releases its "KV blocks" (a counter you decrement) on
   cancellation. Assert the counter returns to zero.
4. Distinguish `context.Canceled` from `context.DeadlineExceeded` in your logs and in the
   status code you return — they mean different things operationally.
5. A test that cancels mid-flight and asserts every resource was released.
6. No goroutine leaks after cancellation.

## What this is testing

The single most valuable Go idiom for inference infrastructure. Cancellation that stops at
the first hop is worse than useless: the client sees a fast failure while the GPU keeps
working on output nobody will read. That is the "30% of tokens generated for requests
nobody is waiting on" from the agent-platform mock, and it shows up on the bill.

Requirement 4 matters because the two errors have different remediations. `Canceled` means
the client left — usually fine, possibly a client-side timeout that is too aggressive.
`DeadlineExceeded` means *you* were too slow, and it is your problem.

## Traps

- `context.Background()` inside a request path silently severs the chain. Every function
  on the path takes a `ctx` as its first parameter — no exceptions.
- Storing a `context` in a struct field instead of passing it. The standard library
  explicitly warns against this; know the reasoning.
- `defer cancel()` on every `WithCancel`/`WithTimeout`, always, even when you think the
  context will be cancelled anyway. `go vet` catches the omission and it is a real leak.
