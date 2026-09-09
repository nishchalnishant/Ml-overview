---
module: Practice
topic: Go for Inference
subtopic: Concurrency
status: unread
tags: [practice, go, concurrency, goroutines]
---
# Exercise 2 — Concurrent Batch Scorer

**Time:** 90 minutes. **Solution:** [solutions/02-concurrency.md](solutions/02-concurrency.md)

## Build

A function that scores a slice of inputs against a slow backend (simulate with
`time.Sleep` plus a random failure), running up to `N` concurrently, returning results in
**input order**.

```go
func ScoreBatch(ctx context.Context, inputs []string, concurrency int) ([]Result, error)
```

## Requirements

1. At most `concurrency` in-flight calls at any moment. Verify this with a counter, not
   by assumption.
2. Results in input order regardless of completion order.
3. One backend failure does not lose the other results. Decide and document your policy:
   fail the whole batch, or return partial results with per-item errors.
4. If `ctx` is cancelled, return promptly — do not wait for in-flight work you no longer
   need.
5. No goroutine leaks. Prove it with `goleak` or by checking `runtime.NumGoroutine()`
   before and after.
6. Run `go test -race` and have it pass.

## What this is testing

The worker-pool pattern, and specifically: preserving order via **index-carrying results
or a pre-sized output slice** rather than by appending to a shared slice under a mutex.
Requirement 5 is where most people fail — a goroutine blocked forever on a send to a
channel nobody reads is the single most common Go leak, and it is invisible until
production memory climbs.

## Traps

- Capturing the loop variable. On Go 1.22+ this is fixed; on earlier versions it is the
  classic bug. Know which version you are on and why it matters.
- `sync.WaitGroup` plus an unbuffered error channel deadlocks when more than one worker
  errors. Work out why before reading the solution.
- Writing to distinct indices of a pre-sized slice from multiple goroutines is safe. Be
  able to say why the race detector agrees.
