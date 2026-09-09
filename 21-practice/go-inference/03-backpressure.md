---
module: Practice
topic: Go for Inference
subtopic: Backpressure
status: unread
tags: [practice, go, backpressure, admission-control]
---
# Exercise 3 — Request Queue with Backpressure

**Time:** 90 minutes. **Solution:** [solutions/03-backpressure.md](solutions/03-backpressure.md)

## Build

An admission-controlled queue in front of a fixed-capacity backend.

```go
type Queue struct{ /* ... */ }
func NewQueue(workers, maxDepth int) *Queue
func (q *Queue) Submit(ctx context.Context, req Request) (Response, error)
func (q *Queue) Shutdown(ctx context.Context) error
```

## Requirements

1. `maxDepth` bounds the queue. When full, `Submit` returns `ErrQueueFull` **immediately**
   — it does not block waiting for space.
2. A caller whose `ctx` expires while queued is removed and does not consume a worker.
3. Report queue depth and wait time so they can be exported as metrics.
4. `Shutdown` drains queued work, refuses new submissions, and honors its own context
   deadline.
5. Under sustained overload, latency must not grow without bound — demonstrate this with
   a load test in the test file.

## What this is testing

This is the concept that separates people who have operated a data plane from people who
have not. **A queue is not capacity.** An unbounded queue converts an overload into
unbounded latency, and every request eventually times out having consumed real work —
the system does maximum work and delivers zero value. Bounded queues with fast rejection
mean some requests fail quickly while the rest stay healthy, which is strictly better.

Requirement 2 is the same lesson: dequeuing a request whose caller has already given up
is wasted GPU time. For an inference service, that waste is the expensive kind.

## Discussion questions

Answer these in writing before reading the solution; they are interview questions.

- Why is `maxDepth` a better admission signal than a requests-per-second rate limit?
- What is the right value for `maxDepth`? What does it depend on?
- Little's Law relates queue depth, arrival rate, and latency. Write it and use it to
  pick a depth for a 200 ms backend at 500 rps with a 1-second p99 target.
