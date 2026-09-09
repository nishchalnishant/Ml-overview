---
module: Practice
topic: Go for Inference
subtopic: Concurrency Solution
status: unread
tags: [practice, go, concurrency, solutions]
---
# Solution 2 — Concurrent Batch Scorer

```go
package scorer

import (
	"context"
	"sync"
	"sync/atomic"
)

type Result struct {
	Index int
	Score float64
	Err   error // per-item error: partial results survive one bad input
}

func ScoreBatch(ctx context.Context, inputs []string, concurrency int, call func(context.Context, string) (float64, error)) ([]Result, error) {
	if concurrency <= 0 {
		concurrency = 1
	}
	// Pre-sized output. Each goroutine writes exactly one distinct index, so
	// there is no shared mutable state and no mutex is needed. The race
	// detector agrees because distinct elements of a slice are distinct memory
	// locations -- the slice header itself is only read.
	out := make([]Result, len(inputs))

	sem := make(chan struct{}, concurrency) // counting semaphore = the pool
	var wg sync.WaitGroup
	var inFlight, maxSeen int64

	for i, in := range inputs {
		// Go 1.22+ gives each iteration its own i and in. On <=1.21 you must
		// shadow them (i, in := i, in) or pass them as arguments.
		select {
		case <-ctx.Done():
			// Stop dispatching. Already-started work will observe ctx itself.
			out[i] = Result{Index: i, Err: ctx.Err()}
			continue
		case sem <- struct{}{}:
		}

		wg.Add(1)
		go func() {
			defer wg.Done()
			defer func() { <-sem }()

			if n := atomic.AddInt64(&inFlight, 1); n > atomic.LoadInt64(&maxSeen) {
				atomic.StoreInt64(&maxSeen, n) // approximate; test asserts <= concurrency
			}
			defer atomic.AddInt64(&inFlight, -1)

			score, err := call(ctx, in)
			out[i] = Result{Index: i, Score: score, Err: err}
		}()
	}

	wg.Wait() // every goroutine is accounted for: no leak possible
	return out, ctx.Err()
}
```

## Why this shape

**Order without sorting.** The output slice is allocated up front and each goroutine owns
one index. Appending to a shared slice under a mutex would give completion order and
require a sort afterward; this gives input order for free and removes the lock entirely.

**Partial results, not all-or-nothing.** `Result.Err` is per item. For a batch scorer this
is almost always right: one malformed input should not discard 999 good scores. The
alternative — `errgroup` with the first error cancelling everything — is right when the
items are parts of one logical answer. State which you chose and why; interviewers ask.

**Why the semaphore is acquired in the loop, not inside the goroutine.** Acquiring inside
would spawn `len(inputs)` goroutines immediately, all blocked on `sem`. For a million
inputs that is a million goroutines' worth of stacks. Acquiring in the loop bounds
goroutine count, not just concurrent calls.

## The traps, resolved

**The unbuffered-error-channel deadlock.** The naive version is:

```go
errCh := make(chan error)          // unbuffered
// ... each worker: if err != nil { errCh <- err }
wg.Wait()                          // main blocks here
close(errCh)
```

The first erroring worker blocks forever on the send because nothing reads `errCh` until
after `wg.Wait()`, and `wg.Wait()` never returns because that worker never calls
`wg.Done()`. Classic deadlock. Fixes: buffer the channel to `len(inputs)`, read it in a
separate goroutine, or — as above — do not use a channel for errors at all.

**Proving no leaks.** Add to the test file:

```go
func TestMain(m *testing.M) { goleak.VerifyTestMain(m) }
```

`go.uber.org/goleak` fails the test if any goroutine outlives it. Do this once and you
will catch the whole class.

**`-race` is not optional.** Run every concurrency test with it. It only detects races
that actually occur during the run, so exercise the concurrent path with enough inputs
(≥100) to make collisions likely.

## Follow-up the interviewer asks

*"What if a single call hangs forever?"* — `ctx` must carry a per-call timeout, not just a
batch timeout, or one stuck backend holds a worker slot for the life of the batch. Wrap
each `call` in `context.WithTimeout`. This is exercise 6.
