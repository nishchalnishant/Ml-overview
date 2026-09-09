---
module: Practice
topic: Go for Inference
subtopic: Context Solution
status: unread
tags: [practice, go, context, cancellation, solutions]
---
# Solution 6 — Cancellation All the Way Down

```go
package gateway

import (
	"context"
	"errors"
	"log/slog"
	"net/http"
	"sync/atomic"
	"time"
)

// kvBlocks stands in for GPU memory. The test asserts it returns to zero.
var kvBlocks atomic.Int64

// --- Hop 1: HTTP handler --------------------------------------------------

func (s *Server) handle(w http.ResponseWriter, r *http.Request) {
	// r.Context() is ALREADY cancelled when the client disconnects. Every
	// downstream context must descend from it; context.Background() here would
	// silently sever the chain and is the single most common version of this bug.
	ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
	defer cancel() // always, unconditionally -- go vet flags the omission

	out, err := s.enqueue(ctx, r.URL.Query().Get("prompt"))
	switch {
	case err == nil:
		w.Write([]byte(out))

	case errors.Is(err, context.Canceled):
		// The CLIENT left. Nothing is broken on our side. 499 is nginx's
		// non-standard code for this; the important part is that it is not
		// counted as a server error, or your error-rate SLO alerts on user
		// behavior you do not control.
		slog.Info("client cancelled", "path", r.URL.Path)
		w.WriteHeader(499)

	case errors.Is(err, context.DeadlineExceeded):
		// WE were too slow. This is our problem and it belongs in the error budget.
		slog.Warn("deadline exceeded", "path", r.URL.Path, "budget", "30s")
		w.WriteHeader(http.StatusGatewayTimeout)

	default:
		w.WriteHeader(http.StatusInternalServerError)
	}
}

// --- Hop 2: queue ---------------------------------------------------------

func (s *Server) enqueue(ctx context.Context, prompt string) (string, error) {
	// Tighter than the parent. If the backend budget is 25s and the parent is
	// 30s, a backend timeout surfaces as a backend timeout -- attributable --
	// rather than as the outer deadline firing, which tells you nothing about
	// which hop was slow.
	ctx, cancel := context.WithTimeout(ctx, 25*time.Second)
	defer cancel()

	select {
	case s.slots <- struct{}{}:
		defer func() { <-s.slots }()
	case <-ctx.Done():
		return "", ctx.Err() // gave up while queued; no worker was consumed
	}
	return s.backend(ctx, prompt)
}

// --- Hop 3: backend call --------------------------------------------------

func (s *Server) backend(ctx context.Context, prompt string) (string, error) {
	ctx, cancel := context.WithTimeout(ctx, 20*time.Second)
	defer cancel()
	return s.generate(ctx, prompt)
}

// --- Hop 4: simulated GPU work -------------------------------------------

func (s *Server) generate(ctx context.Context, prompt string) (string, error) {
	blocks := int64(len(prompt)/16 + 1)
	kvBlocks.Add(blocks)
	// The defer is what makes requirement 3 true on EVERY exit path: normal
	// return, cancellation, deadline, or panic. Releasing at the bottom of the
	// happy path only would leak on exactly the cases that matter.
	defer kvBlocks.Add(-blocks)

	var out []byte
	tick := time.NewTicker(20 * time.Millisecond)
	defer tick.Stop()

	for i := 0; i < 100; i++ {
		select {
		case <-ctx.Done():
			return string(out), ctx.Err() // partial output plus the reason
		case <-tick.C:
			out = append(out, 'x')
		}
	}
	return string(out), nil
}
```

## The test that proves it

```go
func TestCancellationReleasesEverything(t *testing.T) {
	defer goleak.VerifyNone(t)

	before := runtime.NumGoroutine()
	ctx, cancel := context.WithCancel(context.Background())

	done := make(chan error, 1)
	go func() { _, err := srv.enqueue(ctx, strings.Repeat("a", 512)); done <- err }()

	time.Sleep(100 * time.Millisecond) // let it get into generate()
	if kvBlocks.Load() == 0 {
		t.Fatal("expected blocks to be held mid-flight")
	}
	cancel()

	if err := <-done; !errors.Is(err, context.Canceled) {
		t.Fatalf("got %v, want context.Canceled", err)
	}
	// Requirement 3: the resource is back.
	if n := kvBlocks.Load(); n != 0 {
		t.Fatalf("leaked %d KV blocks", n)
	}
	if after := runtime.NumGoroutine(); after > before {
		t.Fatalf("leaked goroutines: %d -> %d", before, after)
	}
}
```

## Why this is the exercise that matters most

Cancellation that stops at hop 1 is worse than no cancellation. The client gets a fast
error and moves on; the GPU keeps generating tokens nobody will read, holding KV blocks
that queued requests are waiting for. You have converted a client disconnect into reduced
capacity for everyone else. At scale this is the "a third of our token generation was for
abandoned requests" finding — invisible in latency metrics, extremely visible on the bill.

The chain only works if it is unbroken. One `context.Background()` anywhere on the path,
one function that takes no `ctx`, one goroutine spawned without inheriting it, and
everything below that point keeps running.

## Canceled vs DeadlineExceeded

They have opposite remediations and must never be aggregated:

| | Means | Who fixes it |
|---|---|---|
| `Canceled` | client went away | usually nobody; check for over-aggressive client timeouts |
| `DeadlineExceeded` | we exceeded our budget | you — capacity, batching, or the budget itself |

Counting `Canceled` as a 5xx makes your error rate track user behavior. Counting
`DeadlineExceeded` as a client problem hides real degradation. Both are common and both
are bad.

## Why context does not go in a struct field

A `context` carries a *request's* deadline and cancellation. A struct outlives any single
request, so a stored context is either stale (cancelled, breaking later requests) or
immortal (never cancelled, defeating the point). It also makes the lifetime invisible at
the call site. The standard library's own guidance is explicit: pass it as the first
parameter, name it `ctx`, and never store it. The single narrow exception is a struct that
*is* one request — and even then, passing is clearer.
