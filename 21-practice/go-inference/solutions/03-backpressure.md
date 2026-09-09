---
module: Practice
topic: Go for Inference
subtopic: Backpressure Solution
status: unread
tags: [practice, go, backpressure, solutions]
---
# Solution 3 — Request Queue with Backpressure

```go
package queue

import (
	"context"
	"errors"
	"sync"
	"time"
)

var (
	ErrQueueFull    = errors.New("queue full")
	ErrShuttingDown = errors.New("shutting down")
)

type Request struct{ Payload string }
type Response struct{ Body string }

type item struct {
	ctx      context.Context
	req      Request
	resp     chan Response
	errc     chan error
	enqueued time.Time
}

type Queue struct {
	items    chan *item
	work     func(context.Context, Request) (Response, error)
	wg       sync.WaitGroup
	closing  chan struct{}
	once     sync.Once
	waitHist func(time.Duration) // metric sink
}

func NewQueue(workers, maxDepth int, work func(context.Context, Request) (Response, error)) *Queue {
	q := &Queue{
		items:    make(chan *item, maxDepth), // the bound IS the buffered channel
		work:     work,
		closing:  make(chan struct{}),
		waitHist: func(time.Duration) {},
	}
	for i := 0; i < workers; i++ {
		q.wg.Add(1)
		go q.worker()
	}
	return q
}

func (q *Queue) Depth() int { return len(q.items) }

func (q *Queue) Submit(ctx context.Context, req Request) (Response, error) {
	it := &item{
		ctx: ctx, req: req,
		resp:     make(chan Response, 1), // buffered: worker never blocks replying
		errc:     make(chan error, 1),    // even if the caller has already left
		enqueued: time.Now(),
	}

	select {
	case <-q.closing:
		return Response{}, ErrShuttingDown
	default:
	}

	// The default case is the whole point: shed immediately rather than block.
	select {
	case q.items <- it:
	default:
		return Response{}, ErrQueueFull
	}

	select {
	case r := <-it.resp:
		return r, nil
	case err := <-it.errc:
		return Response{}, err
	case <-ctx.Done():
		// The caller gives up. The item stays in the channel but the worker will
		// check it.ctx and discard it without doing any work.
		return Response{}, ctx.Err()
	}
}

func (q *Queue) worker() {
	defer q.wg.Done()
	for it := range q.items {
		// Requirement 2: never spend a worker on an abandoned request.
		if err := it.ctx.Err(); err != nil {
			it.errc <- err
			continue
		}
		q.waitHist(time.Since(it.enqueued))

		resp, err := q.work(it.ctx, it.req)
		if err != nil {
			it.errc <- err
		} else {
			it.resp <- resp
		}
	}
}

func (q *Queue) Shutdown(ctx context.Context) error {
	q.once.Do(func() {
		close(q.closing) // Submit starts refusing
		close(q.items)   // workers drain the buffer, then their range loops end
	})
	done := make(chan struct{})
	go func() { q.wg.Wait(); close(done) }()
	select {
	case <-done:
		return nil
	case <-ctx.Done():
		return ctx.Err() // drain took too long; caller decides whether to hard-exit
	}
}
```

## Why bounded rejection beats blocking

An unbounded queue does not add capacity; it converts an overload into latency. Requests
sit until they time out, and every one of them consumed real work first — the system does
maximum work for zero delivered value. This is congestive collapse, and it is why every
serious data plane sheds load rather than queueing it.

Bounded + fast rejection means a well-defined fraction of requests fails in microseconds
while the admitted ones keep meeting their SLO. A caller that gets `ErrQueueFull` in 200 µs
can retry another region, degrade gracefully, or fail fast to its own caller. A caller that
gets a 30-second timeout can do none of those things.

## Little's Law

`L = λW` — queue depth equals arrival rate times time in system.

For the exercise's numbers: 200 ms backend, 500 rps, 1-second p99 target. Service time is
0.2 s, so the latency budget leaves 0.8 s of queueing. At 500 rps, a request waiting 0.8 s
sits behind `500 × 0.8 = 400` requests. So `maxDepth = 400` is the depth at which the
*last* admitted request just barely meets p99; anything deeper admits requests that are
guaranteed to miss. In practice set it below that — 200–300 — because service time is
distributed, not constant, and the tail is what you are protecting.

The general form: **maxDepth ≈ workers × (latencyBudget / serviceTime)**. Notice it depends
on your SLO, not on your memory. Sizing a queue by "how much RAM can I spare" is the
mistake that produces the unbounded-latency failure.

## Why depth, not rate

A requests-per-second limit is a guess about capacity made in advance, in a config file,
by someone who does not know what today's traffic mix looks like. Queue depth is a
*measurement* of whether you are currently keeping up. If backends slow down, depth rises
and you shed automatically; a static rps limit keeps admitting at the same rate into a
system that can no longer serve it.

For LLM inference this is sharper still: one request can cost 1000× another, so rps is
nearly uncorrelated with load. Depth — or better, KV-cache occupancy — tracks the thing
that actually saturates.
