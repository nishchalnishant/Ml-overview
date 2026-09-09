---
module: Practice
topic: Go for Inference
subtopic: Routing Solution
status: unread
tags: [practice, go, routing, solutions]
---
# Solution 5 — Prefix-Affinity Router

```go
package router

import (
	"container/list"
	"context"
	"errors"
	"math"
	"sync"
	"time"
)

var ErrNoHealthyReplica = errors.New("no healthy replica")

type Replica struct {
	ID      string
	Addr    string
	load    float64 // 0..1, written by the background poller only
	healthy bool
	mu      sync.RWMutex
}

func (r *Replica) snapshot() (load float64, healthy bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.load, r.healthy
}

type Router struct {
	mu       sync.RWMutex
	replicas []*Replica
	// Bounded LRU: prefix hash -> replica ID. Requirement 6. A plain map keyed
	// by prompt grows without limit and is a slow memory leak in production.
	cache    map[uint64]*list.Element
	lru      *list.List
	maxCache int
	w1, w2   float64
}

type entry struct {
	key    uint64
	prefix string
	repID  string
}

func New(maxCache int, w1, w2 float64) *Router {
	return &Router{
		cache: make(map[uint64]*list.Element), lru: list.New(),
		maxCache: maxCache, w1: w1, w2: w2,
	}
}

// Pick reads only local state. It never makes a network call, so a sick backend
// can never add latency to the request path. This is the control-plane /
// data-plane split: health is gathered out of band, consulted in memory.
func (r *Router) Pick(prompt string) (*Replica, error) {
	r.mu.RLock()
	reps := r.replicas       // slice header copy; elements are stable pointers
	matches := r.matchLocked(prompt)
	r.mu.RUnlock()

	var best *Replica
	bestScore := math.Inf(-1)

	for _, rep := range reps {
		load, healthy := rep.snapshot()
		if !healthy {
			continue
		}
		// Normalize the prefix match so w1 and w2 are comparable: a raw
		// character count would swamp a 0..1 load factor at any sane weight.
		affinity := float64(matches[rep.ID]) / float64(len(prompt)+1)
		score := r.w1*affinity - r.w2*load
		if score > bestScore {
			best, bestScore = rep, score
		}
	}
	if best == nil {
		return nil, ErrNoHealthyReplica
	}
	r.record(prompt, best.ID)
	return best, nil
}
```

`matchLocked` walks the LRU entries and returns, per replica, the longest common prefix
length between `prompt` and any prefix that replica has served. A production version uses
a radix trie keyed on token IDs; the map-of-prefixes version is fine for the exercise and
makes the eviction question explicit.

```go
func (r *Router) record(prompt, repID string) {
	key := hash(prompt)
	r.mu.Lock()
	defer r.mu.Unlock()
	if el, ok := r.cache[key]; ok {
		r.lru.MoveToFront(el)
		el.Value.(*entry).repID = repID
		return
	}
	el := r.lru.PushFront(&entry{key: key, prefix: prompt, repID: repID})
	r.cache[key] = el
	for r.lru.Len() > r.maxCache { // the bound
		old := r.lru.Back()
		r.lru.Remove(old)
		delete(r.cache, old.Value.(*entry).key)
	}
}

// Poll runs in its own goroutine, off the request path.
func (r *Router) Poll(ctx context.Context, every time.Duration) {
	t := time.NewTicker(every)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			r.mu.RLock()
			reps := r.replicas
			r.mu.RUnlock()
			for _, rep := range reps {
				load, err := probe(ctx, rep.Addr) // network call lives HERE, not in Pick
				rep.mu.Lock()
				if err != nil {
					rep.healthy = false
				} else {
					rep.healthy, rep.load = true, load
				}
				rep.mu.Unlock()
			}
		}
	}
}
```

## Why affinity is worth this complexity

A 2000-token shared system prompt costs a full prefill on a cache miss and nothing on a
hit. At typical prefill rates that is roughly 800 ms versus ~80 ms TTFT — a 10× difference
on the metric users feel, achieved with no extra hardware. That is why every serious LLM
router does prefix-aware placement.

But affinity alone is a hotspot generator: the most popular prefix routes every request to
one replica while the rest idle. The `-w2*load` term is what stops that. The weights encode
how much TTFT you will trade for evenness.

## Tuning w1 and w2

Start with `w1 = 1.0`, `w2 = 1.0` on normalized inputs, then watch two signals:

- **Cache hit rate falling while replicas are evenly loaded** → `w2` too high, you are
  scattering requests that should have stuck.
- **One replica at 90% while others sit at 20%** → `w1` too high; affinity has overridden
  load and you are queueing behind a hot replica while capacity idles.

The observable that settles it is p99 TTFT, not hit rate. Hit rate is a proxy; if it rises
while p99 TTFT rises too, you have optimized the proxy and hurt the goal.

## Restarts

A replica restart empties its KV cache, but your prefix table still claims the entries.
You now route for affinity that no longer exists and pay a full prefill anyway — worse
than random, because you also created a hotspot. Detect it: track a monotonic
`start_epoch` in the health response and flush that replica's entries when it changes.
Never rely on a TTL alone, which is a race between eviction and misrouting.

## RWMutex vs Mutex

`RWMutex` wins here because `Pick` (read) vastly outnumbers `record`/`Poll` (write). It
loses when critical sections are very short — the extra bookkeeping costs more than the
contention it saves — and it loses under a high write rate, where writer starvation and
cacheline traffic on the shared reader counter make it slower than a plain `Mutex`.
Measure with `go test -bench` before assuming.
