---
module: Practice
topic: Go for Inference
subtopic: Routing
status: unread
tags: [practice, go, routing, load-balancing]
---
# Exercise 5 — Prefix-Affinity Router

**Time:** 90 minutes. **Solution:** [solutions/05-routing.md](solutions/05-routing.md)

## Build

A router that picks a backend replica for an incoming prompt, trading off cache affinity
against load.

```go
func (r *Router) Pick(prompt string) (*Replica, error)
```

## Requirements

1. Track, per replica, which prompt prefixes it has recently served.
2. Score each replica as `w1 * prefixMatchLen - w2 * loadFactor` and choose the best.
3. Replicas report load asynchronously; `Pick` must never block on a health check.
4. Concurrency-safe under many simultaneous `Pick` calls — `go test -race` clean.
5. Unhealthy replicas are excluded; if all are unhealthy, return a clear error rather
   than routing into a black hole.
6. Bound the prefix-tracking memory. It must not grow forever.

## What this is testing

The reason this routing exists: sending a request to a replica that already holds its KV
prefix skips prefill entirely, which for a long shared system prompt is the difference
between 800 ms and 80 ms TTFT. But pure affinity creates a hotspot, so the score must
balance the two.

Requirement 3 is the control-plane/data-plane invariant in miniature — a router that
synchronously asks a replica how loaded it is has put a network round trip on the request
path and made every backend a hard dependency. State must be pushed or polled in the
background and read from local memory.

## Traps

- `sync.RWMutex` over `sync.Mutex`: reads vastly outnumber writes here. Be able to say
  when `RWMutex` is actually worse (short critical sections, high write rate).
- Requirement 6 is where naive implementations fall over. A map keyed by full prompt grows
  unboundedly; you need an LRU bound or a trie with eviction.

## Discussion

- What happens to your prefix table when a replica restarts? How do you find out?
- How do you pick `w1` and `w2`? What observable would tell you they are wrong?
