---
module: AI Infrastructure
topic: Distributed Systems
subtopic: Service Architecture Grpc Go
status: unread
tags: [ai-infrastructure, grpc, golang, streaming, backpressure, microservices]
prerequisites: [slos-and-observability]
---
# Service Architecture: gRPC and Go

The data plane is a network service before it is an ML system. These JDs ask for "expert-level Golang or Python" and gRPC proficiency because the components you build are proxies, routers, and schedulers — not models.

> **Scope note.** This covers the concurrency and RPC patterns a serving data plane actually uses, at the level an interview probes. It is not a substitute for writing Go. If you have not shipped Go, the coding round is a real risk that reading will not close — build a small streaming proxy and a worker pool before the interview.

---

## 1. Why gRPC for a Data Plane

Internal inference traffic is not REST-shaped. gRPC fits because:

- **HTTP/2 multiplexing** — many concurrent streams on one connection, no head-of-line blocking at the application layer, and no connection-per-request cost at high fan-out.
- **Server streaming** is the natural shape for token generation: one request, a stream of tokens, with the connection held open and flushed incrementally.
- **Protobuf** gives a compact binary encoding and a typed schema contract across services, which matters when a router, scheduler, and engine are written by different teams.
- **Deadlines propagate.** A client deadline travels with the call, so a downstream service knows how long it has and can abandon work nobody is waiting for.

Where REST still wins: the public edge, where clients expect HTTP/JSON and SSE. The common architecture is REST or SSE at the edge, gRPC internally.

---

## 2. Streaming Patterns

Four call types; token generation uses server streaming.

```protobuf
service Inference {
  rpc Generate(GenerateRequest) returns (stream Token);
}
```

**What to get right:**

**Cancellation must propagate to the GPU.** A client disconnecting mid-generation should stop the work, not leave a sequence generating tokens nobody will read while holding KV memory. In gRPC the context is cancelled; the engine must observe that and evict the sequence. Failing to wire this is a common and expensive bug — abandoned sequences are pure waste that shows up as mysterious capacity loss.

**Deadlines, not timeouts.** A deadline is an absolute point in time that propagates across hops; a timeout is per-hop and lets total latency accumulate unboundedly across a chain. Always set deadlines, and always derive downstream deadlines from the inbound one.

**Backpressure.** gRPC streams have flow control: if the receiver does not read, the sender eventually blocks. That is correct behaviour — it must propagate back into the scheduler rather than being papered over with an unbounded buffer. An unbounded queue converts backpressure into memory growth and then into an OOM, which is strictly worse than slowing down.

**Errors mid-stream.** Once tokens have been sent, you cannot change the status code. The stream needs an in-band way to signal partial failure, and clients need to handle a truncated-but-successful response.

---

## 3. Go for Data-Plane Services

Go dominates this layer for concrete reasons: goroutines make per-request concurrency cheap, the runtime is a single static binary, latency-focused GC suits proxies, and the Kubernetes/gRPC/observability ecosystem is Go-native. Python holds the ML layer; Go holds the plumbing around it.

**The patterns that appear in interviews:**

**Context is the spine.** `context.Context` carries deadline, cancellation, and request-scoped values through every call. Pass it as the first parameter, always; select on `ctx.Done()` in any loop that can block. This is how cancellation reaches the GPU.

```go
func (s *Server) Generate(req *pb.GenerateRequest, stream pb.Inference_GenerateServer) error {
    ctx := stream.Context()
    tokens, err := s.engine.Start(ctx, req)   // engine must honour ctx
    if err != nil {
        return status.Errorf(codes.ResourceExhausted, "admission: %v", err)
    }
    for {
        select {
        case <-ctx.Done():
            return ctx.Err()                  // client gone — engine cancels via ctx
        case tok, ok := <-tokens:
            if !ok {
                return nil                    // generation complete
            }
            if err := stream.Send(tok); err != nil {
                return err                    // send blocks under flow control: backpressure
            }
        }
    }
}
```

**Bounded worker pools, not unbounded goroutines.** Spawning a goroutine per request is idiomatic and fine for I/O, but work that consumes a scarce resource (GPU slots, KV blocks) must go through a bounded pool or a semaphore. That bound *is* your admission control.

**Channels for handoff, mutexes for state.** A buffered channel as a work queue gives natural backpressure when full. Shared mutable state — a routing table, a cache index — is usually simpler and faster behind a `sync.RWMutex` than behind a channel.

**Errors are values.** No exceptions; every call returns an error you must handle. Wrap with `%w` to preserve the chain. Verbose, but it makes failure paths explicit, which is why it suits infrastructure code.

**The classic bugs** worth being able to name: goroutine leaks from a goroutine blocked on a channel nobody closes (bound them with context), loop-variable capture in older Go versions, and unbounded channel buffers hiding backpressure.

---

## 4. Service Decomposition

A typical data plane splits into:

| Service | Responsibility | Scales with |
|---|---|---|
| Gateway | Auth, rate limit, protocol translation | QPS |
| Router | Replica selection, cache affinity | QPS |
| Scheduler | Admission, queueing, priority | Queue depth |
| Engine workers | The GPUs | Model size × replicas |
| Controller | Placement, config, health | Fleet size |

**The boundary rule:** split where scaling profiles or failure domains differ, not by conceptual tidiness. Router and scheduler are often one process because a network hop between them adds latency to every request for no isolation benefit. The controller is always separate — different SLO, different failure domain, and it must not be in the request path.

**Statelessness is the goal for the request path**, with one caveat: routing wants *soft* state (which replica has which prefix warm). Keep it soft — a cache that can be rebuilt, never a correctness dependency — so any router can serve any request and losing one costs efficiency rather than correctness.

---

## 5. Resilience Controls

Standard controls, with the inference-specific twist on each:

**Retries.** Safe for admission failures and connection errors before any token is emitted. **Unsafe once generation has started** — a retry re-runs prefill and doubles the cost of your most expensive requests. Never retry on timeout without knowing whether work began; use hedging only for short, idempotent calls.

**Circuit breakers.** Trip on a replica's error rate or latency to stop sending it work. The inference-specific signal is that a wedged GPU may pass health checks while producing nothing, so the breaker should also consider tokens-produced-per-second, not just errors.

**Timeouts must scale with the request.** A fixed 30-second timeout is wrong when output length varies by orders of magnitude. Bound TTFT and inter-token gap separately rather than total duration — a stalled stream is a failure; a long-but-progressing one is not.

**Graceful shutdown.** On SIGTERM, stop accepting new requests, drain in-flight generations up to a bound, then exit. Killing in-flight sequences wastes all compute spent on them, so drain windows here are longer than for typical web services and the deployment's termination grace period must accommodate that.

---

## 6. Interview Angles

### Q: A client disconnects mid-generation. Trace what should happen. [Medium]
The transport notices the broken stream and cancels the request context. The handler is selecting on `ctx.Done()`, so it returns promptly instead of continuing to push tokens into a dead stream. Critically, that cancellation has to reach the engine: the sequence must be removed from the running batch and its KV blocks freed, because otherwise it keeps occupying a batch slot and KV memory generating tokens nobody will read. That is why the context has to be threaded all the way from the gRPC handler into the scheduler rather than stopping at the service boundary. On the metrics side I would record it as a cancellation, not an error, so it does not consume the availability error budget — but I would still track the cancellation rate and wasted tokens, because a spike means clients are timing out, which is usually a symptom of something upstream.

### Q: Why would you not retry a failed inference request? [Medium]
Because the cost of a retry is asymmetric and the safety of it depends on how far the request got. Retrying after prefill has run means paying the most expensive part of the request twice, and under load — which is when failures cluster — retries add work exactly when the system is already saturated, turning a partial degradation into a retry storm. Retries are safe only for failures that occurred before any work began: connection errors, admission rejections, requests still queued. Once the first token has been emitted the request is not idempotent from a cost perspective and arguably not from a semantic one either, since sampling makes the second attempt a different response. The right pattern is retry only on pre-generation failures, with backoff and a budget cap, and use explicit 429s so clients back off rather than hammer.

### Q: Why Go rather than Python for the data plane, given the ML ecosystem is Python? [Medium]
Because the data plane is not doing ML — it is doing network I/O, routing, and scheduling, and the GPU work lives behind it in an engine process regardless of what language calls it. For that workload Go's advantages are concrete: cheap per-request concurrency without the GIL constraining CPU-bound request handling, predictable latency-oriented GC that matters for a proxy sitting on every request's critical path, single static binaries that simplify deployment, and first-class gRPC and Kubernetes ecosystems. Python remains the right language for the model layer and for anything touching the ML libraries, so the usual split is Python inside the engine and Go for the gateway, router, and controller around it. The honest caveat is that this is a team decision as much as a technical one — a Python shop with strong async experience can build a competent data plane in Python, and the consistency may be worth more than the runtime difference.

---

## Where to Next

- [SLOs and Observability](07-slos-and-observability.md) — instrumenting these services
- [Interview Drills](09-interview-drills.md) — end-to-end design exercises
- [Distributed Inference Frameworks](02-distributed-inference-frameworks.md) — what these services orchestrate
