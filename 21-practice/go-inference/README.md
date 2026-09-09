---
module: Practice
topic: Go for Inference
status: unread
tags: [practice, go, grpc, inference, infrastructure]
---
# Go for Inference Infrastructure

A build-along track: eight exercises that assemble into one working streaming
inference gateway. Not a Go tutorial — it assumes you can read Go and teaches the
subset that inference-infrastructure roles actually interview on.

## The honest framing

Reading this will not make you fluent in Go. Nothing except writing Go does that. What
this track *can* do is get you to the point where you can discuss a Go data plane
credibly, read an existing codebase, and write a correct concurrent handler under
interview conditions. That is the realistic target, and it is enough for most
inference-infra loops, which weight systems reasoning above language mastery.

If you have three weeks, do all eight. If you have three days, do exercises 2, 4, and 6 —
concurrency, streaming, and cancellation are what get asked.

## Why Go at all

Inference control planes and gateways are overwhelmingly Go: llm-d, Dynamo's routing
layer, KServe, Kubernetes itself, most gRPC gateways. The reason is narrow and worth
being able to state — goroutines make per-connection concurrency cheap enough to hold
tens of thousands of open streams, `context` gives you cancellation that propagates
across service boundaries, and static binaries with no runtime deploy trivially in
containers. Python holds the *model*; Go holds everything around it.

## The exercises

Each exercise has a spec here and a reference implementation in
[solutions/](solutions/). Build first.

| # | Exercise | What it teaches |
|---|---|---|
| 1 | [Token counter service](01-basics.md) | Structs, interfaces, errors, `net/http` |
| 2 | [Concurrent batch scorer](02-concurrency.md) | Goroutines, channels, `WaitGroup`, `errgroup` |
| 3 | [Request queue with backpressure](03-backpressure.md) | Buffered channels, `select`, admission control |
| 4 | [Streaming gRPC token API](04-grpc-streaming.md) | Protobuf, server streaming, flow control |
| 5 | [Prefix-affinity router](05-routing.md) | Maps, sorting, scoring, `sync.RWMutex` |
| 6 | [Cancellation all the way down](06-context.md) | `context`, deadlines, cleanup on disconnect |
| 7 | [Metrics and health](07-observability.md) | Prometheus, histograms, `/healthz` vs `/readyz` |
| 8 | [Put it together](08-gateway.md) | Assemble 1–7 into a working gateway |

## Setup

```bash
mkdir inference-gateway && cd inference-gateway
go mod init example.com/inference-gateway
go version   # 1.21+
```

Exercise 4 additionally needs `protoc` with the Go and gRPC plugins.

## How to use the solutions

The solutions are commented with *why*, not *what*. Read one only after your own version
compiles and runs — a solution read before the attempt turns a two-hour learning session
into a ten-minute reading session that teaches nothing.

## Related

- [20-ai-infrastructure/08-service-architecture-grpc-go.md](../../20-ai-infrastructure/08-service-architecture-grpc-go.md) — the conceptual layer under these exercises
- [20-ai-infrastructure/05-smart-routing-and-scheduling.md](../../20-ai-infrastructure/05-smart-routing-and-scheduling.md) — the theory behind exercise 5
- [20-ai-infrastructure/07-slos-and-observability.md](../../20-ai-infrastructure/07-slos-and-observability.md) — the theory behind exercise 7
