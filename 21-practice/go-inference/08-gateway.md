---
module: Practice
topic: Go for Inference
subtopic: Capstone
status: unread
tags: [practice, go, capstone, gateway]
---
# Exercise 8 — Put It Together

**Time:** 4+ hours. **Solution:** [solutions/08-gateway.md](solutions/08-gateway.md)

## Build

One binary combining exercises 1–7: an inference gateway that accepts streaming
generation requests, admission-controls them, routes with prefix affinity to one of
several simulated backends, streams tokens back, propagates cancellation end to end, and
exports metrics.

```
client ──gRPC stream──> gateway ──> [admission] ──> [router] ──> backend replica
                           │                                          │
                           └────────── metrics, tracing ──────────────┘
```

## Requirements

1. Configuration from environment variables with sane defaults and validation at startup —
   fail fast on bad config rather than at first request.
2. Graceful shutdown draining in-flight streams within a deadline.
3. Per-tenant rate limiting on **both** tokens per minute and concurrency. Understand why
   one alone is insufficient.
4. Backends registered and health-checked in the background.
5. A load-test script demonstrating behavior under overload: rejection rate rises, latency
   for admitted requests stays flat.
6. A README stating your SLOs and which metric you would page on.

## What this is testing

Whether the pieces compose. Every individual exercise works in isolation; the gateway is
where you discover that your router's mutex is held across a network call, that shutdown
deadlocks against the queue drain, or that cancellation stops at the routing hop.

Requirement 3 is a real interview question: requests-per-second limits do not bound work
for LLM inference, because one 100k-token request costs more than a thousand short ones.
Token-based limits bound the work but not the concurrent KV footprint, which is what
actually exhausts a GPU. You need both.

Requirement 5 is the demonstration that matters. If your overload behavior is "everything
gets slower," the admission control is not doing its job — that is the unbounded-queue
failure from exercise 3 reappearing at system scale.

## Talking about it

This is a portfolio artifact. Be able to answer, in two minutes:

- What is the request path, hop by hop, and where can it fail?
- What did you measure, and what did the numbers say?
- What would break at 100× the load, and what would you change first?
- What did you deliberately not build, and why?

That last question is the one that separates a demo from engineering judgment. See
[../../17-projects/](../../17-projects/) for how to frame this as a project story.
