---
module: Practice
topic: Go for Inference
subtopic: Observability
status: unread
tags: [practice, go, prometheus, metrics, slo]
---
# Exercise 7 — Metrics and Health

**Time:** 75 minutes. **Solution:** [solutions/07-observability.md](solutions/07-observability.md)

## Build

Instrument the service from the previous exercises with Prometheus metrics and correct
health endpoints.

## Requirements

1. Metrics: request rate, error rate, latency **histogram** (not a gauge or an average),
   queue depth, in-flight requests, and tokens generated.
2. For a streaming endpoint, measure **TTFT and inter-token latency separately** from
   total duration.
3. `/healthz` (liveness) and `/readyz` (readiness) with genuinely different semantics.
4. Structured JSON logs including a request ID that appears on every line for a request.
5. Label cardinality stays bounded. No user IDs, prompt hashes, or raw paths as labels.
6. Expose a "goodput" metric: requests completed **within SLO**, not merely completed.

## What this is testing

Requirement 2 is the domain-specific one. Total request duration for a streaming response
conflates "the model was slow to start" with "the response was long," and those have
opposite fixes. TTFT is dominated by queueing and prefill; inter-token latency is
dominated by batch composition and decode speed.

Requirement 3 is the one that causes real outages. Liveness failing means "restart me";
readiness failing means "stop sending traffic, I am still alive." Wiring a dependency
check into liveness means a slow database restarts your entire fleet in a loop.

Requirement 6 is the metric distinction interviewers probe: 10,000 tokens/sec where half
the requests blow their latency SLO is worse than 6,000 tokens/sec where all of them meet
it. Raw throughput can be improved by making everyone slower, which is why goodput is what
you page on.

## Traps

- Histogram buckets must be chosen for your actual latency distribution. Prometheus
  defaults top out at 10 seconds — useless for a 60-second generation.
- Averaging percentiles across instances is mathematically meaningless. Aggregate the
  histograms and compute the quantile from the total.
- `promhttp` on the same port as your data plane means a metrics scrape competes with
  serving traffic. Consider a separate listener.
