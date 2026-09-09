---
module: Practice
topic: Go for Inference
subtopic: Observability Solution
status: unread
tags: [practice, go, prometheus, metrics, solutions]
---
# Solution 7 — Metrics and Health

```go
package obs

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	// Labels are BOUNDED: model and status are small closed sets. Never a user
	// ID, prompt hash, or raw path -- each distinct label combination is a
	// separate time series, and unbounded cardinality kills Prometheus.
	Requests = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "inference_requests_total",
	}, []string{"model", "status"})

	// TTFT: queueing + prefill. Buckets chosen for THIS distribution -- the
	// Prometheus defaults stop at 10s, which is useless for generation.
	TTFT = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "inference_ttft_seconds",
		Buckets: []float64{.01, .025, .05, .1, .25, .5, 1, 2.5, 5, 10},
	}, []string{"model"})

	// Inter-token latency: batch composition and decode speed. Different
	// causes, different fixes, so it must be a different metric.
	ITL = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "inference_inter_token_seconds",
		Buckets: []float64{.005, .01, .02, .04, .08, .16, .32},
	}, []string{"model"})

	Duration = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "inference_request_duration_seconds",
		Buckets: []float64{.1, .5, 1, 2, 5, 10, 30, 60, 120},
	}, []string{"model", "status"})

	QueueDepth = promauto.NewGauge(prometheus.GaugeOpts{Name: "inference_queue_depth"})
	InFlight   = promauto.NewGauge(prometheus.GaugeOpts{Name: "inference_in_flight"})
	Tokens     = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "inference_tokens_total",
	}, []string{"model", "kind"}) // kind = prompt | completion

	// Goodput: completed AND within SLO. Raw throughput can be improved by
	// making everyone slower; this cannot.
	Goodput = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "inference_requests_within_slo_total",
	}, []string{"model"})
)

const ttftSLO = 1.0 // seconds

func RecordStream(model string, ttft, total float64, outTokens int, status string) {
	Requests.WithLabelValues(model, status).Inc()
	Duration.WithLabelValues(model, status).Observe(total)
	if status != "ok" {
		return
	}
	TTFT.WithLabelValues(model).Observe(ttft)
	if outTokens > 1 {
		ITL.WithLabelValues(model).Observe((total - ttft) / float64(outTokens-1))
	}
	Tokens.WithLabelValues(model, "completion").Add(float64(outTokens))
	if ttft <= ttftSLO {
		Goodput.WithLabelValues(model).Inc()
	}
}
```

## Health endpoints

```go
// Liveness: is this PROCESS wedged? Nothing else. It must not touch any
// dependency -- a failing liveness probe means "kill and restart me", and
// wiring a database check in here turns a slow database into a fleet-wide
// crashloop that removes all capacity at the exact moment you need it.
mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusOK)
})

// Readiness: should I receive traffic RIGHT NOW? This may fail and recover
// freely. Failing it sheds traffic without losing the process, its warm caches,
// or its in-flight work.
mux.HandleFunc("GET /readyz", func(w http.ResponseWriter, r *http.Request) {
	switch {
	case !modelLoaded.Load():
		http.Error(w, "model still loading", http.StatusServiceUnavailable)
	case queue.Depth() >= maxDepth:
		http.Error(w, "saturated", http.StatusServiceUnavailable)
	case draining.Load():
		http.Error(w, "draining", http.StatusServiceUnavailable)
	default:
		w.WriteHeader(http.StatusOK)
	}
})
```

The `draining` case is why readiness exists at all: on `SIGTERM` you set it *before*
starting the shutdown drain, so the load balancer stops sending new requests while the
process is still alive to finish the ones it has.

## Structured logging with a request ID

```go
func WithRequestID(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		id := r.Header.Get("X-Request-ID")
		if id == "" {
			id = uuid.NewString()
		}
		// The logger goes in the CONTEXT so every downstream layer emits the
		// same ID without threading a parameter through every signature.
		log := slog.With("request_id", id)
		ctx := context.WithValue(r.Context(), logKey{}, log)
		w.Header().Set("X-Request-ID", id) // returned so users can quote it in tickets
		next.ServeHTTP(w, r.WithContext(ctx))
	})
}
```

## Why TTFT and ITL must be separate

Total duration for a streaming response conflates two unrelated things. A 60-second
request may be 59 s of fast tokens on a long answer (healthy) or 55 s stuck in a queue
before the first token (an outage). Same p99 duration, opposite diagnosis.

- **TTFT regressing** → queueing, admission, prefill, or routing. Look at queue depth and
  cache hit rate.
- **ITL regressing** → decode-side: batch composition, KV pressure, a memory-bandwidth
  ceiling, or a noisy co-tenant. Look at KV occupancy and batch size.

Alerting on total duration alone means you cannot tell these apart at 3 a.m.

## Why goodput

Throughput is maximized by batching aggressively, and aggressive batching raises latency
for everyone in the batch. A system doing 10,000 tok/s with half its requests missing the
TTFT SLO is worse than one doing 6,000 tok/s meeting all of them — the first is producing
tokens for users who have already given up. Goodput is the only one of the two that cannot
be gamed by making users wait, which is why it is what you page on.

## Histogram traps

**Buckets are a design decision.** They must bracket your SLO. If your target is p99 TTFT
< 1 s and your buckets are `[.005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5, 10]`, you can
resolve around the threshold. With default buckets ending at 10 s you cannot distinguish
11 s from 300 s, and every incident looks identical.

**Never average percentiles.** `avg(p99)` across ten instances is not the fleet p99 and has
no statistical meaning. Aggregate the raw buckets and compute the quantile once:

```promql
histogram_quantile(0.99, sum by (le, model) (rate(inference_ttft_seconds_bucket[5m])))
```

**Separate the metrics listener.** A `/metrics` scrape that serializes thousands of series
competes with serving traffic for the same handler goroutines and connection budget. Bind
it to a second port, one that is not exposed publicly.
