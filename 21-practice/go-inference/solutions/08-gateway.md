---
module: Practice
topic: Go for Inference
subtopic: Capstone Solution
status: unread
tags: [practice, go, capstone, gateway, solutions]
---
# Solution 8 — Put It Together

This one is deliberately a **skeleton and a set of decisions**, not a full listing. The
exercise is composition; handing you 900 lines removes the part worth doing.

## Layout

```
inference-gateway/
├── cmd/gateway/main.go       wiring, config, signal handling
├── internal/config/          env parsing + validation
├── internal/admit/           queue from ex.3 + per-tenant limiter
├── internal/route/           prefix router from ex.5
├── internal/backend/         gRPC client pool, health polling
├── internal/obs/             metrics from ex.7
├── proto/inference.proto     from ex.4
└── loadtest/main.go          the overload demonstration
```

## Startup: fail fast

```go
type Config struct {
	GRPCAddr     string        `env:"GRPC_ADDR"     default:":9090"`
	MetricsAddr  string        `env:"METRICS_ADDR"  default:":9091"`
	Backends     []string      `env:"BACKENDS"      required:"true"`
	MaxQueue     int           `env:"MAX_QUEUE"     default:"256"`
	Workers      int           `env:"WORKERS"       default:"64"`
	TTFTBudget   time.Duration `env:"TTFT_BUDGET"   default:"1s"`
	DrainTimeout time.Duration `env:"DRAIN_TIMEOUT" default:"30s"`
}

func (c Config) Validate() error {
	if len(c.Backends) == 0 {
		return errors.New("BACKENDS is required")
	}
	if c.MaxQueue < c.Workers {
		// A queue smaller than the worker pool can never keep workers busy;
		// it is always a config mistake, and catching it at boot beats
		// discovering it as unexplained idle GPUs three weeks later.
		return fmt.Errorf("MAX_QUEUE (%d) < WORKERS (%d)", c.MaxQueue, c.Workers)
	}
	if c.DrainTimeout >= 60*time.Second {
		return errors.New("DRAIN_TIMEOUT must be under the k8s grace period")
	}
	return nil
}
```

Validate before binding a port. A gateway that starts, passes readiness, takes traffic, and
*then* discovers its backend list is empty has already failed requests.

## The request path

```go
func (g *Gateway) Generate(req *pb.GenerateRequest, stream pb.Inference_GenerateServer) error {
	ctx := stream.Context()
	tenant := tenantFrom(ctx)
	start := time.Now()

	// 1. Per-tenant limits: BOTH, see below.
	rel, err := g.limits.Acquire(ctx, tenant, estimateTokens(req))
	if err != nil {
		return status.Error(codes.ResourceExhausted, err.Error())
	}
	defer rel()

	// 2. Admission. Fast rejection, never blocking.
	slot, err := g.queue.Admit(ctx)
	if errors.Is(err, admit.ErrQueueFull) {
		obs.Requests.WithLabelValues(req.Model, "rejected").Inc()
		return status.Error(codes.ResourceExhausted, "queue full, retry")
	}
	defer slot.Release()

	// 3. Routing. Local state only -- no network call on this path.
	rep, err := g.router.Pick(req.Prompt)
	if err != nil {
		return status.Error(codes.Unavailable, "no healthy backend")
	}

	// 4. Fan the SAME ctx downstream. Client disconnect now reaches the GPU.
	up, err := rep.Client.Generate(ctx, req)
	if err != nil {
		return err
	}

	// 5. Relay, measuring TTFT separately from total.
	var ttft time.Duration
	n := 0
	for {
		tok, err := up.Recv()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			obs.RecordStream(req.Model, ttft.Seconds(), time.Since(start).Seconds(), n, "error")
			return err
		}
		if n == 0 {
			ttft = time.Since(start)
		}
		if err := stream.Send(tok); err != nil {
			return err // downstream client gone; defers free everything
		}
		n++
	}
	obs.RecordStream(req.Model, ttft.Seconds(), time.Since(start).Seconds(), n, "ok")
	return nil
}
```

Note there is no explicit cancellation code. That is the point of exercise 6: `ctx` is
threaded through every hop, so a disconnect at step 5 tears down step 4 automatically.

## Why both token and concurrency limits

**Requests per second bounds nothing.** One 100k-token request costs more GPU than a
thousand 50-token ones. An rps limit is nearly uncorrelated with the resource you are
protecting.

**Tokens per minute bounds total work but not instantaneous footprint.** A tenant within
its TPM budget can still open 500 simultaneous long-context streams. Each holds KV blocks
for its whole lifetime, and KV — not compute — is what runs out first. The result is an
OOM or a mass preemption while the tenant is technically compliant.

**Concurrency bounds footprint but not cost.** Ten concurrent streams generating 100k
tokens each, forever, is ten slots and an enormous bill.

So: TPM caps spend and long-run fairness; max-concurrent-streams caps peak KV. Neither
substitutes for the other, and this is a standard interview question — see
[../../system-design/02-llm-serving-mocks.md](../../system-design/02-llm-serving-mocks.md).

## Shutdown ordering

Order is the whole difficulty, and getting it wrong deadlocks:

1. Set `draining` → `/readyz` starts returning 503. **Then wait one or two LB probe
   intervals**, or the balancer keeps sending traffic to a server that has stopped
   accepting it.
2. `grpcServer.GracefulStop()` in a goroutine — stops new RPCs, lets streams finish.
3. Drain the queue with `DrainTimeout`.
4. Hard-stop anything still running and exit.

The classic deadlock: calling `GracefulStop()` *before* draining the queue. Queued requests
need to start new backend RPCs, `GracefulStop` refuses them, the drain never completes, and
shutdown hangs until `SIGKILL`.

## Load test — the deliverable that proves it

```
$ go run ./loadtest -rps 200 -duration 60s
p50 TTFT   180ms   p99 TTFT   890ms   rejected 0.0%

$ go run ./loadtest -rps 800 -duration 60s
p50 TTFT   210ms   p99 TTFT   960ms   rejected 61.2%
```

**That is the correct shape.** 4× the offered load, rejection rate rises, and latency for
admitted requests stays flat. If instead you see p99 climbing to 8 s with 0% rejected, your
admission control is not engaging and you have rebuilt the unbounded queue from exercise 3
at system scale.

## Composition bugs you should expect to hit

These are the actual point of the capstone; each is a real failure in real gateways.

- **Mutex held across a network call.** `router.Pick` looks fine until you inline the health
  probe inside the lock and every request serializes behind one slow backend.
- **Cancellation stopping at the relay.** Forgetting to pass `stream.Context()` to the
  upstream client makes the backend keep generating after the client is gone. Test it: kill
  the client and watch whether backend token counters keep rising.
- **Metrics under the lock.** `obs.QueueDepth.Set()` inside the queue's critical section
  adds Prometheus's own locking to your hot path.
- **Shutdown vs drain deadlock.** Above.
- **A goroutine per stream that outlives the stream.** `goleak` in the integration test.

## SLOs to state in the README

| Metric | Target |
|---|---|
| p99 TTFT | < 1 s |
| p99 inter-token latency | < 100 ms |
| Availability (non-5xx, excluding client cancels) | 99.9% |
| Rejection rate at nominal load | < 0.1% |

**Page on goodput**, not throughput or raw error rate. Throughput can be raised by making
every user slower; error rate is polluted by client cancellations you do not control.
Goodput — requests completed within SLO — moves only when users are actually being served
worse.
