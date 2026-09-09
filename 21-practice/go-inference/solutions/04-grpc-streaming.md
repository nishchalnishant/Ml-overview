---
module: Practice
topic: Go for Inference
subtopic: gRPC Streaming Solution
status: unread
tags: [practice, go, grpc, streaming, solutions]
---
# Solution 4 — Streaming gRPC Token API

## Proto

```protobuf
syntax = "proto3";
package inference.v1;
option go_package = "example.com/inference-gateway/gen/inference/v1;inferencev1";

service Inference {
  rpc Generate(GenerateRequest) returns (stream Token);
}

message GenerateRequest {
  string prompt     = 1;
  int32  max_tokens = 2;
  string model      = 3;
}

message Token {
  string text          = 1;
  int32  index         = 2;
  string finish_reason = 3;  // "" until the last message: "stop" | "length"
}
```

```
protoc --go_out=. --go_grpc_out=. proto/inference.proto
```

## Server

```go
func (s *Server) Generate(req *pb.GenerateRequest, stream pb.Inference_GenerateServer) error {
	ctx := stream.Context() // cancelled when the client disconnects. This is the signal.

	max := int(req.GetMaxTokens())
	if max <= 0 || max > 4096 {
		return status.Error(codes.InvalidArgument, "max_tokens must be in (0, 4096]")
	}

	tokens, err := s.engine.Start(ctx, req.GetPrompt())
	if err != nil {
		return status.Errorf(codes.Internal, "engine start: %v", err)
	}
	defer s.engine.Release(tokens) // frees simulated KV blocks on EVERY exit path

	for i := 0; i < max; i++ {
		select {
		case <-ctx.Done():
			// Client is gone. Returning here runs the defer, releasing KV.
			// Without this, generation continues into a dead stream, burning
			// GPU and holding blocks other requests are queued for.
			return status.FromContextError(ctx.Err()).Err()

		case tok, ok := <-tokens:
			if !ok {
				return stream.Send(&pb.Token{Index: int32(i), FinishReason: "stop"})
			}
			// Send's error is the OTHER disconnect signal, and it can fire
			// before ctx does. Never ignore it.
			if err := stream.Send(&pb.Token{Text: tok, Index: int32(i)}); err != nil {
				return err
			}
		}
	}
	return stream.Send(&pb.Token{Index: int32(max), FinishReason: "length"})
}
```

## Client

```go
func Run(ctx context.Context, c pb.InferenceClient, prompt string) (string, error) {
	ctx, cancel := context.WithTimeout(ctx, 60*time.Second)
	defer cancel() // cancel() here is what tells the SERVER to stop, via HTTP/2 RST_STREAM

	stream, err := c.Generate(ctx, &pb.GenerateRequest{Prompt: prompt, MaxTokens: 256})
	if err != nil {
		return "", err
	}

	var sb strings.Builder
	var first time.Duration
	start := time.Now()

	for {
		tok, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			break // clean end of stream, NOT an error
		}
		if err != nil {
			return sb.String(), err // partial output plus the reason it stopped
		}
		if first == 0 {
			first = time.Since(start) // TTFT: the number users actually feel
		}
		sb.WriteString(tok.Text)
		if tok.FinishReason != "" {
			break
		}
	}
	return sb.String(), nil
}
```

## Notes

**Why streaming at all.** A 500-token response at 50 tok/s takes 10 seconds. Unary returns
everything at t=10s. Streaming returns the first token at t=0.2s. Identical total time,
completely different perceived latency — and it is the reason every chat API is streaming.
This also changes what you measure: see [07-observability.md](07-observability.md).

**Two disconnect signals, not one.** `ctx.Done()` fires on RST_STREAM or transport loss;
`Send` returns an error when the write fails. Handle both. A loop that only checks `ctx`
can spin on a half-closed connection; a loop that only checks `Send` may not notice
cancellation while blocked waiting for the next token from the engine.

**`finish_reason` rather than a sentinel token.** A magic string like `"<|end|>"` is
indistinguishable from a model that legitimately emits it, and it forces every client to
string-match. A field is typed, extensible (`content_filter`, `tool_calls`), and survives
the addition of a new stop reason without breaking old clients.

**Status codes, not in-band errors.** Mid-stream failure returns a gRPC status. Clients get
`codes.ResourceExhausted` (retry elsewhere), `codes.DeadlineExceeded` (you were slow), or
`codes.Internal` (do not retry blindly). Encoding this as a token forces every client to
parse text to decide whether to retry.

**Flow control.** HTTP/2 has per-stream windows, so a slow client eventually makes `Send`
block, which propagates backpressure to the engine — usually what you want. If you would
rather drop a slow consumer than let it hold a GPU slot, put a small buffered channel
between engine and `Send`, and abort with `codes.ResourceExhausted` when it fills. Say
which you chose.

## The bidirectional stretch

Bidi requires a second goroutine reading `stream.Recv()` while the first writes, and now
you have shared generation state under concurrent access, ordering questions about when
mid-stream input takes effect, and no clean point to release KV. Most production inference
APIs are server-streaming only for exactly this reason — the complexity is real and the
use case is rare.
