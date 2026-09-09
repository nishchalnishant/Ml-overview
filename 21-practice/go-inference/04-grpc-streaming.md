---
module: Practice
topic: Go for Inference
subtopic: gRPC Streaming
status: unread
tags: [practice, go, grpc, streaming, protobuf]
---
# Exercise 4 — Streaming gRPC Token API

**Time:** 2 hours. **Solution:** [solutions/04-grpc-streaming.md](solutions/04-grpc-streaming.md)

## Build

A gRPC service that streams generated tokens, mirroring how a real inference server
returns output.

```protobuf
service Inference {
  rpc Generate(GenerateRequest) returns (stream Token);
}
```

The "model" can emit words from a fixed sentence with a delay per token.

## Requirements

1. Server-streaming RPC that emits tokens one at a time, not batched at the end.
2. Each `Token` carries text, index, and a `finish_reason` on the last message
   (`stop` / `length` / `error`).
3. `max_tokens` in the request is enforced.
4. A client disconnect stops generation — do not keep producing into a dead stream.
5. Errors mid-stream use proper gRPC status codes, not a magic token value.
6. A client that reads the stream and reassembles the output.

## What this is testing

Why streaming exists at all: TTFT is what users perceive, and a unary RPC that returns
after the full generation makes a fast model feel slow. Requirement 4 is the
production-critical one — on a real server, continuing to generate for a disconnected
client burns GPU and holds KV cache blocks that other requests need. This is the same
concern as mock 4 stage 2 in
[../system-design/02-llm-serving-mocks.md](../system-design/02-llm-serving-mocks.md).

## Traps

- The stream's `Send` returns an error when the client is gone. Check it every time and
  return immediately; ignoring it is exactly the bug in requirement 4.
- `stream.Context()` is cancelled on disconnect. That is your signal, and it is what you
  should be selecting on.
- Flow control: a slow client makes `Send` block. Decide what you want — block, buffer, or
  drop — and be able to defend it.

## Stretch

Make it bidirectional streaming so the client can send additional input mid-generation.
Notice how much harder the concurrency gets, and why most inference APIs do not do this.
