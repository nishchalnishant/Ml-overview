---
module: Practice
topic: Go for Inference
subtopic: Basics
status: unread
tags: [practice, go, basics]
---
# Exercise 1 — Token Counter Service

**Time:** 60 minutes. **Solution:** [solutions/01-basics.md](solutions/01-basics.md)

## Build

An HTTP service with one endpoint, `POST /count`, taking
`{"text": "...", "model": "..."}` and returning
`{"tokens": 42, "model": "gpt-4", "chars": 180}`.

Tokenization can be a stub — approximate as `len(text)/4` — but structure the code so a
real tokenizer drops in without touching the handler.

## Requirements

1. A `Tokenizer` **interface** with a `Count(text string) (int, error)` method, and two
   implementations: the character-approximation one and a whitespace-splitting one.
2. Model selection picks the tokenizer. An unknown model returns HTTP 400 with a JSON
   error body, not a Go panic and not a plain-text 500.
3. Empty text is valid and returns 0. Missing the `text` field entirely is a 400.
4. Request body larger than 1 MB is rejected with 413 before being read into memory.
5. Graceful shutdown: `SIGTERM` stops accepting new connections and lets in-flight
   requests finish, with a 10-second cap.

## What this is testing

Interfaces defined by the consumer rather than the implementation, errors as return
values rather than exceptions, and the fact that graceful shutdown is not optional in a
service that will be rolled by Kubernetes. Requirement 4 is the one candidates skip, and
it is the one that matters in a data plane — an unbounded read is a memory-exhaustion
vector.

## Stretch

Add a `/count/batch` endpoint taking an array. Notice what you want to do about
concurrency; that is exercise 2.
