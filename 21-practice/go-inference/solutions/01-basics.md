---
module: Practice
topic: Go for Inference
subtopic: Basics Solution
status: unread
tags: [practice, go, basics, solutions]
---
# Solution 1 — Token Counter Service

```go
package main

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"
)

// Tokenizer is declared here, in the package that CONSUMES it, not in a package
// that provides implementations. That is the Go convention and it is why the
// handler never has to import a tokenizer package.
type Tokenizer interface {
	Count(text string) (int, error)
}

type charApprox struct{ perToken int }

func (c charApprox) Count(text string) (int, error) {
	if c.perToken <= 0 {
		return 0, errors.New("invalid chars-per-token")
	}
	return (len(text) + c.perToken - 1) / c.perToken, nil
}

type wordSplit struct{}

func (wordSplit) Count(text string) (int, error) { return len(strings.Fields(text)), nil }

type server struct{ tokenizers map[string]Tokenizer }

type countRequest struct {
	Text  *string `json:"text"` // pointer so absent and "" are distinguishable
	Model string  `json:"model"`
}

type countResponse struct {
	Tokens int    `json:"tokens"`
	Model  string `json:"model"`
	Chars  int    `json:"chars"`
}

func writeErr(w http.ResponseWriter, code int, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(map[string]string{"error": msg})
}

func (s *server) handleCount(w http.ResponseWriter, r *http.Request) {
	// MaxBytesReader caps the read at the source. Checking Content-Length is not
	// enough: it is client-supplied and a chunked body omits it entirely.
	r.Body = http.MaxBytesReader(w, r.Body, 1<<20)

	var req countRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		var mbe *http.MaxBytesError
		if errors.As(err, &mbe) {
			writeErr(w, http.StatusRequestEntityTooLarge, "body exceeds 1MB")
			return
		}
		writeErr(w, http.StatusBadRequest, "malformed JSON")
		return
	}
	if req.Text == nil {
		writeErr(w, http.StatusBadRequest, "field 'text' is required")
		return
	}
	tok, ok := s.tokenizers[req.Model]
	if !ok {
		writeErr(w, http.StatusBadRequest, "unknown model: "+req.Model)
		return
	}
	n, err := tok.Count(*req.Text)
	if err != nil {
		writeErr(w, http.StatusInternalServerError, "tokenization failed")
		return
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(countResponse{Tokens: n, Model: req.Model, Chars: len(*req.Text)})
}

func main() {
	s := &server{tokenizers: map[string]Tokenizer{
		"gpt-4":      charApprox{perToken: 4},
		"word-count": wordSplit{},
	}}

	mux := http.NewServeMux()
	mux.HandleFunc("POST /count", s.handleCount)

	// Timeouts are not optional. A server with no ReadTimeout can be held open
	// forever by a client that opens a connection and sends one byte per minute.
	srv := &http.Server{
		Addr:              ":8080",
		Handler:           mux,
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       15 * time.Second,
		WriteTimeout:      30 * time.Second,
		IdleTimeout:       60 * time.Second,
	}

	go func() {
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			panic(err)
		}
	}()

	stop := make(chan os.Signal, 1) // buffered: signal.Notify never blocks, so an
	signal.Notify(stop, syscall.SIGTERM, syscall.SIGINT) // unbuffered chan drops it
	<-stop

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	_ = srv.Shutdown(ctx) // stops listeners, waits for in-flight handlers
}
```

## Notes

**Why the `*string`.** `json.Decode` leaves a missing field at its zero value, so a
plain `string` cannot distinguish `{"text": ""}` (valid, answer 0) from `{}` (a 400).
A pointer is the idiomatic fix and this distinction shows up constantly in real APIs.

**Why `MaxBytesReader` and not a length check.** The point is to bound memory before it
is allocated. `Content-Length` is advisory; a chunked or lying client bypasses it.

**Why graceful shutdown matters here.** Kubernetes sends `SIGTERM` and waits
`terminationGracePeriodSeconds` before `SIGKILL`. A server that exits immediately on
`SIGTERM` drops every in-flight request on every deploy — visible to users as a
five-nines-destroying error spike on an otherwise routine rollout. The 10-second cap
must be shorter than the grace period or the cap is meaningless.

**What a reviewer would flag next.** No request ID, no structured logging, no metrics.
Those are exercise 7.
