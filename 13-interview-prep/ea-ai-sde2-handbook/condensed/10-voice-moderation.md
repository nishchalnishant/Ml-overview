# Interview 10 — Real-Time Voice Toxicity Moderation (Condensed)

Design a real-time system that listens to VoIP voice chat (e.g. Apex Legends), detects hate speech/harassment, and mutes the offending player within seconds — building on an existing text-moderation system.

## Clarifying Questions to Ask
- Process 100% of voice traffic? → No — only reported players or low Trust Score players (cost prohibitive otherwise).
- Latency SLA? → Mute within 3–5 seconds of the utterance.
- STT-then-text, or audio-native model? → STT (Whisper/Vosk) → existing text toxicity model for v1.
- Can we store raw audio? → No persistent storage unless reported (GDPR/CCPA); in-memory only otherwise.
- Overlapping voices / diarization needed? → No — client sends per-player separate audio streams.

## Core Architecture
```
Game Client (opus, 200ms chunks) → Voice Gateway (gRPC/WebSocket, sticky sessions)
   → Streaming STT (Vosk/Kaldi or streaming Whisper) → partial transcripts
   → Existing Text Toxicity Model (score every ~1s) → if >0.95
   → Action Service → mute webhook to game server
```
- Streaming STT is the key ML choice — batch STT (wait-for-silence → WAV → API) blows the 3–5s SLA for continuous talkers.
- VAD (WebRTC VAD) gates audio before STT — drop silence/breathing chunks, cuts STT cost by orders of magnitude.
- Sticky-session load balancing required — recognizer state (KaldiRecognizer) lives in-memory per connection/pod.
- Fail-open on toxicity API failure — never block voice chat over a moderation outage.

## Talking Points That Signal Seniority
- Proactively flags that transcribing 100% of voice is 100x costlier than text — proposes VAD + selective/triggered monitoring instead of blanket coverage.
- States streaming (not batch) STT is mandatory given the SLA, and explains why batch fails for continuous speakers.
- Raises accent/dialect bias in STT as a false-positive risk — proposes never auto-banning off voice alone, only auto-muting current match + human review before long bans.
- Designs a GDPR-safe ephemeral ring buffer (e.g., Redis, last 60s) that only persists audio when a violation actually triggers.
- Notices partial-result evaluation spams the toxicity API and proposes throttling by word-delta/time plus prefix-score caching.
- Suggests audio-native tone/emotion analysis as a v2 check to catch "friendly banter" false positives (text says threat, tone says laughing).
- Mentions speaker verification to prevent "sibling grabbed the mic" ban-appeal disputes.
- Distinguishes CPU (Vosk, linear/easy K8s scaling) vs GPU (Whisper, needs dynamic cross-player batching — hard for streaming).

## Top 3 Tradeoffs
- STT+text pipeline vs audio-native classifier: text loses tone/sarcasm signal but reuses existing model; audio-native captures tone but needs scarce labeled hate-speech audio.
- Partial-result evaluation vs waiting for full sentence: partials hit the SLA but multiply API calls; full-sentence waits reduce load but miss the SLA for continuous talkers.
- Client-side (edge) STT vs cloud STT: edge is cheap and private but trivially bypassed by a hacked client; cloud is tamper-resistant but expensive at scale.

## Toughest Follow-ups
**Q: Partial transcripts ("You"→"You are"→"You are a"...) generate 4x API calls per sentence — this crashes the text API at scale. Fix it?**
A: Throttle by delta, not by every partial — only re-score when the transcript grows by ~3+ words or 2 seconds have elapsed. Cache the previously-scored prefix and only evaluate the new suffix/delta, avoiding full re-scoring of unchanged text.

**Q: GDPR requires immediate audio deletion, but muted players will appeal and CS needs to hear the clip. How do you reconcile?**
A: Keep only a rolling ephemeral in-memory/Redis ring buffer (e.g., last 60s). On a mute trigger, dump just that window to a secure S3 bucket with a 30-day lifecycle for CS review; if no violation occurs, audio never touches disk — compliance by construction, not policy.

**Q: A player speaks very slowly, breaking the STT's chunking so it emits meaningless fragments that dodge the toxicity filter. Fix?**
A: The VAD's silence/hangover threshold is too aggressive, cutting the utterance mid-sentence. Increase the "end-of-utterance" silence padding (e.g., 1.5s of pure silence) so slow cadences stitch back into one string before scoring.

## Biggest Pitfall
Proposing to send an LLM (or any heavyweight audio-in model) over raw audio for every player's full voice stream, ignoring cost/bandwidth/SLA reality — this alone tanks the interview to No Hire regardless of other strengths.
