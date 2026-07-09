# Interview 10 — Real-Time Voice Toxicity Moderation
**EA SDE-2 AI Engineer · Estimated Duration: 75 minutes**

---

## Part 1 — Problem Statement

You are an AI Engineer on the Trust & Safety team. We already have a text chat moderation system (Interview 03), but players are increasingly using voice chat (VoIP) in games like Apex Legends. Voice toxicity is a massive problem.

Your task is to **design and implement a real-time Voice Toxicity Moderation system** that detects hate speech or extreme harassment in voice chat and mutes the offending player.

---

## Part 2 — Intentionally Missing Information

The following critical details are **deliberately omitted**. A strong candidate will ask about all of them:

- Latency constraints (How fast must we mute them? 1 second? 10 seconds?)
- Audio volume/throughput (Are we processing 100% of VoIP traffic? That is petabytes of audio).
- Privacy/Legal (Can we record and store voice data?)
- STT vs Direct Audio ML (Do we convert to text first, or run ML directly on audio spectrograms?)
- Languages supported.
- Action taken (Mute the player globally, or just warn them?)

---

## Part 3 — Ideal Clarifying Questions

> Interviewer will reveal answers only when directly asked.

1. **"Are we processing 100% of all voice traffic globally?"**
   → *Answer: Good catch. Processing 100% of voice traffic is prohibitively expensive. We only process audio if a player in the lobby hits the 'Report' button, or if the player has a low Trust Score.*

2. **"What is the latency SLA?"**
   → *Answer: Near real-time. If a player is screaming slurs, we want to mute them within 3-5 seconds of the utterance.*

3. **"Do we convert to text (STT) or use audio-native models?"**
   → *Answer: STT (Speech-to-Text) followed by our existing text toxicity model is preferred for v1. Audio-native is an option if you can justify it.*

4. **"Are there privacy constraints on storing the audio?"**
   → *Answer: Yes. We can hold it in memory for processing, but we cannot write raw audio to disk unless the player was reported, due to GDPR/CCPA.*

5. **"How do we handle overlapping voices?"**
   → *Answer: The game client sends separate audio streams for each player. We don't need to do speaker diarization.*

---

## Part 4 — Expected Assumptions

- **Architecture:** Streaming architecture (WebSockets or gRPC streaming).
- **Pipeline:** Game Client -> Audio Stream -> STT Engine (Whisper / Kaldi) -> Text Moderation Model -> Action Engine (Mute API).
- **Cost control:** Audio processing is expensive. We only stream audio for high-risk lobbies or when manually triggered.

---

## Part 5 — High-Level Solution

```
  [Game Client (High-Risk Player)]
       │ (opus encoded audio chunks, 200ms)
       ▼
  [Voice Gateway API (gRPC / WebSockets)]
       │
       ▼
  [STT Service (Streaming)]
  ┌────────────────────────────────────────────────────────┐
  │ Uses a streaming STT model (e.g., Kaldi or Whisper V3) │
  │ Accumulates chunks, yields partial text transcripts    │
  └────────────────────────────────────────────────────────┘
       │ (Text string)
       ▼
  [Toxicity Service (Existing)] ➔ Evaluates text string
       │
       ▼ (if Toxic > 0.95)
  [Action Service] ➔ Calls Game Server to Mute player
```

**Core ML Component:** A streaming Speech-to-Text pipeline. A standard batch STT (waiting for the user to stop speaking, saving a WAV file, and sending it to an API) will fail the 3-5 second SLA if the user speaks continuously for 20 seconds. We must use a streaming STT algorithm.

---

## Part 6 — Step-by-Step Implementation

### Step 1: Ingestion
- The game client records mic audio, encodes it to Opus (highly compressed voice codec), and streams it via UDP/TCP to a central ingestion gateway.
- To save costs, the client only opens this socket if instructed by the server (e.g., triggered by a report).

### Step 2: Streaming STT
- We cannot wait for end-of-utterance.
- We use a streaming STT engine like `Vosk` (Kaldi-based) or a modified streaming `Whisper` implementation (using local attention windows).
- The engine processes incoming audio chunks and emits `partial_results` (e.g., "you are a", "you are a bad", "you are a badword").

### Step 3: Text Moderation & Action
- Send the `partial_results` to the Text Toxicity model every 1 second.
- If the toxicity score crosses the threshold (0.95), fire a webhook to the Game Server to apply a VoIP mute to that player_id.

---

## Part 7 — Complete Python Code

```python
"""
voice_moderation_worker.py - Streaming STT and Toxicity Evaluation
"""
import logging
import json
import asyncio
from typing import AsyncGenerator
import websockets # Simulating a streaming audio connection
from vosk import Model, KaldiRecognizer # Fast, offline, streaming STT
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config & Setup
# ---------------------------------------------------------------------------
STT_MODEL_PATH = "/models/vosk-model-small-en-us"
TOXICITY_API = "http://toxicity-service/v1/moderate"
GAME_SERVER_API = "http://game-orchestrator/v1/mute"

logger.info("Loading STT model into memory...")
model = Model(STT_MODEL_PATH)

# ---------------------------------------------------------------------------
# Core Logic
# ---------------------------------------------------------------------------
async def check_toxicity(text: str, player_id: str) -> bool:
    """Calls the existing text moderation API."""
    if len(text.strip()) < 3:
        return False
        
    try:
        # Real-time HTTP call (could be gRPC in prod)
        resp = requests.post(TOXICITY_API, json={
            "player_id": player_id,
            "game_id": "apex_legends",
            "message": text
        }, timeout=0.1)
        
        data = resp.json()
        return data.get("is_toxic", False)
    except Exception as e:
        logger.error(f"Toxicity API failed: {e}")
        return False

def execute_mute(player_id: str):
    """Triggers the game server to mute the player."""
    logger.warning(f"MUTING PLAYER {player_id} due to voice toxicity.")
    # requests.post(f"{GAME_SERVER_API}/{player_id}")

async def process_audio_stream(websocket, path):
    """
    Handles a single WebSocket connection containing an audio stream 
    from a specific player.
    """
    # In a real game, player_id is extracted from auth headers
    player_id = "player_xyz" 
    
    # Initialize streaming recognizer (16kHz PCM audio)
    rec = KaldiRecognizer(model, 16000)
    
    logger.info(f"Started audio stream for {player_id}")
    
    try:
        async for audio_chunk in websocket:
            # rec.AcceptWaveform returns True if silence is detected (full phrase)
            # returns False if it's still processing a continuous phrase
            if rec.AcceptWaveform(audio_chunk):
                # Full result (user paused speaking)
                res = json.loads(rec.Result())
                text = res.get("text", "")
            else:
                # Partial result (user is currently speaking)
                res = json.loads(rec.PartialResult())
                text = res.get("partial", "")
                
            # Evaluate toxicity on the running buffer
            if text:
                is_toxic = await check_toxicity(text, player_id)
                if is_toxic:
                    execute_mute(player_id)
                    await websocket.send("MUTE_TRIGGERED")
                    # Terminate connection as they are muted
                    break
                    
    except websockets.exceptions.ConnectionClosed:
        logger.info(f"Stream closed for {player_id}")
    except Exception as e:
        logger.error(f"Stream error: {e}")

# ---------------------------------------------------------------------------
# Server Entrypoint
# ---------------------------------------------------------------------------
async def main():
    logger.info("Starting Voice Gateway on ws://0.0.0.0:8765")
    async with websockets.serve(process_audio_stream, "0.0.0.0", 8765):
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Part 8 — Deployment

### Scaling Voice Processing
- STT is extremely CPU intensive. A standard CPU core can process roughly 1-2 concurrent realtime streams using Whisper, or ~10 concurrent streams using Vosk.
- If 10,000 players are being monitored, we need **1,000 CPU cores**.
- **Kubernetes:** Deploy as a `DaemonSet` or a highly scaled `Deployment`. HPA must scale aggressively based on active WebSocket connections, NOT just CPU, because WebSocket connections are stateful and long-lived.

### Load Balancing
- WebSockets require sticky sessions. The Load Balancer (e.g., NGINX / Envoy) must route packets for a specific `player_id` to the exact same Pod that holds their `KaldiRecognizer` state in memory.

---

## Part 9 — Unit Testing

```python
import json
from unittest.mock import patch, AsyncMock
import pytest
from voice_moderation_worker import check_toxicity

@pytest.mark.asyncio
@patch('voice_moderation_worker.requests.post')
async def test_check_toxicity_toxic(mock_post):
    # Mock the API response
    mock_post.return_value.json.return_value = {"is_toxic": True}
    
    result = await check_toxicity("some bad words", "p1")
    assert result == True

@pytest.mark.asyncio
async def test_check_toxicity_too_short():
    # Should not hit the API for 1 letter
    result = await check_toxicity("a", "p1")
    assert result == False
```

---

## Part 10 — Integration Testing

- Prepare a `.wav` file containing a known toxic phrase (e.g., 5 seconds long, toxicity happens at second 3).
- Write a Python test script that streams the `.wav` file to the WebSocket endpoint in 200ms chunks (simulating real-time network).
- Assert that the server returns `MUTE_TRIGGERED` at approximately the 3.5-second mark, proving the streaming transcription + moderation loop works within the SLA.

---

## Part 11 — Scaling Discussion

| Axis | Strategy |
|------|----------|
| **Cost of STT** | Transcribing audio is 100x more expensive than text moderation. We must reduce the funnel. Use VAD (Voice Activity Detection). WebRTC VAD runs in microseconds. If a 200ms chunk contains only silence or breathing, drop it. Do not send it to the STT model. |
| **GPU vs CPU** | Whisper V3 runs best on GPUs. If using Whisper, we must batch audio chunks across multiple players dynamically. Batching streaming audio is notoriously difficult. If using CPU (Vosk), scaling is linear and easier to manage in Kubernetes. |

---

## Part 12 — Tradeoffs

| Decision | Tradeoff |
|----------|----------|
| STT Pipeline vs Audio-Native Model | Converting to text loses tone (e.g., sarcastic vs aggressive). An Audio-Native classifier (Raw Audio -> Toxicity Score) captures tone and is faster, but is vastly harder to train because you need millions of labeled *audio* clips of hate speech, which are hard to acquire. Text STT leverages our existing text model. |
| Partial Results vs Full Sentences | Evaluating `partial_results` hits the Toxicity API multiple times for the same sentence (increasing API load), but guarantees the 3-second SLA. Waiting for a full sentence reduces API load but fails the SLA if the player talks continuously. |
| Edge (Client) vs Cloud ML | Running STT locally on the player's console saves us millions in cloud costs and ensures privacy. However, cheat devs can easily bypass or disable the client-side model, granting them immunity. Cloud is secure but expensive. |

---

## Part 13 — Alternative Approaches

1. **Client-Side STT:** The game client (using OS-level APIs like Windows Speech Recognition or Apple Speech) transcribes the voice locally. It sends only the *text* to the cloud Toxicity API. Saves 99% of cloud costs. Fails if the user hacks the client to never send text.
2. **Keyword Spotting (KWS):** Instead of full STT (which predicts every word), train a tiny Wake-Word model (like Alexa/Siri) to only listen for the top 50 worst slurs. Very fast, uses almost no CPU. Fails on nuanced harassment.

---

## Part 14 — Failure Scenarios

| Failure | Impact | Mitigation |
|---------|--------|-----------|
| Accent / Dialect Bias | A Scottish player is incorrectly transcribed as saying a slur and gets banned. | False positives in Voice STT are notoriously high for non-American accents. Never auto-ban based on voice. Auto-mute for the current match, but flag the audio for human review before applying a 30-day ban. |
| Background Noise | Game sounds (gunfire) or music break the STT. | Apply DSP noise-cancellation (e.g., RNNoise) *before* the VAD and STT steps. |
| Toxicity API Crash | System cannot evaluate text | Fail open. Allow voice chat to continue. |

---

## Part 15 — Debugging

**Symptom:** The system is muting players incorrectly. Logs show the STT engine is transcribing random letters like "a a a h g g g" and the Text Model thinks it's toxic.

**Debugging steps:**
1. Check the audio encoding. Did the game client update to 48kHz audio, but the KaldiRecognizer is hardcoded to expect 16kHz? Sample rate mismatch destroys STT models.
2. Check the VAD. Is the system sending static noise to the STT model?
3. Update the STT initialization code to dynamically read the sample rate from the WebSocket headers.

---

## Part 16 — Monitoring

| Metric | Alert Threshold |
|--------|----------------|
| `audio_processing_latency_ms` | > 1000ms → STT engine is falling behind realtime |
| `vad_drop_rate` | < 10% → Warning (Are we processing silence?) |
| `stt_cpu_utilization` | > 85% → Scale out HPA immediately |
| `false_positive_appeals` | Monitor CS tickets. High volume means STT bias. |

---

## Part 17 — Production Improvements

1. **Speaker Verification:** Ensure the voice matches the account owner to prevent "My little brother screamed into the mic" ban appeals.
2. **Audio-Native Tone Analysis:** Run a parallel lightweight classifier that predicts the speaker's emotional state (Angry vs Happy) based on pitch and volume. If the text model says "I'm going to kill you" but the emotional model says "Happy/Laughing", override the mute (it's friendly banter).

---

## Part 18 — Follow-up Questions

> *Interviewer asks these after the initial solution is presented.*

1. **"The Toxicity API evaluates partial sentences. The STT engine outputs: 'You', then 'You are', then 'You are a', then 'You are a badword'. That's 4 API calls for one sentence. At scale, this crashes our Text API. How do you optimize this?"**
2. **"To comply with GDPR, we must delete audio immediately. However, if a player is muted, they will appeal the ban. Customer Support needs to hear the audio to verify. How do you design a compliant storage pipeline?"**
3. **"A player figures out that speaking very, very slowly breaks the chunking logic of the streaming STT, causing it to output meaningless half-words that bypass the toxicity filter. How do you fix this?"**

---

## Part 19 — Ideal Answers

**Q1 (Partial Result Spam):**
> "We shouldn't send every partial update. We should throttle based on token count or time. For example, only send a string to the Toxicity API if it has grown by at least 3 words since the last check, OR if 2 seconds have passed. Furthermore, we can cache the prefix. If 'You are a' was already scored as 0.1, we only evaluate the delta."

**Q2 (GDPR and Appeals):**
> "We use an ephemeral ring buffer (e.g., Redis expiring keys, or an in-memory buffer) that holds exactly the last 60 seconds of audio. If the ML system triggers a mute, we dump that specific 60-second buffer to a secure S3 bucket with a 30-day lifecycle policy for CS review. If no mute is triggered, the audio naturally drops out of the ring buffer and is never written to disk, ensuring GDPR compliance by only storing data tied to a specific policy violation."

**Q3 (Slow Speech Bypass):**
> "The STT engine's Acoustic Model relies on standard phonetic timing. If they speak too slowly, the VAD might classify the gaps between words as silence, cutting the sentence into fragments (e.g., 'bad' ... 'word'). We need to tune the VAD's `padding` or `hangover` threshold—forcing it to wait at least 1.5 seconds of pure silence before declaring an end-of-utterance. This stitches the slow words back into a single string for the Text model."

---

## Part 20 — Evaluation Rubric

### Strong Hire
- Anticipates the cost problem of 100% audio processing and suggests selective monitoring or client-side VAD.
- Correctly identifies that a Streaming STT (not batch) is mandatory for the SLA.
- Provides a robust solution for the partial-results API spam (Q1).
- Answers the GDPR ring-buffer question perfectly.

### Hire
- Sets up a solid STT -> Text Pipeline.
- Uses WebSockets to handle the streaming audio.
- Understands that CPU scaling for STT is different from normal REST APIs.
- Code is mostly complete, missing only some async optimization details.

### Lean Hire
- Suggests waiting for the player to stop speaking, saving a `.wav` file, and sending it to an API. (Requires prompting to realize this fails the 3-5 second SLA).
- Struggles with the concept of `partial_results`.

### Lean No Hire
- Suggests using an LLM to analyze the audio directly (too slow, too expensive).
- Ignores the cost and bandwidth implications of sending 60Hz audio for 50 million players.

### No Hire
- Fails to design a pipeline.
- Does not understand what STT is.
