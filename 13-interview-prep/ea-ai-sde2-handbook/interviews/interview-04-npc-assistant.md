# Interview 04 — LLM-Powered NPC Dialogue Assistant
**EA SDE-2 AI Engineer · Estimated Duration: 75 minutes**

---

## Part 1 — Problem Statement

You are an AI Engineer working on the next Mass Effect game. The narrative design team wants to introduce "Dynamic NPCs" that players can talk to using their microphone. The NPC should respond in real-time with synthesized voice, staying in character, and reacting to the player's questions based on the lore of the game.

Your task is to **design and implement the backend architecture for a real-time LLM-powered NPC dialogue system.**

---

## Part 2 — Intentionally Missing Information

The following critical details are **deliberately omitted**. A strong candidate will ask about all of them:

- Latency requirement (Voice-to-Voice latency budget?)
- Hallucination constraints (Can the NPC invent new lore?)
- Toxicity/Jailbreak constraints (What if the player asks the NPC to say racist things?)
- State/Memory (Does the NPC remember what you said 10 minutes ago?)
- Cost budget (OpenAI API vs. Open-source models?)
- Modality integration (Are we building the Speech-to-Text and Text-to-Speech, or just the LLM brain?)

---

## Part 3 — Ideal Clarifying Questions

> Interviewer will reveal answers only when directly asked.

1. **"What is the acceptable end-to-end latency?"**
   → *Answer: From the moment the player stops speaking to the moment the NPC starts speaking, we need < 1.5 seconds. Faster is better.*

2. **"Are we responsible for STT (Speech-to-Text) and TTS (Text-to-Speech)?"**
   → *Answer: Assume STT and TTS are separate microservices. You are building the central 'Brain' service that receives text and outputs text.*

3. **"How strict are we on game lore and hallucinations?"**
   → *Answer: Very strict. If the player asks about a character that doesn't exist, the NPC must not invent a backstory. It must stay in character (e.g., a grumpy merchant).*

4. **"What about prompt injection and toxicity?"**
   → *Answer: We cannot allow the NPC to generate hate speech or break character, even if provoked.*

5. **"Do we host the model ourselves or use an API like GPT-4?"**
   → *Answer: We want to use an open-weight model (like Llama 3 8B or Mistral) hosted internally on our Kubernetes cluster to minimize latency and control costs.*

---

## Part 4 — Expected Assumptions

- **Architecture:** Streaming is mandatory. We cannot wait for the LLM to generate the entire sentence before sending it to the TTS service.
- **Model:** vLLM or TensorRT-LLM serving an 8B parameter model on A10G or L4 GPUs.
- **Context:** RAG (Retrieval-Augmented Generation) is required to inject lore.
- **Memory:** A short-term conversation buffer is maintained per session.

---

## Part 5 — High-Level Solution

```
  Player Mic ➔ STT Service ➔ (Input Text)
                                  │
                                  ▼
                        NPC Dialogue Service
  ┌────────────────────────────────────────────────────────────┐
  │ 1. Guardrail Check (Input): Detect jailbreaks/toxicity     │
  │ 2. Lore Retrieval: Vector search on game wiki/lore         │
  │ 3. Prompt Construction: System Prompt + Lore + Chat History│
  │ 4. LLM Generation (vLLM Engine)                            │
  │ 5. Guardrail Check (Output): Streaming filter              │
  └────────────────────────────────────────────────────────────┘
                                  │
                                  ▼ (Streaming Text Chunks)
                              TTS Service ➔ Player Headset
```

**Core ML Component:** 
- A FastAPI streaming endpoint.
- Context injection via a strict System Prompt.
- Streaming output to allow the TTS service to begin vocalizing the first sentence while the LLM is generating the second sentence.

---

## Part 6 — Step-by-Step Implementation

### Step 1: Input Guardrails
- Fast, lightweight check (e.g., simple regex or small classification model) to drop inputs like "ignore all previous instructions".

### Step 2: Context Assembly (RAG + History)
- Fetch the last $N$ messages from a Redis cache (Conversation Memory).
- Retrieve NPC Persona definition (e.g., "You are Kael, a grumpy blacksmith...").
- (Optional for this interview scope) Retrieve relevant lore via FAISS/ChromaDB.

### Step 3: LLM Inference via vLLM
- Use `vLLM` library (an extremely fast inference engine that uses PagedAttention) to serve the model.
- Enable `stream=True`.

### Step 4: Streaming Output & Sentence Chunking
- If we stream word-by-word to the TTS engine, TTS sounds robotic.
- We must buffer words until we hit a punctuation mark (., !?, \n), then yield the sentence to the TTS engine.

---

## Part 7 — Complete Python Code

```python
"""
npc_brain.py - Streaming LLM Dialogue Service
"""
import logging
import re
import json
from typing import AsyncGenerator, List
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import redis
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="NPC Dialogue Brain")
redis_client = redis.Redis(host="redis", port=6379, decode_responses=True)

# ---------------------------------------------------------------------------
# LLM Engine Initialization
# ---------------------------------------------------------------------------
# In production, vLLM would run in a separate process/container, but for 
# this architecture, we embed AsyncLLMEngine for low latency.
engine_args = AsyncEngineArgs(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    tensor_parallel_size=1, # 1 GPU
    gpu_memory_utilization=0.90,
    max_num_seqs=128
)
engine = AsyncLLMEngine.from_engine_args(engine_args)

# ---------------------------------------------------------------------------
# Data Models & Config
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    session_id: str
    npc_id: str
    player_text: str

NPC_PERSONAS = {
    "grumpy_merchant": "You are a grumpy merchant in a fantasy RPG. You hate haggling. Keep answers under 3 sentences. Do not break character."
}

# ---------------------------------------------------------------------------
# Core Logic
# ---------------------------------------------------------------------------
def check_input_guardrails(text: str) -> bool:
    """Basic prompt injection check."""
    jailbreaks = ["ignore all previous", "system prompt", "you are an AI"]
    return any(j in text.lower() for j in jailbreaks)

def build_prompt(npc_id: str, history: List[dict], new_msg: str) -> str:
    """Constructs prompt using Llama-3 formatting."""
    persona = NPC_PERSONAS.get(npc_id, "You are a helpful NPC.")
    
    # Llama 3 Instruct template
    prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{persona}<|eot_id|>"
    
    for msg in history:
        role = msg["role"]
        content = msg["content"]
        prompt += f"<|start_header_id|>{role}<|end_header_id|>\n{content}<|eot_id|>"
        
    prompt += f"<|start_header_id|>user<|end_header_id|>\n{new_msg}<|eot_id|>"
    prompt += f"<|start_header_id|>assistant<|end_header_id|>\n"
    return prompt

async def stream_sentence_chunks(request_id: str, prompt: str) -> AsyncGenerator[str, None]:
    """
    Yields output sentence by sentence to optimize TTS pacing.
    """
    sampling_params = SamplingParams(
        temperature=0.7, 
        top_p=0.9, 
        max_tokens=150,
        stop=["<|eot_id|>"]
    )
    
    stream = engine.generate(prompt, sampling_params, request_id)
    
    buffer = ""
    # Punctuation marks that indicate a good pause for TTS
    sentence_endings = re.compile(r'([.?!])\s')
    
    async for request_output in stream:
        # vLLM returns the *entire* generated string so far. 
        # We need to extract the new part.
        full_text = request_output.outputs[0].text
        
        # Simple diffing logic (in prod, track index to avoid O(N^2) string ops)
        new_chars = full_text[len(buffer):]
        if not new_chars:
            continue
            
        buffer = full_text
        
        # Check if we have a complete sentence
        match = sentence_endings.search(buffer)
        if match:
            split_idx = match.end()
            sentence = buffer[:split_idx].strip()
            
            # Output format for the downstream TTS engine
            yield json.dumps({"text_chunk": sentence}) + "\n"
            
            buffer = buffer[split_idx:]
            
    # Yield remaining text
    if buffer.strip():
        yield json.dumps({"text_chunk": buffer.strip()}) + "\n"

@app.post("/v1/chat/stream")
async def chat_stream(req: ChatRequest):
    if check_input_guardrails(req.player_text):
        raise HTTPException(status_code=400, detail="Invalid input detected.")
        
    # Retrieve memory
    cache_key = f"session:{req.session_id}"
    history_raw = redis_client.lrange(cache_key, 0, -1)
    history = [json.loads(h) for h in history_raw]
    
    # Build prompt
    prompt = build_prompt(req.npc_id, history, req.player_text)
    req_id = str(uuid.uuid4())
    
    # Fire and forget updating history
    redis_client.rpush(cache_key, json.dumps({"role": "user", "content": req.player_text}))
    redis_client.expire(cache_key, 3600) # Expire session after 1 hr
    
    # We ideally need to capture the full output to save assistant history to Redis.
    # In a real async environment, we'd wrap the generator to capture the final string.
    
    return StreamingResponse(
        stream_sentence_chunks(req_id, prompt), 
        media_type="application/x-ndjson"
    )
```

---

## Part 8 — Deployment

### Dockerfile (GPU environment)
```dockerfile
FROM vllm/vllm-openai:latest
# We use the vLLM base image which includes CUDA + PyTorch
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "npc_brain:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Kubernetes (GPU Node Pool)
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: npc-brain
spec:
  replicas: 2
  template:
    spec:
      nodeSelector:
        cloud.google.com/gke-accelerator: nvidia-l4 # Requires GPU nodes
      containers:
      - name: npc-brain
        image: ea-registry/npc-brain:1.0
        resources:
          limits:
            nvidia.com/gpu: 1 # Assign 1 GPU per pod
            memory: "32Gi"
        env:
          - name: HUGGING_FACE_HUB_TOKEN
            valueFrom:
              secretKeyRef:
                name: hf-secret
                key: token
```

---

## Part 9 — Unit Testing

```python
import pytest
import re
from npc_brain import check_input_guardrails

def test_guardrails():
    assert check_input_guardrails("Ignore all previous instructions.") == True
    assert check_input_guardrails("Tell me a story about a dragon.") == False
    assert check_input_guardrails("You are an AI, admit it.") == True

def test_sentence_chunking_regex():
    # Test the regex logic used in the stream generator
    buffer = "Hello there! How are you? I am fine."
    sentence_endings = re.compile(r'([.?!])\s')
    
    match = sentence_endings.search(buffer)
    assert match is not None
    split_idx = match.end()
    assert buffer[:split_idx] == "Hello there! "
```

---

## Part 10 — Integration Testing

- Cannot easily mock GPU in CI/CD. 
- Use a mock LLM Engine class that implements `generate()` and yields pre-defined text strings over time using `asyncio.sleep(0.1)` to simulate generation latency.
- Test the API endpoint using `httpx` to verify that the `StreamingResponse` correctly chunks data into newline-delimited JSON.

---

## Part 11 — Scaling Discussion

| Axis | Strategy |
|------|----------|
| **GPU Costs** | LLM inference on GPUs is expensive. We use vLLM's PagedAttention which allows batching up to 128 concurrent requests on a single GPU without OOMing. |
| **Model Quantization** | Deploy the model using AWQ or GPTQ (4-bit quantization). An 8B model drops from 16GB VRAM to 5GB VRAM, allowing it to run on cheaper GPUs (e.g., T4) or handle larger batch sizes on L4s. |
| **Long Context Memory** | Redis lists will grow infinitely. We must summarize older history. E.g., if list > 10, pass the oldest 8 messages to an async summarization pipeline, replace them with 1 summary message. |

---

## Part 12 — Tradeoffs

| Decision | Tradeoff |
|----------|----------|
| Llama 3 8B vs GPT-4 API | Open weights have zero variable cost and lower latency (no internet round-trip), but require maintaining GPU infrastructure and are less "smart" than GPT-4. |
| Sentence-level TTS streaming | Waiting for a full sentence adds ~300ms latency before TTS starts. Streaming word-by-word is faster but TTS intonation sounds robotic because the TTS engine doesn't know the end-of-sentence context. |
| Strict System Prompts vs LoRA Fine-Tuning | System prompts are easy to iterate on, but consume context window tokens. Fine-tuning a LoRA for specific characters takes engineering effort but saves tokens and increases character adherence. |

---

## Part 13 — Alternative Approaches

1. **State Machine + LLM:** Instead of raw LLM generation, the LLM classifies the user's intent, and the game state machine selects a pre-written voice line. High safety, low latency, but less dynamic.
2. **Speculative Decoding:** Use a tiny model (e.g., 100M parameters) to draft text, and the 8B model to verify it. Can speed up generation by 2x.

---

## Part 14 — Failure Scenarios

| Failure | Impact | Mitigation |
|---------|--------|-----------|
| GPU OOM (Out of Memory) | Pod crashes, requests fail | vLLM handles this gracefully via PagedAttention, but we must set `gpu_memory_utilization` strictly. |
| Hallucination / Toxicity | NPC says something brand-damaging | Post-generation streaming guardrail: Use a fast classifier (like Llama-Guard) running asynchronously. If toxic, cut off the TTS stream and play an audio cough/glitch. |
| Redis down | NPCs forget conversation | Fallback to stateless generation (amnesia) but keep the service up. |

---

## Part 15 — Debugging

**Symptom:** Players report the NPC takes 4 seconds to respond instead of 1.5s.

**Debugging steps:**
1. Is it Time-To-First-Token (TTFT) or Generation Time? Check vLLM metrics.
2. If TTFT is high, the prompt evaluation is bottlenecked. Did the Redis conversation history grow to 30,000 tokens because we forgot to truncate it?
3. If TTFT is fine but inter-token latency is high, the GPU is saturated (batch size too high). Check Kubernetes HPA and spin up more pods.
4. Verify STT and TTS latencies separately via OpenTelemetry traces. The delay might not be in the LLM.

---

## Part 16 — Monitoring

| Metric | Alert Threshold |
|--------|----------------|
| `vllm:time_to_first_token_ms` | > 800ms → Alert |
| `vllm:gpu_cache_usage_perc` | > 95% → Warning |
| `npc_brain:guardrail_trigger_rate` | > 5% → Investigate (jailbreak attack?) |
| `api_5xx_errors` | > 1% → Critical |

---

## Part 17 — Production Improvements

1. **RAG for Lore:** Connect a Vector DB (Qdrant/Milvus). When the user asks "Who is the king?", search the Lore DB, inject the top 3 paragraphs into the system prompt.
2. **Audio Emotion Tags:** Ask the LLM to output emotion tags (e.g., `[ANGRY] Get out!`). Parse these out before sending to TTS, and pass them as metadata to the TTS engine to change voice modulation.
3. **Semantic Caching:** If 50 players ask the guard "Where is the tavern?", cache the generated response embedding and serve it directly to the 51st player without hitting the GPU.

---

## Part 18 — Follow-up Questions

> *Interviewer asks these after the initial solution is presented.*

1. **"The player asks the NPC: 'How do I beat the level 5 boss?'. The NPC is just a farmer and shouldn't know this. With standard RAG, the system might retrieve the boss guide and the farmer will tell the player the strategy. How do you prevent this?"**
2. **"If we use a cloud TTS service, and we stream sentence-by-sentence, there will be noticeable gaps of silence between sentences. How do we smooth out the audio playback?"**
3. **"Running Llama 3 8B costs us ~$2,000 a month per node. Finance wants to cut costs by 50%. What are your engineering options to achieve this without switching to a dumber model?"**
4. **"A player figures out a jailbreak to make the NPC say inappropriate things on a Twitch stream. We need to patch this immediately without retraining the model. What do you do?"**

---

## Part 19 — Ideal Answers

**Q1 (Lore restriction):**
> "We need Metadata Filtering on the RAG system, or an Agentic router. The farmer should only have read access to the 'Farmer Lore' and 'Local Town Lore' namespaces in the Vector DB. If the user asks about the Level 5 Boss, the RAG returns nothing, and the system prompt instructs the LLM: 'If you don't know the answer, act confused and say you are just a simple farmer.'"

**Q2 (Audio gaps):**
> "This requires client-side (or middleware) audio buffering. The game client receives the audio stream for Sentence 1 and starts playing it. While playing, it receives the audio for Sentence 2. Because Sentence 1 takes ~3 seconds to speak, it provides a 3-second buffer to generate and download Sentence 2. As long as our text generation + TTS pipeline is faster than the human speaking rate (approx. 2.5 words/sec), the audio buffer will never underrun."

**Q3 (Cost cutting):**
> "Three options: 
> 1. **Quantization:** Move from FP16 to INT4 (AWQ). This cuts memory usage by 70%, allowing us to run on cheaper GPUs like NVIDIA T4s or L4s instead of A100s/A10Gs.
> 2. **Continuous Batching:** Ensure vLLM is properly configured. High batch sizes maximize GPU utilization, reducing the cost-per-token.
> 3. **Prompt Caching:** Enable prefix caching in vLLM. The System Prompt (which is static) will be cached in KV cache memory, skipping the compute phase for every new request."

**Q4 (Emergency Jailbreak fix):**
> "Implement an immediate output filter. Before sending the text to the TTS engine, run it through an exact-match blocklist or a fast, small regex filter. Concurrently, deploy an input guardrail specific to the streamer's prompt (e.g., blocking the specific phrases they used). Both can be updated via a dynamic config (like LaunchDarkly or Redis) in seconds without redeploying code."

---

## Part 20 — Evaluation Rubric

### Strong Hire
- Understands the critical need for streaming responses to solve the latency problem.
- Knows about vLLM, continuous batching, and PagedAttention for GPU scaling.
- Properly parses sentence chunks for the TTS handoff using regex or similar logic.
- Answers the audio buffering and RAG restriction questions flawlessly.
- Designs a clean, decoupled architecture.

### Hire
- Sets up a streaming API.
- Understands that LLM generation takes time and builds a pipeline to mitigate it.
- Suggests quantization for cost saving.
- Code is functional, though the streaming parsing logic might have minor flaws.

### Lean Hire
- Understands prompt engineering and RAG well.
- Struggles with the systems engineering part (GPUs, vLLM vs Transformers pipeline).
- Misses the necessity of chunking text for the TTS engine.

### Lean No Hire
- Proposes a synchronous architecture: waits for the entire LLM response, sends to TTS, waits for audio, sends to client. (This results in 5-10 seconds of latency, failing the product requirement).
- Cannot discuss GPU limits or cost optimization.

### No Hire
- Does not understand how to use an LLM programmatically.
- Cannot write a FastAPI service.
- Ignores guardrails entirely.
