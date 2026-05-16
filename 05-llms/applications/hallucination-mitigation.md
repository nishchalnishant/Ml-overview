# Hallucination Mitigation in LLMs

A systematic treatment of why large language models hallucinate, how to detect it, and the full stack of techniques — from decoding constraints to RLHF — used to reduce it.

---

## 1. What is Hallucination

A hallucination occurs when a model generates text that is fluent and plausible-sounding but factually incorrect or unverifiable.

### Taxonomy

| Type | Definition | Example |
|---|---|---|
| **Intrinsic** | Output contradicts the provided source/context | Summarizer reverses a stated date |
| **Extrinsic** | Output adds information absent from or unverifiable against source | Model invents a citation |
| **Faithful hallucination** | Consistent with context but factually wrong in the world | Faithfully summarizes a wrong claim in the document |
| **Factual hallucination** | Contradicts world knowledge | States Einstein was born in France |

### Why It Matters

- Safety-critical domains (medicine, law, finance) cannot tolerate fabricated facts
- User trust degrades rapidly once hallucinations are noticed
- RAG systems can amplify hallucinations if retrieval is noisy

---

## 2. Why LLMs Hallucinate

### Exposure Bias (Teacher Forcing vs Autoregressive Inference)

During training with teacher forcing the model always sees the ground-truth previous token. At inference it sees its own (possibly wrong) previous token. This distribution mismatch accumulates errors over long sequences — the model was never trained to recover from its own mistakes.

```
Training:   P(w_t | w_1*, w_2*, ..., w_{t-1}*)   ← ground truth prefix
Inference:  P(w_t | w_1^, w_2^, ..., w_{t-1}^)   ← model's own output
```

### Training Data Issues

- **Memorization vs. generalization**: Models memorize frequent surface patterns. For rare facts the model interpolates, producing plausible but wrong outputs.
- **Data contamination**: Inconsistent facts across web-scraped sources create conflicting training signal.
- **Sycophancy from RLHF**: Human raters sometimes reward confident, fluent answers — training models to assert rather than hedge.

### Decoding Artifacts

| Decoding Parameter | Effect on Hallucination |
|---|---|
| High temperature | More randomness → more factual drift |
| High top-p (nucleus) | Samples from broader distribution → less grounded tokens |
| Greedy / beam search | Can still hallucinate; repetition and mode collapse |
| Repetition penalty | Forces novel tokens, occasionally off-topic ones |

The softmax over a 100k-token vocabulary means low-probability but plausible tokens are always reachable.

### Knowledge Cutoff

The model has no access to information after its training cutoff. Queries about recent events force the model to extrapolate, which increases fabrication. This is the primary motivation for RAG.

---

## 3. Detection Methods

### 3.1 Self-Consistency

Sample N responses to the same prompt. Measure agreement across responses. High variance on factual claims signals uncertainty.

```python
from collections import Counter

def self_consistency_check(model, prompt, n=10, extract_fn=lambda x: x):
    responses = [model.generate(prompt) for _ in range(n)]
    answers = [extract_fn(r) for r in responses]
    counts = Counter(answers)
    majority, freq = counts.most_common(1)[0]
    confidence = freq / n
    return majority, confidence
```

Works well for closed-form questions (math, dates). Harder for open-ended generation.

---

### 3.2 SelfCheckGPT

**Key idea**: If a statement is factual, re-sampling the model should produce consistent outputs. Inconsistency across samples indicates hallucination — no external knowledge base required.

**Algorithm**:
1. Generate a primary response `r_0`
2. Sample `N` additional responses `{r_1, ..., r_N}` at non-zero temperature
3. For each sentence `s_i` in `r_0`, measure how consistently it appears in `{r_1, ..., r_N}`
4. High inconsistency score → hallucination flag

```python
import numpy as np
from sentence_transformers import SentenceTransformer, util

def selfcheck_gpt(model, prompt, n_samples=5, threshold=0.75):
    """
    Returns per-sentence hallucination scores for the primary response.
    Score close to 1.0 = likely hallucination.
    """
    primary = model.generate(prompt, temperature=0.0)
    samples = [model.generate(prompt, temperature=0.7) for _ in range(n_samples)]

    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    primary_sentences = split_sentences(primary)
    results = []

    for sentence in primary_sentences:
        s_emb = encoder.encode(sentence, convert_to_tensor=True)
        consistencies = []

        for sample in samples:
            sample_sentences = split_sentences(sample)
            if not sample_sentences:
                consistencies.append(0.0)
                continue
            samp_embs = encoder.encode(sample_sentences, convert_to_tensor=True)
            # max cosine similarity of this sentence against any sentence in sample
            sim = util.cos_sim(s_emb, samp_embs).max().item()
            consistencies.append(sim)

        avg_consistency = np.mean(consistencies)
        hallucination_score = 1.0 - avg_consistency
        results.append({
            "sentence": sentence,
            "hallucination_score": hallucination_score,
            "flagged": hallucination_score > (1 - threshold)
        })

    return results


def split_sentences(text):
    # Placeholder — use nltk.sent_tokenize or spacy in practice
    return [s.strip() for s in text.split(".") if s.strip()]
```

Original paper also uses NLI and n-gram variants of the consistency measure.

---

### 3.3 NLI-Based Factual Checking

Use a Natural Language Inference model to check whether each generated claim is **entailed by**, **neutral to**, or **contradicted by** a reference source.

```
claim  →  NLI model  →  {entailment, neutral, contradiction}
source ↗
```

```python
from transformers import pipeline

nli = pipeline("text-classification", model="cross-encoder/nli-deberta-v3-small")

def check_claim_against_source(claim: str, source: str) -> dict:
    result = nli(f"{source} [SEP] {claim}")[0]
    return {
        "label": result["label"],   # ENTAILMENT / NEUTRAL / CONTRADICTION
        "score": result["score"],
        "hallucinated": result["label"] == "CONTRADICTION"
    }
```

Limitation: NLI models trained on SNLI/MultiNLI may not generalize to domain-specific text.

---

### 3.4 Reference-Based Metrics

| Metric | Mechanism | Limitation |
|---|---|---|
| ROUGE-{1,2,L} | N-gram overlap with reference | Does not capture semantic correctness |
| BERTScore | Contextual embedding cosine similarity | Can miss factual errors if surface is similar |
| **FactScore** | Decompose → retrieve → verify each atomic claim | Requires retrieval corpus; slow |

**FactScore pipeline**:

```
generated text
    → atomic claim decomposition (LLM)       e.g. "Marie Curie won the Nobel Prize in 1903"
    → retrieval from Wikipedia per claim
    → NLI / LLM verification: supported or not
    → FactScore = fraction of supported atomic claims
```

---

## 4. Retrieval-Augmented Generation (RAG)

Ground each response in retrieved documents rather than purely parametric memory. The key insight: retrieval externalizes the knowledge store, making facts inspectable and updateable without retraining.

### Architecture

```
User query
    │
    ▼
[Retriever]  ──────────────────────────────────────┐
  Dense (DPR, BGE, E5)                             │
  Sparse (BM25)                                    │
  Hybrid (RRF fusion)                              │
    │                                              │
    ▼                                              │
[Top-K documents]                                  │
    │                                              │
    ▼                                              │
[Context assembly]  ←── chunk + rerank (optional) │
    │                                              │
    ▼                                              │
[LLM]  ← system prompt with retrieved context ────┘
    │
    ▼
[Response grounded in retrieved docs]
```

### Code Sketch

```python
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class SimpleRAG:
    def __init__(self, documents: list[str], llm_fn, top_k: int = 3):
        self.documents = documents
        self.llm_fn = llm_fn
        self.top_k = top_k
        self.encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")
        self._build_index()

    def _build_index(self):
        embeddings = self.encoder.encode(self.documents, normalize_embeddings=True)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)   # inner product = cosine on normalized vecs
        self.index.add(embeddings.astype(np.float32))

    def retrieve(self, query: str) -> list[str]:
        q_emb = self.encoder.encode([query], normalize_embeddings=True)
        _, indices = self.index.search(q_emb.astype(np.float32), self.top_k)
        return [self.documents[i] for i in indices[0]]

    def answer(self, query: str) -> str:
        context_docs = self.retrieve(query)
        context = "\n\n".join(f"[Doc {i+1}]: {d}" for i, d in enumerate(context_docs))
        prompt = (
            f"Answer the question using ONLY the documents below. "
            f"If the answer is not in the documents, say 'I don't know'.\n\n"
            f"{context}\n\nQuestion: {query}\nAnswer:"
        )
        return self.llm_fn(prompt)
```

### RAG Failure Modes

- Retriever returns irrelevant documents → model ignores context and hallucinates anyway
- Long context dilution → model attends to noisy parts
- Context faithfulness vs. world knowledge conflict → model may override retrieved fact
- Closed-book fallback → model answers from memory when context seems weak

---

## 5. Prompt Engineering

### Chain-of-Thought (CoT)

Step-by-step reasoning forces the model to articulate intermediate claims, each of which can be verified. Reduces hallucination on multi-step reasoning tasks.

```
Bad:  "When was the Eiffel Tower built? Answer:"
Good: "When was the Eiffel Tower built? Think step by step before answering."
```

Hallucination rate on arithmetic and multi-hop QA drops substantially with CoT (Wei et al., 2022).

### "I Don't Know" Few-Shot Examples

```python
system_prompt = """You are a factual assistant. If you are uncertain, say exactly
"I don't know" rather than guessing. Examples:

Q: What is the capital of France?
A: Paris.

Q: Who won the 2031 World Cup?
A: I don't know — that event is beyond my knowledge cutoff.

Q: What is 17 × 24?
A: 408.

Q: What is the exact population of Mars colony Alpha?
A: I don't know — no such colony exists as of my knowledge cutoff.
"""
```

### Explicit Uncertainty Instructions

```
"If you are less than 80% confident in a claim, preface it with 'I believe' or
'I think'. If you cannot verify a claim from the provided context, say so explicitly."
```

### Cite Sources Instruction

```
"After each factual statement, include [Doc N] referring to the document number
in the context. Only make claims you can cite."
```

---

## 6. RLHF and RLAIF

### RLHF (Reinforcement Learning from Human Feedback)

Human raters label responses for factual accuracy. A reward model is trained on these preferences. The LLM policy is fine-tuned via PPO to maximize reward.

```
Prompt → LLM → response_A, response_B
                     ↓
               Human: A > B (A is more accurate)
                     ↓
               Reward model trained on Bradley-Terry pairs
                     ↓
               PPO fine-tunes LLM to maximize R(response)
```

When the reward model penalizes hallucinations explicitly, factuality improves. However, reward hacking is a real failure mode — models learn to appear confident rather than be accurate.

### Constitutional AI / RLAIF (Anthropic)

Replace human labelers with an AI critic guided by a constitution of principles.

1. **Critique**: Model generates a response, then critiques it against a principle (e.g., "Does this response contain unverified claims?")
2. **Revision**: Model revises its response based on the critique
3. **Preference data**: (original, revised) pairs used to train a preference model
4. **RL fine-tuning**: Policy trained against the AI preference model

```
response → critique prompt → critique text → revision prompt → revised response
                                                  ↓
                                         (original, revised) → PM training
```

### Limitations of RLHF/RLAIF for Hallucination

- Reward models can themselves hallucinate or be inconsistent
- Hard to define a precise reward signal for factual accuracy at scale
- Models may learn to avoid factual claims entirely (excessive hedging)
- Distribution shift: reward model trained on one domain may not generalize

---

## 7. Factuality Fine-Tuning

### Core Idea

Instead of relying on RL, directly fine-tune on datasets where outputs are verified to be factually correct. Penalize or exclude hallucinated outputs from training.

### Notable Approaches

| Method | Mechanism |
|---|---|
| **FLAME** | Factuality-aware fine-tuning; down-weights training examples with hallucinated claims |
| **FLAN-style factuality** | Instruction tuning on verified QA datasets (Natural Questions, TriviaQA) |
| **FactTune** | Fine-tune with FactScore as the signal; select only high-FactScore training examples |
| **Truthful SFT** | SFT on TruthfulQA-style adversarial examples with correct answers |

### Data Selection Strategy

```python
def factuality_filter(examples, threshold=0.85):
    """
    Keep only examples where FactScore >= threshold.
    Used to build a clean SFT dataset.
    """
    verified = []
    for ex in examples:
        score = compute_factscore(ex["response"], ex["source"])
        if score >= threshold:
            verified.append(ex)
    return verified
```

Trade-off: filtering aggressively reduces dataset size and may hurt coverage of edge cases.

---

## 8. Calibration

A well-calibrated model's expressed confidence matches its actual accuracy. If a model says it is 90% confident on 100 claims, roughly 90 should be correct.

### Expected Calibration Error (ECE)

```
ECE = Σ_b (|B_b| / n) * |acc(B_b) - conf(B_b)|

where B_b = bucket of predictions with confidence in [b, b+1/M)
      acc  = fraction correct in bucket
      conf = mean confidence in bucket
```

Low ECE = well calibrated. Large LLMs are often overconfident (underestimate uncertainty).

### Confidence Elicitation

```python
# Verbalized probability — ask the model directly
prompt = (
    "Answer the following question and then state your confidence as a percentage.\n"
    "Q: In what year did World War II end?\n"
    "A: [your answer]. Confidence: [X]%"
)

# Token probability — use log-probabilities of answer tokens
import math

def token_confidence(model, prompt, answer_token):
    logprobs = model.get_logprobs(prompt)
    prob = math.exp(logprobs.get(answer_token, -float("inf")))
    return prob
```

### Verbalized Uncertainty Phrases

Training or prompting models to use hedging language:

- "I believe, but am not certain, that..."
- "As of my knowledge cutoff..."
- "I don't have reliable information on this."
- "You may want to verify this with a current source."

Models fine-tuned on data that includes these phrases produce better-calibrated verbalized uncertainty (Kadavath et al., 2022 — "Language Models (Mostly) Know What They Know").

---

## 9. Constrained Decoding

Force the output to conform to a schema, preventing the model from generating tokens that would violate structural or factual constraints.

### JSON Mode / Structured Outputs

OpenAI, Anthropic, and open-source serving frameworks support grammar-constrained generation that guarantees valid JSON matching a schema. This eliminates a class of format-level hallucinations (invented keys, wrong types).

```python
from pydantic import BaseModel
from openai import OpenAI

class FactualClaim(BaseModel):
    claim: str
    confidence: float          # 0.0 – 1.0
    source_cited: bool

client = OpenAI()
response = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[{"role": "user", "content": "State one fact about photosynthesis."}],
    response_format=FactualClaim,
)
claim: FactualClaim = response.choices[0].message.parsed
```

### Grammar-Constrained Decoding (Outlines / LMQL)

**Outlines** uses a finite-state machine derived from a regex or JSON schema. At each decoding step, only tokens that keep the FSM in a valid state are allowed — all others get `-inf` logit.

```python
import outlines

model = outlines.models.transformers("mistralai/Mistral-7B-v0.1")

# Only allow responses matching this regex — no hallucinated formats
generator = outlines.generate.regex(
    model,
    r"The answer is (yes|no)\. Confidence: (0\.\d+|1\.0)\."
)

result = generator("Is the Eiffel Tower in Paris?")
# → "The answer is yes. Confidence: 0.97."
```

**LMQL** embeds constraints as logical predicates evaluated during beam search:

```
argmax
  "Answer: [ANSWER]"
where
  len(ANSWER) < 20 and
  ANSWER in {"yes", "no", "unknown"}
```

### Trade-offs

- Hard constraints can force grammatically valid but semantically wrong outputs
- Reduces generation diversity
- Very effective for structured extraction tasks; less so for open-ended generation

---

## 10. Evaluation Benchmarks

### TruthfulQA (Lin et al., 2022)

- 817 adversarial questions designed to elicit common misconceptions
- Humans tend to answer incorrectly due to widespread falsehoods
- Metrics: % truthful, % informative, truthful × informative
- Weakness: static, GPT-4 already saturates some categories

### HaluEval

- Pairs of (hallucinated, non-hallucinated) outputs across QA, summarization, dialogue
- Tests whether models can identify which response hallucmates
- Useful for evaluating detector models

### FActScore (Min et al., 2023)

- Open-domain biography generation benchmark
- FactScore = % of atomic claims supported by Wikipedia retrieval + NLI
- Model score: GPT-4 ~73%, Llama-2-13B ~55% at release

### FELM (Factual Error Localization and Mitigation)

- Focuses on fine-grained error localization within a response
- Labels individual sentences as factual or hallucinated
- Tests both detection and localization accuracy

### Summary Table

| Benchmark | Task | Primary Metric | Notes |
|---|---|---|---|
| TruthfulQA | Closed-ended QA | % Truthful | Adversarial misconceptions |
| HaluEval | Classification | Accuracy | Hallucination detection |
| FActScore | Open-ended generation | FactScore (0–1) | Wikipedia-grounded |
| FELM | Span-level detection | F1 (sentence-level) | Error localization |

---

## 11. Key Interview Points

**Definitions**
- Hallucination = fluent but factually incorrect or unverifiable output. Distinguish intrinsic (contradicts source) from extrinsic (unverifiable addition).
- Faithful ≠ factual: a model can faithfully reproduce a wrong claim from a document.

**Root Causes**
- Exposure bias from teacher forcing creates a train/inference distribution mismatch.
- Knowledge cutoff forces extrapolation on recent facts.
- High temperature and top-p sampling increase variance and factual drift.
- RLHF sycophancy can reward confident-sounding wrong answers.

**Detection**
- Self-consistency: high variance across samples signals uncertainty.
- SelfCheckGPT: no external KB needed — inconsistency across samplings = hallucination.
- NLI-based: entailment model checks claim vs. source; scalable but model-dependent.
- FactScore: gold standard for open-ended generation, but expensive.

**Mitigation Stack (ordered by intervention cost)**
1. Prompt engineering (cheapest — CoT, uncertainty instructions, cite-sources)
2. RAG (externalizes knowledge, reduces parametric reliance)
3. Constrained decoding (schema enforcement, no retraining needed)
4. Calibration (verbalized uncertainty, confidence elicitation)
5. Factuality fine-tuning (requires curated data)
6. RLHF/RLAIF (most expensive, reward hacking risk)

**RAG Gotchas**
- Retriever quality is the ceiling; bad retrieval → worse grounding than no RAG.
- Models still hallucinate when context is long or noisy (lost in the middle problem).
- Hybrid retrieval (dense + BM25 + RRF) outperforms dense-only in most benchmarks.

**Calibration**
- ECE measures gap between confidence and accuracy; lower is better.
- Large models are often overconfident; RLHF sometimes makes this worse.
- Verbalized uncertainty ("I believe...") can be elicited by few-shot or fine-tuning.

**Constrained Decoding**
- Outlines uses FSM-masked logits; guarantees valid schema but can't fix semantic errors.
- JSON mode eliminates format hallucinations, not factual ones.

**Benchmarks to cite**
- TruthfulQA for adversarial misconceptions, FActScore for open-ended factuality, HaluEval for detector evaluation, FELM for error localization.
