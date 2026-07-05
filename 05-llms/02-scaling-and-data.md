---
module: Llms
topic: Scaling And Data
subtopic: ""
status: unread
tags: [llms, ml, scaling-and-data]
---
# Scaling Laws and Training Data

---

## 1. The Empirical Observation That Changed How Everyone Trains Models

In 2022, DeepMind published the Chinchilla paper (Hoffmann et al., "Training Compute-Optimal Large Language Models"). The central result was not a theory — it was an observation: **GPT-3 is massively undertrained.** A 70B parameter model trained on 1.4T tokens (Chinchilla itself) achieves lower loss than GPT-3's 175B model trained on 300B tokens — at the same total training compute. You can build a better model at one-third the parameter count if you train it on enough data.

This was not what the field had assumed. OpenAI's 2020 scaling laws (Kaplan et al.) had suggested that for a fixed compute budget, you should scale model size as aggressively as possible and accept fewer training tokens. Chinchilla found this was wrong because Kaplan et al. had not varied the training duration long enough to observe the data-scaling regime.

---

## 2. Kaplan vs. Chinchilla: What Each Got Right

### Kaplan et al. (OpenAI, 2020): the first principles

Kaplan ran experiments varying model size, dataset size, and compute independently and found that test loss follows power laws in each:

$$L(N) \approx \left(\frac{N_c}{N}\right)^{\alpha_N}, \quad L(D) \approx \left(\frac{D_c}{D}\right)^{\alpha_D}$$

The key finding: parameters scale more efficiently than data — adding parameters to a fixed dataset reduces loss faster than adding data to a fixed parameter count. Conclusion: spend your compute on more parameters.

This led to GPT-3: 175B parameters, only 300B training tokens (~1.7 tokens per parameter).

### Chinchilla: the corrected picture

Hoffmann et al. ran a broader set of experiments, including models trained to genuine convergence rather than a fixed compute budget. They found that Kaplan's estimate of the compute cost of a forward pass was incorrect, and that data and parameters should be scaled equally.

The compute approximation they used:

$$C \approx 6ND$$

(6 FLOPs per parameter per token: 2 for the forward pass, 4 for the backward pass). For a compute-optimal training run:

$$N_{opt} \propto C^{0.5}, \quad D_{opt} \propto C^{0.5}$$

**The practical rule of thumb**: ~20 training tokens per parameter for compute-optimal training. Chinchilla (70B parameters) was trained on 1.4T tokens: exactly 20 tokens per parameter.

### Working backward from the observation

The insight is not "here is a formula." The insight is: Kaplan was right that both parameters and data matter, but wrong about their relative scaling. When Chinchilla ran properly controlled experiments across a wider range of parameter/data combinations, the crossover point where data scaling overtakes parameter scaling appeared much earlier than Kaplan suggested.

---

## 3. Why Everyone Ignores Compute-Optimal Training in Practice

**The problem Chinchilla solves is the wrong problem for a product company.** Chinchilla optimizes for minimizing loss at fixed training compute. But once you've trained a model, you run inference on it billions of times. Inference cost scales with model size at every request. A 70B model costs roughly 10× more to serve than a 7B model per token. Training cost is paid once; inference cost is paid continuously.

**The reframe**: if your goal is the best model for a given *inference* compute budget, you should use a smaller model and train it for much longer than Chinchilla-optimal. The smaller model is cheaper to serve; the extra training tokens make it more capable than a larger undertrained model at the same inference cost.

| Model | Parameters | Tokens | Tokens/Param | Status under Chinchilla |
| :--- | :--- | :--- | :--- | :--- |
| GPT-3 | 175B | 300B | 1.7 | Severely undertrained |
| Chinchilla | 70B | 1.4T | 20 | Compute-optimal |
| LLaMA 2 7B | 7B | 2T | ~285 | Deliberately overtrained |
| LLaMA 3 70B | 70B | 15T | ~214 | Inference-optimized |

LLaMA 3 7B trained on 15T tokens outperforms Chinchilla-optimal models several times its size at identical inference compute. That's the point.

---

## 4. Emergent Abilities: Real Phenomenon or Measurement Artifact?

**The observation**: certain capabilities appear to jump discontinuously at a threshold model scale, rather than improving gradually. Models below ~$10^{22}$ training FLOPs cannot reliably do 3-digit arithmetic; models above that threshold can. The jump looks sharp enough to call "emergent."

**The core insight (Schaeffer et al., 2023)**: emergent abilities are often an artifact of discontinuous evaluation metrics, not discontinuous model capabilities. The underlying probability of getting a problem right improves smoothly with scale. But if the metric is "exact match accuracy" (all-or-nothing credit), a model that is 70% right on each step of a 3-step problem has near-zero accuracy until it crosses ~90% per step, at which point accuracy jumps to ~73%. The discontinuity is in the evaluation, not the model.

**What this means in practice**: use calibrated, continuous metrics where possible. Binary exact-match metrics can make your model look worse than it is at intermediate scales, which can lead you to incorrectly conclude that a capability has "not emerged yet" when it is actually improving gradually.

| Claimed emergent ability | Threshold (training FLOPs) |
| :--- | :--- |
| 3-digit arithmetic | ~$10^{22}$ |
| Multi-step reasoning (chain-of-thought) | ~$10^{23}$ |
| Few-shot instruction following | ~$10^{23}$ |
| Calibrated uncertainty | ~$10^{24}$ |

These thresholds should be read as "where exact-match metrics start showing clear improvement" not as genuine discontinuities in model capability.

---

## 5. The Token Scarcity Problem

**The problem**: the internet is finite. Epoch et al. estimated that high-quality English text on the web amounts to roughly $4-17 \times 10^{13}$ tokens. Frontier models have already consumed most of it. If the current scaling trajectory continues, data will be the bottleneck before compute is.

**What this actually means**: you can't simply collect more data — the high-quality, naturally-occurring English text roughly available to train on has a ceiling. The solutions all involve either generating new data or using other modalities.

### Synthetic data

Models trained purely on web text learn to imitate the average internet author. A capable teacher model generating synthetic training data produces cleaner reasoning chains, harder edge cases, and more consistent style than naturally-occurring text.

The quality ceiling: a model can't be significantly smarter than its teacher model purely from synthetic data without a verification signal. Verification changes this.

**Verified synthetic data** (highest quality): generate candidate solutions, verify them programmatically or mathematically, keep only verified correct examples:

```python
def generate_verified_math_problem() -> dict | None:
    problem = llm.complete("Generate a grade-8 algebra problem with a unique integer solution.")
    solution_code = llm.complete(f"Write Python to solve: {problem}\nPrint the answer.")

    try:
        result = subprocess.run(["python", "-c", solution_code],
                                capture_output=True, timeout=5)
        if result.returncode == 0:
            return {"problem": problem, "solution": result.stdout.decode()}
    except Exception:
        pass
    return None  # Reject unverifiable problems
```

With verification, you can generate arbitrarily many correct math and code training examples. This is how DeepSeek-R1 and similar models bootstrapped their reasoning capabilities.

### Constitutional AI and RLAIF

> Covered in depth in [training-process.md](01-training-process.md) (Alignment section). In brief: rather than human annotators, use a model to critique and revise its own responses against a list of principles, then use (original, revised) pairs as DPO/RLHF preference data. Risk: systematic critic biases propagate into training data.

### Multi-modal data

Text tokens from the web are scarce; video frames are not. A single hour of video has far more information content than a comparable duration of text. Training on video, audio, and scientific figures provides data at orders of magnitude more volume — but requires solving harder alignment problems (how do you connect visual and textual representations?).

---

## 6. Data Quality: Why a 7B Model Can Beat a 70B Model

**The problem**: raw internet text contains enormous amounts of low-value content — SEO spam, boilerplate, near-duplicate pages, machine-translated garbage, and low-quality forum posts. Training on this data forces the model to model noise, which consumes capacity that could be spent on signal.

**The empirical evidence**: Phi-1 (1.3B parameters, trained on "textbook quality" synthetic and curated data) matched or exceeded models 10× its size on coding benchmarks. LIMA (65B model SFT on 1,000 carefully curated examples) matched RLHF-aligned models. Quality dominates quantity past a baseline.

### The data pipeline

```
Raw sources (web, books, code, papers)
        │
        ▼
URL/domain filtering (block spam, adult, malware domains)
        │
        ▼
Language identification (fastText classifier)
        │
        ▼
Quality filtering (perplexity-based + heuristics)
        │
        ▼
Deduplication (exact + fuzzy, at paragraph level)
        │
        ▼
PII removal (names, phone numbers, emails)
        │
        ▼
Domain mixing (set weights per source type)
        │
        ▼
Tokenization and sharding
```

### Quality filtering

**Heuristic filters** (fast, run first):
- Remove documents shorter than 100 or longer than 100,000 tokens
- Filter by character-level repetition ratio (> 20% flagged as degenerate)
- Remove documents where most lines are punctuation or numbers
- Filter by word overlap with a known-quality reference corpus

**Classifier filters** (accurate, run on heuristic survivors):

```python
import fasttext

# Train a binary classifier: Wikipedia/books = quality, random web = noise
quality_model = fasttext.train_supervised(
    input="quality_training_data.txt",   # format: __label__quality text
    lr=0.1, epoch=5, wordNgrams=2, dim=100,
)

def passes_quality_filter(text: str, threshold: float = 0.7) -> bool:
    label, prob = quality_model.predict(text.replace("\n", " "))
    return label[0] == "__label__quality" and prob[0] >= threshold
```

### Deduplication

**The problem**: near-duplicate documents waste compute (the model sees near-identical gradients multiple times) and cause memorization (the model can verbatim reproduce training text that appeared many times).

MinHash LSH handles fuzzy deduplication at scale:

```python
from datasketch import MinHash, MinHashLSH

def create_minhash(text: str, num_perm: int = 128) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for word in text.lower().split():
        m.update(word.encode("utf-8"))
    return m

lsh = MinHashLSH(threshold=0.8, num_perm=128)
kept = []

for i, doc in enumerate(documents):
    m = create_minhash(doc)
    if not lsh.query(m):        # no near-duplicates found
        lsh.insert(str(i), m)
        kept.append(doc)
    # else: near-duplicate, discard
```

For trillion-token corpora: exact deduplication uses SHA-256 hashing at the document level; fuzzy deduplication uses the MinHash approach at the paragraph level.

### Domain mixing

Different source types contribute different capabilities. Mix weights are tuned empirically:

| Domain | Primary contribution | Typical weight |
| :--- | :--- | :--- |
| Web text (filtered) | General knowledge, language fluency | 40–60% |
| Books | Long-range coherence, structured reasoning | 10–20% |
| Code (GitHub) | Code generation, logical/formal reasoning | 10–20% |
| Scientific papers | STEM reasoning, specialized knowledge | 5–10% |
| Wikipedia | Factual accuracy, clean prose | 5–10% |
| Math (proofs, competition problems) | Mathematical reasoning | 1–5% |

Upsampling code and math disproportionately improves reasoning benchmarks — these domains have high information density and ground-truth verifiability that teaches the model structured reasoning patterns.

---

## 7. Test-Time Compute Scaling: The New Frontier

**The problem after Chinchilla**: you've trained the model optimally, served it cheaply. Is there any way to get more capability at inference time without training a bigger model?

**The observation**: giving the model more tokens to "think" before producing an answer reliably improves performance on hard reasoning tasks. This is not a new insight about architecture — it follows directly from the fact that more computation produces better outputs. The question is whether you can systematically allocate more inference compute.

### Chain-of-thought

The simplest form: prompt the model to reason step by step before answering. The reasoning tokens are not in the final output but they change the distribution of the final token's context. Empirically:

$$\text{Performance} \approx f(\log(\text{thinking tokens}))$$

The relationship is roughly logarithmic — doubling the reasoning tokens gives diminishing but real returns.

### Best-of-N sampling

Generate $N$ candidate responses, select the best using a reward model or verifier:

```python
def best_of_n(prompt: str, model, reward_model, n: int = 8) -> str:
    candidates = [model.complete(prompt) for _ in range(n)]
    scores = [reward_model.score(prompt, c) for c in candidates]
    return candidates[scores.index(max(scores))]
```

Best-of-8 from a 7B model often matches best-of-1 from a 70B model at comparable total compute cost. The ratio of verification cost to generation cost determines the efficiency of this tradeoff.

**What breaks**: best-of-N only works if the verifier is accurate. A weak reward model will select the response that games the reward model, not the genuinely best response.

### Monte Carlo Tree Search for reasoning

For formal reasoning tasks (mathematical proof, verified code), explore multiple reasoning paths as a tree, backpropagate correctness signals, and prune poor branches. Used in AlphaProof to solve International Mathematical Olympiad problems. Expensive but qualitatively beyond what best-of-N achieves on hard structured problems.

---

## 8. Compute-Efficient Architectures: Scaling the Capability/FLOPs Ratio

When scaling token count hits data walls and scaling parameters hits memory walls, architectural improvements become the marginal gains.

| Architecture | Core efficiency mechanism | Tradeoff |
| :--- | :--- | :--- |
| MoE (Mixture of Experts) | Active parameters << total; sparse routing | Memory unchanged; routing overhead; load balancing |
| Sparse attention | $O(T \log T)$ vs $O(T^2)$ | Loses global context per layer |
| State Space Models (Mamba) | $O(T)$ inference via selective state | Weaker at in-context learning than attention |
| Linear attention | $O(T)$ via kernel approximation | Quality gap vs softmax attention in practice |

---

## 9. The Scaling Frontier

| Dimension | Current frontier | Binding constraint |
| :--- | :--- | :--- |
| Parameters | ~1–2T (estimated for GPT-4) | Memory, communication overhead |
| Context length | 1M tokens (Gemini 1.5) | "Lost in the middle" attention degradation |
| Training tokens | ~15T (LLaMA 3) | High-quality web text near exhaustion |
| Reasoning (inference-time) | Extended thinking (o1, R1, DeepSeek) | Verification accuracy at hard problems |
| Modalities | Text + image + audio (GPT-4o) | Video, 3D, physical-world grounding |

## Flashcards

**Remove documents shorter than 100 or longer than 100,000 tokens?** #flashcard
Remove documents shorter than 100 or longer than 100,000 tokens

**Filter by character-level repetition ratio (> 20% flagged as degenerate)?** #flashcard
Filter by character-level repetition ratio (> 20% flagged as degenerate)

**Remove documents where most lines are punctuation or numbers?** #flashcard
Remove documents where most lines are punctuation or numbers

**Filter by word overlap with a known-quality reference corpus?** #flashcard
Filter by word overlap with a known-quality reference corpus
