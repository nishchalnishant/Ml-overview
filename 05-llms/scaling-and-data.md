# Scaling Laws and Training Data

Understanding scaling laws tells you how to spend a compute budget optimally. Understanding data quality tells you why a 7B model trained on carefully curated data can outperform a 70B model trained on internet slop.

---

## 1. The Neural Scaling Laws

### Kaplan et al. (OpenAI, 2020)

The original scaling laws showed that test loss follows a power law in model size $N$, dataset size $D$, and compute $C$:

$$L(N) \approx \left(\frac{N_c}{N}\right)^{\alpha_N}, \quad L(D) \approx \left(\frac{D_c}{D}\right)^{\alpha_D}, \quad L(C) \approx \left(\frac{C_c}{C}\right)^{\alpha_C}$$

Key finding: **parameters dominated**. For a fixed compute budget, scale the model as large as possible and train for fewer tokens.

This led to GPT-3 (175B parameters, trained on 300B tokens) — compute-optimal under Kaplan but severely under-trained relative to what we now know.

### Chinchilla (DeepMind, 2022)

Hoffmann et al. showed Kaplan's laws used a flawed compute model. Re-running with proper accounting:

$$C \approx 6ND$$

For a compute-optimal training run, **model size and training tokens should scale equally**:

$$N_{opt} \propto C^{0.5}, \quad D_{opt} \propto C^{0.5}$$

**The ~20 tokens-per-parameter rule:** for every 1 parameter, train on approximately 20 tokens to be compute-optimal.

| Model | Parameters | Tokens | Tokens/Param | Status |
| :--- | :--- | :--- | :--- | :--- |
| GPT-3 | 175B | 300B | 1.7 | Severely under-trained |
| Chinchilla | 70B | 1.4T | 20 | Compute-optimal |
| LLaMA 2 7B | 7B | 2T | ~285 | Heavily over-trained for inference |
| LLaMA 3 70B | 70B | 15T | ~214 | Inference-optimized |

**Why over-train beyond Chinchilla-optimal?** Inference cost scales with model size, not training compute. A smaller, well-trained model costs less to serve at scale. LLaMA 3 7B trained on 15T tokens outperforms compute-optimal 40B models at inference time.

### Emergent Abilities

Some capabilities appear abruptly at certain scales, not gradually:

| Ability | Emergence threshold (FLOP) |
| :--- | :--- |
| 3-digit arithmetic | ~$10^{22}$ |
| Multi-step reasoning (chain-of-thought) | ~$10^{23}$ |
| Few-shot instruction following | ~$10^{23}$ |
| Calibrated uncertainty | ~$10^{24}$ |

**Caveat (Schaeffer et al., 2023):** emergent abilities may be artifacts of discontinuous metrics. Switching to smooth metrics often reveals gradual improvement. The discontinuity is in the evaluation, not necessarily the model.

---

## 2. Scaling Beyond Text: Multi-Modal and Reasoning

### The Token Scarcity Problem

High-quality human text on the internet is finite. Epoch et al. estimates 4–17 × 10^13 tokens of English web text exist; current frontier models have consumed most of it. Solutions:

1. **Synthetic data:** generate training data using stronger models
2. **Multi-modal data:** video, audio, scientific figures — far more abundant than text
3. **Self-play and RL:** generate hard problems and solutions iteratively (AlphaProof-style)
4. **Code and math:** structured data with ground-truth verification, excellent for reasoning

### Compute-Efficient Architectures

| Architecture | Efficiency gain | Mechanism |
| :--- | :--- | :--- |
| **MoE (Mixture of Experts)** | Active params << total params | Route each token to top-k experts |
| **Sparse attention** | $O(n \log n)$ vs $O(n^2)$ | Attend to limited token windows |
| **State Space Models (Mamba)** | $O(n)$ inference | Selective state compression |
| **Linear attention** | $O(n)$ | Approximate softmax with kernel trick |

---

## 3. Data Quality and Curation

Quality matters far more than quantity beyond a baseline. Phi-1 (1.3B) and LIMA (65B SFT) demonstrated that carefully curated small datasets can match or beat much larger models trained on raw internet data.

### The Data Pipeline

```
Raw sources (web, books, code, papers)
        │
        ▼
URL/Domain filtering (block spam, adult, malware domains)
        │
        ▼
Language identification (fastText classifier)
        │
        ▼
Quality filtering (perplexity threshold, heuristics)
        │
        ▼
Deduplication (exact + fuzzy, at paragraph level)
        │
        ▼
PII removal (names, phone numbers, emails)
        │
        ▼
Mixing and sampling (set domain weights)
        │
        ▼
Tokenization and sharding
```

### Quality Filtering

**Heuristic filters (fast):**
- Remove documents < 100 tokens or > 100k tokens
- Filter by character-level repetition ratio (> 0.2 flagged as degenerate)
- Remove lines that are mostly punctuation or numbers
- Filter by word overlap with known-quality reference corpus

**Classifier filters (accurate):**
```python
import fasttext

# Train a binary classifier on Wikipedia (quality=1) vs random web (quality=0)
model = fasttext.train_supervised(
    input="quality_training_data.txt",  # format: __label__quality text
    lr=0.1,
    epoch=5,
    wordNgrams=2,
    dim=100,
)

def filter_quality(text: str, threshold: float = 0.7) -> bool:
    label, prob = model.predict(text.replace("\n", " "))
    return label[0] == "__label__quality" and prob[0] >= threshold
```

### Deduplication

Training on duplicate data wastes compute and can cause memorization/privacy issues.

**MinHash LSH for near-duplicate detection:**

```python
from datasketch import MinHash, MinHashLSH

def create_minhash(text: str, num_perm: int = 128) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for word in text.lower().split():
        m.update(word.encode("utf-8"))
    return m

lsh = MinHashLSH(threshold=0.8, num_perm=128)

for i, doc in enumerate(documents):
    m = create_minhash(doc)
    result = lsh.query(m)
    if not result:  # no near-duplicates found
        lsh.insert(str(i), m)
        keep.append(doc)
    # else: near-duplicate, discard
```

For large-scale deduplication (trillion-token corpora), exact deduplication uses SHA-256 hashing; fuzzy deduplication uses n-gram shingling + LSH.

### Domain Mixing

Different domains contribute different capabilities. Mix weights are typically tuned empirically:

| Domain | Contribution | Typical weight |
| :--- | :--- | :--- |
| Web text (filtered) | General knowledge, language fluency | 40-60% |
| Books | Long-range coherence, structured reasoning | 10-20% |
| Code (GitHub) | Code generation, logical reasoning | 10-20% |
| Scientific papers (ArXiv) | STEM reasoning, specialized knowledge | 5-10% |
| Wikipedia | Factual accuracy, clean prose | 5-10% |
| Math (proofs, problems) | Mathematical reasoning | 1-5% |

Upsampling high-quality domains disproportionately improves reasoning benchmarks.

---

## 4. Synthetic Data

### Why Synthetic Data Works

Models trained purely on web text learn to imitate the average internet author. Synthetic data from capable models introduces:
- Cleaner formatting and step-by-step reasoning
- Harder edge cases than naturally occur in the wild
- Consistent style aligned with the target task

### Self-Instruct and Alpaca-style Generation

```python
import openai

def generate_instruction_pairs(seed_instructions: list[str], n: int = 100) -> list[dict]:
    """Generate (instruction, response) pairs for SFT."""
    
    prompt = f"""Generate {n} diverse instruction-response pairs for training an AI assistant.
    
Seed examples:
{chr(10).join(f'- {inst}' for inst in seed_instructions[:5])}

Requirements:
- Varied difficulty and topics
- Clear, complete responses
- Mix of question types (explain, write, analyze, solve)

Output JSON array: [{{"instruction": "...", "response": "..."}}]"""
    
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)["pairs"]
```

### Constitutional AI and RLAIF

Instead of human feedback, use the model itself to critique and revise responses:

1. Sample a response from the SFT model
2. Ask the model to critique the response against a list of principles ("Is this helpful? Harmless? Honest?")
3. Ask the model to revise the response based on the critique
4. Use revised responses as preference data for DPO/RLHF

This scales alignment cheaply without human annotators.

### Verified Synthetic Data (Math/Code)

The highest-quality synthetic data has **ground-truth verifiability**:

```python
def generate_math_problem() -> dict:
    """Generate a math problem where the answer can be verified programmatically."""
    problem = llm.complete("Generate a grade-8 algebra problem with a unique integer solution.")
    
    # Execute the solution in a Python interpreter to verify
    solution_code = llm.complete(f"Write Python code to solve: {problem}\nPrint the answer.")
    
    try:
        result = subprocess.run(["python", "-c", solution_code], 
                              capture_output=True, timeout=5)
        if result.returncode == 0:
            return {"problem": problem, "solution": result.stdout.decode()}
    except:
        pass
    return None  # Reject unverifiable problems
```

---

## 5. Scaling Inference: Test-Time Compute

Post-Chinchilla insight: you can trade inference compute for better answers.

### Chain-of-Thought Scaling

More thinking tokens at inference → better reasoning:

$$\text{Performance} \propto \log(\text{thinking tokens})$$

OpenAI o1, DeepSeek-R1, and Qwen-QwQ demonstrate that inference-time search substantially improves math and coding benchmarks.

### Best-of-N Sampling

Generate $N$ candidates, select the best using a verifier or reward model:

```python
def best_of_n(prompt: str, n: int = 8, judge_model: str = "gpt-4o") -> str:
    candidates = [llm.complete(prompt) for _ in range(n)]
    
    # Use a reward model or LLM judge to select best
    scores = [reward_model.score(prompt, c) for c in candidates]
    return candidates[scores.index(max(scores))]
```

**Effective compute scaling:** best-of-8 with a 7B model often matches best-of-1 with a 70B model at similar compute cost.

### Monte Carlo Tree Search (MCTS) for Reasoning

Explore multiple reasoning paths, backpropagate correctness signals, prune poor branches. Used in AlphaProof to solve IMO problems. Expensive but effective for very hard mathematical reasoning.

---

## 6. The Scaling Frontier

| Dimension | Current frontier | Open problems |
| :--- | :--- | :--- |
| **Parameters** | ~1-2T (GPT-4 speculated) | Memory wall, communication overhead |
| **Context length** | 1M tokens (Gemini 1.5) | "Lost in the middle" attention decay |
| **Training tokens** | ~15T (LLaMA 3) | Token scarcity — approaching internet exhaustion |
| **Reasoning** | Test-time compute (o1, R1) | Reliable multi-step verification |
| **Modalities** | Text + image + audio (GPT-4o) | Video, 3D, physical world grounding |

> [!TIP]
> **Interview structure:** Scaling = three levers (parameters, data, compute). Chinchilla showed data matters as much as size. Inference efficiency inverts the optimal tradeoff — smaller, overtrained models win in production. Test-time compute is the new frontier for capability improvements without training new models.
