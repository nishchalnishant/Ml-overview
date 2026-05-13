# LLM Evaluation and Benchmarks

Evaluating LLMs is hard because language is open-ended. A correct answer can be stated in many ways, safety failures are subtle, and benchmark contamination inflates scores. This file covers: standard benchmarks, automated metrics, LLM-as-judge, human evaluation, and domain-specific evaluation design.

---

## 1. Standard Benchmarks

### Knowledge and Reasoning

| Benchmark | Task | Metric | Notes |
| :--- | :--- | :--- | :--- |
| **MMLU** | 57-subject multiple-choice (57k questions) | Accuracy | GPT-4: ~86%, Llama 3 70B: ~82% |
| **BIG-Bench Hard** | 23 hard reasoning tasks (symbolic, logical) | Exact match | Resistant to scaling — models plateau |
| **ARC-Challenge** | Grade-school science multiple-choice | Accuracy | Requires reasoning beyond facts |
| **HellaSwag** | Complete the next sentence | Accuracy | Tests world-model commonsense |
| **WinoGrande** | Coreference resolution with common sense | Accuracy | Adversarially filtered |

### Math and Code

| Benchmark | Task | Metric | Notes |
| :--- | :--- | :--- | :--- |
| **GSM8K** | Grade-school math word problems (8.5k) | Exact match | Tests multi-step reasoning chains |
| **MATH** | Competition-level mathematics | Exact match | Harder than GSM8K; requires symbolic reasoning |
| **HumanEval** | 164 Python coding problems | pass@k | Measures functional correctness; k usually 1,10,100 |
| **MBPP** | 374 crowd-sourced Python problems | pass@k | More diverse than HumanEval |
| **SWE-Bench** | Real GitHub issues with test suites | % resolved | Reflects actual software engineering ability |

**pass@k formula:**

$$\text{pass@}k = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}$$

where $n$ total samples are drawn and $c$ pass the tests. Estimates probability that at least one of $k$ samples passes.

### Long Context

| Benchmark | Task | Notes |
| :--- | :--- | :--- |
| **SCROLLS** | Summarization, QA over long documents | Tests 10k-100k token context |
| **RULER** | Synthetic long-context tasks (needle-in-haystack variants) | Separates context length from reasoning |
| **LongBench** | 6 task categories over 16 datasets | Multilingual long-context understanding |

**Needle-in-a-haystack:** embed a single fact in a long irrelevant document, ask the model to retrieve it. Reveals positional attention decay — most models degrade at the middle of long contexts.

### Alignment and Safety

| Benchmark | What it measures | Notes |
| :--- | :--- | :--- |
| **TruthfulQA** | Factual accuracy under adversarial queries | Common misconceptions; models often fail |
| **BBQ** | Social bias in QA (ambiguous contexts) | Measures demographic bias direction and magnitude |
| **BOLD** | Toxicity in open-ended completions | Fairness across demographic groups |
| **MT-Bench** | Multi-turn instruction following (80 questions, 8 categories) | GPT-4 judges on 1-10 scale |

---

## 2. Automated Text Metrics

### BLEU (Bilingual Evaluation Understudy)

$$\text{BLEU} = \text{BP} \cdot \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)$$

where $p_n$ is modified n-gram precision, $w_n = 1/N$, and BP is a brevity penalty. Scores between 0-1 (higher = better).

**Limitations:** rewards n-gram overlap, misses synonyms and paraphrases. A 0.3 BLEU on MT is decent; a 0.1 BLEU on summarization may be competitive.

### ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

- **ROUGE-N:** n-gram recall between hypothesis and reference
- **ROUGE-L:** longest common subsequence-based F1

$$\text{ROUGE-L} = \frac{(1+\beta^2) R_{lcs} P_{lcs}}{R_{lcs} + \beta^2 P_{lcs}}$$

ROUGE is standard for summarization evaluation. Does not capture semantic meaning.

### BERTScore

Uses contextual embeddings from BERT to compute token similarity:

$$P_{\text{BERT}} = \frac{1}{|\hat{y}|} \sum_{\hat{y}_j \in \hat{y}} \max_{y_i \in y} \mathbf{x}_{\hat{y}_j}^\top \mathbf{x}_{y_i}$$

BERTScore correlates better with human judgment than BLEU/ROUGE. Expensive — requires a BERT forward pass per example.

```python
from bert_score import score

P, R, F1 = score(
    cands=["The cat sat on the mat"],
    refs=["A feline rested on the rug"],
    lang="en",
    verbose=True
)
print(f"BERTScore F1: {F1.mean():.4f}")
```

### Perplexity

$$\text{PPL}(X) = \exp\left(-\frac{1}{T} \sum_{t=1}^{T} \log P_\theta(x_t \mid x_{<t})\right)$$

Lower perplexity = the model assigns higher probability to the test sequence. Measures language model quality, not task performance. GPT-2: ~29 PPL on WikiText-103; GPT-3: ~20 PPL.

---

## 3. LLM-as-Judge

Human evaluation is expensive and slow. A strong LLM (typically GPT-4 or Claude Opus) grades outputs instead.

### Absolute Scoring

```python
import openai
import json

def llm_judge_absolute(question: str, answer: str, criteria: list[str]) -> dict:
    criteria_str = "\n".join(f"- {c}" for c in criteria)
    prompt = f"""You are evaluating a model's answer to a question.

Question: {question}
Answer: {answer}

Rate on each criterion from 1 (poor) to 5 (excellent):
{criteria_str}

Return JSON: {{"scores": {{"criterion": score}}, "reasoning": "..."}}"""
    
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)
```

### Pairwise Comparison (Arena-style)

```python
def llm_judge_pairwise(question: str, answer_a: str, answer_b: str) -> str:
    prompt = f"""Compare two answers to the question below.

Question: {question}

Answer A: {answer_a}

Answer B: {answer_b}

Which answer is better? Consider: accuracy, helpfulness, clarity, and completeness.
Reply with only 'A', 'B', or 'tie'."""
    
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()
```

### MT-Bench (Multi-Turn Evaluation)

80 carefully designed questions across 8 categories: writing, roleplay, extraction, reasoning, math, coding, STEM, humanities. Each conversation has two turns.

```
Turn 1: "What is the difference between deep copy and shallow copy in Python?"
Turn 2: "How would this behavior differ with a custom class that overrides __copy__?"
```

GPT-4 judges each answer on a 1-10 scale. The multi-turn aspect tests coherence and memory across context.

### Biases in LLM Judges

| Bias | Description | Mitigation |
| :--- | :--- | :--- |
| **Self-preference** | Model favors answers that match its own style | Use a different judge model |
| **Position bias** | Prefers answer A when listed first | Swap order, average both results |
| **Verbosity bias** | Prefers longer answers regardless of quality | Explicit length-irrelevant scoring criteria |
| **Sycophancy** | Agrees with stated human opinion | Use neutral prompts without stated preferences |

---

## 4. LMSYS Chatbot Arena

Anonymous side-by-side comparisons where users chat with two models simultaneously and vote on which is better. Elo ratings are computed from win/loss outcomes.

$$\text{Expected score} = \frac{1}{1 + 10^{(R_B - R_A)/400}}$$

**Why it matters:** captures nuance, tone, formatting preferences, and user satisfaction that static benchmarks miss. Reflects real-world user preference distribution.

**Limitations:** subject to selection bias (who votes), prompt distribution may not match deployment domain, strategic users can game rankings.

---

## 5. Benchmark Contamination

A critical validity threat: test questions appearing in pretraining corpora.

**Detection methods:**
1. **N-gram overlap:** check if test examples appear in training data (requires access to pretraining corpus)
2. **Membership inference attacks:** probe whether the model has "memorized" specific test examples
3. **Canonical vs. paraphrased versions:** if performance drops sharply on paraphrased but semantically equivalent questions, contamination is likely
4. **Chronological splits:** train only on data before a cutoff date; test on questions released after

**Example:** GSM8K questions appearing verbatim on math tutoring forums that were crawled into Common Crawl. Models may achieve high scores by memorizing answers rather than reasoning.

---

## 6. Domain-Specific Evaluation

Standard benchmarks do not reflect performance in specialized domains. Design domain evals when:
- Facts are highly specialized (medical, legal, financial)
- Errors have asymmetric costs (false negatives in cancer screening vs. false positives)
- Style and format requirements are strict

### Evaluation Stack for Domain Models

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from datasets import Dataset

# Collect (question, generated answer, retrieved context, ground truth) tuples
eval_data = Dataset.from_dict({
    "question": questions,
    "answer": model_answers,
    "contexts": retrieved_contexts,   # list of lists of strings
    "ground_truth": reference_answers,
})

results = evaluate(eval_data, metrics=[
    faithfulness,         # answer grounded in context?
    answer_relevancy,     # answer addresses the question?
    context_precision,    # retrieved chunks are relevant?
    context_recall,       # context contains necessary information?
])
```

### Medical Domain Checklist

| Aspect | Metric | Tool |
| :--- | :--- | :--- |
| Clinical accuracy | Physician agreement rate | Expert annotation |
| Factual groundedness | Citation accuracy | Manual fact-check |
| Harm avoidance | Dangerous advice rate | Red-team annotation |
| Benchmark | PubMedQA, MedQA-USMLE accuracy | Standard benchmarking |

---

## 7. Evaluation Design Principles

**1. Never evaluate on the training distribution.** Use a held-out test set that was collected after training cutoff or kept strictly separate.

**2. Use multiple complementary metrics.** No single metric captures everything. At minimum: task accuracy + LLM-judge quality + safety evaluation.

**3. Calibration matters more than accuracy for decision-critical uses.** A model that says "90% confident" should be right 90% of the time.

**4. Evaluate failure modes, not just average performance.** Slice metrics by demographic groups, input length, topic, and edge cases.

**5. Set a regression baseline.** Before deploying an updated model, verify it doesn't regress on categories that the current model handles well.

```python
def regression_check(baseline_scores: dict, new_scores: dict, threshold: float = 0.02) -> list:
    """Returns categories where new model regresses by more than threshold."""
    regressions = []
    for category, baseline in baseline_scores.items():
        new = new_scores.get(category, 0)
        if baseline - new > threshold:
            regressions.append({
                "category": category,
                "baseline": baseline,
                "new": new,
                "delta": new - baseline
            })
    return regressions
```

> [!TIP]
> **Interview structure:** When asked about LLM evaluation — (1) distinguish automated benchmarks (task-specific, reproducible) from human evaluation (ground truth but slow) from LLM judges (scalable but biased), (2) name benchmark contamination as a validity threat, (3) explain why domain evals require custom design. The sophistication is in knowing the limitations of each approach.
