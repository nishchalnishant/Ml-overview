---
module: Llms
topic: Evaluation Benchmarks
subtopic: ""
status: unread
tags: [llms, ml, evaluation-benchmarks]
---
# LLM Evaluation and Benchmarks

---

## 1. The Core Problem: What Does "Better" Even Mean?

**The problem**: a model outputs a string. You want to know if that string is good. But "good" is not a well-defined function. "Good" depends on whether the answer is factually correct, whether it's in the right format, whether it's appropriate for the audience, whether it avoids harms, whether the reasoning is sound. No single number captures all of this.

Every evaluation method makes a tradeoff: automated metrics are fast and reproducible but miss nuance; human evaluation is ground truth but slow and expensive; LLM judges scale cheaply but inherit the judge model's biases. The sophistication in evaluation is knowing which method is appropriate for which question, and knowing the failure modes of each.

---

## 2. Standard Benchmarks: Knowledge and Reasoning

Standard benchmarks use multiple-choice or short-answer formats so that "correctness" can be defined unambiguously. This makes them fast and reproducible — but also gameable through contamination.

### MMLU (Massive Multitask Language Understanding)

57 academic subjects, 57,000 multiple-choice questions from undergraduate and professional exams. Tests breadth of knowledge, not depth. A model can score high on MMLU by pattern-matching to the most plausible-sounding answer without understanding the material.

Use it for: coarse ranking of general knowledge across models. Don't use it for: measuring reasoning ability or domain expertise.

### BIG-Bench Hard

A subset of BIG-Bench tasks chosen specifically because they resisted the original scaling trend — performance plateaued even as models got larger. The 23 tasks include symbolic reasoning, logical deduction, and unusual word problems. Harder to game via scale alone.

Use it for: testing genuine reasoning rather than knowledge recall.

### ARC-Challenge

Grade-school science questions, but specifically the subset that a word-frequency baseline gets wrong. A model that just matches the question to the most common words in the answer can't solve these.

### HellaSwag

Complete the next sentence from a situation description. The wrong answers are generated adversarially to be plausible. Tests commonsense world model — the implicit understanding that fire is hot, chairs are for sitting, and so on.

### WinoGrande

Coreference resolution: fill in a blank in a sentence where the correct answer depends on commonsense reasoning ("The trophy doesn't fit in the suitcase because ___ is too large"). Adversarially filtered to remove questions that simple heuristics can answer.

---

## 3. Standard Benchmarks: Math and Code

### GSM8K

8,500 grade-school math word problems requiring multi-step arithmetic reasoning. Ground truth is a single integer. Tests the ability to decompose a problem and execute a chain of steps correctly.

**Why it's still useful despite being "solved"**: at lower scales (< 7B), performance varies meaningfully. The failure mode it tests — losing track of state across multi-step reasoning — is real.

### MATH

Competition-level mathematics (AMC, AIME, Olympiad problems). Requires symbolic manipulation and multi-step proof construction. Much harder than GSM8K — GPT-4 levels performance on GSM8K is approximately GPT-3 levels on MATH.

### HumanEval and MBPP

HumanEval: 164 Python coding problems from OpenAI. Given a function signature and docstring, generate a function body. Tested with unit tests.

MBPP: 374 crowd-sourced Python problems. More diverse task types.

The **pass@k** metric: how likely is it that at least one of $k$ generated samples passes all tests?

$$\text{pass@}k = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}$$

where $n$ total samples are generated and $c$ pass all tests. At $k=1$: this is just the pass rate. At $k=10$: measures whether the model can find a correct solution in 10 tries — more forgiving, better for capability assessment.

**What breaks**: HumanEval problems are simple (write a function to count vowels) and many have appeared verbatim in public discussions of the benchmark. A model that memorized the answers can score high without being able to code.

### SWE-Bench

Real GitHub issues with corresponding test suites. The model is given a repository and an issue description; success is measured by whether the modified code passes the test suite. This actually reflects software engineering ability. Scoring significantly lower here than on HumanEval is a signal that the model can write functions but can't reason about a codebase.

---

## 4. Long Context Benchmarks

**The problem that emerges at scale**: models trained on long contexts don't uniformly use the entire context. Attention decays — tokens at the beginning and end of a document are better attended than tokens in the middle. A model that "supports 128K tokens" may perform poorly on information that appears in the middle of a 128K document.

### Needle-in-a-Haystack

The simplest probe: embed a single fact ("The secret word is 'banana'") at a specific position in a long irrelevant document, then ask the model to retrieve it. Plot retrieval accuracy as a function of fact position within the document. Most models show a characteristic U-shape: high accuracy near the beginning and end, degraded accuracy in the middle.

This benchmark is almost too simple — retrieving a single fact is far easier than real long-document reasoning — but it quickly reveals whether a model has genuine long-context attention or just claims to.

### RULER

Synthetic tasks that separate "can the model use long context" from "can the model solve the underlying reasoning problem." Tests: multi-hop retrieval (fact A points to fact B points to the answer), aggregation (find all instances of a pattern), and ordering (place events from the document in sequence).

### SCROLLS and LongBench

Naturalistic tasks: summarization and QA over long documents. More realistic than synthetic needles but harder to analyze — poor performance could be due to insufficient context use, poor summarization ability, or domain mismatch.

---

## 5. Alignment and Safety Benchmarks

### TruthfulQA

817 questions designed to probe common human misconceptions — areas where the statistically likely answer is wrong. Categories include health myths, historical distortions, and common scientific misunderstandings. A model trained to predict likely text often confidently states these misconceptions. TruthfulQA measures whether alignment training has corrected this.

**What breaks**: it's possible to score well on TruthfulQA by being overly cautious — refusing to answer or adding excessive hedges to everything. This is "truthful" in a degenerate sense. The benchmark doesn't penalize unhelpful non-answers well.

### BBQ (Bias Benchmark for QA)

Questions set in ambiguous social contexts where the correct answer is "not enough information," but models often default to stereotyped responses. Measures bias direction and magnitude across demographic groups. Identifies whether a model treats different groups asymmetrically.

### MT-Bench

80 carefully designed multi-turn questions across 8 categories: writing, roleplay, extraction, reasoning, math, coding, STEM, humanities. A GPT-4 judge scores each response 1–10. The multi-turn structure tests whether the model can maintain coherence across a conversation — harder to fake than single-turn responses.

---

## 6. Automated Text Metrics: When to Use Them and When Not To

Automated metrics are fast and reproducible but correlate poorly with human judgment for generative tasks. Use them as lightweight signals, not ground truth.

### BLEU

Measures n-gram overlap between a generated translation and reference translations:

$$\text{BLEU} = \text{BP} \cdot \exp\!\left(\sum_{n=1}^{N} w_n \log p_n\right)$$

where $p_n$ is modified n-gram precision and BP is a brevity penalty. Designed for machine translation where exact phrasing matters.

**The problem**: BLEU rewards lexical similarity, not semantic correctness. "The dog bit the man" and "The man bit the dog" have identical BLEU relative to either reference. For summarization and open-ended generation, BLEU correlates poorly with human quality judgments. Treat BLEU scores as a filter (detecting very bad outputs) not an evaluator.

### ROUGE

ROUGE-L measures the longest common subsequence between hypothesis and reference. Standard for summarization evaluation.

$$\text{ROUGE-L} = \frac{(1+\beta^2)\, R_{lcs}\, P_{lcs}}{R_{lcs} + \beta^2 P_{lcs}}$$

Better than BLEU for summarization because it measures coverage (recall-oriented), but still misses semantic equivalence.

### BERTScore

Uses contextual embeddings to compute semantic similarity between hypothesis and reference tokens. Correlates meaningfully better with human judgment than BLEU or ROUGE:

$$P_{\text{BERT}} = \frac{1}{|\hat{y}|} \sum_{\hat{y}_j \in \hat{y}} \max_{y_i \in y}\, \mathbf{x}_{\hat{y}_j}^\top \mathbf{x}_{y_i}$$

```python
from bert_score import score

P, R, F1 = score(
    cands=["The cat sat on the mat"],
    refs=["A feline rested on the rug"],
    lang="en",
)
print(f"BERTScore F1: {F1.mean():.4f}")
# Handles semantic equivalence that BLEU would score near-zero
```

Cost: requires a BERT forward pass per example. For large-scale evaluation this adds up.

### Perplexity

$$\text{PPL}(X) = \exp\!\left(-\frac{1}{T} \sum_{t=1}^{T} \log P_\theta(x_t \mid x_{<t})\right)$$

Lower perplexity means the model assigns higher probability to the test sequence. Measures how well the model fits a distribution — not how good the model's outputs are for a task. Useful for: comparing model checkpoints during pretraining, measuring domain adaptation. Not useful for: comparing models of different families, since they use different tokenizers and vocabularies.

---

## 7. LLM-as-Judge: Scalable Evaluation With Known Biases

**The problem with human evaluation**: it requires annotators, is slow (hours per task), is expensive, and is hard to reproduce exactly. For rapid iteration during development, you need something faster.

**The core insight**: a capable LLM can assess response quality on many dimensions — factual accuracy, clarity, helpfulness, appropriateness — better than automated metrics, and much faster than humans. The cost is that the judge model has its own biases.

### Absolute scoring

```python
import openai, json

def llm_judge_absolute(question: str, answer: str, criteria: list[str]) -> dict:
    criteria_str = "\n".join(f"- {c}" for c in criteria)
    prompt = f"""Rate the following answer on each criterion from 1 (poor) to 5 (excellent).

Question: {question}
Answer: {answer}

Criteria:
{criteria_str}

Return JSON: {{"scores": {{"criterion": score}}, "reasoning": "..."}}"""

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)
```

### Pairwise comparison

More reliable than absolute scoring because it avoids the need to calibrate a scale. The judge only needs to decide which of two responses is better:

```python
def llm_judge_pairwise(question: str, answer_a: str, answer_b: str) -> str:
    prompt = f"""Which answer is better? Consider accuracy, helpfulness, and clarity.

Question: {question}
Answer A: {answer_a}
Answer B: {answer_b}

Reply with only 'A', 'B', or 'tie'."""

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()
```

### Known biases in LLM judges

| Bias | What it does | Mitigation |
| :--- | :--- | :--- |
| Self-preference | Model favors outputs stylistically similar to its own | Use a different model family as judge |
| Position bias | Prefers the first answer when presented A vs B | Run both orderings (A,B) and (B,A), take majority |
| Verbosity bias | Prefers longer answers regardless of quality | Explicit instruction: "do not consider length" |
| Sycophancy | Agrees when prompted "isn't answer A clearly better?" | Use neutral, opinionated-free prompts |

**Position bias is the most dangerous for systematic evaluation**: if you always put candidate A first, the judge will be biased toward A. Always randomize order and average both orderings.

---

## 8. LMSYS Chatbot Arena: Ground-Truth Human Preference at Scale

**The problem with curated benchmarks**: they measure performance on a fixed set of questions chosen by researchers. Real users ask different questions, care about different things, and have different thresholds for what counts as "good." Benchmark performance may not predict user satisfaction.

**The design**: anonymous side-by-side comparisons. Users interact with two unknown models simultaneously, then vote on which was better. Elo ratings are computed from win/loss outcomes.

$$\text{Expected score}(A) = \frac{1}{1 + 10^{(R_B - R_A)/400}}$$

**Why it matters**: Arena captures preferences over the actual distribution of questions users ask — not researcher-designed problems. It measures tone, formatting, personality, and response style in addition to accuracy. Models that score high on MMLU but have awkward response styles may rank lower in Arena than their benchmark scores predict.

**What breaks**: Arena is subject to selection bias — the user population that visits the Arena website is not representative of all users or all deployment domains. A model optimized for Arena users may not be optimal for medical or legal applications. Also, strategic voting (users who know which model they're testing) can inflate ratings.

---

## 9. Benchmark Contamination: The Validity Threat Nobody Likes Talking About

**The problem**: a test that has been in the training set is not a test. It's a recall exercise. If GSM8K problems appeared verbatim on math tutoring forums crawled into Common Crawl, and your model trained on Common Crawl, your GSM8K score measures memorization as much as reasoning.

**Why this is hard to detect**: you'd need to check every test example against the full training corpus. Most organizations don't publicly release their training data, making independent verification impossible.

**Detection methods available without training data access**:

1. **Canonical vs. paraphrased**: test on semantically equivalent rephrased versions of benchmark questions. If performance drops significantly on the paraphrase but not the original, contamination is likely.

2. **Chronological splits**: train only on data before a cutoff date; evaluate on questions released after. The model can't have seen questions that didn't exist when training data was collected.

3. **Membership inference**: probe whether the model assigns unusually high probability to benchmark questions relative to similarly-constructed non-benchmark questions. High probability is evidence (not proof) of memorization.

4. **Novel benchmarks**: benchmarks released after a model's training cutoff are less likely to be contaminated. LiveCodeBench (continuously updated coding problems) and similar "living" benchmarks are designed to resist contamination.

**The practical implication**: treat benchmark numbers as upper bounds on true capability, especially for widely-publicized benchmarks. When two models score within 1–2% of each other on a contaminated benchmark, the difference is likely noise.

---

## 10. Domain-Specific Evaluation: Why Standard Benchmarks Fail in Production

**The problem**: a model deployed for medical documentation is not evaluated by how well it solves grade-school math problems. MMLU has a medicine subset, but clinical documentation requires accuracy on very specific factual claims, adherence to formatting standards, and avoidance of harmful advice — none of which MMLU tests.

**When to build a custom eval**:
- Errors have asymmetric costs (a false negative in a cancer screening context is far worse than a false positive)
- Facts are highly specialized (cardiology drug interactions vs. general medical knowledge)
- Style and format requirements are strict

### RAG evaluation (RAGAS)

For retrieval-augmented generation, you need to separately evaluate: did the retriever find relevant chunks? Did the model use those chunks faithfully? Did the answer actually address the question?

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset

eval_data = Dataset.from_dict({
    "question": questions,
    "answer": model_answers,
    "contexts": retrieved_contexts,   # list of lists of strings
    "ground_truth": reference_answers,
})

results = evaluate(eval_data, metrics=[
    faithfulness,         # Is the answer grounded in the retrieved context?
    answer_relevancy,     # Does the answer address the question?
    context_precision,    # Are the retrieved chunks actually relevant?
    context_recall,       # Does the context contain the necessary information?
])
```

Faithfulness is the most important metric for safety-critical applications — a model that makes up information not in the context is dangerous regardless of how relevant or fluent the answer is.

---

## 11. Evaluation Design Principles

**Never evaluate on the training distribution.** This is obvious but violated constantly — test sets that were collected before the training cutoff, released publicly, and then crawled into training data are compromised.

**Use multiple complementary metrics.** Task accuracy tells you if the model is right; LLM-judge quality tells you if it's helpful and well-formatted; safety evaluation tells you if it's harmful. Any single metric is gameable.

**Calibration matters more than accuracy for decision-critical uses.** A model that says "90% confident" should be right 90% of the time. A model with 80% accuracy but perfect calibration is more useful in a high-stakes setting than a model with 85% accuracy that is confidently wrong 30% of the time.

**Slice by subgroup, not just average.** Aggregate performance hides demographic disparities, domain-specific failures, and distribution-shift vulnerabilities. A model with 85% average accuracy might have 60% accuracy on the specific input types that appear in your production traffic.

**Set a regression baseline before each deployment.** Verify the new model doesn't regress on categories the current model handles well:

```python
def regression_check(baseline_scores: dict, new_scores: dict, threshold: float = 0.02) -> list:
    """Returns categories where the new model regresses by more than threshold."""
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

A 2% regression on a critical category can matter far more than a 5% improvement on aggregate metrics.

## Flashcards

**When should you build a custom eval instead of relying on standard benchmarks?** #flashcard
When errors have asymmetric costs (e.g., false negatives in cancer screening are far worse than false positives), when facts are highly specialized (e.g., cardiology drug interactions vs. general medical knowledge), or when style/format requirements are strict.

**Why does BLEU correlate poorly with human judgment on open-ended generation?** #flashcard
BLEU rewards lexical n-gram overlap, not semantic correctness — "The dog bit the man" and "The man bit the dog" score identically against either as reference. Treat it as a filter for very bad outputs, not an evaluator of quality.

**What is pass@k and why is pass@10 more informative than pass@1 for coding benchmarks?** #flashcard
Pass@k is the probability that at least one of k generated samples passes all unit tests. Pass@1 measures raw first-try accuracy; pass@10 measures whether the model can find a correct solution given multiple attempts, which better reflects real usage where a developer can try again.

**Why is position bias the most dangerous bias in LLM-as-judge evaluation, and how do you mitigate it?** #flashcard
A judge tends to prefer whichever answer is shown first, so a fixed ordering (always candidate A first) silently biases every comparison toward A. Mitigate by randomizing order and running both (A,B) and (B,A), then averaging or taking the majority.

**What is benchmark contamination, and what's the most reliable way to detect it without access to the training data?** #flashcard
Contamination is when test examples leaked into training data, turning the test into a recall exercise rather than a measure of capability. Since training data is rarely public, use proxies: test on paraphrased (not verbatim) versions of benchmark questions, use chronological splits (train before a cutoff, test on newer data), or rely on continuously-updated "living" benchmarks like LiveCodeBench.

**Why does the LMSYS Chatbot Arena Elo ranking sometimes disagree with MMLU rankings?** #flashcard
Arena measures real user preference across the actual distribution of questions people ask, including tone, formatting, and personality — not just factual accuracy on curated questions. A model can score high on MMLU but have an awkward or unhelpful response style that users rank lower in head-to-head comparisons.

**Why is calibration sometimes more important than raw accuracy in high-stakes deployments?** #flashcard
A well-calibrated model that says "90% confident" is right 90% of the time, so its confidence is actionable. A model with higher accuracy but poor calibration can be confidently wrong, which is more dangerous in decision-critical settings even though its average accuracy looks better.
