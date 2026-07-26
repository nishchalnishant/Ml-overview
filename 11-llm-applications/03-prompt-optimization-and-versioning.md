---
module: LLM Applications
topic: Applications
subtopic: Prompt Optimization And Versioning
status: unread
tags: [llms, ml, applications-prompt-optimizati]
---
# Prompt Optimization and Versioning

Systematic prompt engineering — moving from ad-hoc manual tuning to reproducible, evaluated, optimized prompt pipelines.

---

## 1. Why Systematic Prompt Engineering

**The problem with ad-hoc prompting:**
- "Improved" prompts aren't validated on held-out data
- No version control — can't roll back when a prompt update breaks downstream tasks
- No measurement — model performance measured informally or not at all
- Duplicated effort — multiple teams write variations of the same prompts

**Systematic approach:**
```
Define task → Collect eval set → Baseline prompt → 
Auto-optimize (DSPy / APE) → Human review → 
A/B test in production → Version in registry
```

---

## 2. Prompt Versioning

**Treat prompts as code with semantic versioning:**

```python
# prompt_registry.py
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class PromptVersion:
    name: str
    version: str          # semver: "1.2.3"
    template: str
    model: str
    temperature: float
    created_at: datetime
    eval_score: Optional[float]
    author: str
    parent_version: Optional[str]  # lineage tracking

class PromptRegistry:
    def __init__(self, backend="s3"):
        self.store = {}  # simplified; use S3/DB in production
    
    def register(self, prompt: PromptVersion):
        key = f"{prompt.name}:{prompt.version}"
        self.store[key] = prompt
        return key
    
    def get(self, name: str, version: str = "latest"):
        if version == "latest":
            # Find highest semver for this name
            versions = [v for k, v in self.store.items() if k.startswith(name)]
            return sorted(versions, key=lambda x: x.version)[-1]
        return self.store[f"{name}:{version}"]
    
    def promote(self, name: str, version: str, stage: str):
        """Promote to staging/production."""
        prompt = self.get(name, version)
        assert prompt.eval_score is not None, "Cannot promote without eval score"
        assert prompt.eval_score >= 0.8, f"Eval score {prompt.eval_score} below threshold"
        # Tag as production
        self.store[f"{name}:production"] = prompt
```

**Directory structure:**
```
prompts/
├── summarization/
│   ├── v1.0.0.yaml      # original
│   ├── v1.1.0.yaml      # added CoT instruction
│   ├── v2.0.0.yaml      # complete rewrite
│   └── production -> v1.1.0.yaml  # symlink
├── classification/
│   └── ...
└── eval_sets/
    └── summarization_100.jsonl
```

---

## 3. LLM-as-Judge Evaluation

**Use a stronger LLM to score outputs at scale.**

```python
JUDGE_PROMPT = """You are evaluating the quality of a text summary.

Original text: {source}
Summary: {summary}

Rate the summary on these dimensions (1-5):
1. Faithfulness: Does the summary accurately represent the source? No hallucinations?
2. Completeness: Does it cover the key points?
3. Conciseness: Is it appropriately brief without losing key information?

Respond in JSON:
{{"faithfulness": <1-5>, "completeness": <1-5>, "conciseness": <1-5>, "reasoning": "<1-2 sentences>"}}"""

def llm_judge_eval(
    predictions: list[dict],
    judge_model: str = "claude-opus-4-7",
    n_samples: int = 100
):
    """Evaluate model outputs using an LLM judge."""
    scores = []
    for sample in predictions[:n_samples]:
        prompt = JUDGE_PROMPT.format(
            source=sample["source"],
            summary=sample["prediction"]
        )
        response = call_llm(judge_model, prompt)
        result = json.loads(response)
        scores.append(result)
    
    # Aggregate
    df = pd.DataFrame(scores)
    return {
        "mean_faithfulness": df["faithfulness"].mean(),
        "mean_completeness": df["completeness"].mean(),
        "mean_conciseness": df["conciseness"].mean(),
        "overall": df[["faithfulness", "completeness", "conciseness"]].mean(axis=1).mean()
    }
```

**LLM-as-judge pitfalls:**
- **Position bias:** judges favor the first response in pairwise comparisons (randomize order and average)
- **Verbosity bias:** judges favor longer outputs even when they add no value
- **Self-preference:** a model judges its own outputs more favorably
- **Calibration:** map judge scores to human labels using a held-out correlation set

**Mitigation:**
```python
def pairwise_judge(source, response_a, response_b, judge_model):
    """Pairwise comparison with position debiasing."""
    # AB order
    score_ab = judge_pairwise(source, response_a, response_b, judge_model)
    # BA order
    score_ba = judge_pairwise(source, response_b, response_a, judge_model)
    
    # Average with position flip
    a_wins = (score_ab == "A") + (score_ba == "B")
    b_wins = (score_ab == "B") + (score_ba == "A")
    
    if a_wins > b_wins:
        return "A"
    elif b_wins > a_wins:
        return "B"
    return "TIE"
```

---

## 4. Automatic Prompt Optimization (APE / DSPy)

### APE (Automatic Prompt Engineer)

Generate N candidate prompts, evaluate on held-out set, select best.

```python
def ape_optimize(task_description, examples, eval_fn, n_candidates=20):
    """Automatic Prompt Engineer - generate and select best prompt."""
    
    # Step 1: Generate candidate prompts using LLM
    generation_prompt = f"""Generate {n_candidates} different instruction prompts for this task:
    Task: {task_description}
    Example inputs: {examples[:3]}
    
    Each prompt should approach the task from a different angle.
    Respond with one prompt per line."""
    
    candidates = call_llm("claude-sonnet-4-6", generation_prompt).split("\n")
    
    # Step 2: Evaluate each candidate
    candidate_scores = []
    for candidate in candidates:
        score = eval_fn(candidate, examples)
        candidate_scores.append((candidate, score))
    
    # Step 3: Return top candidate
    best_prompt, best_score = max(candidate_scores, key=lambda x: x[1])
    return best_prompt, best_score
```

### DSPy (Declarative Self-improving Python)

DSPy separates program logic from prompts — optimize the whole pipeline.

```python
import dspy

class FraudExplanationSignature(dspy.Signature):
    """Explain why a transaction was flagged as fraudulent."""
    transaction_features: str = dspy.InputField()
    fraud_score: float = dspy.InputField()
    explanation: str = dspy.OutputField(desc="Clear explanation for the customer")

class FraudExplainer(dspy.Module):
    def __init__(self):
        self.explain = dspy.ChainOfThought(FraudExplanationSignature)
    
    def forward(self, transaction_features, fraud_score):
        return self.explain(
            transaction_features=transaction_features,
            fraud_score=fraud_score
        )

# Compile with optimizer
teleprompter = dspy.BootstrapFewShot(metric=faithfulness_metric)
optimized_module = teleprompter.compile(
    FraudExplainer(),
    trainset=training_examples,
    valset=val_examples
)

# DSPy automatically generates few-shot examples and instructions
print(optimized_module.explain.demos)  # auto-selected few-shot examples
```

**DSPy optimizers comparison:**

| Optimizer | Strategy | Best for |
|---|---|---|
| BootstrapFewShot | Select best few-shot examples | Any task, fast |
| MIPRO | Generate + select instructions + few-shots | High-quality, slower |
| BootstrapFinetune | Distill into fine-tuned model | Latency-sensitive |
| COPRO | Coordinate prompt and chain optimization | Multi-hop reasoning |

---

## 5. Prompt Templates and Few-Shot Selection

**Dynamic few-shot selection** — choose examples most similar to the current query:

```python
from sentence_transformers import SentenceTransformer
import numpy as np

class DynamicFewShotSelector:
    def __init__(self, example_pool, n_shots=5):
        self.examples = example_pool
        self.n_shots = n_shots
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Pre-encode all examples
        self.example_embeddings = self.encoder.encode(
            [ex["input"] for ex in example_pool]
        )
    
    def select(self, query: str) -> list[dict]:
        query_embedding = self.encoder.encode([query])
        
        # Cosine similarity
        scores = np.dot(self.example_embeddings, query_embedding.T).flatten()
        scores /= (
            np.linalg.norm(self.example_embeddings, axis=1) * 
            np.linalg.norm(query_embedding)
        )
        
        # Top-k most similar (avoid identical copies)
        top_k_indices = scores.argsort()[-self.n_shots:][::-1]
        return [self.examples[i] for i in top_k_indices]
```

**Chain-of-thought effectiveness:**
- Simple tasks: direct answer prompting is sufficient (CoT adds latency, no benefit)
- Multi-step reasoning: CoT improves accuracy by 10–40% on benchmarks
- Classification: CoT + "Verify your answer" reduces errors significantly
- Rule: if the task requires >2 reasoning steps, try CoT

---

## 6. Evaluation Framework

```python
class PromptEvaluator:
    def __init__(self, eval_set_path: str):
        with open(eval_set_path) as f:
            self.eval_set = [json.loads(line) for line in f]
    
    def evaluate_prompt(
        self, 
        prompt_template: str,
        model: str,
        metrics: list[str] = ["exact_match", "f1", "llm_judge"]
    ) -> dict:
        predictions = []
        for example in self.eval_set:
            prompt = prompt_template.format(**example["inputs"])
            prediction = call_llm(model, prompt)
            predictions.append({
                "input": example["inputs"],
                "expected": example["target"],
                "prediction": prediction
            })
        
        results = {}
        if "exact_match" in metrics:
            results["exact_match"] = np.mean([
                p["expected"].strip().lower() == p["prediction"].strip().lower()
                for p in predictions
            ])
        if "f1" in metrics:
            results["f1"] = np.mean([
                token_f1(p["expected"], p["prediction"])
                for p in predictions
            ])
        if "llm_judge" in metrics:
            results["llm_judge"] = llm_judge_eval(predictions)
        
        return results
    
    def compare_versions(self, v1_prompt: str, v2_prompt: str, model: str):
        """Statistical significance test for prompt comparison."""
        v1_results = [self._evaluate_single(v1_prompt, ex, model) for ex in self.eval_set]
        v2_results = [self._evaluate_single(v2_prompt, ex, model) for ex in self.eval_set]
        
        from scipy.stats import wilcoxon
        stat, p_value = wilcoxon(v1_results, v2_results)
        
        return {
            "v1_mean": np.mean(v1_results),
            "v2_mean": np.mean(v2_results),
            "delta": np.mean(v2_results) - np.mean(v1_results),
            "p_value": p_value,
            "significant": p_value < 0.05
        }
```

---

## 7. Production Deployment

**Prompt A/B testing:**
```python
class PromptRouter:
    def __init__(self, registry: PromptRegistry):
        self.registry = registry
        self.experiments = {}  # name → {control: v1, treatment: v2, traffic_split: 0.1}
    
    def route(self, prompt_name: str, user_id: str) -> PromptVersion:
        if prompt_name in self.experiments:
            exp = self.experiments[prompt_name]
            # Deterministic assignment by user_id hash
            if hash(user_id) % 100 < exp["traffic_split"] * 100:
                return self.registry.get(prompt_name, exp["treatment"])
        return self.registry.get(prompt_name, "production")
```

**Rollback triggers:**
- LLM-judge score drops > 5%
- User negative feedback rate increases
- Latency P99 exceeds SLA (different prompt → different output length → different TPOT)
- Error rate (JSON parse failures, refused completions) spikes

---

## Interview Angles

**Q: How do you prevent over-fitting when iteratively optimizing a prompt?**
A: Just like in traditional ML, prompt engineering requires a strict train/test split. You must maintain a "dev set" (where you iteratively tweak the prompt and check metrics) and a "test set" (which you only evaluate against when you think you are done). If you tweak a prompt to fix a specific failure in production, you must run it against a regression suite (the dev set) to ensure that the fix didn't degrade performance on other categories. Furthermore, you need statistical significance — if a prompt improves accuracy by 2% on a 50-example eval set, that is likely just noise.

### Q: Compare manual prompt engineering with automatic prompt optimization frameworks like DSPy. When would you invest in DSPy? [Hard]
A: **Manual engineering** treats prompts as hand-crafted strings. It is fast to start but hits a ceiling quickly: if you have a multi-step pipeline (e.g., Retriever -> Reranker -> Summarizer), manually tweaking the Summarizer's prompt might break because it relies on the exact format the Retriever outputs. **DSPy** separates the *program logic* (the pipeline of modules) from the *parameters* (the specific prompt instructions and few-shot examples). It uses an optimizer to automatically compile the best prompt instructions and select the best few-shot examples from your training data to maximize a metric. You invest in DSPy when you have a complex, multi-stage agentic workflow, when you frequently swap underlying models (which breaks hand-crafted prompts), and when you have high-quality labeled training data to guide the optimizer.

**Cross-questions to expect:**

- *You mentioned using LLM-as-a-judge for evaluation. What are the common biases of LLM judges, and how do you mitigate them?* -> The three major biases are **Position Bias** (favoring the first option in a pairwise comparison), **Verbosity Bias** (favoring longer answers even if they are fluff), and **Self-Preference** (a model rating its own outputs higher). Mitigate position bias by evaluating both A-B and B-A orderings and averaging. Mitigate verbosity by explicitly penalizing length in the judge's rubric. Always calibrate the LLM judge against a human-annotated baseline using Pearson correlation.
- *When should you use Chain-of-Thought (CoT) prompting, and when is it a bad idea?* -> CoT is essential for tasks requiring >2 reasoning steps (math, logic, multi-hop QA) because it forces the model to allocate compute (tokens) to intermediate steps. It is a bad idea for simple extraction or classification tasks because it drastically increases latency (Time-To-First-Token and total generation time) without improving accuracy.

**Trap:** Claiming that a prompt "works better" without a defined metric or eval set. Staff-level engineers never deploy a prompt change based on "vibes" or a few manual tests. They treat prompts as code: version-controlled, regression-tested against a dataset, and rolled out via A/B testing with latency and error-rate monitoring.

## Flashcards

**Why is systematic prompt engineering needed instead of ad-hoc tuning?** #flashcard
Ad-hoc "improvements" aren't validated on held-out data, aren't version-controlled (no rollback when a change breaks downstream tasks), aren't measured consistently, and lead to duplicated effort across teams. Systematic engineering treats prompts like code: define task → build eval set → baseline → optimize (DSPy/APE) → human review → A/B test → version in a registry.

**What are the main pitfalls of using an LLM as a judge, and how do you mitigate them?** #flashcard
Position bias (judges favor the first response in pairwise comparisons — fix by randomizing/averaging both orderings), verbosity bias (favors longer outputs regardless of quality), self-preference (a model rates its own outputs higher), and calibration drift (judge scores must be validated against a held-out set of human labels).

**When does chain-of-thought prompting help vs. hurt?** #flashcard
For simple/single-step tasks, CoT just adds latency with no accuracy benefit — direct answering is sufficient. For multi-step reasoning it improves accuracy by 10–40%. Rule of thumb: use CoT only when the task requires more than ~2 reasoning steps; for classification, adding "verify your answer" catches errors cheaply.

**What triggers a prompt rollback in production?** #flashcard
LLM-judge score drops more than 5%, user negative-feedback rate rises, latency P99 exceeds SLA (a new prompt changes output length and therefore TPOT), or error rate spikes (JSON parse failures, refused completions).

**What is APE (Automatic Prompt Engineer) and how does it work?** #flashcard
Generate N candidate prompts with an LLM from a task description, evaluate each on a held-out example set with an eval function, and keep the best-scoring candidate. Simple, fast, but only as good as the eval function and candidate diversity.

**What does DSPy optimize that manual prompt engineering can't?** #flashcard
DSPy separates program logic from prompt text and treats prompts/few-shot examples as learnable parameters optimized end-to-end across chained modules, rather than hand-tuned in isolation per prompt. This lets multi-hop pipelines improve jointly, at the cost of needing labeled training data and more setup time.

**How do you avoid over-fitting a prompt to its eval set?** #flashcard
Use a strict dev/test split (iterate only on dev, evaluate final choice once on held-out test), require statistical significance (Wilcoxon signed-rank, p < 0.05, n ≥ 200) before promoting a change, stratify by difficulty/category, and keep a never-touched "hard examples" set for periodic blind checks.
