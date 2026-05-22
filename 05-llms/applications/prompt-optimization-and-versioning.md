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

## Canonical Interview Q&As

**Q: How would you set up a systematic evaluation pipeline for LLM prompts?**  
A: Four components: (1) Eval set — collect 200–500 examples representative of production traffic, with ground-truth outputs for each; split into dev (iterate) and held-out test (final evaluation only, to avoid over-fitting prompts to it); (2) Metrics — task-specific metrics (F1, BLEU, exact match) for structured outputs, LLM-as-judge for open-ended generation; always run both and calibrate judge scores against human labels; (3) Statistical rigor — use Wilcoxon signed-rank test to check if prompt A vs B difference is statistically significant; require p < 0.05 before promoting; (4) Regression suite — run evaluation on every prompt change before merging, flagging if any category of examples degrades by > 3% even if overall score improves.

**Q: What is DSPy and how is it different from manual prompt engineering?**  
A: DSPy (Declarative Self-improving Python) separates the program logic from the prompt text. You write a program that describes what transformations to apply (chain-of-thought, retrieval, multi-hop reasoning), and DSPy automatically finds the best prompt instructions and few-shot examples by running an optimizer over your training data. The key difference from manual engineering: DSPy treats prompts as learnable parameters, not as hand-crafted strings. It can simultaneously optimize prompts across multiple chained modules, whereas manual engineering optimizes each prompt in isolation, missing interactions. Practical tradeoff: DSPy requires labeled training data and is slower to set up, but produces more robust pipelines that degrade gracefully when the underlying model changes.

**Q: How do you prevent over-fitting to your eval set when optimizing prompts?**  
A: Same principles as ML model evaluation: (1) strict train/test split — iterate only on dev set, evaluate final prompt on test set once; (2) stratify eval set by difficulty and category so improvements generalize; (3) set a minimum sample size for significance — any change claiming >2% improvement needs n ≥ 200 examples; (4) monitor calibration — if your eval set samples from 3 months ago, check if prompt performance on recent traffic matches; (5) use cross-validation for few-shot selection — never include the test example's neighbors in its own few-shot context; (6) track blind test failures — maintain a "hard examples" set that was never used for optimization, and periodically evaluate against it.
