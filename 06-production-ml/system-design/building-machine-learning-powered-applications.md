---
module: Production Ml
topic: System Design
subtopic: Building Machine Learning Powered Applications
status: unread
tags: [productionml, ml, system-design-building-machine]
---
# Building Machine Learning Powered Applications

> **See also:** [MLOps](../mlops.md) | [Deployment Patterns](../deployment-patterns.md) | [Model Governance](../model-governance.md)

*(Based on Ameisen, "Building Machine Learning Powered Applications")*

---

## 1. The Problem with Starting from the Model

### The Problem

An ML team spends three months building and training a model. They hand it to product. The product team says: "This doesn't solve the problem we have." The model was technically excellent and practically useless.

### The Core Insight

The model is not the product. The product is a system that changes what a user can do. Start from the user outcome and work backward to the model, not forward from the data.

### The Mechanics

**The backward design sequence:**

1. **Concrete use case:** What specific task will the user do differently because of this system?
2. **Decision or prediction:** What decision is being made, or what quantity is being predicted?
3. **Minimum useful output:** What is the simplest output that still makes the decision better?
4. **Data required:** What training signal would let us learn that mapping?
5. **Model:** What architecture learns that mapping from that training signal?

**The product spec question:** "If the model were replaced by a human expert, what would that expert need to know, and what would they produce?"

Answer this before touching a dataset.

### What Breaks

**Starting from available data:** you build a model that predicts what the data easily supports, not what the product needs.

**Starting from the model architecture:** you constrain the problem to what the architecture handles well, rather than what the user needs.

**Defining success as model accuracy:** a model that is 95% accurate but makes its errors on the 5% of cases that matter most to the user has failed.

---

## 2. Building a Simple End-to-End System First

### The Problem

A team spends six months on feature engineering and model selection, then discovers at deployment that their serving infrastructure cannot support the latency requirement. The entire design needs to change.

### The Core Insight

Build a complete end-to-end pipeline as fast as possible — even if every component is naive. The pipeline reveals the real engineering constraints. Then iterate.

### The Mechanics

**Ameisen's rule:** have a working end-to-end system before doing any sophisticated modeling.

**Minimum viable pipeline:**

```python
# Stage 1: Dumbest possible end-to-end system
# Goal: find every integration point and constraint before optimizing anything

class MVPWritingAssistant:
    """
    ML Writing Assistant — MVP version.
    Predicts whether a sentence is too complex and suggests simplification.
    """

    def __init__(self):
        # Start with a rule — no model needed yet
        # This establishes the API contract that the real model will fulfill
        self.max_words_per_sentence = 25

    def predict(self, sentence: str) -> dict:
        """The interface is fixed here. The model behind it can be replaced."""
        word_count = len(sentence.split())
        is_complex = word_count > self.max_words_per_sentence

        return {
            "is_complex": is_complex,
            "confidence": 1.0 if is_complex else 0.0,
            "suggestion": "Consider breaking this sentence." if is_complex else None,
            "word_count": word_count
        }

# Stage 2: Replace the rule with a model
# The interface (predict method signature) does not change
class MLWritingAssistant:
    def __init__(self, model_path: str):
        import joblib
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(model_path + ".vectorizer")

    def predict(self, sentence: str) -> dict:
        features = self.vectorizer.transform([sentence])
        proba = self.model.predict_proba(features)[0]
        is_complex = proba[1] > 0.5

        return {
            "is_complex": bool(is_complex),
            "confidence": float(proba[1]),
            "suggestion": self._generate_suggestion(sentence) if is_complex else None,
            "word_count": len(sentence.split())
        }
```

**What the MVP reveals:**
- API contract between model and product
- Actual latency requirements (rule runs in microseconds; model in milliseconds)
- Data format issues in the real input stream
- Edge cases: empty strings, very long documents, non-English input
- Downstream failures: what happens when the model returns `None`?

### What Breaks

**Polishing the model before building the integration:** you discover integration problems at the worst possible time.

**No rule-based baseline:** you cannot tell if the model is adding value over the rule. Baseline accuracy is the minimum a model must beat to justify its operational cost.

---

## 3. Iterating on Data Before Iterating on Models

### The Problem

A model plateaus at 78% accuracy. The team tries 12 different architectures over two months. Accuracy stays at 78%. Then someone examines the training data and finds that 15% of labels are wrong. Fixing the labels takes two days and raises accuracy to 87%.

### The Core Insight

Model architecture contributes incrementally to performance. Data quality contributes multiplicatively. Fix data first.

### The Mechanics

**Data debugging workflow:**

```python
class DataDebugger:
    """
    Systematic workflow for finding data problems before trying new models.
    """

    def run_full_audit(self, X_train, y_train, X_val, y_val, model) -> dict:
        issues = {}

        # 1. Find hard examples: cases where the model is most wrong
        val_preds = model.predict_proba(X_val)[:, 1]
        errors = abs(val_preds - y_val)
        hardest_idx = errors.argsort()[-100:][::-1]  # top-100 hardest cases
        issues["hard_examples"] = X_val.iloc[hardest_idx]

        # 2. Check label consistency: find similar examples with different labels
        issues["label_conflicts"] = self._find_label_conflicts(X_train, y_train)

        # 3. Check representation: are some subgroups underrepresented?
        issues["representation"] = self._check_representation(X_train, y_train)

        # 4. Check for data leakage
        issues["leakage_candidates"] = self._find_leakage(X_train, y_train, X_val, y_val)

        return issues

    def _find_label_conflicts(self, X, y, similarity_threshold=0.95):
        """Find near-duplicate examples with different labels."""
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        sim_matrix = cosine_similarity(X)
        conflicts = []
        for i in range(len(X)):
            for j in range(i+1, len(X)):
                if sim_matrix[i, j] > similarity_threshold and y.iloc[i] != y.iloc[j]:
                    conflicts.append((i, j, sim_matrix[i, j]))
        return conflicts

    def _find_leakage(self, X_train, y_train, X_val, y_val):
        """
        Red flag: a feature that correlates >0.9 with the label.
        Often indicates the feature is derived from the label.
        """
        import pandas as pd
        correlations = X_train.corrwith(y_train).abs()
        return correlations[correlations > 0.9].index.tolist()
```

**Data collection priorities:**

| Issue | Solution | Priority |
|-------|----------|----------|
| Label errors | Re-label, measure inter-annotator agreement | Highest |
| Missing subgroups | Collect targeted examples | High |
| Distribution mismatch | Collect data matching deployment distribution | High |
| Insufficient data | More data in worst-performing segments | Medium |
| Feature noise | Clean or bin noisy features | Medium |

### What Breaks

**More data for already-covered distributions:** doubling training data rarely doubles performance if the model already generalizes on that distribution. Add data in segments where the model fails.

**Fixing data without re-evaluating:** fixing labels in training without also auditing validation set labels produces an optimistic accuracy report.

---

## 4. Writing Evaluations Before Knowing the Model

### The Problem

A team ships a text summarization model. Users complain the summaries are wrong. The team built one metric: ROUGE. ROUGE is high. ROUGE measures n-gram overlap, not factual correctness. The metric optimized for was not the thing users needed.

### The Core Insight

Evaluation is a product decision, not a modeling decision. Define it before training.

### The Mechanics

**Evaluation hierarchy:**

```
Business KPI (what we actually care about)
    ↓
Product metric (measurable proxy for business KPI)
    ↓
Model metric (optimizable surrogate for product metric)
    ↓
Slice evaluations (model metric broken down by segment)
```

**Example: writing assistant**

```
Business KPI:       User documents rated "clear" by readers increase by 15%
Product metric:     Suggestion acceptance rate in A/B test
Model metric:       F1 on held-out complex/simple sentence classification
Slice evaluations:  F1 broken down by domain (legal, medical, general), sentence length bucket
```

**Building a slice evaluation framework:**

```python
class SliceEvaluator:
    """
    Evaluate model performance broken down by data slices.
    Forces examination of performance beyond overall aggregate.
    """

    def __init__(self, model, test_data: pd.DataFrame, label_col: str):
        self.model = model
        self.data = test_data
        self.label_col = label_col

    def evaluate_all_slices(self, slice_definitions: dict) -> pd.DataFrame:
        """
        slice_definitions: {slice_name: (column, value_or_condition)}
        """
        from sklearn.metrics import f1_score, roc_auc_score
        results = []

        # Overall baseline
        all_preds = self.model.predict(self.data.drop(columns=[self.label_col]))
        all_labels = self.data[self.label_col]
        results.append({
            "slice": "overall",
            "n": len(self.data),
            "f1": f1_score(all_labels, all_preds),
            "auc": roc_auc_score(all_labels, all_preds)
        })

        # Each slice
        for slice_name, (col, condition) in slice_definitions.items():
            if callable(condition):
                mask = self.data[col].apply(condition)
            else:
                mask = self.data[col] == condition

            slice_data = self.data[mask]
            if len(slice_data) < 30:
                results.append({"slice": slice_name, "n": len(slice_data), "f1": None, "auc": None,
                                 "note": "insufficient sample"})
                continue

            preds = self.model.predict(slice_data.drop(columns=[self.label_col]))
            labels = slice_data[self.label_col]
            results.append({
                "slice": slice_name,
                "n": len(slice_data),
                "f1": f1_score(labels, preds),
                "auc": roc_auc_score(labels, preds)
            })

        return pd.DataFrame(results).sort_values("f1")
```

**Why slice evaluation before training:** defining slices before training prevents you from constructing slices that flatter the model's known strengths.

### What Breaks

**Optimizing a single aggregate metric:** high average F1 with catastrophic failure on one segment is invisible in the aggregate.

**Post-hoc evaluation:** after training, teams find slices where the model performs well and emphasize those. Pre-defined slices prevent this selection bias.

---

## 5. Debugging Failures Systematically

### The Problem

A model produces wrong outputs. "The model is wrong" is not actionable. The wrong outputs come from one of four places, and each requires a different fix.

### The Core Insight

Every model failure is attributable to one of: wrong framing, bad data, bad features, or bad model. Diagnose before fixing.

### The Mechanics

**Failure attribution taxonomy:**

```python
class FailureDiagnostic:
    """
    Walk through the attribution tree to find the root cause.
    """

    def diagnose(self, case: dict) -> str:
        """
        case: {input, true_label, model_prediction, model_confidence}
        Returns the attributed failure category.
        """

        # Step 1: Is this case even within scope?
        if not self._is_in_scope(case["input"]):
            return "FRAMING: Input is out-of-distribution for the problem definition"

        # Step 2: Is the label correct?
        if self._label_is_ambiguous(case["input"], case["true_label"]):
            return "DATA: Label is ambiguous or incorrect"

        # Step 3: Do features capture what is needed?
        if not self._features_are_sufficient(case["input"]):
            return "FEATURES: Feature set does not contain signal needed for this case"

        # Step 4: Would more training examples of this type help?
        if self._is_underrepresented(case["input"]):
            return "DATA: This pattern is underrepresented in training data"

        # Step 5: Model-specific failure
        return "MODEL: Model is not learning the right pattern; consider architecture change"

    def _is_in_scope(self, input_data) -> bool:
        """Is this input within the distribution the model was designed for?"""
        # Check against feature distribution bounds, language, domain
        raise NotImplementedError

    def _label_is_ambiguous(self, input_data, label) -> bool:
        """Would two expert annotators agree on this label?"""
        # If labeling instructions don't cover this case, it's a data problem
        raise NotImplementedError

    def _features_are_sufficient(self, input_data) -> bool:
        """If a human saw only these features, could they make the right decision?"""
        # Feature sufficiency test — use human judgment
        raise NotImplementedError

    def _is_underrepresented(self, input_data) -> bool:
        """Is this pattern rare in training data?"""
        raise NotImplementedError
```

**The manual error analysis protocol:**

1. Take the top 100 hardest validation examples (highest loss)
2. For each, write down why the model failed in plain English
3. Group failures into categories
4. Each category suggests a specific fix:

| Error Category | Fix |
|----------------|-----|
| "Model misses domain-specific context" | Add domain features; collect more domain examples |
| "Edge case not in training data" | Targeted data collection |
| "Label is actually ambiguous" | Refine labeling criteria; reannotate |
| "Input is malformed (noise, encoding)" | Add preprocessing; filter during training |
| "Model predicts wrong class under time pressure" | Add temporal features |

### What Breaks

**Trying a new model architecture as the first response to failure:** architecture is rarely the root cause. Fix framing, data, and features first.

**Sampling random failures instead of hardest failures:** random sampling underrepresents rare failure modes. Always sort by loss magnitude.

---

## 6. Transitioning from Prototype to Product

### The Problem

A Jupyter notebook prototype produces excellent results. Moving it to production takes three months and the performance is different. The pipeline that ran sequentially and single-threaded in the notebook behaves differently under concurrency, streaming input, and different data formats.

### The Core Insight

Notebook code is analysis code. Production code is software. The transition requires rebuilding, not refactoring.

### The Mechanics

**Production checklist:**

```python
# Notebook prototype (what you built first)
def predict_sentiment(text):
    tokens = tokenizer.tokenize(text)
    return model.predict([tokens])[0]

# Production version (what is needed)
class SentimentPredictor:
    """
    Production-grade wrapper with all the things notebooks omit.
    """

    def __init__(self, model_path: str, tokenizer_path: str):
        import joblib
        self.model = joblib.load(model_path)
        self.tokenizer = joblib.load(tokenizer_path)
        self._validate_artifacts()

    def _validate_artifacts(self):
        """Fail fast: verify model artifacts are loadable and have expected interface."""
        test_input = "validation test"
        result = self.predict(test_input)
        assert "label" in result and "confidence" in result, \
            f"Model output schema mismatch: {result}"

    def predict(self, text: str) -> dict:
        # Input validation
        if not isinstance(text, str):
            raise TypeError(f"Expected str, got {type(text)}")
        if len(text) == 0:
            return {"label": "unknown", "confidence": 0.0, "reason": "empty_input"}
        if len(text) > 10000:
            text = text[:10000]  # truncate with documented limit

        try:
            tokens = self.tokenizer.tokenize(text)
            proba = self.model.predict_proba([tokens])[0]
            return {
                "label": "positive" if proba[1] > 0.5 else "negative",
                "confidence": float(max(proba)),
                "model_version": self.model.__version__
            }
        except Exception as e:
            # Fail gracefully — never propagate to the caller
            import logging
            logging.error(f"Prediction failed for input length {len(text)}: {e}")
            return {"label": "error", "confidence": 0.0, "reason": str(e)}

    def predict_batch(self, texts: list) -> list:
        """Batch prediction with individual error isolation."""
        return [self.predict(t) for t in texts]
```

**The three hardest production transitions:**

| Notebook Assumption | Production Reality | Fix |
|--------------------|-------------------|-----|
| Input is always well-formed | Input is corrupted, empty, wrong type | Explicit validation and graceful fallback |
| Single-threaded, sequential | Concurrent requests | Thread safety; per-request state |
| Model is always loaded | Model load fails, OOM, corruption | Health check endpoint; artifact validation on startup |

**Minimum production requirements:**
- Input schema validation (type, range, presence)
- Graceful error handling (never propagate raw exceptions to callers)
- Health check endpoint
- Request logging (for debugging production failures)
- Version information in every response

### What Breaks

**Copying notebook code directly into a Flask endpoint:** global state, mutable defaults, and exception propagation all produce non-deterministic failures under load.

**No input validation:** a single malformed request can crash the server, bringing down all concurrent requests.

---

## 7. Testing ML Systems

### The Problem

A model passes all unit tests. In production it makes correct predictions on clean input and silently produces garbage on slightly different input — different date format, slightly different encoding, one extra column in the feature vector. There were no tests for these cases.

### The Core Insight

ML systems require two levels of testing that traditional software does not: behavioral tests (does the model learn the right thing?) and integration tests (does the pipeline handle the full input distribution?).

### The Mechanics

**Three test categories:**

```python
import pytest

class TestSentimentModel:

    # Category 1: Unit tests — does the pipeline transform correctly?
    def test_tokenizer_handles_empty_string(self, tokenizer):
        result = tokenizer.tokenize("")
        assert result == [] or result is not None  # depends on spec

    def test_feature_scaling_bounded(self, scaler, raw_features):
        scaled = scaler.transform(raw_features)
        assert scaled.min() >= -3.0 and scaled.max() <= 3.0  # expect ~N(0,1) bounds

    # Category 2: Behavioral tests — does the model learn the right pattern?
    # These test invariances that should hold regardless of architecture
    def test_positive_examples_score_above_negative(self, model):
        """The model must rank clear positives above clear negatives."""
        clear_positives = [
            "This product is excellent, I love it",
            "Absolutely wonderful, highly recommend",
        ]
        clear_negatives = [
            "Terrible product, completely broken",
            "Worst purchase I have ever made",
        ]
        pos_scores = [model.predict(t)["confidence"] for t in clear_positives]
        neg_scores = [model.predict(t)["confidence"] for t in clear_negatives]
        assert min(pos_scores) > max(neg_scores), \
            f"Clear positives should outscore clear negatives"

    def test_prediction_is_monotonic_in_signal(self, model):
        """Adding more positive signal should not decrease positive score."""
        base = "Good product"
        stronger = "Very good product, excellent quality"
        assert model.predict(stronger)["confidence"] >= model.predict(base)["confidence"]

    # Category 3: Integration tests — does the full pipeline handle edge cases?
    def test_unicode_input(self, pipeline):
        result = pipeline.predict("très bien 👍 great product")
        assert result is not None and "label" in result

    def test_handles_very_long_input(self, pipeline):
        long_input = "good " * 5000  # 25k characters
        result = pipeline.predict(long_input)
        assert result["label"] in ["positive", "negative", "error"]

    def test_handles_concurrent_requests(self, pipeline):
        import concurrent.futures
        inputs = ["test sentence " + str(i) for i in range(100)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(pipeline.predict, inputs))
        assert all(r is not None for r in results)
        # Verify results are deterministic
        results2 = list(executor.map(pipeline.predict, inputs))
        assert results == results2
```

**Minimum test coverage for ML pipelines:**

| Test Type | What It Catches | Frequency |
|-----------|----------------|-----------|
| Schema tests | Feature columns missing, type changes | Every PR |
| Behavioral invariance tests | Model regression on key patterns | Every PR |
| Slice performance tests | Performance drop on key segments | Every training run |
| Integration tests | Pipeline edge cases | Every PR |
| Load tests | Latency under concurrent load | Pre-deployment |

### What Breaks

**Testing only the model, not the pipeline:** a preprocessing bug can silently change the input distribution without changing model test scores.

**Behavioral tests tied to specific predictions:** "sentence X must return 0.73" breaks on every retrain. Test orderings and invariances, not absolute scores.

---

## 8. Communicating Results

### The Problem

An ML team delivers a model. The stakeholder asks "Is it good enough?" The team answers with AUC = 0.84. The stakeholder does not know what that means relative to the business decision they need to make.

### The Core Insight

Metrics are not communication. The question stakeholders need answered is: "What can we do now that we could not do before, and what is the cost of the errors?"

### The Mechanics

**The calibration translation:**

```
DO NOT SAY:   "Our model achieves AUC of 0.84"
SAY INSTEAD:  "At the operating threshold we propose, the model correctly identifies 
               72% of fraudulent transactions while flagging only 8% of legitimate 
               ones for review. This means our fraud review team spends 40% of their 
               time on actual fraud rather than the current 12%."
```

**Error cost analysis:**

```python
def compute_decision_economics(
    confusion_matrix: dict,
    cost_table: dict
) -> dict:
    """
    Translate classification metrics into business economics.

    confusion_matrix: {tp, fp, tn, fn}
    cost_table: {
        "tp_value": revenue saved per caught fraud,
        "fp_cost": manual review cost per false alarm,
        "fn_cost": loss per missed fraud,
        "tn_value": 0 (correct rejection is free)
    }
    """
    tp = confusion_matrix["tp"]
    fp = confusion_matrix["fp"]
    fn = confusion_matrix["fn"]
    tn = confusion_matrix["tn"]

    model_economics = {
        "revenue_from_caught_fraud": tp * cost_table["tp_value"],
        "cost_of_false_alarms": fp * cost_table["fp_cost"],
        "loss_from_missed_fraud": fn * cost_table["fn_cost"],
        "net_value": (
            tp * cost_table["tp_value"]
            - fp * cost_table["fp_cost"]
            - fn * cost_table["fn_cost"]
        )
    }

    baseline_economics = {
        "flag_everything": {
            "caught_fraud": (tp + fn) * cost_table["tp_value"],
            "false_alarm_cost": (fp + tn) * cost_table["fp_cost"],
            "missed_fraud": 0,
            "net_value": (
                (tp + fn) * cost_table["tp_value"]
                - (fp + tn) * cost_table["fp_cost"]
            )
        },
        "flag_nothing": {
            "net_value": -(tp + fn) * cost_table["fn_cost"]
        }
    }

    model_economics["vs_flag_everything"] = (
        model_economics["net_value"] - baseline_economics["flag_everything"]["net_value"]
    )
    model_economics["vs_flag_nothing"] = (
        model_economics["net_value"] - baseline_economics["flag_nothing"]["net_value"]
    )

    return model_economics
```

### What Breaks

**Reporting accuracy on a balanced test set when production is imbalanced:** fraud is 0.1% of transactions. 99.9% accuracy is achieved by predicting nothing. Report precision and recall at the operating threshold.

**Omitting the baseline comparison:** "our model has 84% accuracy" means nothing without knowing what the rule-based system or random chance achieves on the same task.

## Flashcards

**API contract between model and product?** #flashcard
API contract between model and product

**Actual latency requirements (rule runs in microseconds; model in milliseconds)?** #flashcard
Actual latency requirements (rule runs in microseconds; model in milliseconds)

**Data format issues in the real input stream?** #flashcard
Data format issues in the real input stream

**Edge cases?** #flashcard
empty strings, very long documents, non-English input

**Downstream failures?** #flashcard
what happens when the model returns None?

**Input schema validation (type, range, presence)?** #flashcard
Input schema validation (type, range, presence)

**Graceful error handling (never propagate raw exceptions to callers)?** #flashcard
Graceful error handling (never propagate raw exceptions to callers)

**Health check endpoint?** #flashcard
Health check endpoint

**Request logging (for debugging production failures)?** #flashcard
Request logging (for debugging production failures)

**Version information in every response?** #flashcard
Version information in every response

**fp * cost_table["fp_cost"]?** #flashcard
fp * cost_table["fp_cost"]

**fn * cost_table["fn_cost"]?** #flashcard
fn * cost_table["fn_cost"]

**(fp + tn) * cost_table["fp_cost"]?** #flashcard
(fp + tn) * cost_table["fp_cost"]
