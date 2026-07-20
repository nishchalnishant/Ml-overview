---
module: LLMs
topic: Applications
subtopic: Hallucination Mitigation
status: unread
tags: [llms, ml, applications-hallucination-mit]
---
# Hallucination Mitigation

---

## Why LLMs Hallucinate

**The problem:** a language model is trained to minimize next-token prediction loss. It is not trained to distinguish "I know this" from "this sounds plausible." Both a recalled fact and a fluent confabulation look identical to the training objective — they are evaluated only on whether the output distribution matches the training corpus, not on whether the generated claim is true. The model has no internal oracle to consult.

**The core insight:** hallucination is not a bug introduced into an otherwise correct system. It is the natural behavior of a model optimized for fluency rather than factual accuracy. Any mitigation must either (a) give the model external facts to attend to instead of relying on parametric memory, (b) detect when generated claims are inconsistent with a reliable source, or (c) change what the model is rewarded for during training.

**What breaks without mitigation:** in safety-critical domains (medicine, law, finance), a confidently stated wrong fact is worse than acknowledged uncertainty. User trust degrades rapidly once hallucinations are discovered. RAG systems can amplify hallucinations if the retrieved context is itself wrong or if the model ignores context in favor of its training memory.

---

## Taxonomy

| Type | Definition | Example |
|:---|:---|:---|
| Intrinsic | Output contradicts the provided source/context | Summarizer reverses a date stated in the source document |
| Extrinsic | Output adds information absent from the source | Model invents a citation that does not exist |
| Faithful hallucination | Consistent with context but factually wrong in the world | Faithfully summarizes a wrong claim from the source |
| Factual hallucination | Contradicts world knowledge | States Einstein was born in France |

Note: faithful ≠ factual. A model can accurately reflect a wrong claim in a source document (faithful but factually wrong) or accurately state world knowledge despite a source saying otherwise (factual but unfaithful).

---

## Root Causes

### Teacher Forcing vs. Autoregressive Inference

**The problem:** during training the model always receives the ground-truth previous token as input. At inference it receives its own (possibly wrong) previous token. These are different distributions. A model trained under teacher forcing was never exposed to its own errors as inputs — it was never trained to recover from mistakes.

```
Training:   P(wₜ | w₁*, w₂*, ..., w*ₜ₋₁)  ← ground truth prefix
Inference:  P(wₜ | ŵ₁, ŵ₂, ..., ŵₜ₋₁)    ← model's own output
```

An error at position t becomes the input for position t+1, compounding over long generations.

### Knowledge Cutoff and Rare Facts

**The problem:** models memorize frequent patterns well and interpolate for rare ones. For common facts (the capital of France), memorization is reliable. For rare facts (an obscure historical event), the model may have seen contradictory or insufficient training examples, and will produce a plausible-sounding interpolation that is wrong. Facts after the training cutoff simply do not exist in the model's weights — queries about them force fabrication.

### RLHF Sycophancy

**The problem:** human raters sometimes reward confident, fluent responses more than cautious, hedging ones. A model trained on this signal learns that asserting things confidently is rewarded, even when the assertion is wrong. The model is being trained to seem accurate rather than to be accurate.

### Decoding Parameters

High temperature and high top-p sampling draw tokens from a broader distribution — trading coherence for diversity, which means factual precision also degrades. Greedy decoding does not eliminate hallucination; it can still produce the mode of a wrong distribution.

---

## Detection

### Self-Consistency

**The problem:** we cannot directly measure whether a claim is true without a reference. But we can measure whether the model is consistent about a claim across independent samples.

**The core insight:** if a claim is grounded in the model's actual knowledge, re-sampling should produce the same claim. If the claim is confabulated, different samples will contradict each other. High variance across samples signals unreliable information.

**The mechanics:** sample N responses to the same prompt. Extract the key claim. Measure agreement. Majority answer with high frequency = likely correct; scattered answers = high uncertainty.

**What breaks:** works well for closed-form questions. Breaks for open-ended generation where two responses can be semantically equivalent but lexically different. Does not catch errors the model consistently makes — systematic hallucinations produce high cross-sample consistency.

---

### SelfCheckGPT

**The problem:** self-consistency requires knowing what "the answer" is in advance. For longer generated texts with many claims, extracting and comparing individual claims across N samples is not straightforward.

**The core insight:** each sentence in the primary response can be checked for consistency against independently sampled responses. If a sentence appears in a different form or is contradicted in other samples, it is likely hallucinated. No external knowledge base is required — only the model's own variance.

**The mechanics:**
1. Generate a primary response at temperature 0.
2. Sample N additional responses at temperature > 0.
3. For each sentence in the primary response, measure its maximum semantic similarity to any sentence in each of the N samples.
4. Low average similarity = the sentence is inconsistent across samples = likely hallucinated.

**What breaks:** systematic hallucinations — errors the model makes consistently — produce high cross-sample consistency and are not detected. This method catches random fabrication, not confident misinformation.

---

### NLI-Based Factual Checking

**The problem:** for RAG systems, there is a specific verifiable source: the retrieved documents. The question is whether each generated claim is supported by that source.

**The core insight:** Natural Language Inference (NLI) models classify whether a premise entails, is neutral to, or contradicts a hypothesis. Use an NLI model as the verifier: premise = retrieved source, hypothesis = generated claim. Contradiction = hallucination.

**The mechanics:**
```
retrieved_source + generated_claim → NLI model → {entailment, neutral, contradiction}
```

**What breaks:** NLI models trained on SNLI/MultiNLI may not generalize to domain-specific text. A claim can be "neutral" to the source (neither confirmed nor denied) — is that a hallucination or a valid extension? The NLI model itself can be wrong.

---

### FactScore

**The problem:** a generated sentence may contain multiple atomic claims, some supported and some not. A sentence-level correct/incorrect label misses this granularity.

**The core insight:** decompose the generated text into atomic claims (the smallest verifiable units), retrieve evidence for each claim independently, and verify each against its evidence. FactScore = fraction of atomic claims that are supported.

**The mechanics:**
1. Use an LLM to decompose generated text into atomic claims.
2. For each claim, retrieve relevant passages from Wikipedia via BM25 or dense retrieval.
3. Use an NLI model or LLM to classify each claim as supported or not.
4. FactScore = (supported claims) / (total claims).

**What breaks:** the decomposition step can introduce errors. Retrieval may fail to surface the right evidence for obscure claims. Expensive at scale — full FactScore on a long response requires many retrieval calls.

---

## Mitigation

### Retrieval-Augmented Generation (RAG)

**The problem:** the model's parametric memory is fixed at training time, cannot be updated, cannot be verified, and conflates remembered facts with confabulated text that merely sounds plausible.

**The core insight:** externalize the knowledge store. Give the model the relevant facts directly in the prompt. If the retrieved context contains the answer, the model attends to an explicit, verifiable source rather than reconstructing from distributed weights. The source is also updatable without retraining.

**The mechanics:**
1. Embed the query.
2. Retrieve top-k chunks from a vector index by cosine similarity.
3. Prepend retrieved chunks to the prompt.
4. Instruct the model to answer only from the provided context.

**What breaks:** retrieval quality is the ceiling. If the right chunk is not retrieved, the model falls back to parametric memory. Chunks in the middle of long contexts get ignored (lost-in-the-middle). Conflicting retrieved passages cause hedging or arbitrary selection. Semantic similarity ≠ relevance.

---

### Prompt Engineering

**The problem:** a model capable of expressing uncertainty rarely does so by default, because training data rewards confident assertions.

**The core insight:** explicit instructions in the prompt can activate hedging behavior. The model has seen many examples of hedged language in training; prompting it to use those phrases increases their frequency in output.

**The mechanics:**
- Append "Think step by step before answering" — forces verifiable intermediate steps.
- Include few-shot examples where the model says "I don't know" for uncertain questions.
- Add explicit instructions: "If you are less than 80% confident in a claim, preface it with 'I believe.' If you cannot verify from the provided context, say so."
- Instruct citation: "After each factual claim, include [Doc N] referring to the source document."

**What breaks:** prompt instructions are not enforced mechanically — they increase the probability of desired behaviors but do not guarantee them. A model that is confidently wrong will override hedging instructions. Longer system prompts can dilute the effect.

---

### Constrained Decoding

**The problem:** format-level hallucinations (invented JSON keys, wrong data types, impossible enum values) occur when the model generates output that must conform to a schema but is not mechanically required to do so.

**The core insight:** at each decoding step, mask out any token that would make the output invalid according to the schema. The model cannot generate invalid output because it is physically prevented from doing so.

**The mechanics:** derive a finite-state machine (FSM) from the JSON schema or regex. At each generation step, compute which tokens are valid given the current FSM state. Assign −∞ logit to all other tokens. The model samples only from valid tokens.

**What breaks:** constrained decoding eliminates format hallucinations but not factual ones. A model can generate valid JSON with wrong values. Very tight constraints can force semantically nonsensical outputs if the model is trying to say something the schema does not support.

---

### Calibration

**The problem:** a model that says "I am 90% confident" should be correct 90% of the time at that stated confidence. If it is actually correct only 50% of the time at that stated confidence, the expressed confidence is useless.

**The core insight:** calibration measures the gap between expressed confidence and empirical accuracy. Improving calibration does not change what the model says — it changes how reliably the model's stated confidence predicts correctness.

**The mechanics:**
- **Expected Calibration Error (ECE):** bin predictions by stated confidence; measure the average gap between confidence and accuracy within each bin. Low ECE = well calibrated.
- **Verbalized uncertainty:** train or prompt the model to use phrases ("I believe," "I'm not certain") when it is less confident.
- **Token-level log-probabilities:** for single-token answers, the model's own log-probability of the answer token is a calibration signal — imperfect but better than nothing.

**What breaks:** RLHF training often worsens calibration by rewarding confident assertions. Large models are frequently overconfident. Verbalized confidence ("70% confident") is unreliable as an absolute estimate — it is better used as a relative ranking signal.

---

### Factuality Fine-Tuning

**The problem:** the base training corpus contains falsehoods, contradictions, and unverified claims. A model trained on this data will reproduce them. Instruction tuning with random internet text cannot fix systematic factual errors baked in during pretraining.

**The core insight:** curate training data specifically for factual accuracy. Train or fine-tune only on examples that have been verified against a reliable knowledge source. Down-weight or exclude examples containing hallucinated claims.

**The mechanics:**
- Filter the SFT dataset to keep only examples with FactScore above a threshold.
- Include TruthfulQA-style adversarial examples with correct answers (teaching the model not to repeat common misconceptions).
- Use FactScore as a reward signal: prefer high-FactScore responses during preference data collection.

**What breaks:** aggressive filtering shrinks the dataset — coverage of rare topics decreases. The filtering process itself uses an imperfect verifier. Fine-tuning on a narrow factuality-focused dataset can reduce general capability (catastrophic forgetting).

---

### RLHF and RLAIF

**The problem:** cross-entropy training on human-written text trains the model to match the training distribution. It does not teach the model to prefer true answers over plausible-sounding false ones when both appear in training data.

**The core insight:** collect explicit human judgments of which responses are more factually accurate. Train a reward model on these judgments. Fine-tune the LLM to maximize reward. This directly optimizes for factual accuracy — the property that cross-entropy training ignores.

**The mechanics (RLHF):**
1. SFT: fine-tune on demonstration data.
2. Reward model: train on human preferences (response A preferred over response B) using Bradley-Terry loss.
3. PPO: fine-tune the SFT model to maximize reward, with KL penalty to prevent diverging from the SFT policy.

RLAIF replaces human labelers with an AI critic guided by a constitution of principles — the model critiques its own responses, revisions are preferred over originals, and these (original, revised) pairs train the preference model.

**What breaks:** reward hacking — the model learns to appear factually accurate rather than to be factually accurate. The reward model can be fooled by confident, fluent responses. Hard to define a precise reward signal for factual accuracy at scale.

---

## Evaluation Benchmarks

### TruthfulQA

817 adversarial questions designed to elicit common misconceptions. Questions are designed so the "imitative falsehood" — the answer that sounds true — is wrong. Metrics: % truthful, % informative, truthful × informative.

**Limitation:** static benchmark; frontier models are approaching saturation. Does not test open-ended generation.

### FActScore

Open-domain biography generation. FactScore = % of atomic claims supported by Wikipedia retrieval + NLI verification. GPT-4 ~73%, Llama-2-13B ~55% at release.

**Limitation:** slow and expensive. Limited to Wikipedia-verifiable claims.

### HaluEval

Pairs of (hallucinated, non-hallucinated) outputs across QA, summarization, and dialogue. Tests whether models can distinguish the hallucinated version from the correct one.

**Limitation:** tests detection ability, not generation quality.

### FELM

Span-level hallucination detection. Labels individual sentences as factual or hallucinated. Tests both detection accuracy and localization (which sentence is wrong?).

---

## Mitigation Stack by Cost

| Intervention | Cost | Effect |
|:---|:---|:---|
| Prompt engineering (CoT, cite sources, "I don't know") | Near zero | Reduces surface hallucination; unreliable for confident errors |
| RAG | Medium (infra) | Externalizes knowledge; retrieval quality is the ceiling |
| Constrained decoding | Low (inference compute) | Eliminates format hallucinations; does not fix factual ones |
| Calibration | Low (prompting or light fine-tuning) | Improves uncertainty expression; does not change content |
| Factuality fine-tuning | High (GPU + curated data) | Reduces systematic errors; risks reducing coverage |
| RLHF/RLAIF | Very high (reward model + RL loop) | Most powerful; reward hacking risk |

*Related: [RAG](02-rag.md) | [Tuning and Optimization](10-tuning-optimization.md) | [Agentic Workflows](01-agentic-workflows.md)*

## Flashcards

**Why do LLMs hallucinate in the first place?** #flashcard
They're trained to minimize next-token prediction loss (fluency), not to distinguish "I know this" from "this sounds plausible." A recalled fact and a fluent confabulation look identical to the training objective. Mitigations must either give the model external facts (RAG), detect inconsistency with a reliable source, or change the training reward.

**What's the difference between faithful and factual hallucination?** #flashcard
Faithful = consistent with the provided context but wrong about the world (e.g., accurately summarizing a source that itself contains an error). Factual = contradicts world knowledge regardless of context. A model can be faithful-but-wrong or factual-but-unfaithful (correct about the world while contradicting what the source said).

**Why does teacher forcing during training contribute to hallucination at inference?** #flashcard
Training conditions each token on the ground-truth prefix; inference conditions on the model's own (possibly wrong) previous outputs. The model was never trained to recover from its own errors, so one wrong token can compound across a long generation (exposure bias).

**How do self-consistency and SelfCheckGPT detect hallucination without a reference answer?** #flashcard
Sample N responses to the same prompt (or N variants at temp>0 against one primary response at temp=0) and measure agreement/semantic similarity. High variance across samples signals unreliable/confabulated content. Both fail on systematic hallucinations — errors the model makes consistently produce high cross-sample agreement despite being wrong.

**How does RAG reduce hallucination and what limits its effectiveness?** #flashcard
It externalizes knowledge into the prompt so the model attends to a verifiable, updatable source instead of reconstructing facts from parametric memory. Ceiling is retrieval quality: if the right chunk isn't retrieved, the model falls back to parametric memory; lost-in-the-middle effects and conflicting passages can still cause errors.

**How does constrained decoding prevent hallucination, and what does it not fix?** #flashcard
It derives a finite-state machine from a schema/regex and masks invalid tokens (−∞ logit) at each decode step, so the model physically cannot emit malformed output. It eliminates format hallucinations (bad JSON keys/types) but not factual ones — the model can still produce valid JSON containing wrong values.

**What is calibration and why does RLHF often make it worse?** #flashcard
Calibration is the gap between a model's stated confidence and its empirical accuracy (measured via Expected Calibration Error). RLHF tends to reward confident, fluent responses over hedged ones, so models become more overconfident even as raw capability improves — the model learns to seem accurate rather than be accurate.

**What is FactScore and how is it computed?** #flashcard
It decomposes generated text into atomic claims, retrieves supporting evidence for each (e.g., via Wikipedia BM25/dense retrieval), and classifies each claim as supported or not with an NLI model or LLM. FactScore = fraction of claims supported. More granular than sentence-level judgments but expensive at scale and limited to verifiable claims.

**Rank the hallucination mitigation stack by cost and what each does/doesn't fix.** #flashcard
Cheapest → most expensive: prompt engineering (near-zero cost, reduces surface hallucination but unreliable against confident errors) → constrained decoding / calibration (low cost, fixes format/expression, not content) → RAG (medium infra cost, fixes knowledge-grounding up to retrieval quality) → factuality fine-tuning (high cost, reduces systematic errors, risks losing coverage) → RLHF/RLAIF (highest cost, most powerful, but reward-hacking risk).
