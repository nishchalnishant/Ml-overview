# PART 3: MODEL SELECTION DECISION TREES

## Goal
To teach candidates how to systematically choose the right algorithm/model based on data constraints, latency requirements, and interpretability needs, avoiding the trap of "always using Deep Learning or LLMs."

## Mental Model
**Start simple, justify complexity.** 
Never start with an LLM or Deep Neural Network unless the data is inherently unstructured (images, text) and requires it. Always establish a baseline (Rule-based -> Classical ML -> Deep Learning -> GenAI) and explain *why* the previous tier fails.

## Decision Framework

1. **What is the data type?** (Tabular, Text, Image, Graph)
2. **What is the scale of data?** (< 10K rows, > 10M rows)
3. **What is the latency SLA?** (Strict < 10ms vs flexible)
4. **Is interpretability required?** (Finance/Healthcare require explainability)
5. **Do we have labels?** (Supervised vs Unsupervised)

## Decision Tree

```text
What is the primary data type?
├── TABULAR (Structured)
│   ├── Need high interpretability? -> Logistic Regression / Decision Trees
│   ├── Lots of data, need high performance? -> Gradient Boosted Trees (XGBoost, LightGBM)
│   └── Extreme scale, sparse categorical data? -> Wide & Deep Networks / DLRM
│
├── TEXT (Unstructured)
│   ├── Strict latency (<10ms) & compute bounds? -> TF-IDF + SVM / FastText
│   ├── Standard NLP (Classification, NER)? -> Fine-tuned BERT / RoBERTa
│   ├── Generative / Complex Reasoning? -> LLMs (GPT-4, Llama 3)
│   │   ├── Domain-specific knowledge needed? -> RAG
│   │   └── Specific tone/format needed? -> Fine-tuning (LoRA/QLoRA)
│
├── IMAGES/VIDEO
│   ├── Standard Classification/Detection? -> ResNet / YOLO
│   └── Multi-modal / Generative? -> CLIP / Stable Diffusion
│
└── GRAPH / RELATIONAL
    ├── simple relationships? -> Node2Vec + XGBoost
    └── Complex multi-hop? -> Graph Neural Networks (GNNs)
```

## Flowchart (ASCII): Generative AI vs Classical ML

```text
[Do you need to generate text/code/images?]
       │
  YES ─┴─ NO
   │       │
   ▼       ▼
 [LLMs]  [Is data structured/tabular?]
           │
      YES ─┴─ NO (Text/Image)
       │       │
       ▼       ▼
 [XGBoost]   [Can you use a pre-trained model?]
                   │
              YES ─┴─ NO
               │       │
               ▼       ▼
     [Transfer Lrng] [Train from scratch (ResNet/BERT)]
```

## Engineering Checklist

- [ ] Have I defined a heuristic/rule-based baseline?
- [ ] Have I considered Classical ML (XGBoost/Random Forest) for tabular data?
- [ ] If using Deep Learning, have I justified it? (e.g., capturing complex interactions, unstructured data).
- [ ] If using an LLM, have I considered the cost/latency implications?
- [ ] Did I discuss whether to use RAG vs Fine-tuning?

## Common Mistakes

- **Applying Deep Learning to Tabular Data blindly:** Tree-based models (XGBoost, CatBoost) generally outperform Deep Learning on standard tabular data with less tuning and compute.
- **Using Fine-tuning for Knowledge Injection:** Trying to teach an LLM new facts via fine-tuning instead of using RAG. Fine-tuning is for *behavior/style*, RAG is for *knowledge*.
- **Ignoring the Heuristic Baseline:** Failing to mention that a simple SQL query or regex could solve 80% of the problem.

## Interview Examples

**Prompt:** "Design an AI to classify customer support tickets into 5 categories."
- *Junior:* "I would use a massive LLM like GPT-4 to read the ticket and output the category."
- *Senior:* "I would start with a baseline using FastText or TF-IDF with Logistic Regression—it's incredibly fast, cheap, and easy to deploy. If accuracy isn't sufficient, I would step up to a lightweight fine-tuned transformer like DistilBERT. I would *avoid* massive LLMs for this because it's a simple classification task; using an LLM here introduces unnecessary latency, high token costs, and risks of hallucination or non-deterministic outputs."

## Tradeoffs

| Technique | Pros | Cons | Best For |
| :--- | :--- | :--- | :--- |
| **Heuristics/Rules** | Zero latency, 100% interpretable, easy to fix. | Does not learn, hard to maintain at scale. | Baselines, hotfixes, strict business logic. |
| **XGBoost/LGBM** | State-of-the-art for tabular data, handles missing values, fast to train/serve. | Struggles with unstructured data (images, raw text). | Tabular data, credit scoring, CTR prediction. |
| **Transfer Learning** | Needs little data, leverages powerful representations. | Can be computationally heavy, domain shift issues. | Vision, standard NLP tasks. |
| **LLMs (Zero-shot)** | Highly capable, zero training required, flexible. | Expensive, high latency, hallucinates, non-deterministic. | Prototyping, complex reasoning, generation. |
| **RAG** | Grounds LLMs in facts, easy to update knowledge base. | Adds retrieval latency, dependent on search quality. | Q&A systems, internal docs search. |
| **LoRA/QLoRA** | Parameter-efficient, teaches LLMs specific formats/styles. | Doesn't effectively teach *new facts*. | Style adaptation, structured JSON output. |

## Production Considerations

- **Ensembles:** In production, you rarely use one model. You use a fast model to filter/retrieve, and a heavy model to re-rank/refine.
- **Fallback Mechanisms:** If the LLM times out or hits a rate limit, the system should gracefully fall back to a cached response or a smaller, local model.

## Real-world Examples

- **Search Engines:** Don't run BERT on the entire internet. They use BM25 (Keyword search) or lightweight dual-encoders to retrieve the top 1000 results, then use a heavier cross-encoder (Deep Learning) to re-rank the top 100, then apply business rules (Heuristics) before showing the user.
- **Pricing Algorithms:** Often use Gradient Boosted Trees rather than Neural Networks because stakeholders need feature importance (SHAP values) to understand *why* a price changed.

## Interview Follow-up Questions & Best Answers

**Q: "When would you choose to fine-tune an LLM vs using RAG?"**
*Best Answer:* "I use RAG when the model needs access to external, changing, or proprietary knowledge (like internal company wikis). RAG is cheaper to update—you just update the vector DB. I use Fine-Tuning (like LoRA) when I need the model to learn a specific *behavior*, *tone*, or *format* (like outputting strict XML, or adopting a specific persona) that is hard to fit in a prompt context. Often, production systems use both: a fine-tuned model that utilizes RAG."

**Q: "Why not use a Deep Neural Network for this fraud detection system on tabular data?"**
*Best Answer:* "Deep neural networks are data-hungry and prone to overfitting on tabular data unless specifically designed (like TabNet). Tree-based models like XGBoost handle non-linearities and missing values out-of-the-box, train significantly faster, and are much easier to interpret using SHAP, which is a hard requirement for compliance in fraud detection."
