---
module: References
topic: Research Papers
subtopic: Mlops
status: unread
tags: [references, ml, research-papers-mlops]
---
# MLOps & Production ML — Key Papers

Papers that define how ML gets built, deployed, monitored, and maintained at scale.

---

## Foundational Systems Papers

| Paper | Year | Why It Matters |
|---|---|---|
| [Hidden Technical Debt in Machine Learning Systems (Sculley et al., Google)](https://proceedings.neurips.cc/paper_files/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf) | 2015 | The canonical "ML in production is hard" paper — CACE principle, entanglement, feedback loops |
| [Rules of Machine Learning: Best Practices for ML Engineering (Zinkevich, Google)](http://martin.zinkevich.org/rules_of_ml/rules_of_ml.pdf) | 2017 | 43 practical rules covering feature engineering, training, evaluation, and launch — Google's internal ML style guide |
| [Towards ML Engineering: A Brief History of TensorFlow Extended (TFX)](https://arxiv.org/abs/2012.15566) | 2020 | Google's production ML pipeline — data validation, transform, train, evaluate, serve |

---

## Data Management & Quality

| Paper | Year | Why It Matters |
|---|---|---|
| [Data Validation for Machine Learning (Breck et al., Google)](https://mlsys.org/Conferences/2019/doc/2019/167.pdf) | 2019 | TFDV — detecting schema drift, distribution shifts |
| [A Survey on Data Collection for Machine Learning (Roh et al.)](https://arxiv.org/abs/1811.03402) | 2021 | Data quality taxonomy: labeling, augmentation, collection strategies |
| [Scaling Knowledge Distillation for Industry-Scale Models](https://arxiv.org/abs/2104.04473) | 2021 | Production distillation patterns |
| [Snorkel: Rapid Training Data Creation with Weak Supervision (Ratner et al.)](https://arxiv.org/abs/1711.10160) | 2017 | Programmatic labeling — important when annotation is expensive |

---

## Model Monitoring & Drift Detection

| Paper | Year | Why It Matters |
|---|---|---|
| [Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift (Rabanser et al.)](https://arxiv.org/abs/1810.11953) | 2019 | Systematic comparison of drift detection methods |
| [Continuous Delivery for Machine Learning (Sato et al.)](https://martinfowler.com/articles/cd4ml.html) | 2019 | CD4ML framework — pipelines, testing, deployment for ML |

---

## Experiment Tracking & Reproducibility

| Paper | Year | Why It Matters |
|---|---|---|
| [Challenges in Deploying Machine Learning: a Survey of Case Studies (Paleyes et al.)](https://arxiv.org/abs/2011.09926) | 2022 | Real-world failure modes in ML deployment |
| [DVC: Data Version Control (Ruslan et al.)](https://arxiv.org/abs/2101.02234) | 2021 | Git for data and models |

---

## Serving & Inference

| Paper | Year | Why It Matters |
|---|---|---|
| [Clipper: A Low-Latency Online Prediction Serving System (Crankshaw et al.)](https://www.usenix.org/conference/nsdi17/technical-sessions/presentation/crankshaw) | 2017 | Model serving abstractions — prediction caching, adaptive batching |
| [Triton Inference Server (NVIDIA)](https://arxiv.org/abs/2101.06434) | 2021 | Multi-framework serving at scale |
| [Orca: A Distributed Serving System for Transformer-Based Generative Models (Yu et al.)](https://www.usenix.org/conference/osdi22/presentation/yu) | 2022 | Iteration-level scheduling for LLM serving — precursor to vLLM |
| [Efficient Memory Management for Large Language Model Serving with PagedAttention (Kwon et al.)](https://arxiv.org/abs/2309.06180) | 2023 | vLLM — KV cache as paged virtual memory, 24x throughput |

---

## Evaluation, Fairness & Governance

| Paper | Year | Why It Matters |
|---|---|---|
| [Model Cards for Model Reporting (Mitchell et al., Google)](https://arxiv.org/abs/1810.03993) | 2019 | Standardized model documentation: intended use, performance across subgroups, limitations |
| [Datasheets for Datasets (Gebru et al.)](https://arxiv.org/abs/1803.09010) | 2020 | Provenance documentation for datasets: collection, composition, preprocessing, recommended use |
| [A Survey of Methods for Model Compression (Cheng et al.)](https://arxiv.org/abs/2002.03938) | 2020 | Pruning, quantization, knowledge distillation — production model size reduction strategies |

---

## Feature Stores & Online/Offline Consistency

| Paper | Year | Why It Matters |
|---|---|---|
| [Feature Store for ML (Feast)](https://feast.dev/) | 2020 | Solving train-serve skew with a unified feature store |
| [Operationalizing Machine Learning: An Interview Study (Shankar et al.)](https://arxiv.org/abs/2209.09125) | 2022 | Ethnographic study of how ML teams actually work — gap between tools and practice |

---

## Key Interview Takeaways

**"What's the most important paper every ML engineer should read?"**

Hidden Technical Debt (Sculley et al.) — because it articulates why ML systems rot differently than software: CACE (Changing Anything Changes Everything), feedback loops, undeclared consumers. Every production ML failure traces back to one of these patterns.

**"What's train-serve skew and how do you prevent it?"**

Training on historical data but serving real-time features computed differently. Prevention: feature stores (same code path for training and serving), strict schema validation, integration tests on the serving path.

**"How does PagedAttention work?"**

KV cache grows dynamically during generation — traditional allocation wastes GPU memory. PagedAttention manages KV cache like OS virtual memory: fixed-size pages allocated on demand, no internal fragmentation, enables sharing across parallel requests (beam search, sampling).
