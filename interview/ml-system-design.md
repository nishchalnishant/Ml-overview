# ML system design

Great — here is a practical, curated list of ML models mapped to real ML system-design use-cases, exactly the way they appear in FAANG-style interviews.

This is _not_ a generic ML model list — it’s a system-design-oriented catalog, meaning:

✔ what model to use

✔ for which type of system

✔ why that model fits

✔ variations you can propose during interviews

***

## ✅ ML Models Useful for ML System Design Interviews

_(Grouped by real-world use cases)_

***

## 1. PII Detection / Sensitive-Info Redaction

Used in systems for anonymization, document processing, compliance (GDPR/CCPA).

#### 🔹 Models

*   NER Models (SpaCy / BERT-NER / RoBERTa-NER) \[named entity recognition]

    Extract entities like names, addresses, email, phone numbers.
*   LayoutLM / LayoutLMv3

    For PII detection in PDFs, scans, forms.
*   RCNN / Faster-RCNN _(for images containing PII)_

    Detect passport numbers, license plates, addresses in scanned IDs.
*   CRFs (Conditional Random Fields)

    Traditional but still used for structured documents like invoices.

Why it’s good for interviews

Shows ability to combine text + vision for redaction systems.

***

## 2. Recommendation Systems

Used in YouTube, Netflix, Amazon, Instagram ranking.

#### 🔹 Models

* Matrix Factorization (ALS, BPR)
* Neural Collaborative Filtering (NCF)
* Wide & Deep Models
* DeepFM / xDeepFM
*   Graph Neural Networks (PinSage / GAT)

    For connections between users, items.

#### Why it’s good

Every ML SD interview includes some recommendation component.

***

## 3. Search & Query Understanding Systems

Used in search engines, conversational systems, chatbots.

#### 🔹 Models

* BM25 (baseline)
* Bi-Encoders (Sentence-BERT, MiniLM)
* Cross-Encoders (BERT rankers)
* ColBERT v2
* RAG (Retrieval-Augmented Generation)
* Hybrid Search (Dense + Sparse)

#### When to use

For designing Google-like search or semantic search.

***

## 4. Spam Detection / Fraud Detection

Used by Gmail, payment systems, ad fraud teams.

#### 🔹 Models

* Random Forest / XGBoost / LightGBM (industry standard)
* Graph Neural Networks (fraud rings)
* Autoencoders (anomaly detection)
* Isolation Forest

#### Why strong for interviews

Demonstrates ability to handle imbalanced & adversarial data.

***

## 5. Content Moderation (Toxicity, Violence, NSFW)

Used in social media, online safety, community guidelines.

#### 🔹 Text Moderation

* RoBERTa-Toxicity
* DistilBERT toxic classifiers
* GPT-based safety classifiers

#### 🔹 Image/Video Moderation

* YOLOv8 / EfficientDet (objectionable content)
* CLIP-based zero-shot classifiers
* 3D-CNNs / SlowFast (video violence detection)

***

## 6. Image Recognition / Computer Vision

Used in e-commerce tagging, visual search, manufacturing QA.

#### 🔹 Models

* CNNs (ResNet, EfficientNet)
* Vision Transformers (ViT, Swin Transformer)
* Faster-RCNN, Mask-RCNN (detection & segmentation)
* YOLO series (real-time detection)

***

## 7. Document Understanding / OCR Systems

Used in fintech onboarding, receipt parsing, OCR pipelines.

#### 🔹 Models

* Tesseract + Language Model
* TrOCR (transformer OCR)
* Donut (OCR-free document model)
* LayoutLM (forms, invoices)

***

## 8. NLP Classification / Sequence Tasks

Used in ticket routing, topic classification, sentiment systems.

#### 🔹 Models

* BERT, DistilBERT, RoBERTa
* XLNet, ELECTRA
* Hierarchical Attention Networks (HAN) for long documents
* LSTMs / Bi-LSTMs (legacy but interview-friendly)

***

## 9. Forecasting & Time-Series Models

Used in demand forecasting, pricing, inventory, trading.

#### 🔹 Models

* ARIMA / SARIMA
* Prophet
* LSTM / GRU forecasting models
* Temporal Fusion Transformer (TFT)
* N-BEATS

***

## 10. ML for Ads / Ranking Systems (CTR, CVR, etc.)

Used in Meta/Google Ads pipelines.

#### 🔹 Models

* Logistic Regression + wide features (industry baseline)
* GBDTs + LR (Facebook’s approach)
* Deep CTR models (Wide & Deep, DIN, DIEN)
* Sequential models (Transformers)

***

## 11. Speech / Voice Assistants

Used in Alexa, Siri, Google Assistant.

#### 🔹 Models

* Whisper
* Conformer
* RNN-Transducer
* Wav2Vec 2.0

***

## 12. Reinforcement Learning Systems

Used in ranking, ads bidding, robotics, auto-ML.

#### 🔹 Models

* Deep Q-Networks (DQN)
* PPO (Proximal Policy Optimization)
* A3C / A2C
* AlphaZero-style architectures

***

## 13. Generative Models (text + image)

Used in synthetic data, augmentation, ad-copy generation, coding assistants.

#### 🔹Models

* GPT family
* LLaMA / Mistral / Falcon
* Diffusion Models (Stable Diffusion, DDPM)
* GANs (CycleGAN, StyleGAN)

***

## 14. ML for Privacy / Security

Used in privacy-preserving ML, PII hiding, anomaly detection.

#### 🔹 Models

* DP-SGD (Differential privacy training)
* Homomorphic Encryption ML
* Autoencoders for data anonymization
* CNN/RCNN for image PII hiding
* Named-entity transformers for text redaction

***

## 15. Ranking & Scoring Systems

Used in search, recommendations, feeds.

#### 🔹 Models

* Learning-to-Rank (LambdaMART, RankNet, ListNet)
* Pointwise, Pairwise, Listwise ranking models
* Transformer rankers

***

## 👉 Want this turned into a cheat-sheet PDF?

<br>

I can convert this into a 2-page interview-ready PDF with:

* Model categories
* When to use which
* Interview phrases to justify your choice
* Diagram examples
* Use-case snippets

<br>

Just say “make it a PDF”.
