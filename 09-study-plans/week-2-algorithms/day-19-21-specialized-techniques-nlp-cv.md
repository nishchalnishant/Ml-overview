---
module: Study Plans
topic: Week 2 Algorithms
subtopic: Day 19 21 Specialized Techniques Nlp Cv
status: unread
tags: [studyplans, ml, week-2-algorithms-day-19-21-sp]
---
# Day 19-21: Specialized Techniques (NLP & CV)

## Why This Topic Comes Here

All previous algorithms operated on tabular data — fixed-size numerical feature vectors. NLP and Computer Vision are the two domains where raw input (text, pixels) is not yet in that form and cannot be made so without domain-specific architectures. These topics come at the end of week 2 because they are best understood as applications of everything you have already learned: gradient descent, regularization, evaluation metrics, and ensemble thinking still apply — but the input representations and the architectures that process them are domain-specific. A transformer is backpropagation applied to attention matrices. A CNN is gradient descent over convolutional filters. The algorithms are the same; the inductive biases are new.

---

## Executive Summary

| Domain | Core Task | Standard Feature/Arch |
|--------|-----------|------------------------|
| **NLP** | Text Classification/Translation | Embeddings, Transformers, attention |
| **CV** | Image Classification/Detection | Convolutions (CNN), Pooling |
| **Tabular** | Forecasting/Fraud | XGBoost, LightGBM |

---

## 1. Natural Language Processing (NLP)

### Text Representation

**Why representation comes before architecture:** You cannot feed raw text to any of the models from weeks 1-2. The representation choice determines what information the model can learn. This is the domain-specific analog of feature engineering.

1. **Bag-of-Words (BoW)**: Count word frequencies. High-dimensional, sparse. Loses all word order.
2. **TF-IDF**: Weighs words by importance: $TF(t,d) \times IDF(t) = \frac{f_{t,d}}{|d|} \times \log\frac{N}{df_t}$. Reduces the influence of common words.
3. **Embeddings (Word2Vec/GloVe)**: Dense vectors where semantic relationships are encoded by distance.

**Key insight for BoW/TF-IDF:** These representations treat "bank" in "river bank" and "bank account" identically — the same token gets the same representation regardless of context. This is not a minor limitation: it means the model cannot distinguish meanings that depend on context, which is fundamental to language. This single problem motivated the entire research direction that led to contextual embeddings (ELMo, BERT).

**How to verify understanding:** TF-IDF assigns high weight to rare words. Name a case where a rare word would have high TF-IDF score but be useless or harmful as a feature (e.g., noise in training set). How would you detect this?

**What trips people up:** Thinking embeddings are features in the traditional sense. Word2Vec embeddings encode distributional similarity — words that appear in similar contexts are nearby. "Dog" and "puppy" will be close. "Bank" (financial) and "bank" (river) will be at some average of the two usages. Embeddings capture usage patterns, not meaning in a semantic or logical sense.

### Modern NLP: Transformers

**Why transformers displaced all prior architectures:** RNNs and LSTMs process text sequentially — token by token. This means information about token 1 must travel through tokens 2, 3, ..., N to influence the prediction at position N. In long sequences, this path is too long and the signal vanishes. Transformers solve this by allowing every token to directly attend to every other token in one step. The price is quadratic complexity in sequence length — a tradeoff that motivated recent linear-attention and sparse-attention variants.

- **Self-Attention**: Allows each token to weigh every other token when computing its representation.
  $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$
- **BERT**: Encoder-only. Pre-trained on masked language modeling. Best for understanding tasks (classification, NER, question answering).
- **GPT**: Decoder-only. Pre-trained on next-token prediction (causal). Best for generation tasks.

**Key insight:** The $\sqrt{d_k}$ scaling in the attention formula is not cosmetic. Without it, the dot products of large-dimensional queries and keys grow with $d_k$, pushing the softmax into regions with near-zero gradients (the softmax saturates). Scaling by $\sqrt{d_k}$ keeps the dot products in a range where the softmax has useful gradients. This is the same phenomenon as vanishing gradients in deep networks — managed by normalization.

**How to verify understanding:** BERT is bidirectional (sees both left and right context). GPT is causal (sees only left context). Explain why GPT's causal masking is necessary during pre-training but why the same architecture can be used without masking at inference for classification tasks.

**What trips people up:** Treating fine-tuning as "retraining BERT." Fine-tuning updates all weights of a pre-trained model on a small task-specific dataset. The pre-trained weights are a starting point — the model already knows language; fine-tuning teaches it the task. The alternative (training a transformer from scratch on a small dataset) almost never works.

---

## 2. Computer Vision (CV)

### Why CNN Inductive Biases Match Images

**Why CNNs precede transformers in the CV sequence:** Vision Transformers (ViT) came later and are harder to understand without knowing what problem CNNs solved. CNNs were designed with two assumptions that happen to be true for natural images: (1) useful patterns are local (edges, textures exist within small image regions), and (2) the same pattern is useful regardless of where it appears (translation invariance). These assumptions allowed CNNs to reduce the parameter count from millions (fully connected) to thousands (shared filters), making them learnable from the data sizes available in 2012.

### Convolutional Neural Networks (CNNs)

- **Filters/Kernels**: Shared weights that slide across the input and detect local patterns (edges, textures, shapes). Early layers detect low-level features; later layers combine them into high-level ones.
- **Pooling**: Reduces spatial dimensions. Max pooling keeps the strongest activation in each region, providing approximate translation invariance.
- **Invariance**: The same filter detects a feature regardless of its position in the image.

**Key insight:** The power of CNNs is not the convolution operation itself — it is the weight sharing. A filter that detects a horizontal edge uses the same weights whether the edge is in the top-left or bottom-right of the image. This dramatically reduces the number of parameters and encodes the prior that "what" matters more than "where." Without weight sharing, recognizing a cat in any position would require separate parameters for each position.

**How to verify understanding:** An MLP applied to a 256×256 RGB image has 196,608 input connections per neuron in the first hidden layer. A 3×3 convolutional filter on the same image has 27 (3×3×3) parameters — shared across all positions. Explain why this parameter sharing works and when it would fail.

**What trips people up:** Thinking pooling provides true translation invariance. Max pooling provides approximate, local invariance to small shifts. A cat shifted by 20 pixels can still produce a completely different activation map if the shift crosses a pooling boundary. Proper invariance requires data augmentation during training.

### Vision Transformers (ViT)

ViT applies the Transformer architecture to fixed-size patches of images instead of pixels. Each patch is flattened into a vector, linearly projected, and treated as a "token" — exactly as a word token is treated in NLP.

**Key insight:** ViT drops both locality and translation invariance assumptions — it treats the image as a sequence of patch tokens with no spatial prior built into the architecture. This requires much more data to learn these properties from scratch. On ImageNet (1M images), ResNet-style CNNs outperform ViT. On JFT-300M (300M images), ViT dominates. The inductive biases of CNN are a form of regularization — they help when data is scarce and hurt when data is abundant and the true patterns are global.

**How to verify understanding:** Why does ViT require larger datasets than CNNs to achieve comparable performance? Frame your answer in terms of inductive biases and implicit regularization.

**What trips people up:** Treating CNNs as "old" and ViT as "modern/better." In most practical settings with limited data, CNNs with appropriate augmentation remain competitive or superior. ViT's advantages emerge at scale.

### Data Augmentation

**Key insight:** Data augmentation is not just a trick to make the dataset larger. It directly encodes invariances that you want the model to learn. If you augment with horizontal flips, you are telling the model that left-right orientation is irrelevant. If the task requires distinguishing left-right (e.g., reading text), horizontal flipping is harmful augmentation. Always align augmentation strategy with the invariances that actually hold in the deployment domain.

**How to verify understanding:** You train an image classifier with random rotation augmentation up to 360 degrees. The model is deployed to classify skin lesion images. The task is rotation-invariant (a mole looks the same upside down). Now you deploy the same augmentation to classify street sign images. What goes wrong, and why?

**What trips people up:** Applying the same augmentation pipeline to every vision task. Copy-pasting augmentation from a classification paper into a detection or segmentation task without checking whether the labels need to be transformed alongside the image (bounding boxes must be rotated too, for example).

---

## Interview Questions

**1. "What is TF-IDF and why use it over simple counts?"**
> TF (Term Frequency) measures how often a word appears in a doc. IDF (Inverse Document Frequency) penalizes common words (like "the", "is"). TF-IDF highlights words that are unique/important to a specific document.

**2. "Why are CNNs better than MLPs for images?"**
> MLPs have too many parameters (fully connected) and don't capture spatial locality. CNNs use shared weights (filters) and translation invariance, making them much more efficient and effective for visual patterns.

**3. "What is Data Augmentation in the context of CV?"**
> Artificially increasing the dataset by applying transformations (flips, rotations, crops, color shifts) to the original images. This helps the model generalize better and reduces overfitting.

---

## NLP Quick-Check

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModel

# Classical approach
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
X = vectorizer.fit_transform(corpus)

# Modern approach: get contextual embeddings
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
outputs = model(**inputs)
cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token
```

## Flashcards

**Self-Attention?** #flashcard
Allows each token to weigh every other token when computing its representation.

**BERT?** #flashcard
Encoder-only. Pre-trained on masked language modeling. Best for understanding tasks (classification, NER, question answering).

**GPT?** #flashcard
Decoder-only. Pre-trained on next-token prediction (causal). Best for generation tasks.

**Filters/Kernels?** #flashcard
Shared weights that slide across the input and detect local patterns (edges, textures, shapes). Early layers detect low-level features; later layers combine them into high-level ones.

**Pooling?** #flashcard
Reduces spatial dimensions. Max pooling keeps the strongest activation in each region, providing approximate translation invariance.

**Invariance?** #flashcard
The same filter detects a feature regardless of its position in the image.
