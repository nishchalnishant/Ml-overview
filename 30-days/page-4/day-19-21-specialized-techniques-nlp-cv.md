# Day 19-21: Specialized Techniques (NLP & CV)

##  Executive Summary
| Domain | Core Task | Standard Feature/Arch |
|--------|-----------|------------------------|
| **NLP** | Text Classification/Translation | Embeddings, Transformers, attention |
| **CV** | Image Classification/Detection | Convolutions (CNN), Pooling |
| **Tabular** | Forecasting/Fraud | XGBoost, LightGBM |

---

## 文本 1. Natural Language Processing (NLP)

### Text Representation
1. **Bag-of-Words (BoW)**: Counting word frequencies. High-dim, sparse.
2. **TF-IDF**: Weighs words by importance ($TF \times IDF$).
3. **Embeddings (Word2Vec/GloVe)**: Dense vectors where relationship is captured by distance.

### Modern NLP: Transformers
- **Self-Attention**: Allows the model to weigh different words in a sentence differently when processing a token.
- **BERT**: Encoder-only, good for understanding (Sentiment, NER).
- **GPT**: Decoder-only, good for generation.

---

##  2. Computer Vision (CV)

### Convolutional Neural Networks (CNNs)
- **Filters/Kernels**: Learn features like edges, textures, and patterns.
- **Pooling**: Reduces spatial dimensions (Max Pooling is most common).
- **Invariance**: CNNs allow for translation invariance (an ear is an ear, wherever it is in the picture).

### Vision Transformers (ViT)
Recent breakthrough: Applying the Transformer architecture to patches of images instead of pixels.

---

##  Interview Questions

**1. "What is TF-IDF and why use it over simple counts?"**
> TF (Term Frequency) measures how often a word appears in a doc. IDF (Inverse Document Frequency) penalizes common words (like "the", "is"). TF-IDF highlights words that are unique/important to a specific document.

**2. "Why are CNNs better than MLPs for images?"**
> MLPs have too many parameters (fully connected) and don't capture spatial locality. CNNs use shared weights (filters) and translation invariance, making them much more efficient and effective for visual patterns.

**3. "What is Data Augmentation in the context of CV?"**
> Artificially increasing the dataset by applying transformations (flips, rotations, crops, color shifts) to the original images. This helps the model generalize better and reduces overfitting.

---

##  NLP Quick-Check
```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(corpus)
```
