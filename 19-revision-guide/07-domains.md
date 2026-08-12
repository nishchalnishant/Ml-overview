---
module: Domains
topic: Revision Guide
subtopic: Domains
status: unread
tags: [revision-guide, nlp, cv, time-series, index]
---
# 07 Domains — Revision Guide

Fast revision for the entire `07-domains/` folder: NLP fundamentals, transformer-era NLP, computer vision, and time series. Use the outline to jog the map of topics, then work the graded questions to test recall and applied reasoning.

---

## Structure Overview

```markdown
- **07 Domains**
  - **NLP Fundamentals**
    - Tokenization
      - BPE
      - WordPiece
      - SentencePiece/Unigram
    - Bag-of-Words / TF-IDF
    - Static Word Embeddings
      - Word2Vec
      - GloVe
      - FastText
    - Contextual Embeddings
      - ELMo (BiLSTM)
      - BERT embeddings
    - RNN/LSTM/GRU
    - Seq2Seq
      - Bahdanau Attention
      - Beam Search
    - Stemming/Lemmatization
    - Dependency Parsing
    - NER (BIO tagging)
    - Text Classification
    - Perplexity
    - MT Metrics
      - BLEU
      - ROUGE
      - METEOR
      - BERTScore
    - Dialogue Systems
  - **NLP Transformers**
    - Self-Attention
      - Q/K/V mechanism
      - sqrt(d_k) scaling
      - Multi-Head Attention
    - Positional Encoding
      - Sinusoidal
      - RoPE
    - BERT vs GPT
      - Masked LM
      - Causal LM
    - Encoder-Decoder (T5/BART)
      - Cross-Attention
    - Transformer Improvements
      - Pre-Norm
      - SwiGLU
      - GQA
      - Flash Attention
      - RMSNorm
    - PEFT
      - LoRA
      - Prefix/Prompt Tuning
    - NLP Eval Metrics
  - **Computer Vision**
    - CNN Building Blocks
      - Convolution
      - Activations (ReLU/GELU)
      - Pooling / GAP
      - Receptive Field
      - Depthwise Separable Conv
      - BatchNorm
    - Architecture Evolution
      - VGG
      - Inception
      - ResNet
      - DenseNet
      - EfficientNet
      - ConvNeXt
    - Residual Learning
      - Degradation Problem
      - Bottleneck Block
    - Object Detection
      - R-CNN family
      - Anchor Boxes
      - IoU / NMS / Focal Loss
      - YOLO (one-stage)
      - DETR (set prediction)
    - Segmentation
      - FCN
      - U-Net
      - DeepLab/Atrous
      - Mask R-CNN
    - Vision Transformers
      - DeiT
      - Swin
    - CLIP / VLMs
      - BLIP-2 (Q-Former)
      - LLaVA
    - Image Generation
      - GANs
      - VAEs
      - Diffusion
    - Self-Supervised Learning
      - SimCLR
      - MoCo
      - DINO
      - MAE
    - Data Augmentation
      - Mixup
      - CutMix
    - Eval Metrics (mAP/mIoU/FID)
    - Production (quant/prune/distill)
  - **Time Series**
    - Series Properties
      - Trend
      - Seasonality
      - Stationarity (ADF/KPSS)
      - ACF/PACF
    - Classical Methods
      - ARIMA/SARIMA
      - Exponential Smoothing
      - Prophet
    - Feature Engineering
      - Lag Features
      - Rolling Stats
      - Calendar/Fourier Features
    - Deep Learning Models
      - RNN/LSTM Seq2Seq
      - TCN (dilated conv)
      - Transformers
        - Informer
        - Autoformer
        - PatchTST
        - iTransformer
      - N-BEATS / N-HiTS
    - Anomaly Detection
      - Isolation Forest
      - LSTM Autoencoder
    - Evaluation (MASE/CRPS)
    - CV for Time Series
    - Probabilistic Forecasting
```

---

## Questions

### Easy

1. What problem does subword tokenization solve that word-level and character-level tokenization each fail at?
2. What is the core mechanic of Byte-Pair Encoding?
3. How does WordPiece differ from BPE in choosing merges?
4. What does TF-IDF correct for relative to raw word counts?
5. What is the key limitation of Word2Vec and GloVe that contextual embeddings solve?
6. What are the two Word2Vec training objectives (skip-gram vs CBOW)?
7. Why can't vanilla RNNs retain information over long sequences?
8. What problem do LSTM gates (forget/input/output) address?
9. What is beam search, and why is it better than greedy decoding?
10. What does perplexity measure, and what does a lower value mean?
11. What is the difference between BLEU and ROUGE in what they emphasize (precision vs recall)?
12. What is the BIO tagging scheme used for?
13. Why does a neural network need Query, Key, and Value vectors in self-attention?
14. Why is positional encoding necessary in Transformers?
15. What is the core difference between BERT and GPT in pretraining objective?
16. What does LoRA train, and why is it more efficient than full fine-tuning?
17. Why can't a fully connected layer be used efficiently on raw images?
18. What problem does weight sharing in convolution solve?
19. What is the purpose of pooling in a CNN?
20. What is the "degradation problem" that ResNets address?
21. What is the difference between object detection and semantic segmentation in output format?
22. What is IoU, and what is it used for?
23. What is the core idea behind CLIP's training objective?
24. What is the difference between GANs, VAEs, and diffusion models at a high level?
25. Why is temporal ordering important in time series, and what breaks if you ignore it?
26. What is the difference between trend and seasonality in a time series?
27. What do ACF and PACF plots help you determine?
28. What is the purpose of lag features and rolling statistics in tree-based time series models?
29. Why can't standard k-fold cross-validation be used for time series?
30. What is MASE, and why is it preferred over MAPE for cross-series comparison?

### Medium

1. Why does SentencePiece operate on raw bytes instead of pre-tokenized whitespace-split text, and what problem does this solve for non-whitespace-delimited languages?
2. Compare Word2Vec and GloVe in terms of what statistics each uses to learn embeddings, and explain why their performance tends to be similar in practice.
3. Explain why the sequential nature of LSTMs — not just vanishing gradients — motivated the shift to Transformer self-attention.
4. Walk through why Bahdanau attention is considered the direct precursor to Transformer self-attention, and identify the key structural difference.
5. Why is $\sqrt{d_k}$ scaling necessary in scaled dot-product attention? What happens to the softmax if it's omitted?
6. Explain why multiple attention heads can capture different relationships (syntactic vs semantic) even though each head has the same architecture.
7. Why does RoPE generalize better to unseen sequence lengths than sinusoidal or learned absolute positional embeddings?
8. Explain the reasoning behind why BERT uses bidirectional attention while GPT uses causal masking, tied to their respective pretraining objectives.
9. Why does an encoder-decoder architecture (T5/BART) suit translation/summarization better than a decoder-only model for very long source inputs?
10. Explain the mechanism by which GQA reduces KV cache size, and why this matters for inference throughput.
11. Why does LoRA target `q_proj` and `v_proj` specifically rather than all weight matrices?
12. When would you choose TF-IDF + logistic regression over fine-tuning BERT for a text classification task? Give at least two concrete constraints.
13. Explain why perplexity comparisons are only valid across models sharing the same tokenizer.
14. Why does BERTScore correlate better with human judgment than BLEU/ROUGE for evaluating paraphrases, and what's the tradeoff for using it?
15. Explain why two stacked 3×3 convolutions are preferred over one 5×5 convolution — cover both parameter count and expressiveness.
16. Why does the "effective" receptive field of a CNN tend to be much smaller than its theoretical receptive field?
17. Explain the mathematical intuition for why residual blocks make it easier to learn near-identity mappings than plain stacked layers.
18. Compare ResNet's skip connections to DenseNet's dense connections — what's the key structural and memory tradeoff?
19. Why does compound scaling (EfficientNet) outperform scaling only one of depth/width/resolution independently?
20. Explain why anchor boxes make bounding-box regression an easier optimization problem than predicting raw coordinates.
21. Explain focal loss's mechanism for addressing class imbalance in one-stage detectors — why does the $(1-p_t)^\gamma$ term selectively suppress gradient from easy examples?
22. Why is ROIAlign an improvement over ROI Pooling for instance segmentation specifically (not just detection)?
23. Explain why ViT underperforms CNNs on small datasets but overtakes them at large pretraining scale.
24. Why does Swin Transformer's shifted-window attention need multiple layers before achieving global context, unlike standard ViT attention?
25. Explain the role of the Q-Former in BLIP-2 versus the simpler linear projection in LLaVA — what does this comparison suggest about what actually drives multimodal LLM performance?
26. Why do SimCLR and MoCo require large batches or a memory queue respectively — what failure mode are they both avoiding?
27. Explain why DINO's teacher-student setup with different crop sizes forces semantically meaningful representations without labels.
28. Why does MAE's high masking ratio (75%) force the model to learn global structure rather than local interpolation?
29. Explain why differencing a trending series can sometimes be worse than explicitly modeling the trend and its residuals.
30. Why do multiple seasonalities (e.g., daily + weekly) break single-period methods like SARIMA, and what are two ways to handle them?
31. Explain the exposure bias problem in seq2seq forecasting with teacher forcing, and why scheduled sampling mitigates it.
32. Why does dilated convolution in TCNs grow the receptive field exponentially with depth, and what's the tradeoff versus LSTMs for long-range dependencies?
33. Explain the core idea of ProbSparse attention in Informer — why does most attention weight concentrate on a small subset of queries?
34. Why does PatchTST's channel-independence act as a regularizer, and when does this hurt rather than help?
35. Explain why N-BEATS/N-HiTS often match or beat Transformer-based forecasters despite having no attention mechanism.
36. Why is Isolation Forest a poor fit for detecting anomalies in a strictly temporal sense, and what mitigation is suggested?
37. Explain why quantile regression models can suffer from "quantile crossing," and how it's typically fixed.

### Hard

1. Derive why softmax attention scores collapse toward one-hot vectors without $1/\sqrt{d_k}$ scaling, in terms of the variance of the dot product $q \cdot k$ as a function of $d_k$.
2. RoPE makes the attention score a function of $q_m \cdot k_n = x_m^T R_{m-n} x_n$ — depending only on relative position $m-n$. Explain the derivation step from $R_m^T R_n = R_{m-n}$ and what property of rotation matrices makes this true.
3. You're told a production LLM's KV cache is the throughput bottleneck at high concurrency. Walk through the full chain of reasoning from MHA to GQA to why this specifically fixes the bottleneck, including a rough calculation of cache size reduction for a given (H, G) pair.
4. A BPE tokenizer trained predominantly on English text is deployed on a multilingual chatbot. Predict and explain the specific failure modes for low-resource languages, code, and numeric reasoning tasks — connect each to the tokenizer's training objective.
5. Explain why a decoder-only architecture can, in principle, simulate an encoder-decoder's cross-attention behavior given a sufficiently long context window — and where this simulation breaks down in practice (compute, positional degradation, KV cache growth).
6. You fine-tune BERT for NER and observe the model's F1 dropping specifically on multi-word entities. Trace this to subword tokenization misalignment and describe two different label-alignment strategies and their tradeoffs.
7. Given LoRA's decomposition $W' = W + \frac{\alpha}{r}BA$, explain why setting $\alpha = 2r$ keeps the effective update magnitude roughly rank-independent, and what happens to training dynamics if $\alpha$ is fixed while $r$ is swept from 4 to 64 without adjustment.
8. A researcher claims perplexity improved 20% after switching tokenizers, and concludes the new model is strictly better. Explain precisely why this conclusion is unjustified and what confound is at play.
9. Derive why a 56-layer plain CNN can, in theory, always match a 20-layer CNN's training loss (by setting extra layers to identity) yet fails to in practice — connect this explicitly to gradient magnitudes through repeated nonlinear compositions, not just "vanishing gradients" as a label.
10. Explain quantitatively why DenseNet's concatenation-based connectivity grows memory quadratically with depth, and derive the actual parameter/memory scaling in terms of growth rate $k$ and number of layers $L$.
11. A detection model has mAP@0.5 = 0.80 but mAP@0.5:0.95 = 0.45. Diagnose what's happening geometrically to the predicted boxes and explain what business/product decision would be affected by only reporting mAP@0.5.
12. Explain precisely why Deformable DETR's sparse deformable attention fixes DETR's slow convergence — connect this to what makes vanilla DETR's full attention over flattened feature maps hard to optimize from random initialization.
13. Design an experiment to determine whether ViT's performance gains over ResNet on a downstream task come from the attention mechanism itself or from confounding training-recipe factors (data augmentation, optimizer, normalization) — reference how ConvNeXt's existence complicates a naive "attention wins" conclusion.
14. A CLIP-based zero-shot classifier performs poorly on a niche medical imaging category. Enumerate the specific mechanisms (training data distribution, prompt engineering, contrastive objective granularity) that would cause this and rank them by likely impact.
15. Explain why MAE's asymmetric encoder-decoder design (encoder sees only 25% of patches) is both a computational optimization and a representation-learning design choice — what would change if the encoder saw all patches (visible + mask tokens) like the decoder does?
16. A time series model achieves excellent backtest MASE but degrades sharply in live deployment. Walk through the full checklist of leakage sources (feature computation order, walk-forward gaps, exogenous variable circularity) that could each independently explain this gap.
17. Derive why SARIMA cannot handle two nested seasonalities (e.g., daily period 24 inside weekly period 168) even with two different seasonal orders configured, and explain precisely how Fourier features or TBATS structurally solve this.
18. A retailer switches from point forecasts to quantile regression for inventory planning and observes Q90 < Q50 in some SKU-week combinations. Explain the root cause of quantile crossing and compare isotonic regression post-processing against a joint quantile loss as fixes, including failure modes of each.
19. Explain precisely why channel-independent models (PatchTST) often beat channel-dependent models (iTransformer, VAR) on standard multivariate benchmarks despite modeling less information — what does this imply about the actual signal-to-noise ratio of cross-variable correlations in typical benchmark datasets?
20. A TCN and an LSTM are both proposed for a streaming anomaly detection system requiring per-timestep online inference. Analyze the architectural mismatch between TCNs and streaming inference despite TCNs' training-time parallelism advantage, and explain what changes would be needed to serve a TCN online.
21. N-BEATS' backcast-residual mechanism means each block only sees what previous blocks failed to explain. Explain what happens to model behavior and interpretability if you remove the interpretable basis functions (trend/seasonality) and use the fully generic variant — connect this to a broader bias-variance tradeoff between inductive bias and flexibility seen elsewhere in this folder (e.g., CNN vs ViT).
22. Design a full evaluation protocol (cross-validation scheme, metrics, and leakage checks) for a multivariate demand forecasting system with weather as an exogenous forecast-derived feature, and explain why naively evaluating with ground-truth future weather would overstate real-world accuracy.
23. Compare the failure mode of Informer's ProbSparse attention approximation against the empirical finding that simple linear models beat it on several benchmarks — what does this suggest about when architectural complexity for long-sequence attention is actually justified versus when it's solving a problem that doesn't exist in the data?
