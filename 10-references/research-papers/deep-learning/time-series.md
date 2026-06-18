---
module: References
topic: Research Papers
subtopic: Deep Learning Time Series
status: unread
tags: [references, ml, research-papers-deep-learning, time-series]
---
# Time Series — Key Papers

Foundational and interview-relevant papers for sequence modeling, forecasting, and anomaly detection.

---

## Sequence Modeling Foundations

| Paper | Year | Why It Matters |
|---|---|---|
| [Long Short-Term Memory (Hochreiter & Schmidhuber)](https://www.bioinf.jku.at/publications/older/2604.pdf) | 1997 | Introduced gated cell state — solved vanishing gradient for sequences; still used in production |
| [Sequence to Sequence Learning with Neural Networks (Sutskever et al.)](https://arxiv.org/abs/1409.3215) | 2014 | Encoder-decoder LSTM for variable-length seq2seq — foundation for neural machine translation and forecasting |
| [Empirical Evaluation of Gated Recurrent Neural Networks (Chung et al.)](https://arxiv.org/abs/1412.3555) | 2014 | Systematic GRU vs LSTM comparison — GRUs often match LSTM with fewer parameters |
| [Temporal Convolutional Networks (Bai et al.)](https://arxiv.org/abs/1803.01271) | 2018 | TCNs outperform LSTMs on many sequence tasks; dilated causal convolutions give long receptive field with parallel training |

---

## Transformer-Based Forecasting

| Paper | Year | Why It Matters |
|---|---|---|
| [Temporal Fusion Transformer (Lim et al., Google)](https://arxiv.org/abs/1912.09363) | 2021 | Multi-horizon forecasting with attention over static, observed, and known-future covariates; interpretable feature importances |
| [Informer: Efficient Transformer for Long Sequence Time-Series Forecasting (Zhou et al.)](https://arxiv.org/abs/2012.07436) | 2021 | ProbSparse self-attention reduces O(n²) to O(n log n) for long-horizon forecasting |
| [Autoformer (Wu et al.)](https://arxiv.org/abs/2106.13008) | 2021 | Decomposition architecture: trend-cyclical decomposition + Auto-Correlation as an attention alternative |
| [Are Transformers Effective for Time Series Forecasting? — PatchTST (Nie et al.)](https://arxiv.org/abs/2211.14730) | 2022 | Patching + channel-independence + pre-training; challenges prior transformer forecasting methods |
| [iTransformer (Liu et al.)](https://arxiv.org/abs/2310.06625) | 2024 | Inverted Transformer: applies attention across variates (not time); SOTA on multivariate benchmarks |

---

## Classical Neural Forecasting

| Paper | Year | Why It Matters |
|---|---|---|
| [N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series Forecasting (Oreshkin et al.)](https://arxiv.org/abs/1905.10437) | 2020 | Pure MLP with backward/forward residual links; interpretable trend/seasonality decomposition; won M4 competition |
| [N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting (Challu et al.)](https://arxiv.org/abs/2201.12886) | 2022 | Multi-rate signal sampling + interpolation; faster and more accurate than N-BEATS for long horizons |
| [TimesNet (Wu et al.)](https://arxiv.org/abs/2210.02186) | 2023 | Reshapes 1D time series to 2D to use 2D convolutions; captures both intra- and inter-period variation |

---

## Anomaly Detection in Time Series

| Paper | Year | Why It Matters |
|---|---|---|
| [LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection (Malhotra et al.)](https://arxiv.org/abs/1607.00148) | 2016 | Reconstruction error from LSTM autoencoder as anomaly score — standard industrial approach |
| [Anomaly Transformer (Xu et al.)](https://arxiv.org/abs/2110.02642) | 2022 | Association discrepancy between prior-association and series-association as anomaly criterion |

---

## Probabilistic Forecasting

| Paper | Year | Why It Matters |
|---|---|---|
| [DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks (Salinas et al., Amazon)](https://arxiv.org/abs/1704.04110) | 2020 | Autoregressive RNN outputting probability distributions; global model trained across many related series |
| [Deep and Confident Prediction for Time Series at Uber (Zhu & Laptev)](https://arxiv.org/abs/1709.01907) | 2017 | MC Dropout for confidence intervals in production forecasting |

---

## Foundation Models for Time Series

| Paper | Year | Why It Matters |
|---|---|---|
| [TimesFM (Das et al., Google)](https://arxiv.org/abs/2310.10688) | 2024 | Large pretrained time series model (200M params); zero-shot forecasting competitive with supervised models |
| [Lag-Llama: Towards Foundation Models for Time Series Forecasting (Rasul et al.)](https://arxiv.org/abs/2310.08278) | 2024 | LLaMA-based decoder-only foundation model for univariate probabilistic forecasting |
| [MOIRAI (Salesforce)](https://arxiv.org/abs/2402.02592) | 2024 | Universal time series forecasting transformer; handles variable frequencies and horizons zero-shot |

---

## Key Interview Takeaways

**"LSTM vs Transformer for time series?"**

LSTMs process sequentially (O(n) inference, hard to parallelize training). Transformers parallelize training but suffer from O(n²) attention for long sequences. In practice: TFT or PatchTST for medium-horizon forecasting; LSTMs still appear in low-latency streaming inference pipelines where full-sequence attention is too slow.

**"What is train-test leakage in time series?"**

Using future information in features or splitting data without respecting temporal order. The correct split is always temporal — no shuffling. Cross-validation for time series requires walk-forward (expanding or sliding window) validation, not random k-fold.

**"What are the benchmarks for time series forecasting?"**

ETT (Electricity Transformer Temperature), Exchange Rate, Weather, ILI, Traffic — standardized in the Informer paper. The long-horizon forecasting benchmark suite is now the standard comparison. M4/M5 competitions used MSE and sMAPE as primary metrics.

**"What's the difference between global and local forecasting models?"**

Local: one model per time series (ARIMA, ETS). Global: one model trained on all series simultaneously (DeepAR, N-BEATS). Global models learn cross-series patterns and generalize better with sufficient data; local models are simpler and don't require related series.
