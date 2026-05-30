---
module: Specialized Domains
topic: Time Series
subtopic: ""
status: unread
tags: [time-series, forecasting, arima, lstm, transformer, anomaly-detection, specialized-domains]
---
# Time Series Analysis and Forecasting

---

## Table of Contents

1. [What Makes Time Series Special](#1-what-makes-time-series-special)
2. [Classical Methods](#2-classical-methods)
3. [Decomposition and Stationarity](#3-decomposition-and-stationarity)
4. [Deep Learning for Time Series](#4-deep-learning-for-time-series)
5. [Transformer-Based Forecasters](#5-transformer-based-forecasters)
6. [Anomaly Detection](#6-anomaly-detection)
7. [Evaluation Metrics](#7-evaluation-metrics)
8. [Multi-Step Forecasting Strategies](#8-multi-step-forecasting-strategies)
9. [Common Interview Questions](#9-common-interview-questions)

---

## 1. What Makes Time Series Special

**Temporal dependence:** Observations at time t are correlated with observations at t-1, t-2, ... The i.i.d. assumption underlying most ML breaks. Past values predict future values — this is the signal.

**Non-stationarity:** The statistical properties of the series (mean, variance, autocorrelation structure) may change over time. Models trained on the past may fail when the distribution shifts.

**Seasonality:** Periodic patterns tied to calendar cycles — daily (peak power demand at 7pm), weekly (traffic drops on weekends), yearly (retail sales spike in December). These cycles must be modeled or removed.

**Irregular spacing:** Events (sensor failures, missing data, variable-frequency logs) may not arrive at fixed intervals. Most classical methods assume regular spacing.

**Multiple interacting series:** In practice, forecasting is multivariate — temperature depends on humidity, sales depend on promotions, stock price depends on correlated assets. Capturing cross-series dependencies is hard.

---

## 2. Classical Methods

### AR, MA, ARMA

**AR(p) — Autoregressive:**
$$y_t = c + \sum_{i=1}^{p} \phi_i y_{t-i} + \varepsilon_t$$

Regress y_t on its own past p values. Lag order p controls the memory.

**MA(q) — Moving Average:**
$$y_t = \mu + \varepsilon_t + \sum_{i=1}^{q} \theta_i \varepsilon_{t-i}$$

Regress y_t on past forecast errors (innovations). Not a moving average of y values — a moving average of past shocks.

**ARMA(p,q):** combine both.

### ARIMA

ARIMA(p, d, q) adds differencing to handle non-stationarity:

1. **Differencing (d):** Replace y_t with $\Delta^d y_t$. First difference: $\Delta y_t = y_t - y_{t-1}$. Makes the series stationary.
2. **AR(p)** on the differenced series
3. **MA(q)** on the differenced series

**SARIMA** adds seasonal terms: SARIMA(p,d,q)(P,D,Q)[s] where s is the seasonal period (s=12 for monthly data with yearly seasonality).

**Selecting p, d, q:**
- ADF test (Augmented Dickey-Fuller): test for unit root (non-stationarity). If non-stationary, difference.
- ACF (Autocorrelation Function): MA order q — ACF cuts off after lag q
- PACF (Partial Autocorrelation): AR order p — PACF cuts off after lag p
- AIC/BIC: penalized likelihood for model comparison

**Limitations:**
- Linear model: cannot capture non-linear patterns
- Univariate: cannot use exogenous features natively (use ARIMAX for exogenous variables)
- Fixed seasonality: struggles with multiple seasonal periods (hourly data has daily AND weekly AND yearly seasonality)
- Requires stationarity: pre-processing burden

### Exponential Smoothing

**Simple Exponential Smoothing:** Weighted average of past observations, exponentially decaying weights:
$$\hat{y}_{t+1} = \alpha y_t + (1-\alpha)\hat{y}_t = \alpha \sum_{k=0}^{\infty}(1-\alpha)^k y_{t-k}$$

α∈(0,1) controls how quickly the influence of past observations decays.

**Holt's Linear:** Add trend component — two equations tracking level and trend separately.

**Holt-Winters:** Add seasonal component — three equations for level, trend, and seasonality (additive or multiplicative). Effective for seasonal data with trend.

**ETS (Error, Trend, Seasonality):** Unifying framework for exponential smoothing families. Specifies whether each component is None, Additive, or Multiplicative. Enables likelihood-based model selection and prediction intervals.

### Prophet (Meta, 2017)

**The problem it solves:** Business time series have complex, irregular seasonality (holidays, promotions, product launches) and structural changes that ARIMA handles poorly.

**Model:**
$$y(t) = g(t) + s(t) + h(t) + \varepsilon_t$$

- **g(t):** trend — logistic growth (for saturating trends) or piecewise linear with automatic changepoint detection
- **s(t):** seasonality — Fourier series decomposition at multiple periods
- **h(t):** holiday effects — user-specified holidays with adjustable windows

**Fit via Stan (MAP):** Fast, automatic, requires no tuning. Handles missing data and outliers gracefully.

**When to use:** Business forecasting with calendar effects, irregular data, known holidays, stakeholder-interpretable components. Not suitable for high-frequency or multivariate forecasting.

---

## 3. Decomposition and Stationarity

### Stationarity

A series is **weakly stationary** if:
- Mean is constant: E[y_t] = μ for all t
- Variance is constant: Var(y_t) = σ² for all t
- Autocovariance depends only on lag: Cov(y_t, y_{t+k}) = γ(k) for all t

Most classical methods assume stationarity. **Tests:**
- **ADF test:** H0 = unit root (non-stationary). Reject H0 → stationary.
- **KPSS test:** H0 = stationary. Reject H0 → non-stationary.

**Making a series stationary:**
1. Differencing: removes linear trends
2. Log transformation: stabilizes variance
3. Seasonal differencing: removes seasonal patterns

### STL Decomposition

STL (Seasonal and Trend decomposition using Loess) additively decomposes:
$$y_t = T_t + S_t + R_t$$

where T_t is trend, S_t is seasonal, R_t is residual.

Uses Loess (locally weighted regression) — robust to outliers, flexible to non-linear trends. Handles varying seasonality over time (unlike classical decomposition).

**Use case:** Decompose a series to understand its structure before modeling. Apply model to residuals or de-seasonalized trend.

---

## 4. Deep Learning for Time Series

### LSTM for Sequence Forecasting

Apply an LSTM encoder to the historical window, then predict the next step (or multiple steps) from the final hidden state.

```python
import torch
import torch.nn as nn

class LSTMForecaster(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, horizon):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,    # number of features per time step
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,         # input: (batch, time, features)
            dropout=0.2
        )
        self.fc = nn.Linear(hidden_size, horizon)  # predict H steps ahead
    
    def forward(self, x):
        # x: (batch, lookback, features)
        out, (h_n, c_n) = self.lstm(x)
        # Use final hidden state
        return self.fc(h_n[-1])  # (batch, horizon)
```

**Practical considerations:**
- **Lookback window:** How much history to feed. Too short = underfitting. Too long = noise, memory cost.
- **Normalization:** Normalize each series individually (instance normalization) — global normalization breaks when series have different scales.
- **Multi-step:** Predict H steps at once (direct forecasting) or autoregressively (one step at a time, feed predictions back). Direct is more accurate for long horizons; autoregressive accumulates error.

### TCN (Temporal Convolutional Network)

**The key idea:** Replace recurrence with dilated causal convolutions. Each layer looks at the past only (causal), and dilation exponentially expands the receptive field.

```
Layer 0: dilation=1 — sees 1 step back
Layer 1: dilation=2 — sees 2 steps back
Layer 2: dilation=4 — sees 4 steps back
Layer k: dilation=2^k — sees 2^k steps back
```

With 10 layers: receptive field of 2^10 = 1024 steps using only O(log T) layers.

**Advantages over LSTM:**
- Parallelizable during training (no sequential hidden state dependencies)
- Fixed receptive field (easier to reason about)
- No vanishing gradient across time (gradients flow through residual connections, not hidden states)

**Disadvantage:** Fixed receptive field — cannot dynamically extend for very long dependencies.

```python
class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        padding = (kernel_size - 1) * dilation  # causal padding
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation, padding=padding
        )
        self.chomp1 = Chomp1d(padding)  # remove future padding
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        # After conv + chomp: output only sees past inputs (causal)
        out = self.dropout(self.relu(self.chomp1(self.conv1(x))))
        return out + x  # residual connection
```

### N-BEATS (Oreshkin et al., 2020)

**Pure feed-forward architecture** with no recurrence or convolution — wins M4 forecasting competition.

**Key idea:** Stack of "blocks," each producing a **backcast** (explanation of the historical window) and **forecast** (prediction). Each block removes its backcast from the input signal (residual learning). The final forecast is the sum of all block forecasts.

```
Input window → Block 1 → backcast1, forecast1
Input - backcast1 → Block 2 → backcast2, forecast2
...
Total forecast = Σ forecast_k
```

**Interpretable variant:** Constrain the forecast basis functions to trend (polynomial) and seasonality (Fourier basis). Each block explicitly extracts trend or seasonal components.

---

## 5. Transformer-Based Forecasters

### Why Standard Transformers Are Problematic for Time Series

**The point-wise attention problem:** Standard transformer attention attends over individual time steps. For long series (thousands of steps), this is O(T²) in memory. More importantly, time series has strong **local temporal patterns** — adjacent points are more related than distant points, but attention does not naturally exploit this locality.

**Permutation invariance:** Standard transformers without positional encoding are permutation-invariant — they can't distinguish t=5 from t=50. Positional encoding helps but doesn't fully capture the temporal inductive bias that LSTMs and TCNs have.

### Informer (Zhou et al., 2021)

Designed for long sequence time-series forecasting (LSTF). Key contributions:

**ProbSparse self-attention:** Instead of full O(T²) attention, only compute attention for the top-u "dominant" queries (those with highest query-key dot product spread). Reduces to O(T log T).

**Distilling:** Between encoder layers, halve the sequence length via a strided max-pooling. This allows stacking many layers while keeping computation manageable.

**Direct multi-step decoder:** The decoder generates all H future steps in parallel (not autoregressively) — fed with a "start token" of recent history.

### PatchTST (Nie et al., 2023)

**The insight:** Treat time series like image patches in ViT. Instead of attending over individual time steps, divide the series into non-overlapping patches (e.g., 16-step patches) and attend over patches. This:
1. Reduces sequence length: T/P patches instead of T time steps → O((T/P)²) attention
2. Provides local semantic context: each token represents a meaningful temporal segment

**Instance normalization + channel independence:** Normalize each series independently. Treat each variate independently (no cross-series attention). Achieves SOTA on many benchmarks.

```python
# PatchTST core: divide time series into patches
class PatchEmbedding(nn.Module):
    def __init__(self, patch_len, d_model):
        super().__init__()
        self.patch_len = patch_len
        self.proj = nn.Linear(patch_len, d_model)
    
    def forward(self, x):
        # x: (batch, seq_len, features)
        # Divide into patches
        x = x.unfold(-2, self.patch_len, self.patch_len)  # (batch, n_patches, features, patch_len)
        batch, n_patches, features, pl = x.shape
        x = x.reshape(batch * features, n_patches, pl)  # treat channels independently
        return self.proj(x)  # (batch*features, n_patches, d_model)
```

### iTransformer (Liu et al., 2024)

**Inverted dimensions:** Apply attention across variates (features), not time steps. Each variate's full temporal sequence becomes a token. The transformer learns cross-variate correlations. Time mixing is done by the feed-forward network, not attention.

This flips the usual intuition — attention for feature interaction, FFN for temporal mixing — but achieves SOTA on multivariate benchmarks.

---

## 6. Anomaly Detection

**The task:** Identify time steps where the observed value is significantly unexpected given the historical context.

### Statistical Methods

**Z-score / moving z-score:** Flag observations more than k standard deviations from the local mean. Simple and fast; fails on trending or seasonal data.

**Isolation Forest:** Fits an ensemble of random trees; anomalies are isolated with fewer splits. Works on multivariate data; no temporal structure.

**CUSUM (Cumulative Sum):** Tracks cumulative deviation from expected value. Designed for detecting persistent mean shifts, not point anomalies.

### Reconstruction-Based Deep Learning

**Autoencoder approach:** Train autoencoder on normal (training) data. At test time: compute reconstruction error per time step. Anomalies have high reconstruction error (the model learned to reconstruct normal patterns; abnormal patterns cause high loss).

```python
class TimeSeriesAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, bottleneck):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.LSTM(input_size, hidden_size, batch_first=True),
        )
        self.decoder = nn.Sequential(
            nn.LSTM(bottleneck, hidden_size, batch_first=True),
        )
    
    def anomaly_score(self, x, threshold):
        # x: (batch, time, features)
        reconstruction = self.forward(x)
        mse_per_step = ((x - reconstruction) ** 2).mean(dim=-1)  # (batch, time)
        return mse_per_step > threshold
```

**LSTM-based:** Train LSTM to predict next step. Anomaly score = prediction error at each step. Captures temporal dependencies better than simple autoencoders.

**Threshold selection:**
- Percentile: set threshold at 95th/99th percentile of training reconstruction errors
- Dynamic: compute moving statistics on validation set; adapt threshold over time
- Point anomaly vs. contextual anomaly: a value of 0 at 3pm is not anomalous; at 3am it might be (contextual)

### Deep SVDD

**The idea:** Train a neural network to map normal data points inside a hypersphere of minimum volume. Anomalies map outside.

Loss: minimize the volume of the hypersphere enclosing all training representations — force normal data to map near the center c.

$$\mathcal{L} = \frac{1}{N}\sum_{i=1}^{N}\|f(x_i) - c\|^2$$

At test time: anomaly score = distance from center.

---

## 7. Evaluation Metrics

**Point forecast metrics:**

| Metric | Formula | Notes |
|---|---|---|
| MAE | $\frac{1}{H}\sum\|y_t - \hat{y}_t\|$ | Robust to outliers; same units as y |
| RMSE | $\sqrt{\frac{1}{H}\sum(y_t-\hat{y}_t)^2}$ | Penalizes large errors more; sensitive to outliers |
| MAPE | $\frac{100}{H}\sum\|\frac{y_t-\hat{y}_t}{y_t}\|$ | Percentage error; undefined when y_t=0; asymmetric |
| sMAPE | $\frac{200}{H}\sum\frac{\|y_t-\hat{y}_t\|}{\|y_t\|+\|\hat{y}_t\|}$ | Symmetric MAPE; still unstable near zero |
| MASE | $\frac{\text{MAE}}{\text{MAE}_{\text{naive}}}$ | Scale-free; < 1 means beats naive forecast |

**MASE in detail:**
$$\text{MASE} = \frac{\frac{1}{H}\sum_{t=T+1}^{T+H}|y_t - \hat{y}_t|}{\frac{1}{T-1}\sum_{t=2}^{T}|y_t - y_{t-1}|}$$

MASE compares the model's forecast error to the naive seasonal random walk forecast. MASE < 1 means the model outperforms the naive baseline. It is scale-free (can aggregate across series of different scales) and handles zero values. **Preferred metric for M4/M5 competitions.**

**Probabilistic forecast metrics:**
- **CRPS (Continuous Ranked Probability Score):** Proper scoring rule for distributional forecasts. Reduces to MAE for point forecasts.
- **Coverage:** Fraction of actuals falling within the 90% prediction interval. Should be ~90%.
- **Winkler score:** Penalizes prediction interval width AND coverage simultaneously.

---

## 8. Multi-Step Forecasting Strategies

**The challenge:** Forecast H steps ahead (H=1 is one-step; H=30 is a month). Different strategies have different error accumulation properties.

### DIRECT (Multi-Output)

Train one model per horizon h:
$$\hat{y}_{t+h} = f_h(y_t, y_{t-1}, \ldots, y_{t-p})$$

- No error propagation between steps
- H separate models: H× training cost
- Models do not share information across horizons
- Best for long horizons where error accumulation would dominate

### RECURSIVE (Iterated One-Step)

Train one one-step model. At inference, feed predictions back as inputs:
$$\hat{y}_{t+1} = f(y_t, \ldots), \quad \hat{y}_{t+2} = f(\hat{y}_{t+1}, y_t, \ldots), \ldots$$

- One model — training efficient
- Errors compound: 30-step forecasts amplify early mistakes
- Best for short horizons

### DIRECT-RECURSIVE (MIMO)

Train one model that directly outputs all H steps simultaneously:
$$(\hat{y}_{t+1}, \ldots, \hat{y}_{t+H}) = f(y_t, y_{t-1}, \ldots)$$

- No error propagation (unlike recursive)
- One model (unlike direct)
- Can capture correlations between forecast horizons
- Standard for deep learning forecasters (N-BEATS, Informer, PatchTST all use this)

### Exogenous Variables

Extend models to include external features: time-of-day, day-of-week, weather, economic indicators, known future events (promotions, holidays).

**Handling future unknowns:** For features not known at forecast time, either predict them separately (auxiliary forecast) or restrict to features known in advance (calendar features, pre-scheduled events).

---

## 9. Common Interview Questions

### Q1: What is the difference between AR and MA models? When would you use each?

**AR(p):** regress y_t on its own past p values. Models persistent autocorrelation — today's value is a linear function of recent past. Good for series with strong autocorrelation patterns.

**MA(q):** regress y_t on past q forecast errors. Models short-term shocks: a random shock at t-1 has a direct effect on y_t (through θ₁ε_{t-1}) but no direct effect at t-2 (unlike AR). Good for series where disturbances decay quickly.

Identify using ACF/PACF: MA → ACF cuts off after q; AR → PACF cuts off after p. In practice, use AIC/BIC for order selection.

---

### Q2: How do you handle non-stationarity in time series?

1. **Visual inspection:** plot the series; look for trend and changing variance
2. **ADF test:** if p-value > 0.05, cannot reject unit root hypothesis — difference the series
3. **First differencing:** Δy_t = y_t - y_{t-1} removes linear trend
4. **Seasonal differencing:** Δ_s y_t = y_t - y_{t-s} removes seasonal pattern
5. **Log transform:** stabilizes multiplicative variance (use when variance grows with level)

After differencing, re-test. Most financial/economic series need d=1; some need d=2.

---

### Q3: What is MASE and why is it preferred over MAPE?

MASE divides the model's MAE by the MAE of the naive seasonal random walk on the training data. **MASE < 1 means outperforms the naive baseline.**

**Advantages over MAPE:**
1. **Handles zero values:** MAPE is undefined when y_t = 0 (division by zero). MASE divides by a training-set MAE, always defined.
2. **Symmetric:** MAPE is asymmetric — a 100% overforecast (predict 2, actual 1) is penalized more than a 100% underforecast (predict 0.5, actual 1).
3. **Scale-free:** Can aggregate MASE across series of different scales (e.g., sales of different products) without high-volume series dominating.

---

### Q4: Why are vanilla Transformers problematic for long time series forecasting?

1. **O(T²) complexity:** attention over T=3000 time steps requires 9M attention weights per head — memory explosion for long sequences. Informer (ProbSparse) and PatchTST (patch tokens) address this.
2. **Point-wise tokens lack temporal locality:** adjacent time steps are more correlated than distant ones, but attention treats all positions symmetrically. PatchTST groups nearby steps into patches to give each token local context.
3. **Permutation sensitivity:** sinusoidal positional encodings don't strongly enforce temporal order — the model doesn't "know" that t=5 is adjacent to t=6 the way an LSTM does.

The best solutions (PatchTST, iTransformer) redesign the tokenization strategy rather than the attention mechanism.

---

### Q5: When should you use deep learning vs. classical methods for time series?

**Use classical (ARIMA, ETS, Prophet):**
- Short, univariate series with clear trend/seasonality
- Small data (< a few hundred points per series)
- Need interpretable components (trend/seasonality)
- Fast iteration, no GPU infrastructure

**Use deep learning (LSTM, N-BEATS, PatchTST):**
- Large datasets (many series or very long histories)
- Multivariate with complex cross-series dependencies
- Non-linear patterns that classical models can't capture
- Pre-trained models available (transfer learning from other datasets)

**Practical reality:** N-BEATS and LightGBM with lag features often beat LSTM. Classical methods with ensembling are competitive on standard benchmarks. Deep learning wins when data is abundant and patterns are complex.

---

## Key Papers

| Paper | Year | Contribution |
|---|---|---|
| Box & Jenkins | 1970 | ARIMA — foundational time series framework |
| Holt-Winters | 1960s | Exponential smoothing with trend + seasonality |
| Taylor & Letham | 2018 | Prophet — decomposable forecasting with changepoints |
| Oreshkin et al. | 2020 | N-BEATS — pure feed-forward beating LSTM/ARIMA |
| Zhou et al. | 2021 | Informer — efficient transformer for long sequences |
| Nie et al. | 2023 | PatchTST — patch-based transformer for TS |
| Liu et al. | 2024 | iTransformer — inverted attention over variates |
| Ruff et al. | 2018 | Deep SVDD — deep one-class anomaly detection |
