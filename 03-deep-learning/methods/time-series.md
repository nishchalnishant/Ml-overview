# Time Series Machine Learning — Comprehensive Reference

---

## Table of Contents

1. What Makes Time Series Special
2. Classical Methods
3. Feature Engineering for Time Series
4. RNNs and LSTMs
5. Temporal Convolutional Networks (TCN)
6. Transformer-Based Models
7. N-BEATS and N-HiTS
8. Anomaly Detection
9. Evaluation Metrics
10. Cross-Validation for Time Series
11. Multivariate Forecasting
12. Probabilistic Forecasting
13. Foundation Models for Time Series
14. Production Patterns
15. Common Interview Questions

---

## 1. What Makes Time Series Special

A time series is a sequence of observations indexed by time: `y_1, y_2, ..., y_T`. The critical distinction from standard tabular data is that **order matters**. Shuffling rows destroys information.

### 1.1 Temporal Ordering

Every sample depends on its history. A stock price tomorrow is conditionally dependent on prices today, yesterday, and further back. Treating observations as i.i.d. is a modeling error that leads to:

- **Data leakage**: using future information to predict the past (classic validation mistake)
- **Incorrect confidence intervals**: assuming independence inflates statistical power
- **Incorrect feature construction**: computing a rolling mean on the full dataset before splitting

### 1.2 Trend

A **trend** is a long-run, systematic movement in the mean level of the series.

```
Energy demand (GW):
2000: 400
2005: 420
2010: 445
2015: 470   <-- upward trend
```

Linear trend: `y_t = alpha + beta*t + epsilon_t`

Trend can be:
- **Additive**: fluctuations have constant magnitude around trend
- **Multiplicative**: fluctuations scale proportionally with the level (common in finance)

Decompose multiplicative to additive with a log transform: `log(y_t)`.

### 1.3 Seasonality

**Seasonality** is a repeating pattern at a fixed, known frequency.

| Domain | Period | Frequency |
|---|---|---|
| Retail sales | Weekly | 7 |
| Energy demand | Daily | 24 (hourly) |
| Stock volume | Yearly | 252 (trading days) |
| Server traffic | Weekly + Daily | Multiple |

Multiple seasonality is common and hard. A server metric can have both daily peaks and weekly cycles simultaneously.

### 1.4 Stationarity

A **stationary** series has:
- Constant mean: `E[y_t] = mu` for all `t`
- Constant variance: `Var(y_t) = sigma^2` for all `t`
- Autocovariance depends only on lag, not time: `Cov(y_t, y_{t+k}) = gamma(k)`

**Why it matters**: most classical methods assume stationarity. Neural methods are more robust but still benefit from stationary inputs.

**Tests for stationarity:**

*Augmented Dickey-Fuller (ADF)*: null hypothesis is unit root (non-stationary). A p-value < 0.05 suggests stationarity.

```python
from statsmodels.tsa.stattools import adfuller

result = adfuller(series)
print(f"ADF Statistic: {result[0]:.4f}")
print(f"p-value: {result[1]:.4f}")
# p < 0.05 -> reject unit root -> likely stationary
```

*KPSS test*: null hypothesis is stationarity. Use both ADF and KPSS together to confirm.

**Making a series stationary:**
- First-order differencing: `delta_y_t = y_t - y_{t-1}` removes linear trend
- Seasonal differencing: `delta_s y_t = y_t - y_{t-s}` removes seasonal structure
- Log transform: stabilizes variance
- Box-Cox transform: general variance stabilization

### 1.5 Autocorrelation

**ACF (Autocorrelation Function)**: measures correlation between `y_t` and `y_{t-k}`.
**PACF (Partial Autocorrelation Function)**: measures correlation after removing effects of intermediate lags.

These plots directly guide ARIMA order selection (see Section 2).

```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
plot_acf(series, lags=40, ax=ax1)
plot_pacf(series, lags=40, ax=ax2)
plt.tight_layout()
plt.show()
```

---

## 2. Classical Methods

### 2.1 ARIMA

**AutoRegressive Integrated Moving Average**: ARIMA(p, d, q)

- **AR(p)**: autoregressive term — regress on own past `p` lags
- **I(d)**: integrated term — apply `d`-order differencing to achieve stationarity
- **MA(q)**: moving average term — regress on past `q` forecast errors

Model equation:
```
delta^d y_t = c + phi_1*delta^d y_{t-1} + ... + phi_p*delta^d y_{t-p}
              + theta_1*epsilon_{t-1} + ... + theta_q*epsilon_{t-q} + epsilon_t
```

**Identifying p, d, q from ACF/PACF:**
- `d`: number of differences needed to make series stationary (ADF test)
- `p`: PACF cuts off after lag `p` (AR signature)
- `q`: ACF cuts off after lag `q` (MA signature)

```python
from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(train, order=(2, 1, 2))
result = model.fit()
print(result.summary())

forecast = result.forecast(steps=30)
```

**Information criteria for order selection:**
- AIC (Akaike): `AIC = 2k - 2*ln(L)` — penalizes complexity
- BIC (Bayesian): `BIC = k*ln(n) - 2*ln(L)` — stronger penalty for large `n`

Auto-ARIMA (pmdarima) searches over orders automatically:

```python
import pmdarima as pm

model = pm.auto_arima(train,
                      start_p=0, max_p=5,
                      start_q=0, max_q=5,
                      d=None,           # auto-determine d
                      seasonal=False,
                      information_criterion='aic',
                      stepwise=True)
```

### 2.2 SARIMA

**Seasonal ARIMA**: SARIMA(p, d, q)(P, D, Q)[S]

Adds seasonal AR and MA terms operating at seasonal lag S.

```
SARIMA(1,1,1)(1,1,1)[12]  -- monthly data with annual seasonality
```

Parameters:
- `(P, D, Q)`: seasonal AR order, seasonal differencing, seasonal MA order
- `[S]`: seasonal period (12 for monthly, 4 for quarterly, 7 for daily-with-weekly)

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

model = SARIMAX(train,
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 12))
result = model.fit(disp=False)
forecast = result.get_forecast(steps=12)
conf_int = forecast.conf_int()
```

Limitation: SARIMA handles only one seasonal period. For multiple seasonalities (e.g., hourly data with daily + weekly cycles), consider TBATS or neural methods.

### 2.3 Exponential Smoothing

Weighted average of past observations, with exponentially decaying weights.

**Simple Exponential Smoothing (SES)** — no trend, no seasonality:
```
s_t = alpha * y_t + (1 - alpha) * s_{t-1}
y_hat_{t+1} = s_t
```
`alpha` in (0,1): high alpha = more weight on recent observations.

**Holt's Linear Method** — handles trend:
```
Level:  l_t = alpha * y_t + (1-alpha) * (l_{t-1} + b_{t-1})
Trend:  b_t = beta  * (l_t - l_{t-1}) + (1-beta) * b_{t-1}
Forecast: y_hat_{t+h} = l_t + h * b_t
```

**Holt-Winters (Triple Exponential Smoothing)** — handles trend + seasonality:

Additive version:
```
Level:    l_t = alpha*(y_t - s_{t-m}) + (1-alpha)*(l_{t-1} + b_{t-1})
Trend:    b_t = beta*(l_t - l_{t-1}) + (1-beta)*b_{t-1}
Seasonal: s_t = gamma*(y_t - l_{t-1} - b_{t-1}) + (1-gamma)*s_{t-m}
Forecast: y_hat_{t+h} = l_t + h*b_t + s_{t-m+h}
```

```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing

model = ExponentialSmoothing(train,
                             trend='add',
                             seasonal='add',
                             seasonal_periods=12)
result = model.fit()
forecast = result.forecast(12)
```

Multiplicative variant (`seasonal='mul'`) is better when seasonal amplitude grows with level (e.g., e-commerce sales growing year over year).

### 2.4 Prophet

Facebook Prophet decomposes the series as:
```
y(t) = trend(t) + seasonality(t) + holidays(t) + epsilon_t
```

Trend: piecewise linear or logistic growth with automatic changepoint detection.
Seasonality: Fourier series approximation.

```python
from prophet import Prophet
import pandas as pd

df = pd.DataFrame({'ds': dates, 'y': values})

model = Prophet(
    changepoint_prior_scale=0.05,  # flexibility of trend
    seasonality_prior_scale=10,
    yearly_seasonality=True,
    weekly_seasonality=True
)
model.fit(df)

future = model.make_future_dataframe(periods=90)
forecast = model.predict(future)
```

Prophet is robust to missing data and outliers, easy to tune with domain knowledge (e.g., custom holidays). Not great for very short-horizon or high-frequency data.

---

## 3. Feature Engineering for Time Series

Neural models still benefit enormously from well-engineered features. Gradient boosted trees (LightGBM, XGBoost) with time series features often outperform vanilla neural approaches.

### 3.1 Lag Features

The most direct representation of temporal dependence:

```python
import pandas as pd
import numpy as np

def add_lag_features(df, target_col, lags):
    for lag in lags:
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    return df

# Example: predict energy demand 24h ahead
lags = [1, 2, 3, 6, 12, 24, 48, 168]  # 168 = 1 week in hours
df = add_lag_features(df, 'demand_gw', lags)
```

**Critical**: only shift forward (no negative lags) to avoid leakage. After shift, drop rows with NaN from the start.

### 3.2 Rolling Statistics

Capture local statistics over sliding windows:

```python
def add_rolling_features(df, col, windows):
    for w in windows:
        df[f'{col}_rolling_mean_{w}'] = df[col].shift(1).rolling(w).mean()
        df[f'{col}_rolling_std_{w}']  = df[col].shift(1).rolling(w).std()
        df[f'{col}_rolling_min_{w}']  = df[col].shift(1).rolling(w).min()
        df[f'{col}_rolling_max_{w}']  = df[col].shift(1).rolling(w).max()
    return df

# Note: shift(1) before rolling to prevent leakage
windows = [7, 14, 30, 90]
df = add_rolling_features(df, 'demand_gw', windows)
```

**Expanding window** (cumulative statistics):
```python
df['cumulative_mean'] = df['target'].shift(1).expanding().mean()
```

### 3.3 Calendar Features

Encode temporal position explicitly:

```python
def add_calendar_features(df, datetime_col):
    dt = pd.to_datetime(df[datetime_col])
    df['hour']        = dt.dt.hour
    df['day_of_week'] = dt.dt.dayofweek      # 0=Monday
    df['day_of_month']= dt.dt.day
    df['week_of_year']= dt.dt.isocalendar().week.astype(int)
    df['month']       = dt.dt.month
    df['quarter']     = dt.dt.quarter
    df['year']        = dt.dt.year
    df['is_weekend']  = (dt.dt.dayofweek >= 5).astype(int)
    return df
```

Avoid treating hour/month as raw integers for neural models — they wrap around. Use cyclical encoding:

```python
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
```

This ensures hour 23 is close to hour 0 in feature space.

### 3.4 Fourier Features

Represent seasonality as a sum of sine and cosine waves:

```python
def fourier_features(t, period, n_terms):
    """
    t: array of time indices
    period: seasonality period (e.g., 365.25 for yearly)
    n_terms: number of harmonics
    """
    features = {}
    for k in range(1, n_terms + 1):
        features[f'sin_{period}_{k}'] = np.sin(2 * np.pi * k * t / period)
        features[f'cos_{period}_{k}'] = np.cos(2 * np.pi * k * t / period)
    return pd.DataFrame(features)

# Daily seasonality with 5 harmonics (for hourly data)
t = np.arange(len(df))
fourier = fourier_features(t, period=24, n_terms=5)
df = pd.concat([df, fourier], axis=1)
```

More harmonics = sharper seasonality shape captured, but more parameters.

### 3.5 Target Encoding and External Features

- **Holidays**: binary or categorical feature, important for retail and energy
- **Weather data**: temperature is a strong predictor for energy demand
- **Economic indicators**: leading indicators for demand forecasting
- **Exogenous variables**: known future values (e.g., planned promotions)

```python
# Example: energy demand with temperature covariate
# SARIMAX handles exogenous variables
model = SARIMAX(train_demand,
                exog=train_temperature,
                order=(1,1,1),
                seasonal_order=(1,1,1,24))
result = model.fit()
forecast = result.get_forecast(steps=24, exog=future_temperature)
```

---

## 4. RNNs and LSTMs

### 4.1 Vanilla RNN

An RNN maintains a hidden state `h_t` updated at each timestep:

```
h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b_h)
y_t = W_hy * h_t + b_y
```

**Vanishing gradient problem**: gradients from long-ago timesteps shrink exponentially through backpropagation through time (BPTT). This limits practical memory to ~10-20 timesteps.

### 4.2 LSTM Architecture

Long Short-Term Memory (Hochreiter & Schmidhuber, 1997) introduces a **cell state** `c_t` and three gates:

```
Forget gate:  f_t = sigmoid(W_f * [h_{t-1}, x_t] + b_f)
Input gate:   i_t = sigmoid(W_i * [h_{t-1}, x_t] + b_i)
Cell update:  g_t = tanh(W_g * [h_{t-1}, x_t] + b_g)
Output gate:  o_t = sigmoid(W_o * [h_{t-1}, x_t] + b_o)

Cell state:   c_t = f_t * c_{t-1} + i_t * g_t
Hidden state: h_t = o_t * tanh(c_t)
```

- **Forget gate**: decides what to erase from cell state
- **Input gate + cell update**: decides what new info to add
- **Output gate**: decides what to expose as hidden state

The additive cell state update `c_t = f_t*c_{t-1} + i_t*g_t` prevents vanishing gradients — gradients can flow through without attenuation.

```python
import torch
import torch.nn as nn

class LSTMForecaster(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Use last timestep output
        out = self.fc(lstm_out[:, -1, :])  # (batch, output_size)
        return out

model = LSTMForecaster(input_size=10, hidden_size=64,
                       num_layers=2, output_size=1)
```

**GRU (Gated Recurrent Unit)**: simpler than LSTM with only two gates (reset and update). Often comparable performance with fewer parameters.

### 4.3 Sequence-to-Sequence for Multi-step Forecasting

For predicting `H` steps ahead, two architectures:

**Direct approach**: train `H` separate models, each predicting horizon `h`.
- Pro: no error accumulation
- Con: does not model dependencies between horizons

**Recursive approach**: predict one step, feed prediction back as input.
- Pro: single model
- Con: errors compound over the horizon

**Seq2Seq with encoder-decoder**:

```python
class Seq2SeqLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, horizon):
        super().__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(1, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, 1)
        self.horizon = horizon

    def forward(self, src, tgt=None, teacher_forcing_ratio=0.5):
        # Encode
        _, (h, c) = self.encoder(src)

        # Decode step by step
        outputs = []
        dec_input = src[:, -1:, :1]  # last observed value

        for t in range(self.horizon):
            out, (h, c) = self.decoder(dec_input, (h, c))
            pred = self.output(out)
            outputs.append(pred)

            # Teacher forcing: use ground truth with probability p
            if tgt is not None and torch.rand(1) < teacher_forcing_ratio:
                dec_input = tgt[:, t:t+1, :]
            else:
                dec_input = pred

        return torch.cat(outputs, dim=1)
```

**Teacher forcing**: during training, feed the ground truth as decoder input with probability `p` instead of the model's own prediction. Speeds up training but can cause exposure bias at inference. Schedule the ratio downward during training.

### 4.4 Practical LSTM Tips

- **Input normalization**: scale each feature to zero mean, unit variance using training statistics only. Apply same scaler to validation/test.
- **Stacking**: 2-3 layers usually sufficient; more causes overfitting
- **Lookback window**: start with `2x` the longest seasonal period; tune via validation
- **Dropout**: apply between LSTM layers, not within (use `nn.Dropout` after LSTM)
- **Gradient clipping**: clip gradients to norm 1.0 for stability (`torch.nn.utils.clip_grad_norm_`)

---

## 5. Temporal Convolutional Networks (TCN)

TCNs apply dilated causal convolutions to sequences, offering parallelism that RNNs lack.

### 5.1 Causal Convolution

A **causal** convolution ensures that output at time `t` depends only on inputs at time `<= t` (no future leakage):

```
y_t = sum_{k=0}^{K-1} w_k * x_{t-k}
```

Padding on the left by `K-1` zeros achieves causality.

### 5.2 Dilated Convolution

**Dilation** increases the receptive field exponentially without adding parameters:

```
y_t = sum_{k=0}^{K-1} w_k * x_{t - d*k}
```

With kernel size 2 and exponentially increasing dilation `d = 1, 2, 4, 8, ...`:
- Layer 1 (d=1): sees 2 steps back
- Layer 2 (d=2): sees 4 steps back
- Layer 3 (d=4): sees 8 steps back
- Layer L (d=2^L): sees 2^(L+1) steps back

Receptive field = `1 + (K-1) * sum(dilations)`.

### 5.3 TCN Implementation

```python
class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              dilation=dilation)

    def forward(self, x):
        # x: (batch, channels, seq_len)
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)

class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout=0.2):
        super().__init__()
        self.conv1 = CausalConv1d(in_ch, out_ch, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_ch, out_ch, kernel_size, dilation)
        self.norm1 = nn.LayerNorm(out_ch)
        self.norm2 = nn.LayerNorm(out_ch)
        self.drop  = nn.Dropout(dropout)
        self.skip  = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu  = nn.ReLU()

    def forward(self, x):
        # x: (batch, in_ch, seq_len)
        res = x if self.skip is None else self.skip(x)
        x = self.relu(self.norm1(self.conv1(x).transpose(1,2)).transpose(1,2))
        x = self.drop(x)
        x = self.relu(self.norm2(self.conv2(x).transpose(1,2)).transpose(1,2))
        x = self.drop(x)
        return self.relu(x + res)
```

**TCN advantages over LSTM:**
- Parallelism: all timesteps computed simultaneously (no sequential dependency)
- Stable gradients: no vanishing gradient through time
- Flexible receptive field: controlled by kernel size and dilation schedule

**TCN disadvantages:**
- Fixed receptive field must be set to cover longest dependency
- Less natural for online/streaming inference

---

## 6. Transformer-Based Models

Standard Transformers have `O(L^2)` attention complexity, which is prohibitive for long sequences (L = 1000+ timesteps). Several architectures address this.

### 6.1 Vanilla Transformer Baseline

Self-attention computes:
```
Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
```

For time series, positional encoding injects temporal order. The full sequence is processed in parallel, which is training-efficient but requires the whole context window.

### 6.2 Informer (2021)

**Problem**: `O(L^2)` memory for long sequences.
**Solution**: ProbSparse attention — identify the top-u queries that have the most influence, compute attention only for those.

ProbSparse attention complexity: `O(L * log L)`

Also introduces:
- **Distilling**: halve sequence length across encoder layers
- **Generative decoder**: directly outputs the full forecast horizon (vs. autoregressive)

```python
# Using HuggingFace time_series_transformer (Informer variant)
from transformers import InformerConfig, InformerForPrediction

config = InformerConfig(
    prediction_length=24,
    context_length=96,
    input_size=1,
    num_time_features=4,
    d_model=64,
    encoder_layers=2,
    decoder_layers=1,
    encoder_attention_heads=4,
    decoder_attention_heads=4,
)
model = InformerForPrediction(config)
```

### 6.3 Autoformer (2021)

**Key innovation**: Auto-Correlation mechanism replaces attention.

Instead of token-level attention, computes period-based dependencies via fast Fourier transform:
```
Correlation(Q, K) = IFFT(FFT(Q) * conj(FFT(K)))
```

Also introduces **series decomposition block**: separates trend from seasonality inside every encoder/decoder layer rather than as a preprocessing step.

This inductive bias is strong for periodic series but may not generalize as well on non-periodic data.

### 6.4 PatchTST (2023)

**Key insight**: patches of consecutive time steps are more meaningful tokens than individual timesteps.

```
Sequence of length L -> divide into patches of size P -> L/P tokens
```

Benefits:
- Reduces sequence length by factor `P`
- Each patch token carries local context
- Enables use of standard (full) self-attention with manageable cost
- Channel-independence: each variable processed separately (strong inductive bias)

PatchTST often outperforms earlier Transformer variants while being simpler.

```python
# Pseudo-code for patch embedding
class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, d_model, seq_len):
        super().__init__()
        self.patch_size = patch_size
        n_patches = seq_len // patch_size
        self.proj = nn.Linear(patch_size, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches, d_model))

    def forward(self, x):
        # x: (batch, seq_len)
        B, L = x.shape
        x = x.reshape(B, L // self.patch_size, self.patch_size)
        x = self.proj(x) + self.pos_embed
        return x  # (batch, n_patches, d_model)
```

### 6.5 iTransformer (2024)

**Inverted Transformer**: apply attention across **variables** rather than across time.

Standard Transformer: each token = one timestep, attention = temporal dependencies.
iTransformer: each token = one variable's full time series, attention = inter-variable dependencies.

Why it works for multivariate: the temporal structure is encoded by the feed-forward layer per variable; the attention learns cross-variable correlation patterns.

This is particularly effective when the number of variables is moderate and cross-variable dependencies matter (e.g., multiple sensor readings in a factory).

### 6.6 Transformer Comparison Table

| Model | Complexity | Key Innovation | Best For |
|---|---|---|---|
| Informer | O(L log L) | ProbSparse attention | Long sequences |
| Autoformer | O(L log L) | Auto-Correlation + decomp | Periodic series |
| PatchTST | O((L/P)^2) | Patch tokenization | General forecasting |
| iTransformer | O(M^2 * L) | Inverted attention | Multivariate |

---

## 7. N-BEATS and N-HiTS

### 7.1 N-BEATS (2020)

**Neural Basis Expansion Analysis for Time Series** — pure MLP-based, no recurrence, no convolution.

Architecture:
- Stack of **blocks**, each producing a **backcast** (reconstruction of input) and **forecast**
- Residual connections: each block operates on the residual after previous blocks subtract their backcast
- **Interpretable variant**: basis functions are trend polynomials and Fourier harmonics

```
Block k:
  theta_b, theta_f = MLP(x_residual_k)
  backcast_k = basis_b(theta_b)
  forecast_k  = basis_f(theta_f)
  x_residual_{k+1} = x_residual_k - backcast_k
  
Total forecast = sum(forecast_k for all k)
```

Interpretable stacks:
- **Trend stack**: polynomial basis `[1, t, t^2, t^3]`
- **Seasonality stack**: Fourier basis `[cos(2pi*k*t/P), sin(2pi*k*t/P)]`

```python
# Using neuralforecast library
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS

model = NBEATS(
    h=24,                    # forecast horizon
    input_size=72,           # lookback window
    stack_types=['trend', 'seasonality'],
    n_blocks=[3, 3],
    mlp_units=[[512, 512], [512, 512]],
    n_harmonics=2,
    n_polynomials=2,
    max_steps=1000,
    scaler_type='standard'
)

nf = NeuralForecast(models=[model], freq='H')
nf.fit(df)
forecast = nf.predict()
```

### 7.2 N-HiTS (2022)

**N-HiTS** (Hierarchical Interpolation for Time Series) extends N-BEATS with:

1. **Multi-rate sampling**: different blocks sample the input at different rates (pooling), capturing short and long-range patterns
2. **Hierarchical interpolation**: forecast is generated at coarser resolution then upsampled, reducing parameter count

The key idea is that most forecasting signal lives in low-frequency components. N-HiTS exploits this by allocating more parameters to low-frequency stacks.

```python
from neuralforecast.models import NHITS

model = NHITS(
    h=24,
    input_size=72,
    stack_types=['identity', 'identity', 'identity'],
    n_blocks=[1, 1, 1],
    mlp_units=[[512, 512]] * 3,
    n_pool_kernel_size=[2, 2, 1],    # pooling rates per stack
    n_freq_downsample=[4, 2, 1],     # interpolation rates
    max_steps=1000
)
```

**N-BEATS vs N-HiTS vs Transformers**: on M4 and M5 benchmarks, N-BEATS and N-HiTS frequently outperform Transformer variants while being simpler and faster. The MLP-based approach is surprisingly competitive.

---

## 8. Anomaly Detection

### 8.1 Statistical Tests

**Z-score**: flag observations more than `k` standard deviations from the rolling mean.

```python
def zscore_anomalies(series, window=30, threshold=3.0):
    rolling_mean = series.rolling(window).mean()
    rolling_std  = series.rolling(window).std()
    z_scores = (series - rolling_mean) / rolling_std
    return z_scores.abs() > threshold
```

**Grubbs' test**: formally tests whether a single outlier exists in a normally distributed series. Use iteratively to find multiple outliers.

**Seasonal Hybrid ESD (S-H-ESD)**: Twitter's method — decompose seasonality first, then apply ESD test on residuals. Used in production monitoring.

### 8.2 Isolation Forest

Isolates anomalies by recursively splitting on random features. Anomalies are isolated faster (shorter average path length in the tree).

```python
from sklearn.ensemble import IsolationForest
import numpy as np

# Build feature matrix from lag and rolling features
X = build_features(series)  # shape: (n_samples, n_features)

clf = IsolationForest(
    n_estimators=100,
    contamination=0.05,   # expected fraction of anomalies
    random_state=42
)
clf.fit(X)
anomaly_scores = clf.decision_function(X)  # lower = more anomalous
labels = clf.predict(X)  # -1 = anomaly, 1 = normal
```

Isolation Forest works well on multivariate data (e.g., server CPU + memory + disk I/O).

### 8.3 LSTM Autoencoder

Train an LSTM autoencoder on normal data. Anomalies produce high reconstruction error.

```python
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, seq_len):
        super().__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, input_size, batch_first=True)
        self.seq_len = seq_len

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        _, (h, c) = self.encoder(x)
        # Repeat context vector for each decoder timestep
        context = h[-1].unsqueeze(1).repeat(1, self.seq_len, 1)
        reconstruction, _ = self.decoder(context)
        return reconstruction

# Training
model = LSTMAutoencoder(input_size=1, hidden_size=32, seq_len=50)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# After training on normal data, compute threshold from training reconstruction errors
train_errors = []
with torch.no_grad():
    for batch in train_loader:
        recon = model(batch)
        errors = ((batch - recon) ** 2).mean(dim=(1, 2))
        train_errors.extend(errors.numpy())

threshold = np.percentile(train_errors, 95)

# Anomaly detection
with torch.no_grad():
    recon = model(test_data)
    test_errors = ((test_data - recon) ** 2).mean(dim=(1, 2))
    anomalies = test_errors > threshold
```

### 8.4 VAE for Anomaly Detection

A Variational Autoencoder learns a probabilistic latent space. Anomalies produce high reconstruction loss AND have low probability under the learned prior:

```
anomaly_score = reconstruction_loss + beta * KL_divergence
```

DONUT (2018) and DAGMM apply this idea specifically to multivariate time series.

### 8.5 Anomaly Detection in Production

Key decisions:
1. **Threshold**: set on held-out normal data, typically at 95th or 99th percentile of reconstruction errors. Can use 3-sigma rule.
2. **Smoothing**: apply moving average to anomaly scores to reduce point false positives.
3. **Contextual vs point anomalies**: a value of 1000 CPU at 3 AM is anomalous; at 2 PM it is not. Incorporate time-of-day context.
4. **Drift vs spike**: distinguish gradual drift (concept drift) from sudden spikes.

---

## 9. Evaluation Metrics

### 9.1 Point Forecast Metrics

**MAE (Mean Absolute Error)**:
```
MAE = (1/n) * sum(|y_t - y_hat_t|)
```
Scale-dependent. Robust to outliers relative to MSE. Lower is better.

**RMSE (Root Mean Squared Error)**:
```
RMSE = sqrt((1/n) * sum((y_t - y_hat_t)^2))
```
Same units as target. Penalizes large errors more than MAE. Useful when large errors are especially costly.

**MAPE (Mean Absolute Percentage Error)**:
```
MAPE = (100/n) * sum(|y_t - y_hat_t| / |y_t|)
```
Scale-free. **Problematic when `y_t` is near zero** (division instability) and is asymmetric: overforecasts by the same absolute amount are penalized less than underforecasts.

**sMAPE (Symmetric MAPE)**:
```
sMAPE = (100/n) * sum(2*|y_t - y_hat_t| / (|y_t| + |y_hat_t|))
```
Mitigates asymmetry. Still problematic near zero.

**MASE (Mean Absolute Scaled Error)**:
```
MASE = MAE / (1/(T-s)) * sum(|y_t - y_{t-s}|)
```
Denominator is the MAE of the naive seasonal forecast (predict today = same period last season). MASE > 1 means your model is worse than naive. MASE is the **recommended metric for academic benchmarks** (M-competitions) — scale-free and well-behaved.

```python
def mase(actual, forecast, seasonal_naive_errors):
    mae = np.mean(np.abs(actual - forecast))
    naive_mae = np.mean(np.abs(seasonal_naive_errors))
    return mae / naive_mae
```

### 9.2 Probabilistic Forecast Metrics

**Quantile Loss (Pinball Loss)**:
```
QL_q(y, y_hat_q) = q * (y - y_hat_q)    if y >= y_hat_q
                 = (1-q) * (y_hat_q - y) if y < y_hat_q
```
Measures calibration of quantile `q`. Sum over quantiles gives WQL (Weighted Quantile Loss).

**CRPS (Continuous Ranked Probability Score)**:
```
CRPS(F, y) = integral_{-inf}^{inf} (F(x) - 1[x >= y])^2 dx
```
Where `F` is the predicted CDF. Generalizes MAE to distributional forecasts. A CRPS equal to MAE for a point forecast — so CRPS >= 0 and lower is better.

```python
# Using properscoring library
from properscoring import crps_ensemble
import numpy as np

# samples: (n_samples_per_timestep,) or (n_timesteps, n_ensemble_members)
crps_scores = crps_ensemble(actual, ensemble_samples)
print(f"Mean CRPS: {crps_scores.mean():.4f}")
```

**Calibration**: for an interval forecast, check empirical coverage. If 90% intervals contain the true value 90% of the time, the model is calibrated.

### 9.3 Choosing Metrics

| Situation | Recommended Metric |
|---|---|
| Scale-free comparison across series | MASE or sMAPE |
| Same series, different models | RMSE or MAE |
| High-cost large errors | RMSE |
| Zero-crossing series | RMSE or MASE (not MAPE) |
| Probabilistic forecasts | CRPS or WQL |
| Interval forecasts | Coverage + interval width |

---

## 10. Cross-Validation for Time Series

### 10.1 Why Standard k-Fold Fails

Standard k-fold randomly assigns observations to folds. For time series, this causes:
1. **Future leakage**: training on observations from 2023 to predict 2022
2. **Dependency violation**: train and test share observations from the same temporal neighborhood

### 10.2 Train-Validation-Test Split

The simplest valid approach: use a fixed cutoff.

```python
n = len(df)
train_end = int(n * 0.70)
val_end   = int(n * 0.85)

train = df.iloc[:train_end]
val   = df.iloc[train_end:val_end]
test  = df.iloc[val_end:]
```

Validate on `val`, report final numbers on `test` (touched once).

### 10.3 Time Series Split (Expanding Window)

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5, gap=24)  # gap avoids leakage near cutoff

for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val     = X[val_idx], y[val_idx]

    model.fit(X_train, y_train)
    score = evaluate(model, X_val, y_val)
    print(f"Fold {fold}: {score:.4f}")
```

Each fold expands the training set. The `gap` parameter skips observations near the boundary (important for lag features to avoid leakage).

### 10.4 Walk-Forward Validation (Sliding Window)

Both train and validation windows slide forward, keeping training size constant:

```python
def walk_forward_cv(series, train_size, val_size, step=1):
    results = []
    start = 0
    while start + train_size + val_size <= len(series):
        train = series[start:start + train_size]
        val   = series[start + train_size:start + train_size + val_size]

        model = fit_model(train)
        preds = model.predict(val_size)
        score = evaluate(preds, val)
        results.append(score)

        start += step

    return results
```

Walk-forward is more computationally expensive but better represents production deployment where the model is retrained on a rolling basis.

### 10.5 Backtesting Caveat

**Temporal leakage in feature engineering**: compute rolling statistics and lag features on the full dataset before splitting, and you will have leaked future information into lagged features near the split boundary. Always:

1. Create lag features using `shift()` — safe because it references past only
2. Use `shift(1)` before any rolling computation
3. Set `min_periods` to avoid NaN-propagation

---

## 11. Multivariate Forecasting

### 11.1 Problem Formulation

Given `M` time series: `Y_t = [y_t^1, ..., y_t^M]`, predict `H` steps ahead for all `M` variables.

### 11.2 Channel-Independent vs Channel-Dependent

**Channel-independent (CI)**: treat each variable independently. Same model applied separately to each variable. PatchTST and most linear models operate this way.

**Channel-dependent (CD)**: model cross-variable interactions explicitly. iTransformer, GNNs for time series.

Empirically, CI often wins on benchmarks because it regularizes overfitting. CD is better when variables have known, strong causal relationships.

### 11.3 Vector Autoregression (VAR)

Classical multivariate extension of AR:

```
Y_t = c + A_1 * Y_{t-1} + A_2 * Y_{t-2} + ... + A_p * Y_{t-p} + epsilon_t
```

Each variable is regressed on its own past AND past of all other variables. Number of parameters scales as `M^2 * p` — expensive for large `M`.

```python
from statsmodels.tsa.vector_ar.var_model import VAR

model = VAR(train_data)  # shape: (T, M)
results = model.fit(maxlags=15, ic='aic')
forecast = results.forecast(train_data[-results.k_ar:], steps=12)
```

### 11.4 LightGBM/XGBoost for Multivariate

Often the most practical approach for tabular-izable multivariate forecasting:

1. Build a feature matrix: lag features of all `M` variables, calendar features, rolling stats
2. Train a single gradient boosting model to predict each target variable
3. Use recursive or direct strategy for multi-step

```python
import lightgbm as lgb
from sklearn.multioutput import MultiOutputRegressor

# Build feature matrix (T - max_lag) x (M * n_lags + n_calendar_features)
X_train, y_train = build_feature_matrix(train_data, lags=[1,2,3,24], horizon=1)

model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=31)
model.fit(X_train, y_train)
```

---

## 12. Probabilistic Forecasting

### 12.1 Why Point Forecasts Are Insufficient

A point forecast communicates nothing about uncertainty. For decision-making:
- **Inventory management**: need to know P95 of demand to avoid stockouts
- **Energy grid**: need uncertainty to set reserve capacity
- **Finance**: need the full return distribution for risk management

### 12.2 Quantile Regression

Train the model to predict specific quantiles directly by minimizing pinball loss:

```python
import lightgbm as lgb

quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
quantile_models = {}

for q in quantiles:
    model = lgb.LGBMRegressor(
        objective='quantile',
        alpha=q,
        n_estimators=500
    )
    model.fit(X_train, y_train)
    quantile_models[q] = model

# Prediction interval
lower = quantile_models[0.1].predict(X_test)
upper = quantile_models[0.9].predict(X_test)
```

**Quantile crossing**: independently trained quantile models may cross (Q90 < Q50). Fix with isotonic regression or joint quantile regression.

### 12.3 Deep Learning Probabilistic Methods

**MC Dropout**: enable dropout at inference, run `N` forward passes, get distribution.

```python
def mc_dropout_predict(model, x, n_samples=100):
    model.train()  # keep dropout active
    with torch.no_grad():
        samples = torch.stack([model(x) for _ in range(n_samples)])
    return samples.mean(0), samples.std(0)
```

**Deep AR** (Amazon, 2020): LSTM-based model outputs parameters of a distribution (Gaussian, negative binomial) at each step:

```
p(y_t | y_{<t}, x_t) = N(mu_theta(y_{<t}, x_t), sigma_theta(y_{<t}, x_t))
```

Loss = negative log likelihood.

**Normalizing Flows**: model the full conditional distribution by learning a bijective transformation from a simple base distribution.

### 12.4 Conformal Prediction for Time Series

Conformal prediction provides distribution-free prediction intervals with coverage guarantees.

For time series, use **Adaptive Conformal Inference (ACI)**:

```python
# Simplified conformal prediction for time series
def conformal_forecast(model, calibration_residuals, alpha=0.1):
    """
    alpha: desired error rate (0.1 -> 90% coverage)
    """
    q_hat = np.quantile(np.abs(calibration_residuals), 1 - alpha)

    def predict_interval(x):
        y_hat = model.predict(x)
        return y_hat - q_hat, y_hat + q_hat

    return predict_interval
```

**EnbPI** (Ensemble batch prediction intervals): extends conformal prediction to streaming settings where distribution may shift.

Standard conformal prediction guarantees marginal coverage but not conditional. For time series with non-stationarity, ACI adaptively updates the quantile.

### 12.5 Forecast Combination (Ensembling)

Averaging predictions from multiple models typically outperforms any single model:

```python
models = [arima_model, lstm_model, lightgbm_model, prophet_model]
forecasts = np.stack([m.predict(horizon) for m in models])

# Simple average
ensemble = forecasts.mean(axis=0)

# Weighted by inverse validation error
val_errors = np.array([validate(m) for m in models])
weights = (1 / val_errors) / (1 / val_errors).sum()
ensemble_weighted = (forecasts * weights[:, None]).sum(axis=0)
```

The M4 competition winner used a combination of LSTM outputs with Holt-Winters.

---

## 13. Foundation Models for Time Series

### 13.1 Motivation

Classical and task-specific neural models require retraining for each new dataset. Foundation models trained on large corpora of diverse time series promise:
- **Zero-shot forecasting**: no retraining on target data
- **Few-shot adaptation**: minimal fine-tuning
- **Transfer across domains**: finance, energy, healthcare, weather

### 13.2 TimeGPT-1 (Nixtla, 2023)

Transformer trained on 100 billion time points from diverse public datasets. API-based:

```python
from nixtla import NixtlaClient

client = NixtlaClient(api_key='your-key')
forecast = client.forecast(
    df=df,
    h=24,
    freq='H',
    time_col='timestamp',
    target_col='value'
)
```

Supports zero-shot and fine-tuned modes. Shows strong performance on M4 and ETT benchmarks without training.

### 13.3 Lag-Llama (2024)

Llama-style decoder-only transformer adapted for time series. Key design choices:
- **Patch tokenization**: patches of consecutive timesteps as tokens
- **Distributional output**: models future values as probability distributions (Student-t)
- **Trained on**: large Chronos/Lotsa dataset collection

Zero-shot performance competitive with dataset-specific models on many benchmarks.

```python
from lag_llama.gluon.estimator import LagLlamaEstimator

estimator = LagLlamaEstimator(
    ckpt_path="lag-llama.ckpt",
    prediction_length=24,
    context_length=32,
    n_layer=32,
    n_embd_per_head=32,
    n_head=16,
    num_parallel_samples=100
)
```

### 13.4 TimesFM (Google, 2024)

Decoder-only foundation model (200M parameters) trained on 100 billion real-world time points. Uses a patching approach similar to PatchTST but in an autoregressive framework.

Key claims: zero-shot performance better than many task-specific models on standard benchmarks. Supports variable prediction length without retraining.

### 13.5 Moirai (Salesforce, 2024)

"Universal" forecasting model using a masked encoder approach. Designed to handle:
- Variable context lengths
- Variable prediction lengths
- Multiple frequencies
- Any-variate prediction (variable number of variables at inference)

Uses patch-based tokenization with frequency-specific patches.

### 13.6 Chronos (Amazon, 2024)

Converts time series forecasting into a **language modeling task**:
1. Normalize and quantize time series values into discrete tokens
2. Train a T5-style encoder-decoder on sequences of tokens
3. Decode future tokens autoregressively and map back to values

```python
import torch
from chronos import ChronosPipeline

pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-base",
    device_map="cuda",
    torch_dtype=torch.bfloat16
)

forecast = pipeline.predict(
    context=torch.tensor(series[-512:]),
    prediction_length=24
)
# forecast shape: (num_samples, prediction_length)
```

### 13.7 When to Use Foundation Models

| Situation | Recommendation |
|---|---|
| Cold start, no training data | Foundation model zero-shot |
| Small dataset (<1k observations) | Foundation model + light fine-tuning |
| Large dataset, stable distribution | Task-specific model (N-HiTS, PatchTST) |
| Real-time, low latency | Lightweight model (LightGBM + features) |
| Multiple series, shared patterns | Foundation model or global model |

---

## 14. Production Patterns

### 14.1 Concept Drift

**Concept drift**: the statistical relationship between features and target changes over time.

Types:
- **Sudden drift**: e.g., COVID lockdowns changing energy demand patterns overnight
- **Gradual drift**: e.g., customer behavior shifting over months
- **Recurring drift**: e.g., seasonal patterns returning each year (this is expected and handled by seasonal features)

**Detection methods:**

Page-Hinkley test: cumulative sum over normalized residuals:
```
m_t = sum_{i=1}^t (x_i - mu_0 - delta)
M_T = max(m_1, ..., m_T)
drift detected if m_T - M_T > lambda
```

ADWIN (Adaptive Windowing): maintains a sliding window and detects drift when two sub-windows have significantly different means.

```python
from river.drift import ADWIN

detector = ADWIN()
for error in model_errors:
    detector.update(error)
    if detector.drift_detected:
        print("Drift detected — trigger retraining")
```

### 14.2 Retraining Triggers

| Trigger Type | Description | Use Case |
|---|---|---|
| Scheduled | Retrain every N days/weeks | Stable series, low operational cost |
| Performance-based | Retrain when val error > threshold | When monitoring is available |
| Drift-based | Retrain on drift detection | When distribution shifts detectably |
| Continuous | Retrain on each new observation (online learning) | High-frequency, fast-changing |

**Model decay rate** varies by domain:
- E-commerce demand: days to weeks
- Energy demand: weeks to months (slow drift)
- Financial markets: hours to days

### 14.3 Data Pipeline for Forecasting

```
Raw data sources
       |
   Ingestion + validation (Great Expectations, Pandera)
       |
   Feature engineering (lag, rolling, calendar, Fourier)
       |
   Train / validate / test split (temporal)
       |
   Model training
       |
   Model registry (MLflow, W&B)
       |
   Serving layer (batch / online)
       |
   Monitoring (prediction error tracking, drift detection)
       |
   Retraining trigger
```

### 14.4 Common Production Pitfalls

1. **Leakage in preprocessing**: StandardScaler fit on full dataset before split — use only training data to fit scalers
2. **Lookahead in lag features**: computing `shift(0)` or negative shifts
3. **Missing value handling**: forward-fill creates artificial autocorrelation; interpolation can leak future
4. **Horizon mismatch**: evaluating 1-step-ahead but deploying as 24-step-ahead
5. **Evaluation on too-short test period**: seasonal patterns need at least 2-3 full cycles in test set
6. **Ignoring latency**: a model needing 512 lookback steps may not be feasible for real-time inference

### 14.5 Global vs Local Models

**Local model**: one model per series. Classic ARIMA, Prophet. Works well for a small number of important series. Does not scale to thousands.

**Global model**: one model across all series. LSTM with series identifier, LightGBM, N-BEATS. Can learn shared patterns. Requires careful normalization (instance normalization or global z-score).

```python
# Instance normalization for global models
def normalize_per_series(batch):
    # batch: (batch_size, seq_len)
    mean = batch.mean(dim=1, keepdim=True)
    std  = batch.std(dim=1, keepdim=True) + 1e-8
    return (batch - mean) / std, mean, std

def denormalize(normalized, mean, std):
    return normalized * std + mean
```

**RevIN (Reversible Instance Normalization)**: normalization applied at the encoder input and reversed at the decoder output. Used in PatchTST and N-HiTS.

---

## 15. Common Interview Questions

---

**Q: What is the difference between ARIMA and SARIMA?**

ARIMA(p,d,q) models non-seasonal time series using autoregressive terms (past values), differencing (to achieve stationarity), and moving average terms (past errors). SARIMA(p,d,q)(P,D,Q)[S] adds seasonal counterparts operating at lag S: seasonal AR, seasonal differencing, and seasonal MA. If you have monthly data with clear annual cycles, you need SARIMA with S=12. ARIMA applied to seasonal data without seasonal differencing will show residual autocorrelation at seasonal lags.

---

**Q: Why can't you use standard k-fold cross-validation for time series?**

k-fold randomly shuffles data into folds, breaking temporal order. This causes two problems: (1) the model trains on data from period T+k to predict period T, using future information that would not be available in production; (2) observations close in time are correlated, so random splits bleed information between train and validation sets. Use TimeSeriesSplit or walk-forward validation instead.

---

**Q: How do LSTMs handle the vanishing gradient problem compared to vanilla RNNs?**

Vanilla RNNs backpropagate gradients through time by multiplying the same weight matrix at each step. With sigmoid-like activations and weights < 1, gradients shrink exponentially. LSTMs introduce a cell state updated additively: `c_t = f_t * c_{t-1} + i_t * g_t`. The additive update means the gradient path through the cell state does not involve repeated multiplication by the same weights, allowing gradients to flow unchanged over many timesteps. The forget gate can still zero out old memories, but that is a controlled, learned operation.

---

**Q: What is teacher forcing and when is it a problem?**

Teacher forcing feeds the ground-truth output as decoder input at each step during training, rather than the model's own prediction. This speeds up training because errors do not compound. The problem is **exposure bias**: at inference, the model never saw its own mistakes during training, so small prediction errors early in the sequence get amplified by the model not knowing how to recover. Mitigation: scheduled sampling (gradually reduce teacher forcing ratio during training), or use non-autoregressive decoders like N-BEATS or Informer's generative decoder.

---

**Q: When would you choose a Transformer over an LSTM for time series?**

Transformers are preferred when: (1) long-range dependencies matter and the sequence is long (>100 steps) — LSTMs struggle to retain information over many steps; (2) parallelism is important — Transformers train faster on GPUs; (3) enough data exists — Transformers have more parameters and are more data-hungry. LSTMs are preferred when: (1) data is limited; (2) online/streaming inference is required (Transformers need the full context window); (3) the sequence has clear local structure without long-range dependencies.

---

**Q: Explain the difference between additive and multiplicative decomposition.**

In additive decomposition: `y_t = trend_t + seasonal_t + residual_t`. The seasonal amplitude is constant regardless of the series level. Suitable when the series fluctuates by a fixed amount each period.

In multiplicative decomposition: `y_t = trend_t * seasonal_t * residual_t`. The seasonal amplitude grows proportionally with the level. Suitable for series like revenue or population where percentage swings are more meaningful than absolute ones. Multiplicative can be transformed to additive by taking log: `log(y_t) = log(trend_t) + log(seasonal_t) + log(residual_t)`.

---

**Q: What is MASE and why is it preferred over MAPE?**

MASE scales the forecast error by the MAE of a naive seasonal benchmark: `MASE = MAE / MAE_naive`. Values < 1 mean you beat the naive baseline; values > 1 mean you do not. MAPE is undefined when actual values are zero and is asymmetric (overforecasts and underforecasts of the same absolute size have different MAPE values). MASE avoids both problems and is scale-free, making it valid for comparing across series with different units and magnitudes.

---

**Q: How would you detect and handle concept drift in a production forecasting system?**

Detection: monitor rolling forecast error (e.g., 7-day MAE). Use statistical tests like ADWIN or Page-Hinkley on residuals to detect distribution changes. Compare incoming feature distributions to training distributions using PSI (Population Stability Index) or KL divergence.

Handling: (1) Scheduled retraining on a rolling window of recent data; (2) Drift-triggered retraining when tests fire; (3) Online learning methods (e.g., SGD with decaying learning rate) that continuously update; (4) Ensemble with models trained on different historical windows, weighted by recent performance.

---

**Q: What are the advantages of N-BEATS over LSTM-based models?**

N-BEATS is entirely MLP-based: no recurrence, no attention. This gives: (1) fully parallel computation — all inputs processed simultaneously; (2) interpretability in the interpretable variant — distinct trend and seasonality stacks with explicit basis functions; (3) competitive performance on M4/M5 benchmarks despite architectural simplicity; (4) no sequence-length limitation from vanishing gradients. The backcast mechanism also provides a natural way to decompose the forecast into additive components, which aids model debugging and domain expert review.

---

**Q: How do you approach forecasting with multiple seasonalities?**

Options: (1) **TBATS**: handles multiple seasonal periods with trigonometric Fourier terms; (2) **Prophet**: supports multiple additive Fourier seasonality components with different periods; (3) **Fourier features**: encode each seasonality as a set of sin/cos terms at the appropriate period, then use any regression or neural model; (4) **Deep learning**: LSTM or TCN that can learn multiple periodicities from data if the lookback window is large enough; (5) **STL decomposition**: decompose one seasonality at a time and model the residual separately.

---

**Q: What is conformal prediction and why is it useful for time series?**

Conformal prediction is a framework for constructing prediction intervals with a marginal coverage guarantee: if you request a 90% interval, the true value falls inside it at least 90% of the time, regardless of the model's distributional assumptions. For time series, the key challenge is non-exchangeability (observations are not i.i.d.). Adaptive Conformal Inference (ACI) addresses this by adapting the coverage level over time as coverage errors are observed. Unlike Bayesian intervals, conformal prediction requires no distributional assumptions about residuals.

---
