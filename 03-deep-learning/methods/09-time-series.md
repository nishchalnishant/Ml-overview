---
module: Deep Learning
topic: Methods
subtopic: Time Series
status: unread
tags: [deeplearning, ml, methods-time-series]
---
# Time Series Machine Learning

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

### 1.1 Temporal Ordering

**The problem**: shuffling the rows of a tabular dataset changes nothing. Shuffling a time series destroys it. Standard ML assumptions — i.i.d. samples, random train/test splits, feature statistics computed over the full dataset — all produce silent, hard-to-detect errors when applied to temporal data.

**The core insight**: every observation in a time series is a consequence of what came before. The generative process is `P(y_t | y_{t-1}, y_{t-2}, ..., y_1)`, not `P(y_t)`. Violate this and you get three failure modes:

- **Data leakage**: training on 2023 data to predict 2022. The model learns a perfectly calibrated cheat.
- **Inflated confidence intervals**: assuming independence when adjacent observations are correlated inflates effective sample size.
- **Leaked features**: computing rolling means on the full dataset before splitting injects future signal into past features.

**What breaks first**: computing `df['rolling_mean'] = df['target'].rolling(7).mean()` before splitting — the last 6 timesteps before the split boundary look into the test set.

---

### 1.2 Trend

**The problem**: a model trained on energy demand from 2000-2010 extrapolates a flat mean. But the series has been climbing by ~5 GW per year. The model systematically underforecasts.

**The core insight**: the mean of the series is itself a function of time. The non-stationarity must be removed or modeled before fitting any method that assumes a fixed mean.

**The mechanics**:
- **Additive trend**: `y_t = α + βt + ε_t`. Fluctuations have constant magnitude.
- **Multiplicative trend**: `y_t = α · e^{βt} · ε_t`. Fluctuations scale with level. Apply `log(y_t)` to convert to additive.
- First-order differencing `Δy_t = y_t − y_{t-1}` removes a linear trend.

**What breaks**: differencing removes trend but inflates model complexity when the trend is smooth. Fitting a linear trend explicitly and modeling residuals is often cleaner.

---

### 1.3 Seasonality

**The problem**: a model trained on retail sales data predicts a flat weekday level. It ignores that Sundays are 40% lower and December is 3× the monthly average. Point forecasts are consistently wrong in the same direction at the same calendar positions.

**The core insight**: the series has a repeating structure at a known period. Encode that period explicitly as a feature or model it structurally, or the model must re-discover it from scratch — which requires far more data.

**The mechanics**:

| Domain | Period | Frequency |
|---|---|---|
| Retail sales | Weekly | 7 |
| Energy demand | Daily | 24 (hourly) |
| Stock volume | Yearly | 252 (trading days) |
| Server traffic | Weekly + Daily | Multiple |

Multiple seasonality (daily cycle nested inside weekly) is common and requires either Fourier features or a method that explicitly handles it (TBATS, Prophet).

**What breaks**: single-period methods (SARIMA with one S) fail on multiple seasonalities. Treating time-of-day as a raw integer fails because the model does not know that hour 23 wraps to hour 0.

---

### 1.4 Stationarity

**The problem**: classical forecasting methods — ARIMA, exponential smoothing — are derived under the assumption that `E[y_t]`, `Var(y_t)`, and `Cov(y_t, y_{t+k})` do not change over time. Apply them to a trending or variance-expanding series and parameter estimates are biased.

**The core insight**: make the series stationary before fitting. The transformation is not just preprocessing — it changes which model class is valid.

**The mechanics**:
- **Augmented Dickey-Fuller (ADF)**: null = unit root (non-stationary). Reject at p < 0.05.
- **KPSS**: null = stationary. Both tests together disambiguate.

```python
from statsmodels.tsa.stattools import adfuller

result = adfuller(series)
print(f"ADF Statistic: {result[0]:.4f}")
print(f"p-value: {result[1]:.4f}")
# p < 0.05 -> reject unit root -> likely stationary
```

Transformations to achieve stationarity:
- First differencing `Δy_t = y_t − y_{t-1}`: removes linear trend
- Seasonal differencing `y_t − y_{t-s}`: removes seasonal structure
- Log transform: stabilizes exponentially growing variance
- Box-Cox: general variance stabilization

**What breaks**: over-differencing introduces unnecessary negative autocorrelation and makes a simple series harder to model. One ADF test is not enough — use both ADF and KPSS, and inspect ACF/PACF before deciding.

---

### 1.5 Autocorrelation

**The problem**: in i.i.d. data, yesterday's value tells you nothing about today's. In a time series, it tells you almost everything. Without measuring this dependence explicitly, you cannot choose the right model order.

**The core insight**: the ACF and PACF plots are a direct diagnostic for model selection. They answer: does this series have AR structure, MA structure, or both?

- **ACF (Autocorrelation Function)**: correlation between `y_t` and `y_{t-k}` for each lag `k`
- **PACF (Partial Autocorrelation Function)**: correlation after removing effects of intermediate lags

**Reading the plots**:
- PACF cuts off sharply at lag `p`, ACF decays slowly → AR(p) process
- ACF cuts off at lag `q`, PACF decays slowly → MA(q) process
- Both decay slowly → ARMA(p, q) or need differencing

```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
plot_acf(series, lags=40, ax=ax1)
plot_pacf(series, lags=40, ax=ax2)
plt.tight_layout()
plt.show()
```

**What breaks**: ignoring ACF/PACF and blindly fitting ARIMA(1,1,1) on all series. If the MA(1) coefficient is near zero, you added unnecessary complexity. If you needed MA(3) and fitted MA(1), residuals will still be autocorrelated.

---

## 2. Classical Methods

### 2.1 ARIMA

**The problem**: you want to predict next month's sales. The series is trending upward and shows residual autocorrelation — today's value depends on yesterday's error, not just yesterday's level. No single AR model captures both.

**The core insight**: decompose the prediction problem into three orthogonal parts — trend removal (differencing), autoregression on past values, and autoregression on past errors. ARIMA(p, d, q) handles all three.

**The mechanics**:
- `d`: differencing order to achieve stationarity (read from ADF)
- `p`: AR terms — PACF cuts off at lag `p`
- `q`: MA terms — ACF cuts off at lag `q`

```
Δ^d y_t = c + φ_1 Δ^d y_{t-1} + ... + φ_p Δ^d y_{t-p}
          + θ_1 ε_{t-1} + ... + θ_q ε_{t-q} + ε_t
```

```python
from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(train, order=(2, 1, 2))
result = model.fit()
print(result.summary())

forecast = result.forecast(steps=30)
```

Auto-ARIMA (pmdarima) searches orders automatically using AIC:

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

**What breaks**: ARIMA assumes Gaussian errors and linear dependence. It cannot model nonlinear patterns, multiple seasonalities, or structural breaks. Residual autocorrelation in fitted residuals signals under-specified order — check the Ljung-Box test.

---

### 2.2 SARIMA

**The problem**: ARIMA removes a linear trend and models short-memory autocorrelation. But a monthly electricity series shows the same spike every January, every year — a dependency at lag 12 that ARIMA(p,d,q) terms at lag 1-3 cannot reach.

**The core insight**: add a second set of AR/I/MA terms that operate at the seasonal lag S instead of lag 1. SARIMA(p,d,q)(P,D,Q)[S] is two nested ARIMA models: one for short-term dynamics, one for seasonal dynamics.

**The mechanics**:

```
SARIMA(1,1,1)(1,1,1)[12]  -- monthly data with annual seasonality
```

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

**What breaks**: SARIMA handles exactly one seasonal period. For hourly data with both daily (S=24) and weekly (S=168) cycles, SARIMA fails. Options: TBATS, Fourier features + ARIMAX, or neural models.

---

### 2.3 Exponential Smoothing

**The problem**: ARIMA fits a full statistical model including MA terms that require maximum likelihood estimation. For many practical forecasting tasks — inventory replenishment, budget projections — you want a simpler, interpretable algorithm that gives more weight to recent observations and less to older ones.

**The core insight**: weight past observations by an exponentially decaying factor. The prediction at time `t` is a weighted average of all past values, but the weights decay geometrically so yesterday matters more than last year.

**The mechanics**:

**Simple Exponential Smoothing (SES)** — no trend, no seasonality:
```
s_t = α · y_t + (1 − α) · s_{t-1}
ŷ_{t+1} = s_t
```
`α` ∈ (0,1): high α = more weight on recent observations.

**Holt's Linear Method** — handles trend:
```
Level:    l_t = α · y_t + (1−α) · (l_{t-1} + b_{t-1})
Trend:    b_t = β · (l_t − l_{t-1}) + (1−β) · b_{t-1}
Forecast: ŷ_{t+h} = l_t + h · b_t
```

**Holt-Winters (Triple Exponential Smoothing)** — handles trend + seasonality (additive):
```
Level:    l_t = α · (y_t − s_{t-m}) + (1−α) · (l_{t-1} + b_{t-1})
Trend:    b_t = β · (l_t − l_{t-1}) + (1−β) · b_{t-1}
Seasonal: s_t = γ · (y_t − l_{t-1} − b_{t-1}) + (1−γ) · s_{t-m}
Forecast: ŷ_{t+h} = l_t + h · b_t + s_{t-m+h}
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

Use `seasonal='mul'` when seasonal amplitude grows with the level (e.g., e-commerce sales growing year over year).

**What breaks**: exponential smoothing is strictly univariate. It cannot incorporate exogenous variables (weather, promotions). The forecast horizon is limited — long-horizon forecasts revert to the trend without accounting for new information.

---

### 2.4 Prophet

**The problem**: a data analyst needs to forecast daily website traffic for the next 90 days. The series has a weekly dip on weekends, a dip during holidays, a gradual upward trend with occasional sudden shifts (product launches), and some missing days. ARIMA requires a clean, regularly-spaced, stationary series and manual order selection.

**The core insight**: treat forecasting as a curve-fitting problem with interpretable components. Decompose the series explicitly into trend, seasonality, and holidays. Let domain experts tune each component directly through parameters they understand.

**The mechanics**:
```
y(t) = trend(t) + seasonality(t) + holidays(t) + ε_t
```
- **Trend**: piecewise linear or logistic growth with automatic changepoint detection
- **Seasonality**: Fourier series approximation at yearly, weekly, and daily periods
- **Holidays**: user-supplied date ranges with learned offsets

```python
from prophet import Prophet
import pandas as pd

df = pd.DataFrame({'ds': dates, 'y': values})

model = Prophet(
    changepoint_prior_scale=0.05,  # flexibility of trend (higher = more changepoints)
    seasonality_prior_scale=10,
    yearly_seasonality=True,
    weekly_seasonality=True
)
model.fit(df)

future = model.make_future_dataframe(periods=90)
forecast = model.predict(future)
```

**What breaks**: Prophet's flexibility makes it easy to overfit on changepoints. The default `changepoint_prior_scale=0.05` is conservative — too small and it misses genuine shifts, too large and it fits noise. It is not a forecasting engine for very short-horizon or high-frequency (sub-hourly) data.

---

## 3. Feature Engineering for Time Series

**The problem**: gradient boosted trees (LightGBM, XGBoost) are competitive or superior to neural sequence models on many time series benchmarks — but they operate on fixed-size feature vectors, not sequences. How do you represent temporal history as a feature matrix?

**The core insight**: the model cannot see the sequence directly. You must manufacture the temporal context explicitly as lag features, rolling statistics, and calendar encodings. The quality of these features determines the ceiling of the model's performance.

---

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

**Critical**: only shift forward (positive lags) to avoid leakage. After shift, drop rows with NaN from the start.

**What breaks**: including `shift(0)` or negative lags. After a train/test split, recomputing lag features on the combined dataframe leaks test values into training lag columns near the boundary.

---

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

Expanding window (cumulative statistics):
```python
df['cumulative_mean'] = df['target'].shift(1).expanding().mean()
```

**What breaks**: omitting `shift(1)` before `rolling()`. The rolling window at time `t` includes `y_t` itself, which is the target — direct leakage.

---

### 3.3 Calendar Features

**The problem**: the model needs to know it is Monday, or December, or a holiday. A raw timestamp integer does not encode this. The model cannot learn that "column 1,456,800 = Monday" without help.

**The core insight**: extract calendar features explicitly. Use cyclical encoding for periodic features — raw integers make 23:00 and 00:00 far apart when they should be adjacent.

```python
def add_calendar_features(df, datetime_col):
    dt = pd.to_datetime(df[datetime_col])
    df['hour']         = dt.dt.hour
    df['day_of_week']  = dt.dt.dayofweek
    df['day_of_month'] = dt.dt.day
    df['week_of_year'] = dt.dt.isocalendar().week.astype(int)
    df['month']        = dt.dt.month
    df['quarter']      = dt.dt.quarter
    df['year']         = dt.dt.year
    df['is_weekend']   = (dt.dt.dayofweek >= 5).astype(int)
    return df

# Cyclical encoding: hour 23 should be close to hour 0
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
```

**What breaks**: treating `hour` as a raw integer in a tree model. Trees can learn the 0-23 cutoff but need many extra splits. Neural models with raw integers learn embeddings but cyclical encoding is free and guaranteed.

---

### 3.4 Fourier Features

**The problem**: Holt-Winters handles one seasonal period. Your series has a 24-hour daily cycle AND a 7-day weekly cycle. You need both represented simultaneously without fitting two separate seasonal models.

**The core insight**: any periodic function can be approximated as a sum of sinusoids (Fourier series). Encode each seasonality as a set of sin/cos pairs at harmonics of its period. Feed them as regular features into any model.

```python
def fourier_features(t, period, n_terms):
    features = {}
    for k in range(1, n_terms + 1):
        features[f'sin_{period}_{k}'] = np.sin(2 * np.pi * k * t / period)
        features[f'cos_{period}_{k}'] = np.cos(2 * np.pi * k * t / period)
    return pd.DataFrame(features)

# Daily seasonality with 5 harmonics (for hourly data)
t = np.arange(len(df))
fourier_daily  = fourier_features(t, period=24, n_terms=5)
fourier_weekly = fourier_features(t, period=168, n_terms=3)
df = pd.concat([df, fourier_daily, fourier_weekly], axis=1)
```

More harmonics = sharper seasonality shape captured, but more parameters.

**What breaks**: too many harmonics with too little data — overfitting. Too few harmonics misses sharp seasonal peaks (e.g., the exact hour of maximum demand).

---

### 3.5 External Features

- **Holidays**: binary or categorical flag — critical for retail and energy
- **Weather**: temperature is the dominant predictor for energy demand
- **Known future values**: promotions, scheduled events (these are *exogenous* variables)

```python
# SARIMAX handles exogenous variables
model = SARIMAX(train_demand,
                exog=train_temperature,
                order=(1,1,1),
                seasonal_order=(1,1,1,24))
result = model.fit()
forecast = result.get_forecast(steps=24, exog=future_temperature)
```

**What breaks**: using exogenous variables that are themselves forecasts (e.g., forecast temperature as a feature). You have now chained two uncertain forecasts. The stated accuracy of the combined system will be optimistic if you evaluate with ground-truth covariates.

---

## 4. RNNs and LSTMs

### 4.1 Vanilla RNN

**The problem**: a fully connected network sees one timestep at a time and has no memory. If you concatenate the last `k` timesteps as a feature vector, you must fix `k` at architecture design time and miss anything outside that window. The model has no principled mechanism for varying how far back it looks.

**The core insight**: maintain a hidden state that accumulates information across timesteps. The same function is applied at each step, making the architecture independent of sequence length.

```
h_t = tanh(W_hh · h_{t-1} + W_xh · x_t + b_h)
y_t = W_hy · h_t + b_y
```

**What breaks**: gradients flow back through `W_hh` at every timestep. If `W_hh` has singular values < 1, multiplying through 100 timesteps produces gradients ~0 (vanishing). If > 1, gradients explode. In practice, RNNs cannot retain information beyond ~10-20 timesteps.

---

### 4.2 LSTM Architecture

**The problem**: vanilla RNNs cannot decide what to remember and what to forget. The hidden state is overwritten at every step. A fact learned 100 timesteps ago is either diluted or has vanished from the hidden state entirely.

**The core insight**: separate short-term hidden state from long-term cell state. The cell state flows through time with additive updates (not multiplicative), so gradients can pass through without attenuation. Learned gates decide what to write, what to erase, and what to output.

**The mechanics**:
```
Forget gate:  f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
Input gate:   i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
Cell update:  g_t = tanh(W_g · [h_{t-1}, x_t] + b_g)
Output gate:  o_t = σ(W_o · [h_{t-1}, x_t] + b_o)

Cell state:   c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t    ← additive update
Hidden state: h_t = o_t ⊙ tanh(c_t)
```

The additive cell state update `c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t` is the key: the gradient path through `c_t` does not repeatedly multiply through the same weight matrix, allowing gradients to flow across hundreds of timesteps.

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
        out = self.fc(lstm_out[:, -1, :])  # (batch, output_size)
        return out

model = LSTMForecaster(input_size=10, hidden_size=64, num_layers=2, output_size=1)
```

**GRU**: simpler than LSTM with only two gates (reset and update). Merges cell and hidden state. Often comparable performance with fewer parameters.

**What breaks**: LSTMs are sequential — each step depends on the previous hidden state. This prevents parallelization across the time dimension. Training on long sequences (>1000 steps) is slow. The hidden state is still a fixed-size bottleneck.

---

### 4.3 Sequence-to-Sequence for Multi-step Forecasting

**The problem**: you need a forecast for the next 24 hours. Predicting one step and feeding it back 24 times compounds every error — a small error at step 1 shifts the context for step 2, and so on. By step 24, you are predicting from a distribution you never saw in training.

**The core insight**: an encoder-decoder architecture separates the compression of history from the generation of the forecast horizon. The decoder generates all `H` steps, conditioned on the encoder's summary of the input.

```python
class Seq2SeqLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, horizon):
        super().__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(1, hidden_size, batch_first=True)
        self.output  = nn.Linear(hidden_size, 1)
        self.horizon = horizon

    def forward(self, src, tgt=None, teacher_forcing_ratio=0.5):
        _, (h, c) = self.encoder(src)

        outputs = []
        dec_input = src[:, -1:, :1]  # last observed value

        for t in range(self.horizon):
            out, (h, c) = self.decoder(dec_input, (h, c))
            pred = self.output(out)
            outputs.append(pred)

            if tgt is not None and torch.rand(1) < teacher_forcing_ratio:
                dec_input = tgt[:, t:t+1, :]
            else:
                dec_input = pred

        return torch.cat(outputs, dim=1)
```

**Teacher forcing**: during training, feed ground-truth decoder input with probability `p` instead of the model's own prediction. Speeds training but creates **exposure bias** — at inference, the model encounters its own (imperfect) predictions as inputs, which it never saw during training. Schedule the ratio downward during training.

**What breaks**: exposure bias is not a small effect. Models trained with 100% teacher forcing often produce unstable rollouts at inference because small prediction errors shift the decoder input distribution out of distribution.

---

### 4.4 Practical LSTM Tips

- **Input normalization**: scale features to zero mean, unit variance using training statistics only
- **Stacking**: 2-3 layers usually sufficient; more causes overfitting
- **Lookback window**: start with `2×` the longest seasonal period; tune via validation
- **Gradient clipping**: clip gradients to norm 1.0 (`torch.nn.utils.clip_grad_norm_`)
- **Dropout**: between LSTM layers, not within (use `nn.Dropout` after LSTM)

---

## 5. Temporal Convolutional Networks (TCN)

### 5.1 Causal Convolution

**The problem**: LSTMs are sequential — step `t` must wait for step `t-1` before computing. On a GPU with 10,000 parallel cores, this serialization is wasteful. Training is slow and the long gradient paths cause instability.

**The core insight**: convolutions are parallel. If you constrain the convolution to be *causal* — output at time `t` only depends on inputs at `t` and earlier — you get both parallelism and temporal validity.

```
y_t = Σ_{k=0}^{K-1} w_k · x_{t-k}
```

Pad the left by `K-1` zeros to achieve causality.

**What breaks**: a causal convolution with kernel size `K` can only see `K` steps back. To model long-range dependencies, you need either many layers or large kernels — both expensive.

---

### 5.2 Dilated Convolution

**The problem**: to see 1000 steps back with a causal convolution, you either need 1000 layers or a kernel of size 1000. Both are impractical.

**The core insight**: skip steps. Apply the kernel with spacing (dilation) `d`. With exponentially increasing dilation `d = 1, 2, 4, 8, ...`, each layer doubles the receptive field. After `L` layers with kernel size 2, you see `2^L` steps back.

```
y_t = Σ_{k=0}^{K-1} w_k · x_{t − d·k}
```

With kernel size 2 and dilations 1, 2, 4, 8:
- Layer 1 (d=1): sees 2 steps back
- Layer 2 (d=2): sees 4 steps back
- Layer 3 (d=4): sees 8 steps back
- Layer 4 (d=8): sees 16 steps back

Receptive field = `1 + (K−1) · Σ(dilations)`.

---

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
        res = x if self.skip is None else self.skip(x)
        x = self.relu(self.norm1(self.conv1(x).transpose(1,2)).transpose(1,2))
        x = self.drop(x)
        x = self.relu(self.norm2(self.conv2(x).transpose(1,2)).transpose(1,2))
        x = self.drop(x)
        return self.relu(x + res)
```

**What breaks**: the receptive field is fixed at architecture design time. If a dependency exists outside the receptive field, the model cannot see it. LSTMs are more flexible here — in principle, they can attend to any past step (though in practice they forget). TCNs also work less naturally for online/streaming inference, where you update state one step at a time.

---

## 6. Transformer-Based Models

### 6.1 Vanilla Transformer Baseline

**The problem**: LSTMs see each step sequentially. Transformers compute attention over all pairs simultaneously, giving direct access to any past timestep. But full self-attention is O(L²) in memory and time. For a 1000-timestep series, that is 1 million attention pairs per head per layer — feasible for NLP with 512-token sentences, not for time series with thousands of observations.

**The core insight**: full attention is too expensive for time series, but the O(L²) cost can be reduced with sparse, patched, or inverted attention.

---

### 6.2 Informer (2021)

**The problem**: standard attention at O(L²) is infeasible for forecasting 720 steps from 1440 context steps.

**The core insight**: most attention weights are near zero. Only a small fraction of queries have high attention entropy — they are the informative ones. Compute full attention only for those top-u queries; skip the rest.

**The mechanics**: ProbSparse attention. For each query, estimate its importance by comparing its max attention score to the average. Select the top `u = O(L log L)` queries. Complexity drops from O(L²) to O(L log L).

Also introduces:
- **Distilling**: halve sequence length between encoder layers via max pooling
- **Generative decoder**: predicts the full forecast horizon in one forward pass (not autoregressive)

```python
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

**What breaks**: ProbSparse attention is an approximation. On short sequences where full attention is feasible, simpler models often outperform Informer. Empirical benchmarks (Zeng et al., 2023) showed a simple linear model beats Informer on several ETT datasets.

---

### 6.3 Autoformer (2021)

**The problem**: standard attention treats every token independently. But time series have periodic structure — a strong correlation between `y_t` and `y_{t-P}` for seasonal period `P`. Token-level attention must re-discover this periodicity from scratch.

**The core insight**: replace attention with auto-correlation. Compute period-based dependencies using the Fast Fourier Transform, which naturally captures periodic structure. Decompose trend and seasonality inside every encoder/decoder layer.

```
Correlation(Q, K) = IFFT(FFT(Q) · conj(FFT(K)))
```

**What breaks**: the inductive bias toward periodicity is a strength on periodic series and a weakness on irregular or aperiodic series. If the series lacks clear seasonality, the auto-correlation mechanism provides no advantage.

---

### 6.4 PatchTST (2023)

**The problem**: individual timestep tokens carry little information — a single value at time `t` has no local context. And a long sequence of 500 single-value tokens means 250,000 attention pairs.

**The core insight**: patch the series. Divide the sequence into overlapping windows of size `P`. Each patch becomes one token carrying `P` consecutive timesteps. Reduces token count by factor `P`, and each token now has local temporal context baked in. Apply standard full attention on the shorter sequence.

```
Sequence of length L → divide into patches of size P → L/P tokens
```

```python
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

PatchTST also uses channel-independence: each variable is processed separately. This acts as regularization — the model cannot overfit to spurious cross-variable correlations.

**What breaks**: channel-independence ignores cross-variable correlations entirely. When variables have known, strong causal relationships (e.g., temperature and humidity), this is a handicap.

---

### 6.5 iTransformer (2024)

**The problem**: in a multivariate series, standard Transformers apply attention across time for each variable. But sometimes the most useful signal is across variables at the same time — knowing that sensor A spiked tells you sensor B will spike 5 minutes later. Temporal attention does not capture this.

**The core insight**: invert the attention axis. Let each token represent one variable's entire time series. Apply attention across variables to capture cross-variable interactions. The feed-forward layer handles temporal dynamics per variable.

| | Token | Attention learns |
|---|---|---|
| Standard Transformer | one timestep | temporal dependencies |
| iTransformer | one variable's full series | cross-variable correlations |

**What breaks**: iTransformer's effectiveness depends on whether cross-variable dependencies are the dominant signal. For univariate series or weakly correlated variables, inverting attention provides no benefit.

---

### 6.6 Transformer Comparison

| Model | Complexity | Key Innovation | Best For |
|---|---|---|---|
| Informer | O(L log L) | ProbSparse attention | Long sequences |
| Autoformer | O(L log L) | Auto-Correlation + decomposition | Periodic series |
| PatchTST | O((L/P)²) | Patch tokenization + channel-independence | General forecasting |
| iTransformer | O(M² · L) | Inverted attention across variables | Multivariate |

---

## 7. N-BEATS and N-HiTS

### 7.1 N-BEATS (2020)

**The problem**: RNNs and Transformers are designed for NLP and adapted for time series. The architectural inductive biases (token embeddings, attention masks, cell states) are not natural for numerical sequences. Can a purpose-built architecture, with no recurrence and no attention, beat RNN/Transformer baselines?

**The core insight**: stack MLP blocks where each block explains part of the series. Every block produces a *backcast* (reconstruction of its input) and a *forecast* (contribution to the prediction). The next block receives only the residual — what the previous block could not explain. This is an additive decomposition learned end-to-end.

**The mechanics**:
```
Block k:
  θ_b, θ_f = MLP(x_residual_k)
  backcast_k  = basis_b(θ_b)
  forecast_k  = basis_f(θ_f)
  x_residual_{k+1} = x_residual_k − backcast_k

Total forecast = Σ forecast_k
```

Interpretable variant uses explicit basis functions:
- **Trend stack**: polynomial basis `[1, t, t², t³]`
- **Seasonality stack**: Fourier basis `[cos(2πkt/P), sin(2πkt/P)]`

```python
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS

model = NBEATS(
    h=24,
    input_size=72,
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

**What breaks**: N-BEATS is univariate by default. The generic (non-interpretable) variant has no explicit basis functions and is harder to debug. Like any deep MLP, it is data-hungry relative to classical methods.

---

### 7.2 N-HiTS (2022)

**The problem**: N-BEATS blocks all operate at the same temporal resolution. But a 720-step forecast is dominated by low-frequency trends and seasonality — high-frequency noise contributes little. Allocating equal capacity to all frequencies wastes parameters.

**The core insight**: multi-rate sampling. Different blocks pool the input at different rates. Coarse blocks see the whole history at low resolution and capture long-range trends. Fine blocks see the recent past at full resolution and capture short-range dynamics. The forecast at each scale is interpolated back to the target resolution.

```
Stack 1 (coarse): pool_kernel=8 → captures long-range trend
Stack 2 (medium): pool_kernel=4 → captures seasonal patterns
Stack 3 (fine):   pool_kernel=1 → captures short-term variation
```

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

**N-BEATS vs N-HiTS vs Transformers**: on M4 and M5 benchmarks, N-BEATS and N-HiTS frequently match or outperform Transformer variants while training faster and using fewer parameters. The lesson: architectural simplicity beats complexity when inductive biases are well-matched to the task.

**What breaks**: N-HiTS's multi-rate design assumes low-frequency components dominate, which is true for most economic and demand series but not for all signals. On series with dominant high-frequency content, the coarse stacks waste capacity.

---

## 8. Anomaly Detection

### 8.1 Statistical Tests

**The problem**: you want to flag observations that are unusual. But "unusual" relative to what? The mean of the full series is wrong for a trending or seasonal series — a value perfectly normal in December looks anomalous if you compare it to a July baseline.

**The core insight**: compare each observation to its local expected value — the mean of a recent rolling window, or a seasonal decomposition residual.

**Z-score with rolling window**:
```python
def zscore_anomalies(series, window=30, threshold=3.0):
    rolling_mean = series.rolling(window).mean()
    rolling_std  = series.rolling(window).std()
    z_scores = (series - rolling_mean) / rolling_std
    return z_scores.abs() > threshold
```

**Seasonal Hybrid ESD (S-H-ESD)**: Twitter's method. Decompose seasonality first, then apply the Extreme Studentized Deviate test on residuals. Used in production monitoring.

**What breaks**: Z-score with a fixed threshold fails when the noise variance changes over time. A 3σ threshold calibrated on calm periods misses anomalies during volatile periods.

---

### 8.2 Isolation Forest

**The problem**: you have 20 sensor metrics simultaneously. A single metric anomaly is easy to spot. But an anomaly that appears in combinations — CPU is high AND memory is low AND disk I/O is normal — is invisible to univariate tests.

**The core insight**: anomalies are rare and different. They are easier to isolate with random axis-aligned splits. An anomaly requires fewer splits to isolate (shorter path length in a random tree) than a normal point embedded in a dense cluster.

```python
from sklearn.ensemble import IsolationForest
import numpy as np

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

**What breaks**: Isolation Forest does not model time. Two consecutive anomalous points are treated as independent samples. It has no concept of "this looks normal for a Monday but not for a Sunday." Pair it with calendar features to give it temporal context.

---

### 8.3 LSTM Autoencoder

**The problem**: you have unlabeled time series data. You want to detect anomalies without labeling anomalies — there are too few of them to train a classifier directly.

**The core insight**: train a model to reconstruct normal sequences. Anomalous sequences will have high reconstruction error because the model has only learned to reconstruct the normal distribution.

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
        context = h[-1].unsqueeze(1).repeat(1, self.seq_len, 1)
        reconstruction, _ = self.decoder(context)
        return reconstruction

model = LSTMAutoencoder(input_size=1, hidden_size=32, seq_len=50)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# After training, set threshold from training reconstruction errors
train_errors = []
with torch.no_grad():
    for batch in train_loader:
        recon = model(batch)
        errors = ((batch - recon) ** 2).mean(dim=(1, 2))
        train_errors.extend(errors.numpy())

threshold = np.percentile(train_errors, 95)

with torch.no_grad():
    recon = model(test_data)
    test_errors = ((test_data - recon) ** 2).mean(dim=(1, 2))
    anomalies = test_errors > threshold
```

**What breaks**: the model learns to reconstruct whatever it is trained on. If the training set contains unlabeled anomalies, it learns to reconstruct those too. Pre-clean training data or use a robust loss function.

---

### 8.4 VAE for Anomaly Detection

**The problem**: an LSTM autoencoder gives a reconstruction score but no probability. You cannot say "this event has a 0.1% chance under normal conditions."

**The core insight**: a VAE learns a probabilistic latent space. The anomaly score is not just reconstruction error but also the KL divergence from the latent prior — anomalies encode to unusual regions of latent space.

```
anomaly_score = reconstruction_loss + β · KL_divergence
```

DONUT (2018) and DAGMM apply this to multivariate time series.

**What breaks**: VAE anomaly detection requires careful tuning of `β`. Too high and KL dominates — all anomalies look the same. Too low and reconstruction dominates — reverts to a standard autoencoder.

---

### 8.5 Production Considerations

Key decisions:
1. **Threshold**: set on held-out normal data, typically 95th or 99th percentile of reconstruction errors
2. **Smoothing**: apply moving average to scores to reduce false positives from single-point spikes
3. **Contextual anomalies**: CPU at 100% at 3 AM is anomalous; at 2 PM it is not — incorporate time-of-day features
4. **Drift vs spike**: distinguish gradual drift (concept drift) from sudden spikes — they require different responses

---

## 9. Evaluation Metrics

### 9.1 Point Forecast Metrics

**The problem**: MSE is scale-dependent. A model with MSE=100 on energy demand (GW) is not comparable to a model with MSE=0.01 on stock returns. You cannot benchmark across series or report a single number that means anything.

**The core insight**: use scale-free metrics for cross-series comparison; use scale-dependent metrics when comparing models on a single series.

**MAE**: `(1/n) · Σ|y_t − ŷ_t|`. Robust to outliers. Same units as the target.

**RMSE**: `√((1/n) · Σ(y_t − ŷ_t)²)`. Penalizes large errors more. Use when large misses are especially costly.

**MAPE**: `(100/n) · Σ|y_t − ŷ_t| / |y_t|`. Scale-free but **undefined when `y_t ≈ 0`** and asymmetric: overforecasting by X is penalized less than underforecasting by X.

**sMAPE**: `(100/n) · Σ 2|y_t − ŷ_t| / (|y_t| + |ŷ_t|)`. Reduces asymmetry. Still problematic near zero.

**MASE (Mean Absolute Scaled Error)**:
```
MASE = MAE / (1/(T−s)) · Σ|y_t − y_{t-s}|
```
Denominator is the MAE of the naive seasonal forecast (predict today = same period last season). MASE < 1 = you beat the seasonal naive. MASE is scale-free, symmetric, well-defined at zero, and the standard metric for M-competition benchmarks.

```python
def mase(actual, forecast, seasonal_naive_errors):
    mae = np.mean(np.abs(actual - forecast))
    naive_mae = np.mean(np.abs(seasonal_naive_errors))
    return mae / naive_mae
```

**What breaks**: MASE is undefined if the seasonal naive baseline has zero error (perfectly predictable series). RMSE is distorted by a single large outlier — if you care about median performance, use MAE.

---

### 9.2 Probabilistic Forecast Metrics

**The problem**: a point forecast communicates nothing about uncertainty. For inventory management, you need P95 of demand. For energy grids, you need the full distribution to set reserve capacity.

**The core insight**: evaluate the quality of the predicted distribution, not just its center.

**Quantile Loss (Pinball Loss)**:
```
QL_q(y, ŷ_q) = q · (y − ŷ_q)     if y ≥ ŷ_q
             = (1−q) · (ŷ_q − y)  if y < ŷ_q
```

**CRPS (Continuous Ranked Probability Score)**:
```
CRPS(F, y) = ∫ (F(x) − 1[x ≥ y])² dx
```
Where `F` is the predicted CDF. Equals MAE for point forecasts — lower is better.

```python
from properscoring import crps_ensemble
import numpy as np

# samples: (n_timesteps, n_ensemble_members)
crps_scores = crps_ensemble(actual, ensemble_samples)
print(f"Mean CRPS: {crps_scores.mean():.4f}")
```

**Calibration check**: if 90% prediction intervals contain the true value 90% of the time, the model is calibrated. If coverage < 90%, intervals are too narrow (overconfident).

---

### 9.3 Choosing Metrics

| Situation | Recommended Metric |
|---|---|
| Cross-series comparison | MASE or sMAPE |
| Same series, different models | RMSE or MAE |
| High-cost large errors | RMSE |
| Series with zero-crossing values | RMSE or MASE (not MAPE) |
| Probabilistic forecasts | CRPS or WQL |
| Interval forecasts | Coverage + interval width |

---

## 10. Cross-Validation for Time Series

### 10.1 Why Standard k-Fold Fails

**The problem**: k-fold cross-validation randomly assigns observations to folds. For time series, this means training on 2023 data to predict 2022. A model that perfectly memorizes future values would score perfectly in cross-validation and fail entirely in deployment.

**The core insight**: the train/test boundary must always move forward in time. You can never train on the future to predict the past.

**What breaks specifically**:
1. **Future leakage**: model trains on post-cutoff observations
2. **Dependency violation**: train and test share temporal neighbors with correlated errors
3. **Feature leakage**: lag features computed on the full dataset before splitting

---

### 10.2 Train-Validation-Test Split

The simplest valid approach — fixed cutoffs:

```python
n = len(df)
train_end = int(n * 0.70)
val_end   = int(n * 0.85)

train = df.iloc[:train_end]
val   = df.iloc[train_end:val_end]
test  = df.iloc[val_end:]
```

Tune hyperparameters on `val`. Report final numbers on `test` — touched exactly once.

---

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

Each fold expands the training set. The `gap` parameter skips observations near the boundary — critical when lag features use lags shorter than `gap`.

---

### 10.4 Walk-Forward Validation (Sliding Window)

Both train and validation windows slide forward, keeping training size fixed:

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

Walk-forward is more expensive but better represents a model retrained on a rolling window in production.

**What breaks**: the feature engineering leakage trap. Computing rolling statistics before splitting, then indexing into those pre-computed arrays, is still a leak. Always compute features inside each fold, starting from raw data.

---

## 11. Multivariate Forecasting

### 11.1 Problem Formulation

Given `M` time series `Y_t = [y_t¹, ..., y_tᴹ]`, predict `H` steps ahead for all `M` variables.

---

### 11.2 Channel-Independent vs Channel-Dependent

**The problem**: modeling cross-variable interactions adds parameters and potential overfitting. Should you model each variable separately or jointly?

**The core insight**: cross-variable modeling helps only when cross-variable dependencies are strong and consistent. If variables are weakly correlated, modeling them jointly introduces noise rather than signal.

**Channel-independent (CI)**: same model applied to each variable independently. PatchTST, most linear models. Acts as regularization — cannot overfit to spurious correlations.

**Channel-dependent (CD)**: explicit cross-variable modeling. iTransformer, VAR, GNNs. Better when variables have known causal structure.

Empirically: CI often wins on standard benchmarks because cross-variable correlation is weaker than expected. CD wins when variables are genuinely causally linked (multiple sensors in a controlled system).

---

### 11.3 Vector Autoregression (VAR)

The classical multivariate AR extension:

```
Y_t = c + A₁ · Y_{t-1} + A₂ · Y_{t-2} + ... + Aₚ · Y_{t-p} + ε_t
```

Each variable is regressed on its own past AND the past of all other variables. Parameters scale as `M² · p` — expensive for large `M`.

```python
from statsmodels.tsa.vector_ar.var_model import VAR

model = VAR(train_data)  # shape: (T, M)
results = model.fit(maxlags=15, ic='aic')
forecast = results.forecast(train_data[-results.k_ar:], steps=12)
```

**What breaks**: VAR with M=50 variables and p=10 lags has 25,000 parameters. With T=200 observations, this is severely underdetermined. Regularization (ridge, lasso) or dimensionality reduction is necessary.

---

### 11.4 LightGBM/XGBoost for Multivariate

Often the most practical approach for tabular-izable multivariate forecasting:

1. Build feature matrix: lag features of all `M` variables, calendar features, rolling stats
2. Train a single gradient boosting model per target variable (or multi-output)
3. Use recursive or direct strategy for multi-step

```python
import lightgbm as lgb

X_train, y_train = build_feature_matrix(train_data, lags=[1,2,3,24], horizon=1)

model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=31)
model.fit(X_train, y_train)
```

**What breaks**: tree models cannot extrapolate beyond training range. If a covariate takes values at test time that it never took during training, the tree outputs the value from the nearest training leaf — which may be far from correct.

---

## 12. Probabilistic Forecasting

### 12.1 Why Point Forecasts Are Insufficient

**The problem**: a retailer orders inventory based on the point forecast of demand. The forecast is 1000 units. The model is sometimes off by ±400. The retailer orders 1000 units and stockouts 30% of the time — an unnecessary loss, because if they had known the uncertainty, they would have ordered 1300 units.

**The core insight**: decision-making under uncertainty requires knowing the uncertainty. A point forecast strips out the information needed to make optimal decisions.

---

### 12.2 Quantile Regression

**The core insight**: directly train the model to predict specific quantiles by minimizing the pinball (quantile) loss — an asymmetric loss that rewards the model for placing the `q`-quantile at the correct position.

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

lower = quantile_models[0.1].predict(X_test)
upper = quantile_models[0.9].predict(X_test)
```

**Quantile crossing**: independently trained quantile models may produce Q90 < Q50 in some regions. Fix with isotonic regression post-processing or train with a joint quantile loss.

---

### 12.3 Deep Learning Probabilistic Methods

**MC Dropout**: enable dropout at inference, run `N` forward passes, approximate posterior:
```python
def mc_dropout_predict(model, x, n_samples=100):
    model.train()  # keep dropout active
    with torch.no_grad():
        samples = torch.stack([model(x) for _ in range(n_samples)])
    return samples.mean(0), samples.std(0)
```

**Deep AR** (Amazon): LSTM that outputs distribution parameters (μ, σ) at each step. Loss = negative log likelihood. Naturally produces predictive intervals.

```
p(y_t | y_{<t}, x_t) = N(μ_θ(y_{<t}, x_t), σ_θ(y_{<t}, x_t))
```

**What breaks**: MC Dropout underestimates uncertainty because the approximate posterior is too concentrated. Calibration should be checked against empirical coverage.

---

### 12.4 Conformal Prediction for Time Series

**The problem**: Bayesian and parametric intervals require distributional assumptions. If those assumptions are wrong (non-Gaussian errors, heavy tails), the stated coverage is wrong.

**The core insight**: conformal prediction makes no distributional assumptions. It provides marginal coverage guarantees: request a 90% interval, and 90% of test observations fall inside — regardless of the model's internal assumptions.

```python
def conformal_forecast(model, calibration_residuals, alpha=0.1):
    """alpha: desired error rate (0.1 -> 90% coverage)"""
    q_hat = np.quantile(np.abs(calibration_residuals), 1 - alpha)

    def predict_interval(x):
        y_hat = model.predict(x)
        return y_hat - q_hat, y_hat + q_hat

    return predict_interval
```

**Adaptive Conformal Inference (ACI)**: standard conformal prediction assumes exchangeability (i.i.d.). Time series violates this — the error distribution changes over time. ACI adaptively updates the quantile as coverage errors are observed.

**What breaks**: conformal prediction provides marginal (average) coverage, not conditional coverage. The interval may be too narrow on hard examples and too wide on easy ones.

---

### 12.5 Forecast Combination

**The problem**: every model family has biases. ARIMA misses nonlinear patterns. LSTMs require much data. LightGBM extrapolates poorly. No single model is universally best.

**The core insight**: averaging predictions from diverse models cancels independent errors. The ensemble is often better than any constituent model — proven repeatedly in the M-competitions.

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

The M4 competition winner combined LSTM outputs with Holt-Winters exponential smoothing.

**What breaks**: ensemble weights derived from validation performance can overfit if the validation period is short or not representative of test conditions. Simple equal weighting is surprisingly hard to beat.

---

## 13. Foundation Models for Time Series

### 13.1 Motivation

**The problem**: every time series forecasting project starts from scratch. You collect data, engineer features, train an ARIMA or LSTM, tune hyperparameters, and validate. For a company with 50,000 SKUs or 10,000 sensors, this is infeasible. And for a new series with 30 observations, there is not enough data to train anything.

**The core insight**: train one model on a massive corpus of diverse time series. At inference, it forecasts any new series zero-shot — no retraining required. The model has absorbed the statistical patterns of thousands of domains.

---

### 13.2 TimeGPT-1 (Nixtla, 2023)

Transformer trained on 100 billion time points from diverse public datasets. API-based zero-shot and fine-tuned modes:

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

**What breaks**: API dependency means no offline inference, uncertain pricing at scale, and no control over model updates.

---

### 13.3 Chronos (Amazon, 2024)

**The problem**: language models handle arbitrary sequences of tokens. Time series are sequences of numbers. Can the same architecture work?

**The core insight**: quantize time series values into discrete tokens, then treat forecasting as language modeling. The T5-style encoder-decoder autoregressively generates future tokens, which are then de-quantized back to values.

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

**What breaks**: quantization introduces discretization error. Very fine-grained signals (sub-0.1% changes in sensor readings) lose information in the tokenization step.

---

### 13.4 Lag-Llama (2024)

Llama-style decoder-only transformer adapted for time series. Key design:
- **Patch tokenization**: patches of consecutive timesteps as tokens
- **Distributional output**: Student-t distribution over future values
- **Trained on**: large Chronos/Lotsa dataset collection

Zero-shot performance competitive with dataset-specific models on held-out benchmarks.

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

---

### 13.5 Moirai (Salesforce, 2024)

"Universal" forecasting model using masked encoder. Designed to handle variable context lengths, variable prediction lengths, multiple frequencies, and any number of variables at inference. Uses patch-based tokenization with frequency-specific patch sizes.

---

### 13.6 TimesFM (Google, 2024)

Decoder-only foundation model (200M parameters) trained on 100 billion real-world time points. Autoregressive patching approach. Zero-shot performance reportedly better than many task-specific models on standard benchmarks.

---

### 13.7 When to Use Foundation Models

| Situation | Recommendation |
|---|---|
| Cold start, no training data | Foundation model zero-shot |
| Small dataset (<1k observations) | Foundation model + light fine-tuning |
| Large dataset, stable distribution | Task-specific model (N-HiTS, PatchTST) |
| Real-time, low latency | LightGBM + features |
| Multiple series, shared patterns | Foundation model or global neural model |

**What breaks**: foundation models trained on public datasets may not represent highly domain-specific series (industrial sensor data, rare disease biomarkers). Zero-shot performance often lags fine-tuned task-specific models when data is abundant.

---

## 14. Production Patterns

### 14.1 Concept Drift

**The problem**: you trained a demand forecasting model in January. By October, buying patterns shifted after a major competitor launched. Your model's validation error was 3%; now it is 12%. But your automated alerts only fire at 15%. You shipped wrong forecasts for months before noticing.

**The core insight**: model accuracy in production degrades silently. You must monitor for distribution shifts and prediction error changes continuously, not just at deployment.

**Types**:
- **Sudden drift**: COVID lockdowns changing energy demand overnight
- **Gradual drift**: customer behavior shifting over months
- **Recurring drift**: seasonal patterns returning each year (expected and handled by seasonal features)

**Detection**:

Page-Hinkley test — cumulative sum over normalized residuals:
```
m_t = Σ_{i=1}^t (x_i − μ₀ − δ)
M_T = max(m_1, ..., m_T)
drift detected if m_T − M_T > λ
```

ADWIN — maintains a sliding window and detects drift when two sub-windows have significantly different means:
```python
from river.drift import ADWIN

detector = ADWIN()
for error in model_errors:
    detector.update(error)
    if detector.drift_detected:
        print("Drift detected — trigger retraining")
```

**What breaks**: drift detection based on prediction error requires ground-truth labels, which arrive with a lag. Feature distribution monitoring (PSI, KL divergence on inputs) catches drift earlier, without waiting for labels.

---

### 14.2 Retraining Triggers

| Trigger Type | Description | Use Case |
|---|---|---|
| Scheduled | Retrain every N days/weeks | Stable series, low ops cost |
| Performance-based | Retrain when val error > threshold | When monitoring is available |
| Drift-based | Retrain on drift detection | When distribution shifts detectably |
| Continuous (online) | Update on each observation | High-frequency, fast-changing series |

**Model decay rate** varies by domain:
- E-commerce demand: days to weeks
- Energy demand: weeks to months
- Financial markets: hours to days

---

### 14.3 Data Pipeline

```
Raw data sources
       |
   Ingestion + validation (Great Expectations, Pandera)
       |
   Feature engineering (lag, rolling, calendar, Fourier)
       |
   Temporal train/validate/test split
       |
   Model training
       |
   Model registry (MLflow, W&B)
       |
   Serving (batch / online)
       |
   Monitoring (prediction error, drift detection)
       |
   Retraining trigger
```

---

### 14.4 Common Production Pitfalls

1. **Scaler fit on full dataset**: StandardScaler fit before split — fit only on training data, transform test
2. **Lookahead in lag features**: `shift(0)` or negative shifts
3. **Missing value handling**: forward-fill creates artificial autocorrelation; interpolation can leak future
4. **Horizon mismatch**: evaluating 1-step-ahead but deploying as 24-step-ahead — completely different difficulty
5. **Too-short test period**: seasonal patterns need at least 2-3 full cycles in the test set to evaluate correctly
6. **Latency at inference**: a model requiring 512 lookback steps may miss the SLA for real-time predictions

---

### 14.5 Global vs Local Models

**Local model**: one model per series (ARIMA, Prophet). Works well for a small number of important series. Does not scale to thousands.

**Global model**: one model across all series (LSTM with series identifier, LightGBM, N-BEATS). Can learn shared patterns. Requires careful normalization.

**The problem**: global models see series at different scales. A model trained on sales from 1 to 10 million units cannot directly generalize to a new series ranging from 0.1 to 0.5 units without normalization.

**RevIN (Reversible Instance Normalization)**: normalize at input, reverse normalization at output:
```python
def normalize_per_series(batch):
    # batch: (batch_size, seq_len)
    mean = batch.mean(dim=1, keepdim=True)
    std  = batch.std(dim=1, keepdim=True) + 1e-8
    return (batch - mean) / std, mean, std

def denormalize(normalized, mean, std):
    return normalized * std + mean
```

Used in PatchTST and N-HiTS. The key insight: normalize each instance independently at runtime, not globally — this allows the model to generalize across series at different scales.

**What breaks**: RevIN assumes the test series distribution is stationary enough that the normalization statistics computed on the input window are representative of the forecast window. This fails on sudden level shifts.

---

## 15. Common Interview Questions

---

**Q: What is the difference between ARIMA and SARIMA?**

ARIMA(p,d,q) handles non-seasonal series: AR terms model dependence on past values, differencing removes trend to achieve stationarity, MA terms model dependence on past errors. SARIMA(p,d,q)(P,D,Q)[S] adds a second set of AR/I/MA terms that operate at lag S rather than lag 1. For monthly data with annual seasonality, you need SARIMA with S=12. ARIMA applied to seasonal data without seasonal differencing leaves autocorrelation at seasonal lags — visible in the residual ACF plot.

---

**Q: Why can't you use standard k-fold cross-validation for time series?**

k-fold randomly shuffles observations into folds, breaking temporal order in two ways: (1) the model trains on future observations to predict past ones — future leakage that would not exist in deployment; (2) adjacent observations in time are correlated, so randomly splitting bleeds information across folds. Use TimeSeriesSplit (expanding window) or walk-forward validation (sliding window) to maintain temporal order.

---

**Q: How do LSTMs handle the vanishing gradient problem compared to vanilla RNNs?**

Vanilla RNNs backpropagate through `W_hh` at every timestep. If singular values of `W_hh` are < 1, gradients shrink exponentially — the 100-step-back gradient is essentially zero. LSTMs introduce an additive cell state update: `c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t`. The gradient path through `c_t` does not repeatedly multiply through the same matrix. The forget gate `f_t` can zero out old memories, but that is a controlled, learned operation — not an uncontrolled exponential decay. This allows gradients to flow across hundreds of timesteps.

---

**Q: What is teacher forcing and when is it a problem?**

Teacher forcing feeds ground-truth decoder output as input at each decoding step during training. This speeds training because errors do not compound. The problem is **exposure bias**: at inference, the model never saw its own (imperfect) predictions as inputs during training. A small error at step 1 shifts the input distribution for step 2, which the model has never handled, potentially causing error amplification across the horizon. Mitigation: scheduled sampling (gradually reduce teacher forcing ratio), or use non-autoregressive decoders (N-BEATS, Informer's generative decoder) that avoid the problem entirely.

---

**Q: When would you choose a Transformer over an LSTM for time series?**

Transformers are preferred when: (1) long-range dependencies matter and sequences are long (>100 steps) — LSTMs degrade; (2) parallelism matters during training; (3) enough data exists — Transformers have more parameters and require more data. LSTMs are preferred when: (1) data is limited; (2) online/streaming inference with per-step state update is required; (3) the series has local structure without long-range dependencies, where sequential processing is fine. Note: on standard benchmarks, simple patched Transformers (PatchTST) and even linear models often outperform complex Transformer variants — architecture choice should be empirically validated.

---

**Q: What is MASE and why is it preferred over MAPE?**

MASE = `MAE / MAE_naive_seasonal`. The denominator is the error of predicting today = same period last year. MASE < 1 means you beat the naive baseline. MAPE is undefined when actuals are zero or near-zero, and is asymmetric: a forecast that is 50 above actual has lower MAPE than one that is 50 below actual (because the denominator is `|y_t|`). MASE avoids both issues and is scale-free, making it valid for comparing across series with different units, scales, and value ranges.

---

**Q: How would you detect and handle concept drift in a production forecasting system?**

Detection: monitor rolling forecast error (e.g., 7-day MAE vs. historical baseline). Apply ADWIN or Page-Hinkley on residuals for statistical drift detection. Compare incoming feature distributions to training distributions using PSI (Population Stability Index) or KL divergence — this catches drift before ground-truth labels arrive.

Handling: (1) scheduled retraining on a rolling window; (2) drift-triggered retraining; (3) online learning (SGD with decaying rate, continuously updating the model); (4) ensemble models trained on different historical windows, weighted by recent performance. The choice depends on how fast the distribution shifts and the cost of retraining.

---

**Q: What are the advantages of N-BEATS over LSTM-based models?**

N-BEATS is MLP-based — no recurrence, no attention. This gives: (1) fully parallel computation during training; (2) interpretability in the interpretable variant — explicit trend and seasonality stacks with polynomial and Fourier basis functions; (3) no sequential bottleneck — the model processes all lags simultaneously; (4) no vanishing gradient across time. The backcast mechanism naturally decomposes the forecast into additive components. On M4 and M5 benchmarks, N-BEATS frequently matches or outperforms Transformer variants despite architectural simplicity.

---

**Q: How do you approach forecasting with multiple seasonalities?**

Options: (1) **TBATS**: explicit multiple Fourier seasonality terms with trigonometric state space; (2) **Prophet**: additive Fourier seasonality components at user-specified periods; (3) **Fourier features**: encode each seasonal period as sin/cos pairs, then use any regression or neural model — this is the most flexible approach; (4) **Deep learning**: LSTM or TCN with a lookback window large enough to cover the longest seasonal period; (5) **STL + residual modeling**: decompose one seasonality using STL, model the remainder with a separate model.

---

**Q: What is conformal prediction and why is it useful for time series?**

Conformal prediction constructs prediction intervals with a guaranteed marginal coverage: request 90% coverage, get 90% empirical coverage, regardless of the model's internal distributional assumptions. No Gaussianity required. For time series, standard conformal prediction assumes exchangeability (i.i.d. residuals), which is violated by temporal autocorrelation and non-stationarity. Adaptive Conformal Inference (ACI) addresses this by updating the coverage target over time based on observed coverage errors. Unlike Bayesian intervals, conformal prediction is post-hoc — it wraps any trained point forecast model.
