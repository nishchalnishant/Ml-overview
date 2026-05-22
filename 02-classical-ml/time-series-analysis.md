# Time Series Analysis

Entirely absent from the repository — this is the complete reference covering classical methods through modern foundation models.

---

## 1. Core Concepts

### Stationarity

A time series is **stationary** if its statistical properties (mean, variance, autocorrelation) do not change over time.

**Strictly stationary:** Joint distribution invariant under time shifts.

**Weakly (covariance) stationary:**
$$E[X_t] = \mu \text{ (constant)}$$
$$\text{Var}(X_t) = \sigma^2 \text{ (constant)}$$
$$\text{Cov}(X_t, X_{t+k}) = \gamma(k) \text{ (depends only on lag k, not t)}$$

**Why it matters:** ARIMA and most classical models assume stationarity. Non-stationary series → spurious regressions.

**Augmented Dickey-Fuller (ADF) Test:**
$$\Delta y_t = \alpha + \beta t + \gamma y_{t-1} + \sum_{i=1}^p \delta_i \Delta y_{t-i} + \epsilon_t$$

H₀: γ = 0 (unit root, non-stationary). Reject H₀ → stationary.

```python
from statsmodels.tsa.stattools import adfuller

def test_stationarity(series, name=""):
    result = adfuller(series.dropna())
    print(f"{name} ADF Statistic: {result[0]:.4f}")
    print(f"{name} p-value: {result[1]:.4f}")
    print(f"{'Stationary' if result[1] < 0.05 else 'Non-stationary (difference needed)'}")
    return result[1] < 0.05
```

### Transformations to Achieve Stationarity

| Problem | Transformation |
|---|---|
| Trend | First difference: ΔXt = Xt - Xt-1 |
| Seasonal trend | Seasonal difference: ΔXt = Xt - Xt-s |
| Exponential growth | Log transform: log(Xt) |
| Variance instability | Box-Cox transform |

---

## 2. ACF and PACF

**ACF (Autocorrelation Function):** Correlation of series with itself at lag k.
$$\rho_k = \frac{\text{Cov}(X_t, X_{t-k})}{\text{Var}(X_t)}$$

**PACF (Partial Autocorrelation Function):** Correlation at lag k after removing effects of lags 1..k-1.

**Reading ACF/PACF for model identification:**

| Pattern | Model suggested |
|---|---|
| ACF: tails off; PACF: cuts off at lag p | AR(p) |
| ACF: cuts off at lag q; PACF: tails off | MA(q) |
| Both tail off | ARMA(p, q) |
| Neither tails off (ACF slow decay) | Non-stationary → difference first |

```python
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig, axes = plt.subplots(2, 1, figsize=(12, 6))
plot_acf(series, ax=axes[0], lags=40)
plot_pacf(series, ax=axes[1], lags=40)
plt.tight_layout()
```

---

## 3. ARIMA Models

### AR(p) — Autoregressive

$$X_t = c + \sum_{i=1}^p \phi_i X_{t-i} + \epsilon_t$$

Xt depends on its own past p values plus white noise ε.

**Stationarity condition:** Roots of $1 - \phi_1 z - \phi_2 z^2 - ... - \phi_p z^p = 0$ lie outside the unit circle.

### MA(q) — Moving Average

$$X_t = \mu + \epsilon_t + \sum_{j=1}^q \theta_j \epsilon_{t-j}$$

Xt depends on past q error terms.

### ARMA(p, q)

$$X_t = c + \sum_{i=1}^p \phi_i X_{t-i} + \epsilon_t + \sum_{j=1}^q \theta_j \epsilon_{t-j}$$

### ARIMA(p, d, q)

d = number of differences required for stationarity.

```
ARIMA(2, 1, 1) means:
  - Difference once (d=1): ΔXt = Xt - Xt-1
  - Apply ARMA(2,1) to ΔXt
```

### SARIMA(p, d, q)(P, D, Q, s)

Adds seasonal components at lag s (s=12 for monthly data, s=4 for quarterly):
$$\Phi_P(B^s) \phi_p(B) (1-B)^d (1-B^s)^D X_t = \Theta_Q(B^s) \theta_q(B) \epsilon_t$$

```python
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Fit ARIMA
model = ARIMA(train, order=(2, 1, 2))
result = model.fit()
forecast = result.forecast(steps=30)

# Auto model selection via AIC
from pmdarima import auto_arima
model = auto_arima(train, seasonal=True, m=12, 
                   information_criterion='aic',
                   stepwise=True, trace=True)
```

**Information criteria for model selection:**
$$\text{AIC} = 2k - 2\ln(\hat{L})$$
$$\text{BIC} = k \ln(n) - 2\ln(\hat{L})$$

where k = number of parameters, n = sample size. Lower is better. BIC penalizes complexity more heavily.

---

## 4. Exponential Smoothing

### Simple Exponential Smoothing (SES)

No trend, no seasonality. Forecast = weighted average of past, with exponentially decaying weights.

$$\hat{y}_{t+1} = \alpha y_t + (1-\alpha) \hat{y}_t, \quad \alpha \in (0, 1)$$

Recursively: $\hat{y}_{t+1} = \alpha \sum_{k=0}^{t} (1-\alpha)^k y_{t-k}$

### Holt's Linear Smoothing (Double Exponential)

Adds trend component:
$$l_t = \alpha y_t + (1-\alpha)(l_{t-1} + b_{t-1})$$
$$b_t = \beta(l_t - l_{t-1}) + (1-\beta)b_{t-1}$$
$$\hat{y}_{t+h} = l_t + h \cdot b_t$$

### Holt-Winters (Triple Exponential)

Adds seasonality:
$$l_t = \alpha \frac{y_t}{s_{t-m}} + (1-\alpha)(l_{t-1} + b_{t-1})$$
$$b_t = \beta(l_t - l_{t-1}) + (1-\beta)b_{t-1}$$
$$s_t = \gamma \frac{y_t}{l_t} + (1-\gamma)s_{t-m}$$
$$\hat{y}_{t+h} = (l_t + h \cdot b_t) s_{t-m+h}$$

```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing

model = ExponentialSmoothing(
    train, 
    trend='add',       # or 'mul'
    seasonal='add',    # or 'mul'
    seasonal_periods=12
).fit()
forecast = model.forecast(24)
```

---

## 5. Prophet (Meta)

**Design goal:** Business time series with strong seasonality, holidays, trend changes.

**Decomposition model:**
$$y(t) = g(t) + s(t) + h(t) + \epsilon_t$$

- g(t): trend (piecewise linear or logistic growth)
- s(t): seasonality (Fourier series)
- h(t): holiday effects
- εt: Gaussian noise

**Piecewise linear trend:**
$$g(t) = (k + \mathbf{a}(t)^T \delta) t + (m + \mathbf{a}(t)^T \gamma)$$

where δ are changepoint adjustments, a(t) is an indicator vector for which changepoints have passed.

**Fourier seasonality:**
$$s(t) = \sum_{n=1}^N \left(a_n \cos\frac{2\pi nt}{P} + b_n \sin\frac{2\pi nt}{P}\right)$$

N=10 for annual seasonality, N=3 for weekly.

```python
from prophet import Prophet

df = pd.DataFrame({'ds': dates, 'y': values})

model = Prophet(
    changepoint_prior_scale=0.05,  # flexibility of trend changes
    seasonality_mode='multiplicative',  # or 'additive'
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False
)
model.add_country_holidays(country_name='US')
model.fit(df[df['ds'] < '2024-01-01'])

future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)
```

**Prophet strengths:** Handles missing data, outliers, multiple seasonalities, easy holiday modeling. **Weaknesses:** Doesn't capture autocorrelation directly, can be overfit with too many changepoints.

---

## 6. Evaluation: Walk-Forward Validation

**Never use random train/test split for time series.** It leaks future information.

**Walk-forward validation (time series cross-validation):**

```
Train:  [1, 2, 3, 4, 5]  Test: [6]
Train:  [1, 2, 3, 4, 5, 6]  Test: [7]
Train:  [1, 2, 3, 4, 5, 6, 7]  Test: [8]
...
```

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5, gap=0)
for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Fold {fold}: MAE={mean_absolute_error(y_test, y_pred):.4f}")
```

**Metrics:**

| Metric | Formula | When to use |
|---|---|---|
| MAE | mean\|y - ŷ\| | robust to outliers |
| RMSE | √mean(y-ŷ)² | penalizes large errors |
| MAPE | mean\|y-ŷ\|/\|y\| × 100 | percentage, easy to communicate |
| MASE | MAE / MAE_naive | scale-free, compares to naive forecast |
| sMAPE | 2\|y-ŷ\|/(|y|+|ŷ\|) | symmetric MAPE, handles zeros better |

**MASE (Mean Absolute Scaled Error):**
$$\text{MASE} = \frac{MAE}{\frac{1}{n-1}\sum_{t=2}^n |y_t - y_{t-1}|}$$

MASE < 1: better than naive; MASE > 1: worse than naive.

---

## 7. Deep Learning for Time Series

### Temporal Convolutional Network (TCN)

Causal dilated convolutions with exponentially growing receptive field.

```
Layer 0 (dilation=1): x[t], x[t-1], x[t-2]
Layer 1 (dilation=2): x[t], x[t-2], x[t-4]
Layer 2 (dilation=4): x[t], x[t-4], x[t-8]
```

Receptive field at layer L = 2^L (exponential growth).

```python
class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)
    
    def forward(self, x):
        # Add causal padding (only past, no future)
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)
```

### Transformer for Time Series

**PatchTST (2023):** Split time series into patches (subseries), treat as tokens.
- Patch size 16, no overlap
- Channel independence: each variate processed separately with shared weights
- Linear complexity in sequence length for long-horizon forecasting

**Informer:** Sparse self-attention via ProbSparse (O(L log L) instead of O(L²)).

**Autoformer:** Auto-correlation mechanism as substitute for attention.

### N-BEATS / N-HiTS

Pure MLP architecture with basis function expansion:

$$\hat{y} = \sum_k \theta_k^f \phi_k(t), \quad \hat{x} = \sum_k \theta_k^b \phi_k(t)$$

Interpretable: separate stacks for trend, seasonality. Outperforms many Transformer variants on univariate forecasting (M4 competition).

---

## 8. Foundation Models for Time Series

### Chronos (Amazon, 2024)

- Tokenizes time series values (quantization into bins)
- Uses T5 language model architecture
- Trained on 100K+ time series datasets
- Zero-shot forecasting without task-specific fine-tuning

```python
from chronos import ChronosPipeline

pipeline = ChronosPipeline.from_pretrained("amazon/chronos-t5-small")
forecast = pipeline.predict(
    context=torch.tensor(train_values).unsqueeze(0),
    prediction_length=12,
    num_samples=100  # probabilistic forecast
)
```

### Moirai (Salesforce, 2024)

- Universal forecasting model
- Any frequency, any prediction length
- Patch-based tokenization (like PatchTST)
- Handles multivariate series with mixed frequencies

### TimesFM (Google, 2024)

- 200M parameter foundation model
- Trained on 100B time points (Google Trends, Wikipedia, etc.)
- Competitive zero-shot performance on standard benchmarks

**When to use foundation models vs classical:**
| Scenario | Use |
|---|---|
| Cold start, no historical data | Foundation model (zero-shot) |
| Long history, specific domain | ARIMA/Prophet (interpretable, fast) |
| Complex seasonality, many series | LightGBM/XGBoost with lag features |
| Long-horizon, multivariate | PatchTST/TimesNet |
| Irregular time series | Chronos/Moirai |

---

## 9. Anomaly Detection in Time Series

```python
# STL decomposition + residual thresholding
from statsmodels.tsa.seasonal import STL

stl = STL(series, seasonal=13)
result = stl.fit()
residual = result.resid

# Flag points > k standard deviations from residual mean
k = 3
anomalies = np.abs(residual) > k * residual.std()

# Or use ADWIN for streaming anomaly detection
from river.drift import ADWIN
detector = ADWIN()
for value in stream:
    detector.update(value)
    if detector.drift_detected:
        print(f"Anomaly at step {step}")
```

---

## Canonical Interview Q&As

**Q: Your time series model performs well on the test set but poorly in production. What are the likely causes?**  
A: (1) Data leakage: used random split instead of temporal split — test set contains future information; (2) training-serving skew: features computed differently at training vs serving time; (3) concept drift: the underlying pattern changed (e.g., COVID effect on demand forecasting); (4) evaluation metric mismatch: offline MAPE looks good but you're measuring on high-volume periods while serving includes quiet periods; (5) missing exogenous variables: production needs real-time signals (weather, promotions) that weren't in training. Fix: temporal cross-validation, shadow serve before cutover, monitor PSI on model features.

**Q: When would you use ARIMA vs Prophet vs XGBoost for time series forecasting?**  
A: ARIMA: stationary series, univariate, short horizon (days/weeks), need interpretable coefficients, limited data. Prophet: strong seasonality (holidays, weekly/annual patterns), irregular frequency, non-expert users who need tunable components, robust to missing data. XGBoost with lag features: many simultaneous series (e.g., sales forecast for 10K SKUs), external regressors dominate (price, promotions), complex non-linear patterns, sufficient history. Deep learning (TCN/Transformer): very long horizon, multivariate with complex cross-series dependencies, when data volume justifies training cost.

**Q: How do you choose the order (p, d, q) for ARIMA?**  
A: (1) Test stationarity with ADF — if p-value > 0.05, difference (d++); (2) plot ACF: if it cuts off at lag q, MA(q) component; (3) plot PACF: if it cuts off at lag p, AR(p) component; (4) if both tail off, need ARMA. Use auto_arima (pmdarima) for automated search via AIC/BIC optimization. Validate with walk-forward cross-validation, not just in-sample AIC. Common mistake: over-differencing (d=2 when d=1 suffices) or selecting p too large, leading to overfitting.

**Q: Explain the difference between trend stationarity and difference stationarity.**  
A: Trend-stationary: has a deterministic trend that can be removed by subtracting a fitted trend line. The residuals are stationary. Difference-stationary (unit root): stochastic trend — shocks have permanent effects. Must be differenced (not detrended) to achieve stationarity. ARIMA handles difference-stationary series. Detrending a difference-stationary series doesn't fully remove non-stationarity — the ADF test distinguishes these. Practical implication: differencing is the safe default for financial time series; trend subtraction is appropriate for physical processes with deterministic trends.
