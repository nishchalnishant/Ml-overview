# Time series

#### 📜 What is Time Series Data?

First, time series data (or time series forecasting) is a type of problem where you use a sequence of historical data points (e.g., sales per day, stock price per hour) to predict future values.

The key characteristics of time series data are:

* Time-Dependence: The order of the data matters. Today's value is dependent on yesterday's value.
* Seasonality: A repeating, predictable pattern over a fixed period (e.g., ice cream sales are high every summer).
* Trend: A long-term upward or downward movement (e.g., a company's growing revenue over 10 years).
* Autocorrelation: The relationship between a value and its own past values (or "lags").

The models we use are designed to capture these properties. They fall into three main families.

***

#### 1. Classical Statistical Models

These models are the traditional workhorses of time series forecasting. They are "univariate," meaning they typically use only the past values of the target variable to make a prediction.

**### ARIMA (AutoRegressive Integrated Moving Average)**

* How it Works: This is a combination of three components, defined by three parameters (p, d, q):
  * AR (AutoRegressive - p): Assumes the current value is a linear combination of its _own past values_. (e.g., "Today's sales are 90% of yesterday's sales + 10% of the day before's sales").
  * I (Integrated - d): This component makes the data "stationary" by _differencing_. Stationarity means the series has a constant mean and variance (no trend or seasonality). This is done by subtracting the previous value (e.g., `value_today - value_yesterday`). The `d` parameter is the number of times this is done.
  * MA (Moving Average - q): Assumes the current value is a linear combination of _past prediction errors_. (e.g., "Today's value is the mean + 80% of yesterday's error").
* Pros:
  * Interpretable: The (p,d,q) parameters have clear statistical meaning.
  * Strong Foundation: Based on decades of statistical theory.
  * Works well on simple data: Excellent for many real-world problems (e.g., sales, simple stock/economic indicators) when its assumptions are met.
  * Low data requirement: Can work well on smaller datasets.
* Cons:
  * Assumes Linearity: Cannot capture complex, non-linear patterns.
  * Requires Stationarity: You _must_ pre-process the data to be stationary (remove trend/seasonality), which can be complex.
  * Univariate: In its basic form, it can't use other variables. (The ARIMAX variant allows for "exogenous" variables, but it's still linear).
  * Parameter tuning (p,d,q) is manual: Requires statistical knowledge (e.g., reading ACF/PACF plots).
* Variant (SARIMA): This is Seasonal ARIMA. It's the same as ARIMA but adds another set of (P,D,Q,m) parameters to _specifically_ model the seasonal component. This is the go-to model for data with a clear, repeating cycle (like sales per month).

***

**### Prophet**

* How it Works: Developed by Facebook, this is an automated forecasting tool based on a decomposable additive model: $$ $y(t) = g(t) + s(t) + h(t) + \epsilon_t$ $$
  * $$ $g(t)$ $$: Trend (modeled as piecewise linear or logistic growth).
  * $$ $s(t)$ $$: Seasonality (modeled using Fourier Series, allowing it to fit multiple seasonalities like day-of-week and month-of-year).
  * $$ $h(t)$ $$: Holidays (a user-provided list of special events).
* Pros:
  * Extremely User-Friendly: Designed to be intuitive for analysts, not just statisticians. `fit()`, `predict()`, `plot()`.
  * Handles seasonality well: Can model multiple, complex seasonal patterns.
  * Robust: Handles missing data and outliers automatically.
  * Interpretable: You can easily plot the individual trend, seasonal, and holiday components to see what the model has learned.
  * Fast and requires little tuning.
* Cons:
  * "Black Box" (in a way): It's an automated procedure. If its core assumptions are wrong, it's hard to manually fix or tune.
  * Not a general model: It's specifically for business forecasting and may perform poorly on other types of time series (e.g., highly volatile financial data).
  * No Autocorrelation Modeling: It doesn't explicitly model the AR/MA components, so it can miss some simpler patterns that ARIMA would catch.

***

#### 2. Machine Learning (Feature-Based) Models

This approach completely changes the problem. Instead of a "sequence" model, you convert the time series into a standard tabular (supervised learning) problem.

* How it Works: You must perform Feature Engineering.
  * Target (y): The value at time `t` (e.g., `Sales_Today`).
  * Features (X):
    * Lags: `Sales_Yesterday` (t-1), `Sales_2_Days_Ago` (t-2), etc.
    * Rolling Stats: `7-Day_Moving_Average`, `30-Day_Std_Dev`.
    * Time Features: `Day_of_Week`, `Month`, `Is_Holiday`, `Quarter`.
  * Once you have this table, you can use _any_ standard ML model.
* Models Used: XGBoost (most popular), LightGBM, Random Forest.
* Pros:
  * Non-Linearity: Can easily capture complex, non-linear relationships that ARIMA misses.
  * Multivariate (Its BIGGEST strength): Can _easily_ use hundreds of other variables (exogenous features).
    * Example: Predict `Sales(t)` using `Lags`, `Day_of_Week`, `Competitor_Price(t)`, `Ad_Campaign_Spend(t)`, `Store_Location`.
  * Robust and high-performing: Often wins forecasting competitions.
* Cons:
  * Requires heavy feature engineering: This is the _entire_ challenge. The model is only as good as the features you create. It doesn't "learn" the sequence; it just learns from the features.
  * Loses the sequence: It treats `Lag_1` and `Lag_2` as just two independent columns, not an ordered pair.
  * Poor at Extrapolation: Tree models (like XGBoost) _cannot_ predict a value outside the range they were trained on. If your sales have always been $100-$200, it will never predict $210, even if there's a clear upward trend.

***

#### 3. Deep Learning (Sequence) Models

These models are designed to learn patterns from raw sequential data, automatically learning the features that ML models need to be _given_.

**### RNN / LSTM / GRU**

* How it Works:
  * RNN (Recurrent Neural Network): A basic neural network with a "loop" that maintains a "memory" (hidden state) of past information. (Suffers from short-term memory / vanishing gradients).
  * LSTM & GRU: More advanced RNNs with "gates" that control the memory. They can learn _what_ to remember (long-term) and _what_ to forget, allowing them to capture very long-range dependencies.
* Pros:
  * Natively Sequential: Designed to understand order and time.
  * Learns features automatically: You don't need to manually create lags or rolling averages; the model learns these representations.
  * Non-Linear: Can model highly complex patterns.
  * Stateful: The "memory" is perfect for time series.
* Cons:
  * Data Hungry: Requires _a lot_ of data to outperform classical models.
  * Slow to Train: The recurrent (step-by-step) nature is sequential and hard to parallelize.
  * Complex to Tune: Many hyperparameters (layers, hidden units, etc.) to get right.

**### Transformers**

* How it Works: The new state-of-the-art. Instead of a recurrent loop, Transformers use a Self-Attention Mechanism. This allows the model to look at _all_ time steps at once and "pay attention" to the most relevant past values, even those hundreds of steps in the past.
* Pros:
  * Best-in-class performance on large, complex datasets.
  * Captures Long-Range Dependencies: Far better at this than even LSTMs.
  * Parallelizable: Much faster to train than RNNs (at scale) because it's not sequential.
  * Good at multivariate tasks.
* Cons:
  * Even _more_ data hungry: Needs massive datasets.
  * Very Complex: A "black box" that is hard to tune and interpret.
  * Computational Overkill: Using a Transformer on a simple monthly sales dataset is unnecessary and will likely perform _worse_ than ARIMA.

#### 🚀 How to Choose

| **Model Family** | **When to Use**                                                                                                                                               |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ARIMA / SARIMA   | Your baseline. Start here. Use when you have clear, simple data (e.g., < 5,000 points), a single variable, and clear seasonality/trend.                       |
| Prophet          | When you need a fast, pretty-good, automated forecast with multiple seasonalities (e.g., weekly, yearly) and a list of holidays. Great for business analysts. |
| XGBoost / ML     | When your forecast depends on many other variables (e.g., price, weather, ads) and the relationships are non-linear. This is a powerful, flexible approach.   |
| LSTM / GRU       | When you have lots of data, complex non-linear patterns, and believe long-term memory is key. Good for sensor data, stock volatility, etc.                    |
| Transformer      | When you have a massive, complex dataset (e.g., minute-by-minute financial data, weather forecasting) and need the absolute SOTA performance.                 |
