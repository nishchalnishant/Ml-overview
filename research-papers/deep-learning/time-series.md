# Time series



<table><thead><tr><th width="478">Paper link</th><th>tags</th></tr></thead><tbody><tr><td><a href="https://arxiv.org/abs/2202.03156">Comparative Study of Machine Learning Models for Stock Price Prediction</a></td><td>time seires, stock</td></tr><tr><td><a href="https://arxiv.org/html/2404.08712v1">Machine Learning Methods for Financial Forecasting During Geopolitical Events</a></td><td>time series, Stock</td></tr><tr><td></td><td></td></tr></tbody></table>



***

## <mark style="color:purple;">Comparative Study of Machine Learning Models for Stock Price Prediction</mark>

Here’s a more detailed breakdown of each section of the paper, covering its goals, methods, and findings in stock price prediction using machine learning.

***

#### <mark style="color:red;">**1. Introduction**</mark>

* **Motivation**: The paper begins by discussing the financial incentive behind stock market prediction and the challenges associated with it. A reference is made to Warren Buffett’s million-dollar bet with a hedge fund manager, which showed the difficulty of consistently beating the market. This paper aims to forecast stock prices for educational purposes.
* **Theoretical Background**: Stock prices have historically been viewed as following a “random walk,” implying unpredictability in short-term price changes. The Efficient Market Hypothesis (EMH) argues similarly, stating that future stock prices are unpredictable due to market complexities. The authors, however, take a more practical approach by exploring the applicability of machine learning models on stock price data.
* **Objective**: The study investigates whether machine learning (ML) can accurately predict stock prices. It applies two primary methods suitable for time-series data:
  * A **linear Kalman filter** for low-complexity forecasting.
  * **LSTM architectures** for more advanced, memory-dependent forecasts. The goal is to assess these models’ effectiveness in predicting next-day stock prices.

#### <mark style="color:red;">**2. Methods**</mark>

* **Data**: Historical stock prices (1/1/2011 to 1/1/2021) for stocks from the technology sector (e.g., MSFT, TSLA) and market indices (e.g., NASDAQ) were sourced using the `yfinance` library. TSLA represents a volatile stock, and MSFT represents a non-volatile stock in this study.
* **Kalman Filter**:
  * The Kalman filter is a recursive algorithm often used in time-series prediction to balance measurement and prediction uncertainty. For stock price prediction, the Kalman filter treated the stock’s short-term movement as a random walk.
  * Assumptions include a small variance in the last measured state (the previous day’s stock price) and local variance proportionality of the current state. In practice, the model worked well by using a 3-day local variance.
* **LSTM Models**:
  * The study utilized four LSTM architectures, which are recurrent neural networks (RNNs) optimized for long-term dependencies in data:
    * **Single-layer LSTM**: A basic model with a single memory layer.
    * **Stacked LSTM**: Two or more LSTM layers, allowing deeper learning of data sequences.
    * **Bidirectional LSTM**: Processes data in both forward and backward directions for comprehensive sequence learning.
    * **CNN-LSTM**: Combines convolutional neural networks (CNNs) for feature extraction with LSTM for sequence learning, aiming for enhanced accuracy.

#### <mark style="color:red;">**3. Experiments**</mark>

* **Model Training**: Each model was individually optimized to achieve a balance between simplicity and predictive accuracy.
  * Parameters like node count (64 nodes) and memory size (3) were chosen based on optimization trials.
  * CNN-LSTM involved 1D convolution of stock data to enhance feature extraction before processing with LSTM.
* **Evaluation Metrics**: Models were evaluated on:
  * **Root Mean Square Error (RMSE)**: Lower values indicate better performance.
  * **Mean Absolute Error (MAE)**: Indicates average prediction error, with lower values preferred.
  * **R² value**: Measures fit; values closer to 1 indicate a better fit.
* **Results**:
  * **TSLA (volatile stock)**: The CNN-LSTM model performed best, achieving RMSE of 2.20, MAE of 1.54, and R² of 0.96. Other LSTM models, such as the bidirectional LSTM, also performed well, but the dual-layer LSTM lagged in accuracy, possibly due to increased lag from added complexity.
  * **MSFT (non-volatile stock)**: The Kalman filter and bidirectional LSTM showed excellent performance for non-volatile stocks, with Kalman filter achieving RMSE of 4.78, MAE of 1.19, and R² of 0.99. For non-volatile stocks, simpler models like Kalman filter and bidirectional LSTM provided adequate accuracy without needing complex CNN-LSTM structures.

#### <mark style="color:red;">**4. Applying Trained Models to New Stocks**</mark>

* **Generalization Test**:
  * The study applied trained LSTM models on new data representing different volatility types to assess model generalization.
  * The **bidirectional LSTM and CNN-LSTM** models were applied to S\&P 500 (low volatility) and Russell Microcap Index (high volatility).
  * **Findings**: The models generalized well with R² values close to 0.99, demonstrating that pre-trained LSTM models can effectively predict similarly volatile stocks without additional training. This finding suggests that stock types (low or high volatility) could be clustered, with individual models trained for each category.

#### <mark style="color:red;">**5. Conclusion**</mark>

* **Key Takeaways**:
  * Simple models like the **Kalman filter** are effective for low-volatility stocks, while **LSTM architectures**, particularly the CNN-LSTM, excel at forecasting volatile stocks.
  * **Model Selection by Stock Volatility**: LSTM models are advantageous for high-volatility stocks like TSLA due to their ability to capture complex patterns, while simpler algorithms suffice for stable stocks like MSFT.
  * **Practical Applications**: The results support the potential use of machine learning in automated portfolio management or day-trading algorithms. The authors suggest clustering stocks by volatility to develop specialized predictive models for each category, facilitating automation in finance.
* **Future Work**: Expanding this approach could involve clustering diverse stock types for optimized LSTM models, potentially enabling fully automated portfolio generation based on target return rates.

***

This detailed breakdown shows how the authors tested various machine learning models and their effectiveness in forecasting stock prices, with LSTM-based architectures providing promising results, especially for volatile stocks.

```python
import numpy as np
import pandas as pd
import yfinance as yf
from filterpy.kalman import KalmanFilter
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, Bidirectional
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load the stock data (you can replace 'TSLA' with any stock symbol)
def load_data(ticker='TSLA', start='2011-01-01', end='2021-01-01'):
    data = yf.download(ticker, start=start, end=end)
    data = data[['Close']]
    return data

# Preprocess data for LSTM model
def preprocess_data(data, time_step=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(time_step, len(scaled_data)):
        X.append(scaled_data[i - time_step:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y, scaler

# Kalman filter implementation
def apply_kalman_filter(data):
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.x = np.array([data.iloc[0, 0], 0.])  # initial state (position and velocity)
    kf.F = np.array([[1., 1.], [0., 1.]])   # state transition matrix
    kf.H = np.array([[1., 0.]])             # measurement matrix
    kf.P *= 1000.                           # covariance matrix
    kf.R = 5                                # measurement noise
    kf.Q = 0.1                              # process noise

    kalman_predictions = []
    for price in data['Close']:
        kf.predict()
        kf.update(price)
        kalman_predictions.append(kf.x[0])
    
    return kalman_predictions

# Build LSTM model
def build_lstm_model(input_shape, model_type='single'):
    model = Sequential()
    if model_type == 'single':
        model.add(LSTM(50, return_sequences=False, input_shape=input_shape))
    elif model_type == 'bidirectional':
        model.add(Bidirectional(LSTM(50, return_sequences=False), input_shape=input_shape))
    elif model_type == 'cnn-lstm':
        model.add(Conv1D(64, kernel_size=1, activation='relu', input_shape=input_shape))
        model.add(LSTM(50, return_sequences=False))
    
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Train LSTM model
def train_lstm_model(model, X_train, y_train, epochs=50, batch_size=32):
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1)
    return history

# Plot results
def plot_results(real_data, kalman_pred, lstm_pred, title):
    plt.figure(figsize=(14, 7))
    plt.plot(real_data, label='Real Stock Price')
    plt.plot(kalman_pred, label='Kalman Filter Prediction', linestyle='dashed')
    plt.plot(lstm_pred, label='LSTM Prediction', linestyle='dotted')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

# Main
data = load_data('TSLA')
X_train, y_train, scaler = preprocess_data(data)

# Kalman Filter Prediction
kalman_pred = apply_kalman_filter(data)

# Train LSTM Models
input_shape = (X_train.shape[1], 1)

# Single LSTM model
lstm_model = build_lstm_model(input_shape, model_type='single')
train_lstm_model(lstm_model, X_train, y_train)
lstm_pred = lstm_model.predict(X_train)
lstm_pred = scaler.inverse_transform(lstm_pred)

# CNN-LSTM model
cnn_lstm_model = build_lstm_model(input_shape, model_type='cnn-lstm')
train_lstm_model(cnn_lstm_model, X_train, y_train)
cnn_lstm_pred = cnn_lstm_model.predict(X_train)
cnn_lstm_pred = scaler.inverse_transform(cnn_lstm_pred)

# Plot Results
plot_results(data['Close'], kalman_pred, lstm_pred, 'Kalman vs LSTM (Single Layer) Prediction')
plot_results(data['Close'], kalman_pred, cnn_lstm_pred, 'Kalman vs CNN-LSTM Prediction')

```



***

## <mark style="color:purple;">Machine learning and economic forecasting:</mark>

Certainly! Here’s an in-depth summary that goes through each part of the paper **"Machine Learning and Economic Forecasting: The Role of International Trade Networks"**, detailing the paper's objectives, methodologies, analyses, and findings.

***

#### <mark style="color:red;">Abstract</mark>

This research investigates the impact of international trade networks on economic forecasting, particularly in predicting GDP growth. Using data from 2010 to 2022, it identifies shifts in trade networks driven by increasing trade policy uncertainties. By leveraging supervised machine learning models, the authors demonstrate that trade network-based metrics can improve the accuracy of GDP growth predictions. Non-linear models such as Random Forest, XGBoost, and LightGBM outperform traditional linear models, with network topology descriptors emerging as critical features in refining forecasts.

***

#### <mark style="color:red;">1. Introduction</mark>

The authors discuss the challenges in forecasting economic growth in a complex and interconnected global economy. Traditional econometric models, often based on linear regression and macroeconomic indicators, fall short in capturing the nonlinear relationships and intricacies of global trade interdependencies. This study builds upon two main research areas: **international trade network analysis** and **machine learning for economic forecasting**. The paper innovates by integrating topological metrics from trade networks as features in machine learning models to predict GDP growth, offering a more nuanced understanding of economic interactions and forecasting improvements. This approach is particularly relevant as recent global events, such as the COVID-19 pandemic and geopolitical conflicts, have heightened trade uncertainties.

***

#### <mark style="color:red;">2. Network Analysis of Section-Level International Trade Networks</mark>

**A. Data Description**

The study uses section-level trade data from the United Nations Comtrade database, covering bilateral merchandise trade for nearly 200 countries. The trade data is categorized under the Harmonized System (HS) codes, aggregated to the section level for broader analysis. This categorization allows for detailed analyses while maintaining a manageable dataset. The dataset's monthly data is consolidated to quarterly and annual data for analysis, excluding re-imported and re-exported items to focus on primary trade activities. The five main commodity sections (Mechanical & Electrical, Mineral, Transport, Chemical, and Base Metals) account for over 60% of the trade flow value, representing the largest sections in global trade from 2010 to 2022.

**B. Network Measures**

The authors construct networks where countries are nodes and trade flows are directed edges, weighted by the trade volume. They utilize several network metrics, including:

* **Strength**: Sum of trade flows associated with a node, with in-strength representing imports and out-strength for exports.
* **PageRank**: Centrality measure emphasizing the importance of countries based on import flows.
* **Clustering Coefficient (Transitivity)**: Measures the extent to which trade partners form clusters, indicating the local clustering within networks.
* **Density**: Represents the proportion of actual to potential trade connections, providing a measure of global trade network interconnectedness.
* **Assortativity**: Indicates if countries with similar trade connectivity are linked, showing whether trade interactions are more homogenous or heterogenous.
* **Reciprocity**: Measures the level of mutual trade exchanges.
* **Modularity**: Examines the division of the network into communities, highlighting intra-community trade density versus inter-community connections.

**C. Topological Analysis of Section-Level Trade Networks**

The study observes critical transformations in trade network topology between 2010 and 2022, noting a reversal in trends around 2016-2018. This shift coincides with rising trade uncertainties, influenced by geopolitical events like Brexit, the U.S.-China trade war, and the pandemic. Key observations include:

* **Density**: Initially, trade network density increased until 2017, suggesting more interconnected trade, followed by stagnation or decline, particularly post-2018. The COVID-19 pandemic further exacerbated this decline.
* **Assortativity**: A general trend towards negative assortativity suggests that countries with different levels of trade connections increasingly trade with each other, particularly after 2017.
* **Reciprocity and Clustering**: Both measures indicate growing reciprocal trade partnerships and clustering within trade networks, though clustering declined post-2019.
* **Modularity**: Low modularity values reveal that while countries cluster locally, they are not isolated into distinct trade blocs, indicating fluidity across global trade networks.

***

#### <mark style="color:red;">3. Trade Network Measures as Predictors: A Machine Learning Approach to GDP Growth Forecasting</mark>

The study hypothesizes that including trade network topology measures can improve GDP growth predictions. Two key rationales for this hypothesis are:

1. **Network Position Significance**: A country’s centrality in trade networks could reflect its importance, adaptability, and economic resilience.
2. **Neighboring Economic Conditions**: The economic status of neighboring trade partners can indicate the nation’s economic performance potential.

**Model Selection and Performance**

To test this hypothesis, the authors conduct a "horse race" among various machine learning models: **Linear Regression**, **Elastic Net (Lasso and Ridge regularization)**, **Support Vector Machine (SVM)**, **k-Nearest Neighbor (k-NN)**, **Random Forest**, **XGBoost**, and **LightGBM**. They perform cross-validation and hyperparameter tuning to optimize each model, evaluating them based on **Root Mean Squared Error (RMSE)** and other error metrics. The non-linear models, Random Forest, XGBoost, and LightGBM, emerged as the top-performing models, showing a substantial advantage over linear models.

***

#### <mark style="color:red;">4. Feature Importance Analysis and Model Interpretability with SHAP Values</mark>

**Feature Importance**

Using SHAP values, the authors interpret the feature importance within the top-performing models. They find that network descriptors account for nearly half of the 15 most important features in the Random Forest model, underscoring the predictive power of trade network topology. Key influential features include:

* **Current and recent GDP growth**: These autoregressive features are consistently significant across models, aligning with the concept of economic inertia.
* **Trade Network Density (particularly in Mineral and Chemical sections)**: Trade network density ranks as a critical predictor, where moderate density correlates with positive economic growth predictions, but very high density can negatively impact forecasts.
* **Modularity of the Mechanical & Electrical Network, Population Growth, and Primary Sector’s Role**: These are also among the most influential features, indicating the importance of structural factors within trade networks and demographic trends.

**In-depth Analysis of Random Forest Model**

For the Random Forest model, an additional interpretive layer through SHAP dependence plots reveals:

* **Threshold Effects in Trade Density**: For the Mineral network, moderate density positively influences growth predictions, but an increase beyond a certain threshold correlates with lower growth forecasts.
* **Autoregressive Economic Growth Patterns**: Current and past GDP growth values remain leading indicators, suggesting that recent economic performance has strong predictive power for future growth.
* **Reciprocity and Population Growth**: Higher reciprocity in trade networks and positive population growth values correlate with positive economic outlooks, aligning with conventional economic growth theories.

***

#### <mark style="color:red;">5. Conclusions</mark>

This study contributes a novel approach to economic forecasting by incorporating trade network topology measures into machine learning models. The authors conclude that network-based features significantly enhance the accuracy of economic growth predictions, outperforming traditional models. Non-linear models, especially Random Forest and XGBoost, are more adept at capturing the complexities of trade interactions and economic dynamics.

The paper suggests practical applications of this approach in policymaking, particularly for addressing de-globalization challenges and managing economic uncertainties. Enhanced forecasting models based on trade networks can inform more resilient policies and strategic decisions, benefiting both national and international economic stability.

***

#### <mark style="color:red;">Implications and Future Research</mark>

The paper emphasizes that policymakers need to adopt more sophisticated models that incorporate global trade metrics to navigate de-globalization and economic volatility. Future research could explore more granular trade data, network interactions across other sectors, or additional machine learning techniques to further refine these predictive models.

This comprehensive analysis illustrates that international trade networks are integral to economic forecasting, providing both theoretical and practical advances in understanding global economic dynamics.

