# Supervised Learning

## <mark style="color:purple;">Regression</mark>

* <mark style="color:yellow;">**Algorithms**</mark>
  * Linear regression \[Plane]
* <mark style="color:yellow;">**Metrics**</mark>
  * Mean absolute error \[MAE]
  * Mean squared error \[MSE]
  * Root mean squared error \[RMSE]
  * R squared
  * Adjusted R squared

## <mark style="color:purple;">Classification</mark>

* <mark style="color:yellow;">**Algorithms**</mark>
  * Logistic regression \[Plane]
* <mark style="color:yellow;">**Metrics**</mark>
  * Accuracy
  * Precision
  * Recall
  * F1 scores
  * Confusion matrix
  * ROC curve
  * Area under curve
  * Log loss / logistic loss / cross entropy loss

## Algorithms that can be used for both regression and classification

* Naïve bayes classification \[Probability]
* Support vector machines \[Plane]
* KNN \[Plane]
* Decision tree \[Tree]
* Random forest \[Tree]
*   Ensemble Learning

    * Bagging methods \[bootstrap aggregating]
    * Boosting methods



\############################# Regression ######################

### **Algorithm**

* **Linear regression**
  * Used to predict a continuous value (such as a price or a probability) based on a set of independent variables.
  * Assumes a linear relationship between the input variables and the output variable.
  * Algorithm estimates the coefficient for each input variable which represents the change in the output variable for one unit change in the input variable.
  * These coefficients are learnt from the training data using an optimization technique called gradient descent.
  * **Pros**
    * Relatively simple and easy to implement, making it a good choice for many applications.
    * Can handle large datasets efficiently, since it only requires matrix operations to make predictions.
    * It is interpretable, since the coefficients of the input variables can be used to understand the relative influence of each variable on the output.
    * It performs well on data with a linear relationship between the input and output variables.
  * **Cons**
    * Assumes that the relationship between the input and output variables is linear, which may not be true in all cases.
    * Sensitive to outliers, which can have a large influence on the model if they are not properly handled.
    * Not suitable for modeling complex relationships between the input and output variables.
    * Does not handle categorical variables well, and requires them to be encoded as numerical values before they can be used as input features.
  * **Most suitable**
    * Most suitable when the relationship between the input and output variables is linear, meaning that the change in the output variable is proportional to the change in the input variables.
    * Simple and fast algorithm that can handle large datasets efficiently and is easy to interpret, making it a good choice for many applications.
    * Linear regression is also a good choice when the goal is to understand the relative influence of each input variable on the output.
    * The coefficients of the input variables can be used to understand how each variable impacts the output, which can be useful for feature selection or for understanding the underlying relationships in the data.
    * However, linear regression is limited by the assumption that the relationship between the input and output variables is linear. If the relationship is more complex, a non-linear model such as a decision tree or a neural network may be more appropriate.
    * Linear regression is also sensitive to outliers and may not handle categorical variables well, requiring them to be encoded as numerical values before they can be used as input features.
  *   Maths



      **The Mathematical Model** The basic linear regression model is represented by the equation:

      ```
      Y = β0 + β1X + ε
      ```

      Where:

      * **Y:** The dependent variable (the variable we want to predict).
      * **X:** The independent variable (the variable used to make the prediction).
      * **β0:** The intercept (the value of Y when X is 0).
      * **β1:** The slope (the rate of change of Y with respect to X).
      * **ε:** The error term (the difference between the actual Y value and the predicted Y value).



      **Finding the Best-Fitting Line**

      The key to linear regression is finding the values of β0 and β1 that minimize the sum of the squared errors (also known as the residual sum of squares or RSS). This is often done using the method of least squares.

      *   **Least Squares Method**

          The least squares method involves finding the values of β0 and β1 that minimize the following equation:

          ```
          RSS = Σ(Y - ŷ)²
          ```

          Where:

          * **ŷ:** The predicted value of Y.
            * **Σ:** The sum of the squared differences between the actual Y values and the predicted Y values.
          * **Multiple Linear Regression**

          Multiple linear regression extends the basic model to include multiple independent variables:

          ```
          Y = β0 + β1X1 + β2X2 + ... + βnXn + ε
          ```

          Where:

          * **X1, X2, ..., Xn:** The independent variables.
          * **β1, β2, ..., βn:** The coefficients for each independent variable.

          **Key Concepts**

          * **Correlation:** Measures the strength and direction of the linear relationship between two variables.
          * **R-squared:** A statistical measure that represents the proportion of the variance in the dependent variable that is explained by the independent variable(s).2
          * **Residuals:** The differences between the actual Y values and the predicted Y values.
          * **Assumptions of Linear Regression:**
            * Linearity: The relationship between the variables is linear.
            * Independence: The observations are independent of each other.
            * Homoscedasticity: The variance of the3 residuals is constant across all values of the independent4 variable(s).
            * Normality: The residuals are normally distributed.



{% code overflow="wrap" fullWidth="true" %}
```python
import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # Initialize weights and bias
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient Descent
        for _ in range(self.n_iters):
            y_predicted = np.dot(X, self.weights) + self.bias
            # Calculate gradients
            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted - y)
            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        y_predicted = np.dot(X, self.weights) + self.bias
        return y_predicted

# Example Usage
if __name__ == "__main__":
    # Sample data
    X = np.array([[1, 2], [2, 4], [3, 6], [4, 8], [5, 10]])
    y = np.array([3, 5, 7, 9, 11])

    # Create and train the model
    model = LinearRegression()
    model.fit(X, y)

    # Make predictions
    X_new = np.array([[6, 12]])
    y_pred = model.predict(X_new)
    print("Prediction:", y_pred)
```
{% endcode %}



### <mark style="color:purple;">Metrics</mark>

* <mark style="color:yellow;">**Mean absolute error \[MAE]**</mark>
  *   **Where**:

      * ( N ) is the total number of observation or data points
        * ( O\_i ) is the actual or observed value
        * ( P\_i ) is the predicted value
      * Measures the average absolute difference between predicted and actual values.
      * It gives you a sense of how far off your predictions are from the actual values on average.
      * Unlike Mean Squared Error (MSE), which squares the differences, MAE treats all differences equally, regardless of whether they are overestimations or underestimations.
      * MAE is less sensitive to outliers compared to MSE because it doesn't amplify large errors.



      *   **Mathematical Definition:**

          MAE is calculated as the average of the absolute differences between the predicted values and the actual values. Mathematically,1 it's represented as:

          ```
          MAE = (1/n) * Σ|y_i - ŷ_i| 
          ```

          Where:

          * `n`: is the number of data points
          * `y_i`: is the actual value of the i-th data point
          * `ŷ_i`: is the predicted value of the i-th data point2
          * `Σ`: denotes the summation over all data points

          **Key Points:**

          * **Absolute Value:** The use of the absolute value function (`|...|`) ensures that all errors are treated as positive, regardless of whether the prediction is overestimated or underestimated. This makes MAE less sensitive to outliers compared to Mean Squared Error (MSE).
          * **Average Error:** MAE provides a single, interpretable value that represents the average magnitude of the errors made by the model.
          * **Scale-Dependent:** MAE is measured in the same units as the target variable. This makes it easier to understand the practical significance of the errors.

          **Example:**

          Let's say we have the following actual values and predicted values:

          | Actual (y\_i) | Predicted (ŷ\_i) |
          | ------------- | ---------------- |
          | 5             | 4                |
          | 8             | 9                |
          | 2             | 3                |
          | 10            | 8                |

          Calculating MAE:

          1. Calculate the absolute differences: |5 - 4| = 1, |8 - 9| = 1, |2 - 3| = 1, |10 - 8| = 2
          2. Sum the absolute differences: 1 + 1 + 1 + 2 = 5
          3. Divide by the number of data points: 5 / 4 = 1.25

          Therefore, the MAE for this example is 1.25.

          **In essence, MAE provides a straightforward and robust measure of the average error magnitude in a regression model, making it a valuable tool for evaluating model performance.**


*   <mark style="color:yellow;">**Mean squared error \[MSE]**</mark>

    *
    * **Where**:
      * ( N ) is the total number of data points
      * ( Y\_i ) is the actual value of the dependent variables for the ith data point
      * ( \hat{Y\_i} ) is the predicted value of the dependent variable for the ith data point
    * Commonly used metric in regression analysis that measures the average of the squared differences between predicted and actual values.
    * Provides a measure of how well a regression model is able to approximate the relationship between the independent and dependent variables.
    *   Here's what MSE represents:

        * It quantifies the average of the squared differences between predicted and actual values. Larger errors contribute more to the metric due to the squaring operation.
        * It provides a way to penalize large errors more heavily than smaller ones.
        * It is always a non-negative value, with lower values indicating better model performance.


    *   **Mathematical Definition:**

        MSE is calculated as the average of the squared differences between the predicted values and the actual values. Mathematically, it's represented as:

        ```
        MSE = (1/n) * Σ(y_i - ŷ_i)^2
        ```

        Where:

        * `n`: is the number of data points
        * `y_i`: is the actual value of the i-th data point
        * `ŷ_i`: is the predicted value of the i-th data point1
        * `Σ`: denotes the summation over all data points

        **Key Points:**

        * **Squared Differences:** The squaring of the differences between actual and predicted values emphasizes larger errors, making MSE more sensitive to outliers compared to Mean Absolute Error (MAE).
        * **Average Error:** MSE provides a single, interpretable value representing the average squared error of the model's predictions.
        * **Scale-Dependent:** MSE is measured in squared units of the target variable, which might not be directly interpretable in the same context as the original data.

        **Example:**

        Let's say we have the following actual values and predicted values:

        | Actual (y\_i) | Predicted (ŷ\_i) |
        | ------------- | ---------------- |
        | 5             | 4                |
        | 8             | 9                |
        | 2             | 3                |
        | 10            | 8                |

        Calculating MSE:

        1. Calculate the squared differences: (5 - 4)^2 = 1, (8 - 9)^2 = 1, (2 - 3)^2 = 1, (10 - 8)^2 = 4
        2. Sum the squared differences: 1 + 1 + 1 + 4 = 7
        3. Divide by the number of data points: 7 / 4 = 1.75

        Therefore, the MSE for this example is 1.75.

        **In essence, MSE provides a measure of the average squared error in a regression model, emphasizing larger errors. It's commonly used as a loss function in linear regression due to its mathematical properties that make it easier to optimize.**


*   <mark style="color:yellow;">**Root mean squared error \[RMSE]**</mark>

    *
    * **Where**:
      * ( N ) is the total number of data points
      * ( Y\_i ) is the actual value of the dependent variable for the ith data point
      * ( \hat{Y\_i} ) is the predicted value of the dependent variable for the ith data point
    * Widely used metric in regression analysis that provides a measure of the average magnitude of the error between predicted and actual values.
    * It provides an error measure in the same units as the target variable, making it easy to interpret.
    * It quantifies the typical or root average error between the predicted and actual values.



    *   **Root Mean Squared Error (RMSE)**

        **Mathematical Definition:**

        RMSE is the square root of the average of the squared differences between the predicted values and the actual values. Mathematically,1 it's represented as:

        ```
        RMSE = √[(1/n) * Σ(y_i - ŷ_i)^2]
        ```

        Where:

        * `n`: is the number of data points
        * `y_i`: is the actual value of the i-th data point
        * `ŷ_i`: is the predicted value of the i-th data point2
        * `Σ`: denotes the summation over all data points

        **Key Points:**

        * **Squared Differences:** Similar to MSE, RMSE emphasizes larger errors due to the squaring of the differences.
        * **Square Root:** Taking the square root of the mean squared error brings the units of RMSE back to the same units as the target variable, making it more interpretable.
        * **Sensitivity to Outliers:** Like MSE, RMSE is sensitive to outliers.

        **Example:**

        Let's use the same example as in the MSE explanation:

        | Actual (y\_i) | Predicted (ŷ\_i) |
        | ------------- | ---------------- |
        | 5             | 4                |
        | 8             | 9                |
        | 2             | 3                |
        | 10            | 8                |

        Calculating RMSE:

        1. Calculate the squared differences: (5 - 4)^2 = 1, (8 - 9)^2 = 1, (2 - 3)^2 = 1, (10 - 8)^2 = 4
        2. Sum the squared differences: 1 + 1 + 1 + 4 = 7
        3. Divide by the number of data points: 7 / 4 = 1.75
        4. Take the square root: √1.75 ≈ 1.32

        Therefore, the RMSE for this example is approximately 1.32.

        **In essence, RMSE provides a measure of the average error magnitude in the same units as the target variable, making it a widely used metric for evaluating the performance of regression models.**


* **R&#x20;**<mark style="color:yellow;">**squared**</mark>
  *
  * **Where**:
    * ( SSR ) is the sum of squared residuals (the differences between predicted and actual values).
      * **Residuals**: The term "residual" refers to the difference between the observed value and the predicted value for a specific data point. Squaring and summing these residuals gives us the SSR.
    * ( SST ) is the total sum of squares, which measures the total variability in the dependent variable.
      * Represents the total variation in the dependent variable (or response variable) before accounting for the variation explained by the independent variables.
  * A statistical measure used in regression analysis to evaluate the goodness of fit of a model to the data.
  * It provides insights into how well the model explains the variance in the dependent variable.
  * It should be used in conjunction with other metrics and domain knowledge to gain a comprehensive understanding of the model's performance.
  * **R-squared** is a value between 0 and 1, where:
    * 0: The model does not explain any of the variability in the dependent variable.
    * 1: The model perfectly explains the variability in the dependent variable.
  * **What R-squared represents**:
    * It measures the proportion of the variance in the dependent variable that is predictable from the independent variable(s) in the model.
    * A higher R-squared value indicates a better fit, meaning that a larger proportion of the variability in the dependent variable is explained by the model.
  *   **R-Squared limitations**:

      * It can be artificially inflated by adding more independent variables to the model, even if they don't have a meaningful relationship with the dependent variable.
      * It doesn't indicate the causality of relationships; it only measures the strength of association.
      * It may not be the best metric for models with non-linear relationships or complex patterns.


  *   <mark style="color:yellow;">**R-squared (R²)**</mark>

      R-squared is a statistical measure that represents the proportion of the variance in the dependent variable that is predictable from the independent variable(s).1 In simpler terms, it tells us how well our regression model fits the observed data.

      **Mathematical Definition:**

      R-squared is calculated as:

      ```
      R² = 1 - (SSR / SST)
      ```

      Where:

      *   **SSR (Sum of Squared Residuals):** The sum of the squared differences between the actual values (y\_i) and the predicted values (ŷ\_i). It represents the unexplained variance.

          ```
          SSR = Σ(y_i - ŷ_i)²
          ```
      *   **SST (Total Sum of Squares):** The sum of the squared differences between the actual values (y\_i) and the mean of the actual values (ȳ). It represents the total variance.

          ```
          SST = Σ(y_i - ȳ)²
          ```

      **Interpretation:**

      * **R² ranges from 0 to 1:**
        * **R² = 0:** The model explains none of the variability in the dependent variable.
        * **R² = 1:** The model explains all of the variability in the dependent variable.2
      * **Higher R² values generally indicate a better fit:** A higher R² suggests that the model is better at predicting the dependent variable based on the independent variables.

      **Example:**

      Let's say we have the following actual values (y\_i) and predicted values (ŷ\_i):

      | Actual (y\_i) | Predicted (ŷ\_i) |
      | ------------- | ---------------- |
      | 5             | 4                |
      | 8             | 9                |
      | 2             | 3                |
      | 10            | 8                |

      And the mean of the actual values (ȳ) is 6.25.

      1. **Calculate SSR:**
         * (5 - 4)² = 1
         * (8 - 9)² = 1
         * (2 - 3)² = 1
         * (10 - 8)² = 4
         * SSR = 1 + 1 + 1 + 4 = 7
      2. **Calculate SST:**
         * (5 - 6.25)² = 1.5625
         * (8 - 6.25)² = 3.0625
         * (2 - 6.25)² = 17.5625
         * (10 - 6.25)² = 14.0625
         * SST = 1.5625 + 3.0625 + 17.5625 + 14.0625 = 36.25
      3. **Calculate R²:**
         * R² = 1 - (SSR / SST) = 1 - (7 / 36.25) ≈ 0.807

      Therefore, the R² for this example is approximately 0.807, which means that the model explains about 80.7% of the variability in the dependent variable.

      **In essence, R-squared provides a valuable metric for assessing the goodness-of-fit of a regression model by quantifying the proportion of variance explained by the model.**
*   <mark style="color:yellow;">**Adjusted R squared**</mark>

    * **Where**:
    * ( R^2 ) is the standard R-value
    * ( N ) is the number of observations or data points
    * ( K ) is the number of independent variables in the model
    * Modified version of the standard R-squared (coefficient of determination) that takes into account the number of independent variables in a regression model.
    * It is particularly useful when comparing models with different numbers of predictors.
    * Here's what adjusted R-squared represents:
      * It provides a more accurate measure of the model's goodness of fit, especially when comparing models with different numbers of predictors.
      * It penalizes the addition of unnecessary independent variables to the model. As ( K ) increases, the penalty term ( \frac{(N-1)}{(N-K-1)} ) decreases, which means that Adjusted R-squared becomes closer to R-squared.
    * The main difference between R-squared and Adjusted R-squared is that the latter takes into account the complexity of the model. It provides a more conservative estimate of the proportion of variance explained by the independent variables.
    * While Adjusted R-squared is a useful metric, it should be used in conjunction with other evaluation criteria and domain knowledge to make informed decisions about model selection and performance.



    *   Maths --

        R-squared is a valuable metric for assessing the goodness-of-fit of a regression model, but it has a limitation: it tends to increase as you add more independent variables to the model, even if those variables don't actually improve the model's predictive power. This can lead to overfitting, where the model performs well on the training data but poorly on new, unseen data.

        **Adjusted R-squared addresses this limitation by penalizing the model for including irrelevant predictors.**

        **Mathematical Definition:**

        ```
        Adjusted R² = 1 - [(1 - R²) * (n - 1) / (n - k - 1)]
        ```

        Where:

        * `R²`: is the regular R-squared value
        * `n`: is the number of data points
        * `k`: is the number of independent variables in the model1

        **Key Points:**

        * **Penalizes for Irrelevant Predictors:** The adjusted R-squared value will only increase if the new predictor improves the model's fit more than would be expected by chance. If a predictor doesn't significantly improve the model, the adjusted R-squared may actually decrease.
        * **More Conservative:** Adjusted R-squared is generally more conservative than regular R-squared, providing a more realistic assessment of the model's true predictive power.
        * **Useful for Model Comparison:** Adjusted R-squared is particularly useful when comparing models with different numbers of predictors. It helps to avoid selecting models that simply have more variables but don't necessarily have better predictive performance.

        **In essence, adjusted R-squared provides a more reliable measure of the goodness-of-fit of a regression model, especially when dealing with multiple predictors, by accounting for the number of predictors and preventing overfitting.**

        **Example:**

        Let's say you have two models:

        * **Model A:** R-squared = 0.8, 3 independent variables
        * **Model B:** R-squared = 0.85, 5 independent variables

        While Model B has a slightly higher R-squared, its adjusted R-squared might be lower than Model A's if the two additional variables don't significantly improve the model's predictive power. In this case, the adjusted R-squared would suggest that Model A is a better fit despite having fewer predictors.



\######################### Classification ############################

## <mark style="color:purple;">Algorithm</mark>

### <mark style="color:yellow;">Logistic Regression</mark>

* Logistic regression is a supervised learning algorithm used to predict a binary outcome (such as a yes/no or 0/1 response) based on a set of independent variables.
* It is similar to linear regression, but the output is transformed using a logistic function to ensure it is always between 0 and 1.
* To perform logistic regression, the algorithm estimates the coefficients for each input variable, representing the change in the output variable for a one-unit change in the input variable.
* These coefficients are learned from the training data using an optimization technique called gradient descent.
* Once the model is trained, it can make predictions on new data by plugging in the input values and using the learned coefficients to calculate the predicted output value.
* The predicted output is transformed using the logistic function to give a probability of the positive class. This probability can be thresholded to obtain a binary prediction.
* It is especially useful for cases where the output is binary, but can also be used for multiclass classification by training multiple logistic regression models, one for each class.
* However, logistic regression is limited by the assumption that the relationship between the input and output variables is linear.
* In cases where the relationship is more complex, a non-linear model such as a decision tree or a neural network may be more appropriate.

#### Pros

* Logistic regression is a simple and widely used algorithm effective for many classification tasks.
* It is fast to train and can handle large datasets efficiently.
* It is interpretable, as the coefficients of the input variables can help understand the relative influence of each variable on the output.
* It can handle multiclass classification by training multiple logistic regression models, one for each class.

#### Cons

* Assumes the relationship between the input and output variables is linear, which may not be true in all cases.
* Sensitive to outliers, which can significantly influence the model if not properly handled.
* Not suitable for modeling complex relationships between input and output variables.
* Does not handle categorical variables well; these must be encoded as numerical values before use as input features.

Maths

*   **Logistic Regression: A Deep Dive into the Mathematics**

    **1. The Core Idea**

    Logistic regression is a powerful statistical method used to model the probability of a binary outcome (e.g., success/failure, yes/no) based on one or more predictor variables. It achieves this by transforming the linear combination of predictors into a probability value between 0 and 1.

    **2. The Logistic Function**

    The heart of logistic regression lies in the logistic function (also known as the sigmoid function):

    ```
    P(y = 1 | x) = 1 / (1 + exp(-z))
    ```

    Where:

    * `P(y = 1 | x)`: The probability of the outcome being 1 (positive class) given the input features `x`.
    * `z`: The linear combination of predictors and their coefficients: `z = b0 + b1*x1 + b2*x2 + ... + bn*xn`

    The logistic function maps any real-valued input `z` to a value between 0 and 1, making it suitable for representing probabilities.

    **3. Estimating the Coefficients**

    The primary goal in logistic regression is to estimate the coefficients (b0, b1, b2, ..., bn) that best fit the data. This is typically done using the maximum likelihood estimation (MLE) method.

    **4. Maximum Likelihood Estimation (MLE)**

    MLE aims to find the coefficients that maximize the likelihood of observing the given data. The likelihood function for logistic regression is:

    ```
    L(b0, b1, ..., bn) = Π [P(y_i = 1 | x_i)]^y_i * [1 - P(y_i = 1 | x_i)]^(1-y_i)
    ```

    Where:

    * `y_i`: The actual class label of the i-th data point (0 or 1)
    * `x_i`: The input features of the i-th data point

    To find the coefficients that maximize this likelihood, we often work with the log-likelihood function, which is easier to optimize:

    ```
    log L(b0, b1, ..., bn) = Σ [y_i * log(P(y_i = 1 | x_i)) + (1 - y_i) * log(1 - P(y_i = 1 | x_i))]
    ```

    **5. Optimization Algorithms**

    Various optimization algorithms can be used to find the coefficients that maximize the log-likelihood, such as:

    * **Gradient Descent:** An iterative algorithm that adjusts the coefficients in the direction of the steepest ascent of the log-likelihood function.
    * **Newton-Raphson Method:** A more advanced optimization algorithm that can converge faster than gradient descent.

    **6. Making Predictions**

    Once the coefficients are estimated, we can use the logistic function to predict the probability of the positive class for new data points. A common decision rule is to classify a data point as positive if the predicted probability is greater than 0.5.

    **7. Key Points**

    * Logistic regression is a powerful tool for binary classification problems.
    * The logistic function transforms the linear combination of predictors into probabilities between 0 and 1.
    * MLE is commonly used to estimate the coefficients.
    * Various optimization algorithms can be employed to find the optimal coefficients.
    * The decision boundary in logistic regression is typically a linear hyperplane.

    **In summary, logistic regression provides a robust framework for modeling the relationship between predictor variables and a binary outcome. By understanding the underlying mathematics, you can gain a deeper appreciation for its strengths and limitations.**
* **Example --**
  *   **1. Define the Logistic Regression Model**

      * **Hypothesis:**
        * `h_θ(x) = 1 / (1 + exp(-θ^T x))`
        * Where:
          * `h_θ(x)`: Predicted probability of the positive class (between 0 and 1)
          * `θ`: Vector of model parameters (weights)
          * `x`: Input feature vector (including a bias term)
          * `θ^T x`: Dot product of the weight vector and the feature vector

      **2. Define the Cost Function**

      * **Log-Loss (Cross-Entropy Loss):** A common cost function for logistic regression
        * `J(θ) = -(1/m) * Σ [y_i * log(h_θ(x_i)) + (1 - y_i) * log(1 - h_θ(x_i))]`
        * Where:
          * `m`: Number of training examples
          * `y_i`: Actual class label (0 or 1) of the i-th example
          * `h_θ(x_i)`: Predicted probability for the i-th example

      **3. Calculate the Gradient of the Cost Function**

      * **Partial Derivative:**
        * Calculate the partial derivative of the cost function with respect to each weight parameter (θ\_j). This gives the direction of steepest ascent of the cost function.

      **4. Gradient Descent Algorithm**

      * **Initialize Parameters:**
        * Start with random initial values for the weight vector (θ).
      * **Iterative Updates:**
        * **Calculate Gradient:** Compute the gradient of the cost function using the current parameter values.
        * **Update Weights:**
          * `θ_j := θ_j - α * ∂J(θ) / ∂θ_j`
          * Where:
            * `α`: Learning rate (a small constant)
            * `∂J(θ) / ∂θ_j`: Partial derivative of the cost function with respect to the j-th weight
      * **Repeat:**
        * Repeat the gradient calculation and weight update steps until the cost function converges (stops decreasing significantly) or a maximum number of iterations is reached.

      **Example (Simplified)**

      Let's consider a simple logistic regression model with two features (x1 and x2) and a bias term.

      1. **Initialize:**
         * `θ = [0, 0, 0]` (initial values for bias and two weights)
         * `α = 0.01` (learning rate)
      2. **Iterate:**
         * For each training example (x\_i, y\_i):
           * Calculate `h_θ(x_i)`
           * Calculate the gradient for each weight (∂J(θ) / ∂θ\_j)
           * Update weights: `θ_j = θ_j - α * ∂J(θ) / ∂θ_j`
         * Repeat for a specified number of iterations or until convergence.

      **Note:**

      * This is a simplified illustration. In practice, you would typically use libraries like scikit-learn in Python to implement and train logistic regression models.
      * The actual gradient calculation for logistic regression involves the sigmoid function and can be derived using calculus.

      This example demonstrates the core idea of using gradient descent to find the optimal parameters for a logistic regression model. By iteratively adjusting the weights in the direction that minimizes the cost function, gradient descent helps the model learn to accurately predict the probability of the positive class for new data.



{% code overflow="wrap" fullWidth="true" %}
```python
import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def loss(self, y, y_pred):
        return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

    def gradient_descent(self, X, y, y_pred):
        dw = (1/len(y)) * np.dot(X.T, (y_pred - y))
        db = (1/len(y)) * np.sum(y_pred - y)
        return dw, db

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_model)
            dw, db = self.gradient_descent(X, y, y_pred)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear_model)
        y_pred_class = [1 if i > 0.5 else 0 for i in y_pred]
        return np.array(y_pred_class)

# Example Usage
if __name__ == "__main__":
    # Sample data (replace with your own data)
    X = np.array([[1, 2], [2, 4], [3, 6], [4, 8], [5, 10]])
    y = np.array([0, 0, 0, 1, 1]) 

    # Create and train the model
    model = LogisticRegression()
    model.fit(X, y)

    # Make predictions
    X_new = np.array([[6, 12]])
    y_pred = model.predict(X_new)
    print("Predictions:", y_pred)
```
{% endcode %}

## <mark style="color:purple;">Metrics</mark>

### <mark style="color:yellow;">Accuracy</mark>

* Accuracy = Total Number of Predictions / Number of Correct Predictions
* A commonly used metric for evaluating classification models.
* Measures the proportion of correctly classified instances out of the total number of instances in a dataset.

#### Limitations of Accuracy

* **Imbalanced Classes**: In datasets where classes are imbalanced (i.e., one class dominates), high accuracy may not indicate a good model. Other metrics like precision, recall, or F1-score may be more informative.
* **Misleading in Skewed Datasets**: In cases where one class dominates the dataset, a model that simply predicts the majority class may achieve high accuracy, even though it’s not performing well.
* **Doesn’t Consider Type of Errors**: Accuracy treats false positives and false negatives equally, though they may have different real-world consequences.
* **Not Suitable for Continuous Predictions**: Accuracy is designed for classification tasks, where predictions are discrete class labels. It's not appropriate for regression tasks, where predictions are continuous.

### <mark style="color:yellow;">Ways to Remember TP, TN, FP, FN</mark>

* **True Positive (TP)**:
  * Mnemonic: "Detective's Success"
  * A true positive is when the detective correctly identifies a suspect as guilty.
* **True Negative (TN)**:
  * Mnemonic: "Innocent Bystander"
  * A true negative is when the detective correctly identifies a suspect as innocent.
* **False Positive (FP)**:
  * Mnemonic: "Misguided Accusation"
  * A false positive is when the detective wrongly accuses an innocent person.
* **False Negative (FN)**:
  * Mnemonic: "Missed Culprit"
  * A false negative is when the detective fails to identify a guilty suspect.

### <mark style="color:yellow;">Precision</mark>

* Precision = (True Positives) / (True Positives + False Positives)
* Precision measures the accuracy of positive predictions made by a model.
* It focuses on the accuracy of positive predictions, which is crucial when the cost of false positives is high (e.g., in medical diagnoses).
* A high precision indicates that the model is good at avoiding false positives but may still have false negatives.

### <mark style="color:yellow;">Recall</mark>

* Recall = True Positives / (True Positives + False Negatives)
* Also known as sensitivity or true positive rate.
* Recall measures the model’s ability to identify all relevant instances of a certain class.
* High recall indicates the model is good at identifying positive instances, even if it means accepting a higher rate of false positives.

### <mark style="color:yellow;">F1 Score</mark>

* Combines precision and recall into a single value.
* F1 Score = 2 × (Precision × Recall) / (Precision + Recall)
* The F1 score provides a balance between false positives (precision) and false negatives (recall).
* Range: 0-1, where a higher value indicates better model performance. 1 means both precision and recall are perfect.

### <mark style="color:yellow;">Confusion Matrix</mark>

* A confusion matrix describes the performance of a classification model on a dataset with known true values.

|                    | Actual Positive     | Actual Negative     |
| ------------------ | ------------------- | ------------------- |
| Predicted Positive | True Positive (TP)  | False Positive (FP) |
| Predicted Negative | False Negative (FN) | True Negative (TN)  |

* Useful for understanding the types of errors the model is making.
* From the confusion matrix, various metrics like accuracy, precision, recall, F1-score, etc., can be calculated.

### <mark style="color:yellow;">ROC Curve</mark>

* The Receiver Operating Characteristic (ROC) curve is a graphical representation of a binary classification model's performance across different thresholds.
* It helps visualize the trade-off between the true positive rate (sensitivity) and the false positive rate (1 - specificity).

### <mark style="color:yellow;">Area Under Curve (AUC)</mark>

* The Area Under the ROC Curve (AUC-ROC) quantifies the overall ability of the model to distinguish between the two classes.
* A higher AUC indicates better model performance, with a maximum value of 1 for a perfect classifier.

### <mark style="color:yellow;">Log Loss / Cross Entropy Loss</mark>

* Log Loss measures the accuracy of predicted probabilities compared to the actual probabilities.
* The formula penalizes confident and incorrect predictions more heavily. Predicting a low probability for the actual class incurs a small loss, while predicting a high probability for the actual class incurs a much larger loss.
* Lower Log Loss values indicate better model performance.



\################  both regression and classification ############################

## <mark style="color:purple;">Naive Bayes</mark>

* Naive Bayes is a probabilistic algorithm used for classification and prediction tasks.
* It is based on **Bayes' theorem**, which relates the probability of a hypothesis (or event) to the probabilities of the evidence (or observations) that support it.
* In classification, Naive Bayes assumes that each feature is independent of all other features, given the class label. This "naive" assumption simplifies the calculation of probabilities, making the algorithm computationally efficient.
* The algorithm computes the **posterior probability** of each class label given the observed features, selecting the class label with the highest probability as the predicted class.



*   **Naive Bayes: A Mathematical Dive**

    **1. Core Concept**

    Naive Bayes is a classification algorithm based on Bayes' Theorem. It makes a **strong assumption** that the features used to predict the class are **independent of each other**. This "naive" assumption simplifies the calculations significantly.

    **2. Bayes' Theorem**

    The foundation of Naive Bayes lies in Bayes' Theorem:

    ```
    P(C|X) = [P(X|C) * P(C)] / P(X)
    ```

    Where:

    * **P(C|X):** Posterior probability - The probability of class C given the observed features X.
    * **P(X|C):** Likelihood - The probability of observing features X given that the class is C.
    * **P(C):** Prior probability - The probability of class C occurring.
    * **P(X):** Probability of observing features X.

    **3. Naive Bayes Assumption**

    The "naive" part comes from the assumption that all features are independent given the class:

    ```
    P(X|C) = P(x1|C) * P(x2|C) * ... * P(xn|C)
    ```

    Where:

    * `X` is a vector of features: `X = (x1, x2, ..., xn)`

    **4. Putting it Together**

    Combining Bayes' Theorem and the independence assumption, we get the Naive Bayes classification rule:

    ```
    P(C|X) ∝ P(C) * P(x1|C) * P(x2|C) * ... * P(xn|C)
    ```

    The classifier assigns a class to a new data point by selecting the class `C` that maximizes this probability.

    **5. Example**

    Let's say we want to classify an email as spam or not spam based on the presence of certain words.

    * **Features (X):** "money," "free," "urgent"
    * **Class (C):** "spam" or "not spam"

    Naive Bayes calculates:

    * `P(spam)`: Prior probability of an email being spam.
    * `P(money|spam)`, `P(free|spam)`, `P(urgent|spam)`: Probability of these words occurring in spam emails.
    * `P(not spam)`, `P(money|not spam)`, `P(free|not spam)`, `P(urgent|not spam)`: Probabilities for non-spam emails.

    Then, it calculates `P(spam|X)` and `P(not spam|X)` and assigns the class with the higher probability.

    **Key Points**

    * Naive Bayes is a simple yet effective algorithm, especially for text classification.
    * The independence assumption is a simplification, but it often works well in practice.
    * Naive Bayes can be used for both categorical and continuous features.

    **In essence, Naive Bayes leverages Bayes' Theorem and the independence assumption to efficiently classify data based on the probability of features given different classes.**

#### Bayes' Theorem

The posterior probability can be expressed using Bayes' theorem as:

\[ P(y|x\_1, x\_2, ..., x\_n) = \frac{P(x\_1, x\_2, ..., x\_n|y) P(y)}{P(x\_1, x\_2, ..., x\_n)} ]

where:

* ( y ) is the class label
* ( x\_1, x\_2, ..., x\_n ) are the observed features.

#### Key Concepts

* **Prior Probability (P(B)):** This is our initial belief or knowledge about the probability of event B before considering any new evidence.
* **Likelihood (P(A|B)):** This represents the probability of observing the evidence (event A) if the hypothesis (event B) is true.
* **Posterior Probability (P(B|A)):** This is the updated probability of event B after considering the new evidence (event A).

#### Types of Naive Bayes

1. **Bernoulli Naive Bayes**:
   * Used for binary features (e.g., presence/absence of words).
2. **Multinomial Naive Bayes**:
   * Suitable for discrete features (e.g., accounting for the frequency of words, such as TF-IDF - term frequency-inverse document frequency).
3. **Gaussian Naive Bayes**:
   * For continuous/real-valued features (it stores the average value and standard deviation of each feature for each class).

#### Estimation Techniques

* The probabilities can be estimated from training data using:
  * **Maximum Likelihood Estimation**: Probabilities are estimated by counting occurrences of each feature and feature-class combination.
  * **Bayesian Estimation**: Assumes a prior probability distribution for parameters, and the probabilities are estimated using the posterior distribution based on observed data.

#### Naive Bayes Variants

* **Gaussian Naive Bayes**: Assumes continuous features following a Gaussian distribution.
* **Multinomial Naive Bayes**: Assumes discrete features following a multinomial distribution.
* **Bernoulli Naive Bayes**: Assumes binary features.

#### Pros of Naive Bayes

* Provides a way to update the probability of a hypothesis based on new evidence.
* Effective for large datasets and high-dimensional feature spaces.
* Incorporates prior knowledge or beliefs to improve accuracy.
* Provides a framework for decision-making and risk analysis across various fields (e.g., medicine, finance).
* Used widely in machine learning algorithms (e.g., Bayesian networks, Bayesian optimization).

#### Cons of Naive Bayes

* Requires specification of prior probabilities, which may be subjective and hard to estimate.
* Assumes independence of features, which may not hold in real-world data.
* Computationally expensive, requiring advanced techniques like **Markov Chain Monte Carlo (MCMC)** to approximate posterior distributions.
* Sensitive to the choice of prior distribution and hyperparameters.
* May struggle with high-dimensional data or complex models where posterior distribution cannot be computed analytically.

{% code fullWidth="true" %}
```python
import numpy as np

class NaiveBayes:
    def __init__(self, smoothing=1.0):
        self.smoothing = smoothing
        self.priors = {}
        self.likelihoods = {}

    def fit(self, X, y):
        """
        Trains the Naive Bayes model.

        Args:
            X: Training data features (numpy array).
            y: Training data labels (numpy array).
        """
        n_samples, n_features = X.shape
        classes = np.unique(y)

        # Calculate prior probabilities
        for cls in classes:
            self.priors[cls] = np.sum(y == cls) / n_samples

        # Calculate likelihoods (conditional probabilities)
        self.likelihoods = {}
        for cls in classes:
            self.likelihoods[cls] = {}
            X_cls = X[y == cls]
            for feature in range(n_features):
                values, counts = np.unique(X_cls[:, feature], return_counts=True)
                total_counts = len(X_cls)
                self.likelihoods[cls][feature] = {}
                for value, count in zip(values, counts):
                    self.likelihoods[cls][feature][value] = (count + self.smoothing) / (total_counts + self.smoothing * len(values))

    def predict(self, X):
        """
        Predicts class labels for new data.

        Args:
            X: Test data features (numpy array).

        Returns:
            Predicted class labels (numpy array).
        """
        predictions = []
        for sample in X:
            posteriors = {}
            for cls in self.priors:
                posterior = self.priors[cls]
                for feature, value in enumerate(sample):
                    if value in self.likelihoods[cls][feature]:
                        posterior *= self.likelihoods[cls][feature][value]
                posteriors[cls] = posterior
            predictions.append(max(posteriors, key=posteriors.get))
        return np.array(predictions)

# Example Usage
if __name__ == "__main__":
    # Sample data (replace with your own data)
    X = np.array([[1, 2, 0], [2, 1, 1], [0, 2, 1], [1, 0, 0], [2, 1, 0]])
    y = np.array([0, 1, 1, 0, 0])

    # Create and train the model
    model = NaiveBayes()
    model.fit(X, y)

    # Make predictions
    X_new = np.array([[1, 1, 1]])
    y_pred = model.predict(X_new)
    print("Predictions:", y_pred)
```
{% endcode %}



## <mark style="color:purple;">Support Vector Machines (SVMs)</mark>

* Support Vector Machines (SVMs) are a popular machine learning algorithm used for both classification and regression tasks.
* SVMs work by finding the best hyperplane that separates the data into different classes or predicts a target value for a given input.

#### SVM for Binary Classification

* In binary classification, the hyperplane that maximizes the margin between two classes is selected as the decision boundary.
* The margin is the distance between the hyperplane and the closest instances from each class.
* SVMs seek to maximize this margin while minimising the classification error.

#### Kernels

* SVMs can handle both linear and non-linearly separable data by using different types of **kernel functions**.
* Common kernels include:
  * **Linear kernel**
  * **Polynomial kernel**
  * **Radial Basis Function (RBF) kernel**
* Kernels transform input data into higher-dimensional space where it can be more easily separated.

#### Advantages of SVMs

* SVMs can handle high-dimensional data and are effective even when the number of features exceeds the number of instances.
* They work well for both linearly and non-linearly separable data, offering versatility for a wide range of problems.
* SVMs find the optimal hyperplane, which can lead to better generalization performance and lower overfitting compared to other algorithms.

#### Limitations of SVMs

* SVMs can be sensitive to the choice of kernel function and its parameters.
* They can be computationally expensive, especially with large datasets or non-linearly separable data.
* SVMs can be prone to overfitting when the dataset is noisy or when a complex kernel function is used.

#### Mathematical Foundation of SVMs

1. **Hyperplane in Binary Classification**:
   * The goal of SVMs is to find the hyperplane that maximizes the margin between the two classes.
   *   The equation of the hyperplane is:

       \[ w \cdot x + b = 0 ]

       where:

       * ( w ) is the normal vector to the hyperplane.
       * ( b ) is the bias term.
       * The dot product ( w \cdot x ) represents the projection of ( x ) onto the direction of the normal vector.
2. **Classification**:
   * If the signed distance of ( x ) from the hyperplane is positive, ( x ) is classified as positive; if negative, it is classified as negative.
3. **Optimization Problem**:
   *   To find the optimal hyperplane, we solve the following optimization problem:

       \[ \text{minimize: } \frac{1}{2} ||w||^2 ] subject to: \[ y\_i (w \cdot x\_i + b) \geq 1 \quad \text{for all } i ]

       where:

       * ( ||w|| ) is the L2 norm of the weight vector ( w ),
       * ( y\_i ) is the class label of the i-th instance ((+1) or (-1)),
       * ( x\_i ) is the feature vector of the i-th instance.

       This constraint ensures that all instances are correctly classified with a margin of at least 1.
4.  **Dual Form**:

    * We use **Lagrangian duality** to convert this into a dual form, which involves maximizing a function of the **Lagrange multipliers**. Support vectors (instances that lie on the margin or are misclassified) have non-zero Lagrange multipliers.

    The dual form is:

    \[ \text{maximize: } \sum\_i \alpha\_i - \frac{1}{2} \sum\_i \sum\_j y\_i y\_j \alpha\_i \alpha\_j (x\_i \cdot x\_j) ] subject to: \[ \sum\_i y\_i \alpha\_i = 0 \quad \text{and} \quad 0 \leq \alpha\_i \leq C \quad \text{for all } i ]

    where:

    * ( \alpha\_i ) is the Lagrange multiplier for the i-th instance,
    * ( C ) is a hyperparameter controlling the trade-off between margin maximization and classification error.
5. **Weight and Bias Calculation**:
   * Once the optimal values of the L



{% code fullWidth="true" %}
```python
import numpy as np

class SVM:
    def __init__(self, C=1.0):
        self.C = C  # Regularization parameter
        self.w = None
        self.b = None

    def fit(self, X, y):
        """
        Trains the SVM model using the SMO algorithm.

        Args:
            X: Training data features (numpy array).
            y: Training data labels (-1 or 1).
        """
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        # Implement the SMO algorithm here (simplified version)
        # ... (See detailed explanation and implementation below)

    def predict(self, X):
        """
        Predicts class labels for new data.

        Args:
            X: Test data features (numpy array).

        Returns:
            Predicted class labels (-1 or 1).
        """
        return np.sign(np.dot(X, self.w) + self.b)

# Simplified SMO algorithm (for illustration purposes)
def simplified_smo(X, y, C, tol=1e-3, max_iter=1000):
    n_samples, n_features = X.shape
    alpha = np.zeros(n_samples)
    b = 0

    for _ in range(max_iter):
        num_changed_alphas = 0
        for i in range(n_samples):
            # ... (Calculate gradients, update alphas, etc.)
            # ... (Simplified for illustration)
            pass

        if num_changed_alphas == 0:
            break

    # Calculate weights after training
    w = np.zeros(n_features)
    for i in range(n_samples):
        w += alpha[i] * y[i] * X[i]

    return w, b

# Example Usage
if __name__ == "__main__":
    # Sample data (replace with your own data)
    X = np.array([[1, 2], [2, 4], [3, 6], [4, 8], [5, 10]])
    y = np.array([-1, -1, -1, 1, 1]) 

    # Create and train the model
    svm = SVM()
    svm.fit(X, y)  # Simplified SMO implementation needs to be completed

    # Make predictions
    X_new = np.array([[6, 12]])
    y_pred = svm.predict(X_new)
    print("Predictions:", y_pred)
```
{% endcode %}

## <mark style="color:purple;">K-Nearest Neighbors (KNN)</mark>

* **K-Nearest Neighbors (KNN)** is a non-parametric machine learning algorithm used for both classification and regression tasks.
* It works by finding the **k closest instances** in the training dataset to a new input instance and using their labels or values to make a prediction for the new instance.
* The value of **k** is a hyperparameter that can be tuned to improve the performance of the model.

### Pros of KNN

* **Non-parametric**:
  * KNN does not make any assumptions about the underlying distribution of the data, which makes it versatile for a wide range of problems.
* **Lazy learning**:
  * KNN does not require any training time since it stores all instances and performs computation at prediction time. This characteristic makes it suitable for real-time predictions.
* **Versatility**:
  * KNN works well for both **classification** and **regression** tasks and can be easily extended to handle **multi-class classification** problems.
* **Simplicity**:
  * KNN is easy to implement and interpret, making it a good choice for those new to machine learning.

### Cons of KNN

* **Choice of distance metric**:
  * KNN can be sensitive to the choice of distance metric, such as **Euclidean**, **Manhattan**, or **Minkowski**. Different metrics may perform better or worse depending on the dataset and problem.
* **Computationally expensive**:
  * KNN can be computationally expensive, especially when dealing with large datasets, as it requires calculating distances between the new instance and all training instances.
* **Curse of dimensionality**:
  * KNN struggles with **high-dimensional data**, where the distances between instances can become less meaningful, negatively impacting its performance.
* **Imbalanced datasets**:
  * KNN may be biased towards the majority class in **imbalanced datasets** and may require additional techniques, such as **data resampling** or **class weighting**, to address this issue.import numpy as np

```python
from collections import Counter

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2)**2))

    def predict(self, X_test):
        y_pred = []
        for x in X_test:
            distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            most_common = Counter(k_nearest_labels).most_common(1)
            y_pred.append(most_common[0][0])
        return np.array(y_pred)

# Example Usage
if __name__ == "__main__":
    # Sample data
    X_train = np.array([[1, 2], [2, 3], [3, 1], [4, 3], [5, 2]])
    y_train = np.array([0, 0, 1, 1, 0])
    X_test = np.array([[4, 2]])

    # Create and train the model
    knn = KNN(k=3)
    knn.fit(X_train, y_train)

    # Make predictions
    y_pred = knn.predict(X_test)
    print("Predictions:", y_pred)
```

## <mark style="color:purple;">Decision Tree</mark>

* A **decision tree** is a tree-based model used for **classification** or **regression** tasks. It is a supervised learning algorithm that learns a mapping from input features to output targets based on a set of training examples.
* A decision tree consists of a **root node**, **branches**, and **leaf nodes**. The root node represents the entire dataset, and each branch represents a decision based on a feature value. Each leaf node represents a class label or a numeric value, depending on the task.
* The process of building a decision tree is called **induction**. It involves recursively splitting the data based on the most informative features to create homogeneous subsets of data. The split criteria can be chosen based on various measures of purity or impurity, such as **entropy**, **Gini index**, or **classification error**.
* Once the decision tree is built, it can be used to classify new instances by traversing the tree from the root to a leaf node, based on the feature values of the instance. The leaf node reached by the traversal represents the predicted class label or numeric value.

### Advantages

* **Easy to understand and interpret**: Decision trees can be easily visualized and interpreted, and they help identify the most important features.
* **Handles both continuous and categorical data**: No need for feature scaling or normalization.
* **Versatility**: Decision trees can be used for both classification and regression tasks.
* **Fast training and prediction**: They are scalable and efficient, suitable for large datasets.
* **Non-linear effects**: Decision trees can capture non-linear relationships and interactions between features.
* **Robust to outliers**: Decision trees are resistant to outliers and noise in the data.

### Disadvantages

* **Overfitting**: If the tree is too deep, it can overfit the data, leading to poor generalization on new data.
* **Unstable**: Small variations in data can result in different trees, making the model sensitive.
* **Bias towards categorical features**: Trees tend to favor features with many categories, which can reduce performance for continuous features.
* **Limited extrapolation ability**: Decision trees may not generalize well outside the training data's range and can be sensitive to changes in data distribution.

### Algorithms for Decision Trees

* **ID3 (Iterative Dichotomiser 3)**:
  * Uses **entropy** or the **Gini index** to determine the best split.
  * Splits data recursively until the leaves are pure.
  * Sensitive to noisy and irrelevant features.
* **C4.5**:
  * An extension of ID3 that handles continuous and categorical data.
  * Uses **Gini index** and **gain ratio** to select the best splits.
  * More resistant to overfitting but can be slower to train.
* **CART (Classification and Regression Trees)**:
  * Uses **Gini index** for classification and **mean squared error** for regression.
  * Suitable for both continuous and categorical data.
  * Can overfit if the tree is too deep.
* **CHAID (Chi-squared Automatic Interaction Detector)**:
  * Uses the **chi-squared statistic** to determine the best split.
  * Can handle both continuous and categorical data.
  * More accurate but slower for large datasets.
* **Random Forests**:
  * An ensemble method that builds multiple decision trees and averages their predictions to reduce overfitting.

### Split Criteria for Decision Trees

* **Entropy**:
  * A measure of impurity or randomness of data.
  * High when classes are evenly distributed and low when one class dominates.
  * Information gain is the difference between parent and child node entropy.
* **Gini Index**:
  * A measure of impurity that sums the squared probabilities of class labels.
  * High when classes are evenly distributed, and low when one class dominates.
  * Used in CART and random forests.
* **Classification Error**:
  * Measures the proportion of misclassified instances in a node.
  * More sensitive to the number of classes than entropy or Gini index.
  * Error reduction is the difference between parent and child node errors.

The choice of split criteria depends on the characteristics of the data and the task at hand. Entropy and Gini index are more sensitive to class distribution, while classification error focuses on class number changes.



{% code fullWidth="true" %}
```python
import numpy as np

class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        """
        Builds the decision tree model.

        Args:
            X: Training data features (numpy array).
            y: Training data labels (numpy array).
        """
        self.root = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        num_labels = len(np.unique(y))

        # Stopping criteria
        if (num_labels == 1 
                or num_samples < self.min_samples_split 
                or depth == self.max_depth):
            return Node(value=self._most_common_label(y))

        # Find the best split
        best_split = self._find_best_split(X, y)
        if best_split is None:
            return Node(value=self._most_common_label(y))

        # Split the data
        left_idx, right_idx = best_split['idx_left'], best_split['idx_right']
        X_left, X_right = X[left_idx], X[right_idx]
        y_left, y_right = y[left_idx], y[right_idx]

        # Recursively build subtrees
        left_subtree = self._build_tree(X_left, y_left, depth + 1)
        right_subtree = self._build_tree(X_right, y_right, depth + 1)

        return Node(
            feature_idx=best_split['feature_idx'], 
            threshold=best_split['threshold'], 
            left=left_subtree, 
            right=right_subtree
        )

    def _find_best_split(self, X, y):
        best_gain = -float('inf')
        best_split = None

        num_samples, num_features = X.shape

        for feature_idx in range(num_features):
            for threshold in np.unique(X[:, feature_idx]):
                idx_left = X[:, feature_idx] <= threshold
                idx_right = X[:, feature_idx] > threshold

                gain = self._information_gain(y, idx_left, idx_right)

                if gain > best_gain:
                    best_gain = gain
                    best_split = {
                        'feature_idx': feature_idx,
                        'threshold': threshold,
                        'idx_left': idx_left,
                        'idx_right': idx_right
                    }

        return best_split

    def _information_gain(self, y, idx_left, idx_right):
        # Calculate parent entropy
        parent_entropy = self._entropy(y)

        # Calculate weighted child entropies
        weight_left = len(idx_left) / len(y)
        weight_right = len(idx_right) / len(y)
        child_entropy = (weight_left * self._entropy(y[idx_left]) 
                         + weight_right * self._entropy(y[idx_right]))

        # Calculate information gain
        information_gain = parent_entropy - child_entropy
        return information_gain

    def _entropy(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        entropy = -np.sum([p * np.log2(p) for p in probabilities if p > 0])
        return entropy

    def _most_common_label(self, y):
        return np.argmax(np.bincount(y))

    def predict(self, X):
        """
        Predicts class labels for new data.

        Args:
            X: Test data features (numpy array).

        Returns:
            Predicted class labels (numpy array).
        """
        predictions = []
        for x in X:
            predictions.append(self._predict_single(x, self.root))
        return np.array(predictions)

    def _predict_single(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature_idx] <= node.threshold:
            return self._predict_single(x, node.left)
        else:
            return self._predict_single(x, node.right)

class Node:
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

# Example Usage
if __name__ == "__main__":
    # Sample data (replace with your own data)
    X = np.array([[1, 2], [2, 3], [3, 1], [4, 3], [5, 2]])
    y = np.array([0, 0, 1, 1, 0])

    # Create and train the model
    tree = DecisionTree()
    tree.fit(X, y)

    # Make predictions
    X_test = np.array([[4, 2]])
    y_pred = tree.predict(X_test)
    print("Predictions:", y_pred)
```
{% endcode %}

## <mark style="color:yellow;">Ensemble Learning</mark>

* Ensemble learning is a machine learning technique that combines the predictions of multiple individual models (called base learners) to improve overall predictive performance.
* The idea behind ensemble learning is that by aggregating the opinions of multiple models, the ensemble can often achieve higher accuracy and be more robust than any individual model.

### Bagging vs Boosting

#### Bagging

* Bagging stands for Bootstrap Aggregating, and it involves building multiple models independently and combining their predictions by averaging or taking the majority vote.
* The idea is to reduce the variance of the models by training them on different subsets of the data.
* Bagging is typically used for high-variance models that are prone to overfitting, such as decision trees.
* **Random Forest** is an example of a bagging algorithm that uses decision trees as base models.

#### Boosting

* Boosting involves building multiple weak models sequentially and adjusting the weights of the data points based on the errors of the previous models.
* The idea is to improve the performance of the models by focusing on the difficult examples.
* Boosting is typically used to reduce the bias of models that are prone to underfitting, such as decision stumps (i.e., decision trees with only one split).
* **AdaBoost** and **Gradient Boosting** are examples of boosting algorithms that use decision stumps or shallow decision trees as base models.

### Key Differences Between Bagging and Boosting

* **Bagging** builds multiple models independently, while **boosting** builds them sequentially and adapts the data weights.
* **Bagging** reduces the variance of the models, while **boosting** reduces the bias of the models.
* **Bagging** is typically used for high-variance models that are prone to overfitting, while **boosting** is typically used for low-bias models that are prone to underfitting.
* **Bagging** combines the predictions of the models by averaging or taking the majority vote, while **boosting** combines them by giving more weight to the models that perform well on difficult examples.

### <mark style="color:purple;">Bagging Methods</mark>

#### <mark style="color:yellow;">Random Forest</mark>

* A popular ensemble method that uses decision trees as base models.
* Builds multiple decision trees independently and combines their predictions by averaging or taking the majority vote.
* Each decision tree in a Random Forest is trained on a **bootstrap sample** of the data, which is a random sample with replacement from the original data.
* This process introduces diversity and reduces overfitting.
* Each tree is trained on a random subset of the features, which further reduces overfitting and decorrelates the trees.

**Process of Building a Decision Tree in a Random Forest**

1. Select a random subset of the data (with replacement) and a random subset of the features.
2. Split the data into two subsets based on a randomly selected feature and a randomly selected split point.
3. Repeat step 2 recursively for each subset until a stopping criterion is met (e.g., maximum depth, minimum number of samples per leaf).
4. Output the leaf node (i.e., class label) for each data point that reaches it.

* To make a prediction for a new data point in a Random Forest, the algorithm runs the new data point through each decision tree in the forest and aggregates their predictions by taking the **majority vote** (for classification) or the **average** (for regression).

#### Pros of Random Forest

* **High accuracy**: Effective for high-dimensional and noisy data, captures nonlinear and interaction effects.
* **Low overfitting**: Uses multiple decision trees trained on random subsets, reducing overfitting and improving generalization.
* **Fast training and prediction**: Can be trained and evaluated in parallel, scalable for large datasets, handles missing values.
* **Resistant to outliers**: Less sensitive to outliers and noise compared to other models like linear regression.

#### Cons of Random Forest

* **Black box model**: Complex to interpret and explain, may suffer from bias and instability.
* **Large memory usage**: Can require large memory, especially for larger datasets or a high number of trees.
* **Biased towards categorical features**: May lead to overfitting and suboptimal performance for continuous features.
* **Limited extrapolation ability**: A local model that may not generalize well outside the range of the training data.



```python
import numpy as np
from collections import Counter

class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None

    # ... (Implementation of Decision Tree class as in the previous response) 
    # ... (Including fit, _build_tree, _find_best_split, _information_gain, 
    #       _entropy, _most_common_label, _predict_single, Node class) 

class RandomForest:
    def __init__(self, n_trees=100, max_features='sqrt', 
                 min_samples_split=2, max_depth=None):
        self.n_trees = n_trees
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        """
        Trains the Random Forest model.

        Args:
            X: Training data features (numpy array).
            y: Training data labels (numpy array).
        """
        for _ in range(self.n_trees):
            # Bootstrap sampling
            idx = np.random.choice(len(X), len(X), replace=True)
            X_bootstrap = X[idx]
            y_bootstrap = y[idx]

            # Feature subsetting
            if self.max_features == 'sqrt':
                max_features = int(np.sqrt(X.shape[1]))
            elif self.max_features == 'log2':
                max_features = int(np.log2(X.shape[1]))
            else:
                max_features = self.max_features 

            feature_idx = np.random.choice(X.shape[1], max_features, replace=False)
            X_bootstrap = X_bootstrap[:, feature_idx]

            # Create and train a decision tree
            tree = DecisionTree(
                min_samples_split=self.min_samples_split, 
                max_depth=self.max_depth
            )
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)

    def predict(self, X):
        """
        Predicts class labels for new data.

        Args:
            X: Test data features (numpy array).

        Returns:
            Predicted class labels (numpy array).
        """
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=0, arr=predictions)

# Example Usage
if __name__ == "__main__":
    # Sample data (replace with your own data)
    X = np.array([[1, 2], [2, 3], [3, 1], [4, 3], [5, 2]])
    y = np.array([0, 0, 1, 1, 0])

    # Create and train the model
    rf = RandomForest(n_trees=10) 
    rf.fit(X, y)

    # Make predictions
    X_test = np.array([[4, 2]])
    y_pred = rf.predict(X_test)
    print("Predictions:", y_pred)
```

#### Other Bagging Methods

* **B**<mark style="color:yellow;">**agging Meta-estimator**</mark><mark style="color:yellow;">:</mark> A generic bagging algorithm that can be used with any base model. Trains multiple models on different subsets of the data and combines their predictions by averaging or taking the majority vote.
* <mark style="color:yellow;">**Extra Trees**</mark><mark style="color:yellow;">:</mark> Similar to Random Forest, but it uses extremely randomized trees. It selects the split points randomly to increase model diversity.
* <mark style="color:yellow;">**Bootstrapped SVM**</mark><mark style="color:yellow;">:</mark> Uses support vector machines (SVMs) as base models, trained on bootstrapped samples of data and combines their predictions by averaging.
* <mark style="color:yellow;">**Bagging Ensemble Clustering**</mark><mark style="color:yellow;">:</mark> Uses clustering algorithms as base models, trained on different subsets of the data, and combines their predictions by taking the majority vote.



## <mark style="color:purple;">Boosting Methods</mark>

Boosting is a powerful ensemble learning technique that combines multiple weak learners to create a strong learner. Here are some popular boosting methods:

### <mark style="color:yellow;">1. AdaBoost</mark>

* **AdaBoost (Adaptive Boosting)** trains multiple weak classifiers (e.g., decision stumps) on different subsets of data and adjusts their weights based on performance.
* Combines weak classifiers into a strong classifier using a weighted majority vote.
* **Advantages**:
  * Flexible and applicable to many classification and regression problems.
  * Handles both numerical and categorical data.
  * Less prone to overfitting.
  * Achieves high accuracy with relatively few iterations.
* **Limitations**:
  * Sensitive to noisy data and outliers.
  * Computationally expensive.
  * May not perform well on imbalanced datasets.

```python
import numpy as np

class DecisionStump:
    def __init__(self):
        self.polarity = 1
        self.feature_idx = None
        self.threshold = None
        self.alpha = None

    def predict(self, X):
        n_samples = X.shape[0]
        X_column = X[:, self.feature_idx]
        predictions = np.ones(n_samples)
        predictions[X_column < self.threshold] = -1
        predictions *= self.polarity
        return predictions

class AdaBoost:
    def __init__(self, n_clf=5):
        self.n_clf = n_clf
        self.clfs = []

    def fit(self, X, y):
        n_samples = len(y)
        # Initialize weights
        w = np.full(n_samples, (1 / n_samples))

        for _ in range(self.n_clf):
            clf = DecisionStump()
            min_error = float('inf')

            # Find the best classifier
            for feature_i in range(X.shape[1]):
                X_column = X[:, feature_i]
                thresholds = np.unique(X_column)

                for thr in thresholds:
                    pred = np.ones(n_samples)
                    pred[X_column < thr] = -1
                    misclassified = w[y != pred]
                    error = np.sum(misclassified)

                    if error > 0.5:
                        error = 1 - error
                        pred *= -1

                    if error < min_error:
                        clf.polarity = 1 if pred[0] > 0 else -1
                        clf.feature_idx = feature_i
                        clf.threshold = thr
                        min_error = error

            # Calculate classifier weight
            clf.alpha = 0.5 * np.log((1.0 - min_error) / (min_error + 1e-10))

            # Update sample weights
            w *= np.exp(-clf.alpha * y * clf.predict(X))
            w /= np.sum(w)

            self.clfs.append(clf)

    def predict(self, X):
        clf_preds = [clf.alpha * clf.predict(X) for clf in self.clfs]
        y_pred = np.sign(np.sum(clf_preds, axis=0))
        return y_pred

# Example Usage
if __name__ == "__main__":
    # Sample data (replace with your own data)
    X = np.array([[1, 2], [2, 4], [3, 6], [4, 8], [5, 10]])
    y = np.array([-1, -1, -1, 1, 1]) 

    # Create and train the model
    adaboost = AdaBoost(n_clf=3)  # Number of weak classifiers
    adaboost.fit(X, y)

    # Make predictions
    X_new = np.array([[6, 12]])
    y_pred = adaboost.predict(X_new)
    print("Predictions:", y_pred)
```

### <mark style="color:yellow;">2. Gradient Boosting</mark>

* Builds an ensemble of weak learners (usually decision trees) in a sequential manner, with each learner correcting the errors of the previous learners.
* The algorithm aims to minimize the residual errors of previous models.
* **Advantages**:
  * High accuracy for regression, classification, and ranking tasks.
  * Handles different types of data.
  * Less prone to overfitting.
  * Handles missing data and outliers.
* **Limitations**:
  * Computationally expensive.
  * Requires careful hyperparameter tuning.
  * Sensitive to noisy data and outliers.

```python
import numpy as np

class DecisionTreeRegressor:
    def __init__(self, max_depth=3, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree_ = None

    def fit(self, X, y):
        # ... (Implementation of Decision Tree Regressor)
        # ... (Similar to the DecisionTree class in the previous example, 
        #      but modified for regression)

    def predict(self, X):
        # ... (Implementation of prediction for Decision Tree Regressor)

class GradientBoostingRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, 
                 min_samples_split=2):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees_ = []

    def fit(self, X, y):
        y_pred = np.zeros(len(y))
        for _ in range(self.n_estimators):
            residuals = y - y_pred
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth, 
                min_samples_split=self.min_samples_split
            )
            tree.fit(X, residuals)
            self.trees_.append(tree)
            y_pred += self.learning_rate * tree.predict(X)

    def predict(self, X):
        y_pred = np.zeros(len(X))
        for tree in self.trees_:
            y_pred += self.learning_rate * tree.predict(X)
        return y_pred

# Example Usage
if __name__ == "__main__":
    # Sample data (replace with your own data)
    X = np.array([[1, 2], [2, 3], [3, 1], [4, 3], [5, 2]])
    y = np.array([3, 5, 1, 4, 2])

    # Create and train the model
    gb_regressor = GradientBoostingRegressor()
    gb_regressor.fit(X, y)

    # Make predictions
    X_new = np.array([[4, 2]])
    y_pred = gb_regressor.predict(X_new)
    print("Predictions:", y_pred)
```

### <mark style="color:yellow;">3. XGBoost</mark>

* **XGBoost (eXtreme Gradient Boosting)** is an optimized version of Gradient Boosting designed for speed, scalability, and accuracy.
* Builds an ensemble of decision trees sequentially, using gradient descent optimization and regularization techniques.
* **Advantages**:
  * Scalable and handles large datasets.
  * Efficient and can run on a single machine or distributed systems.
  * Has a built-in cross-validation module for hyperparameter tuning.
* **Limitations**:
  * Sensitive to noisy or irrelevant features, leading to overfitting.
  * Hyperparameter tuning can be time-consuming.
  * Less interpretable due to the complexity of the ensemble.

```python
import numpy as np

class DecisionTreeRegressor:
    def __init__(self, max_depth=3, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree_ = None

    def fit(self, X, y):
        """
        Builds the decision tree model.

        Args:
            X: Training data features (numpy array).
            y: Training data labels (numpy array).
        """
        self.tree_ = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        num_labels = len(np.unique(y))

        # Stopping criteria
        if (num_labels == 1 
                or num_samples < self.min_samples_split 
                or depth == self.max_depth):
            return Node(value=np.mean(y))

        # Find the best split
        best_split = self._find_best_split(X, y)
        if best_split is None:
            return Node(value=np.mean(y))

        # Split the data
        left_idx, right_idx = best_split['idx_left'], best_split['idx_right']
        X_left, X_right = X[left_idx], X[right_idx]
        y_left, y_right = y[left_idx], y_right

        # Recursively build subtrees
        left_subtree = self._build_tree(X_left, y_left, depth + 1)
        right_subtree = self._build_tree(X_right, y_right, depth + 1)

        return Node(
            feature_idx=best_split['feature_idx'], 
            threshold=best_split['threshold'], 
            left=left_subtree, 
            right=right_subtree
        )

    def _find_best_split(self, X, y):
        best_gain = -float('inf')
        best_split = None

        num_samples, num_features = X.shape

        for feature_idx in range(num_features):
            for threshold in np.unique(X[:, feature_idx]):
                idx_left = X[:, feature_idx] <= threshold
                idx_right = X[:, feature_idx] > threshold

                gain = self._mse_gain(y, idx_left, idx_right)

                if gain > best_gain:
                    best_gain = gain
                    best_split = {
                        'feature_idx': feature_idx,
                        'threshold': threshold,
                        'idx_left': idx_left,
                        'idx_right': idx_right
                    }

        return best_split

    def _mse_gain(self, y, idx_left, idx_right):
        parent_mse = np.mean((y - np.mean(y))**2)
        left_mse = np.mean((y[idx_left] - np.mean(y[idx_left]))**2)
        right_mse = np.mean((y[idx_right] - np.mean(y[idx_right]))**2)
        weight_left = len(idx_left) / len(y)
        weight_right = len(idx_right) / len(y)
        child_mse = weight_left * left_mse + weight_right * right_mse
        return parent_mse - child_mse

    def predict(self, X):
        """
        Predicts class labels for new data.

        Args:
            X: Test data features (numpy array).

        Returns:
            Predicted class labels (numpy array).
        """
        predictions = []
        for x in X:
            predictions.append(self._predict_single(x, self.tree_))
        return np.array(predictions)

    def _predict_single(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature_idx] <= node.threshold:
            return self._predict_single(x, node.left)
        else:
            return self._predict_single(x, node.right)

class Node:
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

class XGBoostRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3,
                 min_samples_split=2, gamma=0, lambda_=1, alpha=0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.gamma = gamma  # Minimum loss reduction required to make a further partition on a leaf node
        self.lambda_ = lambda_  # L2 regularization term on weights
        self.alpha = alpha  # L1 regularization term on weights
        self.trees_ = []

    def fit(self, X, y):
        y_pred = np.zeros(len(y))
        for i in range(self.n_estimators):
            residuals = y - y_pred
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth, 
                min_samples_split=self.min_samples_split
            )
            tree.fit(X, residuals)
            self.trees_.append(tree)

            # Calculate leaf scores with regularization 
            # (Simplified, not considering all XGBoost details)
            # ... (Calculate leaf scores considering regularization terms) 

            y_pred += self.learning_rate * tree.predict(X)

    def predict(self, X):
        y_pred = np.zeros(len(X))
        for tree in self.trees_:
            y_pred += self.learning_rate * tree.predict(X)
        return y_pred

# Example Usage
if __name__ == "__main__":
    # Sample data (replace with your own data)
    X = np.array([[1, 2], [2, 3], [3, 1], [4, 3], [5, 2]])
    y = np.array([3, 5, 1, 4, 2])

    # Create and train the model
    xgb_regressor = XGBoostRegressor()
    xgb_regressor.fit(X, y)

    # Make predictions
    X_new = np.array([[4, 2]])
    y_pred = xgb_regressor.predict(X_new)
    print("Predictions:", y_pred)
```

### <mark style="color:yellow;">4. LightGBM</mark>

* **LightGBM** is a fast and memory-efficient gradient boosting framework used for regression, classification, and ranking tasks.
* Uses techniques like **Gradient-based One-Side Sampling (GOSS)** and **Exclusive Feature Bundling (EFB)** to reduce training time and memory usage.
* **Advantages**:
  * Handles various types of data (numerical, categorical, text).
  * Highly efficient with low memory usage.
  * Supports advanced features like early stopping and cross-validation.
  * Fast training on large datasets.
* **Limitations**:
  * Difficult to interpret due to its ensemble structure.
  * Requires careful hyperparameter tuning.

```python
import numpy as np

class DecisionTreeRegressor:
    def __init__(self, max_depth=3, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree_ = None

    def fit(self, X, y):
        """
        Builds the decision tree model.

        Args:
            X: Training data features (numpy array).
            y: Training data labels (numpy array).
        """
        self.tree_ = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        num_labels = len(np.unique(y))

        # Stopping criteria
        if (num_labels == 1 
                or num_samples < self.min_samples_split 
                or depth == self.max_depth):
            return Node(value=np.mean(y))

        # Find the best split
        best_split = self._find_best_split(X, y)
        if best_split is None:
            return Node(value=np.mean(y))

        # Split the data
        left_idx, right_idx = best_split['idx_left'], best_split['idx_right']
        X_left, X_right = X[left_idx], X[right_idx]
        y_left, y_right = y[left_idx], y_right

        # Recursively build subtrees
        left_subtree = self._build_tree(X_left, y_left, depth + 1)
        right_subtree = self._build_tree(X_right, y_right, depth + 1)

        return Node(
            feature_idx=best_split['feature_idx'], 
            threshold=best_split['threshold'], 
            left=left_subtree, 
            right=right_subtree
        )

    def _find_best_split(self, X, y, gamma=0):
        best_gain = -float('inf')
        best_split = None

        num_samples, num_features = X.shape

        for feature_idx in range(num_features):
            for threshold in np.unique(X[:, feature_idx]):
                idx_left = X[:, feature_idx] <= threshold
                idx_right = X[:, feature_idx] > threshold

                gain = self._gain(y, idx_left, idx_right, gamma)

                if gain > best_gain:
                    best_gain = gain
                    best_split = {
                        'feature_idx': feature_idx,
                        'threshold': threshold,
                        'idx_left': idx_left,
                        'idx_right': idx_right
                    }

        return best_split

    def _gain(self, y, idx_left, idx_right, gamma):
        parent_loss = self._loss(y)
        left_loss = self._loss(y[idx_left])
        right_loss = self._loss(y[idx_right])
        weight_left = len(idx_left) / len(y)
        weight_right = len(idx_right) / len(y)
        child_loss = weight_left * left_loss + weight_right * right_loss
        gain = parent_loss - child_loss - gamma  # Include gamma for regularization
        return gain

    def _loss(self, y):
        # Example: Squared loss
        return np.mean((y - np.mean(y))**2) 

    def predict(self, X):
        """
        Predicts class labels for new data.

        Args:
            X: Test data features (numpy array).

        Returns:
            Predicted class labels (numpy array).
        """
        predictions = []
        for x in X:
            predictions.append(self._predict_single(x, self.tree_))
        return np.array(predictions)

    def _predict_single(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature_idx] <= node.threshold:
            return self._predict_single(x, node.left)
        else:
            return self._predict_single(x, node.right)

class Node:
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

class XGBoostRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, 
                 min_samples_split=2, gamma=0, lambda_=1, alpha=0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.gamma = gamma  # Minimum loss reduction required to make a further partition on a leaf node
        self.lambda_ = lambda_  # L2 regularization term on weights
        self.alpha = alpha  # L1 regularization term on weights
        self.trees_ = []

    def fit(self, X, y):
        y_pred = np.zeros(len(y))
        for _ in range(self.n_estimators):
            residuals = y - y_pred
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth, 
                min_samples_split=self.min_samples_split
            )
            tree.fit(X, residuals)
            self.trees_.append(tree)

            # Calculate leaf scores with regularization 
            # (Simplified, not considering all XGBoost details)
            # ... (Calculate leaf scores considering regularization terms) 

            y_pred += self.learning_rate * tree.predict(X)

    def predict(self, X):
        y_pred = np.zeros(len(X))
        for tree in self.trees_:
            y_pred += self.learning_rate * tree.predict(X)
        return y_pred

# Example Usage
if __name__ == "__main__":
    # Sample data (replace with your own data)
    X = np.array([[1, 2], [2, 3], [3, 1], [4, 3], [5, 2]])
    y = np.array([3, 5, 1, 4, 2])

    # Create and train the model
    xgb_regressor = XGBoostRegressor(gamma=0.1, lambda_=1.0) 
    xgb_regressor.fit(X, y)

    # Make predictions
    X_new = np.array([[4, 2]])
    y_pred = xgb_regressor.predict(X_new)
    print("Predictions:", y_pred)
```

### <mark style="color:yellow;">5. CatBoost</mark>

* **CatBoost** is designed to handle categorical features without needing to convert them to numerical values.
* Uses a **permutation-driven** algorithm to handle categorical features more efficiently and **Ordered Boosting** to improve accuracy for ordinal features.
* **Advantages**:
  * High performance, especially for categorical data.
  * Does not require manual conversion of categorical values.
* **Limitations**:
  * Complex and requires careful tuning for optimal performance.

```python
import numpy as np

class DecisionTreeRegressor:
    def __init__(self, max_depth=3, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree_ = None

    def fit(self, X, y):
        """
        Builds the decision tree model.

        Args:
            X: Training data features (numpy array).
            y: Training data labels (numpy array).
        """
        self.tree_ = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        num_labels = len(np.unique(y))

        # Stopping criteria
        if (num_labels == 1 
                or num_samples < self.min_samples_split 
                or depth == self.max_depth):
            return Node(value=np.mean(y))

        # Find the best split
        best_split = self._find_best_split(X, y)
        if best_split is None:
            return Node(value=np.mean(y))

        # Split the data
        left_idx, right_idx = best_split['idx_left'], best_split['idx_right']
        X_left, X_right = X[left_idx], X[right_idx]
        y_left, y_right = y[left_idx], y_right

        # Recursively build subtrees
        left_subtree = self._build_tree(X_left, y_left, depth + 1)
        right_subtree = self._build_tree(X_right, y_right, depth + 1)

        return Node(
            feature_idx=best_split['feature_idx'], 
            threshold=best_split['threshold'], 
            left=left_subtree, 
            right=right_subtree
        )

    def _find_best_split(self, X, y, gamma=0):
        best_gain = -float('inf')
        best_split = None

        num_samples, num_features = X.shape

        for feature_idx in range(num_features):
            for threshold in np.unique(X[:, feature_idx]):
                idx_left = X[:, feature_idx] <= threshold
                idx_right = X[:, feature_idx] > threshold

                gain = self._gain(y, idx_left, idx_right, gamma)

                if gain > best_gain:
                    best_gain = gain
                    best_split = {
                        'feature_idx': feature_idx,
                        'threshold': threshold,
                        'idx_left': idx_left,
                        'idx_right': idx_right
                    }

        return best_split

    def _gain(self, y, idx_left, idx_right, gamma):
        parent_loss = self._loss(y)
        left_loss = self._loss(y[idx_left])
        right_loss = self._loss(y[idx_right])
        weight_left = len(idx_left) / len(y)
        weight_right = len(idx_right) / len(y)
        child_loss = weight_left * left_loss + weight_right * right_loss
        gain = parent_loss - child_loss - gamma  # Include gamma for regularization
        return gain

    def _loss(self, y):
        # Example: Squared loss
        return np.mean((y - np.mean(y))**2) 

    def predict(self, X):
        """
        Predicts class labels for new data.

        Args:
            X: Test data features (numpy array).

        Returns:
            Predicted class labels (numpy array).
        """
        predictions =
        for x in X:
            predictions.append(self._predict_single(x, self.tree_))
        return np.array(predictions)

    def _predict_single(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature_idx] <= node.threshold:
            return self._predict_single(x, node.left)
        else:
            return self._predict_single(x, node.right)

class Node:
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

class CatBoostRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, 
                 min_samples_split=2, gamma=0, iterations=10):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.gamma = gamma 
        self.iterations = iterations 
        self.trees_ =

    def fit(self, X, y):
        y_pred = np.zeros(len(y))
        for _ in range(self.n_estimators):
            residuals = y - y_pred

            #
```





